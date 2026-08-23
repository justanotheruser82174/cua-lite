"""ScaleCUA Chrome profile, settings, and active-tab tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lite.gym.envs.lite.scalecua.src.osworld import judges
from lite.gym.envs.lite.scalecua.src.osworld import verify as scalecua_verify
from lite.gym.envs.lite.scalecua.src.utils import dataset


def _overlays_ready() -> bool:
    """Judge-overlay tests need the imported getters/metrics modules too."""
    return all(
        (root / f"{name}.py").is_file()
        for split in dataset.RUNTIME_SPLITS
        if (root := judges.overlay_dir(split)) is not None
        for name in ("getters", "metrics")
    )


def _is_chrome_prefs_loader_command(command: str) -> bool:
    return (
        "/home/user/chrome-data/Default/Preferences" in command
        and "/home/user/.config/google-chrome/Default/Preferences" in command
    )


def _is_chrome_settings_snapshot_command(command: str) -> bool:
    return "settings-ui" in command and "Runtime.evaluate" in command


class _FakeInterface:
    def __init__(
        self,
        stdout: str | list[str] = "stdout",
        files: dict[str, bytes] | None = None,
    ):
        self.commands: list[str] = []
        self.command_calls: list[dict[str, object]] = []
        self.hotkeys: list[tuple[str, ...]] = []
        self.typed_text: list[str] = []
        self.stdout = stdout
        self.files = files or {}

    async def read_bytes(self, path: str) -> bytes:
        if path in self.files:
            return self.files[path]
        return f"bytes:{path}".encode()

    async def screenshot(self) -> bytes:
        return b"png"

    async def get_screen_size(self):
        return {"width": 800, "height": 600}

    async def run_command(self, command: str, timeout=None):
        self.commands.append(command)
        self.command_calls.append(
            {"command": command, "timeout": timeout}
        )
        if isinstance(self.stdout, list):
            stdout = self.stdout.pop(0) if self.stdout else ""
        else:
            stdout = self.stdout
        return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    async def hotkey(self, *keys: str):
        self.hotkeys.append(tuple(keys))

    async def type_text(self, text: str):
        self.typed_text.append(text)


class _FakeComputer:
    def __init__(
        self,
        stdout: str = "stdout",
        files: dict[str, bytes] | None = None,
    ):
        self.interface = _FakeInterface(stdout=stdout, files=files)


def test_scalecua_chrome_settings_snapshot_prefers_current_settings_dom():
    command = scalecua_verify._chrome_settings_snapshot_command(
        "chrome://settings/security"
    )

    assert "same_settings_url" in command
    assert "eval_snapshot(page, navigate=False)" in command
    assert "eval_snapshot(page, navigate=True)" in command
    assert command.index("eval_snapshot(page, navigate=False)") < command.index(
        "eval_snapshot(page, navigate=True)"
    )


def test_scalecua_chrome_extension_overlay_results_trigger_profile_flush():
    result_type = (
        "chrome_extensions_dev_mode__"
        "b759dda996122d64d46867fb7e10f84a_qw35sft2_93149557"
    )

    assert scalecua_verify._needs_chrome_profile_flush(result_type, {"type": result_type})


def test_scalecua_normalizes_yahoo_regional_search_engine_name():
    assert scalecua_verify._normalize_default_search_engine("duckduckgo.com") == "DuckDuckGo"
    assert scalecua_verify._normalize_default_search_engine("Duck Duck Go") == "DuckDuckGo"
    assert scalecua_verify._normalize_default_search_engine("Yahoo! Hong Kong") == "Yahoo!"
    assert scalecua_verify._normalize_default_search_engine("Yahoo Search") == "Yahoo!"
    assert scalecua_verify._normalize_default_search_engine("Microsoft Bing") == "Bing"
    assert (
        scalecua_verify._normalize_default_search_engine(
            "https://www.startpage.com/sp/search?query=%s"
        )
        == "Startpage"
    )


def test_scalecua_normalizes_generated_chrome_search_engine_labels():
    assert (
        scalecua_verify._normalize_scalecua_result(
            "chrome_search_engine__26a050291c58f47430d6ebca143c45ff",
            "Microsoft Bing (Default)",
        )
        == "Bing"
    )
    assert (
        scalecua_verify._normalize_scalecua_result(
            "default_search_engine",
            "Yahoo! Hong Kong",
        )
        == "Yahoo!"
    )


def test_scalecua_active_url_prefix_preserves_existing_subdomain():
    assert (
        scalecua_verify._apply_goto_prefix(
            "discussions.flightaware.com/t/welcome/8",
            "https://www.",
        )
        == "https://discussions.flightaware.com/t/welcome/8"
    )
    assert (
        scalecua_verify._apply_goto_prefix("drugs.com/npc/", "https://www.")
        == "https://www.drugs.com/npc/"
    )


def test_scalecua_chrome_profile_alias_covers_upstream_path_shapes():
    assert judges._alias_chrome_profile_path(
        "os.path.join(os.getenv('HOME'), '.config', 'google-chrome', 'Default', 'History')"
    ) == "os.path.join('/home/user/chrome-data', 'Default', 'History')"
    assert judges._alias_chrome_profile_path(
        "os.path.expanduser('~') + '/.config/google-chrome/Default/Extensions/'"
    ) == "os.path.expanduser('~') + '/chrome-data/Default/Extensions/'"
    assert (
        judges._alias_chrome_profile_path(
            "/home/ubuntu/.config/google-chrome/Local State"
        )
        == "/home/user/chrome-data/Local State"
    )
    broad_find = (
        'find /root /home -name "Preferences" '
        '-path "*/google-chrome/Default/*" 2>/dev/null | head -1'
    )
    aliased = judges._alias_chrome_profile_path(broad_find)
    assert "/root" not in aliased
    assert "/home/user/chrome-data/Default/Preferences" in aliased
    assert "find /home -name Preferences" in aliased
    assert "| head -1" in aliased


def test_scalecua_bookmark_metric_respects_generated_folder_name():
    metric = judges.resolve_metric("is_expected_bookmarks", "train")
    bookmarks = {
        "bookmark_bar": {
            "children": [
                {
                    "type": "folder",
                    "name": "Caltech Researchers",
                    "children": [
                        {"type": "url", "url": "https://guanzhi.me/"},
                        {"type": "url", "url": "https://tensorlab.cms.caltech.edu/users/anima/"},
                    ],
                }
            ]
        }
    }
    rule = {
        "type": "liked_authors_websites_urls",
        "names": ["Caltech Researchers"],
        "urls": [
            ["https://guanzhi.me/", "https://www.linkedin.com/in/guanzhi-wang/"],
            [
                "https://tensorlab.cms.caltech.edu/users/anima/",
                "https://en.wikipedia.org/wiki/Anima_Anandkumar",
            ],
        ],
    }

    assert metric(bookmarks, rule) == 1.0
    assert metric(bookmarks, {**rule, "names": ["Liked Authors"]}) == 0.0


def test_scalecua_bookmark_metric_ignores_eval_env_for_upstream_rules():
    metric = judges.resolve_metric("is_expected_bookmarks", "train")
    bookmarks = {
        "bookmark_bar": {
            "children": [
                {"type": "folder", "name": "Work", "children": []},
            ]
        }
    }
    rule = {"type": "bookmark_bar_folders_names", "names": ["Work"]}

    assert metric(bookmarks, rule, env=object()) == 1.0


def test_scalecua_bookmark_metric_canonicalizes_bare_and_www_hosts():
    metric = judges.resolve_metric("is_expected_bookmarks", "train")
    bookmarks = {
        "bookmark_bar": {
            "children": [
                {
                    "type": "url",
                    "url": "https://mathsisfun.com/games/2048.html/",
                },
                {
                    "type": "url",
                    "url": "https://example.com/extra",
                },
            ]
        }
    }
    rule = {
        "type": "bookmark_bar_websites_urls",
        "urls": ["https://www.mathsisfun.com/games/2048.html"],
    }

    assert metric(bookmarks, rule) == 1.0
    assert (
        metric(
            bookmarks,
            {**rule, "urls": ["https://www.mathsisfun.com/games/calculus.html"]},
        )
        == 0.0
    )


@pytest.mark.skipif(not _overlays_ready(), reason="lite.scalecua judge overlays not imported")
def test_scalecua_generated_bookmark_title_metric_canonicalizes_url_and_title():
    metric = judges.resolve_metric("is_expected_bookmarks__35253b65", "train")
    bookmarks = {
        "bookmark_bar": {
            "children": [
                {
                    "type": "url",
                    "url": "https://mathsisfun.com/games/2048.html/",
                    "name": "2048",
                }
            ]
        }
    }
    rule = {
        "type": "bookmark_bar_websites_with_titles",
        "bookmarks": [
            {
                "url": "http://www.mathsisfun.com/games/2048.html",
                "title": "2048 Game",
            }
        ],
    }

    assert metric(bookmarks, rule) == 1.0
    assert (
        metric(
            bookmarks,
            {
                **rule,
                "bookmarks": [
                    {
                        "url": "https://www.mathsisfun.com/games/calculus.html",
                        "title": "Calculus",
                    }
                ],
            },
        )
        == 0.0
    )


@pytest.mark.skipif(not _overlays_ready(), reason="lite.scalecua judge overlays not imported")
def test_scalecua_generated_startup_urls_metric_canonicalizes_http_https_www():
    metric = judges.resolve_metric(
        "check_chrome_startup_urls__715fc21b1c707dd79fc5ab9b6e4df514",
        "train",
    )

    assert (
        metric(
            ["https://www.google.com/", "https://github.com/"],
            {"urls": ["http://google.com/", "http://github.com/"]},
        )
        == 1.0
    )
    assert (
        metric(
            ["https://www.google.com/", "https://github.com/", "https://funbrain.com/"],
            {"urls": ["http://google.com/", "http://github.com/"]},
        )
        == 0.0
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not _overlays_ready(), reason="lite.scalecua judge overlays not imported")
async def test_scalecua_generated_startup_url_removed_getter_canonicalizes_url(tmp_path):
    prefs = json.dumps(
        {"session": {"startup_urls": ["https://www.reddit.com/", "https://github.com/"]}}
    ).encode("utf-8")
    env = judges.make_eval_env(
        _FakeComputer(files={"/home/user/chrome-data/Default/Preferences": prefs}),
        str(tmp_path),
    )
    getter = judges.resolve_getter(
        "chrome_startup_url_removed__a85eebd24e563a97c17935bc46126aa1",
        "train",
    )

    present = await judges.call_overlay_getter(
        getter,
        env,
        {
            "type": "chrome_startup_url_removed__a85eebd24e563a97c17935bc46126aa1",
            "check_url": "http://reddit.com/",
        },
        str(tmp_path),
    )
    removed = await judges.call_overlay_getter(
        getter,
        env,
        {
            "type": "chrome_startup_url_removed__a85eebd24e563a97c17935bc46126aa1",
            "check_url": "http://funbrain.com/",
        },
        str(tmp_path),
    )

    assert present == "false"
    assert removed == "true"


@pytest.mark.asyncio
async def test_scalecua_postconfig_chrome_pkill_uses_synchronous_profile_flush(tmp_path):
    computer = _FakeComputer()
    evaluator = {
        "postconfig": [
            {
                "type": "launch",
                "parameters": {"command": ["pkill", "chrome"]},
            }
        ]
    }

    await scalecua_verify._run_postconfig(computer, evaluator, str(tmp_path))

    assert evaluator["_postconfig_done"] is True
    assert len(computer.interface.commands) == 1
    command = computer.interface.commands[0]
    assert "Alt+F4" in command
    assert "pkill -TERM chrome" in command


@pytest.mark.asyncio
async def test_scalecua_execute_python_command_filters_xlib_warning(tmp_path):
    stdout = (
        "Xlib.xauth: warning, no xauthority details available\n"
        "Xlib.xauth: warning, no xauthority details available\n"
        "/home/user/chrome-data/Default/Preferences\n"
    )
    computer = _FakeComputer(stdout=stdout)
    env = judges.make_eval_env(computer, str(tmp_path))

    def getter(eval_env, config):
        return eval_env.controller.execute_python_command(
            "import os; print(os.path.join(os.getenv('HOME'), "
            "'.config/google-chrome/Default/Preferences'))"
        )["output"].strip()

    out = await judges.call_overlay_getter(
        getter,
        env,
        {"type": "custom"},
        str(tmp_path),
    )

    assert out == "/home/user/chrome-data/Default/Preferences"
    assert ".config/google-chrome" not in computer.interface.commands[-1]
    assert "chrome-data/Default/Preferences" in computer.interface.commands[-1]
    assert "pyautogui.FAILSAFE" not in computer.interface.commands[-1]


@pytest.mark.asyncio
async def test_scalecua_run_bash_script_aliases_broad_chrome_profile_find(tmp_path):
    computer = _FakeComputer(stdout="/home/user/chrome-data/Default/Preferences\n")
    env = judges.make_eval_env(computer, str(tmp_path))

    def getter(eval_env, config):
        return eval_env.controller.run_bash_script(
            'find /root /home -name "Preferences" '
            '-path "*/google-chrome/Default/*" 2>/dev/null | head -1'
        )["output"].strip()

    out = await judges.call_overlay_getter(
        getter,
        env,
        {"type": "custom"},
        str(tmp_path),
    )

    assert out == "/home/user/chrome-data/Default/Preferences"
    command = computer.interface.commands[-1]
    assert "/root" not in command
    assert "/home/user/chrome-data/Default/Preferences" in command
    assert "find /home -name Preferences" in command


@pytest.mark.asyncio
async def test_scalecua_chrome_profile_aliases_file_and_command_paths(tmp_path):
    computer = _FakeComputer(stdout="{}")
    env = judges.make_eval_env(computer, str(tmp_path))

    def getter(eval_env, config):
        direct = eval_env.controller.get_file(
            "/home/user/.config/google-chrome/Default/Preferences"
        )
        snap = eval_env.controller.get_file(
            "~/snap/chromium/common/chromium/Local State"
        )
        eval_env.controller.run_bash_script(
            "python3 -c \"import json; "
            "json.load(open('/home/user/.config/google-chrome/Default/Preferences'))\""
        )
        eval_env.controller.execute_python_command(
            "import os; print(os.path.join(os.getenv('HOME'), "
            "'.config/google-chrome/Default/Bookmarks'))"
        )
        return direct, snap

    direct, snap = await judges.call_overlay_getter(
        getter,
        env,
        {"type": "custom"},
        str(tmp_path),
    )

    assert direct == b"bytes:/home/user/chrome-data/Default/Preferences"
    assert snap == b"bytes:/home/user/chrome-data/Local State"
    assert len(computer.interface.commands) == 2
    assert all(".config/google-chrome" not in cmd for cmd in computer.interface.commands)
    assert all("chrome-data" in cmd for cmd in computer.interface.commands)


@pytest.mark.asyncio
async def test_scalecua_controller_get_bookmarks_reads_chrome_data(tmp_path):
    payload = json.dumps(
        {
            "roots": {
                "bookmark_bar": {
                    "children": [
                        {
                            "type": "folder",
                            "name": "Caltech Researchers",
                            "children": [{"type": "url", "url": "https://guanzhi.me/"}],
                        }
                    ]
                }
            }
        }
    )
    computer = _FakeComputer(stdout=payload)
    env = judges.make_eval_env(computer, str(tmp_path))

    def getter(eval_env, config):
        return eval_env.controller.get_bookmarks()

    out = await judges.call_overlay_getter(getter, env, {"type": "custom"}, str(tmp_path))

    assert out["bookmark_bar"]["children"][0]["name"] == "Caltech Researchers"
    assert "/home/user/chrome-data/Default/Bookmarks" in computer.interface.commands[-1]


@pytest.mark.asyncio
async def test_scalecua_canonical_result_type_prefers_base_runner(monkeypatch, tmp_path):
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))
    calls = {}

    def overlay_getter(eval_env, config):
        calls["overlay"] = True
        return "overlay"

    async def base_get_result(computer, config, cache_dir):
        calls["base"] = (computer, config, cache_dir)
        return "chrome://password-manager/passwords"

    monkeypatch.setattr(
        scalecua_verify.judges,
        "resolve_getter",
        lambda result_type, runtime_split: overlay_getter,
    )
    monkeypatch.setattr(scalecua_verify.base_runner, "_get_result", base_get_result)

    out = await scalecua_verify._get_result(
        env,
        {"type": "bookmarks"},
        str(tmp_path),
        "train",
    )

    assert out == "chrome://password-manager/passwords"
    assert "overlay" not in calls
    assert calls["base"][0] is env.computer


@pytest.mark.asyncio
async def test_scalecua_base_runner_config_commands_are_chrome_profile_aliased(
    monkeypatch, tmp_path
):
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))
    calls = {}

    async def base_get_result(computer, config, cache_dir):
        calls["config"] = config
        return "ok"

    monkeypatch.setattr(scalecua_verify.base_runner, "_get_result", base_get_result)

    out = await scalecua_verify._get_result(
        env,
        {
            "type": "vm_command_line",
            "command": "cat /home/user/.config/google-chrome/Default/Bookmarks",
        },
        str(tmp_path),
        "train",
    )

    assert out == "ok"
    assert calls["config"]["command"] == "cat /home/user/chrome-data/Default/Bookmarks"
    assert "xdotool key Tab" in env.computer.interface.commands[0]
    assert "xdotool key Escape" in env.computer.interface.commands[0]
    assert "pkill -TERM chrome" in env.computer.interface.commands[0]


@pytest.mark.asyncio
async def test_scalecua_generated_chrome_profile_getter_flushes_before_eval(
    monkeypatch, tmp_path
):
    computer = _FakeComputer()
    env = judges.make_eval_env(computer, str(tmp_path))

    def generated_getter(eval_env, config):
        assert "xdotool key Tab" in computer.interface.commands[0]
        assert "xdotool key Escape" in computer.interface.commands[0]
        assert "pkill -TERM chrome" in computer.interface.commands[0]
        return "true"

    monkeypatch.setattr(
        scalecua_verify.judges,
        "resolve_getter",
        lambda result_type, runtime_split: generated_getter,
    )

    out = await scalecua_verify._get_result(
        env,
        {
            "type": "chrome_startup_url_removed__a85eebd24e563a97c17935bc46126aa1",
            "check_url": "http://reddit.com/",
        },
        str(tmp_path),
        "train",
    )

    assert out == "true"


@pytest.mark.asyncio
async def test_scalecua_profile_name_getter_blurs_before_killing_chrome(
    monkeypatch, tmp_path
):
    computer = _FakeComputer()
    env = judges.make_eval_env(computer, str(tmp_path))

    async def base_get_result(_computer, config, _cache_dir):
        assert config == {"type": "profile_name"}
        return "Sarah"

    monkeypatch.setattr(scalecua_verify.base_runner, "_get_result", base_get_result)

    out = await scalecua_verify._get_result(
        env,
        {"type": "profile_name"},
        str(tmp_path),
        "train",
    )

    assert out == "Sarah"
    assert len(computer.interface.commands) == 1
    flush_command = computer.interface.commands[0]
    assert flush_command.index("xdotool key Tab") < flush_command.index("pkill -TERM chrome")
    assert flush_command.index("xdotool key Escape") < flush_command.index("pkill -TERM chrome")
    assert "Alt+F4" in flush_command


@pytest.mark.asyncio
async def test_scalecua_default_search_engine_uses_profile_repair_after_flush(
    monkeypatch, tmp_path
):
    computer = _FakeComputer(stdout=["", "DuckDuckGo\n"])
    env = judges.make_eval_env(computer, str(tmp_path))

    async def base_get_result(_computer, config, _cache_dir):
        assert config == {"type": "default_search_engine"}
        return "Google"

    monkeypatch.setattr(scalecua_verify.base_runner, "_get_result", base_get_result)

    out = await scalecua_verify._get_result(
        env,
        {"type": "default_search_engine"},
        str(tmp_path),
        "rl",
    )

    assert out == "DuckDuckGo"
    assert "pkill -TERM chrome" in computer.interface.commands[0]
    assert "Default/Preferences" in computer.interface.commands[1]


@pytest.mark.asyncio
async def test_scalecua_default_search_engine_repairs_startpage_template_url(
    monkeypatch, tmp_path
):
    computer = _FakeComputer(
        stdout=["", "https://www.startpage.com/sp/search?query={searchTerms}\n"]
    )
    env = judges.make_eval_env(computer, str(tmp_path))

    async def base_get_result(_computer, config, _cache_dir):
        assert config == {"type": "default_search_engine"}
        return "Google"

    monkeypatch.setattr(scalecua_verify.base_runner, "_get_result", base_get_result)

    out = await scalecua_verify._get_result(
        env,
        {"type": "default_search_engine"},
        str(tmp_path),
        "train",
    )

    assert out == "Startpage"
    repair_command = computer.interface.commands[1]
    assert "template_url" in repair_command
    assert "default_search_provider" in repair_command
    assert "extensions" not in repair_command


@pytest.mark.asyncio
async def test_scalecua_default_search_engine_candidates_match_expected(
    monkeypatch, tmp_path
):
    candidates = json.dumps(
        ["Google", "https://www.startpage.com/sp/search?query={searchTerms}"]
    )
    computer = _FakeComputer(
        stdout=["", f"__SCALECUA_SEARCH_CANDIDATES__{candidates}\n"]
    )
    env = judges.make_eval_env(computer, str(tmp_path))

    async def base_get_result(_computer, config, _cache_dir):
        assert config == {"type": "default_search_engine"}
        return "Google"

    monkeypatch.setattr(scalecua_verify.base_runner, "_get_result", base_get_result)

    out = await scalecua_verify._get_result(
        env,
        {"type": "default_search_engine"},
        str(tmp_path),
        "train",
    )

    assert out == {
        "__scalecua_candidates__": True,
        "value": "Google",
        "candidates": [
            "Google",
            "https://www.startpage.com/sp/search?query={searchTerms}",
        ],
    }
    metric = judges.resolve_metric("match_in_list", "train")
    assert metric(out, {"expected": ["Startpage", "Startpage.com"]}) == 1.0
    assert metric(out, {"expected": ["DuckDuckGo"]}) == 0.0


@pytest.mark.asyncio
async def test_scalecua_default_search_engine_repairs_empty_candidate_marker_from_ui(
    monkeypatch, tmp_path
):
    computer = _FakeComputer(
        stdout=[
            "",
            "__SCALECUA_SEARCH_CANDIDATES__[]\n",
            json.dumps({"text": "Search engines Name Shortcut Google (Default) google.com"}),
        ]
    )
    env = judges.make_eval_env(computer, str(tmp_path))

    async def base_get_result(_computer, config, _cache_dir):
        assert config == {"type": "default_search_engine"}
        return "Web Store"

    monkeypatch.setattr(scalecua_verify.base_runner, "_get_result", base_get_result)

    out = await scalecua_verify._get_result(
        env,
        {"type": "default_search_engine"},
        str(tmp_path),
        "train",
    )

    assert out == "Google"
    assert "chrome://settings/searchEngines" in computer.interface.commands[-1]


@pytest.mark.asyncio
async def test_scalecua_chrome_experiments_maps_disabled_quic_from_local_state(
    monkeypatch, tmp_path
):
    local_state = json.dumps(
        {"browser": {"enabled_labs_experiments": ["enable-quic@2"]}}
    ).encode("utf-8")
    computer = _FakeComputer(files={"/home/user/chrome-data/Local State": local_state})
    env = judges.make_eval_env(computer, str(tmp_path))

    def generated_getter(_eval_env, _config):
        return ["enable-quic"]

    monkeypatch.setattr(
        scalecua_verify.judges,
        "resolve_getter",
        lambda result_type, runtime_split: generated_getter,
    )

    out = await scalecua_verify._get_result(
        env,
        {"type": "chrome_experiments_exact_match__d7481e87a8a6dde50669ce517c215edf"},
        str(tmp_path),
        "train",
    )

    assert out == ["disable-quic"]
    assert "pkill -TERM chrome" in computer.interface.commands[0]


@pytest.mark.asyncio
async def test_scalecua_enabled_experiments_are_canonicalized_from_local_state(
    monkeypatch, tmp_path
):
    local_state = json.dumps(
        {
            "browser": {
                "enabled_labs_experiments": [
                    "overlay-scrollbars@1",
                    "smooth-scrolling@2",
                ]
            }
        }
    ).encode("utf-8")
    computer = _FakeComputer(files={"/home/user/chrome-data/Local State": local_state})
    env = judges.make_eval_env(computer, str(tmp_path))

    def generated_getter(_eval_env, _config):
        return ["overlay-scrollbars@1", "smooth-scrolling@2"]

    monkeypatch.setattr(
        scalecua_verify.judges,
        "resolve_getter",
        lambda result_type, runtime_split: generated_getter,
    )

    out = await scalecua_verify._get_result(
        env,
        {"type": "enabled_experiments"},
        str(tmp_path),
        "train",
    )

    assert out == ["overlay-scrollbars"]


@pytest.mark.skipif(not _overlays_ready(), reason="lite.scalecua judge overlays not imported")
def test_scalecua_chrome_experiments_metric_accepts_disabled_quic_alias():
    metric = judges.resolve_metric(
        "check_chrome_experiments_exact_match__d7481e87a8a6dde50669ce517c215edf",
        "train",
    )

    assert metric(["enable-quic"], {"experiment_names": ["disable-quic"]}) == 1.0
    assert metric(["enable-quic"], {"experiment_names": ["other-flag"]}) == 0.0


@pytest.mark.skipif(not _overlays_ready(), reason="lite.scalecua judge overlays not imported")
def test_scalecua_chrome_experiments_metrics_accept_current_flag_shapes():
    contains_all = judges.resolve_metric(
        "check_chrome_experiments_contains_all__e998f78abb27064086318477b860256b",
        "train",
    )
    not_contains = judges.resolve_metric(
        "check_chrome_experiments_not_contains__f72b4886483f6a090bcc6a28ba49acde",
        "train",
    )
    contains = judges.resolve_metric(
        "check_chrome_experiments_contains__0a40109ea287ab7bbd8cd9175a7ce6a5",
        "train",
    )
    contains_any = judges.resolve_metric(
        "check_chrome_experiments_contains_any__70b32da51f968d0aa45e836eecc52bdb",
        "train",
    )
    enabled = judges.resolve_metric("check_enabled_experiments", "train")

    assert contains(
        ["enable-parallel-downloading@1"],
        {"experiment_name": "enable-parallel-downloading"},
    ) == 1.0
    assert contains(
        ["enable-parallel-downloading@2"],
        {"experiment_name": "enable-parallel-downloading"},
    ) == 0.0
    assert contains_any(
        ["enable-accelerated-video-decode@1"],
        {"experiment_names": ["enable-gpu-rasterization", "enable-accelerated-video-decode"]},
    ) == 1.0
    assert contains_any(
        ["enable-accelerated-video-decode@2"],
        {"experiment_names": ["enable-accelerated-video-decode"]},
    ) == 0.0
    assert (
        contains_all(
            ["tab-groups-save-v2@1", "tab-organization@1"],
            {"experiment_names": ["tab-groups", "tab-groups-collapse"]},
        )
        == 1.0
    )
    assert not_contains(["smooth-scrolling@2"], {"experiment_name": "smooth-scrolling"}) == 1.0
    assert not_contains(["smooth-scrolling@1"], {"experiment_name": "smooth-scrolling"}) == 0.0
    assert (
        enabled(
            ["tab-groups-save@1", "tab-search-fuzzy-search@1", "side-panel@1"],
            {
                "type": "names",
                "names": ["tab-groups-save", "tab-search-fuzzy-search"],
            },
            env=object(),
        )
        == 1.0
    )


@pytest.mark.skipif(not _overlays_ready(), reason="lite.scalecua judge overlays not imported")
def test_scalecua_extension_name_metric_accepts_hello_extension_alias():
    metric = judges.resolve_metric(
        "check_extension_name__de268faa9ad661d5adcd42e1b7f775e2",
        "train",
    )

    assert metric(["helloExtension"], {"extension_name": "Hello Extensions"}) == 1.0
    assert metric(["Other Extension"], {"extension_name": "Hello Extensions"}) == 0.0


@pytest.mark.asyncio
async def test_scalecua_active_tab_getter_does_not_flush_chrome(
    monkeypatch, tmp_path
):
    computer = _FakeComputer()
    env = judges.make_eval_env(computer, str(tmp_path))

    async def fake_at_xml(_computer):
        return "<xml/>"

    monkeypatch.setattr(scalecua_verify.base_runner, "_get_at_xml", fake_at_xml)
    monkeypatch.setattr(
        scalecua_verify.base_runner,
        "_extract_address_bar_url",
        lambda at_xml: "example.com/path",
    )

    out = await scalecua_verify._get_result(
        env,
        {"type": "active_url_from_accessTree", "goto_prefix": "https://"},
        str(tmp_path),
        "train",
    )

    assert out == "https://example.com/path"
    assert "pkill -TERM chrome" not in "\n".join(computer.interface.commands)


@pytest.mark.asyncio
async def test_scalecua_extension_version_fallback_flushes_and_reads_manifest(
    monkeypatch, tmp_path
):
    class ChromeExtensionInterface(_FakeInterface):
        async def read_bytes(self, path: str) -> bytes:
            if path == "/home/user/Desktop/helloExtension/manifest.json":
                return json.dumps(
                    {
                        "name": "Hello Extensions",
                        "version": "1.0",
                        "description": "Base Level Extension",
                        "manifest_version": 3,
                    }
                ).encode("utf-8")
            return await super().read_bytes(path)

        async def run_command(self, command: str, timeout=None):
            self.commands.append(command)
            self.command_calls.append(
                {"command": command, "timeout": timeout}
            )
            if _is_chrome_prefs_loader_command(command):
                stdout = json.dumps(
                    {
                        "extensions": {
                            "settings": {
                                "mocnij": {
                                    "path": "/home/user/Desktop/helloExtension",
                                    "location": 4,
                                    "state": 1,
                                    "manifest": {},
                                }
                            }
                        }
                    }
                )
            else:
                stdout = ""
            return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    computer = _FakeComputer()
    computer.interface = ChromeExtensionInterface()
    env = judges.make_eval_env(computer, str(tmp_path))

    def generated_getter(eval_env, config):
        assert any("pkill -TERM chrome" in cmd for cmd in computer.interface.commands)
        return ""

    monkeypatch.setattr(
        scalecua_verify.judges,
        "resolve_getter",
        lambda result_type, runtime_split: generated_getter,
    )

    out = await scalecua_verify._get_result(
        env,
        {
            "type": "extension_version__407be0458b7b234fb5401d66a10f5221",
            "extension_name": "Hello Extensions",
        },
        str(tmp_path),
        "train",
    )

    assert out == "1.0"
    assert "xdotool key Tab" in computer.interface.commands[0]
    assert "xdotool key Escape" in computer.interface.commands[0]
    assert "pkill -TERM chrome" in computer.interface.commands[0]


@pytest.mark.asyncio
async def test_scalecua_extension_path_uses_local_profile_reader_not_base(
    monkeypatch, tmp_path
):
    class ChromeExtensionInterface(_FakeInterface):
        async def run_command(self, command: str, timeout=None):
            self.commands.append(command)
            self.command_calls.append(
                {"command": command, "timeout": timeout}
            )
            if _is_chrome_prefs_loader_command(command):
                stdout = json.dumps(
                    {
                        "extensions": {
                            "settings": {
                                "mocnij": {
                                    "path": "/home/user/Desktop/helloExtension",
                                    "location": 4,
                                    "state": 1,
                                    "manifest": {},
                                }
                            }
                        }
                    }
                )
            else:
                stdout = ""
            return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    computer = _FakeComputer()
    computer.interface = ChromeExtensionInterface()
    env = judges.make_eval_env(computer, str(tmp_path))

    async def base_get_result(*args, **kwargs):
        raise AssertionError("SCALE-CUA extension path should not use base runner")

    def official_getter(eval_env, config):
        raise AssertionError("extension path should be handled by local profile reader")

    monkeypatch.setattr(scalecua_verify.base_runner, "_get_result", base_get_result)
    monkeypatch.setattr(
        scalecua_verify.judges,
        "resolve_getter",
        lambda result_type, runtime_split: official_getter,
    )

    out = await scalecua_verify._get_result(
        env,
        {"type": "find_unpacked_extension_path"},
        str(tmp_path),
        "train",
    )

    assert out == ["/home/user/Desktop/helloExtension"]
    assert "xdotool key Tab" in computer.interface.commands[0]
    assert "xdotool key Escape" in computer.interface.commands[0]
    assert "pkill -TERM chrome" in computer.interface.commands[0]


@pytest.mark.asyncio
async def test_scalecua_extension_name_base_result_uses_profile_fallback(
    monkeypatch, tmp_path
):
    class ChromeExtensionInterface(_FakeInterface):
        async def read_bytes(self, path: str) -> bytes:
            if path == "/home/user/Desktop/helloExtension/manifest.json":
                return json.dumps(
                    {
                        "name": "Hello Extensions",
                        "version": "1.0",
                        "description": "Base Level Extension",
                        "manifest_version": 3,
                    }
                ).encode("utf-8")
            return await super().read_bytes(path)

        async def run_command(self, command: str, timeout=None):
            self.commands.append(command)
            self.command_calls.append(
                {"command": command, "timeout": timeout}
            )
            if _is_chrome_prefs_loader_command(command):
                stdout = json.dumps(
                    {
                        "extensions": {
                            "settings": {
                                "mocnij": {
                                    "path": "/home/user/Desktop/helloExtension",
                                    "location": 4,
                                    "state": 1,
                                    "manifest": {},
                                }
                            }
                        }
                    }
                )
            else:
                stdout = ""
            return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    computer = _FakeComputer()
    computer.interface = ChromeExtensionInterface()
    env = judges.make_eval_env(computer, str(tmp_path))

    async def base_get_result(*args, **kwargs):
        return []

    monkeypatch.setattr(scalecua_verify.base_runner, "_get_result", base_get_result)

    out = await scalecua_verify._get_result(
        env,
        {"type": "find_installed_extension_name"},
        str(tmp_path),
        "train",
    )

    assert out == ["Hello Extensions"]
    assert "xdotool key Tab" in computer.interface.commands[0]
    assert "xdotool key Escape" in computer.interface.commands[0]
    assert "pkill -TERM chrome" in computer.interface.commands[0]


@pytest.mark.asyncio
async def test_scalecua_repairs_generated_chrome_extension_getter_shapes(
    monkeypatch, tmp_path
):
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))
    entry = {
        "id": "abc",
        "path": "/home/user/Desktop/helloExtension",
        "state": 1,
        "location": 4,
        "from_webstore": False,
        "manifest": {
            "name": "Hello Extensions",
            "version": "1.0",
            "description": "Base Level Extension",
            "manifest_version": 3,
        },
    }

    async def fake_entries(eval_env):
        return [entry]

    monkeypatch.setattr(scalecua_verify, "_chrome_extension_entries", fake_entries)

    assert await scalecua_verify._repair_chrome_extension_result(
        env,
        {
            "type": "ext_count__2cf92b2479d096498965f5b9ffc3704c",
            "name_pattern": "hello.*extension",
            "path_contains": "Desktop/helloExtension",
        },
        {},
    ) == {
        "found": True,
        "path": "/home/user/Desktop/helloExtension",
        "name": "Hello Extensions",
        "enabled": True,
    }
    assert await scalecua_verify._repair_chrome_extension_result(
        env,
        {
            "type": "ext_manifest_data__86d1adf05912c94007ba506be4096d7e",
            "extension_path": "/home/user/Desktop/helloExtension",
        },
        {},
    ) == entry["manifest"]
    assert await scalecua_verify._repair_chrome_extension_result(
        env,
        {"type": "extension_description__ae6416e4"},
        {},
    ) == {
        "extensions": {"Hello Extensions": "Base Level Extension"},
        "all_names": ["Hello Extensions"],
    }
    assert await scalecua_verify._repair_chrome_extension_result(
        env,
        {"type": "extension_manifest_version__ae6416e4"},
        {},
    ) == {"Hello Extensions": {"manifest_version": 3, "is_unpacked": True}}
    assert await scalecua_verify._repair_chrome_extension_result(
        env,
        {"type": "extension_source_type__ae6416e4"},
        {},
    ) == {"Hello Extensions": "unpacked"}
    assert await scalecua_verify._repair_chrome_extension_result(
        env,
        {"type": "ext_names__3898ddf1f9c99b582b59db314361e457"},
        [],
    ) == ["Hello Extensions"]
    assert await scalecua_verify._repair_chrome_extension_result(
        env,
        {"type": "extension_names__de268faa9ad661d5adcd42e1b7f775e2"},
        [],
    ) == ["Hello Extensions"]
    assert await scalecua_verify._repair_chrome_extension_result(
        env,
        {"type": "extension_names__de268faa9ad661d5adcd42e1b7f775e2"},
        ["helloExtension"],
    ) == ["Hello Extensions"]


@pytest.mark.asyncio
async def test_scalecua_repairs_generated_third_party_cookie_getters_to_mode_one(
    monkeypatch, tmp_path
):
    class ChromeCookieInterface(_FakeInterface):
        async def run_command(self, command: str, timeout=None):
            self.commands.append(command)
            self.command_calls.append(
                {"command": command, "timeout": timeout}
            )
            if _is_chrome_prefs_loader_command(command):
                stdout = json.dumps({"profile": {"cookie_controls_mode": 1}})
            else:
                stdout = ""
            return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    computer = _FakeComputer()
    computer.interface = ChromeCookieInterface()
    env = judges.make_eval_env(computer, str(tmp_path))

    async def generated_getter(*args, **kwargs):
        return "false"

    monkeypatch.setattr(
        scalecua_verify.judges,
        "resolve_getter",
        lambda result_type, runtime_split: generated_getter,
    )

    assert await scalecua_verify._get_result(
        env,
        {"type": "block_third_party_cookies__980fd38b"},
        str(tmp_path),
        "train",
    ) == "true"
    assert await scalecua_verify._get_result(
        env,
        {"type": "block_third_party_cookies__62856a905c85744fe1f63d2f847937c7"},
        str(tmp_path),
        "train",
    ) == "true"


@pytest.mark.asyncio
async def test_scalecua_third_party_cookie_repair_rejects_incognito_only(
    tmp_path,
):
    class ChromeCookieInterface(_FakeInterface):
        async def run_command(self, command: str, timeout=None):
            self.commands.append(command)
            self.command_calls.append(
                {"command": command, "timeout": timeout}
            )
            if _is_chrome_prefs_loader_command(command):
                stdout = json.dumps({"profile": {"cookie_controls_mode": 2}})
            else:
                stdout = ""
            return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    computer = _FakeComputer()
    computer.interface = ChromeCookieInterface()
    env = judges.make_eval_env(computer, str(tmp_path))

    assert await scalecua_verify._repair_chrome_third_party_cookies_result(
        env,
        {"type": "third_party_cookies_blocked__75ed94d8e5e2ff083f81c247646b0b38"},
        "true",
    ) == "false"


@pytest.mark.asyncio
async def test_scalecua_third_party_cookie_repair_updates_dict_shapes(tmp_path):
    class ChromeCookieInterface(_FakeInterface):
        async def run_command(self, command: str, timeout=None):
            self.commands.append(command)
            self.command_calls.append(
                {"command": command, "timeout": timeout}
            )
            if _is_chrome_prefs_loader_command(command):
                stdout = json.dumps({"profile": {"block_third_party_cookies": True}})
            else:
                stdout = ""
            return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    computer = _FakeComputer()
    computer.interface = ChromeCookieInterface()
    env = judges.make_eval_env(computer, str(tmp_path))

    assert await scalecua_verify._repair_chrome_third_party_cookies_result(
        env,
        {"type": "block_third_party_cookies__d0393d13df595b6db99860dc4f30ea7b"},
        {"block_third_party_cookies": False},
    ) == {"block_third_party_cookies": True}
    assert await scalecua_verify._repair_chrome_third_party_cookies_result(
        env,
        {"type": "chrome_third_party_cookies__1394774d"},
        {"block_third_party": False},
    ) == {"block_third_party": True}


@pytest.mark.asyncio
async def test_scalecua_chrome_settings_snapshot_repairs_ui_only_prefs(tmp_path):
    snapshot = {
        "safebrowsing": {"enhanced": True, "enabled": False},
        "security": {
            "generated_https_first_mode_enabled": False,
            "profile_https_only_mode_enabled": True,
        },
        "cookies": {
            "default_content_setting_values_cookies": 4,
            "cookie_controls_mode": 1,
            "block_third_party_cookies": False,
        },
    }
    computer = _FakeComputer(stdout=json.dumps(snapshot))
    env = judges.make_eval_env(computer, str(tmp_path))

    assert await scalecua_verify._repair_chrome_settings_pref_result(
        env, "enable_enhanced_safety_browsing", "false"
    ) == "true"
    assert await scalecua_verify._repair_chrome_settings_pref_result(
        env, "enable_safe_browsing", "false"
    ) == "true"
    assert await scalecua_verify._repair_chrome_settings_pref_result(
        env, "data_delete_automacally", "false"
    ) == "true"
    assert await scalecua_verify._repair_chrome_settings_pref_result(
        env, "third_party_cookies_blocked", "false"
    ) == "true"
    assert await scalecua_verify._repair_chrome_settings_pref_result(
        env, "chrome_https_only_mode__bef59d93", "false"
    ) == "true"

    commands = "\n".join(computer.interface.commands)
    assert _is_chrome_settings_snapshot_command(commands)
    assert "chrome://settings/security" in commands
    assert "chrome://settings/content/siteData" in commands
    assert "chrome://settings/cookies" in commands


def test_scalecua_https_only_mode_reads_current_chrome_pref_shapes():
    assert (
        scalecua_verify._extract_scalecua_chrome_pref(
            "chrome_https_only_mode",
            {"generated": {"https_first_mode_enabled": True}},
        )
        == "true"
    )
    assert (
        scalecua_verify._extract_scalecua_chrome_pref(
            "https_only_mode",
            {"profile": {"https_only_mode_enabled": True}},
        )
        == "true"
    )
    assert (
        scalecua_verify._extract_scalecua_chrome_pref(
            "chrome_https_only_mode",
            {"generated": {"https_first_mode_enabled": False}},
        )
        == "false"
    )


def test_scalecua_third_party_cookie_snapshot_uses_selected_radio_text():
    assert (
        scalecua_verify._extract_chrome_settings_snapshot_result(
            "third_party_cookies_blocked__abc",
            {
                "selected_radio_texts": [
                    "Block third-party cookies Sites can't use your cookies"
                ],
                "cookies": {"cookie_controls_mode": 1},
            },
        )
        == "true"
    )
    assert (
        scalecua_verify._extract_chrome_settings_snapshot_result(
            "third_party_cookies_blocked__abc",
            {
                "selected_radio_texts": [
                    "Block third-party cookies in Incognito mode"
                ],
                "cookies": {"cookie_controls_mode": 2},
            },
        )
        == "false"
    )


@pytest.mark.asyncio
async def test_scalecua_chrome_profile_name_uses_settings_ui_snapshot(tmp_path):
    computer = _FakeComputer(stdout=json.dumps({"profile": {"name": "Sarah"}}))
    env = judges.make_eval_env(computer, str(tmp_path))

    assert await scalecua_verify._repair_chrome_profile_name_result(
        env, "profile_name", "Person 1"
    ) == "Sarah"
    assert _is_chrome_settings_snapshot_command(computer.interface.commands[0])
    assert "chrome://settings/manageProfile" in computer.interface.commands[0]


@pytest.mark.asyncio
async def test_scalecua_local_chrome_pref_getter_uses_ui_snapshot(tmp_path):
    computer = _FakeComputer(
        stdout=[
            json.dumps({"safebrowsing": {"enhanced": False, "enabled": False}}),
            json.dumps({"safebrowsing": {"enhanced": True, "enabled": False}}),
        ]
    )
    env = judges.make_eval_env(computer, str(tmp_path))

    out = await scalecua_verify._get_result(
        env,
        {"type": "enable_enhanced_safety_browsing__abc"},
        str(tmp_path),
        "train",
    )

    assert out == "true"
    commands = "\n".join(computer.interface.commands)
    assert "pkill -TERM chrome" not in commands
    assert _is_chrome_prefs_loader_command(commands)
    assert _is_chrome_settings_snapshot_command(commands)
    assert "chrome://settings/security" in commands


@pytest.mark.asyncio
async def test_scalecua_pre_postconfig_chrome_settings_snapshot_repairs_result(
    tmp_path,
):
    snapshot = {"safebrowsing": {"enhanced": True, "enabled": False}}
    computer = _FakeComputer(stdout=json.dumps(snapshot))
    evaluator = {
        "postconfig": [
            {"type": "launch", "parameters": {"command": ["pkill", "chrome"]}},
            {
                "type": "launch",
                "parameters": {"command": ["google-chrome", "--remote-debugging-port=1337"]},
            },
        ],
        "result": {"type": "enable_enhanced_safety_browsing__abc"},
    }

    state = await scalecua_verify._capture_pre_postconfig_state(computer, evaluator)

    assert state is not None
    assert scalecua_verify._CHROME_SETTINGS_SNAPSHOT_MARKER in state
    assert _is_chrome_settings_snapshot_command("\n".join(computer.interface.commands))
    assert (
        scalecua_verify._repair_chrome_settings_result_from_pre_postconfig_state(
            evaluator["result"],
            "false",
            state,
        )
        == "true"
    )


@pytest.mark.skipif(not _overlays_ready(), reason="lite.scalecua judge overlays not imported")
def test_scalecua_unwraps_generated_extension_version_expected_rule():
    expected = scalecua_verify._normalize_scalecua_expected_rules(
        "check_ext_version__b28217ae", {"expected": "1.0"}
    )
    metric = judges.resolve_metric("check_ext_version__b28217ae", "train")

    assert expected == "1.0"
    assert metric("1.0", expected) == 1.0


def test_scalecua_active_tab_metric_accepts_www_and_trailing_slash_only():
    metric = judges.resolve_metric("is_expected_active_tab", "train")
    expected = {
        "type": "url",
        "url": "https://www.dmv.virginia.gov/licenses-ids/real-id",
    }

    assert (
        metric("https://dmv.virginia.gov/licenses-ids/real-id/", expected)
        == 1.0
    )
    assert (
        metric("https://dmv.virginia.gov/vehicles/registration/exemp-disc-chart", expected)
        == 0
    )
    assert (
        metric(
            "https://dmv.virginia.gov/vehicles/registration",
            {"type": "url", "url": "https://www.dmv.virginia.gov/vehicles"},
        )
        == 1.0
    )
    assert (
        metric(
            "https://dmv.virginia.gov/vehicles/titles",
            {"type": "url", "url": "https://www.dmv.virginia.gov/vehicles"},
        )
        == 0.0
    )
    assert (
        metric(
            "https://www.flightaware.com/live/airport/delays",
            {"type": "url", "url": "https://www.flightaware.com/miserymap/"},
        )
        == 1.0
    )


def test_scalecua_chrome_internal_metric_accepts_settings_subpages():
    metric = judges.resolve_metric("is_expected_active_tab_approximate", "train")
    expected = {"type": "url", "url": "chrome://settings/"}

    assert metric("chrome://settings/manageProfile", expected) == 1.0
    assert metric("chrome://history/", expected) == 0


@pytest.mark.asyncio
async def test_scalecua_active_url_uses_access_tree_not_base_cdp(monkeypatch, tmp_path):
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))

    async def fake_at_xml(computer):
        return "<xml/>"

    monkeypatch.setattr(scalecua_verify.base_runner, "_get_at_xml", fake_at_xml)
    monkeypatch.setattr(
        scalecua_verify.base_runner,
        "_extract_address_bar_url",
        lambda at_xml: "example.com/search?q=ok",
    )

    out = await scalecua_verify._get_result(
        env,
        {"type": "active_url_from_accessTree", "goto_prefix": "https://www."},
        str(tmp_path),
        "train",
    )

    assert out == "https://www.example.com/search?q=ok"
    assert not env.computer.interface.commands


@pytest.mark.asyncio
async def test_scalecua_active_url_falls_back_to_cdp_for_chrome_internal_page(
    monkeypatch, tmp_path
):
    pages = json.dumps(
        [
            {
                "type": "page",
                "title": "Bookmarks",
                "url": "chrome://bookmarks/",
            }
        ]
    )
    env = judges.make_eval_env(_FakeComputer(stdout=pages), str(tmp_path))

    async def fake_at_xml(computer):
        return "<xml/>"

    monkeypatch.setattr(scalecua_verify.base_runner, "_get_at_xml", fake_at_xml)
    monkeypatch.setattr(
        scalecua_verify.base_runner,
        "_extract_address_bar_url",
        lambda at_xml: "Search bookmarks",
    )

    out = await scalecua_verify._get_result(
        env,
        {"type": "active_url_from_accessTree", "goto_prefix": ""},
        str(tmp_path),
        "train",
    )

    assert out == "chrome://bookmarks/"
    assert env.computer.interface.commands == ["curl -s http://localhost:1337/json"]


@pytest.mark.asyncio
async def test_scalecua_active_url_canonicalizes_chrome_internal_page(
    monkeypatch, tmp_path
):
    pages = json.dumps(
        [
            {"type": "page", "title": "History", "url": "chrome://history/"},
            {
                "type": "page",
                "title": "Bookmarks",
                "url": "chrome://bookmarks/",
            }
        ]
    )
    env = judges.make_eval_env(_FakeComputer(stdout=pages), str(tmp_path))

    async def fake_at_xml(computer):
        return "<xml/>"

    monkeypatch.setattr(scalecua_verify.base_runner, "_get_at_xml", fake_at_xml)
    monkeypatch.setattr(
        scalecua_verify.base_runner,
        "_extract_address_bar_url",
        lambda at_xml: "chrome://bookmarks",
    )

    out = await scalecua_verify._get_result(
        env,
        {"type": "active_url_from_accessTree", "goto_prefix": ""},
        str(tmp_path),
        "train",
    )

    assert out == "chrome://bookmarks/"


@pytest.mark.asyncio
async def test_scalecua_active_url_no_at_does_not_use_internal_fallback_for_web_url(
    monkeypatch, tmp_path
):
    pages = json.dumps(
        [{"type": "page", "title": "Settings", "url": "chrome://settings/"}]
    )
    env = judges.make_eval_env(_FakeComputer(stdout=pages), str(tmp_path))

    async def fake_at_xml(computer):
        return None

    monkeypatch.setattr(scalecua_verify.base_runner, "_get_at_xml", fake_at_xml)

    out = await scalecua_verify._get_result(
        env,
        {"type": "active_url_from_accessTree", "goto_prefix": "https://www."},
        str(tmp_path),
        "train",
    )

    assert out is None
    assert not env.computer.interface.commands


@pytest.mark.asyncio
async def test_scalecua_active_url_internal_fallback_ignores_popup_and_ambiguity(
    monkeypatch, tmp_path
):
    pages = json.dumps(
        [
            {
                "type": "page",
                "title": "Popup",
                "url": "chrome://omnibox-popup.top-chrome/",
            },
            {"type": "page", "title": "Bookmarks", "url": "chrome://bookmarks/"},
            {"type": "page", "title": "History", "url": "chrome://history/"},
        ]
    )
    env = judges.make_eval_env(_FakeComputer(stdout=pages), str(tmp_path))

    async def fake_at_xml(computer):
        return "<xml/>"

    monkeypatch.setattr(scalecua_verify.base_runner, "_get_at_xml", fake_at_xml)
    monkeypatch.setattr(
        scalecua_verify.base_runner,
        "_extract_address_bar_url",
        lambda at_xml: None,
    )

    out = await scalecua_verify._get_result(
        env,
        {"type": "active_url_from_accessTree", "goto_prefix": ""},
        str(tmp_path),
        "train",
    )

    assert out is None


@pytest.mark.asyncio
async def test_scalecua_active_tab_info_loads_title_content_from_at_url(
    monkeypatch, tmp_path
):
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))

    async def fake_at_xml(computer):
        return "<xml/>"

    async def fake_page_info(computer, url):
        return {"title": "Title", "url": url, "content": "<html>ok</html>"}

    monkeypatch.setattr(scalecua_verify.base_runner, "_get_at_xml", fake_at_xml)
    monkeypatch.setattr(
        scalecua_verify.base_runner,
        "_extract_address_bar_url",
        lambda at_xml: "https://example.com/page",
    )
    monkeypatch.setattr(
        scalecua_verify.base_runner,
        "_get_page_info_via_cdp",
        fake_page_info,
    )

    out = await scalecua_verify._get_result(
        env,
        {"type": "active_tab_info"},
        str(tmp_path),
        "train",
    )

    assert out == {
        "title": "Title",
        "url": "https://example.com/page",
        "content": "<html>ok</html>",
    }


@pytest.mark.asyncio
async def test_scalecua_open_tabs_keeps_user_visible_chrome_pages(tmp_path):
    pages = json.dumps(
        [
            {"type": "page", "title": "Bookmarks", "url": "chrome://bookmarks/"},
            {
                "type": "page",
                "title": "Popup",
                "url": "chrome://omnibox-popup.top-chrome/",
            },
            {"type": "page", "title": "Site", "url": "https://example.com"},
        ]
    )
    env = judges.make_eval_env(_FakeComputer(stdout=pages), str(tmp_path))

    out = await scalecua_verify._get_result(
        env,
        {"type": "open_tabs_info"},
        str(tmp_path),
        "train",
    )

    assert out == [
        {"title": "Bookmarks", "url": "chrome://bookmarks/"},
        {"title": "Site", "url": "https://example.com"},
    ]


@pytest.mark.asyncio
async def test_scalecua_active_tab_url_parse_uses_access_tree_url(monkeypatch, tmp_path):
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))

    async def fake_at_xml(computer):
        return "<xml/>"

    monkeypatch.setattr(scalecua_verify.base_runner, "_get_at_xml", fake_at_xml)
    monkeypatch.setattr(
        scalecua_verify.base_runner,
        "_extract_address_bar_url",
        lambda at_xml: "https://apple.com/compare?modelList=a,b&other=x",
    )

    out = await scalecua_verify._get_result(
        env,
        {
            "type": "active_tab_url_parse",
            "parse_keys": ["modelList"],
            "split_list": True,
        },
        str(tmp_path),
        "train",
    )

    assert out == {"modelList": ["a", "b"]}


@pytest.mark.asyncio
async def test_scalecua_active_tab_url_parse_accepts_current_ryanair_schema(
    monkeypatch, tmp_path
):
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))
    url = (
        "https://www.ryanair.com/gb/en/trip/flights/select?"
        "originMac=STN&destinationMac=BCN&dateOut=2026-08-15&"
        "adults=1&teens=0&children=0&isReturn=false"
    )

    async def fake_at_xml(computer):
        return "<xml/>"

    monkeypatch.setattr(scalecua_verify.base_runner, "_get_at_xml", fake_at_xml)
    monkeypatch.setattr(
        scalecua_verify.base_runner,
        "_extract_address_bar_url",
        lambda at_xml: url,
    )

    out = await scalecua_verify._get_result(
        env,
        {
            "type": "active_tab_url_parse",
            "parse_keys": [
                "originIata",
                "destinationIata",
                "tpAdults",
                "tpTeens",
                "tpChildren",
                "tpStartDate",
                "isReturn",
            ],
            "replace": {"tpStartDate": "time"},
        },
        str(tmp_path),
        "train",
    )

    assert out == {
        "originIata": "STN",
        "destinationIata": "BCN",
        "tpAdults": "1",
        "tpTeens": "0",
        "tpChildren": "0",
        "time": "2026-08-15",
        "isReturn": "false",
    }


@pytest.mark.asyncio
async def test_scalecua_active_tab_html_parse_requires_matching_active_page(
    monkeypatch, tmp_path
):
    pages = json.dumps(
        [{"type": "page", "title": "Other", "url": "https://other.example"}]
    )
    env = judges.make_eval_env(_FakeComputer(stdout=pages), str(tmp_path))

    async def fake_active_url(eval_env, config):
        return "https://target.example"

    async def should_not_parse(*args, **kwargs):
        raise AssertionError("should not fall back to first CDP page")

    monkeypatch.setattr(
        scalecua_verify,
        "_get_active_url_from_access_tree",
        fake_active_url,
    )
    monkeypatch.setattr(
        scalecua_verify.base_runner,
        "_get_html_parse_via_cdp",
        should_not_parse,
    )

    out = await scalecua_verify._get_result(
        env,
        {"type": "active_tab_html_parse", "category": "class"},
        str(tmp_path),
        "train",
    )

    assert out == {}


@pytest.mark.asyncio
async def test_scalecua_chrome_prefs_use_lite_strict_startup_semantics(tmp_path):
    prefs = json.dumps(
        {
            "enable_do_not_track": True,
            "profile": {"default_content_setting_values": {"cookies": 4}},
            "safebrowsing": {"enabled": False, "enhanced": True},
        }
    )
    env = judges.make_eval_env(_FakeComputer(stdout=prefs), str(tmp_path))

    startup = await scalecua_verify._get_result(
        env,
        {"type": "new_startup_page"},
        str(tmp_path),
        "train",
    )
    delete_data = await scalecua_verify._get_result(
        env,
        {"type": "data_delete_automacally"},
        str(tmp_path),
        "train",
    )
    do_not_track = await scalecua_verify._get_result(
        env,
        {"type": "enable_do_not_track"},
        str(tmp_path),
        "train",
    )
    enhanced_safe_browsing = await scalecua_verify._get_result(
        env,
        {"type": "enable_enhanced_safety_browsing"},
        str(tmp_path),
        "train",
    )
    safe_browsing = await scalecua_verify._get_result(
        env,
        {"type": "enable_safe_browsing"},
        str(tmp_path),
        "train",
    )
    disable_safe_browsing = await scalecua_verify._get_result(
        env,
        {"type": "disable_safe_browsing"},
        str(tmp_path),
        "train",
    )

    assert startup == "false"
    assert delete_data == "true"
    assert do_not_track == "true"
    assert enhanced_safe_browsing == "true"
    assert safe_browsing == "true"
    assert disable_safe_browsing == "true"

    prefs_no_clear = json.dumps({"profile": {"default_content_setting_values": {}}})
    env_no_clear = judges.make_eval_env(
        _FakeComputer(stdout=prefs_no_clear),
        str(tmp_path),
    )
    assert (
        await scalecua_verify._get_result(
            env_no_clear,
            {"type": "data_delete_automacally"},
            str(tmp_path),
            "train",
        )
        == "false"
    )

    prefs_specific_pages = json.dumps(
        {
            "session": {
                "restore_on_startup": 4,
                "startup_urls": ["https://www.google.com/", "https://www.wikipedia.org/"],
            }
        }
    )
    env_specific_pages = judges.make_eval_env(
        _FakeComputer(stdout=prefs_specific_pages),
        str(tmp_path),
    )
    assert (
        await scalecua_verify._get_result(
            env_specific_pages,
            {"type": "new_startup_page"},
            str(tmp_path),
            "train",
        )
        == "false"
    )

    prefs_empty_specific_pages = json.dumps(
        {"session": {"restore_on_startup": 4, "startup_urls": []}}
    )
    env_empty_specific_pages = judges.make_eval_env(
        _FakeComputer(stdout=prefs_empty_specific_pages),
        str(tmp_path),
    )
    assert (
        await scalecua_verify._get_result(
            env_empty_specific_pages,
            {"type": "new_startup_page"},
            str(tmp_path),
            "train",
        )
        == "false"
    )

    prefs_new_tab = json.dumps({"session": {"restore_on_startup": 5}})
    env_new_tab = judges.make_eval_env(
        _FakeComputer(stdout=prefs_new_tab),
        str(tmp_path),
    )
    assert (
        await scalecua_verify._get_result(
            env_new_tab,
            {"type": "new_startup_page"},
            str(tmp_path),
            "train",
        )
        == "true"
    )

    prefs_disabled = json.dumps({"safebrowsing": {"enabled": False, "enhanced": False}})
    env_disabled = judges.make_eval_env(
        _FakeComputer(stdout=prefs_disabled),
        str(tmp_path),
    )
    assert (
        await scalecua_verify._get_result(
            env_disabled,
            {"type": "disable_safe_browsing"},
            str(tmp_path),
            "train",
        )
        == "false"
    )

    prefs_continue = json.dumps({"session": {"restore_on_startup": 1}})
    env_continue = judges.make_eval_env(
        _FakeComputer(stdout=prefs_continue),
        str(tmp_path),
    )
    assert (
        await scalecua_verify._get_result(
            env_continue,
            {"type": "new_startup_page"},
            str(tmp_path),
            "train",
        )
        == "false"
    )


@pytest.mark.asyncio
async def test_scalecua_chrome_hashed_pref_getters_route_to_local_semantics(tmp_path):
    prefs = json.dumps(
        {
            "browser": {"clear_data": {"browsing_data_lifetime": {"enabled": True}}},
            "credentials_enable_service": False,
            "profile": {"cookie_controls_mode": 1},
        }
    )
    env = judges.make_eval_env(_FakeComputer(stdout=prefs), str(tmp_path))

    assert (
        await scalecua_verify._get_result(
            env,
            {"type": "data_delete_automacally__2c85759ee3c771990524833aa219e88b"},
            str(tmp_path),
            "train",
        )
        == "true"
    )
    assert (
        await scalecua_verify._get_result(
            env,
            {"type": "third_party_cookies_blocked__ab79f9e8"},
            str(tmp_path),
            "train",
        )
        == "true"
    )
    assert (
        await scalecua_verify._get_result(
            env,
            {"type": "password_manager_disabled__f026d0"},
            str(tmp_path),
            "train",
        )
        == "true"
    )
    assert (
        await scalecua_verify._get_result(
            env,
            {"type": "password_manager_enabled__f026d0"},
            str(tmp_path),
            "train",
        )
        == "false"
    )

    incognito_only = json.dumps(
        {"profile": {"cookie_controls_mode": 2, "default_content_setting_values": {}}}
    )
    env_incognito_only = judges.make_eval_env(
        _FakeComputer(stdout=incognito_only),
        str(tmp_path),
    )
    assert (
        await scalecua_verify._get_result(
            env_incognito_only,
            {"type": "third_party_cookies_blocked__ab79f9e8"},
            str(tmp_path),
            "train",
        )
        == "false"
    )
    assert (
        await scalecua_verify._get_result(
            env_incognito_only,
            {"type": "data_delete_automacally__2c85759ee3c771990524833aa219e88b"},
            str(tmp_path),
            "train",
        )
        == "false"
    )


def test_scalecua_chrome_homepage_result_ignores_trailing_slash():
    assert (
        scalecua_verify._normalize_scalecua_result(
            "chrome_homepage__homepage_trailing_slash",
            "https://www.wikipedia.org/",
        )
        == "wikipedia.org"
    )
    result, expected = scalecua_verify._normalize_chrome_url_metric_pair(
        {"type": "chrome_homepage__homepage_trailing_slash"},
        "http://wikipedia.org/",
        {"expected": "https://www.wikipedia.org"},
    )
    assert result == expected["expected"] == "wikipedia.org"


@pytest.mark.asyncio
async def test_scalecua_chrome_prefs_loader_uses_active_or_newest_profile_command(tmp_path):
    prefs = json.dumps({"safebrowsing": {"enhanced": True}})
    computer = _FakeComputer(stdout=prefs)
    env = judges.make_eval_env(computer, str(tmp_path))

    out = await scalecua_verify._get_result(
        env,
        {"type": "enable_enhanced_safety_browsing"},
        str(tmp_path),
        "train",
    )

    assert out == "true"
    command = computer.interface.commands[-1]
    assert "pkill -TERM chrome" not in "\n".join(computer.interface.commands)
    assert "ps\", \"-eo\", \"args=\"" in command
    assert "--user-data-dir" in command
    assert "/home/user/chrome-data/Default/Preferences" in command
    assert "/home/user/.config/google-chrome/Default/Preferences" in command


@pytest.mark.asyncio
@pytest.mark.skipif(not _overlays_ready(), reason="lite.scalecua judge overlays not imported")
async def test_scalecua_generated_chrome_pref_getters_use_lite_profile(tmp_path):
    prefs = json.dumps(
        {
            "profile": {"default_content_setting_values": {"cookies": 4}},
            "safebrowsing": {"enabled": False, "enhanced": True},
        }
    ).encode("utf-8")
    env = judges.make_eval_env(
        _FakeComputer(files={"/home/user/chrome-data/Default/Preferences": prefs}),
        str(tmp_path),
    )

    checks = {
        "data_delete_automacally__2c85759ee3c771990524833aa219e88b": "true",
        "enable_enhanced_safety_browsing__6cfa1daf7e75edcbbc01473bc020676e": "true",
        "enable_safe_browsing__c066cfbb11182ed68d7b1c1e09f2d5ad": "true",
        "disable_safe_browsing__eefbb2e3e4c8eef9ec6d58df9ab495a3": "true",
    }
    for result_type, expected in checks.items():
        getter = judges.resolve_getter(result_type, "train")
        assert getter is not None
        assert (
            await judges.call_overlay_getter(
                getter,
                env,
                {"type": result_type},
                str(tmp_path),
            )
            == expected
        )

    disabled = json.dumps({"safebrowsing": {"enabled": False, "enhanced": False}}).encode(
        "utf-8"
    )
    disabled_env = judges.make_eval_env(
        _FakeComputer(files={"/home/user/chrome-data/Default/Preferences": disabled}),
        str(tmp_path),
    )
    getter = judges.resolve_getter(
        "disable_safe_browsing__eefbb2e3e4c8eef9ec6d58df9ab495a3",
        "train",
    )
    assert getter is not None
    assert (
        await judges.call_overlay_getter(
            getter,
            disabled_env,
            {"type": "disable_safe_browsing__eefbb2e3e4c8eef9ec6d58df9ab495a3"},
            str(tmp_path),
        )
        == "false"
    )


@pytest.mark.asyncio
async def test_scalecua_generated_chrome_color_scheme_getters_use_lite_profile(tmp_path):
    # #155 #8 (canonical-first): the canonical `color_scheme` wins over the stale
    # `color_scheme2` mirror, so color_scheme=2 -> dark (was: mirror-first -> system).
    prefs = json.dumps(
        {
            "browser": {"theme": {"color_scheme": 2, "color_scheme2": 0}},
            "enable_do_not_track": True,
        }
    ).encode("utf-8")
    env = judges.make_eval_env(
        _FakeComputer(files={"/home/user/chrome-data/Default/Preferences": prefs}),
        str(tmp_path),
    )

    color_getter = judges.resolve_getter(
        "chrome_color_scheme__c109c66a806aa3929078e7fafeda76b2_qw35sft2_d1928ab1",
        "rl",
    )
    assert color_getter is not None
    assert (
        await judges.call_overlay_getter(
            color_getter,
            env,
            {
                "type": "chrome_color_scheme__c109c66a806aa3929078e7fafeda76b2_qw35sft2_d1928ab1"
            },
            str(tmp_path),
        )
        == "dark"
    )

    appearance_getter = judges.resolve_getter(
        "chrome_appearance_and_dnt__197c5f21c248c0028a67e57d9193addd_qw35sft2_0f0a9cca",
        "rl",
    )
    assert appearance_getter is not None
    assert await judges.call_overlay_getter(
        appearance_getter,
        env,
        {
            "type": "chrome_appearance_and_dnt__197c5f21c248c0028a67e57d9193addd_qw35sft2_0f0a9cca"
        },
        str(tmp_path),
    ) == {"color_scheme": 2, "do_not_track": True}


def test_scalecua_extension_manifest_reroot_config_paths():
    # #154 RC-FN-11: re-root every VM path in a config from the (empty) expected
    # extension dir onto the actual one-deeper unpacked dir; leave unrelated paths.
    from lite.gym.envs.lite.scalecua.src.osworld import judges as J
    old = "/home/user/Projects/page-modifier"
    new = "/home/user/Projects/page-modifier/page-modifier"
    cfg = {
        "manifest_path": old + "/manifest.json",
        "required_files": [old + "/background.js", old + "/icon.png"],
        "unrelated": "/home/user/other/file.txt",
        "exact": old,
    }
    out = J._reroot_config_paths(cfg, old, new)
    assert out["manifest_path"] == new + "/manifest.json"
    assert out["required_files"] == [new + "/background.js", new + "/icon.png"]
    assert out["unrelated"] == "/home/user/other/file.txt"   # untouched
    assert out["exact"] == new


def test_scalecua_extension_manifest_family_prefix_gate():
    # The depth-tolerant override installs by NAME-PREFIX over the whole family,
    # never a per-hash allow-list.
    from types import ModuleType

    from lite.gym.envs.lite.scalecua.src.osworld import judges as J
    m = ModuleType("fake_getters")
    hits = []

    def _mk(name):
        def g(env, config):
            hits.append((name, config))
            return {}
        g.__name__ = name
        return g

    for n in ("get_extension_manifest__abc123", "get_chrome_extension_manifest__def456",
              "get_webext_manifest__0f0f0f", "get_unrelated_thing__999"):
        setattr(m, n, _mk(n))
    J._install_extension_manifest_family_overrides(m)
    # the 3 manifest-family getters are wrapped (depth-tolerant); the unrelated is not.
    assert getattr(m, "get_extension_manifest__abc123")._scalecua_depth_tolerant is True
    assert getattr(m, "get_chrome_extension_manifest__def456")._scalecua_depth_tolerant is True
    assert getattr(m, "get_webext_manifest__0f0f0f")._scalecua_depth_tolerant is True
    assert not getattr(getattr(m, "get_unrelated_thing__999"), "_scalecua_depth_tolerant", False)


def test_scalecua_command_result_aliases_chrome_profile_before_running():
    """`_ControllerShim._command_result` must alias Chrome-profile paths.

    The shim stubs `_run`, so the only thing this test can observe is the
    command string that reaches `interface.run_command` — and the ONLY
    substantive transform `_command_result` applies to it is
    `_alias_chrome_profile_path`. A getter-shaped command ("xsel ...") passes
    through unchanged, so asserting on one cannot see that transform at all
    (deleting the alias call left the whole module green). The second half
    below is the biting assertion; the first only pins the passthrough +
    stdout-cleaning shape.
    """
    from lite.gym.envs.lite.scalecua.src.osworld.judges import _ControllerShim

    class _Loop:
        pass

    class _Interface:
        def __init__(self):
            self.command_calls = []

        def run_command(self, command, timeout=None):
            self.command_calls.append(
                {"command": command, "timeout": timeout}
            )
            return SimpleNamespace(stdout="clipboard", stderr="", returncode=0)

    class _Computer:
        def __init__(self):
            self.interface = _Interface()

    computer = _Computer()
    shim = _ControllerShim(computer, str(Path("/tmp/cache")), _Loop())

    def run(result, *, timeout=None):
        del timeout
        return result

    shim._run = run
    result = shim._command_result("xsel --clipboard --output")

    assert str(result) == "clipboard"
    assert computer.interface.command_calls[-1]["command"] == "xsel --clipboard --output"

    # The official OSWorld getters read the VM's stock Chrome profile; lite
    # launches Chrome on /home/user/chrome-data, so every command crossing this
    # shim must be rewritten or the getter reads an empty/absent profile.
    shim._command_result("cat /home/user/.config/google-chrome/Default/Preferences")
    assert computer.interface.command_calls[-1]["command"] == (
        "cat /home/user/chrome-data/Default/Preferences"
    )
