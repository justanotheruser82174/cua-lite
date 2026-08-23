"""ScaleCUA VM path and command repair tests."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from lite.gym.envs.lite.scalecua.src.osworld import judges
from lite.gym.envs.lite.scalecua.src.osworld import verify as scalecua_verify


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
        self.command_calls.append({"command": command, "timeout": timeout})
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


def test_scalecua_vm_user_home_alias_covers_host_expanded_paths():
    assert (
        judges._alias_vm_user_home_path("/root/.config/GIMP/2.10/gimprc")
        == "/home/user/.config/GIMP/2.10/gimprc"
    )
    assert (
        judges._alias_vm_user_home_path("/home/ubuntu/Documents/report.txt")
        == "/home/user/Documents/report.txt"
    )
    host_home = str(Path.home()).rstrip("/")
    if host_home != "/home/user":
        assert judges._alias_vm_user_home_path(f"{host_home}/Desktop/a.txt") == (
            "/home/user/Desktop/a.txt"
        )


def test_scalecua_thunderbird_profile_find_prefers_default_release():
    command = (
        "find /home/user/.thunderbird -maxdepth 1 -mindepth 1 -type d "
        '2>/dev/null | grep -v "Crash Reports" | head -1'
    )

    rewritten = judges._alias_chrome_profile_path(command)

    assert '-name "*.default-release"' in rewritten
    assert 'grep -v "Crash Reports"' not in rewritten


@pytest.mark.asyncio
async def test_scalecua_overlay_file_path_getter_downloads_to_cache(tmp_path):
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))

    def getter(file_path, eval_env, config):
        assert file_path.endswith("book.xlsx")
        assert eval_env.cache_dir == str(tmp_path)
        return open(file_path, "rb").read()

    out = await judges.call_overlay_getter(
        getter,
        env,
        {"type": "xlsx_custom", "path": "/home/user/Desktop/book.xlsx"},
        str(tmp_path),
    )
    assert out == b"bytes:/home/user/Desktop/book.xlsx"


@pytest.mark.asyncio
async def test_scalecua_overlay_direct_config_path_getter_materializes_vm_file(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    source = tmp_path / "source.xlsx"
    wb = openpyxl.Workbook()
    wb.active.title = "Visible"
    wb.create_sheet("Copied")
    wb.save(source)
    wb.close()

    remote_path = "/home/user/copy_sheet_insert.xlsx"
    computer = _FakeComputer(files={remote_path: source.read_bytes()})
    env = judges.make_eval_env(computer, str(tmp_path))
    config = {"type": "xlsx_sheet_names__test", "path": remote_path}

    def getter(eval_env, config):
        wb = openpyxl.load_workbook(config["path"], data_only=True)
        try:
            return wb.sheetnames
        finally:
            wb.close()

    out = await judges.call_overlay_getter(getter, env, config, str(tmp_path))

    assert out == ["Visible", "Copied"]
    assert config["path"] == remote_path
    cached = list((tmp_path / "_overlay_inputs").glob("copy_sheet_insert.*.xlsx"))
    assert len(cached) == 1


@pytest.mark.asyncio
async def test_scalecua_overlay_csv_config_path_getter_materializes_vm_file(tmp_path):
    remote_path = "/home/user/Desktop/table.csv"
    computer = _FakeComputer(files={remote_path: b"name,score\nAda,99\n"})
    env = judges.make_eval_env(computer, str(tmp_path))

    def getter(eval_env, config):
        with open(config["csv_path"], encoding="utf-8") as handle:
            return handle.read()

    out = await judges.call_overlay_getter(
        getter,
        env,
        {"type": "csv_custom", "csv_path": remote_path},
        str(tmp_path),
    )

    assert out == "name,score\nAda,99\n"
    cached = list((tmp_path / "_overlay_inputs").glob("table.*.csv"))
    assert len(cached) == 1


@pytest.mark.asyncio
async def test_scalecua_overlay_controller_file_getter_keeps_vm_path(tmp_path):
    remote_path = "/home/user/Desktop/book.xlsx"
    computer = _FakeComputer(files={remote_path: b"book-bytes"})
    env = judges.make_eval_env(computer, str(tmp_path))
    seen = {}

    def getter(eval_env, config):
        seen["path"] = config["path"]
        return eval_env.controller.get_file(config["path"])

    out = await judges.call_overlay_getter(
        getter,
        env,
        {"type": "custom", "path": remote_path},
        str(tmp_path),
    )

    assert out == b"book-bytes"
    assert seen["path"] == remote_path


@pytest.mark.asyncio
async def test_scalecua_overlay_get_vm_file_getter_keeps_vm_path(tmp_path):
    remote_path = "/home/user/Desktop/book.xlsx"
    computer = _FakeComputer(files={remote_path: b"book-bytes"})
    env = judges.make_eval_env(computer, str(tmp_path))
    seen = {}

    def get_vm_file(eval_env, config):
        seen["path"] = config["path"]
        return eval_env.controller.get_file(config["path"])

    def getter(eval_env, config):
        return get_vm_file(eval_env, {"path": config["path"]})

    out = await judges.call_overlay_getter(
        getter,
        env,
        {"type": "custom", "path": remote_path},
        str(tmp_path),
    )

    assert out == b"book-bytes"
    assert seen["path"] == remote_path
    assert not (tmp_path / "_overlay_inputs").exists()


@pytest.mark.asyncio
async def test_scalecua_overlay_command_string_file_api_keeps_vm_path(tmp_path):
    remote_path = "/home/user/Desktop/image.png"
    computer = _FakeComputer(files={remote_path: b"image-bytes"})
    env = judges.make_eval_env(computer, str(tmp_path))

    def getter(eval_env, config):
        eval_env.controller.run_bash_script(
            f"python3 - <<'PY'\nfrom PIL import Image\nImage.open({config['path']!r})\nPY"
        )
        return config["path"]

    out = await judges.call_overlay_getter(
        getter,
        env,
        {"type": "custom", "path": remote_path},
        str(tmp_path),
    )

    assert out == remote_path
    assert remote_path in computer.interface.commands[0]
    assert not (tmp_path / "_overlay_inputs").exists()
    assert not (tmp_path / "_overlay_inputs").exists()


def test_scalecua_normalizes_vm_python_binary_to_match_osworld_vm():
    assert scalecua_verify._normalize_python_command(
        ["python", "-c", "import sklearn"],
        shell=False,
    ) == ["python3", "-c", "import sklearn"]
    assert (
        scalecua_verify._normalize_python_command(
            "python -c 'print(1)'",
            shell=False,
        )
        == "python3 -c 'print(1)'"
    )
    assert (
        scalecua_verify._normalize_python_command(
            "python -c 'print(1)'",
            shell=True,
        )
        == "python -c 'print(1)'"
    )


def test_scalecua_repairs_generated_git_root_commit_diff_tree_command():
    command = [
        "/bin/bash",
        "-c",
        'first_files=$(git diff-tree --no-commit-id --name-only -r "$first_hash" 2>/dev/null); '
        'if [ "$first_files" = "README.md" ]; then echo "first_commit_only_readme"; fi',
    ]

    fixed = scalecua_verify._normalize_scalecua_command_config(
        {"type": "vm_command_line", "command": command}
    )

    assert fixed["command"][0:2] == ["/bin/bash", "-c"]
    assert "git diff-tree --root --no-commit-id --name-only -r" in fixed["command"][2]


def test_scalecua_git_root_commit_diff_tree_repair_is_narrow():
    command = [
        "/bin/bash",
        "-c",
        'files=$(git diff-tree --no-commit-id --name-only -r "$hash" 2>/dev/null)',
    ]

    fixed = scalecua_verify._normalize_scalecua_command_config(
        {"type": "vm_command_line", "command": command}
    )

    assert fixed["command"] is command


@pytest.mark.asyncio
async def test_scalecua_vm_command_error_uses_python3_vm_alias(monkeypatch, tmp_path):
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))
    seen = {}

    async def fake_base_get_result(computer, config, cache_dir):
        seen["config"] = config
        return ""

    monkeypatch.setattr(scalecua_verify.base_runner, "_get_result", fake_base_get_result)

    await scalecua_verify._get_result(
        env,
        {"type": "vm_command_error", "command": ["python", "-c", "import sklearn"]},
        str(tmp_path),
        "train",
    )

    assert seen["config"]["command"] == ["python3", "-c", "import sklearn"]


@pytest.mark.asyncio
async def test_scalecua_django_file_count_getter_is_repo_aware(tmp_path):
    computer = _FakeComputer(stdout="42\n")
    env = judges.make_eval_env(computer, str(tmp_path))

    out = await scalecua_verify._get_result(
        env,
        {
            "type": "dir_file_count__00db2192",
            "dir_path": "/home/user/django",
            "pattern": "*.py",
        },
        str(tmp_path),
        "train",
    )

    assert out == 42
    command = computer.interface.commands[0]
    assert "/home/user/django" in command
    assert ".git" in command
    assert "github.com/django/django" in command
    assert 'find "$dir" -type f -name ' in command
    assert "*.py" in command
    assert "-maxdepth 1" not in command


def test_scalecua_does_not_synthesize_timedatectl_status():
    source = Path(scalecua_verify.__file__).read_text()
    dockerfile = (
        Path(__file__).resolve().parents[5] / "lite/gym/envs/lite/osworld/docker/Dockerfile"
    ).read_text()

    assert "_synthetic_timedatectl_status" not in source
    assert "_repair_timedatectl_status_result" not in source
    assert "COPY --chmod=0755 docker/bin/timedatectl /usr/local/bin/timedatectl" in dockerfile


@pytest.mark.asyncio
async def test_scalecua_vm_file_is_path_only_and_allows_zero_byte_files(tmp_path):
    class ZeroComputer(_FakeComputer):
        def __init__(self):
            super().__init__()

        @property
        def interface(self):
            return self._interface

        @interface.setter
        def interface(self, value):
            self._interface = value

    class ZeroInterface(_FakeInterface):
        async def read_bytes(self, path: str) -> bytes:
            return b""

    computer = ZeroComputer()
    computer.interface = ZeroInterface()
    env = judges.make_eval_env(computer, str(tmp_path))

    out = await scalecua_verify._get_result(
        env,
        {"type": "vm_file", "path": "/home/user/Desktop/zero.txt", "dest": "zero.txt"},
        str(tmp_path),
        "train",
    )

    assert out == str(tmp_path / "zero.txt")
    assert (tmp_path / "zero.txt").exists()
    assert (tmp_path / "zero.txt").read_bytes() == b""


@pytest.mark.asyncio
async def test_scalecua_expected_vm_file_uses_dest_for_single_file(tmp_path):
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))

    out = await scalecua_verify._get_expected(
        env,
        {
            "type": "vm_file",
            "path": "/home/user/Desktop/Early Buildings.tex",
            "dest": "download.tex",
        },
        str(tmp_path),
        "train",
    )

    assert out == str(tmp_path / "download.tex")
    assert (
        tmp_path / "download.tex"
    ).read_bytes() == b"bytes:/home/user/Desktop/Early Buildings.tex"


@pytest.mark.asyncio
async def test_scalecua_expected_rule_materializes_vm_reference_paths(tmp_path):
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))

    out = await scalecua_verify._get_expected(
        env,
        {
            "type": "rule",
            "rules": {
                "source_path": "/home/user/Desktop/tilearray.png",
                "crop_box": [0, 320, 962, 960],
            },
        },
        str(tmp_path),
        "train",
    )

    assert out["crop_box"] == [0, 320, 962, 960]
    assert out["source_path"].startswith(str(tmp_path / "_reference_assets"))
    assert out["source_path"].endswith(".png")
    assert open(out["source_path"], "rb").read() == b"bytes:/home/user/Desktop/tilearray.png"


@pytest.mark.asyncio
async def test_scalecua_expected_rule_with_relative_time_is_resolved(monkeypatch, tmp_path):
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))
    calls = []

    def fake_resolve(config):
        calls.append(config)
        return {
            "expected": {
                "start": "LAX",
                "end": "SFO",
                "time": "Sat, Jul 18, 2026",
            },
            "relativeTime": {"from": "tomorrow"},
        }

    monkeypatch.setattr(scalecua_verify, "_resolve_relative_time", fake_resolve)

    out = await scalecua_verify._get_expected(
        env,
        {
            "type": "rule",
            "rules": {
                "expected": {
                    "start": "LAX",
                    "end": "SFO",
                    "time": "{DoW}, {Month} {Day0D}, {Year}",
                },
                "relativeTime": {"from": "tomorrow"},
            },
        },
        str(tmp_path),
        "train",
    )

    assert calls
    assert out["expected"]["time"] == "Sat, Jul 18, 2026"
    assert "{DoW}" not in out["expected"]["time"]


@pytest.mark.asyncio
async def test_scalecua_expected_rule_materializes_gimp_src_and_target_paths(tmp_path):
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))

    out = await scalecua_verify._get_expected(
        env,
        {
            "type": "rule",
            "rules": {
                "src_path": "/home/user/Desktop/white_background_with_object.png",
                "tgt_path": "/home/user/Desktop/red_background_with_object.png",
            },
        },
        str(tmp_path),
        "train",
    )

    assert out["src_path"].startswith(str(tmp_path / "_reference_assets"))
    assert out["tgt_path"].startswith(str(tmp_path / "_reference_assets"))
    assert (
        Path(out["src_path"]).read_bytes()
        == b"bytes:/home/user/Desktop/white_background_with_object.png"
    )
    assert (
        Path(out["tgt_path"]).read_bytes()
        == b"bytes:/home/user/Desktop/red_background_with_object.png"
    )


@pytest.mark.asyncio
async def test_scalecua_expected_rule_author_cache_without_pinned_url_stays_unmaterialized(
    tmp_path,
):
    # No pinned source URL: the reference must NOT be recovered by basename
    # search over agent-writable VM dirs (a policy could plant a same-name
    # file equal to its own output). Fail closed instead.
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))

    author_path = (
        "/home/lvbowen/project/AutoGen/src/envs/osworld_env/cache/"
        "e8172110-ec08-421b-a6f5-842e6451911f/character.png"
    )
    out = await scalecua_verify._get_expected(
        env,
        {"type": "rule", "rules": {"original_path": author_path}},
        str(tmp_path),
        "train",
    )

    assert out["original_path"] == author_path


@pytest.mark.asyncio
async def test_scalecua_expected_rule_prefers_setup_source_for_author_cache(monkeypatch, tmp_path):
    env = judges.make_eval_env(
        _FakeComputer(files={"/home/user/Desktop/heron.jpeg": b"mutated-vm-file"}),
        str(tmp_path),
    )
    url = (
        "https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/"
        "711e0811642364e7aa8f10a8918367d0b626d578/gimp/"
        "58d3eeeb-e9d0-499f-962e-fd0db2a744d8/heron.jpeg"
    )
    env._scalecua_reference_source_urls = {
        "58d3eeeb-e9d0-499f-962e-fd0db2a744d8/heron.jpeg": [url],
    }
    monkeypatch.setattr(
        scalecua_verify,
        "_read_reference_source_url",
        lambda actual_url: b"pristine-source" if actual_url == url else None,
    )

    out = await scalecua_verify._get_expected(
        env,
        {
            "type": "rule",
            "rules": {
                "original_path": (
                    "/home/lvbowen/project/SCALE-CUA/VeriGen/osworld_env/cache/"
                    "58d3eeeb-e9d0-499f-962e-fd0db2a744d8/heron.jpeg"
                )
            },
        },
        str(tmp_path),
        "rl",
    )

    assert out["original_path"].startswith(str(tmp_path / "_reference_assets"))
    assert open(out["original_path"], "rb").read() == b"pristine-source"


@pytest.mark.asyncio
async def test_scalecua_result_config_author_source_cache_path_without_url_stays_unmaterialized(
    monkeypatch, tmp_path
):
    # Same fail-closed contract as the expected-side test above: without a
    # pinned URL the author-cache path passes through unmaterialized.
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))

    def getter(eval_env, config):
        return {"source_path": config["source_cache_path"]}

    monkeypatch.setattr(scalecua_verify.judges, "resolve_getter", lambda *_: getter)

    author_path = (
        "/home/lvbowen/project/AutoGen/src/envs/osworld_env/cache/"
        "d68204bf-11c1-4b13-b48b-d303c73d4bf6/tilearray.png"
    )
    out = await scalecua_verify._get_result(
        env,
        {
            "type": "image_flip_check__08f973926b8f35af5489796c73a6c6e0",
            "path": "/home/user/Desktop/flipped_horizontal.png",
            "source_cache_path": author_path,
        },
        str(tmp_path),
        "train",
    )

    assert out["source_path"] == author_path


def test_scalecua_rcfn15_currency_and_autocorrect_wrappers():
    # #154 RC-FN-15: scoped string normalization wrappers — recover the FN while
    # a verbatim-glyph/verbatim-currency gold stays EXACT (asymmetric guard).
    from lite.gym.envs.lite.scalecua.src.osworld import judges as J

    def m_num(result, expected, **o):
        gold = expected.get("value") if isinstance(expected, dict) else expected
        return 1.0 if str(result).strip() == str(gold).strip() else 0.0

    w = J._wrap_text_file_value_currency_metric(m_num)
    assert w("$4.16", {"value": "4.16"}) == 1.0  # 0->1 recovery ($-prefix)
    assert w("$5.16", {"value": "4.16"}) == 0.0  # neg: wrong number
    assert w("4.16", {"value": "$4.16"}) == 0.0  # neg: verbatim-$ gold stays exact

    def m_txt(result, expected, **o):
        gold = expected.get("value") if isinstance(expected, dict) else expected
        r = result.get("value") if isinstance(result, dict) else result
        return 1.0 if str(r) == str(gold) else 0.0

    wa = J._wrap_autocorrect_text_metric(m_txt)
    assert wa("A – B", "A - B") == 1.0  # 0->1: fold en-dash in ACTUAL
    assert wa("A - B", "A – B") == 0.0  # neg: gold HAS en-dash -> exact preserved
    assert wa("X – Y", "A - B") == 0.0  # neg: wrong text


def test_window_title_getters_wmctrl_hardened():
    """Window-title getters must run wmctrl before xdotool."""
    from lite.gym.envs.lite.scalecua.src.osworld import judges as J

    h = J._harden_window_title_command
    compose = "xdotool search --name 'Write:' getwindowname 2>/dev/null | head -1"
    vscode = (
        "xdotool search --onlyvisible --class 'Code' 2>/dev/null | head -10 | "
        'while read wid; do xdotool getwindowname "$wid" 2>/dev/null; done'
    )
    for cmd, needle in (
        (compose, "grep -F -m1 -e Write:"),
        (vscode, "grep -F -e 'Visual Studio Code'"),
    ):
        out = h(cmd)
        assert out.startswith("wmctrl -l 2>/dev/null |"), out  # wmctrl first
        assert needle in out  # right name pattern
        assert f"|| {{ {cmd}; }}" in out  # xdotool fallback kept
    # already-wmctrl getter and non-window commands are untouched (no double-rewrite)
    working = (
        "wmctrl -l 2>/dev/null | grep -i 'Visual Studio Code' || "
        "xdotool search --name 'Visual Studio Code' getwindowname 2>/dev/null"
    )
    assert h(working) == working
    assert (
        h("gsettings get org.gnome.desktop.session idle-delay")
        == "gsettings get org.gnome.desktop.session idle-delay"
    )


def test_window_title_family_override_wraps_getwindowname_getters():
    """The family installer wraps exactly the getwindowname getters, idempotently."""
    from types import ModuleType

    from lite.gym.envs.lite.scalecua.src.osworld import judges as J

    def get_win(env, config):
        return env.controller.run_bash_script("xdotool search --name 'X' getwindowname")

    def get_other(env, config):
        return env.controller.run_bash_script("gsettings get x y")

    mod = ModuleType("fake_verigen_getters")
    mod.get_win, mod.get_other = get_win, get_other
    J._install_window_title_family_overrides(mod)
    assert getattr(mod.get_win, "_scalecua_wmctrl_hardened", False)
    assert not getattr(mod.get_other, "_scalecua_wmctrl_hardened", False)
    J._install_window_title_family_overrides(mod)  # idempotent (no double-wrap)
    assert getattr(mod.get_win, "_scalecua_wmctrl_hardened", False)


def test_scalecua_alias_grep_hardening():
    # scalecua parity with osworld os_05: rewrite brittle baked bashrc alias-grep
    # evaluators to a quote/flag-order-tolerant canonical form; leave robust
    # siblings + non-alias commands untouched (no new FP surface).
    from lite.gym.envs.lite.scalecua.src.osworld import verify as V

    h = V._harden_alias_check_command
    c1 = "grep -q \"alias ll='ls -la'\" ~/.bashrc && echo 1 || echo 0"
    out1 = h(c1)
    assert out1 != c1 and out1.startswith("bash -c ")  # literal -> rewritten
    assert "sort -u" in out1  # ls flag-set canonicalization
    c2 = 'grep -qE "alias python=(\\")?python3(\\")? *$" ~/.bashrc && echo 1 || echo 0'
    assert h(c2) != c2  # optquote -> rewritten
    # robust sibling (['\"]? form) is NOT one of the two brittle shapes -> untouched
    c3 = (
        'grep -qE "^[[:space:]]*alias[[:space:]]+python=[\'\\"]?python3[\'\\"]?" '
        "~/.bashrc && echo 1 || echo 0"
    )
    assert h(c3) == c3
    assert (
        h("test -d /home/user/x && echo 1 || echo 0") == "test -d /home/user/x && echo 1 || echo 0"
    )
    assert h(None) is None  # non-str passthrough


def test_scalecua_alias_check_runs_against_guarded_bashrc():
    # #154 R4: the hardened alias check must WORK against a real Ubuntu ~/.bashrc,
    # which opens with the `case $- in *i*) ;; *) return;; esac` non-interactive
    # guard. The old `source ~/.bashrc` form `return`ed before the alias line
    # under `bash -c` -> alias never defined -> every alias task FALSE-NEGATIVED.
    # This EXECUTES the rewritten command (the string-only test above missed it).
    import os
    import tempfile

    from lite.gym.envs.lite.scalecua.src.osworld.verify import _build_alias_check_command

    GUARD = "# ~/.bashrc\ncase $- in\n    *i*) ;;\n      *) return;;\nesac\n"

    def score(rc_tail, name, body):
        with tempfile.NamedTemporaryFile("w", suffix=".bashrc", delete=False) as f:
            f.write(GUARD + rc_tail)
            path = f.name
        try:
            cmd = _build_alias_check_command(name, body, path)
            return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
        finally:
            os.unlink(path)

    # literal branch — the case the guard broke; + quote tolerance + neg controls
    assert score("alias ll='ls -la'\n", "ll", "ls -la") == "1"  # was 0 under the guard
    assert score('alias ll="ls -la"\n', "ll", "ls -la") == "1"  # double-quote tolerance
    assert score("\n", "ll", "ls -la") == "0"  # absent -> no FP
    assert score("alias ll='ls -x'\n", "ll", "ls -la") == "0"  # wrong body -> no FP
    # ls-shortflags branch — flag-order tolerant, subset still fails
    assert score("alias ll='ls -al'\n", "ll", "ls -la") == "1"  # -al == -la (set)
    assert score("alias ll='ls -l'\n", "ll", "ls -la") == "0"  # missing flag -> no FP
