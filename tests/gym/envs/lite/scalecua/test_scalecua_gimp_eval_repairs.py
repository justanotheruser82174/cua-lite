"""ScaleCUA GIMP evaluator repair tests."""

from __future__ import annotations

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


@pytest.mark.asyncio
async def test_scalecua_gimp_action_history_uses_pre_postconfig_window_fallback(
    monkeypatch,
):
    async def fake_get_result(eval_env, config, cache_dir, runtime_split):
        return ""

    async def fake_get_expected(eval_env, config, cache_dir, runtime_split):
        return {"include": ["filters-gaussian-blur"], "exclude": ["error"]}

    def metric(result, expected):
        assert "filters-gaussian-blur" in result
        return 1.0

    monkeypatch.setattr(scalecua_verify, "_get_result", fake_get_result)
    monkeypatch.setattr(scalecua_verify, "_get_expected", fake_get_expected)
    monkeypatch.setattr(judges, "resolve_metric", lambda name, runtime_split: metric)

    score = await scalecua_verify.evaluate_scalecua_task(
        _FakeComputer(),
        {
            "func": "check_include_exclude",
            "result": {
                "type": "vm_command_line",
                "command": "cat /home/user/.config/GIMP/2.10/action-history",
            },
            "expected": {"type": "rule"},
        },
        runtime_split="train",
        pre_postconfig_state="__ACTIVE__\nGaussian Blur - GIMP\n",
    )

    assert score == 1.0


@pytest.mark.asyncio
async def test_scalecua_gimp_pre_postconfig_state_runs_only_xdotool():
    computer = _FakeComputer(stdout=["window-state", "snapshot", "direct"])
    evaluator = {
        "func": "check_include_exclude",
        "result": {
            "type": "vm_command_line",
            "command": "cat /home/user/.config/GIMP/2.10/action-history",
        },
    }

    state = await scalecua_verify._capture_pre_postconfig_state(computer, evaluator)

    assert state == "window-state\nsnapshot\ndirect"
    assert len(computer.interface.command_calls) == 3
    assert "xdotool search" in computer.interface.command_calls[0]["command"]


@pytest.mark.asyncio
async def test_scalecua_gimp_action_history_accepts_gnu_window_marker(
    monkeypatch,
):
    async def fake_get_result(eval_env, config, cache_dir, runtime_split):
        return "# GIMP action-history\n# end of action-history"

    async def fake_get_expected(eval_env, config, cache_dir, runtime_split):
        return {"include": ["filters-lens-distortion"], "exclude": ["error"]}

    def metric(result, expected):
        assert "filters-lens-distortion" in result
        return 1.0

    monkeypatch.setattr(scalecua_verify, "_get_result", fake_get_result)
    monkeypatch.setattr(scalecua_verify, "_get_expected", fake_get_expected)
    monkeypatch.setattr(judges, "resolve_metric", lambda name, runtime_split: metric)

    score = await scalecua_verify.evaluate_scalecua_task(
        _FakeComputer(),
        {
            "func": "check_include_exclude",
            "result": {
                "type": "vm_command_line",
                "command": "cat /home/user/.config/GIMP/2.10/action-history",
            },
            "expected": {"type": "rule"},
        },
        runtime_split="train",
        pre_postconfig_state=(
            "__WMCTRL_L__\n"
            "0x02000003  0 machine Lens Distortion\n"
            "0x04000003  0 machine dog.png-1.0 (RGB color, 1 layer) - "
            "GNU Image Manipulation Program\n"
        ),
    )

    assert score == 1.0


def test_scalecua_gimp_window_match_uses_wmctrl_class_and_dialog_title():
    state = "__WMCTRL_LX__\n0x03a00007  0 machine gimp.Gimp-2.10 machine Lens Distortion\n"

    assert scalecua_verify._gimp_action_window_matches("filters-lens-distortion", state)
    assert not scalecua_verify._gimp_action_window_matches("filters-oilify", state)


def test_scalecua_gimp_window_match_accepts_child_dialog_titles():
    state = (
        "__XDO_TOOL_GIMP_WINDOWS__\n"
        "__WINDOW__123\n"
        "Sharpen\n"
        'WM_CLASS(STRING) = "gimp-2.10", "Gimp-2.10"\n'
    )

    assert scalecua_verify._gimp_action_window_matches("filters-unsharp-mask", state)


def test_scalecua_generic_blur_window_match_accepts_gaussian_blur_dialog():
    state = (
        "__ACCESSIBILITY_TREE_DIRECT__\n"
        "application: GNU Image Manipulation Program showing=True visible=True active=False\n"
        "  dialog: Gaussian Blur showing=True visible=True active=True\n"
    )

    assert scalecua_verify._gimp_action_window_matches("filters-blur", state)


def test_scalecua_gimp_window_state_command_checks_both_x_displays():
    command = scalecua_verify._build_gimp_window_state_command()

    assert "DISPLAY=:1" not in command
    assert " :1 :0" in command
    assert "wmctrl -lx" in command
    assert "timeout 1s wmctrl -lx" in command
    assert "xwininfo -root -tree" in command
    assert "_NET_CLIENT_LIST_STACKING" in command
    assert "xdotool search --class" in command
    assert "xdotool search --onlyvisible --name" in command
    assert ".*" in command
    assert "Gimp" in command


def test_scalecua_accessibility_tree_snapshot_command_is_localhost_only():
    command = scalecua_verify._build_accessibility_tree_snapshot_command()

    assert "http://localhost:5000/accessibility" in command
    assert "__ACCESSIBILITY_TREE__" in command


def test_scalecua_accessibility_tree_direct_command_uses_pyatspi():
    command = scalecua_verify._build_accessibility_tree_direct_command()

    assert "bash -lc" in command
    assert "XDG_RUNTIME_DIR" in command
    # `/run/user/1000`, not the locally-invented `/tmp/runtime-user`: the base image
    # creates the real freedesktop path, and upstream's own task JSONs hardcode
    # `/run/user/1000` (424 of them) while never mentioning XDG_RUNTIME_DIR at all —
    # so emitting it is both correct here and closer to upstream.
    assert "/run/user/1000" in command
    assert "/tmp/runtime-user" not in command
    assert "DBUS_SESSION_BUS_ADDRESS" in command
    assert "/tmp/dbus-session-bus-address" in command
    assert "pyatspi" in command
    assert "Registry.getDesktop(0)" in command
    assert "__ACCESSIBILITY_TREE_DIRECT__" in command


def test_scalecua_gimp_config_metric_fallback_handles_nested_and_quoted(tmp_path):
    config = tmp_path / "gimprc"
    config.write_text(
        "\n".join(
            [
                '(theme "Dark")',
                '(default-comment "Created with GIMP")',
                "(default-grid (spacing 20.000000 20.000000))",
                "(default-image (width 1920) (height 1080))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert judges._gimp_config_status_score(config, {"key": "theme", "value": '"Dark"'}) == 1.0
    assert (
        judges._gimp_config_status_score(
            config, {"key": "default-comment", "value": '"Created with GIMP"'}
        )
        == 1.0
    )
    assert (
        judges._gimp_config_status_score(
            config, {"key": ["default-grid", "(spacing"], "value": "20"}
        )
        == 1.0
    )
    assert (
        judges._gimp_config_status_score(
            config,
            {
                "checks": [
                    {"key": ["default-image", "width"], "value": "1920"},
                    {"key": ["default-image", "height"], "value": "1080"},
                ]
            },
        )
        == 1.0
    )


def test_scalecua_gimp_config_metric_wrapper_recovers_generated_helper_gap(tmp_path):
    config = tmp_path / "gimprc"
    config.write_text("(default-image (width 1920) (height 1080))\n", encoding="utf-8")

    def generated_metric(_result, _expected, **_options):
        raise NameError("_verify_single_check__7767eef2")

    metric = judges._wrap_gimp_config_metric(generated_metric)

    assert (
        metric(
            config,
            {
                "checks": [
                    {"key": ["default-image", "width"], "value": "1920"},
                    {"key": ["default-image", "height"], "value": "1080"},
                ]
            },
        )
        == 1.0
    )


@pytest.mark.asyncio
async def test_scalecua_gimp_action_history_fallback_requires_matching_window(
    monkeypatch,
):
    async def fake_get_result(eval_env, config, cache_dir, runtime_split):
        return ""

    async def fake_get_expected(eval_env, config, cache_dir, runtime_split):
        return {"include": ["filters-gaussian-blur"], "exclude": ["error"]}

    def metric(result, expected):
        assert "filters-gaussian-blur" not in result
        return 0.0

    monkeypatch.setattr(scalecua_verify, "_get_result", fake_get_result)
    monkeypatch.setattr(scalecua_verify, "_get_expected", fake_get_expected)
    monkeypatch.setattr(judges, "resolve_metric", lambda name, runtime_split: metric)

    score = await scalecua_verify.evaluate_scalecua_task(
        _FakeComputer(),
        {
            "func": "check_include_exclude",
            "result": {
                "type": "vm_command_line",
                "command": "cat /home/user/.config/GIMP/2.10/action-history",
            },
            "expected": {"type": "rule"},
        },
        runtime_split="train",
        pre_postconfig_state="__ACTIVE__\nLibreOffice Calc\n",
    )

    assert score == 0.0


@pytest.mark.asyncio
async def test_scalecua_gimp_config_getters_gracefully_flush_once(monkeypatch, tmp_path):
    computer = _FakeComputer()
    env = judges.make_eval_env(computer, str(tmp_path))

    async def base_get_result(_computer, config, _cache_dir):
        assert config == {"type": "gimp_config_file"}
        assert "xdotool search --class" in computer.interface.commands[0]
        assert "Gimp" in computer.interface.commands[0]
        assert "ctrl+q" in computer.interface.commands[0]
        assert "alt+d" in computer.interface.commands[0]
        return '(theme "Dark")\n'

    def generated_getter(_eval_env, config):
        assert config == {"type": "gimp_theme_setting__abc"}
        return {"config": '(theme "Dark")\n'}

    monkeypatch.setattr(scalecua_verify.base_runner, "_get_result", base_get_result)
    monkeypatch.setattr(
        scalecua_verify.judges,
        "resolve_getter",
        lambda result_type, runtime_split: generated_getter,
    )

    first = await scalecua_verify._get_result(
        env,
        {"type": "gimp_config_file"},
        str(tmp_path),
        "rl",
    )
    second = await scalecua_verify._get_result(
        env,
        {"type": "gimp_theme_setting__abc"},
        str(tmp_path),
        "rl",
    )

    assert first == '(theme "Dark")\n'
    assert second == {"config": '(theme "Dark")\n'}
    assert len(computer.interface.commands) == 1
    assert "pkill -TERM chrome" not in computer.interface.commands[0]


def test_scalecua_gimp_config_flush_detection_covers_generated_and_paths():
    assert scalecua_verify._needs_gimp_config_flush(
        "gimp_theme_setting__ff19a",
        {"type": "gimp_theme_setting__ff19a"},
    )
    assert scalecua_verify._needs_gimp_config_flush(
        "vm_command_line",
        {"command": "cat /home/user/.config/GIMP/2.10/gimprc"},
    )
    assert not scalecua_verify._needs_gimp_config_flush(
        "vm_command_line",
        {"command": "cat /home/user/chrome-data/Default/Preferences"},
    )
