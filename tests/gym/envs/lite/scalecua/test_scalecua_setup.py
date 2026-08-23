"""ScaleCUA setup and setup-error contract tests."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from lite.gym.envs.lite.scalecua import main as M
from lite.gym.envs.lite.scalecua.src.osworld import setup as scalecua_setup
from lite.gym.envs.lite.scalecua.src.utils import assets
from lite.gym.errors import ScaleCuaTaskError, is_retryable, lite_error_from_payload


async def test_setup_rejects_excluded_task_before_dispatch():
    task = SimpleNamespace(
        task_id="scalecua_excluded",
        metadata={"others": {"exclude_reason": "upstream_generated_eval_bug"}},
    )

    with pytest.raises(ScaleCuaTaskError, match="upstream_generated_eval_bug"):
        await scalecua_setup.setup_fn(task, computer=SimpleNamespace())


def test_scalecua_task_errors_are_nonretryable_and_round_trip():
    error = ScaleCuaTaskError(
        "bad setup",
        phase="config",
        kind="unsupported_action",
        returncode=7,
    )

    assert is_retryable(error) is False
    reconstructed = lite_error_from_payload(error.to_payload())
    assert isinstance(reconstructed, ScaleCuaTaskError)
    assert reconstructed.phase == "config"
    assert reconstructed.kind == "unsupported_action"
    assert reconstructed.returncode == 7


def test_scalecua_setup_repairs_generated_pyautogui_list_hotkeys():
    action = {
        "type": "execute",
        "parameters": {
            "command": (
                "python3 -c \"import pyautogui; "
                "pyautogui.hotkey(['shift', 'ctrl', 'e']); "
                "pyautogui.hotkey([\\\"ctrl\\\", \\\"q\\\"])\""
            )
        },
    }

    repaired = scalecua_setup._repair_generated_pyautogui_hotkeys(action)

    assert repaired is not action
    command = repaired["parameters"]["command"]
    assert "pyautogui.hotkey(*['shift', 'ctrl', 'e'])" in command
    assert 'pyautogui.hotkey(*[\\"ctrl\\", \\"q\\"])' in command


def test_scalecua_setup_raises_on_failed_execute_result():
    with pytest.raises(ScaleCuaTaskError, match="postconfig\\[2\\].*failed"):
        scalecua_setup._raise_on_dispatch_failure(
            {"status": "ok", "returncode": 1, "stderr": "boom"},
            phase="postconfig",
            index=2,
            action_type="execute",
        )


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


def test_scalecua_rejects_non_default_image_parameter(monkeypatch):
    captured: dict[str, object] = {}
    checked: list[str] = []

    def fake_parent_init(self, *args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)

    monkeypatch.setattr(M, "_check_base_image", lambda: checked.append(M._IMAGE))
    monkeypatch.setattr(M.LiteOsworldEnv, "__init__", fake_parent_init)

    with pytest.raises(ValueError, match="always uses the configured lite.osworld image"):
        M.LiteScaleCuaEnv(
            image="cua-lite/lite.osworld:ignored",
            display_resolution=(1920, 1080),
        )

    assert checked == []
    assert captured == {}


def test_scalecua_accepts_default_latest_image(monkeypatch):
    captured: dict[str, object] = {}
    checked: list[str] = []

    def fake_parent_init(self, *args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)

    monkeypatch.setattr(M, "_check_base_image", lambda: checked.append(M._IMAGE))
    monkeypatch.setattr(M.LiteOsworldEnv, "__init__", fake_parent_init)

    env = M.LiteScaleCuaEnv(image=M._IMAGE, display_resolution=(1920, 1080))

    assert checked == [M._IMAGE]
    assert captured["image"] is None
    assert captured["computer_config"]["image"] == M._IMAGE
    assert captured["env_id"] == "lite.scalecua"
    assert env._max_resets_per_container == M._MAX_RESETS_PER_CONTAINER


def test_scalecua_rejects_non_osworld_resolution():
    with pytest.raises(ValueError, match="1920x1080"):
        M.LiteScaleCuaEnv(
            image="cua-lite/lite.osworld:latest",
            display_resolution=(1280, 720),
        )


def test_scalecua_ensure_services_does_not_gate_on_default_image(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(M, "_check_desktop_env", lambda: calls.append("desktop"))
    monkeypatch.setattr(M, "_check_base_image", lambda: calls.append("image"))

    M.ensure_services("lite.scalecua")

    assert calls == ["desktop"]


def test_scalecua_install_script_uses_latest_osworld_image():
    repo = Path(__file__).resolve().parents[5]
    script = repo / "lite/gym/envs/lite/scalecua/scripts/install.sh"

    text = script.read_text()

    assert 'OSWORLD_IMAGE="cua-lite/lite.osworld:latest"' in text
    assert "require_explicit_non_latest_images" not in text
    assert "non-latest" not in text
    assert "LITE_SCALECUA_OSWORLD_IMAGE" not in text
    assert "LITE_OSWORLD_IMAGE" not in text
    assert "LITE_OSWORLD_BASE_IMAGE" not in text
    assert "LITE_SANDBOX_IMAGE" not in text
    assert 'bash "$OSWORLD_INSTALL" pull' in text


def test_install_script_contract_text():
    text = (assets.ENV_DIR / "scripts" / "install.sh").read_text()
    assert "cua-lite/lite.scalecua" not in text
    assert "fixedbase" not in text
    assert "cua-lite/lite.osworld:latest" in text
    assert "exit 2" in text
    assert '"$REPO_ROOT[gym]"' in text
    build_body = re.search(r"build\(\) \{\n(?P<body>.*?)\n\}", text, re.S).group("body")
    assert build_body.index("ensure_base_image") < build_body.index("provision")
    rebuild_body = re.search(r"rebuild\(\) \{\n(?P<body>.*?)\n\}", text, re.S).group("body")
    assert rebuild_body.index("ensure_base_image") < rebuild_body.index("provision")
    import_body = re.search(r"import_tasks\(\) \{\n(?P<body>.*?)\n\}", text, re.S).group("body")
    assert 'bash "$TASKS_HELPER" generate "$@"' in import_body
    assert "assets.sh" not in text
    assert "assets)" not in text
    status_body = re.search(r"status\(\) \{\n(?P<body>.*?)\n\}", text, re.S).group("body")
    pull_body = re.search(r"pull\(\) \{\n(?P<body>.*?)\n\}", text, re.S).group("body")
    assert "ensure_base_image" not in status_body
    assert "ensure_base_image" not in pull_body


@pytest.mark.asyncio
async def test_scalecua_setup_file_output_verification_allows_empty_files():
    computer = _FakeComputer(stdout="OK\n")

    await scalecua_setup._verify_file_outputs(
        computer,
        {
            "type": "download",
            "parameters": {"files": [{"path": "/home/user/Desktop/empty.txt"}]},
        },
        phase="config",
        index=0,
    )

    assert "test -e '/home/user/Desktop/empty.txt'" in computer.interface.commands[-1]
