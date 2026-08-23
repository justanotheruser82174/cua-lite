"""Small live smoke tests for a built lite.osworld image.

These are intentionally narrower than oracle validation: they prove that the
private image boots GNOME, that the baked exec-stdio screenshot/input ops still
drive the real X session, and that the VS Code baseline suppresses first-run
Welcome UI. Override the image with ``LITE_OSWORLD_TEST_IMAGE``.

Every ``attach`` here takes the DEFAULT identity, and must keep taking it:
``attach``'s ``exec_user`` defaults to ``"user"`` and lite.osworld ships
``SandboxBaseEnv.EXEC_USER = "user"``, so the shipped session is a desktop-user
session (only cuaworld overrides it to ``root``). The default-root investigation
``5e29d622`` briefly pinned ``exec_user="root"`` at three sites here, which drove
screenshot / input / clipboard / VS Code through a ROOT session that still had
``HOME=/home/user`` — the state-poisoning shape ``f756eb9f`` reverted, and the
opposite of what a smoke test is for. It also made these tests assert something
untrue: ``run_command`` runs as the server's uid, so the ``id -u`` == 1000 probes
below (and ``_clean_user_cmd``'s ``test "$(id -u)" = 1000``) can only hold on the
user session. Do not reintroduce the argument; ``test_lite_osworld.py::
test_live_smoke_attaches_as_the_shipped_exec_user`` fails in the default suite
if you do.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import subprocess
import time
import uuid
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.live

_IMAGE = os.environ.get("LITE_OSWORLD_TEST_IMAGE", "cua-lite/lite.osworld:latest")
_DISPLAY = "DISPLAY=:1"
_CLEAN_USER_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_VSCODE_MAIN_UI_NEEDLES = ("Explorer", "Activity Bar", "Editor")
_VSCODE_WELCOME_NEEDLES = ("Welcome", "Get Started", "Make It Yours", "AI Agents")


def _image_present() -> bool:
    return shutil.which("docker") is not None and subprocess.run(
        ["docker", "image", "inspect", _IMAGE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def _exec(name: str, script: str, *, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "exec", name, "bash", "-lc", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _clean_user_cmd(script: str, **env: str) -> str:
    merged = {
        "HOME": "/home/user",
        "USER": "user",
        "LOGNAME": "user",
        "DISPLAY": ":1",
        "PATH": _CLEAN_USER_PATH,
        **env,
    }
    assigns = " ".join(f"{key}={shlex.quote(value)}" for key, value in merged.items())
    return f"env -i {assigns} /bin/bash -c {shlex.quote(script)}"


def _wait_for_desktop_ready(name: str) -> None:
    deadline = time.monotonic() + 180
    last = subprocess.CompletedProcess([], 1, "", "desktop probe did not run")
    while time.monotonic() < deadline:
        last = _exec(
            name,
            (
                "test -f /tmp/gnome-ready && "
                "pgrep -u user -f gnome-shell >/dev/null && "
                "ss -ltn '( sport = :5000 )' | tail -n +2 | grep -q ."
            ),
            timeout=15,
        )
        if last.returncode == 0:
            return
        time.sleep(2)

    logs = subprocess.run(
        ["docker", "logs", name],
        capture_output=True,
        text=True,
        timeout=30,
    )
    pytest.fail(
        "lite.osworld desktop did not become ready:\n"
        f"{last.stdout}\n{last.stderr}\n{logs.stdout}\n{logs.stderr}"
    )


def _start_osworld_container(prefix: str) -> str:
    name = f"{prefix}-{uuid.uuid4().hex[:12]}"
    started = subprocess.run(
        [
            "docker",
            "run",
            "--detach",
            "--name",
            name,
            # match the env's configs/default.yaml (8GB); a GEGL/office op
            # peaks past 4g and the container is SIGKILLed mid-test.
            "--memory",
            "8g",
            "--cpus",
            "2",
            "-e",
            "VNC_RESOLUTION=1920x1080",
            _IMAGE,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert started.returncode == 0, started.stderr
    try:
        _wait_for_desktop_ready(name)
    except Exception:
        subprocess.run(
            ["docker", "rm", "-f", "-v", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
        raise
    return name


@pytest.fixture(scope="module")
def booted_osworld_container() -> str:
    if not _image_present():
        pytest.skip(f"{_IMAGE} is not built")

    name = _start_osworld_container("lite-osworld-smoke")
    try:
        yield name
    finally:
        subprocess.run(
            ["docker", "rm", "-f", "-v", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )


@pytest.fixture
def clipboard_osworld_container() -> str:
    if not _image_present():
        pytest.skip(f"{_IMAGE} is not built")

    name = _start_osworld_container("lite-osworld-clipboard")
    try:
        yield name
    finally:
        subprocess.run(
            ["docker", "rm", "-f", "-v", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )


async def _wait_for_command(
    interface,
    command: str,
    *,
    timeout_s: float = 30.0,
):
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        last = await interface.run_command(command, timeout=10)
        if last.returncode == 0:
            return last
        await asyncio.sleep(1)
    stdout = "" if last is None else last.stdout
    stderr = "" if last is None else last.stderr
    pytest.fail(f"command did not pass before timeout:\n{command}\n{stdout}\n{stderr}")


async def _cleanup(interface, command: str) -> None:
    try:
        await interface.run_command(f"{command}; true", timeout=10)
    except Exception:
        pass


async def _wait_for_vscode_accessibility(interface, *, timeout_s: float = 30.0) -> str:
    deadline = time.monotonic() + timeout_s
    last_stdout = ""
    last_stderr = ""
    last_windows = ""
    while time.monotonic() < deadline:
        windows = await interface.run_command("wmctrl -l", timeout=10)
        last_windows = windows.stdout
        if "Welcome - Visual Studio Code" in windows.stdout:
            pytest.fail(f"VS Code Welcome window appeared:\n{windows.stdout}")

        at = await interface.run_command(
            "python3 - <<'PY'\n"
            "import json, urllib.request\n"
            "data = json.load(urllib.request.urlopen("
            "'http://localhost:5000/accessibility', timeout=20))\n"
            "print(data['AT'])\n"
            "PY",
            timeout=30,
        )
        last_stdout = at.stdout
        last_stderr = at.stderr
        if at.returncode == 0:
            matched_welcome = [
                needle for needle in _VSCODE_WELCOME_NEEDLES if needle in at.stdout
            ]
            if matched_welcome:
                pytest.fail(
                    "VS Code first-run UI appeared in accessibility tree: "
                    f"{matched_welcome}\n{at.stdout[:8000]}"
                )
            if "Visual Studio Code" in at.stdout and any(
                needle in at.stdout for needle in _VSCODE_MAIN_UI_NEEDLES
            ):
                return at.stdout
        await asyncio.sleep(1)

    log = await interface.run_command(
        "cat /tmp/vscode-smoke.log 2>/dev/null || true",
        timeout=10,
    )
    pytest.fail(
        "VS Code accessibility tree did not expose the main workbench before "
        f"timeout; stderr={last_stderr!r}\n"
        f"windows:\n{last_windows}\n"
        f"vscode log:\n{log.stdout}\n"
        f"accessibility:\n{last_stdout[:8000]}"
    )


@pytest.mark.asyncio
async def test_exec_stdio_screenshot_and_input_drive_x_session(
    booted_osworld_container: str,
) -> None:
    from lite.gym.sandbox.exec_stdio.client import attach

    handle = await attach(booted_osworld_container)
    try:
        screenshot = await handle.interface.screenshot()
        assert screenshot.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(screenshot) > 10_000

        marker = f"R4_SMOKE_{uuid.uuid4().hex[:8]}"
        marker_q = shlex.quote(marker)
        title = f"R4Smoke-{uuid.uuid4().hex[:8]}"
        title_q = shlex.quote(title)

        launch = (
            "rm -f /tmp/r4-input.txt /tmp/r4-terminal.log /tmp/r4-terminal-probe.txt; "
            f"{_DISPLAY} gnome-terminal --title {title_q} -- "
            "bash -lc 'cat > /tmp/r4-input.txt; "
            "{ id -u; id -un; stat -c \"%U:%G:%u:%g\" /tmp/r4-input.txt; "
            "printf \"PATH=%s\\n\" \"$PATH\"; "
            "for c in xdotool xclip xsel jq pdftk convert; do "
            "command -v \"$c\" >/dev/null 2>&1 && echo \"LEAK:$c\"; "
            "done; } > /tmp/r4-terminal-probe.txt' >/tmp/r4-terminal.log 2>&1 &"
        )
        launched = await handle.interface.run_command(launch, timeout=10)
        assert launched.returncode == 0, launched.stderr

        await _wait_for_command(
            handle.interface,
            f"{_DISPLAY} xdotool search --name {title_q} windowactivate --sync",
        )
        await handle.interface.type_text(marker)
        await handle.interface.press_key("Return")
        await handle.interface.hotkey("ctrl", "d")

        found = await _wait_for_command(
            handle.interface,
            f"grep -F {marker_q} /tmp/r4-input.txt",
            timeout_s=20,
        )
        assert marker in found.stdout

        probe = await _wait_for_command(
            handle.interface,
            "test -s /tmp/r4-terminal-probe.txt && cat /tmp/r4-terminal-probe.txt",
            timeout_s=20,
        )
        lines = probe.stdout.splitlines()
        assert lines[0] == "1000"
        assert lines[1] == "user"
        assert lines[2] == "user:user:1000:1000"
        path_line = next(line for line in lines if line.startswith("PATH="))
        assert "/opt/env" not in path_line
        assert not any(line.startswith("LEAK:") for line in lines)
    finally:
        await _cleanup(
            handle.interface,
            "pkill -u user -f 'gnome-terminal|/tmp/r4-input.txt' 2>/dev/null; "
            "rm -f /tmp/r4-input.txt /tmp/r4-terminal.log /tmp/r4-terminal-probe.txt",
        )
        await handle.stop()


@pytest.mark.asyncio
async def test_716a6079_clipboard_setup_installs_xsel_and_oracle_getter_roundtrips(
    clipboard_osworld_container: str,
    tmp_path,
) -> None:
    import lite.gym.envs.lite.osworld.main  # noqa: F401
    from lite.gym.envs.lite.osworld.src.eval.runner import _get_result
    from lite.gym.envs.lite.osworld.src.utils.dispatch import dispatch_action
    from lite.gym.sandbox import lookup_task
    from lite.gym.sandbox.exec_stdio.client import attach

    task = lookup_task("lite.osworld", "osworld_multi_apps_716a6079")
    assert task.setup_fn is not None
    assert any(
        step.get("parameters", {}).get("command")
        == "echo user | sudo -S apt install xsel && xsel -bc"
        for step in task.metadata["config"]
    )

    handle = await attach(clipboard_osworld_container)
    try:
        pre = await handle.interface.run_command(
            _clean_user_cmd(
                'test "$(id -u)" = 1000 && '
                'case "$PATH" in */opt/env*) exit 2;; esac; '
                "! command -v xsel >/dev/null 2>&1"
            ),
            timeout=20,
        )
        assert pre.returncode == 0, pre.stdout + pre.stderr

        await task.setup_fn(task, handle)

        doc = await handle.interface.run_command(
            "test -s /home/user/Data3/List3/secret.docx",
            timeout=20,
        )
        assert doc.returncode == 0, doc.stdout + doc.stderr

        marker = f"R1_CLIP_{uuid.uuid4().hex}"
        user_read = await handle.interface.run_command(
            _clean_user_cmd(
                "set -euo pipefail; "
                'test "$(id -u)" = 1000; '
                'case "$PATH" in */opt/env*) exit 2;; esac; '
                'test "$(command -v xsel)" = /usr/bin/xsel; '
                'printf "%s" "$MARKER" | xsel -bi; '
                "xsel -bo",
                MARKER=marker,
            ),
            timeout=20,
        )
        assert user_read.returncode == 0, user_read.stdout + user_read.stderr
        assert user_read.stdout == marker

        getter = task.metadata["evaluator"]["expected"]
        assert getter == {
            "type": "vm_command_line",
            "command": "xsel --clipboard --output",
            "shell": "true",
        }
        getter_calls = []

        class RecordingInterface:
            def __getattr__(self, name):
                return getattr(handle.interface, name)

            async def run_command(self, command, timeout=None):
                getter_calls.append(command)
                return await handle.interface.run_command(
                    command,
                    timeout=timeout,
                )

        got = await _get_result(
            SimpleNamespace(interface=RecordingInterface()),
            getter,
            str(tmp_path),
        )
        assert got == marker
        assert getter_calls == ["xsel --clipboard --output"]

        oracle_action = task.metadata["others"]["oracle_actions"][0]
        assert (
            oracle_action["parameters"]["command"]
            == "echo -n '/home/user/Data3/List3/secret.docx' | "
            "DISPLAY=:1 xclip -selection clipboard -i && sleep 1"
        )
        getter_calls.clear()
        oracle_resp = await dispatch_action(
            SimpleNamespace(interface=RecordingInterface()),
            oracle_action,
            cache_dir=str(tmp_path),
        )
        assert oracle_resp.get("returncode") == 0, oracle_resp
        assert getter_calls == [
            "echo -n '/home/user/Data3/List3/secret.docx' | "
            "DISPLAY=:1 xclip -selection clipboard -i && sleep 1"
        ]

        getter_calls.clear()
        expected_path = task.metadata["evaluator"]["result"]["rules"]["expected"]
        got_oracle = await _get_result(
            SimpleNamespace(interface=RecordingInterface()),
            getter,
            str(tmp_path),
        )
        assert got_oracle == expected_path
        assert getter_calls == ["xsel --clipboard --output"]
    finally:
        await _cleanup(
            handle.interface,
            f"{_DISPLAY} xsel -bc 2>/dev/null || true",
        )
        await handle.stop()


@pytest.mark.asyncio
async def test_vscode_baseline_starts_without_welcome_overlay(
    booted_osworld_container: str,
) -> None:
    from lite.gym.sandbox.exec_stdio.client import attach

    handle = await attach(booted_osworld_container)
    try:
        await _cleanup(
            handle.interface,
            "pkill -f '[c]ode' 2>/dev/null; rm -f /tmp/vscode-smoke.log",
        )

        launched = await handle.interface.run_command(
            f"{_DISPLAY} code /home/user >/tmp/vscode-smoke.log 2>&1 &",
            timeout=10,
        )
        assert launched.returncode == 0, launched.stderr

        await _wait_for_command(
            handle.interface,
            "wmctrl -l | grep -F 'Visual Studio Code'",
            timeout_s=60,
        )
        windows = await handle.interface.run_command("wmctrl -l", timeout=10)
        assert "Visual Studio Code" in windows.stdout
        assert "Welcome - Visual Studio Code" not in windows.stdout

        await _wait_for_vscode_accessibility(handle.interface)
    finally:
        await _cleanup(
            handle.interface,
            "pkill -f '[c]ode' 2>/dev/null; rm -f /tmp/vscode-smoke.log",
        )
        await handle.stop()
