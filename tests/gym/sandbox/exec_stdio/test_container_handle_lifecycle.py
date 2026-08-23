"""Container-handle ownership of the agent-facing bash shell."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import uuid

import pytest

_OSWORLD_IMAGE = os.environ.get("LITE_OSWORLD_TEST_IMAGE", "cua-lite/lite.osworld:latest")


def _docker_image_present(image: str) -> bool:
    return (
        shutil.which("docker") is not None
        and subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


@pytest.mark.asyncio
async def test_container_ownership_closes_agent_shell_on_teardown() -> None:
    from lite.gym.sandbox.exec_stdio.client import _ContainerHandle

    closed: list[str] = []

    class _FakeSession:
        async def close(self) -> None:
            closed.append("session")

    handle = _ContainerHandle("cont", _FakeSession(), agent_user="user")  # type: ignore[arg-type]
    shell = handle.agent_shell
    assert handle.agent_shell is shell

    async def _close() -> None:
        closed.append("agent_shell")

    shell.close = _close  # type: ignore[method-assign]

    await handle.stop()

    assert closed == ["agent_shell", "session"]
    assert handle._agent_shell is None


@pytest.mark.asyncio
@pytest.mark.live
async def test_fresh_episode_gets_clean_bash_shell() -> None:
    if importlib.util.find_spec("lite.gym.sandbox.exec_stdio.agent_shell") is None:
        pytest.skip("agent_shell channel not importable")
    if not _docker_image_present(_OSWORLD_IMAGE):
        pytest.skip(f"{_OSWORLD_IMAGE} not built")

    from lite.gym.sandbox.exec_stdio.client import attach

    name = f"lite-clean-shell-{uuid.uuid4().hex[:12]}"
    started = subprocess.run(
        [
            "docker",
            "run",
            "--detach",
            "--name",
            name,
            "--memory",
            "8g",
            "--cpus",
            "2",
            "-e",
            "VNC_RESOLUTION=1920x1080",
            _OSWORLD_IMAGE,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert started.returncode == 0, started.stderr
    try:
        handle = await attach(name)
        try:
            await handle.agent_shell.run("export EP1_MARKER=poison")
            first = await handle.agent_shell.run('printf %s "$EP1_MARKER"')
            assert "poison" in getattr(first, "output", str(first))
            assert "/opt/env" not in handle.agent_shell.login_path
        finally:
            await handle.stop()

        handle = await attach(name)
        try:
            second = await handle.agent_shell.run('printf %s "$EP1_MARKER"')
            assert getattr(second, "output", str(second)).strip() == ""
        finally:
            await handle.stop()
    finally:
        subprocess.run(
            ["docker", "rm", "-f", "-v", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
