"""SandboxBaseEnv.step ordering for mixed shell and GUI tool calls."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from lite.core.tools import make_tool_call
from lite.gym.sandbox.base import SandboxBaseEnv
from lite.gym.sandbox.types import SandboxTaskConfig


def _task() -> SandboxTaskConfig:
    return SandboxTaskConfig(
        task_id="t",
        instruction="use bash before reading the desktop",
        computer={"image": "x"},
    )


def _unbooted_env() -> SandboxBaseEnv:
    env = SandboxBaseEnv.__new__(SandboxBaseEnv)
    env._display_resolution = (1000, 1000)
    env._computer_config = None
    env._env_id = None
    env.bind(_task(), extra_tools=["bash"])
    env._post_action_delay = 0.0
    env._max_steps = 10
    env._step_count = 0
    env._debug = False
    env._evaluate_step_fn = None
    env._evaluate_final_fn = None
    return env


class _FakeShell:
    def __init__(self, fs: dict[str, str], order: list[tuple[str, str]]) -> None:
        self._fs = fs
        self._order = order

    async def run(self, command: str) -> SimpleNamespace:
        self._order.append(("bash", command))
        self._fs["/tmp/x"] = command
        return SimpleNamespace(output="written", returncode=0)


class _FakeInterface:
    def __init__(self, fs: dict[str, str], order: list[tuple[str, str]]) -> None:
        self._fs = fs
        self._order = order

    async def get_screen_size(self) -> dict[str, int]:
        return {"width": 1000, "height": 1000}

    async def get_cursor_position(self) -> dict[str, int]:
        self._order.append(("cursor_position", self._fs.get("/tmp/x", "<empty>")))
        return {"x": 0, "y": 0}

    async def screenshot(self) -> bytes:
        seen = self._fs.get("/tmp/x", "<empty>")
        self._order.append(("screenshot", seen))
        return seen.encode()

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - unexpected call
        raise AssertionError(f"unexpected interface call: {name}")


@pytest.mark.asyncio
async def test_computer_bash_execute_in_tool_call_order() -> None:
    fs: dict[str, str] = {}
    order: list[tuple[str, str]] = []
    env = _unbooted_env()
    env._computer = SimpleNamespace(
        interface=_FakeInterface(fs, order),
        agent_shell=_FakeShell(fs, order),
    )

    result = await env.step(
        [
            make_tool_call("bash", {"command": "HELLO_FROM_BASH"}, call_id="b1"),
            make_tool_call(
                "computer",
                {"actions": [{"action": "cursor_position"}]},
                call_id="c1",
            ),
        ]
    )

    assert order == [
        ("bash", "HELLO_FROM_BASH"),
        ("cursor_position", "HELLO_FROM_BASH"),
        ("screenshot", "HELLO_FROM_BASH"),
    ]
    assert [item.tool_call_id for item in result.results] == ["b1", "c1"]
    assert result.results[0].text == "written"
    assert result.results[0].images == []
    assert result.results[1].images == [b"HELLO_FROM_BASH"]
