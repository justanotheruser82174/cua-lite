from __future__ import annotations

from types import SimpleNamespace

import pytest

from lite.core.tools import make_tool_call
from lite.core.tools.schemas import make_tool_schema
from lite.gym.sandbox.base import SandboxBaseEnv
from lite.gym.sandbox.types import SandboxTaskConfig
from lite.gym.utils.feedback.ingress import make_internal_terminate_action


class _FakeInterface:
    def __init__(self):
        self.calls = []

    async def get_screen_size(self):
        return {"width": 100, "height": 100}

    async def screenshot(self):
        return b"shot"

    async def left_click(self, x, y):
        self.calls.append(("left_click", x, y))

    async def type_text(self, text):
        self.calls.append(("type_text", text))


class _FailingClickInterface(_FakeInterface):
    async def left_click(self, x, y):
        self.calls.append(("left_click", x, y))
        raise RuntimeError("display pipe closed")


class _FakeAgentShell:
    def __init__(self):
        self.calls = []

    async def run(self, command):
        self.calls.append(command)
        return SimpleNamespace(output="shell output\n", returncode=0)


def _sandbox_env(interface: _FakeInterface | None = None) -> SandboxBaseEnv:
    env = SandboxBaseEnv.__new__(SandboxBaseEnv)
    env._computer = SimpleNamespace(interface=interface or _FakeInterface())
    env._display_resolution = (100, 100)
    env._post_action_delay = 0.0
    env._max_steps = 10
    env._step_count = 0
    env._debug = False
    env._evaluate_step_fn = None
    env._evaluate_final_fn = None
    env._task = SandboxTaskConfig(
        task_id="t",
        instruction="do it",
        computer={},
        extra_tool_schemas=[],
    )
    return env


def _sandbox_env_with_valid_actions(
    valid_actions: list[str] | None,
    interface: _FakeInterface | None = None,
) -> SandboxBaseEnv:
    env = _sandbox_env(interface)
    env._valid_actions = valid_actions
    return env


def _bash_schema() -> dict:
    return make_tool_schema(
        "bash",
        description="Run a bash command.",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments", "expected_error", "expect_current_image"),
    [
        ("response", {"text": "done"}, "response is not available in this task.", False),
        ("terminate", {"status": "success"}, "terminate is not available in this task.", False),
        ("open_app", {"app_name": "Settings"}, "unknown tool: open_app", False),
        ("goto", {"url": "https://example.com"}, "unknown tool: goto", False),
        ("bash", {"command": "pwd"}, "bash is not available in this task.", False),
        # R2(a): the only GUI action in this table, so the only row that earns
        # the current observation. The three above are text-surface tools --
        # R2(b) -- and must keep answering with the error alone.
        ("tap", {"coordinate": [500, 500]}, "unsupported action: tap", True),
    ],
)
async def test_inactive_or_unknown_standalone_tool_returns_d6_error_without_terminating(
    name, arguments, expected_error, expect_current_image
):
    env = SandboxBaseEnv.__new__(SandboxBaseEnv)
    env._computer = SimpleNamespace(interface=_FakeInterface())
    env._display_resolution = (100, 100)
    env._post_action_delay = 0.0
    env._max_steps = 10
    env._step_count = 0
    env._debug = False
    env._evaluate_step_fn = None
    env._evaluate_final_fn = None
    env._task = SandboxTaskConfig(
        task_id="t",
        instruction="do it",
        computer={},
        extra_tool_schemas=[],
    )

    r = await env.step([
        make_tool_call(name, arguments, call_id=f"call_{name}"),
    ])

    assert r.terminated is False
    assert r.results[0].tool_call_id == f"call_{name}"
    assert (r.results[0].images[-1] if r.results[0].images else None) == (
        b"shot" if expect_current_image else None
    )
    assert r.results[0].text is None
    assert r.results[0].error == expected_error
    assert r.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments", "schema"),
    [
        (
            "response",
            {"text": "done"},
            make_tool_schema(
                "response",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            ),
        ),
        (
            "terminate",
            {"status": "success"},
            make_tool_schema(
                "terminate",
                parameters={
                    "type": "object",
                    "properties": {"status": {"type": "string"}},
                    "required": ["status"],
                },
            ),
        ),
    ],
)
async def test_active_finish_tool_terminates_with_no_tool_result(
    name, arguments, schema
):
    env = SandboxBaseEnv.__new__(SandboxBaseEnv)
    env._computer = SimpleNamespace(interface=_FakeInterface())
    env._display_resolution = (100, 100)
    env._post_action_delay = 0.0
    env._max_steps = 10
    env._step_count = 0
    env._debug = False
    env._evaluate_step_fn = None
    env._evaluate_final_fn = None
    env._task = SandboxTaskConfig(
        task_id="t",
        instruction="do it",
        computer={},
        extra_tool_schemas=[schema],
    )

    r = await env.step([
        make_tool_call(name, arguments, call_id=f"call_{name}"),
    ])

    assert r.terminated is True
    assert r.truncated is False
    assert r.info["stop_reason"] == name
    # A terminal call gets NO tool result: it ended the episode, so there is no
    # next decision for an observation to inform, and
    # ``devs/migration/verify.py`` refuses a tool result for a terminal call.
    # What this test defends is unchanged: the ACTIVE canonical finish tool is
    # accepted (not gated away) and terminates with its own ``stop_reason``.
    assert r.results == []


@pytest.mark.asyncio
async def test_loop_detect_private_terminate_bypasses_inactive_extra_gate():
    env = SandboxBaseEnv.__new__(SandboxBaseEnv)
    env._computer = SimpleNamespace(interface=_FakeInterface())
    env._display_resolution = (100, 100)
    env._post_action_delay = 0.0
    env._max_steps = 10
    env._step_count = 0
    env._debug = False
    env._evaluate_step_fn = None
    env._evaluate_final_fn = None
    env._task = SandboxTaskConfig(
        task_id="t",
        instruction="do it",
        computer={},
        extra_tool_schemas=[],
    )

    r = await env.step([make_internal_terminate_action()])

    assert r.terminated is True
    assert r.truncated is False
    assert r.results == []


@pytest.mark.asyncio
async def test_batched_computer_unpacks_actions_but_returns_one_top_level_result():
    iface = _FakeInterface()
    env = SandboxBaseEnv.__new__(SandboxBaseEnv)
    env._computer = SimpleNamespace(interface=iface)
    env._display_resolution = (100, 100)
    env._post_action_delay = 0.0
    env._max_steps = 10
    env._step_count = 0
    env._debug = False
    env._evaluate_step_fn = None
    env._evaluate_final_fn = None
    env._task = SandboxTaskConfig(
        task_id="t",
        instruction="do it",
        computer={},
        extra_tool_schemas=[],
    )

    r = await env.step([
        make_tool_call(
            "computer",
            {
                "actions": [
                    {"action": "click", "coordinate": [1000, 1000]},
                    {"action": "type", "text": "ok"},
                ]
            },
            call_id="action_0",
        ),
    ])

    assert iface.calls == [
        ("left_click", 99, 99),
        ("type_text", "ok"),
    ]
    assert len(r.results) == 1
    assert r.results[0].tool_call_id == "action_0"
    assert r.results[0].images[-1] == b"shot"


@pytest.mark.asyncio
async def test_direct_sandbox_valid_actions_empty_rejects_action_without_dispatch():
    iface = _FakeInterface()
    env = _sandbox_env_with_valid_actions([], iface)

    r = await env.step([
        make_tool_call(
            "computer",
            {"actions": [{"action": "click", "coordinate": [1000, 1000]}]},
            call_id="action_0",
        ),
    ])

    assert iface.calls == []
    assert r.terminated is False
    assert len(r.results) == 1
    assert r.results[0].tool_call_id == "action_0"
    assert r.results[0].images[-1] == b"shot"
    assert r.results[0].error == (
        "invalid action: click; choose an available action for this task"
    )
    assert r.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_bad_key_returns_tool_error_result_without_crashing():
    iface = _FakeInterface()
    env = _sandbox_env(iface)

    r = await env.step([
        make_tool_call("key", {"keys": ["not_a_real_key"]}, call_id="bad_key"),
    ])

    assert r.terminated is False
    assert iface.calls == []
    assert len(r.results) == 1
    assert r.results[0].tool_call_id == "bad_key"
    assert r.results[0].images[-1] == b"shot"
    assert r.results[0].text is None
    assert "invalid arguments for key:" in r.results[0].error
    assert "unknown key token 'not_a_real_key'" in r.results[0].error
    assert r.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_scalar_key_returns_tool_error_result_without_dispatch():
    iface = _FakeInterface()
    env = _sandbox_env(iface)

    r = await env.step([
        make_tool_call("key", {"keys": "enter"}, call_id="bad_key"),
    ])

    assert r.terminated is False
    assert iface.calls == []
    assert len(r.results) == 1
    assert r.results[0].tool_call_id == "bad_key"
    assert r.results[0].images[-1] == b"shot"
    assert r.results[0].text is None
    assert r.results[0].error == (
        "invalid arguments for key: "
        "key.keys must be a list of strings, not a string"
    )
    assert r.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_bad_click_coordinate_returns_tool_error_without_dispatch():
    iface = _FakeInterface()
    env = _sandbox_env(iface)

    r = await env.step([
        make_tool_call(
            "computer",
            {"actions": [{"action": "click"}]},
            call_id="bad_click",
        ),
    ])

    assert r.terminated is False
    assert iface.calls == []
    assert len(r.results) == 1
    assert r.results[0].tool_call_id == "bad_click"
    assert r.results[0].images[-1] == b"shot"
    assert r.results[0].text is None
    assert r.results[0].error == (
        "invalid arguments for click: arguments could not be interpreted"
    )
    assert r.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_interface_failure_returns_observation_plus_sanitized_error(caplog):
    iface = _FailingClickInterface()
    env = _sandbox_env(iface)

    r = await env.step([
        make_tool_call(
            "computer",
            {"actions": [{"action": "click", "coordinate": [1000, 1000]}]},
            call_id="action_0",
        ),
    ])

    assert iface.calls == [("left_click", 99, 99)]
    assert r.terminated is False
    assert len(r.results) == 1
    assert r.results[0].tool_call_id == "action_0"
    assert r.results[0].images[-1] == b"shot"
    assert r.results[0].text is None
    assert r.results[0].error == "click failed: execution failed"
    assert "display pipe closed" not in r.results[0].error
    assert r.results[0].metadata == {"is_error": True}
    assert "Action click failed at the interface: display pipe closed" in caplog.text


@pytest.mark.asyncio
async def test_active_bash_uses_agent_shell_and_returns_text_only_result():
    iface = _FakeInterface()
    shell = _FakeAgentShell()
    env = SandboxBaseEnv.__new__(SandboxBaseEnv)
    env._computer = SimpleNamespace(interface=iface, agent_shell=shell)
    env._display_resolution = (100, 100)
    env._post_action_delay = 0.0
    env._max_steps = 10
    env._step_count = 0
    env._debug = False
    env._evaluate_step_fn = None
    env._evaluate_final_fn = None
    env._task = SandboxTaskConfig(
        task_id="t",
        instruction="do it",
        computer={},
        extra_tool_schemas=[_bash_schema()],
    )

    r = await env.step([
        make_tool_call("bash", {"command": "pwd"}, call_id="bash_0"),
    ])

    assert shell.calls == ["pwd"]
    assert len(r.results) == 1
    assert r.results[0].tool_call_id == "bash_0"
    assert r.results[0].images == []
    assert r.results[0].text == "shell output\n"
    assert r.results[0].metadata == {"returncode": 0}


@pytest.mark.asyncio
async def test_a_model_fault_costs_its_own_action_and_no_sibling():
    """R4: a rejected action no longer aborts the batch, and R3: it keeps its frame.

    The middle action names a key that does not exist -- a MODEL fault, caught
    before anything reached the screen. So the state the third action was chosen
    against is exactly the state still on screen, and there is nothing to protect
    it from: it runs. All three actions get a frame, the rejected one repeating
    the screen it did not change.

    This replaces a test that asserted the opposite (the tail must NOT run and
    the turn returns ONE image). That was the "a batch stops at the first
    rejected action" policy, superseded by R4 in /batch.md.
    """
    iface = _FakeInterface()
    env = _sandbox_env(iface)

    r = await env.step([
        make_tool_call(
            "computer",
            {
                "actions": [
                    {"action": "click", "coordinate": [1000, 1000]},
                    {"action": "key", "keys": ["not_a_real_key"]},
                    {"action": "type", "text": "runs anyway"},
                ]
            },
            call_id="action_0",
        ),
    ])

    # the sibling AFTER the fault still reached the backend
    assert iface.calls == [("left_click", 99, 99), ("type_text", "runs anyway")]
    assert len(r.results) == 1
    assert r.results[0].tool_call_id == "action_0"
    # R3: three actions in, three frames out
    assert len(r.results[0].images) == 3
    assert r.results[0].text is None
    assert "unknown key token 'not_a_real_key'" in r.results[0].error
    assert r.results[0].metadata == {"is_error": True}
