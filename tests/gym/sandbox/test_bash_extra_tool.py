"""``env_kwargs.extra_tools: ["bash"]`` — resolution, refusal, and survival.

Covers the four things that had to be true for a bash rollout to work end to end:
its canonical schema resolves for the whole ``SandboxBaseEnv`` family, envs that
cannot honor it refuse LOUDLY, a ``bash(command=...)`` call comes back as a
TEXT-only result paired to its ``call_id``, and that result survives
``lite.osworld``'s step post-processing (which re-pairs results and used to drop
every image-less one).

Run:
    uv run pytest tests/gym/sandbox/test_bash_extra_tool.py -q
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lite.core.tools import make_tool_call
from lite.core.tools.extra_tools import BASH_TOOL_NAME, LiteShellToolSet
from lite.core.tools.schemas import (
    BaseTools,
    make_tool_schema,
    tool_schema_name,
    tool_schema_parameters,
)
from lite.gym.envs.lite.cuagym.main import _CuaGymEnv
from lite.gym.envs.lite.cuaworld.src.software import _make_env_class
from lite.gym.envs.lite.osworld.main import LiteOsworldEnv
from lite.gym.sandbox.base import SandboxBaseEnv
from lite.gym.sandbox.types import SandboxTaskConfig
from lite.gym.utils.feedback.surface import (
    merge_extra_tool_schemas,
    resolve_extra_tools,
)


class _NoTools(BaseTools):
    """An env that declares no extras of its own."""


class _BrowserTools(BaseTools):
    """A browser env's declared set — ``goto`` only, and never ``bash``."""

    _SCHEMAS = {
        "goto": make_tool_schema("goto"),
    }


def _task(**kwargs) -> SandboxTaskConfig:
    return SandboxTaskConfig(
        task_id="t", instruction="do it", computer={"image": "x"}, **kwargs
    )


def _unbooted(cls) -> SandboxBaseEnv:
    """A bindable hand-built instance without constructor probes."""
    env = cls.__new__(cls)
    env._display_resolution = (1920, 1080)
    env._computer_config = None
    env._env_id = None
    return env


class _FakeInterface:
    def __init__(self):
        self.calls: list[tuple] = []

    async def get_screen_size(self):
        return {"width": 100, "height": 100}

    async def screenshot(self):
        return b"shot"

    async def move_cursor(self, x, y):
        self.calls.append(("move_cursor", x, y))


class _FakeAgentShell:
    def __init__(self, output: str = "/home/user\n", returncode: int = 0):
        self.commands: list[str] = []
        self._output = output
        self._returncode = returncode

    async def run(self, command):
        self.commands.append(command)
        return SimpleNamespace(output=self._output, returncode=self._returncode)


class _FailingAgentShell(_FakeAgentShell):
    async def run(self, command):
        self.commands.append(command)
        raise RuntimeError("docker exec failed: container is not running")


def _runnable(env: SandboxBaseEnv, shell: _FakeAgentShell) -> SandboxBaseEnv:
    env._computer = SimpleNamespace(interface=_FakeInterface(), agent_shell=shell)
    env._post_action_delay = 0.0
    env._max_steps = 10
    env._step_count = 0
    env._debug = False
    env._evaluate_step_fn = None
    env._evaluate_final_fn = None
    return env


# ---------------------------------------------------------------------------
# 1. resolves for every SandboxBaseEnv-derived env
# ---------------------------------------------------------------------------
_CUAWORLD_ENV_CLS = _make_env_class("lite.cuaworld.vlc", {"image": "x"})


@pytest.mark.parametrize(
    "cls", [SandboxBaseEnv, _CuaGymEnv, _CUAWORLD_ENV_CLS, LiteOsworldEnv]
)
def test_bash_opt_in_resolves_into_metadata_for_the_sandbox_family(cls):
    env = _unbooted(cls)
    env.bind(_task(), extra_tools=["bash"])
    assert [tool_schema_name(s) for s in env.metadata.extra_tool_schemas] == ["bash"]

    bash = env.metadata.extra_tool_schemas[0]
    assert bash["type"] == "function"
    parameters = tool_schema_parameters(bash)
    assert parameters["required"] == ["command"]
    assert set(parameters["properties"]) == {"command"}
    assert bash["function"]["description"]


@pytest.mark.parametrize(
    "cls", [SandboxBaseEnv, _CuaGymEnv, _CUAWORLD_ENV_CLS, LiteOsworldEnv]
)
def test_bash_stays_off_without_the_opt_in(cls):
    env = _unbooted(cls)
    env.bind(_task())
    assert env.metadata.extra_tool_schemas == []


def test_cuaworld_agent_shell_drops_to_the_unprivileged_desktop_user():
    """cuaworld's exec session is root (upstream ``su - ga`` hooks); the AGENT
    shell must not inherit that identity."""
    assert _CUAWORLD_ENV_CLS.EXEC_USER == "root"
    assert _CUAWORLD_ENV_CLS.AGENT_USER == "ga"
    assert SandboxBaseEnv.AGENT_USER == "user"


def test_bash_is_bind_owned_and_never_constructor_state():
    """Soft env_kwarg: ``extra_tools`` is resolved by ``bind``, never by the
    constructor-state helper — so it can never change the booted container's
    shape."""
    import inspect
    constructor_state_params = set(
        inspect.signature(SandboxBaseEnv._set_constructor_state).parameters
    )
    assert "extra_tools" not in constructor_state_params
    for cls in (LiteOsworldEnv, _CuaGymEnv, _CUAWORLD_ENV_CLS):
        assert "extra_tools" in inspect.signature(cls.bind).parameters


# ---------------------------------------------------------------------------
# 2. envs that cannot honor bash refuse LOUDLY
# ---------------------------------------------------------------------------
def test_envs_without_an_agent_shell_reject_bash_with_a_reason():
    """osworld / osworld_2 are QEMU-VM-in-Docker: ``docker exec`` lands on the VM
    wrapper, not the guest the agent sees. Every non-sandbox env routes its
    ``extra_tools`` through ``resolve_extra_tools``, so one gate covers them all."""
    with pytest.raises(ValueError) as exc:
        resolve_extra_tools(["bash"], tools=_NoTools)
    message = str(exc.value)
    assert "cannot honor extra_tools ['bash']" in message
    assert "osworld_2" in message and "QEMU" in message
    assert "unknown extra_tools" not in message

    # ...and an env with a wired ``executable`` set that omits bash (browsergym /
    # webgym / webharbor) hits the same gate, not the generic unknown message.
    with pytest.raises(ValueError, match=r"cannot honor extra_tools \['bash'\]"):
        resolve_extra_tools(
            ["bash"],
            tools=_BrowserTools,
            env_name="webgym",
            executable=frozenset({"goto"}),
        )


def test_non_bash_unknown_names_keep_the_generic_message():
    with pytest.raises(ValueError, match="unknown extra_tools"):
        resolve_extra_tools(["nope"], tools=_NoTools)


# ---------------------------------------------------------------------------
# 3. a bash call is a TEXT-only result paired to its call_id
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_inactive_bash_returns_error_only_feedback():
    env = _unbooted(SandboxBaseEnv)
    env.bind(_task())
    shell = _FakeAgentShell()
    _runnable(env, shell)

    r = await env.step([
        make_tool_call("bash", {"command": "pwd"}, call_id="bash_0"),
    ])

    assert shell.commands == []
    assert [res.tool_call_id for res in r.results] == ["bash_0"]
    assert r.results[0].images == []
    assert r.results[0].text is None
    assert r.results[0].metadata == {"is_error": True}
    assert r.results[0].error == "bash is not available in this task."


@pytest.mark.asyncio
async def test_inactive_finish_tool_returns_error_only_feedback():
    env = _unbooted(SandboxBaseEnv)
    env.bind(_task())
    shell = _FakeAgentShell()
    _runnable(env, shell)

    r = await env.step([
        make_tool_call("terminate", {"status": "success"}, call_id="term_0"),
    ])

    assert r.terminated is False
    assert [res.tool_call_id for res in r.results] == ["term_0"]
    assert r.results[0].images == []
    assert r.results[0].metadata == {"is_error": True}
    assert r.results[0].error == "terminate is not available in this task."
    assert r.info["executed_actions"][0] == {
        "call": "noop",
        "args": {"name": "terminate", "reason": "inactive extra tool"},
    }


@pytest.mark.asyncio
async def test_unknown_non_gui_tool_is_error_only():
    env = _unbooted(SandboxBaseEnv)
    env.bind(_task())
    shell = _FakeAgentShell()
    _runnable(env, shell)

    r = await env.step([
        make_tool_call("foo", call_id="foo_0"),
    ])

    assert [res.tool_call_id for res in r.results] == ["foo_0"]
    assert r.results[0].images == []
    assert r.results[0].text is None
    assert r.results[0].metadata == {"is_error": True}
    assert r.results[0].error == "unknown tool: foo"
    assert r.info["executed_actions"][0] == {
        "call": "noop",
        "args": {"name": "foo", "reason": "unknown tool"},
    }


@pytest.mark.asyncio
async def test_bash_call_returns_a_text_only_result_paired_to_its_call_id():
    env = _unbooted(SandboxBaseEnv)
    env.bind(_task(), extra_tools=["bash"])
    shell = _FakeAgentShell(output="/home/user\n")
    _runnable(env, shell)

    r = await env.step([
        make_tool_call("bash", {"command": "pwd"}, call_id="call_bash"),
    ])

    assert shell.commands == ["pwd"]
    assert [res.tool_call_id for res in r.results] == ["call_bash"]
    assert r.results[0].images == []
    assert r.results[0].text == "/home/user\n"
    assert r.results[0].metadata == {"returncode": 0}


@pytest.mark.asyncio
async def test_bash_result_survives_truncated_sandbox_step():
    env = _unbooted(SandboxBaseEnv)
    env.bind(_task(), extra_tools=["bash"])
    shell = _FakeAgentShell(output="")
    _runnable(env, shell)
    env._max_steps = 1

    r = await env.step([
        make_tool_call("bash", {"command": "true"}, call_id="bash_0"),
    ])

    assert r.truncated is True
    assert r.info["stop_reason"] == "max_steps"
    assert [res.tool_call_id for res in r.results] == ["bash_0"]
    assert r.results[0].images == []
    assert r.results[0].text == ""
    assert r.results[0].metadata == {"returncode": 0}


@pytest.mark.asyncio
async def test_bash_result_survives_lite_osworld_step_delegation():
    """lite.osworld delegates normal actions to SandboxBaseEnv without rebuilding results."""
    env = _unbooted(LiteOsworldEnv)
    env.bind(_task(), extra_tools=["bash"])
    shell = _FakeAgentShell(output="hello\n")
    _runnable(env, shell)

    r = await env.step([
        make_tool_call(
            "computer",
            {"actions": [{"action": "mouse_move", "coordinate": [100, 100]}]},
            call_id="gui_0",
        ),
        make_tool_call("bash", {"command": "echo hello"}, call_id="bash_0"),
    ])

    assert shell.commands == ["echo hello"]
    by_id = {res.tool_call_id: res for res in r.results}
    assert set(by_id) == {"gui_0", "bash_0"}
    # GUI call keeps the post-step screenshot; bash stays text-only.
    assert ("move_cursor", 10, 10) in env._computer.interface.calls
    assert by_id["gui_0"].images[-1] == b"shot"
    assert by_id["bash_0"].images == []
    assert by_id["bash_0"].text == "hello\n"
    assert by_id["bash_0"].metadata == {"returncode": 0}


@pytest.mark.asyncio
async def test_invalid_bash_arguments_do_not_inherit_gui_screenshot_in_lite_osworld():
    env = _unbooted(LiteOsworldEnv)
    env.bind(_task(), extra_tools=["bash"])
    shell = _FakeAgentShell()
    _runnable(env, shell)

    r = await env.step([
        make_tool_call(
            "computer",
            {"actions": [{"action": "mouse_move", "coordinate": [100, 100]}]},
            call_id="gui_0",
        ),
        make_tool_call("bash", {"command": ["pwd"]}, call_id="bash_0"),
    ])

    by_id = {res.tool_call_id: res for res in r.results}
    assert set(by_id) == {"gui_0", "bash_0"}
    assert ("move_cursor", 10, 10) in env._computer.interface.calls
    assert by_id["gui_0"].images[-1] == b"shot"
    assert by_id["bash_0"].images == []
    assert by_id["bash_0"].text is None
    assert by_id["bash_0"].metadata == {"is_error": True}
    assert by_id["bash_0"].error == (
        "invalid arguments for bash: bash.arguments.command must be a string"
    )


@pytest.mark.asyncio
async def test_bash_rejects_schema_extra_arguments_without_running():
    env = _unbooted(LiteOsworldEnv)
    env.bind(_task(), extra_tools=["bash"])
    shell = _FakeAgentShell()
    _runnable(env, shell)

    r = await env.step([
        make_tool_call(
            "bash",
            {"command": "echo should-not-run", "restart": False},
            call_id="bash_0",
        ),
    ])

    assert shell.commands == []
    assert [res.tool_call_id for res in r.results] == ["bash_0"]
    assert r.results[0].images == []
    assert r.results[0].text is None
    assert r.results[0].metadata == {"is_error": True}
    assert r.results[0].error == (
        "invalid arguments for bash: "
        "bash.arguments got unexpected keys: ['restart']"
    )


@pytest.mark.asyncio
async def test_bash_result_survives_lite_osworld_truncated_bash_only_step():
    env = _unbooted(LiteOsworldEnv)
    env.bind(_task(), extra_tools=["bash"])
    shell = _FakeAgentShell(output="done\n")
    _runnable(env, shell)
    env._max_steps = 1

    r = await env.step([
        make_tool_call("bash", {"command": "echo done"}, call_id="bash_0"),
    ])

    assert r.truncated is True
    assert [res.tool_call_id for res in r.results] == ["bash_0"]
    assert r.results[0].images == []
    assert r.results[0].text == "done\n"
    assert r.results[0].metadata == {"returncode": 0}


@pytest.mark.asyncio
async def test_failed_bash_stays_a_paired_error_not_a_hard_raise():
    env = _unbooted(LiteOsworldEnv)
    env.bind(_task(), extra_tools=["bash"])
    shell = _FakeAgentShell(output="nope\n", returncode=127)
    _runnable(env, shell)

    r = await env.step([
        make_tool_call("bash", {"command": "nosuchcmd"}, call_id="bash_0"),
    ])

    assert r.terminated is False
    assert [res.tool_call_id for res in r.results] == ["bash_0"]
    assert "[exit 127]" in r.results[0].text


@pytest.mark.asyncio
async def test_lite_osworld_t2_release_rows_text_cli_failure():
    env = _unbooted(LiteOsworldEnv)
    env.bind(_task(), extra_tools=["bash", "response", "terminate"])
    shell = _FakeAgentShell(output="command not found\n", returncode=127)
    _runnable(env, shell)

    r = await env.step([
        make_tool_call(
            "bash",
            {"command": "definitely-not-a-command"},
            call_id="text_cli_failure",
        ),
    ])

    assert shell.commands == ["definitely-not-a-command"]
    assert r.terminated is False
    assert r.truncated is False
    assert [res.tool_call_id for res in r.results] == ["text_cli_failure"]
    assert r.results[0].images == []
    assert r.results[0].text == "command not found\n\n[exit 127]"
    assert r.results[0].error is None
    assert r.results[0].metadata == {"returncode": 127}


@pytest.mark.asyncio
async def test_invalid_bash_arguments_stay_text_only_without_screenshot():
    env = _unbooted(LiteOsworldEnv)
    env.bind(_task(), extra_tools=["bash"])
    shell = _FakeAgentShell()
    _runnable(env, shell)

    r = await env.step([
        make_tool_call("bash", {"command": ["pwd"]}, call_id="bash_0"),
    ])

    assert shell.commands == []
    assert [res.tool_call_id for res in r.results] == ["bash_0"]
    assert r.results[0].images == []
    assert r.results[0].text is None
    assert r.results[0].metadata == {"is_error": True}
    assert r.results[0].error == (
        "invalid arguments for bash: bash.arguments.command must be a string"
    )


@pytest.mark.asyncio
async def test_bash_execution_exception_is_sanitized_and_text_only():
    env = _unbooted(LiteOsworldEnv)
    env.bind(_task(), extra_tools=["bash"])
    shell = _FailingAgentShell()
    _runnable(env, shell)

    r = await env.step([
        make_tool_call("bash", {"command": "pwd"}, call_id="bash_0"),
    ])

    assert shell.commands == ["pwd"]
    assert [res.tool_call_id for res in r.results] == ["bash_0"]
    assert r.results[0].images == []
    assert r.results[0].text is None
    assert r.results[0].metadata == {"is_error": True}
    assert r.results[0].error == "bash failed: execution failed"
    assert "docker" not in r.results[0].error
    assert "container" not in r.results[0].error


@pytest.mark.asyncio
async def test_sandbox_rejects_duplicate_call_ids_before_side_effects():
    env = _unbooted(SandboxBaseEnv)
    env.bind(_task(), extra_tools=["bash"])
    shell = _FakeAgentShell()
    _runnable(env, shell)

    with pytest.raises(ValueError, match="duplicate tool call id: dup"):
        await env.step([
            make_tool_call("bash", {"command": "echo side-effect"}, call_id="dup"),
            make_tool_call("wait", call_id="dup"),
        ])

    assert shell.commands == []
    assert env._step_count == 0


@pytest.mark.asyncio
async def test_lite_osworld_rejects_duplicate_local_and_delegated_ids_before_step():
    env = _unbooted(LiteOsworldEnv)
    env.bind(_task(), extra_tools=[])
    shell = _FakeAgentShell()
    _runnable(env, shell)

    with pytest.raises(ValueError, match="duplicate tool call id: dup"):
        await env.step([
            make_tool_call("report_infeasible", {"reason": "missing"}, call_id="dup"),
            make_tool_call(
                "computer",
                {"actions": [{"action": "screenshot"}]},
                call_id="dup",
            ),
        ])

    assert shell.commands == []
    assert env._step_count == 0


# ---------------------------------------------------------------------------
# 4. the concatenate fix
# ---------------------------------------------------------------------------
def test_env_owned_extras_keep_the_base_bash_slice():
    env = _unbooted(LiteOsworldEnv)
    env.bind(_task(), extra_tools=["report_infeasible", "bash", "terminate"])
    names = [tool_schema_name(s) for s in env.metadata.extra_tool_schemas]
    assert names == ["report_infeasible", "bash", "terminate"]


def test_task_declared_extras_are_concatenated_not_replaced():
    """A wholesale ``dataclasses.replace`` at either builder drops a slice."""
    declared = make_tool_schema(
        "task_local",
        parameters={"type": "object", "properties": {}},
    )
    env = _unbooted(LiteOsworldEnv)
    env.bind(_task(extra_tool_schemas=[declared]), extra_tools=["bash"])
    names = [tool_schema_name(s) for s in env.metadata.extra_tool_schemas]
    assert names == ["task_local", "bash"]


def test_merge_extra_tool_schemas_is_order_preserving_and_idempotent():
    bash = LiteShellToolSet.get_tool_schema(BASH_TOOL_NAME)
    assert merge_extra_tool_schemas([bash], [bash]) == [bash]
    assert merge_extra_tool_schemas(None, [bash], []) == [bash]
    assert resolve_extra_tools(
        ["bash"],
        tools=_NoTools,
        executable=frozenset({BASH_TOOL_NAME}),
    ) == [bash]
