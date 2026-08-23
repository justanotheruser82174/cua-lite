"""The unknown-action policy is direction-ASYMMETRIC, and deliberately so.

``convert_tool_calls_from_agent`` (agent -> lite) parses a CLOSED set: the
model's own native action table. A name outside it is a hallucination, so the
GPT/Claude action spaces raise at their own parse boundary.

``convert_tool_calls_to_agent`` (lite -> agent) re-emits an OPEN set. Its
fallthrough is the env ``extra_tool`` channel — ``terminate``, ``goto``,
``open_app`` and any env-local tool declared through ``env_kwargs.extra_tools``
— whose membership the action space structurally cannot know. Raising there
would delete the agent's non-GUI capability, so it passes through as bare
function-call agent wire and takes no strictness flag at all.

Flattening the asymmetry in either direction is a regression, not a cleanup.

Run:
    uv run pytest tests/agents/core/action_space/test_unknown_action_direction_policy.py -v
"""

from __future__ import annotations

import pytest

from lite.agents.core.action_space.base import ActionSpaceRegistry
from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_arguments, tool_call_name

RESOLUTION = (1000, 1000)

#: Action-space keys whose ``from_agent`` rejects unknown native names.
#: Measured over all registered keys; nothing outside this set raises today.
#: ``gemini@*`` raises here too, even though its agent never reaches this branch
#: for an unknown name -- the mobile loop routes env extras first and passes the
#: rest through -- so the space's own contract stays pinned alongside its twins.
#:
#: The two ``@mobile`` entries are the point of the parity test below: a mobile
#: space must reject a nameless-call fallthrough just like its other native
#: branches. ``claude@mobile`` used to silently swallow an unknown nameless
#: action while its ``gpt@mobile`` twin raised under an opt-in flag.
UNKNOWN_NATIVE_KEYS = (
    "claude@desktop",
    "claude@desktop@point",
    "claude@mobile",
    "claude@browser",
    "claude@browser@point",
    "gemini@desktop",
    "gemini@mobile",
    "gemini@browser",
    "gpt@desktop",
    "gpt@desktop@point",
    "gpt@mobile",
    "gpt@browser",
    "gpt@browser@point",
)

#: The provider twins whose ``from_agent`` unknown-action policy must not diverge.
MOBILE_TWINS = ("claude@mobile", "gpt@mobile")

#: Names that reach the ``to_agent`` fallthrough as legitimate env extra tools.
EXTRA_TOOL_CALLS = (
    make_tool_call("terminate", {"status": "success"}),
    make_tool_call("goto", {"url": "http://example.com"}),
    make_tool_call("open_app", {"app_name": "Chrome"}),
)


@pytest.fixture(autouse=True, scope="module")
def _registered():
    from lite.agents.bootstrap import register_all

    register_all()


def _all_keys() -> list[str]:
    return sorted(ActionSpaceRegistry.list_expanded())


class TestToAgentNeverRaises:
    """lite -> agent: an unrecognized name is data, not an error."""

    def test_no_action_space_raises_on_an_unknown_name(self):
        offenders = []
        for key in _all_keys():
            space = ActionSpaceRegistry.get(key)
            call = make_tool_call("totally_bogus_action", {"foo": 1})
            try:
                space.convert_tool_calls_to_agent([call], resolution=RESOLUTION)
            except Exception as exc:  # noqa: BLE001 - the failure IS the report
                offenders.append(f"{key}: {type(exc).__name__}: {exc}")
        assert offenders == []

    @pytest.mark.parametrize("key", ["gpt@desktop", "claude@desktop"])
    @pytest.mark.parametrize("call", EXTRA_TOOL_CALLS, ids=tool_call_name)
    def test_env_extra_tool_survives_the_fallthrough(self, key, call):
        """The very branch that "passes unknown names through" IS this channel."""
        space = ActionSpaceRegistry.get(key)
        emitted = space.convert_tool_calls_to_agent([call], resolution=RESOLUTION)
        assert emitted == [{"name": tool_call_name(call), "arguments": tool_call_arguments(call)}]


class TestFromAgentRejectsUnknownNative:
    """agent -> lite: a name outside the native table is a model error."""

    @pytest.mark.parametrize("key", UNKNOWN_NATIVE_KEYS)
    def test_unknown_native_name_raises_by_default(self, key):
        space = ActionSpaceRegistry.get(key)
        action = {"action": "totally_bogus_action", "type": "totally_bogus_action"}
        with pytest.raises(ValueError, match="unknown"):
            space.convert_tool_calls_from_agent([action], resolution=RESOLUTION)

    @pytest.mark.parametrize("key", MOBILE_TWINS)
    def test_nameless_call_fallthrough_rejects_unknown_actions(self, key):
        """A call with no ``name`` is a bare action dict and takes the loop's
        last branch. It must reject unknown native actions like the named branch."""
        space = ActionSpaceRegistry.get(key)
        nameless = {"action": "totally_bogus_action"}
        assert "name" not in nameless
        with pytest.raises(ValueError, match="unknown"):
            space.convert_tool_calls_from_agent([nameless], resolution=RESOLUTION)

    def test_mobile_twins_agree_on_unknown_action_errors(self):
        """claude@mobile and gpt@mobile are twins; a drop in one and an error
        in the other is a divergence, not a dialect difference."""
        nameless = {"action": "totally_bogus_action"}
        raised = {}
        for key in MOBILE_TWINS:
            space = ActionSpaceRegistry.get(key)
            try:
                space.convert_tool_calls_from_agent([nameless], resolution=RESOLUTION)
                raised[key] = False
            except ValueError:
                raised[key] = True
        assert len(set(raised.values())) == 1, raised
