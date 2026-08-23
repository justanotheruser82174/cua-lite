"""``unknown_wrapper_action_batch`` owns one shape, and refuses to fake it.

Run:
    uv run pytest tests/agents/core/action_space/test_unknown_wrapper_action.py -v

The module keeps an unknown provider-native wrapper action PAIRABLE: the child
action name is the model's own action value, so env ingress rejects it by name
and the model sees the rejection instead of the call vanishing.

A call carrying NO action value is a different fact. There is no name to reject
by, so the pairable contract cannot hold. Defaulting the name to ``""`` built a
child that ``validate_lite_action_batch_structure`` calls unrenderable — which
did not fail at conversion but one turn LATER, when the turn was replayed into
history, killing the episode with an uncaught ``ValueError``. Observed on a
Qwen3.8 lite.osworld rollout: ``computer_use(command="true",
coordinate=[263, 798])`` — the model hallucinated a parameter name and omitted
``action`` entirely.
"""

from __future__ import annotations

import pytest

from lite.agents.bootstrap import register_all
from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.core.action_space.utils.unknown_wrapper_action import (
    unknown_wrapper_action_batch,
)
from lite.core.tools.action_space import (
    LITE_COMPUTER_ACTION_BATCH_TOOL_NAME,
    LITE_MOBILE_ACTION_BATCH_TOOL_NAME,
    validate_lite_action_batch_structure,
)
from lite.core.tools.calls import tool_call_arguments, tool_call_name

register_all()


@pytest.mark.parametrize(
    "batch_tool_name",
    [LITE_COMPUTER_ACTION_BATCH_TOOL_NAME, LITE_MOBILE_ACTION_BATCH_TOOL_NAME],
)
def test_unknown_action_value_stays_pairable(batch_tool_name):
    """The designed path: an unknown VERB keeps its name so env ingress can
    reject it by name and the model sees that rejection."""
    call = unknown_wrapper_action_batch(
        batch_tool_name, {"action": "frobnicate", "coordinate": [1, 2]}
    )
    assert tool_call_name(call) == batch_tool_name
    children = tool_call_arguments(call)["actions"]
    assert children == [{"action": "frobnicate", "coordinate": [1, 2]}]
    # Deliberately invalid by NAME, but structurally renderable — that is the
    # whole point, and it is what makes the rejection reach the model.
    _, error = validate_lite_action_batch_structure(
        batch_tool_name, tool_call_arguments(call)
    )
    assert error is None


@pytest.mark.parametrize(
    "args",
    [
        {"command": "true", "coordinate": [263, 798]},  # verbatim from the rollout
        {"action": "", "coordinate": [1, 2]},
        {"action": None},
        {"action": 3},
        {},
    ],
    ids=["no-action-key", "empty", "none", "non-string", "empty-args"],
)
def test_no_action_value_raises_the_named_parse_error(args):
    """The other fact: no verb at all is not a usable call, so the shape must
    not be constructed. ``ModelToolCallParseError`` is the channel the agent
    loop already turns into a clean parse-failure final."""
    with pytest.raises(ModelToolCallParseError):
        unknown_wrapper_action_batch(LITE_COMPUTER_ACTION_BATCH_TOOL_NAME, args)


def test_the_named_error_is_the_one_the_agent_loop_catches():
    """``lite/agents/core/agent/base.py`` catches ``ModelToolCallParseError``
    ahead of incidental ``ValueError`` bugs, which still propagate."""
    assert issubclass(ModelToolCallParseError, ValueError)


def test_an_unnamed_child_would_have_been_unrenderable():
    """Pins WHY the empty default was fatal rather than merely ugly: nothing
    downstream can project a nameless child back to ``action=<verb>``."""
    _, error = validate_lite_action_batch_structure(
        LITE_COMPUTER_ACTION_BATCH_TOOL_NAME,
        {"actions": [{"action": "", "command": "true"}]},
    )
    assert error is not None
    assert "must be a non-empty string" in error.reason
