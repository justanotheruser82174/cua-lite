"""Duration ceilings have ONE owner, and the model is told every one of them.

Policy: ``lite.core.tools.action_space.duration`` owns the whole action-name ->
ceiling table, including ``swipe``/``drag``. Those two declare no ``duration``
parameter of their own, but the action-batch tool merges every child's
properties into one shared ``actions.items`` object, so the model can attach
``duration`` to any action item and the backends enforce these ceilings on it.
A ceiling the model can trip must appear in the advertised clause.

Run:
    uv run pytest tests/core/tools/action_space/test_duration_cap_contract.py -q
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lite.core.tools.action_space.base import (
    LiteDesktopActionSet,
    LiteMobileActionSet,
)
from lite.core.tools.action_space.duration import (
    ACTION_SCHEMA_DURATION_CAP_DESCRIPTION,
    ACTION_SCHEMA_DURATION_CAPS_SECONDS,
)
from lite.core.tools.schemas import tool_schema_parameters


def _batch_duration_property(actions_cls, action_batch_tool_name):
    schema = actions_cls.get_tool_schema(action_batch_tool_name)
    return tool_schema_parameters(schema)["properties"]["actions"]["items"]["properties"][
        "duration"
    ]


def _rendered_child_properties(actions_cls, action_name):
    (schema,) = actions_cls.get_tool_schemas(include=[action_name])
    return tool_schema_parameters(schema)["properties"]["actions"]["items"]["properties"]


def test_backend_enforcement_reads_the_core_table_it_does_not_restate_it():
    """Closure guard: a second ``ACTION_SCHEMA_DURATION_CAPS_SECONDS`` literal in
    the backend layer is what let the advertised clause and the enforced numbers
    drift apart. The backend must hold the SAME object, not an equal copy."""
    from lite.gym.utils.backend import model_inputs

    assert (
        model_inputs.ACTION_SCHEMA_DURATION_CAPS_SECONDS
        is ACTION_SCHEMA_DURATION_CAPS_SECONDS
    )


def test_the_ceiling_table_is_written_down_exactly_once():
    """Same guard, widened past ``model_inputs`` to every reader.

    Four production modules resolve a ceiling from this table -- the backend
    model-input owner plus the three Android dispatchers -- and each one caps a
    value the advertised clause already names, so all four must READ the owner.
    A backend-local table would be a second spelling of one fact: it can hold
    numbers the clause does not state, which is precisely how a model came to
    have its whole batch rejected for a rule it was never told.
    """
    repo_root = Path(__file__).resolve().parents[4]
    assignment = re.compile(r"^\s*ACTION_SCHEMA_DURATION_CAPS_SECONDS\s*(:[^=]+)?=")

    definitions = sorted(
        str(path.relative_to(repo_root))
        for path in repo_root.joinpath("lite").rglob("*.py")
        if any(assignment.match(line) for line in path.read_text().splitlines())
    )

    assert definitions == ["lite/core/tools/action_space/duration.py"]


@pytest.mark.parametrize(
    ("actions_cls", "action_batch_tool_name"),
    [(LiteDesktopActionSet, "computer"), (LiteMobileActionSet, "mobile")],
)
def test_batch_schema_advertises_every_duration_ceiling(
    actions_cls,
    action_batch_tool_name,
):
    """The batch schema merges all children's ``duration`` props into ONE."""
    description = _batch_duration_property(actions_cls, action_batch_tool_name)[
        "description"
    ]

    assert ACTION_SCHEMA_DURATION_CAP_DESCRIPTION in description
    assert "wait <= 30" in description
    assert "hold_key <= 5" in description


@pytest.mark.parametrize(
    ("actions_cls", "action_name"),
    [
        (LiteDesktopActionSet, "wait"),
        (LiteDesktopActionSet, "hold_key"),
        (LiteMobileActionSet, "wait"),
        (LiteMobileActionSet, "long_press"),
    ],
)
def test_each_duration_declaring_action_states_its_ceiling(actions_cls, action_name):
    prop = _rendered_child_properties(actions_cls, action_name)["duration"]

    assert ACTION_SCHEMA_DURATION_CAP_DESCRIPTION in prop["description"]


def test_advertised_ceilings_match_the_enforced_numbers():
    for action_name, cap in ACTION_SCHEMA_DURATION_CAPS_SECONDS.items():
        assert f"{action_name} <= {cap:g}" in ACTION_SCHEMA_DURATION_CAP_DESCRIPTION


def test_the_rejection_consequence_is_advertised_and_scoped_to_one_action():
    """R4: a rejected duration costs ITSELF, not the batch — say exactly that.

    The old wording promised "the rest of the batch is not executed", which was
    true under the abort policy R4 repealed. Leaving it would make the prompt
    lie to the model about a rule that no longer holds, which is prompt-surface
    drift, not a stale comment.
    """
    assert "that action does not run" in ACTION_SCHEMA_DURATION_CAP_DESCRIPTION
    assert "rest of the batch" not in ACTION_SCHEMA_DURATION_CAP_DESCRIPTION


def test_gesture_durations_are_advertised_even_though_no_child_declares_one():
    """``swipe``/``drag`` take ``duration`` only through the merged batch item.

    The child schemas carry no ``duration`` parameter, so nothing in the shape of
    the tool tells the model these two are capped -- only the clause does. A
    ceiling reachable from the batch item and silent in the clause is exactly the
    rejection the model cannot predict.
    """
    for action_name in ("swipe", "drag"):
        assert ACTION_SCHEMA_DURATION_CAPS_SECONDS[action_name] == 5.0
        assert f"{action_name} <= 5" in ACTION_SCHEMA_DURATION_CAP_DESCRIPTION

    assert "duration" not in _rendered_child_properties(LiteDesktopActionSet, "drag")
    assert "duration" not in _rendered_child_properties(LiteMobileActionSet, "swipe")


@pytest.mark.parametrize(
    ("actions_cls", "action_batch_tool_name"),
    [(LiteDesktopActionSet, "computer"), (LiteMobileActionSet, "mobile")],
)
def test_the_merged_batch_item_lets_duration_reach_any_action(
    actions_cls,
    action_batch_tool_name,
):
    """Why the clause must enumerate: the item schema is one flat object.

    ``duration`` sits beside every other child's properties with only ``action``
    required and no per-action discrimination, so ``{"action": "swipe",
    "duration": 10}`` is schema-legal and reaches the backend cap.
    """
    parameters = tool_schema_parameters(
        actions_cls.get_tool_schema(action_batch_tool_name)
    )
    item = parameters["properties"]["actions"]["items"]

    assert "duration" in item["properties"]
    assert item["required"] == ["action"]
