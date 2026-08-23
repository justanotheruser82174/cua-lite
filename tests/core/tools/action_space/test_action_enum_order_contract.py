"""The core GUI action ORDER is a contract with the model, and is pinned here.

``LiteDesktopActionSet``/``LiteMobileActionSet`` emit ONE action-batch tool (``computer`` /
``mobile``) whose ``actions[].action`` property carries an ``enum`` of every
action. That enum is rendered straight from ``_SCHEMAS`` insertion order, i.e.
from the order the ``@tool`` methods happen to be declared in
``lite/core/tools/action_space/base.py`` — so moving one ``def`` past another
rewrites the tool schema every model reads, at both training and inference time.

Nothing pinned that. ``DESKTOP_/LITE_MOBILE_ACTION_NAMES_IN_SCHEMA_ORDER`` are
``tuple(<Actions>._SCHEMAS)``, so every existing order assert derives its
expectation from the very thing it checks and stays green through any reorder
(measured: reversing ``LiteDesktopActionSet._SCHEMAS`` left the whole suite green).
The per-family schema goldens pin the families that REDECLARE their own actions,
not this core catalog.

WHAT A FAILURE MEANS — do not re-bless it by pasting the new order in.
A red assert here means the prompt every model sees has changed: the ``enum``
ordering is input tokens, so it shifts SFT/RL targets, invalidates comparison
with rollouts collected under the old order, and can move eval numbers on models
that were trained against it. That is an owner decision about the action-space
contract, not a golden update. If the reorder is intended, change the literal in
the SAME commit as the reorder and say so in the commit message.

Run:
    uv run pytest tests/core/tools/action_space/test_action_enum_order_contract.py -q
"""

from __future__ import annotations

import pytest

from lite.core.tools.action_space import LiteDesktopActionSet, LiteMobileActionSet
from lite.core.tools.schemas import tool_schema_name, tool_schema_parameters

# --- HAND-WRITTEN ON PURPOSE ------------------------------------------------
#
# Hand-written literals are usually wrong when they restate something the code
# already computes. This is the opposite case, and the exception is the whole
# point of the file: the value below is not a derivation of anything, it is a
# CONTRACT with the model, asserted from outside the source it guards.
#
# Any expression that reads ``_SCHEMAS`` (or ``get_action_names()``, or
# ``LITE_DESKTOP_ACTION_NAMES_IN_SCHEMA_ORDER``, which is ``tuple(_SCHEMAS)``) recomputes
# the expectation from the thing under test and can never redden on a reorder.
# So do NOT "clean this up" into a comprehension over the class. The literal
# must be typed by a human who decided this order is the one shipped.
_DESKTOP_ACTION_ENUM = [
    "key",
    "key_down",
    "key_up",
    "type",
    "hold_key",
    "mouse_move",
    "click",
    "drag",
    "mouse_down",
    "mouse_up",
    "scroll",
    "wait",
    "screenshot",
    "cursor_position",
]

# Mobile is model-visible in exactly the same way — same action-batch renderer,
# same ``actions[].action.enum`` slot, just the ``mobile`` tool —
# so it gets the same pin rather than a weaker one.
_MOBILE_ACTION_ENUM = [
    "tap",
    "long_press",
    "type",
    "swipe",
    "drag",
    "pinch",
    "system_button",
    "wait",
    "screenshot",
]

_PINNED_ACTION_ENUMS = (
    ("desktop", LiteDesktopActionSet, "computer", _DESKTOP_ACTION_ENUM),
    ("mobile", LiteMobileActionSet, "mobile", _MOBILE_ACTION_ENUM),
)


def _rendered_action_enum(tools) -> list[str]:
    """The action enum as the PROVIDER receives it — the output, not the input.

    Asserting against ``_SCHEMAS`` would pin the input to the renderer and miss
    a reorder introduced by the renderer itself, so this walks the emitted
    schema all the way down to the property the model actually reads.
    """
    (schema,) = tools.get_tool_schemas()
    return tool_schema_parameters(schema)["properties"]["actions"]["items"]["properties"]["action"][
        "enum"
    ]


@pytest.mark.parametrize("label,tools,wrapper,expected", _PINNED_ACTION_ENUMS)
def test_model_visible_action_enum_order_is_pinned(
    label: str, tools, wrapper: str, expected: list[str]
) -> None:
    (schema,) = tools.get_tool_schemas()
    assert tool_schema_name(schema) == wrapper, f"{label}: action-batch tool renamed"

    assert _rendered_action_enum(tools) == expected, (
        f"{label}: the {wrapper} action enum the model reads has changed. This "
        "is a prompt change for every training and eval run on this action "
        "space — see the module docstring before updating the literal."
    )


@pytest.mark.parametrize("label,tools,wrapper,expected", _PINNED_ACTION_ENUMS)
def test_filtering_the_enum_preserves_the_pinned_order(
    label: str, tools, wrapper: str, expected: list[str]
) -> None:
    """A subset stays in catalog order, never in the caller's order.

    ``include=``/``valid_actions=`` narrow the enum on every env that restricts
    the action space, so the contract has to hold for the narrowed prompt too —
    otherwise the order is pinned only for the unrestricted case that few
    rollouts actually use.
    """
    # Reversed request, catalog answer: the pinned relative order survives.
    subset = list(reversed(expected))[:4]
    narrowed = tools.get_tool_schemas(include=subset)
    (schema,) = narrowed
    enum = tool_schema_parameters(schema)["properties"]["actions"]["items"]["properties"]["action"][
        "enum"
    ]
    assert enum == [name for name in expected if name in set(subset)], label

    filtered = tools.filter_child_action_enum(tools.get_tool_schemas(), subset)
    (schema,) = filtered
    enum = tool_schema_parameters(schema)["properties"]["actions"]["items"]["properties"]["action"][
        "enum"
    ]
    assert enum == [name for name in expected if name in set(subset)], label
