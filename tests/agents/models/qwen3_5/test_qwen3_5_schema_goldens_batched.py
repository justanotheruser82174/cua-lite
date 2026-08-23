"""Qwen3.5 provider-native wrapper schema guard for action-batch Lite calls.

Qwen3.5's wrapper tool (``computer_use`` desktop) is a PER-ACTION schema: one
``action`` string whose ``enum`` lists the inner verbs, plus flat sibling params
(``coordinate``/``text``/...). One native tool call = one action.

Approved Lite paths may batch GUI actions as ``computer{actions:[...]}``, but
qwen-native prompts stay per-action. Qwen adapters unwrap a canonical batch into
multiple native wrapper calls on render; they do not mutate the qwen tool schema
into an ``actions`` array.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \\
        tests/agents/models/qwen3_5/test_qwen3_5_schema_goldens_batched.py \\
        -p no:cacheprovider -q
"""

from __future__ import annotations

import pytest

from lite.agents.models.qwen3_5.action_space import Qwen3_5DesktopActionSpace
from lite.core.tools.schemas import tool_schema_name, tool_schema_parameters

_DESKTOP_ENUM = [
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "triple_click",
    "scroll",
    "hscroll",
    "wait",
    "terminate",
    "answer",
]

FAMILIES = {
    "qwen3_5_desktop": (Qwen3_5DesktopActionSpace, "computer_use", _DESKTOP_ENUM),
}


def _wrapper_tool(action_space_cls, tool_name: str) -> dict:
    schemas = action_space_cls().get_tool_schemas()
    matches = [s for s in schemas if tool_schema_name(s) == tool_name]
    assert matches, f"{action_space_cls.__name__}: no tool named {tool_name!r}"
    return matches[0]


@pytest.mark.parametrize("family", list(FAMILIES), ids=list(FAMILIES))
def test_qwen_native_per_action_schema_stays_per_action(family: str) -> None:
    cls, tool_name, enum = FAMILIES[family]
    params = tool_schema_parameters(_wrapper_tool(cls, tool_name))
    props = params["properties"]

    assert params["required"] == ["action"]
    assert props["action"]["type"] == "string"
    assert props["action"]["enum"] == enum
    assert "actions" not in props
