"""GPT desktop schema and native conversion-order goldens."""

from __future__ import annotations

from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
from lite.core.tools.calls import tool_call_arguments


def _assert_canonical(schemas):
    assert all(set(schema) == {"type", "function"} for schema in schemas)
    assert all(schema["type"] == "function" for schema in schemas)
    assert all(isinstance(schema["function"], dict) for schema in schemas)


def test_gpt_desktop_schema_golden():
    actual = GPTDesktopActionSpace().get_tool_schemas()
    _assert_canonical(actual)
    assert actual == []


def test_gpt_native_conversion_order_locked():
    native = [
        {"type": "click", "x": 10, "y": 20},
        {"type": "double_click", "x": 13, "y": 23},
        {"type": "right_click", "x": 11, "y": 21},
        {"type": "type", "text": "hello"},
        {"type": "keypress", "keys": ["ENTER"]},
        {"type": "scroll", "x": 30, "y": 40, "scroll_y": 300},
        {"type": "move", "x": 50, "y": 60},
        {"type": "drag", "start_x": 1, "start_y": 2, "end_x": 3, "end_y": 4},
        {"type": "screenshot"},
        {"type": "wait"},
    ]

    [call] = GPTDesktopActionSpace().convert_tool_calls_from_agent(
        native,
        resolution=(1000, 1000),
    )

    assert [action["action"] for action in tool_call_arguments(call)["actions"]] == [
        "click",
        "click",
        "click",
        "type",
        "key",
        "scroll",
        "mouse_move",
        "drag",
        "screenshot",
        "wait",
    ]
