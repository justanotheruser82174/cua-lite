"""Claude desktop schema and native conversion-order goldens."""

from __future__ import annotations

from lite.agents.models.claude.action_space import ClaudeDesktopActionSpace
from lite.core.tools.calls import tool_call_arguments


def _assert_canonical(schemas):
    assert all(set(schema) == {"type", "function"} for schema in schemas)
    assert all(schema["type"] == "function" for schema in schemas)
    assert all(isinstance(schema["function"], dict) for schema in schemas)


def test_claude_desktop_schema_golden():
    actual = ClaudeDesktopActionSpace().get_tool_schemas()
    _assert_canonical(actual)
    assert actual == []


def test_claude_native_conversion_order_locked():
    native = [
        {"action": "left_click", "coordinate": [10, 20]},
        {"action": "right_click", "coordinate": [11, 21]},
        {"action": "middle_click", "coordinate": [12, 22]},
        {"action": "double_click", "coordinate": [13, 23]},
        {"action": "type", "text": "hello"},
        {"action": "key", "text": "ENTER"},
        {
            "action": "scroll",
            "coordinate": [30, 40],
            "scroll_direction": "down",
            "scroll_amount": 3,
        },
        {"action": "left_click_drag", "start_coordinate": [1, 2], "end_coordinate": [3, 4]},
        {"action": "mouse_move", "coordinate": [50, 60]},
        {"action": "left_mouse_down", "coordinate": [70, 80]},
        {"action": "left_mouse_up", "coordinate": [90, 100]},
        {"action": "screenshot"},
        {"action": "wait"},
    ]

    [call] = ClaudeDesktopActionSpace().convert_tool_calls_from_agent(
        native,
        resolution=(1000, 1000),
    )

    assert [action["action"] for action in tool_call_arguments(call)["actions"]] == [
        "click",
        "click",
        "click",
        "click",
        "type",
        "key",
        "scroll",
        "drag",
        "mouse_move",
        "mouse_move",
        "mouse_down",
        "mouse_move",
        "mouse_up",
        "screenshot",
        "wait",
    ]
