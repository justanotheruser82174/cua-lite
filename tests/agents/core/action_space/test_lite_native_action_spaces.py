"""
Tests for CUA-lite native action spaces (desktop, mobile, point, bbox).

Covers:
  1. Tool call creation (static methods)
  2. Seam conversion (canonical Lite / bare agent-wire)
  3. Deep copy isolation
  4. Mobile action space

Run:
    uv run pytest tests/agents/core/action_space/test_lite_native_action_spaces.py -v
"""

from __future__ import annotations

from lite.agents.core.action_space.base import (
    LiteBBoxActionSpace,
    LiteDesktopActionSpace,
    LiteMobileActionSpace,
    LitePointActionSpace,
)
from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.schemas import tool_schema_name


def _only_action(tool_call: dict) -> dict:
    actions = tool_call_arguments(tool_call)["actions"]
    assert len(actions) == 1
    return actions[0]


# =============================================================================
# 1. Desktop: tool call creation
# =============================================================================


class TestDesktopToolCalls:
    """Test LiteDesktopActionSpace static tool call methods."""

    def test_click_basic(self):
        tc = LiteDesktopActionSpace.click(coordinate=[500, 300])
        assert tool_call_name(tc) == "computer"
        action = _only_action(tc)
        assert action["action"] == "click"
        assert action["coordinate"] == [500, 300]
        assert "button" not in action  # default omitted

    def test_click_right(self):
        tc = LiteDesktopActionSpace.click(coordinate=[100, 200], button="right")
        assert _only_action(tc)["button"] == "right"

    def test_click_double(self):
        tc = LiteDesktopActionSpace.click(coordinate=[100, 200], clicks=2)
        assert _only_action(tc)["clicks"] == 2

    def test_type(self):
        tc = LiteDesktopActionSpace.type(text="hello")
        assert tool_call_name(tc) == "computer"
        action = _only_action(tc)
        assert action["action"] == "type"
        assert action["text"] == "hello"

    def test_key(self):
        tc = LiteDesktopActionSpace.key(keys=["ctrl", "c"])
        assert _only_action(tc)["keys"] == ["ctrl", "c"]

    def test_scroll(self):
        tc = LiteDesktopActionSpace.scroll(direction="down", amount=3, coordinate=[500, 500])
        action = _only_action(tc)
        assert action["direction"] == "down"
        assert action["amount"] == 3
        assert action["coordinate"] == [500, 500]

    def test_drag(self):
        tc = LiteDesktopActionSpace.drag(coordinate=[800, 600], start_coordinate=[100, 100])
        action = _only_action(tc)
        assert action["coordinate"] == [800, 600]
        assert action["start_coordinate"] == [100, 100]

    def test_mouse_move(self):
        tc = LiteDesktopActionSpace.mouse_move(coordinate=[300, 400])
        assert _only_action(tc)["coordinate"] == [300, 400]

    def test_wait(self):
        tc = LiteDesktopActionSpace.wait(duration=2.0)
        assert _only_action(tc)["duration"] == 2.0

    def test_screenshot(self):
        tc = LiteDesktopActionSpace.screenshot()
        assert tool_call_name(tc) == "computer"
        assert _only_action(tc)["action"] == "screenshot"

    def test_terminate(self):
        tc = make_tool_call("terminate", {"status": "success"})
        assert tool_call_arguments(tc)["status"] == "success"

    def test_response(self):
        tc = make_tool_call("response", {"text": "42"})
        assert tool_call_arguments(tc)["text"] == "42"

    def test_cursor_position(self):
        tc = LiteDesktopActionSpace.cursor_position()
        assert tool_call_name(tc) == "computer"
        assert _only_action(tc)["action"] == "cursor_position"

    def test_click_constructor_emits_batched_lite_tool_call(self):
        tc = LiteDesktopActionSpace.click(coordinate=[500, 300])
        assert set(tc) == {"type", "function"}
        assert tool_call_name(tc) == "computer"
        assert _only_action(tc) == {"action": "click", "coordinate": [500, 300]}

    def test_desktop_action_space_does_not_expose_finish_schemas_by_default(self):
        actions = LiteDesktopActionSpace.get_declared_action_schema_names()
        schemas = LiteDesktopActionSpace.get_tool_schemas()
        assert "terminate" not in actions
        assert "response" not in actions
        assert {tool_schema_name(s) for s in schemas}.isdisjoint({"terminate", "response"})


# =============================================================================
# 2. Desktop: seam conversion
# =============================================================================


class TestDesktopSeamConversion:
    """LiteDesktopActionSpace unwraps to bare agent-wire and wraps it back."""

    def setup_method(self):
        self.space = LiteDesktopActionSpace()

    def test_to_agent_unwraps_to_bare_agent_wire(self):
        calls = [LiteDesktopActionSpace.click(coordinate=[500, 300])]
        out = self.space.convert_tool_calls_to_agent(calls)
        assert out == [
            {
                "name": "computer",
                "arguments": {"actions": [{"action": "click", "coordinate": [500, 300]}]},
            }
        ]
        assert out is not calls

    def test_from_agent_wraps_bare_agent_wire(self):
        calls = [
            {
                "name": "computer",
                "arguments": {"actions": [{"action": "key", "keys": ["ctrl", "c"]}]},
            }
        ]
        out = self.space.convert_tool_calls_from_agent(calls)
        assert out == [LiteDesktopActionSpace.key(keys=["ctrl", "c"])]
        assert out is not calls

    def test_deep_copy_isolation(self):
        calls = [LiteDesktopActionSpace.click(coordinate=[500, 300])]
        out = self.space.convert_tool_calls_to_agent(calls)
        out[0]["arguments"]["actions"][0]["coordinate"][0] = 999
        assert tool_call_arguments(calls[0])["actions"][0]["coordinate"][0] == 500


# =============================================================================
# 3. Mobile action space
# =============================================================================


class TestMobileToolCalls:
    """Test LiteMobileActionSpace static tool call methods."""

    def test_tap(self):
        tc = LiteMobileActionSpace.tap(coordinate=[500, 300])
        assert tool_call_name(tc) == "mobile"
        action = _only_action(tc)
        assert action["action"] == "tap"
        assert action["coordinate"] == [500, 300]

    def test_long_press(self):
        tc = LiteMobileActionSpace.long_press(coordinate=[100, 200], duration=1.5)
        assert _only_action(tc)["duration"] == 1.5

    def test_swipe(self):
        tc = LiteMobileActionSpace.swipe(start_coordinate=[100, 500], coordinate=[100, 100])
        action = _only_action(tc)
        assert action["start_coordinate"] == [100, 500]
        assert action["coordinate"] == [100, 100]

    def test_open_app(self):
        tc = make_tool_call("open_app", {"app_name": "Settings"})
        assert tool_call_arguments(tc)["app_name"] == "Settings"

    def test_system_button_home(self):
        tc = LiteMobileActionSpace.system_button(button="Home")
        assert tool_call_name(tc) == "mobile"
        assert _only_action(tc) == {"action": "system_button", "button": "Home"}

    def test_system_button_back(self):
        tc = LiteMobileActionSpace.system_button(button="Back")
        assert tool_call_name(tc) == "mobile"
        assert _only_action(tc) == {"action": "system_button", "button": "Back"}

    def test_system_button_enter(self):
        tc = LiteMobileActionSpace.system_button(button="Enter")
        assert tool_call_name(tc) == "mobile"
        assert _only_action(tc) == {"action": "system_button", "button": "Enter"}

    def test_mobile_available_actions(self):
        actions = LiteMobileActionSpace.get_declared_action_schema_names()
        for name in ["tap", "long_press", "type", "swipe", "system_button", "wait"]:
            assert name in actions
        for name in ["open_app", "response", "terminate"]:
            assert name not in actions

    def test_mobile_seam_conversion(self):
        space = LiteMobileActionSpace()
        calls = [LiteMobileActionSpace.tap(coordinate=[500, 300])]
        out = space.convert_tool_calls_to_agent(calls)
        assert out == [
            {
                "name": "mobile",
                "arguments": {
                    "actions": [{"action": "tap", "coordinate": [500, 300], "clicks": 1}]
                },
            }
        ]
        assert out is not calls
        assert space.convert_tool_calls_from_agent(out) == calls

    def test_tap_constructor_emits_batched_lite_tool_call(self):
        tc = LiteMobileActionSpace.tap(coordinate=[500, 300])
        assert set(tc) == {"type", "function"}
        assert tool_call_name(tc) == "mobile"
        assert _only_action(tc) == {"action": "tap", "coordinate": [500, 300], "clicks": 1}


# =============================================================================
# 4. Point / BBox action spaces
# =============================================================================


class TestPointActionSpace:
    def test_point_call(self):
        tc = LitePointActionSpace.point(coordinate=[850, 120])
        assert tool_call_name(tc) == "point"
        assert tool_call_arguments(tc)["coordinate"] == [850, 120]

    def test_point_seam_conversion(self):
        space = LitePointActionSpace()
        calls = [LitePointActionSpace.point(coordinate=[850, 120])]
        agent_calls = [{"name": "point", "arguments": {"coordinate": [850, 120]}}]
        assert space.convert_tool_calls_to_agent(calls) == agent_calls
        assert space.convert_tool_calls_from_agent(agent_calls) == calls


class TestBBoxActionSpace:
    def test_bbox_call(self):
        tc = LiteBBoxActionSpace.bbox(coordinate=[380, 450, 620, 520])
        assert tool_call_name(tc) == "bbox"
        assert tool_call_arguments(tc)["coordinate"] == [380, 450, 620, 520]

    def test_bbox_seam_conversion(self):
        space = LiteBBoxActionSpace()
        calls = [LiteBBoxActionSpace.bbox(coordinate=[380, 450, 620, 520])]
        agent_calls = [{"name": "bbox", "arguments": {"coordinate": [380, 450, 620, 520]}}]
        assert space.convert_tool_calls_to_agent(calls) == agent_calls
        assert space.convert_tool_calls_from_agent(agent_calls) == calls
