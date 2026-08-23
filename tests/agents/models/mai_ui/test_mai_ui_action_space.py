"""
Tests for MAIUIMobileActionSpace.

Covers:
  1. Registry + single mobile_use schema + action enum
  2. Tool call creation (mobile_use)
  3. Tool call conversion: cua-lite <-> MAI-UI mobile (forward + reverse)
  4. Coordinate rescale 999 <-> 1000
  5. Swipe direction synthesis (Option A)
  6. system_button case mapping
  7. terminate status mapping (success/failure <-> success/fail)
  8. Round-trip equality (with rescale tolerance)

Run:
    uv run pytest tests/agents/models/mai_ui/test_mai_ui_action_space.py -v
"""

from __future__ import annotations

import pytest
from agents._support.valid_actions_gating import (
    OPEN_APP_SCHEMA,
    RESPONSE_SCHEMA,
    TERMINATE_SCHEMA,
    agent_adapter_for,
)

from lite.agents.core.action_space.base import ActionSpaceRegistry, LiteMobileActionSpace
from lite.agents.models.mai_ui.action_space import (
    _BUTTON_LITE_TO_MAI,
    _BUTTON_MAI_TO_LITE,
    MAIUIMobileActionSpace,
    _direction_from_endpoints,
    _endpoints_from_direction,
)
from lite.core.tools import make_tool_call
from lite.core.tools.action_space import (
    lite_action_batch_child_name_errors,
    validate_lite_action_batch_structure,
)
from lite.core.tools.calls import (
    tool_call_arguments,
    tool_call_name,
    validate_lite_tool_call,
)
from lite.core.tools.schemas import (
    make_tool_schema,
    tool_schema_name,
    tool_schema_parameters,
)


def _only_mobile_action(tool_calls):
    assert len(tool_calls) == 1
    call = tool_calls[0]
    assert tool_call_name(call) == "mobile"
    actions = tool_call_arguments(call)["actions"]
    assert len(actions) == 1
    return actions[0]


def _canonical_args(tool_call, name="mobile_use"):
    assert validate_lite_tool_call(tool_call, "tool_call", require_id=False) is None
    assert tool_call_name(tool_call) == name
    return tool_call_arguments(tool_call)


@pytest.mark.parametrize(
    "call,schema",
    [
        (make_tool_call("response", {"text": "42"}), RESPONSE_SCHEMA),
        (make_tool_call("terminate", {"status": "success"}), TERMINATE_SCHEMA),
    ],
)
def test_mai_ui_finish_render_delegates_to_family_converter(call, schema) -> None:
    message = {"role": "assistant", "content": [], "tool_calls": [call]}

    adapter = agent_adapter_for("mai_ui@mobile@use", "mobile")
    adapter.convert_message_to_agent(message)

    adapter = agent_adapter_for(
        "mai_ui@mobile@use",
        "mobile",
        extra_tool_schemas=[schema],
    )
    adapter.convert_message_to_agent(message)


def test_mai_ui_open_app_render_does_not_require_matching_extra_schema() -> None:
    message = {
        "role": "assistant",
        "content": [],
        "tool_calls": [make_tool_call("open_app", {"app_name": "Settings"})],
    }

    adapter = agent_adapter_for("mai_ui@mobile@use", "mobile")
    adapter.convert_message_to_agent(message)

    adapter = agent_adapter_for(
        "mai_ui@mobile@use",
        "mobile",
        extra_tool_schemas=[OPEN_APP_SCHEMA],
    )
    adapter.convert_message_to_agent(message)


# =============================================================================
# 1. Registry + schema
# =============================================================================


class TestRegistryAndSchema:
    def test_registry_key(self):
        space = ActionSpaceRegistry.get("mai_ui@mobile")
        assert isinstance(space, MAIUIMobileActionSpace)

    def test_single_tool(self):
        schemas = MAIUIMobileActionSpace.get_tool_schemas()
        assert len(schemas) == 1
        assert tool_schema_name(schemas[0]) == "mobile_use"

    def test_action_enum(self):
        schema = MAIUIMobileActionSpace.get_tool_schemas()[0]
        actions = set(tool_schema_parameters(schema)["properties"]["action"]["enum"])
        # MAI-UI has these 10 actions in v1 (no key, no ask_user, no double_click).
        expected = {
            "click",
            "long_press",
            "type",
            "swipe",
            "drag",
            "open",
            "system_button",
            "wait",
            "terminate",
            "answer",
        }
        assert actions == expected

    def test_button_enum_lowercase(self):
        """MAI-UI uses lowercase button names (vs Qwen3VL's Capitalized)."""
        schema = MAIUIMobileActionSpace.get_tool_schemas()[0]
        buttons = set(tool_schema_parameters(schema)["properties"]["button"]["enum"])
        assert buttons == {"back", "home", "menu", "enter"}

    def test_status_enum_fail_not_failure(self):
        """MAI-UI uses 'fail' (not 'failure') in terminate status."""
        schema = MAIUIMobileActionSpace.get_tool_schemas()[0]
        statuses = set(tool_schema_parameters(schema)["properties"]["status"]["enum"])
        assert statuses == {"success", "fail"}

    def test_swipe_direction_enum(self):
        schema = MAIUIMobileActionSpace.get_tool_schemas()[0]
        dirs = set(tool_schema_parameters(schema)["properties"]["direction"]["enum"])
        assert dirs == {"up", "down", "left", "right"}


# =============================================================================
# 2. Tool call creation
# =============================================================================


class TestToolCalls:
    def test_click(self):
        tc = MAIUIMobileActionSpace.mobile_use(action="click", coordinate=[499, 499])
        args = _canonical_args(tc)
        assert args["action"] == "click"
        assert args["coordinate"] == [499, 499]

    def test_swipe_with_anchor(self):
        tc = MAIUIMobileActionSpace.mobile_use(
            action="swipe",
            direction="up",
            coordinate=[500, 800],
        )
        args = _canonical_args(tc)
        assert args["direction"] == "up"
        assert args["coordinate"] == [500, 800]

    def test_swipe_no_anchor(self):
        """Anchor is optional in MAI-UI."""
        tc = MAIUIMobileActionSpace.mobile_use(action="swipe", direction="down")
        args = _canonical_args(tc)
        assert args["direction"] == "down"
        assert "coordinate" not in args

    def test_drag(self):
        tc = MAIUIMobileActionSpace.mobile_use(
            action="drag",
            start_coordinate=[100, 200],
            end_coordinate=[700, 200],
        )
        args = _canonical_args(tc)
        assert args["start_coordinate"] == [100, 200]
        assert args["end_coordinate"] == [700, 200]

    def test_type(self):
        tc = MAIUIMobileActionSpace.mobile_use(action="type", text="hello")
        assert _canonical_args(tc)["text"] == "hello"

    def test_open_uses_text_field(self):
        """MAI-UI uses 'text' for app name (not 'app_name')."""
        tc = MAIUIMobileActionSpace.mobile_use(action="open", text="Chrome")
        args = _canonical_args(tc)
        assert args["text"] == "Chrome"
        assert "app_name" not in args

    def test_system_button_lowercase(self):
        tc = MAIUIMobileActionSpace.mobile_use(action="system_button", button="back")
        assert _canonical_args(tc)["button"] == "back"

    def test_terminate_fail(self):
        tc = MAIUIMobileActionSpace.mobile_use(action="terminate", status="fail")
        assert _canonical_args(tc)["status"] == "fail"

    def test_none_args_omitted(self):
        tc = MAIUIMobileActionSpace.mobile_use(action="wait")
        # Only action is set; everything else None and dropped.
        assert _canonical_args(tc) == {"action": "wait"}


# =============================================================================
# 3. Coordinate rescale
# =============================================================================


class TestCoordinateRescale:
    @staticmethod
    def _rendered_tap_coordinate(coord):
        space = MAIUIMobileActionSpace()
        [call] = space.convert_tool_calls_to_agent(
            [LiteMobileActionSpace.tap(coordinate=coord)]
        )
        return call["arguments"]["coordinate"]

    @staticmethod
    def _parsed_click_coordinate(coord):
        space = MAIUIMobileActionSpace()
        parsed = space.convert_tool_calls_from_agent([
            {
                "name": "mobile_use",
                "arguments": {"action": "click", "coordinate": coord},
            }
        ])
        return _only_mobile_action(parsed)["coordinate"]

    def test_to_mai_zero(self):
        assert self._rendered_tap_coordinate([0, 0]) == [0, 0]

    def test_to_mai_max(self):
        assert self._rendered_tap_coordinate([1000, 1000]) == [999, 999]

    def test_to_mai_midpoint(self):
        # 500 * 999 / 1000 = 499.5 -> rounds to 500 (banker's) or 499? Python's
        # int(round(499.5)) is 500 (round-half-to-even); but round(499.5) is 500.
        # Actually round(499.5) in Python 3 is 500 (banker's: closest even).
        result = self._rendered_tap_coordinate([500, 500])
        assert result[0] in (499, 500)

    def test_from_mai_zero(self):
        assert self._parsed_click_coordinate([0, 0]) == [0, 0]

    def test_from_mai_max(self):
        # 999 * 1000 / 999 = 1000
        assert self._parsed_click_coordinate([999, 999]) == [1000, 1000]

    def test_from_mai_bbox_to_midpoint(self):
        # 4-element bbox should collapse to midpoint
        # ((100+200)/2, (300+400)/2) in MAI scale -> (150, 350) -> rescaled
        result = self._parsed_click_coordinate([100, 300, 200, 400])
        # midpoint in MAI: [150, 350] -> in cua-lite: [150*1000/999, 350*1000/999]
        assert result == [int(round(150 * 1000 / 999)), int(round(350 * 1000 / 999))]

    def test_round_trip_loses_at_most_one(self):
        for cua in [[0, 0], [250, 750], [500, 500], [1000, 1000]]:
            mai = self._rendered_tap_coordinate(cua)
            back = self._parsed_click_coordinate(mai)
            assert abs(back[0] - cua[0]) <= 1
            assert abs(back[1] - cua[1]) <= 1


# =============================================================================
# 4. Swipe direction synthesis (Option A)
# =============================================================================


class TestSwipeDirection:
    def test_finger_up(self):
        # finger goes from y=800 -> y=200 (dy = -600 < 0)
        assert _direction_from_endpoints([500, 800], [500, 200]) == "up"

    def test_finger_down(self):
        assert _direction_from_endpoints([500, 200], [500, 800]) == "down"

    def test_finger_left(self):
        assert _direction_from_endpoints([800, 500], [200, 500]) == "left"

    def test_finger_right(self):
        assert _direction_from_endpoints([200, 500], [800, 500]) == "right"

    def test_endpoints_from_up_direction(self):
        start, end = _endpoints_from_direction("up", [500, 500])
        assert start == [500, 500]
        # Finger up = end_y < start_y
        assert end[1] < start[1]

    def test_endpoints_from_down_direction(self):
        start, end = _endpoints_from_direction("down", [500, 500])
        assert end[1] > start[1]

    def test_endpoints_from_left_direction(self):
        start, end = _endpoints_from_direction("left", [500, 500])
        assert end[0] < start[0]

    def test_endpoints_from_right_direction(self):
        start, end = _endpoints_from_direction("right", [500, 500])
        assert end[0] > start[0]

    def test_endpoints_no_anchor_uses_center(self):
        start, end = _endpoints_from_direction("up", None)
        assert start == [500, 500]


# =============================================================================
# 5. Tool call conversion (forward: cua-lite -> MAI-UI)
# =============================================================================


class TestForwardConversion:
    def setup_method(self):
        self.space = MAIUIMobileActionSpace()

    def test_tap_to_click_with_rescale(self):
        # cua-lite [500, 300] -> MAI-UI [499, 299] (or [500, 300] depending on rounding)
        tc = [LiteMobileActionSpace.tap(coordinate=[500, 300])]
        result = self.space.convert_tool_calls_to_agent(tc)
        args = result[0]["arguments"]
        assert args["action"] == "click"
        # coordinates rescaled by 0.999
        assert args["coordinate"] == [int(round(500 * 999 / 1000)), int(round(300 * 999 / 1000))]

    def test_mobile_batch_to_agent_unwraps(self):
        tc = make_tool_call(
            "mobile",
            {
                "actions": [
                    {"action": "tap", "coordinate": [500, 300]},
                    {"action": "type", "text": "hello"},
                ]
            },
        )
        result = self.space.convert_tool_calls_to_agent([tc])
        assert [r["arguments"]["action"] for r in result] == ["click", "type"]
        assert result[1]["arguments"]["text"] == "hello"

    def test_long_press_to_agent(self):
        tc = [LiteMobileActionSpace.long_press(coordinate=[100, 200], duration=2.0)]
        result = self.space.convert_tool_calls_to_agent(tc)
        args = result[0]["arguments"]
        assert args["action"] == "long_press"
        assert args["time"] == 2.0

    def test_swipe_finger_up_to_direction_up(self):
        # finger goes from (500, 800) up to (500, 200) -> direction "up"
        tc = [
            LiteMobileActionSpace.swipe(
                start_coordinate=[500, 800],
                coordinate=[500, 200],
            )
        ]
        result = self.space.convert_tool_calls_to_agent(tc)
        args = result[0]["arguments"]
        assert args["action"] == "swipe"
        assert args["direction"] == "up"
        # Anchor = finger start
        assert args["coordinate"] == [int(round(500 * 999 / 1000)), int(round(800 * 999 / 1000))]

    def test_drag_to_agent_keeps_both_endpoints(self):
        """Canonical ``drag`` uses MAI-UI's endpoint-carrying ``drag``. Without a
        branch it falls through to the warn-and-drop tail and vanishes from the
        rendered trajectory. Unlike ``swipe``, nothing collapses to a direction.
        """
        tc = [
            LiteMobileActionSpace.drag(
                start_coordinate=[100, 200],
                coordinate=[700, 200],
            )
        ]
        result = self.space.convert_tool_calls_to_agent(tc)
        args = result[0]["arguments"]
        assert args["action"] == "drag"
        assert args["start_coordinate"] == [
            int(round(100 * 999 / 1000)),
            int(round(200 * 999 / 1000)),
        ]
        assert args["end_coordinate"] == [
            int(round(700 * 999 / 1000)),
            int(round(200 * 999 / 1000)),
        ]

    def test_swipe_finger_left_to_direction_left(self):
        tc = [
            LiteMobileActionSpace.swipe(
                start_coordinate=[800, 500],
                coordinate=[200, 500],
            )
        ]
        result = self.space.convert_tool_calls_to_agent(tc)
        assert result[0]["arguments"]["direction"] == "left"

    def test_open_app_to_agent_uses_text_field(self):
        tc = [make_tool_call("open_app", {"app_name": "Chrome"})]
        result = self.space.convert_tool_calls_to_agent(tc)
        args = result[0]["arguments"]
        assert args["action"] == "open"
        assert args["text"] == "Chrome"
        assert "app_name" not in args

    def test_system_button_capitalized_to_lowercase(self):
        for cua_btn, mai_btn in [("Home", "home"), ("Back", "back"), ("Enter", "enter")]:
            tc = [LiteMobileActionSpace.system_button(button=cua_btn)]
            result = self.space.convert_tool_calls_to_agent(tc)
            assert result[0]["arguments"]["button"] == mai_btn

    def test_every_mapped_system_button_round_trips(self):
        """The four MAI-UI buttons keep their canonical capitalization."""
        for cua_btn in _BUTTON_LITE_TO_MAI:
            tc = [LiteMobileActionSpace.system_button(button=cua_btn)]
            agent_tcs = self.space.convert_tool_calls_to_agent(tc)
            restored = self.space.convert_tool_calls_from_agent(agent_tcs)
            assert _only_mobile_action(restored) == {
                "action": "system_button",
                "button": cua_btn,
            }

    def test_recent_system_button_raises(self):
        """MAI-UI has no 'Recent' button (prompt options: back/home/menu/enter).

        It used to escape as lowercase 'recent', which the reverse map cannot
        decode and which is not a legal canonical button either.
        """
        tc = [LiteMobileActionSpace.system_button(button="Recent")]
        with pytest.raises(ValueError, match="no native system button"):
            self.space.convert_tool_calls_to_agent(tc)
        assert "Recent" not in _BUTTON_LITE_TO_MAI
        assert "recent" not in _BUTTON_MAI_TO_LITE

    def test_response_to_answer(self):
        tc = [make_tool_call("response", {"text": "42"})]
        result = self.space.convert_tool_calls_to_agent(tc)
        args = result[0]["arguments"]
        assert args["action"] == "answer"
        assert args["text"] == "42"

    def test_terminate_failure_to_fail(self):
        tc = [make_tool_call("terminate", {"status": "failure"})]
        result = self.space.convert_tool_calls_to_agent(tc)
        assert result[0]["arguments"]["status"] == "fail"

    def test_terminate_success_to_success(self):
        tc = [make_tool_call("terminate", {"status": "success"})]
        result = self.space.convert_tool_calls_to_agent(tc)
        assert result[0]["arguments"]["status"] == "success"


# =============================================================================
# 6. Tool call conversion (reverse: MAI-UI -> cua-lite)
# =============================================================================


class TestReverseConversion:
    def setup_method(self):
        self.space = MAIUIMobileActionSpace()

    def test_click_from_agent_with_rescale(self):
        tc = [
            {
                "name": "mobile_use",
                "arguments": {
                    "action": "click",
                    "coordinate": [499, 299],
                },
            }
        ]
        result = self.space.convert_tool_calls_from_agent(tc)
        action = _only_mobile_action(result)
        assert action["action"] == "tap"
        # 499 * 1000 / 999 = 500.0; 299 * 1000 / 999 = 299.299... -> 299
        assert action["coordinate"] == [
            int(round(499 * 1000 / 999)),
            int(round(299 * 1000 / 999)),
        ]

    def test_swipe_direction_up_synthesizes_endpoints(self):
        tc = [
            {
                "name": "mobile_use",
                "arguments": {
                    "action": "swipe",
                    "direction": "up",
                    "coordinate": [500, 500],
                },
            }
        ]
        result = self.space.convert_tool_calls_from_agent(tc)
        args = _only_mobile_action(result)
        assert args["action"] == "swipe"
        # finger goes up = end_y < start_y
        assert args["coordinate"][1] < args["start_coordinate"][1]

    def test_swipe_direction_no_anchor_uses_center(self):
        tc = [
            {
                "name": "mobile_use",
                "arguments": {
                    "action": "swipe",
                    "direction": "down",
                },
            }
        ]
        result = self.space.convert_tool_calls_from_agent(tc)
        args = _only_mobile_action(result)
        assert args["start_coordinate"] == [500, 500]
        assert args["coordinate"][1] > 500

    def test_drag_from_agent(self):
        tc = [
            {
                "name": "mobile_use",
                "arguments": {
                    "action": "drag",
                    "start_coordinate": [100, 200],
                    "end_coordinate": [700, 200],
                },
            }
        ]
        result = self.space.convert_tool_calls_from_agent(tc)
        args = _only_mobile_action(result)
        assert args["action"] == "drag"
        # Rescaled
        assert args["start_coordinate"] == [
            int(round(100 * 1000 / 999)),
            int(round(200 * 1000 / 999)),
        ]
        assert args["coordinate"] == [int(round(700 * 1000 / 999)), int(round(200 * 1000 / 999))]

    def test_open_text_to_open_app(self):
        tc = [{"name": "mobile_use", "arguments": {"action": "open", "text": "Chrome"}}]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert tool_call_name(result[0]) == "open_app"
        assert tool_call_arguments(result[0])["app_name"] == "Chrome"

    def test_answer_to_response(self):
        tc = [{"name": "mobile_use", "arguments": {"action": "answer", "text": "42"}}]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert tool_call_name(result[0]) == "response"
        assert tool_call_arguments(result[0])["text"] == "42"

    def test_top_level_answer_to_response(self):
        tc = [{"name": "answer", "arguments": {"text": "42"}}]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert result == [make_tool_call("response", {"text": "42"})]

    def test_top_level_native_click_converts_like_the_wrapped_call(self):
        """Flat output — the ``mobile_use`` wrapper dropped, the native action
        value used as the tool name — runs the same dispatch branch. Dropping it
        wasted the turn with no model-visible feedback."""
        tc = [{"name": "click", "arguments": {"coordinate": [499, 299]}}]
        assert self.space.convert_tool_calls_from_agent(tc) == (
            self.space.convert_tool_calls_from_agent(
                [{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [499, 299]}}]
            )
        )
        assert _only_mobile_action(self.space.convert_tool_calls_from_agent(tc))["action"] == "tap"

    def test_active_extra_with_action_argument_passes_through(self):
        schema = make_tool_schema(
            "report_problem",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["action", "reason"],
                "additionalProperties": False,
            },
        )
        args = {"action": "click", "reason": "ambiguous target"}
        tc = [{"name": "report_problem", "arguments": args}]

        result = self.space.convert_tool_calls_from_agent(
            tc,
            active_extra_tool_names={"report_problem"},
            active_extra_tool_schemas=[schema],
        )

        assert tool_call_name(result[0]) == "report_problem"
        assert tool_call_arguments(result[0]) == args

    def test_active_extra_requires_full_schema_satisfaction(self):
        schema = make_tool_schema(
            "report_problem",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["report"]},
                    "coordinate": {"type": "array"},
                    "reason": {"type": "string"},
                },
                "required": ["action", "coordinate", "reason"],
                "additionalProperties": False,
            },
        )
        args = {
            "action": "click",
            "coordinate": [499, 299],
            "reason": "ambiguous target",
        }
        tc = [{"name": "report_problem", "arguments": args}]

        result = self.space.convert_tool_calls_from_agent(
            tc,
            active_extra_tool_names={"report_problem"},
            active_extra_tool_schemas=[schema],
        )

        action = _only_mobile_action(result)
        assert action["action"] == "tap"
        assert action["coordinate"] == [499, 299]

    def test_action_shaped_non_wrapper_falls_back_to_action_switch(self):
        tc = [
            {
                "name": "not_mobile_use",
                "arguments": {
                    "action": "click",
                    "coordinate": [499, 299],
                },
            }
        ]

        action = _only_mobile_action(self.space.convert_tool_calls_from_agent(tc))
        assert action["action"] == "tap"

    def test_unsupported_top_level_name_stays_a_standalone_call(self):
        """A name that satisfies no active extra schema and spells no native
        action value keeps its wire name, exactly as in every sibling family.
        Returning ``[]`` deleted it before any call id reached the model, so the
        turn read downstream as a no-tool-call parse-failure final."""
        tc = [{"name": "search_web", "arguments": {"query": "weather"}}]
        assert self.space.convert_tool_calls_from_agent(tc) == [
            make_tool_call("search_web", {"query": "weather"})
        ]

    def test_unknown_wrapped_action_becomes_invalid_action_batch(self):
        """The unknown-action policy: the model's own action value becomes the
        batch child name, so env ingress rejects it and the model sees the
        rejection instead of the action vanishing."""
        tc = [{"name": "mobile_use", "arguments": {"action": "UNSUPPORTED", "value": "x"}}]
        out = self.space.convert_tool_calls_from_agent(tc)

        assert len(out) == 1
        assert tool_call_name(out[0]) == "mobile"
        assert tool_call_arguments(out[0]) == {
            "actions": [{"action": "UNSUPPORTED", "value": "x"}],
        }
        children, error = validate_lite_action_batch_structure(
            "mobile",
            tool_call_arguments(out[0]),
        )
        assert len(children) == 1
        error = lite_action_batch_child_name_errors("mobile", children).get(0)
        assert error is not None
        assert error.child_action_name == "UNSUPPORTED"

    def test_system_button_lowercase_to_capitalized(self):
        for mai_btn, lite_btn in [("home", "Home"), ("back", "Back"), ("enter", "Enter")]:
            tc = [
                {
                    "name": "mobile_use",
                    "arguments": {
                        "action": "system_button",
                        "button": mai_btn,
                    },
                }
            ]
            result = self.space.convert_tool_calls_from_agent(tc)
            assert _only_mobile_action(result)["button"] == lite_btn

    def test_system_button_menu_passes_through(self):
        """MAI-UI lowercase 'menu' should normalize to capitalized 'Menu'."""
        tc = [
            {
                "name": "mobile_use",
                "arguments": {
                    "action": "system_button",
                    "button": "menu",
                },
            }
        ]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert _only_mobile_action(result) == {"action": "system_button", "button": "Menu"}

    def test_terminate_fail_to_failure(self):
        tc = [
            {
                "name": "mobile_use",
                "arguments": {
                    "action": "terminate",
                    "status": "fail",
                },
            }
        ]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert tool_call_arguments(result[0])["status"] == "failure"


# =============================================================================
# 7. filter_tool_schemas_for_valid_actions
# =============================================================================


class TestFilterToolSchemasForValidActions:
    def test_empty_valid_actions_keeps_only_extra_tool_action_values(self):
        """GUI-only filter: an empty ``valid_actions`` empties the GUI half and
        leaves the extra-tool action values, which it does not own."""
        schemas = MAIUIMobileActionSpace.get_tool_schemas()
        filtered = MAIUIMobileActionSpace.filter_tool_schemas_for_valid_actions(schemas, [])
        assert set(tool_schema_parameters(filtered[0])["properties"]["action"]["enum"]) == (
            frozenset(MAIUIMobileActionSpace.MAI_ACTION_VALUE_TO_EXTRA_TOOL_NAMES)
        )

    def test_gui_action_preserves_extra_tool_action_values(self):
        schemas = MAIUIMobileActionSpace.get_tool_schemas()
        filtered = MAIUIMobileActionSpace.filter_tool_schemas_for_valid_actions(schemas, ["tap"])
        enum = set(tool_schema_parameters(filtered[0])["properties"]["action"]["enum"])
        assert enum == {"click", "open", "answer", "terminate"}

    def test_finish_names_do_not_open_valid_actions_filter(self):
        """A canonical finish name in ``valid_actions`` opens NO GUI entry (and
        does not conjure its native spelling either — that needs an extra)."""
        schemas = MAIUIMobileActionSpace.get_tool_schemas()
        filtered = MAIUIMobileActionSpace.filter_tool_schemas_for_valid_actions(
            schemas, ["response"]
        )
        assert set(tool_schema_parameters(filtered[0])["properties"]["action"]["enum"]) == (
            frozenset(MAIUIMobileActionSpace.MAI_ACTION_VALUE_TO_EXTRA_TOOL_NAMES)
        )


# =============================================================================
# 8. Round-trip
# =============================================================================


class TestRoundTrip:
    @pytest.mark.parametrize(
        "cua_tc",
        [
            LiteMobileActionSpace.tap(coordinate=[100, 100]),
            LiteMobileActionSpace.tap(coordinate=[1000, 1000]),
            LiteMobileActionSpace.long_press(coordinate=[500, 500], duration=2.0),
            LiteMobileActionSpace.type(text="hello world"),
            make_tool_call("open_app", {"app_name": "Settings"}),
            LiteMobileActionSpace.system_button(button="Home"),
            LiteMobileActionSpace.system_button(button="Back"),
            LiteMobileActionSpace.system_button(button="Enter"),
            make_tool_call("response", {"text": "answer"}),
            LiteMobileActionSpace.wait(duration=1.5),
            make_tool_call("terminate", {"status": "success"}),
            make_tool_call("terminate", {"status": "failure"}),
        ],
    )
    def test_round_trip_no_lossy_fields(self, cua_tc):
        """Round-trip preserves the action name and key non-coordinate args."""
        space = MAIUIMobileActionSpace()
        agent_tcs = space.convert_tool_calls_to_agent([cua_tc])
        restored = space.convert_tool_calls_from_agent(agent_tcs)
        assert len(restored) == 1
        assert tool_call_name(restored[0]) == tool_call_name(cua_tc)
        if tool_call_name(cua_tc) == "mobile":
            orig_action = tool_call_arguments(cua_tc)["actions"][0]
            new_action = tool_call_arguments(restored[0])["actions"][0]
            assert new_action["action"] == orig_action["action"]
            for k, v in orig_action.items():
                if k == "action":
                    continue
                if k in {"coordinate", "start_coordinate"}:
                    for o, n in zip(v, new_action[k]):
                        assert abs(o - n) <= 1
                else:
                    assert new_action.get(k) == v, f"field {k} differs"
            return
        # Coordinates may drift by ±1 due to 999<->1000 rescale; compare other fields exactly.
        orig_args = tool_call_arguments(cua_tc)
        new_args = tool_call_arguments(restored[0])
        for k in orig_args:
            if k == "coordinate":
                # Allow ±1 drift per axis
                for o, n in zip(orig_args[k], new_args[k]):
                    assert abs(o - n) <= 1
            else:
                assert new_args.get(k) == orig_args[k], f"field {k} differs"

    def test_swipe_round_trip_lossy(self):
        """Swipe is intentionally lossy (direction-only forward, synthesized endpoints reverse).

        We only verify that direction is preserved, not exact coordinates.
        """
        space = MAIUIMobileActionSpace()
        # finger goes up
        cua_tc = LiteMobileActionSpace.swipe(
            start_coordinate=[500, 800],
            coordinate=[500, 200],
        )
        agent_tcs = space.convert_tool_calls_to_agent([cua_tc])
        assert agent_tcs[0]["arguments"]["direction"] == "up"
        restored = space.convert_tool_calls_from_agent(agent_tcs)
        # The restored swipe should still go up (start_y > end_y)
        new_args = tool_call_arguments(restored[0])["actions"][0]
        assert new_args["start_coordinate"][1] > new_args["coordinate"][1]


# =============================================================================
# 9. Malformed-but-recoverable provider output
# =============================================================================


class TestMalformedProviderOutput:
    def setup_method(self):
        self.space = MAIUIMobileActionSpace()

    @pytest.mark.parametrize(
        "name,args",
        [
            ("click", {"coordinate": [500, 300]}),
            ("long_press", {"coordinate": [500, 300], "time": 2}),
            ("type", {"text": "hello"}),
            ("wait", {"time": 2}),
        ],
    )
    def test_flat_native_action_value_used_as_tool_name_converts(self, name, args):
        """MAI-UI exposes its whole surface as ONE ``mobile_use`` wrapper. When
        the model drops the wrapper and uses the action VALUE as the tool name
        the call used to vanish entirely; it must run the same dispatch branch
        as nested output."""
        flat = self.space.convert_tool_calls_from_agent([{"name": name, "arguments": args}])
        nested = self.space.convert_tool_calls_from_agent(
            [{"name": "mobile_use", "arguments": {"action": name, **args}}]
        )
        assert flat == nested
        assert flat != []

    def test_wrong_wrapper_name_with_nested_action_converts(self):
        out = self.space.convert_tool_calls_from_agent(
            [{"name": "mobile", "arguments": {"action": "click", "coordinate": [499, 299]}}]
        )
        assert _only_mobile_action(out)["action"] == "tap"

    def test_flat_answer_becomes_response(self):
        out = self.space.convert_tool_calls_from_agent(
            [{"name": "answer", "arguments": {"text": "42"}}]
        )
        assert out == [make_tool_call("response", {"text": "42"})]

    def test_flat_terminate_becomes_canonical_terminate(self):
        """The old ``name == "answer" and "text" in args`` special case claimed
        ``answer`` but left the sibling finish channel dropped; both now route
        through the family's own dispatch."""
        out = self.space.convert_tool_calls_from_agent(
            [{"name": "terminate", "arguments": {"status": "fail"}}]
        )
        assert out == [make_tool_call("terminate", {"status": "failure"})]

    def test_active_extra_tool_outranks_a_colliding_native_action_value(self):
        """A browsergym ``type(bid=...)`` is the env's tool, not the native
        ``type`` the dispatch would read as a keyboard action."""
        schema = make_tool_schema(
            "type",
            description="Type into an element by bid.",
            parameters={
                "type": "object",
                "properties": {"bid": {"type": "string"}},
                "required": ["bid"],
            },
        )
        out = self.space.convert_tool_calls_from_agent(
            [{"name": "type", "arguments": {"bid": "a51"}}],
            active_extra_tool_names={"type"},
            active_extra_tool_schemas=[schema],
        )
        assert out == [make_tool_call("type", {"bid": "a51"})]
