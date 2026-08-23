"""
Tests for Qwen3VL action spaces (desktop + mobile).

Covers, for both ``Qwen3VLDesktopActionSpace`` (``computer_use``) and
``Qwen3VLMobileActionSpace`` (``mobile_use``):

  1. Registry and available actions
  2. Tool call creation
  3. Tool call conversion: cua-lite ↔ Qwen3VL (round-trip)
  4. Single tool schema

Run:
    uv run pytest tests/agents/models/qwen3_vl/test_qwen3_vl_action_space.py -v
"""

from __future__ import annotations

import pytest

from lite.agents.core.action_space.base import (
    ActionSpaceRegistry,
    LiteDesktopActionSpace,
    LiteMobileActionSpace,
)
from lite.agents.models.qwen3_5.action_space import (
    Qwen3_5DesktopActionSpace,
    Qwen3_5MobileActionSpace,
)
from lite.agents.models.qwen3_vl.action_space import (
    Qwen3VLDesktopActionSpace,
    Qwen3VLMobileActionSpace,
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


def _single_action(tool_calls, batch_name):
    assert len(tool_calls) == 1
    call = tool_calls[0]
    assert tool_call_name(call) == batch_name
    actions = tool_call_arguments(call)["actions"]
    assert len(actions) == 1
    return actions[0]


def _canonical_args(tool_call, name):
    assert validate_lite_tool_call(tool_call, "tool_call", require_id=False) is None
    assert tool_call_name(tool_call) == name
    return tool_call_arguments(tool_call)


@pytest.mark.parametrize(
    "space_cls,wrapper",
    [
        (Qwen3VLDesktopActionSpace, "computer_use"),
        (Qwen3VLMobileActionSpace, "mobile_use"),
        (Qwen3_5DesktopActionSpace, "computer_use"),
        (Qwen3_5MobileActionSpace, "mobile_use"),
    ],
)
def test_active_top_level_finish_names_preserve_env_extra_calls(space_cls, wrapper):
    del wrapper
    space = space_cls()
    answer_schema = make_tool_schema(
        "answer",
        description="Env-owned answer tool.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "format": {"type": "string"},
            },
            "required": ["text", "format"],
            "additionalProperties": False,
        },
    )
    terminate_schema = make_tool_schema(
        "terminate",
        description="Env-owned terminate tool.",
        parameters={
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
            "additionalProperties": False,
        },
    )
    calls = [
        {"name": "answer", "arguments": {"text": "42", "format": "short"}},
        {"name": "terminate", "arguments": {"reason": "blocked"}},
    ]

    assert space.convert_tool_calls_from_agent(
        calls,
        active_extra_tool_names={"answer", "terminate"},
        active_extra_tool_schemas=[answer_schema, terminate_schema],
    ) == [
        make_tool_call("answer", {"text": "42", "format": "short"}),
        make_tool_call("terminate", {"reason": "blocked"}),
    ]


@pytest.mark.parametrize(
    "space_cls,wrapper",
    [
        (Qwen3VLDesktopActionSpace, "computer_use"),
        (Qwen3VLMobileActionSpace, "mobile_use"),
        (Qwen3_5DesktopActionSpace, "computer_use"),
        (Qwen3_5MobileActionSpace, "mobile_use"),
    ],
)
def test_inactive_wrapper_answer_still_maps_to_response(space_cls, wrapper):
    space = space_cls()
    result = space.convert_tool_calls_from_agent(
        [{"name": wrapper, "arguments": {"action": "answer", "text": "42"}}]
    )

    assert result == [make_tool_call("response", {"text": "42"})]


# =============================================================================
# Desktop
# =============================================================================


class TestDesktopRegistryAndActions:
    def test_registry_key(self):
        space = ActionSpaceRegistry.get("qwen3_vl@desktop")
        assert isinstance(space, Qwen3VLDesktopActionSpace)

    def test_single_tool(self):
        schemas = Qwen3VLDesktopActionSpace.get_tool_schemas()
        assert len(schemas) == 1
        assert tool_schema_name(schemas[0]) == "computer_use"

    def test_action_enum(self):
        schema = Qwen3VLDesktopActionSpace.get_tool_schemas()[0]
        actions = set(tool_schema_parameters(schema)["properties"]["action"]["enum"])
        expected = {
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
            "answer",
            "terminate",
        }
        assert actions == expected


# =============================================================================
# 2. Tool call creation
# =============================================================================


class TestDesktopToolCalls:
    def test_left_click(self):
        tc = Qwen3VLDesktopActionSpace.computer_use(action="left_click", coordinate=[500, 300])
        args = _canonical_args(tc, "computer_use")
        assert args["action"] == "left_click"
        assert args["coordinate"] == [500, 300]

    def test_type(self):
        tc = Qwen3VLDesktopActionSpace.computer_use(action="type", text="hello")
        assert _canonical_args(tc, "computer_use")["text"] == "hello"

    def test_terminate(self):
        tc = Qwen3VLDesktopActionSpace.computer_use(action="terminate", status="success")
        assert _canonical_args(tc, "computer_use")["status"] == "success"

    def test_none_args_omitted(self):
        tc = Qwen3VLDesktopActionSpace.computer_use(action="left_click")
        assert "coordinate" not in _canonical_args(tc, "computer_use")


# =============================================================================
# 3. Tool call conversion
# =============================================================================


class TestDesktopToolCallConversion:
    def setup_method(self):
        self.space = Qwen3VLDesktopActionSpace()

    def test_click_to_left_click(self):
        tc = [LiteDesktopActionSpace.click(coordinate=[500, 300])]
        result = self.space.convert_tool_calls_to_agent(tc)
        assert result[0]["arguments"]["action"] == "left_click"
        assert result[0]["arguments"]["coordinate"] == [500, 300]

    def test_right_click(self):
        tc = [LiteDesktopActionSpace.click(coordinate=[100, 200], button="right")]
        result = self.space.convert_tool_calls_to_agent(tc)
        assert result[0]["arguments"]["action"] == "right_click"

    def test_double_click(self):
        tc = [LiteDesktopActionSpace.click(coordinate=[100, 200], clicks=2)]
        result = self.space.convert_tool_calls_to_agent(tc)
        assert result[0]["arguments"]["action"] == "double_click"

    def test_left_click_from_agent(self):
        tc = [
            {
                "name": "computer_use",
                "arguments": {
                    "action": "left_click",
                    "coordinate": [500, 300],
                },
            }
        ]
        result = self.space.convert_tool_calls_from_agent(tc)
        action = _single_action(result, "computer")
        assert action["action"] == "click"
        assert action["coordinate"] == [500, 300]

    @pytest.mark.parametrize("space_cls", [Qwen3VLDesktopActionSpace, Qwen3_5DesktopActionSpace])
    @pytest.mark.parametrize(
        "raw_keys,expected",
        [
            ("ctrl++", ["ctrl", "+"]),
            ("ctrl+-", ["ctrl", "-"]),
            ("ctrl+=", ["ctrl", "="]),
        ],
    )
    def test_key_string_chord_uses_core_normalizer(self, space_cls, raw_keys, expected):
        result = space_cls().convert_tool_calls_from_agent(
            [{"name": "computer_use", "arguments": {"action": "key", "keys": raw_keys}}]
        )
        assert _single_action(result, "computer") == {"action": "key", "keys": expected}

    @pytest.mark.parametrize("space_cls", [Qwen3VLDesktopActionSpace, Qwen3_5DesktopActionSpace])
    @pytest.mark.parametrize("raw_keys", ["ctrl left", "ctrl -"])
    def test_key_string_rejects_phrase_like_tokens(self, space_cls, raw_keys):
        with pytest.raises(ValueError, match="unknown key token"):
            space_cls().convert_tool_calls_from_agent(
                [{"name": "computer_use", "arguments": {"action": "key", "keys": raw_keys}}]
            )

    def test_terminate_from_agent(self):
        tc = [
            {
                "name": "computer_use",
                "arguments": {
                    "action": "terminate",
                    "status": "failure",
                },
            }
        ]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert tool_call_name(result[0]) == "terminate"
        assert tool_call_arguments(result[0])["status"] == "failure"

    @pytest.mark.parametrize(
        "name,args",
        [
            ("left_click", {"coordinate": [500, 300]}),
            ("right_click", {"coordinate": [500, 300]}),
            ("middle_click", {"coordinate": [500, 300]}),
            ("double_click", {"coordinate": [500, 300]}),
            ("triple_click", {"coordinate": [500, 300]}),
            ("hscroll", {"pixels": 500}),
            ("left_click_drag", {"coordinate": [500, 300]}),
        ],
    )
    def test_top_level_qwen_native_action_value_used_as_tool_name_converts(self, name, args):
        """Flat output converts exactly like nested output.

        Qwen sometimes drops the ``computer_use`` wrapper and puts the native
        action value in the tool NAME. The name is native vocabulary
        (``left_click``, ``hscroll``), so it is read as ``action`` and runs the
        same dispatch branch — the alternative is handing the env a tool it does
        not have and wasting the turn.
        """
        flat = self.space.convert_tool_calls_from_agent([{"name": name, "arguments": args}])
        nested = self.space.convert_tool_calls_from_agent(
            [{"name": "computer_use", "arguments": {"action": name, **args}}]
        )
        assert flat == nested
        assert tool_call_name(flat[0]) == "computer"

    def test_wrong_wrapper_name_with_nested_action_converts(self):
        """The other malformed-but-recoverable shape: nesting kept, name wrong."""
        result = self.space.convert_tool_calls_from_agent(
            [{"name": "computer", "arguments": {"action": "left_click", "coordinate": [5, 6]}}]
        )
        assert _single_action(result, "computer") == {"action": "click", "coordinate": [5, 6]}

    def test_top_level_non_native_name_stays_a_standalone_tool(self):
        """The recovery reaches only NATIVE action values. A name outside that
        vocabulary is a real standalone tool (env extra, custom tool) and its
        Lite name IS its wire name."""
        tc = [{"name": "summarize", "arguments": {"text": "hi"}}]
        assert self.space.convert_tool_calls_from_agent(tc) == [
            make_tool_call("summarize", {"text": "hi"})
        ]

    def test_active_extra_tool_outranks_a_colliding_native_action_value(self):
        """A browsergym ``type(bid=...)`` is the env's tool, not the native
        ``type`` the dispatch would read as a keyboard action."""
        schema = {
            "type": "function",
            "function": {
                "name": "type",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {"bid": {"type": "string"}},
                    "required": ["bid"],
                },
            },
        }
        result = self.space.convert_tool_calls_from_agent(
            [{"name": "type", "arguments": {"bid": "a51"}}],
            active_extra_tool_names={"type"},
            active_extra_tool_schemas=[schema],
        )
        assert result == [make_tool_call("type", {"bid": "a51"})]

    def test_unknown_wrapped_action_becomes_invalid_feedback_batch(self):
        result = self.space.convert_tool_calls_from_agent(
            [
                {
                    "name": "computer_use",
                    "arguments": {"action": "screen_record", "path": "/tmp/out.mp4"},
                }
            ]
        )

        assert len(result) == 1
        assert tool_call_name(result[0]) == "computer"
        assert tool_call_arguments(result[0]) == {
            "actions": [{"action": "screen_record", "path": "/tmp/out.mp4"}],
        }
        children, error = validate_lite_action_batch_structure(
            "computer",
            tool_call_arguments(result[0]),
        )
        assert len(children) == 1
        error = lite_action_batch_child_name_errors("computer", children).get(0)
        assert error is not None
        assert error.child_action_name == "screen_record"

    def test_non_native_standalone_tool_names_pass_through_for_env_feedback(self):
        tc = [
            {"name": "click", "arguments": {"coordinate": [500, 300]}},
            {"name": "summarize", "arguments": {"text": "hello"}},
        ]
        result = self.space.convert_tool_calls_from_agent(
            tc,
            active_extra_tool_names={"click", "summarize"},
        )
        assert result == [
            make_tool_call("click", {"coordinate": [500, 300]}),
            make_tool_call("summarize", {"text": "hello"}),
        ]

    def test_adjacent_native_gui_calls_batch_without_crossing_standalone_extra(self):
        tc = [
            {
                "name": "computer_use",
                "arguments": {
                    "action": "left_click",
                    "coordinate": [100, 200],
                },
            },
            {"name": "computer_use", "arguments": {"action": "type", "text": "a"}},
            {"name": "bash", "arguments": {"command": "pwd"}},
            {
                "name": "computer_use",
                "arguments": {
                    "action": "left_click",
                    "coordinate": [300, 400],
                },
            },
            {"name": "goto", "arguments": {"url": "https://example.com"}},
            {
                "name": "computer_use",
                "arguments": {
                    "action": "terminate",
                    "status": "success",
                },
            },
        ]

        result = self.space.convert_tool_calls_from_agent(tc)

        assert result == [
            make_tool_call(
                "computer",
                {
                    "actions": [
                        {"action": "click", "coordinate": [100, 200]},
                        {"action": "type", "text": "a"},
                    ],
                },
            ),
            make_tool_call("bash", {"command": "pwd"}),
            make_tool_call(
                "computer",
                {
                    "actions": [
                        {"action": "click", "coordinate": [300, 400]},
                    ],
                },
            ),
            make_tool_call("goto", {"url": "https://example.com"}),
            make_tool_call("terminate", {"status": "success"}),
        ]
        assert self.space.convert_tool_calls_to_agent(result) == [
            {
                "name": "computer_use",
                "arguments": {
                    "action": "left_click",
                    "coordinate": [100, 200],
                },
            },
            {"name": "computer_use", "arguments": {"action": "type", "text": "a"}},
            {"name": "bash", "arguments": {"command": "pwd"}},
            {
                "name": "computer_use",
                "arguments": {
                    "action": "left_click",
                    "coordinate": [300, 400],
                },
            },
            {"name": "goto", "arguments": {"url": "https://example.com"}},
            {
                "name": "computer_use",
                "arguments": {
                    "action": "terminate",
                    "status": "success",
                },
            },
        ]

    def test_answer_from_agent(self):
        tc = [{"name": "computer_use", "arguments": {"action": "answer", "text": "42"}}]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert tool_call_name(result[0]) == "response"
        assert tool_call_arguments(result[0])["text"] == "42"

    @pytest.mark.parametrize(
        "direction,exp_action,exp_pixels",
        [
            ("down", "scroll", -500),
            ("up", "scroll", 500),
            ("left", "hscroll", -500),  # horizontal -> hscroll (was wrongly a vertical scroll)
            ("right", "hscroll", 500),
        ],
    )
    def test_scroll_native_style(self, direction, exp_action, exp_pixels):
        """lite scroll -> qwen3_vl must be NATIVE style: NO coordinate, int
        ``pixels``, and horizontal (left/right) -> ``hscroll`` with the right
        sign. Regression for the GPT->lite->qwen3_vl export that carried a
        ``coordinate`` (element-anchored scroll) and a float ``pixels``, and
        rendered horizontal scrolls as vertical ones.
        """
        # float amount + a coordinate, exactly as GPT-origin element-scrolls carry
        tc = [LiteDesktopActionSpace.scroll(direction=direction, amount=5.0, coordinate=[7, 9])]
        args = self.space.convert_tool_calls_to_agent(tc)[0]["arguments"]
        assert args["action"] == exp_action
        assert args["pixels"] == exp_pixels
        assert isinstance(args["pixels"], int)  # not float
        assert "coordinate" not in args  # native scroll has no coordinate
        # and direction + amount survive the round-trip
        back = self.space.convert_tool_calls_from_agent(self.space.convert_tool_calls_to_agent(tc))
        action = _single_action(back, "computer")
        assert action["direction"] == direction
        assert action["amount"] == 5

    @pytest.mark.parametrize(
        "cua_tc",
        [
            LiteDesktopActionSpace.click(coordinate=[500, 300]),
            LiteDesktopActionSpace.click(coordinate=[100, 200], button="right"),
            LiteDesktopActionSpace.click(coordinate=[100, 200], clicks=2),
            LiteDesktopActionSpace.type(text="hello"),
            LiteDesktopActionSpace.key(keys=["ctrl", "v"]),
            LiteDesktopActionSpace.key(keys=["ctrl", "+"]),
            LiteDesktopActionSpace.wait(duration=1.5),
            make_tool_call("terminate", {"status": "success"}),
            make_tool_call("response", {"text": "answer"}),
        ],
    )
    def test_round_trip(self, cua_tc):
        space = Qwen3VLDesktopActionSpace()
        agent_tcs = space.convert_tool_calls_to_agent([cua_tc])
        restored = space.convert_tool_calls_from_agent(agent_tcs)
        assert len(restored) == 1
        if tool_call_name(cua_tc) in {"click", "key", "type", "wait"}:
            assert _single_action(restored, "computer") == {
                "action": tool_call_name(cua_tc),
                **tool_call_arguments(cua_tc),
            }
        else:
            assert tool_call_name(restored[0]) == tool_call_name(cua_tc)
            assert tool_call_arguments(restored[0]) == tool_call_arguments(cua_tc)


# =============================================================================
# Mobile
# =============================================================================


class TestMobileRegistryAndActions:
    def test_registry_key(self):
        space = ActionSpaceRegistry.get("qwen3_vl@mobile")
        assert isinstance(space, Qwen3VLMobileActionSpace)

    def test_single_tool(self):
        schemas = Qwen3VLMobileActionSpace.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "mobile_use"

    def test_action_enum(self):
        """Qwen3-VL mobile action enum includes native app launch as ``open``."""
        schema = Qwen3VLMobileActionSpace.get_tool_schemas()[0]
        actions = set(tool_schema_parameters(schema)["properties"]["action"]["enum"])
        expected = {
            "click",
            "long_press",
            "swipe",
            "type",
            "open",
            "answer",
            "system_button",
            "wait",
            "terminate",
        }
        assert actions == expected
        assert "key" not in actions

    def test_screen_resolution_999(self):
        """Mobile cookbook declares 999×999 in the tool description."""
        schema = Qwen3VLMobileActionSpace.get_tool_schemas()[0]
        assert "999x999" in schema["function"]["description"]


class TestMobileToolCalls:
    def test_click(self):
        tc = Qwen3VLMobileActionSpace.mobile_use(action="click", coordinate=[500, 300])
        args = _canonical_args(tc, "mobile_use")
        assert args["action"] == "click"
        assert args["coordinate"] == [500, 300]

    def test_swipe(self):
        tc = Qwen3VLMobileActionSpace.mobile_use(
            action="swipe",
            coordinate=[100, 200],
            coordinate2=[100, 800],
        )
        args = _canonical_args(tc, "mobile_use")
        assert args["coordinate"] == [100, 200]
        assert args["coordinate2"] == [100, 800]

    def test_type(self):
        tc = Qwen3VLMobileActionSpace.mobile_use(action="type", text="hello")
        assert _canonical_args(tc, "mobile_use")["text"] == "hello"

    def test_system_button(self):
        tc = Qwen3VLMobileActionSpace.mobile_use(action="system_button", button="Back")
        assert _canonical_args(tc, "mobile_use")["button"] == "Back"

    def test_terminate(self):
        tc = Qwen3VLMobileActionSpace.mobile_use(action="terminate", status="success")
        assert _canonical_args(tc, "mobile_use")["status"] == "success"

    def test_none_args_omitted(self):
        tc = Qwen3VLMobileActionSpace.mobile_use(action="click")
        args = _canonical_args(tc, "mobile_use")
        assert "coordinate" not in args
        assert "text" not in args

    def test_long_press(self):
        tc = Qwen3VLMobileActionSpace.mobile_use(
            action="long_press",
            coordinate=[500, 500],
            time=2.0,
        )
        assert _canonical_args(tc, "mobile_use")["time"] == 2.0


class TestMobileToolCallConversion:
    def setup_method(self):
        self.space = Qwen3VLMobileActionSpace()

    # -- to agent --

    def test_tap_to_click(self):
        tc = [LiteMobileActionSpace.tap(coordinate=[500, 300])]
        result = self.space.convert_tool_calls_to_agent(tc)
        assert result[0]["arguments"]["action"] == "click"
        assert result[0]["arguments"]["coordinate"] == [500, 300]

    def test_long_press_to_agent(self):
        tc = [LiteMobileActionSpace.long_press(coordinate=[100, 200], duration=2.0)]
        result = self.space.convert_tool_calls_to_agent(tc)
        assert result[0]["arguments"]["action"] == "long_press"
        assert result[0]["arguments"]["time"] == 2.0

    def test_swipe_to_agent(self):
        tc = [
            LiteMobileActionSpace.swipe(
                start_coordinate=[100, 200],
                coordinate=[100, 800],
            )
        ]
        result = self.space.convert_tool_calls_to_agent(tc)
        args = result[0]["arguments"]
        assert args["action"] == "swipe"
        assert args["coordinate"] == [100, 200]
        assert args["coordinate2"] == [100, 800]

    def test_system_button_home_to_agent(self):
        tc = [LiteMobileActionSpace.system_button(button="Home")]
        result = self.space.convert_tool_calls_to_agent(tc)
        assert result[0]["arguments"]["action"] == "system_button"
        assert result[0]["arguments"]["button"] == "Home"

    def test_system_button_back_to_agent(self):
        tc = [LiteMobileActionSpace.system_button(button="Back")]
        result = self.space.convert_tool_calls_to_agent(tc)
        assert result[0]["arguments"]["action"] == "system_button"
        assert result[0]["arguments"]["button"] == "Back"

    def test_system_button_enter_to_agent(self):
        tc = [LiteMobileActionSpace.system_button(button="Enter")]
        result = self.space.convert_tool_calls_to_agent(tc)
        assert result[0]["arguments"]["action"] == "system_button"
        assert result[0]["arguments"]["button"] == "Enter"

    def test_open_app_to_agent_uses_native_open(self):
        """Canonical ``open_app`` renders as Qwen's native mobile ``open``."""
        tc = [make_tool_call("open_app", {"app_name": "Chrome"})]
        result = self.space.convert_tool_calls_to_agent(tc)
        assert result == [{"name": "mobile_use", "arguments": {"action": "open", "text": "Chrome"}}]

    def test_response_to_answer(self):
        tc = [make_tool_call("response", {"text": "42"})]
        result = self.space.convert_tool_calls_to_agent(tc)
        assert result[0]["arguments"]["action"] == "answer"
        assert result[0]["arguments"]["text"] == "42"

    # -- from agent --

    def test_click_from_agent(self):
        tc = [
            {
                "name": "mobile_use",
                "arguments": {
                    "action": "click",
                    "coordinate": [500, 300],
                },
            }
        ]
        result = self.space.convert_tool_calls_from_agent(tc)
        action = _single_action(result, "mobile")
        assert action["action"] == "tap"
        assert action["coordinate"] == [500, 300]

    def test_swipe_from_agent(self):
        tc = [
            {
                "name": "mobile_use",
                "arguments": {
                    "action": "swipe",
                    "coordinate": [100, 200],
                    "coordinate2": [100, 800],
                },
            }
        ]
        result = self.space.convert_tool_calls_from_agent(tc)
        action = _single_action(result, "mobile")
        assert action["action"] == "swipe"
        assert action["start_coordinate"] == [100, 200]
        assert action["coordinate"] == [100, 800]

    def test_system_button_back_from_agent(self):
        tc = [
            {
                "name": "mobile_use",
                "arguments": {
                    "action": "system_button",
                    "button": "Back",
                },
            }
        ]
        result = self.space.convert_tool_calls_from_agent(tc)
        action = _single_action(result, "mobile")
        assert action["action"] == "system_button"
        assert action["button"] == "Back"

    def test_system_button_home_from_agent(self):
        tc = [
            {
                "name": "mobile_use",
                "arguments": {
                    "action": "system_button",
                    "button": "Home",
                },
            }
        ]
        result = self.space.convert_tool_calls_from_agent(tc)
        action = _single_action(result, "mobile")
        assert action["action"] == "system_button"
        assert action["button"] == "Home"

    def test_answer_from_agent(self):
        tc = [{"name": "mobile_use", "arguments": {"action": "answer", "text": "42"}}]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert tool_call_name(result[0]) == "response"
        assert tool_call_arguments(result[0])["text"] == "42"

    def test_unknown_wrapped_action_becomes_invalid_feedback_batch(self):
        result = self.space.convert_tool_calls_from_agent(
            [
                {
                    "name": "mobile_use",
                    "arguments": {"action": "UNSUPPORTED", "value": "x"},
                }
            ]
        )

        assert len(result) == 1
        assert tool_call_name(result[0]) == "mobile"
        assert tool_call_arguments(result[0]) == {
            "actions": [{"action": "UNSUPPORTED", "value": "x"}],
        }
        children, error = validate_lite_action_batch_structure(
            "mobile",
            tool_call_arguments(result[0]),
        )
        assert len(children) == 1
        error = lite_action_batch_child_name_errors("mobile", children).get(0)
        assert error is not None
        assert error.child_action_name == "UNSUPPORTED"

    def test_native_open_from_agent_reaches_env_owned_open_app(self):
        """Adapter conversion canonicalizes native open; env owns availability."""
        tc = [
            {
                "name": "mobile_use",
                "arguments": {
                    "action": "open",
                    "text": "Settings",
                },
            }
        ]
        schema = make_tool_schema(
            "open_app",
            description="Open app.",
            parameters={
                "type": "object",
                "properties": {"app_name": {"type": "string", "enum": ["Settings"]}},
                "required": ["app_name"],
            },
        )

        result = self.space.convert_tool_calls_from_agent(
            tc,
            active_extra_tool_names=set(),
            active_extra_tool_schemas=[],
        )

        assert result == [make_tool_call("open_app", {"app_name": "Settings"})]

        result = self.space.convert_tool_calls_from_agent(
            tc,
            active_extra_tool_names={"open_app"},
            active_extra_tool_schemas=[schema],
        )

        assert result == [make_tool_call("open_app", {"app_name": "Settings"})]

        env_owned_invalid_enum = self.space.convert_tool_calls_from_agent(
            tc,
            active_extra_tool_names={"open_app"},
            active_extra_tool_schemas=[
                make_tool_schema(
                    "open_app",
                    description="Open app.",
                    parameters={
                        **tool_schema_parameters(schema),
                        "properties": {"app_name": {"type": "string", "enum": ["Chrome"]}},
                    },
                )
            ],
        )

        assert env_owned_invalid_enum == [make_tool_call("open_app", {"app_name": "Settings"})]

    def test_native_open_case_normalizes_unique_app_enum_match(self):
        schema = make_tool_schema(
            "open_app",
            description="Open app.",
            parameters={
                "type": "object",
                "properties": {
                    "app_name": {
                        "type": "string",
                        "enum": ["Markor", "Simple Calendar Pro"],
                    },
                },
                "required": ["app_name"],
            },
        )

        result = self.space.convert_tool_calls_from_agent(
            [
                {
                    "name": "mobile_use",
                    "arguments": {
                        "action": "open",
                        "text": "simple calendar pro",
                    },
                }
            ],
            active_extra_tool_names={"open_app"},
            active_extra_tool_schemas=[schema],
        )

        assert result == [make_tool_call("open_app", {"app_name": "Simple Calendar Pro"})]

    def test_open_app_wrapper_spelling_reaches_env_owned_open_app(self):
        schema = make_tool_schema(
            "open_app",
            description="Open app.",
            parameters={
                "type": "object",
                "properties": {"app_name": {"type": "string", "enum": ["Markor"]}},
                "required": ["app_name"],
            },
        )
        tc = [
            {
                "name": "mobile_use",
                "arguments": {
                    "action": "open_app",
                    "text": "markor",
                },
            }
        ]

        assert self.space.convert_tool_calls_from_agent(
            tc,
            active_extra_tool_names=set(),
            active_extra_tool_schemas=[],
        ) == [make_tool_call("open_app", {"app_name": "markor"})]

        assert self.space.convert_tool_calls_from_agent(
            tc,
            active_extra_tool_names={"open_app"},
            active_extra_tool_schemas=[schema],
        ) == [make_tool_call("open_app", {"app_name": "Markor"})]

    def test_terminate_from_agent(self):
        tc = [
            {
                "name": "mobile_use",
                "arguments": {
                    "action": "terminate",
                    "status": "failure",
                },
            }
        ]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert tool_call_name(result[0]) == "terminate"
        assert tool_call_arguments(result[0])["status"] == "failure"

    @pytest.mark.parametrize(
        "name,args",
        [
            ("click", {"coordinate": [500, 300]}),
            ("long_press", {"coordinate": [500, 300], "time": 2}),
            ("type", {"text": "hello"}),
            ("open", {"text": "Settings"}),
            ("answer", {"text": "done"}),
        ],
    )
    def test_top_level_native_action_value_used_as_tool_name_converts(self, name, args):
        """Flat output converts exactly like nested output — see the desktop
        twin. ``open`` matters most here: it is the native spelling of the env's
        ``open_app`` extra tool, so losing this path loses app launching."""
        flat = self.space.convert_tool_calls_from_agent([{"name": name, "arguments": args}])
        nested = self.space.convert_tool_calls_from_agent(
            [{"name": "mobile_use", "arguments": {"action": name, **args}}]
        )
        assert flat == nested

    def test_lite_canonical_gui_name_stays_standalone(self):
        """``tap`` is the CANONICAL Lite spelling; the native value is ``click``.
        Only native vocabulary is read as an action, so ``tap`` stays a
        standalone tool — matching the layer split the whole from-agent path
        rests on."""
        tc = [{"name": "tap", "arguments": {"coordinate": [500, 300]}}]
        assert self.space.convert_tool_calls_from_agent(tc) == [
            make_tool_call("tap", {"coordinate": [500, 300]})
        ]

    def test_active_extra_tool_outranks_a_colliding_native_action_value(self):
        """An env ``click(bid=...)`` is the env's tool, not the native ``click``
        the dispatch would read as a tap."""
        schema = {
            "type": "function",
            "function": {
                "name": "click",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {"bid": {"type": "string"}},
                    "required": ["bid"],
                },
            },
        }
        result = self.space.convert_tool_calls_from_agent(
            [{"name": "click", "arguments": {"bid": "a51"}}],
            active_extra_tool_names={"click"},
            active_extra_tool_schemas=[schema],
        )
        assert result == [make_tool_call("click", {"bid": "a51"})]

    def test_adjacent_native_gui_calls_batch_without_crossing_standalone_extra(self):
        tc = [
            {
                "name": "mobile_use",
                "arguments": {
                    "action": "click",
                    "coordinate": [100, 200],
                },
            },
            {"name": "mobile_use", "arguments": {"action": "type", "text": "a"}},
            {"name": "mobile_use", "arguments": {"action": "open", "text": "Settings"}},
            {"name": "mobile_use", "arguments": {"action": "wait", "time": 1}},
            {"name": "mobile_use", "arguments": {"action": "answer", "text": "done"}},
            {
                "name": "mobile_use",
                "arguments": {
                    "action": "click",
                    "coordinate": [300, 400],
                },
            },
        ]

        result = self.space.convert_tool_calls_from_agent(tc)

        assert result == [
            make_tool_call(
                "mobile",
                {
                    "actions": [
                        {"action": "tap", "coordinate": [100, 200], "clicks": 1},
                        {"action": "type", "text": "a"},
                    ],
                },
            ),
            make_tool_call("open_app", {"app_name": "Settings"}),
            make_tool_call(
                "mobile",
                {
                    "actions": [
                        {"action": "wait", "duration": 1.0},
                    ],
                },
            ),
            make_tool_call("response", {"text": "done"}),
            make_tool_call(
                "mobile",
                {
                    "actions": [
                        {"action": "tap", "coordinate": [300, 400], "clicks": 1},
                    ],
                },
            ),
        ]
        assert self.space.convert_tool_calls_to_agent(result) == tc

    # -- round-trip --

    @pytest.mark.parametrize(
        "cua_tc",
        [
            LiteMobileActionSpace.tap(coordinate=[500, 300]),
            LiteMobileActionSpace.long_press(coordinate=[100, 200], duration=2.0),
            LiteMobileActionSpace.swipe(start_coordinate=[100, 200], coordinate=[100, 800]),
            LiteMobileActionSpace.type(text="hello"),
            LiteMobileActionSpace.system_button(button="Home"),
            LiteMobileActionSpace.system_button(button="Back"),
            make_tool_call("open_app", {"app_name": "Chrome"}),
            make_tool_call("response", {"text": "answer"}),
            LiteMobileActionSpace.wait(duration=1.5),
            make_tool_call("terminate", {"status": "success"}),
        ],
    )
    def test_round_trip(self, cua_tc):
        space = Qwen3VLMobileActionSpace()
        agent_tcs = space.convert_tool_calls_to_agent([cua_tc])
        restored = space.convert_tool_calls_from_agent(agent_tcs)
        assert len(restored) == 1
        if tool_call_name(cua_tc) in {
            "long_press",
            "swipe",
            "system_button",
            "tap",
            "type",
            "wait",
        }:
            assert _single_action(restored, "mobile") == {
                "action": tool_call_name(cua_tc),
                **tool_call_arguments(cua_tc),
            }
        else:
            assert tool_call_name(restored[0]) == tool_call_name(cua_tc)
            assert tool_call_arguments(restored[0]) == tool_call_arguments(cua_tc)
