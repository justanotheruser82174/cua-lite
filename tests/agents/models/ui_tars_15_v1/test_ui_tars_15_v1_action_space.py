"""
Tests for UITars15V1DesktopActionSpace.

Covers:
  1. Registry and available actions
  2. Tool call creation
  3. Tool call conversion: CUA-lite ↔ UITars v1 (round-trip)
  4. Text format: box tokens (v1 uses <|box_start|>)
  5. format_message_as_text

Run:
    uv run pytest tests/agents/models/ui_tars_15_v1/test_ui_tars_15_v1_action_space.py -v
"""

from __future__ import annotations

import pytest
from agents._support.valid_actions_gating import (
    RESPONSE_SCHEMA,
    TERMINATE_SCHEMA,
    agent_adapter_for,
)

from lite.agents.bootstrap import register_all
from lite.agents.core.action_space.base import ActionSpaceRegistry, LiteDesktopActionSpace
from lite.agents.models.ui_tars_15_v1.action_space import (
    UITars15V1DesktopActionSpace,
    UITars15V1MobileActionSpace,
)
from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.schemas import tool_schema_parameters

# Adapter rendering tests need adapter registry side effects even when this file
# runs alone in an xdist worker.
register_all()


def _only_desktop_action(tool_calls):
    assert len(tool_calls) == 1
    call = tool_calls[0]
    assert tool_call_name(call) == "computer"
    actions = tool_call_arguments(call)["actions"]
    assert len(actions) == 1
    return actions[0]


# =============================================================================
# 1. Registry and available actions
# =============================================================================


class TestRegistryAndActions:
    def test_registry_key(self):
        space = ActionSpaceRegistry.get("ui_tars_15_v1@desktop")
        assert isinstance(space, UITars15V1DesktopActionSpace)

    def test_available_actions(self):
        actions = UITars15V1DesktopActionSpace.get_declared_action_schema_names()
        for name in [
            "click",
            "left_double",
            "right_single",
            "drag",
            "hotkey",
            "type",
            "scroll",
            "wait",
            "finished",
        ]:
            assert name in actions

    def test_has_call_user(self):
        """v1 retains call_user (matches upstream OSWorld/mm_agents/uitars15_v1.py
        L100 CALL_USER = "call_user" and the v1 SFT prompt example at L43)."""
        actions = UITars15V1DesktopActionSpace.get_declared_action_schema_names()
        assert "call_user" in actions

    def test_finished_has_content_param(self):
        """v1 finished() has optional content parameter (unlike original ui_tars)."""
        schema = UITars15V1DesktopActionSpace.get_tool_schema("finished")
        assert schema["type"] == "function"
        assert "content" in tool_schema_parameters(schema)["properties"]


# =============================================================================
# 2. Tool call creation
# =============================================================================


class TestToolCalls:
    def test_click(self):
        tc = UITars15V1DesktopActionSpace.click(start_box=[500, 300])
        assert tool_call_name(tc) == "click"
        assert tool_call_arguments(tc)["start_box"] == [500, 300]

    def test_finished_with_content(self):
        tc = UITars15V1DesktopActionSpace.finished(content="task done")
        assert tool_call_arguments(tc)["content"] == "task done"

    def test_finished_without_content(self):
        tc = UITars15V1DesktopActionSpace.finished()
        assert "content" not in tool_call_arguments(tc)


# =============================================================================
# 3. Tool call conversion
# =============================================================================


class TestToolCallConversion:
    def setup_method(self):
        self.space = UITars15V1DesktopActionSpace()

    def test_click_to_agent(self):
        tc = [LiteDesktopActionSpace.click(coordinate=[500, 300])]
        result = self.space.convert_tool_calls_to_agent(tc)
        assert result[0]["name"] == "click"
        assert result[0]["arguments"]["start_box"] == [500, 300]

    def test_computer_batch_to_agent_unwraps(self):
        tc = make_tool_call(
            "computer",
            {
                "actions": [
                    {"action": "click", "coordinate": [500, 300]},
                    {"action": "type", "text": "hello"},
                ]
            },
        )
        result = self.space.convert_tool_calls_to_agent([tc])
        assert [r["name"] for r in result] == ["click", "type"]
        assert result[1]["arguments"]["content"] == "hello"

    def test_response_to_finished_with_content(self):
        """CUA-lite response → UITars v1 finished(content=...)."""
        tc = [make_tool_call("response", {"text": "42"})]
        result = self.space.convert_tool_calls_to_agent(tc)
        assert result[0]["name"] == "finished"
        assert result[0]["arguments"]["content"] == "42"

    def test_key_glyph_round_trip_uses_space_separated_wire(self):
        tc = [LiteDesktopActionSpace.key(keys=["ctrl", "+"])]
        agent_tcs = self.space.convert_tool_calls_to_agent(tc)

        assert agent_tcs[0]["arguments"]["key"] == "ctrl +"
        assert self.space.convert_tool_calls_from_agent(agent_tcs) == tc

    @pytest.mark.parametrize(
        "adapter_key,platform",
        [
            ("ui_tars_15_v1@desktop@use", "desktop"),
            ("ui_tars_15_v1@mobile@use", "mobile"),
        ],
    )
    @pytest.mark.parametrize(
        "call,schema",
        [
            (make_tool_call("response", {"text": "42"}), RESPONSE_SCHEMA),
            (make_tool_call("terminate", {"status": "success"}), TERMINATE_SCHEMA),
        ],
    )
    def test_finish_render_delegates_to_family_converter(
        self,
        adapter_key,
        platform,
        call,
        schema,
    ) -> None:
        message = {"role": "assistant", "content": [], "tool_calls": [call]}

        adapter = agent_adapter_for(adapter_key, platform)
        adapter.convert_message_to_agent(message)

        adapter = agent_adapter_for(adapter_key, platform, extra_tool_schemas=[schema])
        adapter.convert_message_to_agent(message)

    def test_finished_with_content_from_agent(self):
        """UITars v1 finished(content=...) → CUA-lite response."""
        tc = [{"name": "finished", "arguments": {"content": "result"}}]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert tool_call_name(result[0]) == "response"
        assert tool_call_arguments(result[0])["text"] == "result"

    def test_finished_without_content_from_agent(self):
        """UITars v1 finished() → CUA-lite terminate(success)."""
        tc = [{"name": "finished", "arguments": {}}]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert tool_call_name(result[0]) == "terminate"
        assert tool_call_arguments(result[0])["status"] == "success"

    @pytest.mark.parametrize(
        "key,expected",
        [
            ("+", ["+"]),
            ("ctrl +", ["ctrl", "+"]),
            ("ctrl -", ["ctrl", "-"]),
            ("ctrl ,", ["ctrl", ","]),
            ("ctrl left", ["ctrl", "left"]),
        ],
    )
    def test_hotkey_from_agent_punctuation_uses_space_wire(self, key, expected):
        result = self.space.convert_tool_calls_from_agent(
            [{"name": "hotkey", "arguments": {"key": key}}]
        )

        assert _only_desktop_action(result) == {"action": "key", "keys": expected}

    def test_hotkey_from_agent_preserves_plus_chord_string_for_core_normalizer(self):
        result = self.space.convert_tool_calls_from_agent(
            [{"name": "hotkey", "arguments": {"key": "ctrl++"}}]
        )

        assert _only_desktop_action(result) == {"action": "key", "keys": ["ctrl", "+"]}

    def test_hotkey_from_agent_rejects_non_string_key(self):
        with pytest.raises(ValueError, match="hotkey requires key string"):
            self.space.convert_tool_calls_from_agent(
                [{"name": "hotkey", "arguments": {"key": ["ctrl", "c"]}}]
            )

    @pytest.mark.parametrize(
        "cua_tc",
        [
            LiteDesktopActionSpace.click(coordinate=[500, 300]),
            LiteDesktopActionSpace.click(coordinate=[100, 200], button="right"),
            LiteDesktopActionSpace.click(coordinate=[100, 200], clicks=2),
            LiteDesktopActionSpace.type(text="hello world"),
            LiteDesktopActionSpace.key(keys=["ctrl", "v"]),
            LiteDesktopActionSpace.key(keys=["ctrl", "+"]),
        ],
    )
    def test_round_trip(self, cua_tc):
        space = UITars15V1DesktopActionSpace()
        agent_tcs = space.convert_tool_calls_to_agent([cua_tc])
        restored = space.convert_tool_calls_from_agent(agent_tcs)
        assert len(restored) == 1
        assert tool_call_name(restored[0]) == tool_call_name(cua_tc)
        assert tool_call_arguments(restored[0]) == tool_call_arguments(cua_tc)


# =============================================================================
# 4. Text format
# =============================================================================


class TestTextFormat:
    def setup_method(self):
        self.space = UITars15V1DesktopActionSpace()

    def test_click_format(self):
        tc = self.space.convert_tool_calls_to_agent(
            [LiteDesktopActionSpace.click(coordinate=[500, 300])]
        )[0]
        text = self.space.format_tool_call_as_text(tc)
        # Bare `(x,y)` — matches raw UI-TARS-1.5 model output; no `<|box_start|>`.
        assert text == "click(start_box='(500,300)')"
        assert "<|box_start|>" not in text

    def test_scroll_format_preserves_direction(self):
        # Even when direction matches the default ("down"), re-rendering
        # must keep it — otherwise next-turn context loses the direction.
        tc = self.space.convert_tool_calls_to_agent(
            [
                LiteDesktopActionSpace.scroll(
                    coordinate=[1041, 542],
                    direction="down",
                    amount=5,
                )
            ]
        )[0]
        text = self.space.format_tool_call_as_text(tc)
        assert "direction='down'" in text
        assert "start_box='(1041,542)'" in text

    def test_renderer_takes_the_bare_projection_not_a_canonical_call(self):
        """Input shape is this family's contract, not an implementation detail.

        The base renderer is the mirror image (canonical only), so a canonical
        call must fail here rather than render something plausible.
        """
        with pytest.raises(KeyError):
            self.space.format_tool_call_as_text(
                LiteDesktopActionSpace.click(coordinate=[500, 300])
            )

    def test_plural_path_carries_the_same_shape(self):
        """``format_tool_calls_as_text`` adds no shape handling of its own."""
        tcs = self.space.convert_tool_calls_to_agent([
            LiteDesktopActionSpace.click(coordinate=[500, 300]),
            LiteDesktopActionSpace.type(text="hello"),
        ])

        assert self.space.format_tool_calls_as_text(tcs) == (
            "click(start_box='(500,300)')\n\ntype(content='hello')"
        )

    def test_format_message_as_text(self):
        tc = self.space.convert_tool_calls_to_agent(
            [LiteDesktopActionSpace.click(coordinate=[500, 300])]
        )[0]
        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "Click it."}],
            "tool_calls": [tc],
        }
        text = self.space.format_message_as_text(msg)
        assert "Thought:" not in text  # no inline_reasoning → no Thought: line
        assert "Click it." in text
        assert "Action:" in text
        assert "(500,300)" in text


# =============================================================================
# 6. Mobile batch conversion
# =============================================================================


class TestMobileBatchConversion:
    def test_mobile_batch_to_agent_unwraps(self):
        space = UITars15V1MobileActionSpace()
        tc = make_tool_call(
            "mobile",
            {
                "actions": [
                    {"action": "tap", "coordinate": [500, 300]},
                    {"action": "type", "text": "hello"},
                ]
            },
        )
        result = space.convert_tool_calls_to_agent([tc])
        assert [r["name"] for r in result] == ["click", "type"]
        assert result[1]["arguments"]["content"] == "hello"

    def test_mobile_system_button_from_agent_is_canonical_batch(self):
        space = UITars15V1MobileActionSpace()
        result = space.convert_tool_calls_from_agent(
            [
                {"name": "press_home", "arguments": {}},
            ]
        )
        assert result == [
            make_tool_call(
                "mobile",
                {
                    "actions": [
                        {"action": "system_button", "button": "Home"},
                    ]
                },
            ),
        ]

    def test_unsupported_mobile_system_button_cannot_render_as_flat_tool(self):
        space = UITars15V1MobileActionSpace()
        with pytest.raises(ValueError, match="press_home and press_back"):
            space.convert_tool_calls_to_agent(
                [
                    make_tool_call(
                        "mobile",
                        {
                            "actions": [
                                {"action": "system_button", "button": "Enter"},
                            ]
                        },
                    )
                ]
            )
