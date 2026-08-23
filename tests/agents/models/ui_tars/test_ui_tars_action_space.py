"""
Tests for UITarsDesktopActionSpace (original UI-TARS).

Covers:
  1. Registry and available actions
  2. Tool call creation
  3. Tool call conversion: CUA-lite ↔ UITars (round-trip)
  4. Text format: box tokens in format_tool_call_as_text
  5. format_message_as_text

Run:
    uv run pytest tests/agents/models/ui_tars/test_ui_tars_action_space.py -v
"""

from __future__ import annotations

import json

import pytest
from agents._support.valid_actions_gating import RESPONSE_SCHEMA, agent_adapter_for

from lite.agents.bootstrap import register_all
from lite.agents.core.action_space.base import (
    ActionSpaceRegistry,
    LiteDesktopActionSpace,
    LiteMobileActionSpace,
    LitePointActionSpace,
)
from lite.agents.models.ui_tars.action_space import (
    UITarsDesktopActionSpace,
    UITarsDesktopGroundingPointActionSpace,
    UITarsMobileActionSpace,
    UITarsMobileGroundingPointActionSpace,
)
from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.schemas import tool_schema_name, tool_schema_parameters

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
        space = ActionSpaceRegistry.get("ui_tars@desktop")
        assert isinstance(space, UITarsDesktopActionSpace)

    def test_available_actions(self):
        actions = UITarsDesktopActionSpace.get_declared_action_schema_names()
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
            "call_user",
        ]:
            assert name in actions

    def test_tool_schemas_count(self):
        schemas = UITarsDesktopActionSpace.get_tool_schemas()
        assert (
            len(schemas) >= 10
        )  # click, left_double, right_single, drag, hotkey, type, scroll, wait, finished, call_user
        click_schema = next(s for s in schemas if tool_schema_name(s) == "click")
        assert click_schema["type"] == "function"
        assert "start_box" in tool_schema_parameters(click_schema)["properties"]


# =============================================================================
# 2. Tool call creation
# =============================================================================


class TestToolCalls:
    def test_click(self):
        tc = UITarsDesktopActionSpace.click(start_box=[500, 300])
        assert tool_call_name(tc) == "click"
        assert tool_call_arguments(tc)["start_box"] == [500, 300]

    def test_left_double(self):
        tc = UITarsDesktopActionSpace.left_double(start_box=[100, 200])
        assert tool_call_name(tc) == "left_double"

    def test_right_single(self):
        tc = UITarsDesktopActionSpace.right_single(start_box=[100, 200])
        assert tool_call_name(tc) == "right_single"

    def test_drag(self):
        tc = UITarsDesktopActionSpace.drag(start_box=[100, 200], end_box=[300, 400])
        assert tool_call_arguments(tc)["start_box"] == [100, 200]
        assert tool_call_arguments(tc)["end_box"] == [300, 400]

    def test_hotkey(self):
        tc = UITarsDesktopActionSpace.hotkey(key="ctrl c")
        assert tool_call_arguments(tc)["key"] == "ctrl c"

    def test_type(self):
        tc = UITarsDesktopActionSpace.type(content="hello")
        assert tool_call_arguments(tc)["content"] == "hello"

    def test_wait(self):
        tc = UITarsDesktopActionSpace.wait()
        assert tool_call_name(tc) == "wait"

    def test_finished(self):
        tc = UITarsDesktopActionSpace.finished()
        assert tool_call_name(tc) == "finished"

    def test_call_user(self):
        tc = UITarsDesktopActionSpace.call_user()
        assert tool_call_name(tc) == "call_user"


# =============================================================================
# 3. Tool call conversion
# =============================================================================


class TestToolCallConversion:
    def setup_method(self):
        self.space = UITarsDesktopActionSpace()

    # -- to_agent --

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
        assert all(set(r) == {"name", "arguments"} for r in result)

    def test_right_click_to_agent(self):
        tc = [LiteDesktopActionSpace.click(coordinate=[100, 200], button="right")]
        result = self.space.convert_tool_calls_to_agent(tc)
        assert result[0]["name"] == "right_single"

    def test_double_click_to_agent(self):
        tc = [LiteDesktopActionSpace.click(coordinate=[100, 200], clicks=2)]
        result = self.space.convert_tool_calls_to_agent(tc)
        assert result[0]["name"] == "left_double"

    def test_type_to_agent(self):
        tc = [LiteDesktopActionSpace.type(text="hello")]
        result = self.space.convert_tool_calls_to_agent(tc)
        assert result[0]["name"] == "type"
        assert result[0]["arguments"]["content"] == "hello"

    def test_key_to_hotkey(self):
        tc = [LiteDesktopActionSpace.key(keys=["ctrl", "a"])]
        result = self.space.convert_tool_calls_to_agent(tc)
        assert result[0]["name"] == "hotkey"
        assert result[0]["arguments"]["key"] == "ctrl a"

    def test_key_glyph_round_trip_uses_space_separated_wire(self):
        tc = [LiteDesktopActionSpace.key(keys=["ctrl", "+"])]
        agent_tcs = self.space.convert_tool_calls_to_agent(tc)

        assert agent_tcs[0]["arguments"]["key"] == "ctrl +"
        assert self.space.convert_tool_calls_from_agent(agent_tcs) == tc

    def test_terminate_to_finished(self):
        tc = [make_tool_call("terminate", {"status": "success"})]
        result = self.space.convert_tool_calls_to_agent(tc)
        assert result[0]["name"] == "finished"

    def test_wait_to_agent(self):
        tc = [LiteDesktopActionSpace.wait(duration=2.0)]
        result = self.space.convert_tool_calls_to_agent(tc)
        assert result[0]["name"] == "wait"

    def test_unsupported_desktop_action_to_agent_fails_loudly(self):
        tc = [LiteDesktopActionSpace.mouse_move(coordinate=[500, 300])]
        with pytest.raises(ValueError, match="cannot render canonical tool 'mouse_move'"):
            self.space.convert_tool_calls_to_agent(tc)

    def test_response_render_fails_loudly(self) -> None:
        """UI-TARS desktop ``finished()`` has no lossless answer-text channel."""
        call = make_tool_call("response", {"text": "42"})
        with pytest.raises(
            ValueError, match=r"^UI-TARS desktop cannot render canonical tool 'response'$"
        ):
            UITarsDesktopActionSpace().convert_tool_calls_to_agent([call])

        adapter = agent_adapter_for(
            "ui_tars@desktop@use",
            "desktop",
            extra_tool_schemas=[RESPONSE_SCHEMA],
        )
        message = {"role": "assistant", "content": [], "tool_calls": [call]}
        with pytest.raises(
            ValueError, match=r"^UI-TARS desktop cannot render canonical tool 'response'$"
        ):
            adapter.convert_message_to_agent(message)

    def test_desktop_never_silently_discards_answer_text(self) -> None:
        """Rendering ``response(text=...)`` either preserves the answer or raises."""
        answer = "the-answer-is-42"
        call = make_tool_call("response", {"text": answer})
        message = {"role": "assistant", "content": [], "tool_calls": [call]}

        space = UITarsDesktopActionSpace()
        try:
            rendered = space.convert_tool_calls_to_agent([call])
        except ValueError as exc:
            assert "cannot render canonical tool 'response'" in str(exc)
        else:  # pragma: no cover - only reachable if the raise is ever removed
            assert answer in json.dumps(rendered), rendered

        adapter = agent_adapter_for(
            "ui_tars@desktop@use",
            "desktop",
            extra_tool_schemas=[RESPONSE_SCHEMA],
        )
        try:
            agent_message = adapter.convert_message_to_agent(message)
        except ValueError as exc:
            assert "cannot render canonical tool 'response'" in str(exc)
        else:  # pragma: no cover - only reachable if the raise is ever removed
            assert answer in json.dumps(agent_message), agent_message

        mobile = UITarsMobileActionSpace().convert_tool_calls_to_agent([call])
        assert answer in json.dumps(mobile), mobile

    # -- from_agent --

    def test_click_from_agent(self):
        tc = [{"name": "click", "arguments": {"start_box": [500, 300]}}]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert _only_desktop_action(result) == {"action": "click", "coordinate": [500, 300]}

    def test_left_double_from_agent(self):
        tc = [{"name": "left_double", "arguments": {"start_box": [100, 200]}}]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert _only_desktop_action(result)["clicks"] == 2

    def test_right_single_from_agent(self):
        tc = [{"name": "right_single", "arguments": {"start_box": [100, 200]}}]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert _only_desktop_action(result)["button"] == "right"

    def test_hotkey_from_agent(self):
        tc = [{"name": "hotkey", "arguments": {"key": "ctrl c"}}]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert _only_desktop_action(result) == {"action": "key", "keys": ["ctrl", "c"]}

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

    def test_type_from_agent(self):
        tc = [{"name": "type", "arguments": {"content": "hello"}}]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert _only_desktop_action(result) == {"action": "type", "text": "hello"}

    def test_finished_from_agent(self):
        tc = [{"name": "finished", "arguments": {}}]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert tool_call_name(result[0]) == "terminate"
        assert tool_call_arguments(result[0])["status"] == "success"

    def test_call_user_from_agent(self):
        tc = [{"name": "call_user", "arguments": {}}]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert tool_call_name(result[0]) == "terminate"
        assert tool_call_arguments(result[0])["status"] == "failure"

    def test_canonical_gui_name_from_agent_flows_to_env_surface(self):
        # ``key`` is canonical Lite desktop, but original UI-TARS native output
        # should use ``hotkey``. Preserve parsed model output and let the env own
        # execution/error feedback instead of silently dropping it.
        result = self.space.convert_tool_calls_from_agent(
            [
                {"name": "key", "arguments": {"keys": ["ctrl", "c"]}},
            ]
        )
        assert _only_desktop_action(result) == {"action": "key", "keys": ["ctrl", "c"]}

    # -- round-trip --

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
        space = UITarsDesktopActionSpace()
        agent_tcs = space.convert_tool_calls_to_agent([cua_tc])
        restored = space.convert_tool_calls_from_agent(agent_tcs)
        assert len(restored) == 1
        assert tool_call_name(restored[0]) == tool_call_name(cua_tc)
        assert tool_call_arguments(restored[0]) == tool_call_arguments(cua_tc)


# =============================================================================
# 4. Text format: box tokens
# =============================================================================


class TestTextFormat:
    def setup_method(self):
        self.space = UITarsDesktopActionSpace()

    def test_click_format(self):
        tc = self.space.convert_tool_calls_to_agent(
            [LiteDesktopActionSpace.click(coordinate=[500, 300])]
        )[0]
        text = self.space.format_tool_call_as_text(tc)
        # Bare `(x,y)` — matches raw UI-TARS model output; no `<|box_start|>`.
        assert text == "click(start_box='(500,300)')"
        assert "<|box_start|>" not in text

    def test_drag_format(self):
        tc = self.space.convert_tool_calls_to_agent(
            [
                LiteDesktopActionSpace.drag(
                    start_coordinate=[100, 200],
                    coordinate=[300, 400],
                )
            ]
        )[0]
        text = self.space.format_tool_call_as_text(tc)
        assert text == "drag(start_box='(100,200)', end_box='(300,400)')"
        assert "<|box_start|>" not in text

    def test_scroll_format_preserves_direction(self):
        # Even when direction matches the default ("down"), re-rendering
        # must keep it — otherwise next-turn context loses the direction.
        tc = self.space.convert_tool_calls_to_agent(
            [
                LiteDesktopActionSpace.scroll(
                    coordinate=[376, 873],
                    direction="down",
                    amount=5,
                )
            ]
        )[0]
        text = self.space.format_tool_call_as_text(tc)
        assert "direction='down'" in text
        assert "start_box='(376,873)'" in text

    def test_type_no_coord(self):
        tc = self.space.convert_tool_calls_to_agent(
            [LiteDesktopActionSpace.type(text="hello")]
        )[0]
        text = self.space.format_tool_call_as_text(tc)
        assert "hello" in text
        assert "start_box" not in text

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
        # ``action_description`` renders as the raw prose (no ``Thought:``
        # relabel); ``Action:`` comes from tool_calls.
        tc = self.space.convert_tool_calls_to_agent(
            [LiteDesktopActionSpace.click(coordinate=[500, 300])]
        )[0]
        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "Click the button."}],
            "tool_calls": [tc],
        }
        text = self.space.format_message_as_text(msg)
        assert "Thought:" not in text  # no inline_reasoning → no Thought: line
        assert "Click the button." in text
        assert "Action:" in text
        assert "(500,300)" in text


# =============================================================================
# 6. Mobile batch conversion
# =============================================================================


class TestMobileBatchConversion:
    def test_mobile_batch_to_agent_unwraps(self):
        space = UITarsMobileActionSpace()
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
        assert all(set(r) == {"name", "arguments"} for r in result)

    def test_unsupported_mobile_action_to_agent_fails_loudly(self):
        space = UITarsMobileActionSpace()
        tc = [LiteMobileActionSpace.system_button(button="Enter")]
        with pytest.raises(ValueError, match="cannot render system_button 'Enter'"):
            space.convert_tool_calls_to_agent(tc)

    def test_canonical_gui_name_from_agent_flows_to_env_surface(self):
        # ``tap`` is canonical Lite mobile, but original UI-TARS native output
        # should use ``click``. Preserve parsed model output and let the env own
        # execution/error feedback instead of silently dropping it.
        space = UITarsMobileActionSpace()
        result = space.convert_tool_calls_from_agent(
            [
                {"name": "tap", "arguments": {"coordinate": [500, 300]}},
            ]
        )
        assert len(result) == 1
        assert tool_call_name(result[0]) == "mobile"
        assert tool_call_arguments(result[0])["actions"] == [
            {"action": "tap", "coordinate": [500, 300]},
        ]


class TestGroundingProjectionBoundary:
    @pytest.mark.parametrize(
        "space_cls",
        [
            UITarsDesktopGroundingPointActionSpace,
            UITarsMobileGroundingPointActionSpace,
        ],
    )
    def test_point_to_agent_uses_bare_model_function_projection(self, space_cls):
        result = space_cls().convert_tool_calls_to_agent(
            [
                LitePointActionSpace.point(coordinate=[850, 120]),
            ]
        )

        assert result == [{"name": "click", "arguments": {"start_box": [850, 120]}}]
        assert set(result[0]) == {"name", "arguments"}
