"""
Tests for UI-TARS 1.5 v1 adapters.

Covers:
  1. Registry: correct adapter classes for all ui_tars_15_v1 keys
  2. System prompt: v1 has no <think> tags, different from v2
  3. parse_raw_assistant_response: no <think> handling
  4. convert_sample_to_agent: tool_calls to text conversion
  5. convert_message_from_agent: round-trip
  6. Grounding bbox/point: tool_calls embedded into content
  7. Trajectory unroll
  8. Batched conversion

Run:
    uv run pytest tests/agents/models/ui_tars_15_v1/test_ui_tars_15_v1_adapter.py -v
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import pytest
from lite_samples import (
    build_lite_trajectory,
    sample_grounding_action_desktop,
    sample_grounding_bbox,
    sample_grounding_point,
    sample_trajectory_long,
    sample_trajectory_two_turns,
    sample_understanding,
)

from lite.agents.core.adapter import (
    AgentAdapterRegistry,
    AsIsAdapter,
    BaseAgentAdapter,
)
from lite.agents.models.ui_tars.protocol import UITarsHistoryProtocol
from lite.agents.models.ui_tars_15_v1.adapter import (
    USE_SYSTEM_PROMPT,
    USE_USER_PROMPT,
    UITars15V1DesktopGroundingPointAdapter,
    UITars15V1DesktopUseAdapter,
    UITars15V1GroundingBBoxAdapter,
    UITars15V1MobileUseAdapter,
)
from lite.core import (
    LiteCUAMetadata,
    LiteSample,
)
from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.extra_tools import LiteFinishToolSet


def _single_batched_action(message, wrapper_name):
    call = message["tool_calls"][0]
    assert tool_call_name(call) == wrapper_name
    return tool_call_arguments(call)["actions"][0]


def _mobile_prompt_only_sample():
    sample = sample_trajectory_two_turns()
    return dataclasses.replace(
        sample,
        messages=[sample.messages[0]],
        metadata=dataclasses.replace(
            sample.metadata,
            dims=(LiteCUAMetadata.Platform.MOBILE, LiteCUAMetadata.TaskType.USE),
        ),
    )


def _single_computer_action(message):
    return _single_batched_action(message, "computer")


def _single_mobile_action(message):
    return _single_batched_action(message, "mobile")


# =============================================================================
# 1. Registry
# =============================================================================


class TestRegistry:
    """Test adapter registry resolution for ui_tars_15_v1 keys."""

    @pytest.mark.parametrize(
        "key,expected_cls",
        [
            ("ui_tars_15_v1@desktop@grounding.point", UITars15V1DesktopGroundingPointAdapter),
            ("ui_tars_15_v1@desktop@use", UITars15V1DesktopUseAdapter),
            ("ui_tars_15_v1@browser@grounding.point", UITars15V1DesktopGroundingPointAdapter),
            ("ui_tars_15_v1@browser@use", UITars15V1DesktopUseAdapter),
            ("ui_tars_15_v1@desktop@grounding.bbox", UITars15V1GroundingBBoxAdapter),
            ("ui_tars_15_v1@browser@grounding.bbox", UITars15V1GroundingBBoxAdapter),
        ],
    )
    def test_registry_returns_correct_class(self, key, expected_cls):
        adapter = AgentAdapterRegistry.get(key)
        assert isinstance(adapter, BaseAgentAdapter)
        assert type(adapter) is expected_cls

    def test_understanding_is_pass_through(self):
        for key in [
            "ui_tars_15_v1@desktop@understanding",
            "ui_tars_15_v1@browser@understanding",
        ]:
            adapter = AgentAdapterRegistry.get(key)
            assert type(adapter) is AsIsAdapter, f"Expected AsIsAdapter for {key}"

    def test_trajectory_has_system_prompt(self):
        adapter = AgentAdapterRegistry.get("ui_tars_15_v1@desktop@use")
        assert adapter.system_prompt == USE_SYSTEM_PROMPT


# =============================================================================
# 2. System Prompt
# =============================================================================


class TestPrompts:
    """Test v1 prompt structure (matches OSWorld)."""

    def test_system_prompt_is_simple(self):
        """System prompt matches OSWorld: 'You are a helpful assistant.'"""
        assert USE_SYSTEM_PROMPT == "You are a helpful assistant."

    def test_no_think_tags_in_user_prompt(self):
        assert "<think>" not in USE_USER_PROMPT
        assert "</think>" not in USE_USER_PROMPT

    def test_user_prompt_has_thought_action_format(self):
        assert "Thought:" in USE_USER_PROMPT
        assert "Action:" in USE_USER_PROMPT

    def test_user_prompt_has_placeholders(self):
        assert "{action_space}" in USE_USER_PROMPT
        assert "{instruction}" in USE_USER_PROMPT

    def test_user_prompt_has_user_instruction_header(self):
        assert "## User Instruction" in USE_USER_PROMPT


# =============================================================================
# 3. parse_raw_assistant_response
# =============================================================================


class TestParseRawAssistantResponse:
    """Test parse_raw_assistant_response for v1 format (no <think> tags).

    parse_raw_assistant_response now returns a trivial wrapper:
        {"role": "assistant", "content": [{"type": "text", "text": raw}]}
    All parsing (Thought:/Action:, coordinate normalization) is in
    convert_message_from_agent. Tests verify the trivial wrapper, then
    chain through convert_message_from_agent for structured assertions.
    """

    def setup_method(self):
        self.adapter = AgentAdapterRegistry.get("ui_tars_15_v1@desktop@use")

    def test_basic_thought_and_action(self):
        response = "Thought: Click the search bar.\nAction: click(start_box='(500,100)')"
        result = self.adapter.parse_raw_assistant_response(response)
        # Trivial wrapper: raw text preserved verbatim
        assert result["role"] == "assistant"
        assert result["content"] == [{"type": "text", "text": response}]
        assert "tool_calls" not in result
        # Structured assertions via convert_message_from_agent
        lite = self.adapter.convert_message_from_agent(result)
        assert "search bar" in lite["content"][0]["text"]
        action = _single_computer_action(lite)
        assert action["action"] == "click"
        assert action["coordinate"] == [500, 100]

    def test_no_reasoning_content(self):
        """v1 responses should not have reasoning_content (no <think> tags)."""
        response = "Thought: Click the button.\nAction: click(start_box='(300,200)')"
        result = self.adapter.parse_raw_assistant_response(response)
        assert "reasoning_content" not in result

    def test_finished_action(self):
        response = "Thought: Done.\nAction: finished(content='task completed')"
        result = self.adapter.parse_raw_assistant_response(response)
        assert result["content"] == [{"type": "text", "text": response}]
        lite = self.adapter.convert_message_from_agent(result)
        # finished(content=...) maps to response(text=...) in CUA-lite
        assert tool_call_name(lite["tool_calls"][0]) == "response"
        assert tool_call_arguments(lite["tool_calls"][0])["text"] == "task completed"

    def test_wait_action(self):
        response = "Thought: Wait for load.\nAction: wait()"
        result = self.adapter.parse_raw_assistant_response(response)
        assert result["content"] == [{"type": "text", "text": response}]
        lite = self.adapter.convert_message_from_agent(result)
        assert _single_computer_action(lite)["action"] == "wait"

    def test_hotkey_action(self):
        response = "Thought: Copy text.\nAction: hotkey(key='ctrl c')"
        result = self.adapter.parse_raw_assistant_response(response)
        assert result["content"] == [{"type": "text", "text": response}]
        lite = self.adapter.convert_message_from_agent(result)
        # hotkey maps to key in CUA-lite, with key string split into list
        action = _single_computer_action(lite)
        assert action["action"] == "key"
        assert action["keys"] == ["ctrl", "c"]

    def test_type_action(self):
        response = "Thought: type search query.\nAction: type(content='hello world')"
        result = self.adapter.parse_raw_assistant_response(response)
        assert result["content"] == [{"type": "text", "text": response}]
        lite = self.adapter.convert_message_from_agent(result)
        action = _single_computer_action(lite)
        assert action["action"] == "type"
        assert action["text"] == "hello world"

    def test_empty_response(self):
        result = self.adapter.parse_raw_assistant_response("")
        assert result["role"] == "assistant"
        assert result["content"] == [{"type": "text", "text": ""}]
        assert "tool_calls" not in result

    def test_no_action(self):
        response = "Thought: I'm thinking..."
        result = self.adapter.parse_raw_assistant_response(response)
        assert result["content"] == [{"type": "text", "text": response}]
        assert "tool_calls" not in result
        lite = self.adapter.convert_message_from_agent(result)
        assert "thinking" in lite["content"][0]["text"]


# =============================================================================
# 4. Full pipeline: parse → convert_from_agent
# =============================================================================


class TestFullParsePipeline:
    """Test parse_raw_assistant_response → convert_message_from_agent chain."""

    def setup_method(self):
        self.adapter = AgentAdapterRegistry.get("ui_tars_15_v1@desktop@use")

    def test_parse_then_convert_click(self):
        response = "Thought: Click the button.\nAction: click(start_box='(500,300)')"
        agent_msg = self.adapter.parse_raw_assistant_response(response)
        lite_msg = self.adapter.convert_message_from_agent(agent_msg)

        assert lite_msg["role"] == "assistant"
        action = _single_computer_action(lite_msg)
        assert action["action"] == "click"
        assert action["coordinate"] == [500, 300]

    def test_parse_then_convert_finished(self):
        response = "Thought: Task done.\nAction: finished(content='result')"
        agent_msg = self.adapter.parse_raw_assistant_response(response)
        lite_msg = self.adapter.convert_message_from_agent(agent_msg)

        tc = lite_msg["tool_calls"][0]
        assert tool_call_name(tc) == "response"
        assert tool_call_arguments(tc)["text"] == "result"


# =============================================================================
# 5. convert_sample_to_agent
# =============================================================================


class TestConvertSampleToAgent:
    """Last-step view of ``unroll(sample)`` (predict-time render)."""

    def _last_step(self, key, sample):
        return AgentAdapterRegistry.get(key).unroll(sample).steps[-1]

    def test_converts_tool_calls_to_text(self):
        sample = sample_grounding_action_desktop()
        msgs = self._last_step("ui_tars_15_v1@desktop@use", sample)
        # Grounding adapter ships ``user_prompt_template=GROUNDING_USER_PROMPT``
        # as a class-level default, so the unrolled step is
        # [user(template), user(image), assistant]. Find the assistant by role.
        assistant_msg = next(m for m in msgs if m["role"] == "assistant")
        assert "tool_calls" not in assistant_msg
        assert isinstance(assistant_msg["content"], list)
        assert "Action:" in assistant_msg["content"][0]["text"]

    def test_trajectory_adds_system_prompt(self):
        sample = sample_trajectory_two_turns()
        msgs = self._last_step("ui_tars_15_v1@desktop@use", sample)
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"][0]["text"] == "You are a helpful assistant."

    def test_trajectory_user_prompt_with_instruction(self):
        sample = sample_trajectory_two_turns()
        msgs = self._last_step("ui_tars_15_v1@desktop@use", sample)
        user_prompt_msg = msgs[1]
        assert user_prompt_msg["role"] == "user"
        text = user_prompt_msg["content"][0]["text"]
        assert "## User Instruction" in text
        assert "## Action Space" in text
        assert "Thought:" in text

    def test_trajectory_image_separate_from_instruction(self):
        sample = sample_trajectory_two_turns()
        msgs = self._last_step("ui_tars_15_v1@desktop@use", sample)
        img_msg = msgs[2]
        assert img_msg["role"] == "user"
        assert any(item.get("type") == "image" for item in img_msg["content"])
        assert not any(item.get("type") == "text" for item in img_msg["content"])

    # test_grounding_no_system_prompt removed — new GroundingPoint adapter
    # ships user_prompt_template; first message is a templated user, not
    # a raw image-bearing user. Covered by test_grounding_point.py.

    def test_output_has_required_fields(self):
        sample = sample_grounding_point()
        adapter = AgentAdapterRegistry.get("ui_tars_15_v1@desktop@grounding.point")
        out = adapter.unroll(sample)
        assert isinstance(out.processed_images, list)
        assert len(out.processed_images) == len(sample.images)
        assert isinstance(out.steps, list)
        assert out.steps and len(out.steps[-1]) > 0

    def test_trajectory_long_windowing(self):
        """With 6 turns and full_history_size=5, old turns should have no images."""
        sample = sample_trajectory_long(num_turns=6)
        msgs = self._last_step("ui_tars_15_v1@desktop@use", sample)
        assert len(msgs) < len(sample.messages) + 2
        assert msgs[0]["role"] == "system"
        user_with_images = [
            m
            for m in msgs
            if m["role"] == "user" and any(i.get("type") == "image" for i in m.get("content", []))
        ]
        assert len(user_with_images) == 5


# =============================================================================
# 6. Grounding bbox/point: tool_calls embedded into content
# =============================================================================


class TestGroundingBBoxPointTextMode:
    """Test that grounding.bbox and grounding.point adapters embed tool_calls into content text."""

    def _last_step(self, key, sample):
        return AgentAdapterRegistry.get(key).unroll(sample).steps[-1]

    def test_bbox_tool_calls_embedded(self):
        sample = sample_grounding_bbox()
        msgs = self._last_step("ui_tars_15_v1@desktop@grounding.bbox", sample)
        assistant_msg = msgs[1]
        assert "tool_calls" not in assistant_msg
        assert assistant_msg["content"][0]["text"] == "Action: click(start_box='(380,450,620,520)')"

    def test_point_tool_calls_embedded(self):
        """point tool_call embedded as UI-TARS-1.5 native click(start_box=) text."""
        sample = sample_grounding_point()
        msgs = self._last_step("ui_tars_15_v1@desktop@grounding.point", sample)
        assistant_msg = next(m for m in msgs if m["role"] == "assistant")
        assert "tool_calls" not in assistant_msg
        text = assistant_msg["content"][0]["text"]
        assert "click" in text  # lite point → UI-TARS click(start_box=)

    def test_understanding_passes_through(self):
        sample = sample_understanding()
        msgs = self._last_step("ui_tars_15_v1@desktop@understanding", sample)
        assert msgs == sample.messages


# =============================================================================
# 7. Trajectory Unroll
# =============================================================================


class TestTrajectoryUnroll:
    """Test ``unroll`` for v1 trajectory adapter."""

    def setup_method(self):
        self.adapter = AgentAdapterRegistry.get("ui_tars_15_v1@desktop@use")

    def test_two_turns_yields_two_steps(self):
        sample = sample_trajectory_two_turns()
        steps = self.adapter.unroll(sample).steps
        assert len(steps) == 2

    def test_unrolled_steps_have_system_prompt(self):
        sample = sample_trajectory_two_turns()
        steps = self.adapter.unroll(sample).steps
        for step in steps:
            assert step[0]["role"] == "system"

    def test_unrolled_assistant_messages_are_text_format(self):
        sample = sample_trajectory_two_turns()
        steps = self.adapter.unroll(sample).steps
        for step in steps:
            for msg in step:
                if msg.get("role") == "assistant":
                    assert "tool_calls" not in msg
                    assert isinstance(msg["content"], list)
                    assert "Action:" in msg["content"][0]["text"]

    def test_long_trajectory(self):
        sample = sample_trajectory_long(num_turns=6)
        steps = self.adapter.unroll(sample).steps
        assert len(steps) == 6

    def test_empty_messages_returns_empty(self):
        sample = dataclasses.replace(sample_trajectory_two_turns(), messages=[])
        assert self.adapter.unroll(sample).steps == []


# =============================================================================
# 8. Batched conversion
# =============================================================================


class TestPerSampleConversion:
    """Test ``unroll(sample).steps``."""

    def test_trajectory_one_row_two_turns(self):
        adapter = AgentAdapterRegistry.get("ui_tars_15_v1@desktop@use")
        sample = sample_trajectory_two_turns()
        out = adapter.unroll(sample)
        assert len(out.steps) == 2

    def test_grounding_one_row_one_step(self):
        adapter = AgentAdapterRegistry.get("ui_tars_15_v1@desktop@grounding.point")
        sample = sample_grounding_point()
        out = adapter.unroll(sample)
        assert out.steps and len(out.steps[-1]) > 0


# =============================================================================
# 8b. valid_actions field (GUI-only gate on the rendered action-space block)
# =============================================================================


class TestValidActions:
    """Neither ``valid_actions`` nor the active extra tools filter the rendered
    ``## Action Space`` block — every byte of it is SFT text."""

    KEY = "ui_tars_15_v1@desktop@use"

    @staticmethod
    def _user_prompt_text(adapter, sample):
        msgs = adapter.unroll(sample).steps[-1]
        assert msgs[1]["role"] == "user"
        return msgs[1]["content"][0]["text"]

    def test_default_field_is_none(self):
        adapter = AgentAdapterRegistry.get(self.KEY)
        assert adapter.metadata.valid_actions is None

    def test_default_unrolls_byte_exact_to_unfiltered(self):
        sample = sample_trajectory_two_turns()
        a_default = AgentAdapterRegistry.get(self.KEY)
        a_explicit_none = AgentAdapterRegistry.get(
            self.KEY, metadata=LiteCUAMetadata(dims=("desktop", "use"), valid_actions=None)
        )
        assert self._user_prompt_text(a_default, sample) == self._user_prompt_text(
            a_explicit_none, sample
        )

    def _action_space_block(self, **metadata_kwargs):
        """The substituted ``{action_space}`` block, isolated from the template."""
        adapter = AgentAdapterRegistry.get(
            self.KEY,
            metadata=LiteCUAMetadata(dims=("desktop", "use"), **metadata_kwargs),
        )
        text = self._user_prompt_text(adapter, sample_trajectory_two_turns())
        return text.split("## Action Space", 1)[1].split("## Note", 1)[0]

    @pytest.mark.parametrize(
        "valid_actions",
        [[], ["click"], ["click", "drag", "key", "type", "scroll", "wait"]],
    )
    def test_valid_actions_never_changes_the_action_space_block(self, valid_actions):
        """The GUI gate narrows what the env ADMITS, not what the model is shown.

        The behaviour this replaces dropped every row outside the gate's image:
        ``valid_actions=["click"]`` left only the three click variants, and the
        finish rows went at EVERY gate, because the env config boundary rejects
        a standalone tool name in ``valid_actions``.
        """
        block = self._action_space_block(valid_actions=valid_actions)
        assert block == self._action_space_block()
        assert "finished(content='xxx')" in block
        assert "call_user()" in block

    def test_gate_leaves_the_whole_user_prompt_byte_identical(self):
        sample = sample_trajectory_two_turns()
        unfiltered = self._user_prompt_text(AgentAdapterRegistry.get(self.KEY), sample)
        gated = self._user_prompt_text(
            AgentAdapterRegistry.get(self.KEY, metadata=LiteCUAMetadata(valid_actions=["click"])),
            sample,
        )
        assert gated == unfiltered

    @pytest.mark.parametrize(
        "finish_tools",
        [[], ["terminate"], ["response"], ["response", "terminate"]],
    )
    def test_active_extra_tools_never_change_the_action_space_block(self, finish_tools):
        """The trained blob renders WHOLE for every combination of active extra
        tools — the negative half of the contract.

        Trimming it to the finish FORM the env can accept was measured worse on
        lite.osworld: advertising only bare ``finished()`` under
        ``extra_tools: ["terminate"]`` removed 16 inactive-tool rejections but
        collapsed finish attempts from 19 to 2, leaving 5 of 8 trajectories to
        burn ``max_steps``. ``finished(content='xxx')`` is the form UI-TARS was
        trained on, so it is the form that makes the model finish at all.
        """
        block = self._action_space_block(
            extra_tool_schemas=[LiteFinishToolSet.get_tool_schema(name) for name in finish_tools],
        )
        assert block == self._action_space_block()
        assert "finished(content='xxx')" in block
        assert "call_user()" in block


# =============================================================================
# 9. Characterization tests — UI-TARS 1.5 v1 data-flow goldens
# =============================================================================
#
# Goldens below were captured by running the current cua-lite code (adapter +
# action_space + protocol) on each UI-TARS 1.5 v1 action shape and recording
# the observed output verbatim. They lock in CURRENT behavior so a planned
# refactor trips if anything changes.
#
# # OBSERVED notes document round-trip lossiness exhibited by the current
# code. They are NOT bugs being xfail'd — they are the observed behavior of
# the current pipeline captured live.

_FINISH_TOOL_SCHEMAS = [
    LiteFinishToolSet.get_tool_schema("response"),
    LiteFinishToolSet.get_tool_schema("terminate"),
]


def _finish_metadata(platform: str = "desktop") -> LiteCUAMetadata:
    return LiteCUAMetadata(dims=(platform, "use"), extra_tool_schemas=list(_FINISH_TOOL_SCHEMAS))


def _desktop_use_adapter() -> UITars15V1DesktopUseAdapter:
    return UITars15V1DesktopUseAdapter(metadata=_finish_metadata("desktop"))


def _mobile_use_adapter() -> UITars15V1MobileUseAdapter:
    return UITars15V1MobileUseAdapter(metadata=_finish_metadata("mobile"))


# ----- Desktop raw-wire goldens (one representative raw per action) ---------
_DESKTOP_RAW: dict[str, str] = {
    "click": ("Thought: Click the button.\nAction: click(start_box='(500,300)')"),
    "left_double": ("Thought: Double click.\nAction: left_double(start_box='(400,400)')"),
    "right_single": ("Thought: Right click.\nAction: right_single(start_box='(300,200)')"),
    "drag": ("Thought: Drag across.\nAction: drag(start_box='(100,200)', end_box='(700,200)')"),
    "hotkey": ("Thought: Copy text.\nAction: hotkey(key='ctrl c')"),
    "type": ("Thought: Enter text.\nAction: type(content='hello world')"),
    "scroll_up": ("Thought: Scroll up.\nAction: scroll(start_box='(500,500)', direction='up')"),
    "wait": ("Thought: Wait a sec.\nAction: wait()"),
    "finished": ("Thought: Task done.\nAction: finished(content='result text')"),
    "finished_no_content": ("Thought: Task done.\nAction: finished()"),
    "call_user": ("Thought: Need help.\nAction: call_user()"),
}

# ----- Mobile raw-wire goldens ---------------------------------------------
_MOBILE_RAW: dict[str, str] = {
    "click": ("Thought: Tap the icon.\nAction: click(start_box='(500,500)')"),
    "long_press": ("Thought: Long press.\nAction: long_press(start_box='(100,200)', time='2')"),
    "type": ("Thought: Enter text.\nAction: type(content='hello')"),
    "scroll": ("Thought: Swipe up.\nAction: scroll(start_box='(500,800)', end_box='(500,200)')"),
    "press_home": ("Thought: Go home.\nAction: press_home()"),
    "press_back": ("Thought: Go back.\nAction: press_back()"),
    "finished": ("Thought: Done.\nAction: finished(content='42')"),
    "finished_no_content": ("Thought: Done.\nAction: finished()"),
}


# Trajectory builder shared across the adapter families (raw -> parse ->
# convert_from_agent). v1 covers both desktop + mobile via the ``platform`` kwarg.
_build_lite_trajectory = build_lite_trajectory


# =============================================================================
# 9.1. Raw -> parse -> LiteMessage fields (desktop + mobile)
# =============================================================================


class TestRawToLiteMessage:
    """For each action, the LiteMessage (post parse + convert_from_agent) has
    the expected canonical cua-lite fields."""

    desktop = _desktop_use_adapter()
    mobile = _mobile_use_adapter()

    @pytest.mark.parametrize("action", list(_DESKTOP_RAW.keys()))
    def test_desktop_parse_produces_assistant(self, action):
        raw = _DESKTOP_RAW[action]
        parsed = self.desktop.parse_raw_assistant_response(raw)
        lite = self.desktop.convert_message_from_agent(parsed)
        assert lite["role"] == "assistant"
        # OBSERVED: v1 never emits reasoning_content from a Thought:/Action: pair
        # when the Thought text is a single line (ui_tars_15_v1.py:499-510).
        assert "reasoning_content" not in lite
        assert len(lite["tool_calls"]) == 1

    @pytest.mark.parametrize("action", list(_MOBILE_RAW.keys()))
    def test_mobile_parse_produces_assistant(self, action):
        raw = _MOBILE_RAW[action]
        parsed = self.mobile.parse_raw_assistant_response(raw)
        lite = self.mobile.convert_message_from_agent(parsed)
        assert lite["role"] == "assistant"
        assert "reasoning_content" not in lite
        assert len(lite["tool_calls"]) == 1

    def test_desktop_click_lite_fields(self):
        lite = self.desktop.convert_message_from_agent(
            self.desktop.parse_raw_assistant_response(_DESKTOP_RAW["click"]),
        )
        action = _single_computer_action(lite)
        assert action["action"] == "click"
        assert action["coordinate"] == [500, 300]

    def test_desktop_left_double_lite_fields(self):
        lite = self.desktop.convert_message_from_agent(
            self.desktop.parse_raw_assistant_response(_DESKTOP_RAW["left_double"]),
        )
        action = _single_computer_action(lite)
        assert action["action"] == "click"
        assert action["clicks"] == 2
        assert action["coordinate"] == [400, 400]

    def test_desktop_right_single_lite_fields(self):
        lite = self.desktop.convert_message_from_agent(
            self.desktop.parse_raw_assistant_response(_DESKTOP_RAW["right_single"]),
        )
        action = _single_computer_action(lite)
        assert action["action"] == "click"
        assert action["button"] == "right"

    def test_desktop_drag_lite_fields(self):
        lite = self.desktop.convert_message_from_agent(
            self.desktop.parse_raw_assistant_response(_DESKTOP_RAW["drag"]),
        )
        args = _single_computer_action(lite)
        assert args["action"] == "drag"
        assert args["start_coordinate"] == [100, 200]
        assert args["coordinate"] == [700, 200]

    def test_desktop_hotkey_lite_fields(self):
        lite = self.desktop.convert_message_from_agent(
            self.desktop.parse_raw_assistant_response(_DESKTOP_RAW["hotkey"]),
        )
        action = _single_computer_action(lite)
        assert action["action"] == "key"
        # OBSERVED (lite/agents/models/ui_tars_15_v1/action_space.py:299): hotkey key
        # string is split on whitespace into a list of keys.
        assert action["keys"] == ["ctrl", "c"]

    def test_desktop_type_lite_fields(self):
        lite = self.desktop.convert_message_from_agent(
            self.desktop.parse_raw_assistant_response(_DESKTOP_RAW["type"]),
        )
        action = _single_computer_action(lite)
        assert action["action"] == "type"
        assert action["text"] == "hello world"

    def test_desktop_scroll_lite_fields(self):
        lite = self.desktop.convert_message_from_agent(
            self.desktop.parse_raw_assistant_response(_DESKTOP_RAW["scroll_up"]),
        )
        args = _single_computer_action(lite)
        assert args["action"] == "scroll"
        assert args["direction"] == "up"
        # OBSERVED (lite/agents/models/ui_tars_15_v1/action_space.py:309): scroll
        # reverse path injects amount=5 (not present in the raw).
        assert args["amount"] == 5

    def test_desktop_wait_lite_fields(self):
        lite = self.desktop.convert_message_from_agent(
            self.desktop.parse_raw_assistant_response(_DESKTOP_RAW["wait"]),
        )
        action = _single_computer_action(lite)
        assert action["action"] == "wait"
        # OBSERVED (lite/agents/models/ui_tars_15_v1/action_space.py:314): wait reverse
        # path injects duration=5.
        assert action["duration"] == 5

    def test_desktop_finished_with_content_becomes_response(self):
        lite = self.desktop.convert_message_from_agent(
            self.desktop.parse_raw_assistant_response(_DESKTOP_RAW["finished"]),
        )
        tc = lite["tool_calls"][0]
        assert tool_call_name(tc) == "response"
        assert tool_call_arguments(tc)["text"] == "result text"

    def test_desktop_finished_without_content_becomes_terminate(self):
        lite = self.desktop.convert_message_from_agent(
            self.desktop.parse_raw_assistant_response(_DESKTOP_RAW["finished_no_content"]),
        )
        tc = lite["tool_calls"][0]
        assert tool_call_name(tc) == "terminate"
        assert tool_call_arguments(tc)["status"] == "success"

    def test_desktop_call_user_becomes_terminate_failure(self):
        lite = self.desktop.convert_message_from_agent(
            self.desktop.parse_raw_assistant_response(_DESKTOP_RAW["call_user"]),
        )
        tc = lite["tool_calls"][0]
        # OBSERVED (lite/agents/models/ui_tars_15_v1/action_space.py:322-323):
        # call_user reverses to terminate(status=failure, reason=call_user).
        assert tool_call_name(tc) == "terminate"
        assert tool_call_arguments(tc)["status"] == "failure"
        assert tool_call_arguments(tc)["reason"] == "call_user"

    def test_mobile_click_becomes_tap(self):
        lite = self.mobile.convert_message_from_agent(
            self.mobile.parse_raw_assistant_response(_MOBILE_RAW["click"]),
        )
        action = _single_mobile_action(lite)
        assert action["action"] == "tap"
        assert action["coordinate"] == [500, 500]

    def test_mobile_long_press_lite_fields(self):
        lite = self.mobile.convert_message_from_agent(
            self.mobile.parse_raw_assistant_response(_MOBILE_RAW["long_press"]),
        )
        action = _single_mobile_action(lite)
        assert action["action"] == "long_press"
        # OBSERVED (lite/agents/models/ui_tars_15_v1/action_space.py:549): time string
        # is coerced to float duration.
        assert action["duration"] == 2.0

    def test_mobile_scroll_becomes_swipe(self):
        lite = self.mobile.convert_message_from_agent(
            self.mobile.parse_raw_assistant_response(_MOBILE_RAW["scroll"]),
        )
        args = _single_mobile_action(lite)
        assert args["action"] == "swipe"
        assert args["start_coordinate"] == [500, 800]
        assert args["coordinate"] == [500, 200]

    def test_mobile_press_home_becomes_system_button(self):
        lite = self.mobile.convert_message_from_agent(
            self.mobile.parse_raw_assistant_response(_MOBILE_RAW["press_home"]),
        )
        action = _single_mobile_action(lite)
        assert action["action"] == "system_button"
        assert action["button"] == "Home"

    def test_mobile_press_back_becomes_system_button(self):
        lite = self.mobile.convert_message_from_agent(
            self.mobile.parse_raw_assistant_response(_MOBILE_RAW["press_back"]),
        )
        action = _single_mobile_action(lite)
        assert action["action"] == "system_button"
        assert action["button"] == "Back"


# =============================================================================
# 9.2. LiteMessage -> UI-TARS v1 agent -> raw text
# =============================================================================


class TestLiteMessageToRaw:
    """For each action, the rendered agent message (post convert_to_agent)
    byte-matches the documented expected raw."""

    desktop = _desktop_use_adapter()
    mobile = _mobile_use_adapter()

    # Most actions round-trip exactly; documented-lossy ones render differently.
    _EXPECTED_DESKTOP: dict[str, str] = {
        "click": _DESKTOP_RAW["click"],
        "left_double": _DESKTOP_RAW["left_double"],
        "right_single": _DESKTOP_RAW["right_single"],
        "drag": _DESKTOP_RAW["drag"],
        "hotkey": _DESKTOP_RAW["hotkey"],
        "type": _DESKTOP_RAW["type"],
        "scroll_up": _DESKTOP_RAW["scroll_up"],
        "wait": _DESKTOP_RAW["wait"],
        "finished": _DESKTOP_RAW["finished"],
        "finished_no_content": _DESKTOP_RAW["finished_no_content"],
        # OBSERVED: call_user has no reverse path; LITE terminate(failure,...)
        # forwards to finished() (no content) — lossy collapse.
        "call_user": "Thought: Need help.\nAction: finished()",
    }

    _EXPECTED_MOBILE: dict[str, str] = {
        "click": _MOBILE_RAW["click"],
        # long_press time round-trip is byte-exact: raw string '2' parses as
        # duration=2.0 (float, LiteMobileActionSpace contract), and
        # `compact_number` collapses it back to '2' on convert_to_agent.
        "long_press": _MOBILE_RAW["long_press"],
        "type": _MOBILE_RAW["type"],
        "scroll": _MOBILE_RAW["scroll"],
        "press_home": _MOBILE_RAW["press_home"],
        "press_back": _MOBILE_RAW["press_back"],
        "finished": _MOBILE_RAW["finished"],
        "finished_no_content": _MOBILE_RAW["finished_no_content"],
    }

    @pytest.mark.parametrize("action", list(_DESKTOP_RAW.keys()))
    def test_desktop_render_matches_golden(self, action):
        raw = _DESKTOP_RAW[action]
        parsed = self.desktop.parse_raw_assistant_response(raw)
        lite = self.desktop.convert_message_from_agent(parsed)
        agent_msg = self.desktop.convert_message_to_agent(lite)
        # convert_message_to_agent collapses tool_calls into a single text content part
        assert "tool_calls" not in agent_msg
        assert agent_msg["content"][0]["text"] == self._EXPECTED_DESKTOP[action]

    @pytest.mark.parametrize("action", list(_MOBILE_RAW.keys()))
    def test_mobile_render_matches_golden(self, action):
        raw = _MOBILE_RAW[action]
        parsed = self.mobile.parse_raw_assistant_response(raw)
        lite = self.mobile.convert_message_from_agent(parsed)
        agent_msg = self.mobile.convert_message_to_agent(lite)
        assert "tool_calls" not in agent_msg
        assert agent_msg["content"][0]["text"] == self._EXPECTED_MOBILE[action]


# =============================================================================
# 9.3. Raw round-trip (raw -> parse -> from -> to -> render)
# =============================================================================


class TestRawRoundTrip:
    """Round-trip each action raw through the full conversion pipeline."""

    desktop = _desktop_use_adapter()
    mobile = _mobile_use_adapter()

    # Desktop actions that are exact byte-match.
    _EXACT_DESKTOP = [
        "click",
        "left_double",
        "right_single",
        "drag",
        "hotkey",
        "type",
        "scroll_up",
        "wait",
        "finished",
        "finished_no_content",
    ]

    @pytest.mark.parametrize("action", _EXACT_DESKTOP)
    def test_desktop_exact_round_trip(self, action):
        raw = _DESKTOP_RAW[action]
        parsed = self.desktop.parse_raw_assistant_response(raw)
        lite = self.desktop.convert_message_from_agent(parsed)
        agent_msg = self.desktop.convert_message_to_agent(lite)
        assert agent_msg["content"][0]["text"] == raw

    def test_desktop_scroll_default_direction_preserved(self):
        """scroll(direction='down') round-trip preserves the direction kwarg.
        Previously lossy: default filtering dropped direction
        when it matched the default, causing next-turn context to lose the
        scroll direction. UITars15V1DesktopActionSpace.scroll no longer passes
        _DEFAULTS, so all explicit kwargs are retained byte-exact on re-render."""
        raw = "Thought: Scroll.\nAction: scroll(start_box='(500,500)', direction='down')"
        parsed = self.desktop.parse_raw_assistant_response(raw)
        lite = self.desktop.convert_message_from_agent(parsed)
        agent_msg = self.desktop.convert_message_to_agent(lite)
        assert agent_msg["content"][0]["text"] == raw

    def test_desktop_call_user_is_lossy(self):
        """OBSERVED: call_user reverses to terminate(failure, reason=call_user)
        which has no forward path → forward renders as finished()."""
        raw = _DESKTOP_RAW["call_user"]
        parsed = self.desktop.parse_raw_assistant_response(raw)
        lite = self.desktop.convert_message_from_agent(parsed)
        agent_msg = self.desktop.convert_message_to_agent(lite)
        assert agent_msg["content"][0]["text"] == "Thought: Need help.\nAction: finished()"

    _EXACT_MOBILE = [
        "click",
        "type",
        "scroll",
        "press_home",
        "press_back",
        "finished",
        "finished_no_content",
    ]

    @pytest.mark.parametrize("action", _EXACT_MOBILE)
    def test_mobile_exact_round_trip(self, action):
        raw = _MOBILE_RAW[action]
        parsed = self.mobile.parse_raw_assistant_response(raw)
        lite = self.mobile.convert_message_from_agent(parsed)
        agent_msg = self.mobile.convert_message_to_agent(lite)
        assert agent_msg["content"][0]["text"] == raw

    def test_mobile_long_press_time_preserves_integer_form(self):
        """long_press `time='2'` -> duration=2.0 (float) on parse but renders
        back as `time='2'` thanks to ``compact_number`` on the convert_to_agent
        path (ui_tars_15_v1.py:492)."""
        raw = _MOBILE_RAW["long_press"]
        parsed = self.mobile.parse_raw_assistant_response(raw)
        lite = self.mobile.convert_message_from_agent(parsed)
        agent_msg = self.mobile.convert_message_to_agent(lite)
        assert agent_msg["content"][0]["text"] == raw  # byte-exact round-trip


# =============================================================================
# 9.4. Unroll structure (3-turn trajectory)
# =============================================================================


class TestUnrollStructure:
    """Validate top-level shape of unroll_sample on a 3-turn trajectory."""

    desktop = _desktop_use_adapter()

    def _sample(self):
        return _build_lite_trajectory(
            self.desktop,
            [
                _DESKTOP_RAW["click"],
                _DESKTOP_RAW["type"],
                _DESKTOP_RAW["finished"],
            ],
        )

    def test_num_steps_equals_num_turns(self):
        steps = self.desktop.unroll(self._sample()).steps
        assert len(steps) == 3

    def test_per_step_role_structure(self):
        """OBSERVED: for a 3-turn trajectory (fits in full_history_size=5) the
        template + protocol split the first user message into (text-only,
        image-only). Step i has 2 + 2*(i+1) messages post-system."""
        steps = self.desktop.unroll(self._sample()).steps
        expected_roles = [
            ["system", "user", "user", "assistant"],
            ["system", "user", "user", "assistant", "user", "assistant"],
            ["system", "user", "user", "assistant", "user", "assistant", "user", "assistant"],
        ]
        actual = [[m["role"] for m in step] for step in steps]
        assert actual == expected_roles

    def test_per_step_image_counts(self):
        """UITars keeps all images within the 5-turn window (no summaries)."""
        steps = self.desktop.unroll(self._sample()).steps
        expected = [1, 2, 3]
        actual = []
        for step in steps:
            n = sum(
                1
                for m in step
                for c in (m.get("content") if isinstance(m.get("content"), list) else [])
                if isinstance(c, dict) and c.get("type") == "image"
            )
            actual.append(n)
        assert actual == expected

    def test_all_assistant_msgs_are_text_format(self):
        """Every unrolled assistant message has content as a Thought:/Action:
        single text content part (tool_calls already folded by
        convert_message_to_agent)."""
        steps = self.desktop.unroll(self._sample()).steps
        for step in steps:
            for m in step:
                if m.get("role") != "assistant":
                    continue
                assert "tool_calls" not in m
                text = m["content"][0]["text"]
                assert "Thought:" in text
                assert "Action:" in text


# =============================================================================
# 9.5. Unroll byte-exact target per action
# =============================================================================


class TestUnrollByteExactTargetPerAction:
    """For each action, a 1-turn trajectory's unrolled target renders to the
    action's expected raw (accounting for documented lossy fields)."""

    desktop = _desktop_use_adapter()
    mobile = _mobile_use_adapter()

    @pytest.mark.parametrize("action", list(_DESKTOP_RAW.keys()))
    def test_desktop_target_renders_to_expected(self, action):
        raw = _DESKTOP_RAW[action]
        sample = _build_lite_trajectory(self.desktop, [raw], platform="desktop")
        steps = self.desktop.unroll(sample).steps
        assert len(steps) == 1
        target = [m for m in steps[0] if m.get("role") == "assistant"][-1]
        assert target["content"][0]["text"] == TestLiteMessageToRaw._EXPECTED_DESKTOP[action]

    @pytest.mark.parametrize("action", list(_MOBILE_RAW.keys()))
    def test_mobile_target_renders_to_expected(self, action):
        raw = _MOBILE_RAW[action]
        sample = _build_lite_trajectory(self.mobile, [raw], platform="mobile")
        steps = self.mobile.unroll(sample).steps
        assert len(steps) == 1
        target = [m for m in steps[0] if m.get("role") == "assistant"][-1]
        assert target["content"][0]["text"] == TestLiteMessageToRaw._EXPECTED_MOBILE[action]


# =============================================================================
# 9.6. Unroll leakage check
# =============================================================================


class TestUnrollNoLeakage:
    """Target i's distinguishing info must NOT appear in sample i's prompt."""

    desktop = _desktop_use_adapter()

    _RAW_0 = "Thought: Tap first.\nAction: click(start_box='(111,222)')"
    _RAW_1 = "Thought: type something.\nAction: type(content='UNIQUETEXT1')"
    _RAW_2 = "Thought: All done.\nAction: finished(content='UNIQUETEXT2')"

    def _sample(self):
        return _build_lite_trajectory(
            self.desktop,
            [self._RAW_0, self._RAW_1, self._RAW_2],
        )

    @staticmethod
    def _message_text_blob(msg: dict[str, Any]) -> str:
        parts: list[str] = []
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    parts.append(c.get("text", ""))
        for tc in msg.get("tool_calls", []) or []:
            parts.append(str(tool_call_name(tc)))
            parts.append(str(tool_call_arguments(tc)))
        return "\n".join(parts)

    def test_target_markers_absent_from_prompt(self):
        steps = self.desktop.unroll(self._sample()).steps
        markers = {
            0: "(111,222)",  # coord appears in target 0 only
            1: "UNIQUETEXT1",
            2: "UNIQUETEXT2",
        }
        for i, step in enumerate(steps):
            prompt_msgs = step[:-1]
            blob = "\n".join(self._message_text_blob(m) for m in prompt_msgs)
            assert markers[i] not in blob, f"step {i} leaked marker {markers[i]!r}:\n{blob}"


# =============================================================================
# 9.7. Protocol windowing — full_history_size=5
# =============================================================================


class TestProtocolWindowing:
    """Exercise UITarsHistoryProtocol at the boundary where windowing engages.

    full_history_size=5 (v1 default). Turns beyond the window collapse to
    assistant-only messages (user images dropped). The first user is always
    kept but stripped of images.
    """

    desktop = _desktop_use_adapter()

    def _build_input(self, num_turns: int) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = []
        raw = _DESKTOP_RAW["click"]
        for i in range(num_turns):
            if i == 0:
                msgs.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "index": 0},
                            {"type": "text", "text": "task"},
                        ],
                    }
                )
            else:
                msgs.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "index": i},
                        ],
                    }
                )
            parsed = self.desktop.parse_raw_assistant_response(raw)
            msgs.append(self.desktop.convert_message_from_agent(parsed))
        return msgs

    def _roles_and_counts(self, num_turns: int):
        proto = UITarsHistoryProtocol(full_history_size=5)
        out = proto.process_messages(self._build_input(num_turns))
        roles = [m["role"] for m in out]
        imgs = []
        for m in out:
            c = m.get("content")
            if isinstance(c, list):
                imgs.append(sum(1 for x in c if isinstance(x, dict) and x.get("type") == "image"))
            else:
                imgs.append(0)
        return roles, imgs

    def test_5_turns_fits_window_exactly(self):
        """5 assistant turns == full_history_size=5 → no windowing engages;
        OBSERVED first user keeps its image (no image-stripping)."""
        roles, imgs = self._roles_and_counts(5)
        # 5 user + 5 assistant = 10 messages
        assert roles == ["user", "assistant"] * 5
        # All 5 user messages retain images (no windowing branch taken).
        assert imgs == [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    def test_6_turns_triggers_windowing(self):
        """OBSERVED: 6 assistant turns; window_start=6-5=1. Turn 0 assistant-only
        (user image dropped from turn 0), turns 1-5 keep user(image)+assistant."""
        roles, imgs = self._roles_and_counts(6)
        # First user (text stripped of images) + a0 + 5*(user+assistant)
        assert roles == ["user", "assistant"] * 6
        # OBSERVED: turn 0 user is kept but its image is stripped.
        assert imgs == [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    def test_7_turns_triggers_windowing(self):
        """OBSERVED: 7 turns, window_start=2. Turns 0,1 collapse to
        assistant-only after the first (image-stripped) user bubble."""
        roles, imgs = self._roles_and_counts(7)
        assert roles == [
            "user",  # turn 0 user (text-only)
            "assistant",  # turn 0 assistant
            "assistant",  # turn 1 assistant (no preceding user)
            "user",
            "assistant",  # turn 2
            "user",
            "assistant",  # turn 3
            "user",
            "assistant",  # turn 4
            "user",
            "assistant",  # turn 5
            "user",
            "assistant",  # turn 6
        ]
        assert imgs == [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]


# =============================================================================
# 9.8. Mutation purity — inputs must not be mutated
# =============================================================================


class TestMutationPurity:
    desktop = _desktop_use_adapter()

    def _assistant_lite(self) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "tool_calls": [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [500, 300]}]},
                )
            ],
        }

    def _assistant_agent(self) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "tool_calls": [
                {
                    "name": "click",
                    "arguments": {"start_box": [500, 300]},
                }
            ],
        }

    def test_convert_to_agent_pure(self):
        msg = self._assistant_lite()
        snap = copy.deepcopy(msg)
        _ = self.desktop.convert_message_to_agent(msg)
        assert msg == snap

    def test_convert_from_agent_pure(self):
        msg = self._assistant_agent()
        snap = copy.deepcopy(msg)
        _ = self.desktop.convert_message_from_agent(msg)
        assert msg == snap

    def test_protocol_process_messages_pure(self):
        proto = UITarsHistoryProtocol(full_history_size=5)
        raw = _DESKTOP_RAW["click"]
        msgs: list[dict[str, Any]] = []
        for i in range(3):
            if i == 0:
                msgs.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "index": 0},
                            {"type": "text", "text": "task"},
                        ],
                    }
                )
            else:
                msgs.append({"role": "user", "content": [{"type": "image", "index": i}]})
            parsed = self.desktop.parse_raw_assistant_response(raw)
            msgs.append(self.desktop.convert_message_from_agent(parsed))
        snap = copy.deepcopy(msgs)
        _ = proto.process_messages(msgs)
        assert msgs == snap

    def test_unroll_pure(self):
        sample = _build_lite_trajectory(
            self.desktop,
            [_DESKTOP_RAW["click"], _DESKTOP_RAW["type"]],
        )
        snap_messages = copy.deepcopy(sample.messages)
        snap_images = copy.deepcopy(sample.images)
        _ = self.desktop.unroll(sample)
        assert sample.messages == snap_messages
        assert sample.images == snap_images


# =============================================================================
# 9.9. Sample independence
# =============================================================================


class TestSampleIndependence:
    """unroll twice returns independent step outputs; mutating one does not
    affect the other."""

    desktop = _desktop_use_adapter()

    def _sample(self) -> LiteSample:
        return _build_lite_trajectory(
            self.desktop,
            [_DESKTOP_RAW["click"], _DESKTOP_RAW["type"]],
        )

    def test_two_unroll_calls_are_independent(self):
        sample = self._sample()
        first = self.desktop.unroll(sample).steps
        second = self.desktop.unroll(sample).steps
        # Mutate first's last step's last message.
        first[-1][-1]["content"] = "MUTATED"
        assert second[-1][-1]["content"] != "MUTATED"

    def test_siblings_are_independent(self):
        steps = self.desktop.unroll(self._sample()).steps
        # Mutate step 0's target content.
        steps[0][-1]["content"] = "SAMPLE_0_MUTATED"
        # Step 1's earlier-turn assistant (a copy of step 0's action) must
        # still contain its original "Action:" string.
        earlier = [m for m in steps[1] if m.get("role") == "assistant"][0]
        assert earlier["content"] != "SAMPLE_0_MUTATED"
        assert "Action:" in earlier["content"][0]["text"]


# =============================================================================
# 9.10. Edge cases
# =============================================================================


class TestEdgeCases:
    desktop = _desktop_use_adapter()
    mobile = _mobile_use_adapter()

    def test_empty_messages_unrolls_to_nothing(self):
        empty = LiteSample(
            metadata=LiteCUAMetadata(dims=("desktop", "use"), others={"resolution": [1920, 1080]}),
            messages=[],
            images=[],
        )
        assert self.desktop.unroll(empty).steps == []

    def test_single_turn_trajectory(self):
        """1-turn sample: system + user(text) + user(image) + target assistant."""
        sample = _build_lite_trajectory(self.desktop, [_DESKTOP_RAW["click"]])
        steps = self.desktop.unroll(sample).steps
        assert len(steps) == 1
        roles = [m["role"] for m in steps[0]]
        assert roles == ["system", "user", "user", "assistant"]

    def test_long_trajectory_windowing_kicks_in(self):
        """10-turn sample. full_history_size=5. Step 9 (last) has 10 turns
        in its prompt — windowing keeps 5 most-recent with images."""
        sample = _build_lite_trajectory(
            self.desktop,
            [_DESKTOP_RAW["click"]] * 10,
        )
        steps = self.desktop.unroll(sample).steps
        assert len(steps) == 10
        last = steps[-1]
        n_user_with_image = sum(
            1
            for m in last
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(c.get("type") == "image" for c in m["content"])
        )
        # Windowing retains exactly full_history_size=5 user(image) msgs.
        assert n_user_with_image == 5

    def test_assistant_without_tool_calls_passes_through(self):
        """Assistant message with only ``action_description`` and no
        tool_calls: ``convert_message_to_agent`` (via
        ``format_message_as_text``) renders the raw prose as-is (no
        ``Thought:`` relabel — that prefix is reserved for
        ``InlineReasoningContent``). The reverse path produces structured
        content."""
        msg = {"role": "assistant", "content": [{"type": "action_description", "text": "text"}]}
        agent = self.desktop.convert_message_to_agent(msg)
        assert agent == {
            "role": "assistant",
            "content": [{"type": "text", "text": "text"}],
        }
        back = self.desktop.convert_message_from_agent(agent)
        assert back["role"] == "assistant"
        assert any("text" in (c.get("text", "") or "") for c in back["content"])
        assert "tool_calls" not in back

    def test_mobile_finished_no_content_round_trip_exact(self):
        """Regression: mobile finished() with no content round-trips exactly."""
        raw = _MOBILE_RAW["finished_no_content"]
        parsed = self.mobile.parse_raw_assistant_response(raw)
        lite = self.mobile.convert_message_from_agent(parsed)
        agent = self.mobile.convert_message_to_agent(lite)
        assert agent["content"][0]["text"] == raw


# =============================================================================
# 10. valid_actions field — Mobile adapter
# =============================================================================


class TestValidActionsMobile:
    """v1 mobile mirror: no gate reaches the rendered ``## Action Space`` block.
    Mobile cua-lite names are ``tap``/``swipe``/``system_button``/etc., distinct
    from desktop's ``click``/``drag``/``key``."""

    KEY = "ui_tars_15_v1@mobile@use"

    @staticmethod
    def _user_prompt_text(adapter, sample):
        msgs = adapter.unroll(sample).steps[-1]
        assert msgs[1]["role"] == "user"
        return msgs[1]["content"][0]["text"]

    def test_default_field_is_none(self):
        adapter = AgentAdapterRegistry.get(self.KEY)
        assert adapter.metadata.valid_actions is None

    def test_default_unrolls_byte_exact_to_unfiltered(self):
        sample = _mobile_prompt_only_sample()
        a_default = AgentAdapterRegistry.get(self.KEY)
        a_explicit_none = AgentAdapterRegistry.get(
            self.KEY, metadata=LiteCUAMetadata(dims=("mobile", "use"), valid_actions=None)
        )
        assert self._user_prompt_text(a_default, sample) == self._user_prompt_text(
            a_explicit_none, sample
        )

    def _action_space_block(self, **metadata_kwargs):
        """The substituted ``{action_space}`` block, isolated from the template."""
        adapter = AgentAdapterRegistry.get(
            self.KEY,
            metadata=LiteCUAMetadata(dims=("mobile", "use"), **metadata_kwargs),
        )
        text = self._user_prompt_text(adapter, _mobile_prompt_only_sample())
        return text.split("## Action Space", 1)[1].split("## Note", 1)[0]

    @pytest.mark.parametrize(
        "valid_actions",
        [[], ["tap"], ["system_button"], ["tap", "long_press", "type", "swipe", "system_button"]],
    )
    def test_valid_actions_never_changes_the_action_space_block(self, valid_actions):
        """Mirror of the desktop rule. ``valid_actions=["tap"]`` used to leave
        only the ``click`` row on the wire and ``["system_button"]`` only the two
        press rows; the finish row went at every gate."""
        block = self._action_space_block(valid_actions=valid_actions)
        assert block == self._action_space_block()
        assert "finished(content='')" in block

    def test_gate_leaves_the_whole_user_prompt_byte_identical(self):
        sample = _mobile_prompt_only_sample()
        unfiltered = self._user_prompt_text(AgentAdapterRegistry.get(self.KEY), sample)
        gated = self._user_prompt_text(
            AgentAdapterRegistry.get(self.KEY, metadata=LiteCUAMetadata(valid_actions=["tap"])),
            sample,
        )
        assert gated == unfiltered

    @pytest.mark.parametrize(
        "finish_tools",
        [[], ["terminate"], ["response"], ["response", "terminate"]],
    )
    def test_active_extra_tools_never_change_the_action_space_block(self, finish_tools):
        """The mobile blob renders WHOLE for every combination of active extra
        tools — see the desktop mirror for the measured reason."""
        block = self._action_space_block(
            extra_tool_schemas=[LiteFinishToolSet.get_tool_schema(name) for name in finish_tools],
        )
        assert block == self._action_space_block()
        assert "finished(content='')" in block
