"""
Tests for UI-TARS (original) adapters.

Covers:
  1. Registry: correct adapter classes for all ui_tars keys
  2. System prompt and user prompt template
  3. parse_raw_assistant_response: sanity check (trivial text wrapper)
  4. Full pipeline: parse_raw → convert_message_from_agent
  5. convert_sample_to_agent: tool_calls to text conversion
  6. Grounding bbox/point: tool_calls embedded into content
  7. Trajectory unroll
  8. Batched conversion

Run:
    uv run pytest tests/agents/models/ui_tars/test_ui_tars_adapter.py -v
"""

from __future__ import annotations

import dataclasses

import pytest
from lite_samples import (
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
from lite.agents.models.ui_tars.adapter import (
    USE_SYSTEM_PROMPT,
    USE_USER_PROMPT,
    UITarsDesktopGroundingPointAdapter,
    UITarsDesktopUseAdapter,
    UITarsGroundingBBoxAdapter,
    UITarsMobileGroundingPointAdapter,
    UITarsMobileUseAdapter,
)
from lite.core import LiteCUAMetadata
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.extra_tools import LiteFinishToolSet


def _single_computer_action(message):
    call = message["tool_calls"][0]
    assert tool_call_name(call) == "computer"
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


# =============================================================================
# 1. Registry
# =============================================================================


class TestRegistry:
    """Test adapter registry resolution for ui_tars keys."""

    @pytest.mark.parametrize(
        "key,expected_cls",
        [
            ("ui_tars@desktop@grounding.point", UITarsDesktopGroundingPointAdapter),
            ("ui_tars@desktop@use", UITarsDesktopUseAdapter),
            ("ui_tars@browser@grounding.point", UITarsDesktopGroundingPointAdapter),
            ("ui_tars@browser@use", UITarsDesktopUseAdapter),
            ("ui_tars@mobile@grounding.point", UITarsMobileGroundingPointAdapter),
            ("ui_tars@mobile@use", UITarsMobileUseAdapter),
            ("ui_tars@desktop@grounding.bbox", UITarsGroundingBBoxAdapter),
            ("ui_tars@browser@grounding.bbox", UITarsGroundingBBoxAdapter),
        ],
    )
    def test_registry_returns_correct_class(self, key, expected_cls):
        adapter = AgentAdapterRegistry.get(key)
        assert isinstance(adapter, BaseAgentAdapter)
        assert type(adapter) is expected_cls

    def test_understanding_is_pass_through(self):
        for key in [
            "ui_tars@desktop@understanding",
            "ui_tars@browser@understanding",
        ]:
            adapter = AgentAdapterRegistry.get(key)
            assert type(adapter) is AsIsAdapter, f"Expected AsIsAdapter for {key}"

    def test_trajectory_has_system_prompt(self):
        adapter = AgentAdapterRegistry.get("ui_tars@desktop@use")
        assert adapter.system_prompt == USE_SYSTEM_PROMPT


# =============================================================================
# 2. System Prompt
# =============================================================================


class TestPrompts:
    """Test UITars prompt structure."""

    def test_system_prompt_is_simple(self):
        assert USE_SYSTEM_PROMPT == "You are a helpful assistant."

    def test_user_prompt_has_thought_action_format(self):
        assert "Thought:" in USE_USER_PROMPT
        assert "Action:" in USE_USER_PROMPT

    def test_user_prompt_has_placeholders(self):
        assert "{action_space}" in USE_USER_PROMPT
        assert "{instruction}" in USE_USER_PROMPT

    def test_user_prompt_has_action_space_placeholder(self):
        """UITars user prompt has {action_space} which will contain box tokens at runtime."""
        assert "{action_space}" in USE_USER_PROMPT


# =============================================================================
# 3. parse_raw_assistant_response (trivial wrapper sanity check)
# =============================================================================


class TestParseRawAssistantResponse:
    """parse_raw_assistant_response returns a trivial text wrapper.

    All structured parsing (Thought/Action, tool_calls) is in
    convert_message_from_agent. This class has a single sanity check.
    """

    def setup_method(self):
        self.adapter = AgentAdapterRegistry.get("ui_tars@desktop@use")

    def test_returns_trivial_text_wrapper(self):
        raw = (
            "Thought: Click the search bar.\n"
            "Action: click(start_box='<|box_start|>(500,100)<|box_end|>')"
        )
        result = self.adapter.parse_raw_assistant_response(raw)
        assert result == {
            "role": "assistant",
            "content": [{"type": "text", "text": raw}],
        }


# =============================================================================
# 4. Full pipeline: parse → convert_from_agent
# =============================================================================


class TestFullParsePipeline:
    """Test parse_raw_assistant_response → convert_message_from_agent chain."""

    def setup_method(self):
        self.adapter = AgentAdapterRegistry.get("ui_tars@desktop@use")

    def _parse(self, raw: str):
        return self.adapter.convert_message_from_agent(
            self.adapter.parse_raw_assistant_response(raw)
        )

    def test_click(self):
        result = self._parse(
            "Thought: Click the button.\n"
            "Action: click(start_box='<|box_start|>(500,300)<|box_end|>')"
        )
        assert result["role"] == "assistant"
        action = _single_computer_action(result)
        assert action["action"] == "click"
        assert action["coordinate"] == [500, 300]

    def test_multi_line_thought_all_goes_to_inline_reasoning(self):
        """Full ``Thought:`` text (including any trailing summary line) maps
        to a single ``InlineReasoningContent`` part — UI-TARS's SFT slot
        is prompted CoT, not action description. Structured action lives
        in ``tool_calls``."""
        result = self._parse(
            "Thought: I need to find the settings icon.\n"
            "Click the settings icon.\n"
            "Action: click(start_box='<|box_start|>(300,200)<|box_end|>')"
        )
        reasoning = next(
            (c["text"] for c in result["content"] if c.get("type") == "inline_reasoning"),
            None,
        )
        assert reasoning == ("I need to find the settings icon.\nClick the settings icon.")
        assert not any(c.get("type") == "action_description" for c in result["content"])

    def test_box_tokens_stripped(self):
        """Box tokens should be stripped before coordinate parsing."""
        result = self._parse(
            "Thought: Click.\nAction: click(start_box='<|box_start|>(700,400)<|box_end|>')"
        )
        assert _single_computer_action(result)["coordinate"] == [700, 400]

    def test_finished_action(self):
        result = self._parse("Thought: Done.\nAction: finished()")
        tc = result["tool_calls"][0]
        assert tool_call_name(tc) == "terminate"
        assert tool_call_arguments(tc)["status"] == "success"

    def test_wait_action(self):
        result = self._parse("Thought: Wait for load.\nAction: wait()")
        action = _single_computer_action(result)
        assert action["action"] == "wait"
        assert action["duration"] == 5

    def test_hotkey_action(self):
        result = self._parse("Thought: Copy text.\nAction: hotkey(key='ctrl c')")
        action = _single_computer_action(result)
        assert action["action"] == "key"
        assert action["keys"] == ["ctrl", "c"]

    def test_type_action(self):
        result = self._parse("Thought: type search query.\nAction: type(content='hello world')")
        action = _single_computer_action(result)
        assert action["action"] == "type"
        assert action["text"] == "hello world"

    def test_empty_response(self):
        result = self._parse("")
        assert result["role"] == "assistant"
        assert "tool_calls" not in result

    def test_no_action(self):
        raw = "Thought: I'm thinking..."
        result = self._parse(raw)
        assert "tool_calls" not in result
        # No tool_call means terminal prose, not an action turn. Even a
        # protocol-looking ``Thought:`` line must stay visible to
        # no_tool_call_final_text instead of being hidden as inline reasoning.
        assert result["content"] == [{"type": "text", "text": raw}]


# =============================================================================
# 5. convert_sample_to_agent
# =============================================================================


class TestConvertSampleToAgent:
    """Last-step view of ``unroll(sample)``."""

    def _last_step(self, key, sample):
        adapter = AgentAdapterRegistry.get(key)
        return adapter.unroll(sample).steps[-1]

    def test_converts_tool_calls_to_text(self):
        # Multi-action SFT-style sample (click + type) — exercises the
        # navigation adapter's full action vocabulary.
        sample = sample_grounding_action_desktop()
        msgs = self._last_step("ui_tars@desktop@use", sample)
        assistant_msg = next(m for m in msgs if m["role"] == "assistant")
        assert "tool_calls" not in assistant_msg
        assert assistant_msg["content"] == [
            {"type": "text", "text": assistant_msg["content"][0]["text"]}
        ]
        assert "Action:" in assistant_msg["content"][0]["text"]

    def test_trajectory_adds_system_prompt(self):
        sample = sample_trajectory_two_turns()
        msgs = self._last_step("ui_tars@desktop@use", sample)
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"][0]["text"] == "You are a helpful assistant."

    def test_trajectory_user_prompt_with_instruction(self):
        sample = sample_trajectory_two_turns()
        msgs = self._last_step("ui_tars@desktop@use", sample)
        user_prompt_msg = msgs[1]
        assert user_prompt_msg["role"] == "user"
        text = user_prompt_msg["content"][0]["text"]
        assert "Thought:" in text
        assert "Action:" in text

    def test_trajectory_image_separate_from_instruction(self):
        sample = sample_trajectory_two_turns()
        msgs = self._last_step("ui_tars@desktop@use", sample)
        img_msg = msgs[2]
        assert img_msg["role"] == "user"
        assert any(item.get("type") == "image" for item in img_msg["content"])
        assert not any(item.get("type") == "text" for item in img_msg["content"])

    # ``test_grounding_no_system_prompt`` removed — the new
    # GroundingPointAdapter ships ``user_prompt_template=GROUNDING_USER_PROMPT``
    # so the unrolled step starts with a templated user message (not a raw
    # image-bearing one). Behaviour now covered in ``test_grounding_point.py``.

    def test_output_has_required_fields(self):
        sample = sample_grounding_point()
        adapter = AgentAdapterRegistry.get("ui_tars@desktop@grounding.point")
        out = adapter.unroll(sample)
        assert isinstance(out.processed_images, list)
        assert len(out.processed_images) == len(sample.images)
        assert isinstance(out.steps, list)
        assert out.steps and len(out.steps[-1]) > 0

    def test_trajectory_long_windowing(self):
        """With 6 turns and full_history_size=5, old turns should have no images."""
        sample = sample_trajectory_long(num_turns=6)
        msgs = self._last_step("ui_tars@desktop@use", sample)
        # Output should have fewer messages than input (windowing removes old user msgs)
        assert len(msgs) < len(sample.messages) + 2  # +2 for system + user_prompt
        # Should still have system prompt and user prompt
        assert msgs[0]["role"] == "system"
        # Window images = 5, so 5 user(image) messages in window
        user_with_images = [
            m
            for m in msgs
            if m["role"] == "user" and any(i.get("type") == "image" for i in m.get("content", []))
        ]
        assert len(user_with_images) == 5  # full_history_size=5


# =============================================================================
# 6. Grounding bbox/point: tool_calls embedded into content
# =============================================================================


class TestGroundingBBoxPointTextMode:
    """Test that grounding.bbox and grounding.point adapters embed tool_calls into content text."""

    def _last_step(self, key, sample):
        adapter = AgentAdapterRegistry.get(key)
        return adapter.unroll(sample).steps[-1]

    def test_bbox_tool_calls_embedded(self):
        """bbox tool_call embedded as the UI-TARS wire ``click(start_box=)`` text.

        Sample carries a lite ``bbox(coordinate=)`` call; the action space
        converts it to the four-value box spelling the family's parser reads
        back, so the rendered turn round-trips into model context.
        """
        sample = sample_grounding_bbox()
        msgs = self._last_step("ui_tars@desktop@grounding.bbox", sample)
        assistant_msg = msgs[1]
        assert "tool_calls" not in assistant_msg
        text = assistant_msg["content"][0]["text"]
        assert text == "Action: click(start_box='(380,450,620,520)')"

    def test_point_tool_calls_embedded(self):
        """point tool_call should be embedded as text via the grounding harness.

        Sample carries a lite ``point(coord)`` call; the action_space converts
        to UI-TARS native ``click(start_box=)`` text. Find the assistant by
        role since the unrolled step is [user(template), user(image), assistant].
        """
        sample = sample_grounding_point()
        msgs = self._last_step("ui_tars@desktop@grounding.point", sample)
        assistant_msg = next(m for m in msgs if m["role"] == "assistant")
        assert "tool_calls" not in assistant_msg
        text = assistant_msg["content"][0]["text"]
        assert "click" in text  # converted from lite point → UI-TARS click(start_box=)

    def test_understanding_passes_through(self):
        """Understanding samples should pass through unchanged (AsIsAdapter)."""
        sample = sample_understanding()
        msgs = self._last_step("ui_tars@desktop@understanding", sample)
        assert msgs == sample.messages


# =============================================================================
# 7. Trajectory Unroll
# =============================================================================


class TestTrajectoryUnroll:
    """Test ``unroll`` for UITars trajectory adapter."""

    def setup_method(self):
        self.adapter = AgentAdapterRegistry.get("ui_tars@desktop@use")

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
                    text = msg["content"][0]["text"]
                    assert "Action:" in text

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
        adapter = AgentAdapterRegistry.get("ui_tars@desktop@use")
        sample = sample_trajectory_two_turns()
        out = adapter.unroll(sample)
        assert len(out.steps) == 2

    def test_grounding_one_row_one_step(self):
        adapter = AgentAdapterRegistry.get("ui_tars@desktop@grounding.point")
        sample = sample_grounding_point()
        out = adapter.unroll(sample)
        assert out.steps and len(out.steps[-1]) > 0


# =============================================================================
# 9. valid_actions field (GUI-only gate on the rendered action-space block)
# =============================================================================


class TestValidActions:
    """Neither ``valid_actions`` nor the active extra tools filter the rendered
    ``## Action Space`` block — every byte of it is SFT text."""

    KEY = "ui_tars@desktop@use"

    @staticmethod
    def _user_prompt_text(adapter, sample):
        """Pull the navigation user-prompt text out of the rendered last step."""
        msgs = adapter.unroll(sample).steps[-1]
        # msgs[0] is system, msgs[1] is the navigation user prompt (text only).
        assert msgs[1]["role"] == "user"
        return msgs[1]["content"][0]["text"]

    def test_default_field_is_none(self):
        """Field defaults to None (i.e. filter disabled)."""
        adapter = AgentAdapterRegistry.get(self.KEY)
        assert adapter.metadata.valid_actions is None

    def test_default_unrolls_byte_exact_to_unfiltered(self):
        """Without a valid_actions kwarg, the rendered user prompt must match
        the rendered prompt obtained by explicitly passing ``valid_actions=None``.
        This is the strongest no-regression check we can write at this layer
        — it pins down the new code path's neutrality."""
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
        assert "finished()" in block
        assert "call_user()" in block

    def test_gate_leaves_the_whole_user_prompt_byte_identical(self):
        sample = sample_trajectory_two_turns()
        unfiltered = self._user_prompt_text(
            AgentAdapterRegistry.get(self.KEY),
            sample,
        )
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
        lite.osworld: it removed 16 inactive-tool rejections but collapsed
        finish attempts from 19 to 2, leaving 5 of 8 trajectories to burn
        ``max_steps``. The advertised finish rows are what make the model
        declare completion at all.
        """
        block = self._action_space_block(
            extra_tool_schemas=[LiteFinishToolSet.get_tool_schema(name) for name in finish_tools],
        )
        assert block == self._action_space_block()
        assert "finished()" in block
        assert "call_user()" in block


# =============================================================================
# 10. valid_actions field — Mobile adapter
# =============================================================================


class TestValidActionsMobile:
    """Mirror of :class:`TestValidActions` for the mobile adapter
    (``ui_tars@mobile@use``). Mobile has a different cua-lite
    name set (``tap``, ``swipe``, ``system_button``...) and a different
    native action-space text (no ``call_user``)."""

    KEY = "ui_tars@mobile@use"

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
        the ``click`` row alone on the wire and ``["system_button"]`` only the
        two press rows; the finish row went at every gate."""
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
