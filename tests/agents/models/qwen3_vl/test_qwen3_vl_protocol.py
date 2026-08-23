"""
Tests for Qwen3VLHistoryProtocol (Qwen-style summarized rolling window).

Covers:
  1. Window size reduces messages
  2. Larger window keeps more
  3. Short trajectory (no summarization needed)
  4. Summary text contains instruction and previous actions
  5. reasoning_content preservation in kept turns
  6. Empty / None input

Run:
    uv run pytest tests/agents/models/qwen3_vl/test_qwen3_vl_protocol.py -v
"""

from __future__ import annotations

from lite_samples import (
    sample_trajectory_long,
    sample_trajectory_two_turns,
    sample_trajectory_with_reasoning,
)

from lite.agents.models.qwen3_vl.protocol import Qwen3VLHistoryProtocol


class TestQwen3VLHistoryProtocol:
    def test_window_size_reduces_messages(self):
        sample = sample_trajectory_long(num_turns=6)
        messages = sample.messages
        assert len(messages) == 12

        p1 = Qwen3VLHistoryProtocol(full_history_size=1)
        p2 = Qwen3VLHistoryProtocol(full_history_size=2)
        out1 = p1.process_messages(messages)
        out2 = p2.process_messages(messages)

        assert len(out1) < len(out2)
        assert len(out1) >= 2
        assert len(out2) >= 4

    def test_larger_window_keeps_more(self):
        sample = sample_trajectory_long(num_turns=6)
        messages = sample.messages
        out2 = Qwen3VLHistoryProtocol(full_history_size=2).process_messages(messages)
        out3 = Qwen3VLHistoryProtocol(full_history_size=3).process_messages(messages)
        assert len(out3) >= len(out2)

    def test_short_trajectory_no_truncation(self):
        """With 2 turns and full_history_size=4, no summarization should happen."""
        sample = sample_trajectory_two_turns()
        messages = sample.messages
        out = Qwen3VLHistoryProtocol(full_history_size=4).process_messages(messages)
        # Should still have user+assistant pairs, just normalized first user message
        assert len(out) == len(messages)

    def test_summary_text_contains_instruction(self):
        """When summarized, first user message should contain the original instruction."""
        sample = sample_trajectory_long(num_turns=6)
        messages = sample.messages
        out = Qwen3VLHistoryProtocol(full_history_size=2).process_messages(messages)
        first_user = next(m for m in out if m.get("role") == "user")
        text = ""
        for item in first_user.get("content", []):
            if item.get("type") == "text":
                text = item.get("text", "")
                break
        assert "Instruction:" in text
        assert "Previous actions:" in text

    def test_summary_text_contains_previous_actions(self):
        """Summarized text should mention steps from old turns."""
        sample = sample_trajectory_long(num_turns=6)
        messages = sample.messages
        out = Qwen3VLHistoryProtocol(full_history_size=2).process_messages(messages)
        first_user = next(m for m in out if m.get("role") == "user")
        text = ""
        for item in first_user.get("content", []):
            if item.get("type") == "text":
                text = item.get("text", "")
                break
        assert "Step 1:" in text

    def test_reasoning_content_preserved(self):
        sample = sample_trajectory_with_reasoning()
        messages = sample.messages
        out = Qwen3VLHistoryProtocol(full_history_size=2).process_messages(messages)
        reasoning_count = sum(1 for m in out if m.get("reasoning_content"))
        assert reasoning_count >= 1

    def test_empty_input(self):
        assert Qwen3VLHistoryProtocol().process_messages([]) == []

    # ----- boundary: window == turns, window > turns -----

    def test_window_equals_num_turns(self):
        """When full_history_size == num_turns, no summarization (but first msg normalized)."""
        sample = sample_trajectory_long(num_turns=4)
        messages = sample.messages
        protocol = Qwen3VLHistoryProtocol(full_history_size=4)
        out = protocol.process_messages(messages)
        assert len(out) == len(messages)

    def test_window_greater_than_num_turns(self):
        """When full_history_size > num_turns, all messages preserved (normalized)."""
        sample = sample_trajectory_long(num_turns=3)
        messages = sample.messages
        protocol = Qwen3VLHistoryProtocol(full_history_size=10)
        out = protocol.process_messages(messages)
        assert len(out) == len(messages)

    def test_zero_window_keeps_no_images(self):
        """``full_history_size=0`` means keep NONE, like every sibling protocol.

        A window start of ``image_indices[-full_history_size]`` wraps at 0 to
        ``image_indices[0]`` and keeps the WHOLE trajectory -- the exact inverse
        of the knob.
        """

        messages = sample_trajectory_long(num_turns=6).messages
        out = Qwen3VLHistoryProtocol(full_history_size=0).process_messages(messages)

        images = [
            part
            for m in out
            for part in (m.get("content") or [])
            if isinstance(part, dict) and part.get("type") == "image"
        ]
        assert images == []
        assert len(out) == len(
            Qwen3VLHistoryProtocol(full_history_size=0).process_messages(messages)
        )

    # ----- minimal cases -----

    def test_single_turn(self):
        """Single turn should pass through (normalized)."""
        sample = sample_trajectory_long(num_turns=1)
        messages = sample.messages
        protocol = Qwen3VLHistoryProtocol(full_history_size=2)
        out = protocol.process_messages(messages)
        assert len(out) == len(messages)

    # ----- system message -----

    def test_system_message_preserved(self):
        """System message should be preserved at the start."""
        sample = sample_trajectory_long(num_turns=6)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            *sample.messages,
        ]
        protocol = Qwen3VLHistoryProtocol(full_history_size=2)
        out = protocol.process_messages(messages)
        assert out[0]["role"] == "system"
        assert out[0]["content"][0]["text"] == "You are a helpful assistant."

    # ----- incomplete last turn -----

    def test_incomplete_last_turn(self):
        """Last turn with user but no assistant should be handled."""
        sample = sample_trajectory_long(num_turns=4)
        messages = sample.messages
        messages.append(
            {
                "role": "user",
                "content": [{"type": "image", "index": 99}],
            }
        )
        protocol = Qwen3VLHistoryProtocol(full_history_size=2)
        out = protocol.process_messages(messages)
        assert out[-1]["role"] == "user"

    # ----- summary content verification -----

    def test_no_previous_actions_when_no_truncation(self):
        """When no turns are truncated, previous_actions should be 'None'."""
        sample = sample_trajectory_two_turns()
        messages = sample.messages
        out = Qwen3VLHistoryProtocol(full_history_size=4).process_messages(messages)
        first_user = next(m for m in out if m.get("role") == "user")
        text = ""
        for item in first_user.get("content", []):
            if item.get("type") == "text":
                text = item.get("text", "")
                break
        assert "Previous actions:\nNone" in text
