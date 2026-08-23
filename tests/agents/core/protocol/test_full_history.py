"""
Tests for FullHistoryProtocol.

Covers:
  1. Preserves all messages
  2. Deep copy isolation
  3. reasoning_content preservation
  4. Empty input

Run:
    uv run pytest tests/agents/core/protocol/test_full_history.py -v
"""

from __future__ import annotations

from lite_samples import (
    sample_trajectory_long,
    sample_trajectory_with_reasoning,
)

from lite.agents.core.protocol.base import FullHistoryProtocol


class TestFullHistoryProtocol:
    def setup_method(self):
        self.protocol = FullHistoryProtocol()

    def test_preserves_all_messages(self):
        sample = sample_trajectory_long(num_turns=6)
        messages = sample.messages
        out = self.protocol.process_messages(messages)
        assert len(out) == len(messages)

    def test_deep_copy(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
        out = self.protocol.process_messages(messages)
        out[0]["content"][0]["text"] = "MUTATED"
        assert messages[0]["content"][0]["text"] == "Hi"

    def test_reasoning_content_preserved(self):
        sample = sample_trajectory_with_reasoning()
        messages = sample.messages
        out = self.protocol.process_messages(messages)
        assert len(out) == len(messages)
        reasoning_count = sum(1 for m in out if m.get("reasoning_content"))
        assert reasoning_count >= 2

    def test_empty_input(self):
        assert self.protocol.process_messages([]) == []

