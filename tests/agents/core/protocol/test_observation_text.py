"""
Tests for observation text support in history protocols.

Covers:
  1. filter_history_content default policy

Run:
    uv run pytest tests/agents/core/protocol/test_observation_text.py -v
"""

from __future__ import annotations

import copy

from lite.agents.core.protocol.window import (
    filter_history_content,
)

# =============================================================================
# filter_history_content default policy
# =============================================================================

class TestFilterHistoryContent:
    def test_keeps_image_text_and_metadata(self):
        msg = {"role": "user", "content": [
            {"type": "image", "index": 0},
            {"type": "text", "text": "obs"},
            {"type": "metadata", "data": {"page_title": "Google"}},
            {"type": "other", "data": "x"},
        ]}
        out = filter_history_content(msg)
        assert len(out["content"]) == 3
        types = {item["type"] for item in out["content"]}
        assert types == {"image", "text", "metadata"}

    def test_image_only_message_is_passed_through_unchanged(self):
        msg = {"role": "user", "content": [{"type": "image", "index": 0}]}
        out_and = filter_history_content(msg)
        assert out_and["content"] == [{"type": "image", "index": 0}]

    def test_role_tool_keeps_empty_text_output(self):
        msg = {"role": "tool", "tool_call_id": "call_0", "content": [
            {"type": "text", "text": ""},
            {"type": "metadata", "data": {"returncode": 0}},
        ]}
        out = filter_history_content(msg)
        assert out["content"] == [
            {"type": "text", "text": ""},
            {"type": "metadata", "data": {"returncode": 0}},
        ]

    def test_user_observation_still_drops_empty_text(self):
        msg = {"role": "user", "content": [
            {"type": "image", "index": 0},
            {"type": "text", "text": ""},
        ]}
        assert filter_history_content(msg)["content"] == [{"type": "image", "index": 0}]

    def test_does_not_mutate_original(self):
        msg = {"role": "user", "content": [
            {"type": "image", "index": 0},
            {"type": "text", "text": "obs"},
        ]}
        original = copy.deepcopy(msg)
        filter_history_content(msg)
        assert msg == original
