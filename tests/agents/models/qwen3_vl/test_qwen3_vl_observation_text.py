"""
Qwen3VL observation text support.

Covers:
  1. Qwen3VLHistoryProtocol keep_text_with_images default policy
  2. Legacy pure-GUI mode: explicit keep_text_with_images=False strips regular
     image+text history text

Run:
    uv run pytest tests/agents/models/qwen3_vl/test_qwen3_vl_observation_text.py -v
"""

from __future__ import annotations

from lite_samples import (
    sample_trajectory_long,
    sample_trajectory_with_obs_text,
)

from lite.agents.models.qwen3_vl.protocol import Qwen3VLHistoryProtocol


class TestQwen3VLObsText:
    def test_obs_text_preserved_by_default(self):
        """By default, windowed user messages keep observation text."""
        sample = sample_trajectory_with_obs_text(num_turns=6)
        protocol = Qwen3VLHistoryProtocol(full_history_size=2)
        out = protocol.process_messages(sample.messages)

        user_msgs = [m for m in out if m.get("role") == "user"]
        last_user = user_msgs[-1]
        text_items = [item for item in last_user["content"] if item.get("type") == "text"]
        assert len(text_items) == 1
        assert "webpage" in text_items[0]["text"] or "Navigation" in text_items[0]["text"]

    def test_obs_text_stripped_in_explicit_legacy_mode(self):
        """With keep_text_with_images=False, windowed image+text user messages are image-only."""
        sample = sample_trajectory_with_obs_text(num_turns=6)
        protocol = Qwen3VLHistoryProtocol(full_history_size=2, keep_text_with_images=False)
        out = protocol.process_messages(sample.messages)

        user_msgs = [m for m in out if m.get("role") == "user"]
        last_user = user_msgs[-1]
        for item in last_user["content"]:
            assert item["type"] == "image"

    def test_obs_text_preserved_when_enabled(self):
        """With keep_text_with_images=True, windowed user messages keep observation text."""
        sample = sample_trajectory_with_obs_text(num_turns=6)
        protocol = Qwen3VLHistoryProtocol(full_history_size=2, keep_text_with_images=True)
        out = protocol.process_messages(sample.messages)

        user_msgs = [m for m in out if m.get("role") == "user"]
        last_user = user_msgs[-1]
        text_items = [item for item in last_user["content"] if item.get("type") == "text"]
        assert len(text_items) == 1
        assert "webpage" in text_items[0]["text"] or "Navigation" in text_items[0]["text"]

    def test_no_obs_text_window_user_messages_image_only(self):
        """Without obs text (OSWorld-style), windowed user messages should remain image-only."""
        sample = sample_trajectory_long(num_turns=6)
        protocol = Qwen3VLHistoryProtocol(full_history_size=2)
        out = protocol.process_messages(sample.messages)

        user_msgs = [m for m in out if m.get("role") == "user"]
        if len(user_msgs) > 1:
            last_user = user_msgs[-1]
            for item in last_user["content"]:
                assert item["type"] == "image"


class TestWebGymObsText:
    def test_obs_text_preserved_by_default(self):
        """By default, windowed user messages keep observation text."""
        sample = sample_trajectory_with_obs_text(num_turns=8)
        protocol = Qwen3VLHistoryProtocol(full_history_size=4)
        out = protocol.process_messages(sample.messages)

        user_msgs = [m for m in out if m.get("role") == "user"]
        has_obs_text = False
        for user_msg in user_msgs[1:]:
            for item in user_msg.get("content", []):
                if item.get("type") == "text" and (
                    "webpage" in item["text"] or "Navigation" in item["text"]
                ):
                    has_obs_text = True
        assert has_obs_text

    def test_obs_text_stripped_in_explicit_legacy_mode(self):
        """With keep_text_with_images=False, windowed user messages are image-only."""
        sample = sample_trajectory_with_obs_text(num_turns=8)
        protocol = Qwen3VLHistoryProtocol(full_history_size=4, keep_text_with_images=False)
        out = protocol.process_messages(sample.messages)

        user_msgs = [m for m in out if m.get("role") == "user"]
        for user_msg in user_msgs[1:]:
            for item in user_msg.get("content", []):
                assert item["type"] == "image"

    def test_obs_text_preserved_when_enabled(self):
        """With keep_text_with_images=True, windowed user messages keep observation text."""
        sample = sample_trajectory_with_obs_text(num_turns=8)
        protocol = Qwen3VLHistoryProtocol(full_history_size=4, keep_text_with_images=True)
        out = protocol.process_messages(sample.messages)

        user_msgs = [m for m in out if m.get("role") == "user"]
        has_obs_text = False
        for user_msg in user_msgs[1:]:
            for item in user_msg.get("content", []):
                if item.get("type") == "text" and (
                    "webpage" in item["text"] or "Navigation" in item["text"]
                ):
                    has_obs_text = True
        assert has_obs_text

    def test_no_obs_text_remains_image_only(self):
        """Without observation text, windowed user messages remain image-only."""
        sample = sample_trajectory_long(num_turns=8)
        protocol = Qwen3VLHistoryProtocol(full_history_size=4)
        out = protocol.process_messages(sample.messages)

        user_msgs = [m for m in out if m.get("role") == "user"]
        if len(user_msgs) > 1:
            last_user = user_msgs[-1]
            for item in last_user["content"]:
                assert item["type"] == "image"
