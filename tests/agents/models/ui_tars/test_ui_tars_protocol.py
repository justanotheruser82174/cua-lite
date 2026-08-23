"""
Tests for UITarsHistoryProtocol.

Covers:
  1. Windowing: old turns keep assistant-only, recent turns keep full
  2. First user message preserved (instruction text, no image)
  3. Short trajectory (no windowing)
  4. reasoning_content preservation
  5. Empty / None input

Run:
    uv run pytest tests/agents/models/ui_tars/test_ui_tars_protocol.py -v
"""

from __future__ import annotations

from lite_samples import (
    sample_trajectory_long,
    sample_trajectory_two_turns,
    sample_trajectory_with_reasoning,
)

from lite.agents.models.ui_tars.protocol import UITarsHistoryProtocol
from lite.core.tools import make_tool_call


class TestUITarsHistoryProtocol:
    def test_windowing_reduces_messages(self):
        """With 6 turns and full_history_size=2, output should have fewer messages than input."""
        sample = sample_trajectory_long(num_turns=6)
        messages = sample.messages
        protocol = UITarsHistoryProtocol(full_history_size=2)
        out = protocol.process_messages(messages)
        assert len(out) < len(messages)

    def test_larger_window_keeps_more(self):
        sample = sample_trajectory_long(num_turns=6)
        messages = sample.messages
        out2 = UITarsHistoryProtocol(full_history_size=2).process_messages(messages)
        out3 = UITarsHistoryProtocol(full_history_size=3).process_messages(messages)
        assert len(out3) >= len(out2)

    def test_first_user_message_has_text_no_image(self):
        """First user message should have instruction text, images stripped."""
        sample = sample_trajectory_long(num_turns=6)
        messages = sample.messages
        protocol = UITarsHistoryProtocol(full_history_size=2)
        out = protocol.process_messages(messages)
        first_user = out[0]
        assert first_user["role"] == "user"
        has_text = any(item.get("type") == "text" for item in first_user.get("content", []))
        has_image = any(item.get("type") == "image" for item in first_user.get("content", []))
        assert has_text
        assert not has_image

    def test_old_turns_are_assistant_only(self):
        """Old turns (before window) should only have assistant messages, no user/image."""
        sample = sample_trajectory_long(num_turns=6)
        messages = sample.messages
        protocol = UITarsHistoryProtocol(full_history_size=2)
        out = protocol.process_messages(messages)
        # After first user message, old turns are consecutive assistant messages
        # Count consecutive assistant messages after the first user
        consecutive_assistants = 0
        for msg in out[1:]:
            if msg["role"] == "assistant":
                consecutive_assistants += 1
            else:
                break
        # With 6 turns and window=2, old turns = 4, so 4 consecutive assistant messages
        assert consecutive_assistants == 4

    def test_recent_turns_have_images(self):
        """Recent turns (in window) should have user messages with images."""
        sample = sample_trajectory_long(num_turns=6)
        messages = sample.messages
        protocol = UITarsHistoryProtocol(full_history_size=2)
        out = protocol.process_messages(messages)
        # Find user messages after the old assistant block
        user_msgs_in_window = [
            m
            for m in out
            if m["role"] == "user"
            and any(item.get("type") == "image" for item in m.get("content", []))
        ]
        assert len(user_msgs_in_window) == 2  # full_history_size=2

    def test_short_trajectory_no_windowing(self):
        """With 2 turns and full_history_size=5, all messages preserved."""
        sample = sample_trajectory_two_turns()
        messages = sample.messages
        protocol = UITarsHistoryProtocol(full_history_size=5)
        out = protocol.process_messages(messages)
        assert len(out) == len(messages)

    def test_reasoning_content_preserved(self):
        sample = sample_trajectory_with_reasoning()
        messages = sample.messages
        protocol = UITarsHistoryProtocol(full_history_size=5)
        out = protocol.process_messages(messages)
        reasoning_count = sum(1 for m in out if m.get("reasoning_content"))
        assert reasoning_count >= 2

    def test_empty_input(self):
        assert UITarsHistoryProtocol().process_messages([]) == []

    def test_default_window_size(self):
        protocol = UITarsHistoryProtocol()
        assert protocol.full_history_size == 5

    # ----- boundary: window == turns, window > turns -----

    def test_window_equals_num_turns(self):
        """When full_history_size == num_turns, no windowing should occur."""
        sample = sample_trajectory_long(num_turns=4)
        messages = sample.messages
        protocol = UITarsHistoryProtocol(full_history_size=4)
        out = protocol.process_messages(messages)
        assert len(out) == len(messages)

    def test_window_greater_than_num_turns(self):
        """When full_history_size > num_turns, all messages preserved."""
        sample = sample_trajectory_long(num_turns=3)
        messages = sample.messages
        protocol = UITarsHistoryProtocol(full_history_size=10)
        out = protocol.process_messages(messages)
        assert len(out) == len(messages)

    # ----- minimal cases -----

    def test_window_size_1(self):
        """full_history_size=1 should keep only the last turn as full."""
        sample = sample_trajectory_long(num_turns=4)
        messages = sample.messages
        protocol = UITarsHistoryProtocol(full_history_size=1)
        out = protocol.process_messages(messages)
        # first user (text only) + 3 old assistants + 1 user(image) + 1 assistant = 6
        assert len(out) == 6
        assert out[0]["role"] == "user"  # first user with instruction text
        # old assistant messages
        for i in range(1, 4):
            assert out[i]["role"] == "assistant"
        # recent turn
        assert out[4]["role"] == "user"
        assert out[5]["role"] == "assistant"

    def test_single_turn(self):
        """Single turn should pass through as-is."""
        sample = sample_trajectory_long(num_turns=1)
        messages = sample.messages
        protocol = UITarsHistoryProtocol(full_history_size=3)
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
        protocol = UITarsHistoryProtocol(full_history_size=2)
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
        protocol = UITarsHistoryProtocol(full_history_size=2)
        out = protocol.process_messages(messages)
        assert out[-1]["role"] == "user"

    def test_recent_tool_result_keeps_image_text_and_error(self):
        """Current role:tool feedback keeps its screenshot and error text."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "task"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": "click first"}],
                "tool_calls": [
                    make_tool_call(
                        "computer",
                        {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                        call_id="call_0000",
                    )
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0000",
                "content": [
                    {"type": "image", "index": 1},
                    {
                        "type": "text",
                        "text": (
                            "CURRENT_IMAGE_TEXT\n\n## Error from previous action:\nclick failed"
                        ),
                    },
                    {"type": "metadata", "data": {"is_error": True}},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": "recover"}],
            },
        ]

        out = UITarsHistoryProtocol(full_history_size=1).process_messages(messages)
        projected = next(
            message
            for message in out
            if any(
                part.get("text", "").startswith("CURRENT_IMAGE_TEXT")
                for part in message.get("content", [])
            )
        )

        assert projected["role"] == "tool"
        assert any(
            part.get("type") == "image" and part.get("index") == 1 for part in projected["content"]
        )
        assert any("click failed" in part.get("text", "") for part in projected["content"])
        assert any(
            part.get("type") == "metadata" and part.get("data", {}).get("is_error") is True
            for part in projected["content"]
        )
