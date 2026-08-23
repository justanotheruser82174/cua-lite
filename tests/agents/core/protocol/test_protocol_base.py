"""
Tests for BaseProtocol and ProtocolRegistry.

Covers:
  1. ProtocolRegistry: get, list
  2. group_into_turns (lite.core.messages — BaseProtocol no longer copies it)
  3. Window metadata-retention policy
  4. The message-window boundary: `[]` is the only "no messages" value

Run:
    uv run pytest tests/agents/core/protocol/test_protocol_base.py -v
"""

from __future__ import annotations

import pytest

from lite.agents.core.protocol import BaseProtocol, ProtocolRegistry
from lite.agents.core.protocol import window as protocol_window
from lite.agents.core.protocol.base import FullHistoryProtocol
from lite.agents.models.qwen3_vl.protocol import Qwen3VLHistoryProtocol
from lite.agents.models.ui_tars.protocol import UITarsHistoryProtocol
from lite.core.messages import group_into_turns

# =============================================================================
# 1. ProtocolRegistry
# =============================================================================

class TestProtocolRegistry:
    def test_list_includes_all_protocols(self):
        keys = ProtocolRegistry.list()
        for key in ["lite.history", "qwen3_vl.history", "ui_tars.history"]:
            assert key in keys

    def test_get_full_history(self):
        protocol = ProtocolRegistry.get("lite.history")
        assert isinstance(protocol, FullHistoryProtocol)

    def test_get_qwen3_vl_history(self):
        protocol = ProtocolRegistry.get("qwen3_vl.history")
        assert isinstance(protocol, Qwen3VLHistoryProtocol)

    def test_get_ui_tars_history(self):
        protocol = ProtocolRegistry.get("ui_tars.history")
        assert isinstance(protocol, UITarsHistoryProtocol)

    def test_get_with_kwargs(self):
        protocol = ProtocolRegistry.get("qwen3_vl.history", full_history_size=3)
        assert protocol.full_history_size == 3

    def test_get_registry_key(self):
        assert FullHistoryProtocol.get_registry_key() == "lite.history"
        assert Qwen3VLHistoryProtocol.get_registry_key() == "qwen3_vl.history"
        assert UITarsHistoryProtocol.get_registry_key() == "ui_tars.history"

# =============================================================================
# 2. group_into_turns
# =============================================================================

class TestGroupIntoTurns:
    def test_pairs_user_assistant(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "assistant", "content": [{"type": "action_description", "text": "Hello"}]},
            {"role": "user", "content": []},
            {"role": "assistant", "content": []},
        ]
        turns = group_into_turns(messages)
        assert len(turns) == 2
        assert turns[0]["observations"] == [messages[0]]
        assert turns[0]["assistant"] == messages[1]
        assert turns[1]["observations"] == [messages[2]]
        assert turns[1]["assistant"] == messages[3]

    def test_empty(self):
        assert group_into_turns([]) == []

    def test_single_user_no_assistant(self):
        messages = [{"role": "user", "content": []}]
        turns = group_into_turns(messages)
        assert len(turns) == 1
        assert turns[0]["observations"] == messages
        assert turns[0]["assistant"] is None

    def test_odd_message_count(self):
        messages = [
            {"role": "user", "content": []},
            {"role": "assistant", "content": []},
            {"role": "user", "content": []},
        ]
        turns = group_into_turns(messages)
        assert len(turns) == 2
        assert turns[1]["assistant"] is None

    def test_instruction_extraction_requires_a_user_observation(self):
        class ProbeProtocol(BaseProtocol):
            def process_messages(self, messages, **kwargs):
                return messages

            def extract_instruction(self, turn):
                return self._extract_instruction(turn)

        protocol = ProbeProtocol()

        with pytest.raises(ValueError, match="role:user instruction"):
            protocol.extract_instruction({
                "observations": [
                    {
                        "role": "tool",
                        "tool_call_id": "call_0",
                        "content": [{"type": "text", "text": "tool result"}],
                    }
                ],
                "assistant": None,
            })

    def test_instruction_extraction_reads_user_observation_text(self):
        class ProbeProtocol(BaseProtocol):
            def process_messages(self, messages, **kwargs):
                return messages

            def extract_instruction(self, turn):
                return self._extract_instruction(turn)

        protocol = ProbeProtocol()

        assert protocol.extract_instruction({
            "observations": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "real instruction"}],
                }
            ],
            "assistant": None,
        }) == "real instruction"

    def test_base_protocol_has_no_tool_result_summary_policy(self):
        assert not hasattr(BaseProtocol, "tool_result_summary_text")

# =============================================================================
# 3. Window metadata-retention policy (one policy, every role, every entry point)
# =============================================================================

class TestWindowMetadataRetentionPolicy:
    """Metadata is per-call/page state, never optional prose — kept for every
    role, whether a caller goes through ``filter_history_content`` or calls
    ``keep_images_and_tool_text`` directly. Regression guard for the split where
    ``keep_images_and_tool_text`` used to drop metadata on non-tool messages."""

    def _user_with_image_and_metadata(self) -> dict:
        return {
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "ambient observation"},
                {"type": "metadata", "data": {"page_title": "Example"}},
            ],
        }

    def test_keep_images_and_tool_text_keeps_metadata_on_non_tool_role(self):
        message = self._user_with_image_and_metadata()

        filtered = protocol_window.keep_images_and_tool_text(message)

        assert filtered["content"] == [
            {"type": "image", "index": 0},
            {"type": "metadata", "data": {"page_title": "Example"}},
        ]

    def test_filter_history_content_matches_keep_images_and_tool_text(self):
        message = self._user_with_image_and_metadata()

        assert protocol_window.filter_history_content(
            message,
            strip_regular_image_text=True,
        ) == protocol_window.keep_images_and_tool_text(message)

# =============================================================================
# 4. Message-window boundary: `[]` is the only "no messages" value
# =============================================================================

#: One representative of each ``process_messages`` implementation in
#: ``protocol/base.py``: the full-history path and the turn-window frame every
#: history family inherits.
_PROTOCOL_ENTRY_POINTS = [
    FullHistoryProtocol,
    Qwen3VLHistoryProtocol,
    UITarsHistoryProtocol,
]

class TestNoneIsNotAnEmptyMessageList:
    """``None`` is a caller-code mistake, not a second spelling of ``[]``.

    An empty window is ``[]`` at every protocol entry point. Were ``None``
    normalized instead, the full-history path would answer ``None`` and the
    turn-window path ``[]`` — two more representations of "no messages" that let
    a dropped message list travel on unnoticed.
    """

    @pytest.mark.parametrize("protocol_cls", _PROTOCOL_ENTRY_POINTS)
    def test_empty_list_is_the_empty_window(self, protocol_cls):
        assert protocol_cls().process_messages([]) == []

    @pytest.mark.parametrize("protocol_cls", _PROTOCOL_ENTRY_POINTS)
    def test_none_is_rejected(self, protocol_cls):
        with pytest.raises(TypeError, match=r"expects a list of messages, got NoneType"):
            protocol_cls().process_messages(None)
