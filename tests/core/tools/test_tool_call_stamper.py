"""Preproc publish-gate regressions for the shared Lite tool-call stamper."""

from __future__ import annotations

import pytest

from lite.core.errors import ToolCallValidationError
from lite.core.tools import make_tool_call
from lite.core.tools.calls import stamp_messages_tool_call_ids, tool_call_id


def test_stamp_messages_tool_call_ids_fills_every_call_uniquely() -> None:
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "go"}]},
        {
            "role": "assistant",
            "tool_calls": [
                make_tool_call("computer", {"actions": [{"action": "screenshot"}]}),
                make_tool_call("computer", {"actions": [{"action": "screenshot"}]}),
            ],
        },
        {
            "role": "assistant",
            "tool_calls": [
                make_tool_call("computer", {"actions": [{"action": "screenshot"}]}),
            ],
        },
    ]
    assert stamp_messages_tool_call_ids(messages, preserve=True) is messages
    ids = [tool_call_id(c) for m in messages for c in m.get("tool_calls") or []]
    assert len(ids) == 3
    assert all(isinstance(i, str) and i for i in ids)
    assert len(set(ids)) == 3


def test_stamp_messages_tool_call_ids_preserves_existing_ids_without_collision() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "screenshot"}]},
                    call_id="call_0000",
                ),
                make_tool_call("computer", {"actions": [{"action": "screenshot"}]}),
            ],
        },
    ]
    stamp_messages_tool_call_ids(messages, preserve=True)
    calls = messages[0]["tool_calls"]
    assert tool_call_id(calls[0]) == "call_0000"
    assert tool_call_id(calls[1]) != "call_0000"


def test_stamp_messages_tool_call_ids_rejects_tool_call_id_in_preproc_policy() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "tool_call_id": "legacy_1",
                    "type": "function",
                    "function": {
                        "name": "computer",
                        "arguments": {"actions": [{"action": "screenshot"}]},
                    },
                },
            ],
        },
    ]

    with pytest.raises(ToolCallValidationError, match="tool_call_id"):
        stamp_messages_tool_call_ids(messages, preserve=True)
