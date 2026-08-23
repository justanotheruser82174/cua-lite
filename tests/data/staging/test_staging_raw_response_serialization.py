"""Staging raw_response serialization coverage."""

from __future__ import annotations

import json

from lite.data.staging import coerce_messages


def test_coerce_messages_preserves_json_string_raw_response_null_evidence() -> None:
    """JSON-string messages are opaque storage, not an Arrow-padding repair target."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "visible",
                    "provider_payload": {"explicit_null": None},
                }
            ],
            "tool_calls": [],
            "raw_response": {
                "text": "VERBATIM RAW",
                "adapter_key": "qwen3_vl@desktop@use",
                "provider_null": None,
            },
        },
    ]

    assert coerce_messages(json.dumps(messages)) == messages
