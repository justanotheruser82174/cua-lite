"""JSONL round-trip coverage for LiteSample."""

from __future__ import annotations

import json

from lite.core import LiteCUAMetadata, LiteSample
from lite.core.tools import make_tool_call


def _assistant_with_raw(text: str, adapter_key: str) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "action_description", "text": "x"}],
        "tool_calls": [
            make_tool_call(
                "computer",
                {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                call_id="call_0000",
            )
        ],
        "raw_response": {"text": text, "adapter_key": adapter_key},
    }


def _make_sample(messages: list[dict]) -> dict:
    sample = LiteSample(
        metadata=LiteCUAMetadata(
            dims=(
                LiteCUAMetadata.Platform.DESKTOP.value,
                LiteCUAMetadata.TaskType.USE.value,
            ),
            others={"id": "test"},
        ),
        images=[],
        messages=messages,
    )
    d = sample.to_dict()
    d.pop("images", None)
    return d


def test_jsonl_roundtrip_preserves_raw_response() -> None:
    """``json.dumps -> json.loads -> LiteSample.from_dict`` preserves raw_response."""
    raw_text = '<tool_call>{"name":"x"}</tool_call>'
    adapter_key = "qwen3_vl@desktop@use"

    sample_dict = _make_sample(
        [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            _assistant_with_raw(raw_text, adapter_key),
        ]
    )

    encoded = json.dumps(sample_dict)
    decoded = json.loads(encoded)

    roundtrip = LiteSample.from_dict(decoded)
    assistant = roundtrip.messages[1]
    assert assistant["raw_response"]["text"] == raw_text
    assert assistant["raw_response"]["adapter_key"] == adapter_key
