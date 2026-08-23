from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from lite.core import LiteCUAMetadata
from lite.core.errors import LiteContractError
from lite.core.messages.image_refs import validate_image_references
from lite.data.staging import coerce_messages, to_plain, write_partition
from lite.data.utils.rows import validate_tool_calls


def test_coerce_messages_preserves_decoded_padding_evidence_without_repairing_calls() -> None:
    messages = [
        {"role": "user", "content": [], "tool_calls": None},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_0000",
                    "type": "function",
                    "function": {
                        "name": "computer",
                        "arguments": {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "goto", "arguments": None}}],
        },
        {
            "role": "tool",
            "content": [{"type": "image", "index": 0}],
            "tool_calls": None,
            "call_id": "call_0000",
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Done."}],
            "tool_calls": [],
        },
    ]

    out = coerce_messages(messages)

    assert out == messages
    with pytest.raises(ValueError, match="type must be 'function'"):
        validate_tool_calls(out)


def test_coerce_messages_preserves_role_tool_key_order_and_padding_evidence() -> None:
    messages = [
        {
            "content": [{"type": "text", "text": "ok"}],
            "tool_call_id": "call_0000",
            "role": "tool",
            "tool_calls": None,
        },
    ]

    out = coerce_messages(messages)

    assert list(out[0]) == ["content", "tool_call_id", "role", "tool_calls"]
    assert out == messages


def test_write_partition_orders_canonical_tool_messages_and_calls(tmp_path) -> None:
    row = {
        "images": [],
        "messages": [
            {
                "content": [{"type": "text", "text": "ok"}],
                "tool_call_id": "call_0000",
                "role": "tool",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {"arguments": {}, "name": "response"},
                        "type": "function",
                        "id": "call_0000",
                    }
                ],
            },
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    }
    path = tmp_path / "ordered.parquet"

    write_partition([row], path)

    stored = pd.read_parquet(path).iloc[0]
    messages = json.loads(stored["messages"])
    assert list(messages[0]) == ["role", "tool_call_id", "content"]
    call = messages[1]["tool_calls"][0]
    assert list(call) == ["id", "type", "function"]
    assert list(call["function"]) == ["name", "arguments"]


def test_coerce_messages_normalizes_integral_float_image_refs_only() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 0.0, "text": None},
                {"type": "text", "text": "start", "index": None},
            ],
            "tool_calls": None,
        },
        {
            "role": "tool",
            "tool_call_id": "call_0000",
            "content": [{"type": "image", "index": 1.0, "text": None}],
        },
    ]

    out = coerce_messages(messages)

    assert out[0]["content"] == [
        {"type": "image", "index": 0, "text": None},
        {"type": "text", "text": "start", "index": None},
    ]
    assert out[1]["content"] == [{"type": "image", "index": 1, "text": None}]
    assert type(out[0]["content"][0]["index"]) is int
    assert type(out[1]["content"][0]["index"]) is int
    validate_image_references(out, ["img0.png", "img1.png"])


@pytest.mark.parametrize("bad_index", [0.5, -1.0, True, "0", None, "0.0"])
def test_coerce_messages_keeps_invalid_image_refs_invalid(bad_index) -> None:
    messages = [{"role": "user", "content": [{"type": "image", "index": bad_index}]}]

    out = coerce_messages(messages)

    assert out[0]["content"][0]["index"] == bad_index
    with pytest.raises(LiteContractError, match="non-negative"):
        validate_image_references(out, ["img0.png"])


def test_to_plain_handles_numpy_scalars() -> None:
    assert to_plain(np.int64(7)) == 7
    assert to_plain(np.bool_(True)) is True
    assert to_plain(np.str_("x")) == "x"
