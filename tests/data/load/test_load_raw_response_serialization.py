"""load_file_as_dataset raw_response serialization coverage."""

from __future__ import annotations

import json

from lite.core import LiteCUAMetadata, LiteSample
from lite.core.tools import make_tool_call
from lite.data.load import load_file_as_dataset
from lite.utils.parquet import write_records_to_parquet


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
    """Build a Parquet-ready dict (LiteSample.to_dict format) with no images."""
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


def test_parquet_roundtrip_preserves_raw_response(tmp_path) -> None:
    """Parquet write/read preserves the nested raw_response struct."""
    raw_text = "VERBATIM RAW"
    adapter_key = "qwen3_vl@desktop@use"
    sample_dict = _make_sample(
        [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            _assistant_with_raw(raw_text, adapter_key),
        ]
    )

    path = tmp_path / "data.parquet"
    write_records_to_parquet(
        [sample_dict],
        path,
        json_fields=("messages", "metadata"),
    )

    ds = load_file_as_dataset(path)
    row = ds[0]
    messages = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]
    assistant = next(m for m in messages if m.get("role") == "assistant")
    assert assistant["raw_response"]["text"] == raw_text
    assert assistant["raw_response"]["adapter_key"] == adapter_key


def test_parquet_sparse_mix_schema_unified(tmp_path) -> None:
    """Rows with and without raw_response must survive one Parquet schema."""
    raw_text = "ONLY IN ROW 0"
    adapter_key = "qwen3_vl@desktop@use"

    sample_with = _make_sample(
        [
            {"role": "user", "content": [{"type": "text", "text": "a"}]},
            _assistant_with_raw(raw_text, adapter_key),
        ]
    )
    sample_without = _make_sample(
        [
            {"role": "user", "content": [{"type": "text", "text": "b"}]},
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": "y"}],
                "tool_calls": [
                    make_tool_call(
                        "computer",
                        {"actions": [{"action": "click", "coordinate": [3, 4]}]},
                        call_id="call_0000",
                    )
                ],
            },
        ]
    )

    path = tmp_path / "mixed.parquet"
    write_records_to_parquet(
        [sample_with, sample_without],
        path,
        json_fields=("messages", "metadata"),
    )

    ds = load_file_as_dataset(path)
    assert len(ds) == 2

    m0 = json.loads(ds[0]["messages"]) if isinstance(ds[0]["messages"], str) else ds[0]["messages"]
    asst0 = next(m for m in m0 if m["role"] == "assistant")
    assert asst0["raw_response"]["text"] == raw_text
    assert asst0["raw_response"]["adapter_key"] == adapter_key

    m1 = json.loads(ds[1]["messages"]) if isinstance(ds[1]["messages"], str) else ds[1]["messages"]
    asst1 = next(m for m in m1 if m["role"] == "assistant")
    assert asst1.get("raw_response") is None, (
        f"expected no raw_response on row 1, got {asst1.get('raw_response')!r}"
    )
