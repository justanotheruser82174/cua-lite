"""Regression tests for lite.data.merge transport opacity and Arrow types.

Usage:
    uv run pytest tests/data/merge/test_merge.py
"""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from lite.core import LiteCUAMetadata
from lite.core.tools import make_tool_call, make_tool_schema
from lite.data.merge import _merge_jsonl, _merge_parquet
from lite.data.staging import iter_parquet_rows


def _row(tool_name: str, args: dict, schema_name: str, schema_props: dict) -> dict:
    return {
        "images": [],
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [make_tool_call(tool_name, args, call_id=f"call_{tool_name}")],
            },
            {
                "role": "tool",
                "tool_call_id": f"call_{tool_name}",
                "content": [{"type": "text", "text": "ok"}],
            },
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[
                make_tool_schema(
                    schema_name,
                    parameters={
                        "type": "object",
                        "properties": schema_props,
                        "required": list(schema_props),
                    },
                )
            ],
            valid_actions=None,
            others={},
        ).to_dict(),
    }


def test_merge_parquet_serializes_message_and_metadata_columns(tmp_path) -> None:
    rows = [
        _row(
            "computer",
            {"actions": [
                {"action": "click", "coordinate": [1, 2]},
                {"action": "type", "text": "hello"},
            ]},
            "bash",
            {"cmd": {"type": "string"}},
        ),
        _row(
            "goto",
            {"url": "https://example.com"},
            "goto",
            {"url": {"type": "string"}},
        ),
    ]
    inputs = []
    for i, row in enumerate(rows):
        path = tmp_path / f"in{i}.jsonl"
        path.write_text(json.dumps(row) + "\n")
        inputs.append(str(path))

    out = tmp_path / "merged.parquet"
    _merge_parquet(inputs, str(out))

    merged = list(iter_parquet_rows(out))
    assert len(merged) == 2
    for got, expected in zip(merged, rows):
        assert isinstance(got["messages"], str)
        assert isinstance(got["metadata"], str)
        assert json.loads(got["messages"]) == expected["messages"]
        assert json.loads(got["metadata"]) == expected["metadata"]


# ---------------------------------------------------------------------------
# export_sft schema: the `processed_images` column must stay 64-bit
# ---------------------------------------------------------------------------

_STEP_STRUCT = pa.struct([
    ("prompt", pa.string()),
    ("image_indices", pa.list_(pa.int64())),
    ("response", pa.string()),
    ("response_tokens", pa.list_(pa.int64())),
    ("reward", pa.float64()),
    ("status", pa.string()),
    ("prompt_tokens", pa.list_(pa.int64())),
])

#: Mirrors ``lite/train/export/export_sft.py::_output_features``. The point of
#: the fixture is the ``processed_images`` type: ``large_list<large_binary>`` is
#: declared deliberately there, because past 2 GiB of combined blobs a
#: 32-bit-offset column is silently chunked and
#: ``ParquetFile.iter_batches(batch_size=...)`` — slime's ``PROMPT_DATA`` reader
#: (``slime/slime/utils/data.py``, ``batch_size=4096``) — then raises "Nested
#: data conversions not implemented for chunked array outputs".
_SFT_SCHEMA = pa.schema([
    ("_error", pa.string()),
    ("processed_images", pa.large_list(pa.large_binary())),
    ("steps", pa.list_(_STEP_STRUCT)),
    ("metadata", pa.string()),
])


def _write_sft_parquet(path, *, blobs: list[bytes], tag: str) -> None:
    """Write a 1-row parquet with the export_sft schema (tiny fake PNG bytes)."""
    table = pa.table(
        {
            "_error": [""],
            "processed_images": [blobs],
            "steps": [[{
                "prompt": f"<|image|>{tag}",
                "image_indices": list(range(len(blobs))),
                "response": f"click {tag}",
                "response_tokens": [1, 2, 3],
                "reward": 1.0,
                "status": "success",
                "prompt_tokens": [4, 5],
            }]],
            "metadata": [json.dumps(LiteCUAMetadata(dims=("desktop", "use")).to_dict())],
        },
        schema=_SFT_SCHEMA,
    )
    pq.write_table(table, path)


def _sft_inputs(tmp_path) -> tuple[list[str], list[list[bytes]]]:
    blob_sets = [[b"\x89PNG-a0", b"\x89PNG-a1"], [b"\x89PNG-b0"]]
    paths = []
    for i, blobs in enumerate(blob_sets):
        path = tmp_path / f"sft{i}.parquet"
        _write_sft_parquet(path, blobs=blobs, tag=f"t{i}")
        # Precondition: the producer really does declare 64-bit offsets.
        assert pa.types.is_large_list(
            pq.ParquetFile(path).schema_arrow.field("processed_images").type)
        paths.append(str(path))
    return paths, blob_sets


def test_merge_parquet_keeps_processed_images_a_large_list(tmp_path) -> None:
    """The type gate: merge must not downgrade large_list<large_binary>."""
    inputs, _ = _sft_inputs(tmp_path)
    out = tmp_path / "merged.parquet"
    _merge_parquet(inputs, str(out))

    field = pq.ParquetFile(out).schema_arrow.field("processed_images")
    assert pa.types.is_large_list(field.type), field.type
    assert pa.types.is_large_binary(field.type.value_type), field.type


def test_merged_sft_parquet_survives_slime_iter_batches(tmp_path) -> None:
    """The consumer gate: read the merged file the way slime actually reads it."""
    inputs, blob_sets = _sft_inputs(tmp_path)
    out = tmp_path / "merged.parquet"
    _merge_parquet(inputs, str(out))

    pf = pq.ParquetFile(out)
    rows = []
    for batch in pf.iter_batches(batch_size=4096):
        rows.extend(batch.to_pylist())

    assert [row["processed_images"] for row in rows] == blob_sets
    assert [json.loads(row["metadata"])["dims"][0] for row in rows] == ["desktop"] * 2
    assert [row["steps"][0]["image_indices"] for row in rows] == [[0, 1], [0]]


def test_merge_jsonl_accepts_parquet_inputs(tmp_path) -> None:
    """parquet -> JSONL: rows must come out as plain Python, not numpy."""
    src = tmp_path / "canonical.parquet"
    pq.write_table(
        pa.table({
            "images": pa.array([["a/b.png"]]),
            "messages": pa.array([json.dumps([{"role": "user", "content": "hi"}])]),
            "metadata": pa.array([
                json.dumps(LiteCUAMetadata(dims=("desktop", "use")).to_dict())
            ]),
        }),
        src,
    )

    out = tmp_path / "merged.jsonl"
    _merge_jsonl([str(src)], str(out))

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows == [{
        "images": ["a/b.png"],
        "messages": json.dumps([{"role": "user", "content": "hi"}]),
        "metadata": json.dumps(LiteCUAMetadata(dims=("desktop", "use")).to_dict()),
    }]
