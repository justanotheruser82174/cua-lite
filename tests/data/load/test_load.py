"""
Tests for lite.data.load: load_file_as_dataset (parquet + JSONL), load_yaml_config.

Uses tmp_path and samples from lite_samples where applicable. Run from repo root:
  uv run pytest tests/data/load/test_load.py -v
"""

from __future__ import annotations

import json

from lite_samples import sample_trajectory_one_turn, sample_understanding

from lite.core import LiteCUAMetadata
from lite.core.tools import make_tool_call, make_tool_schema
from lite.data.load import load_file_as_dataset
from lite.data.staging import coerce_messages
from lite.utils.parquet import write_records_to_parquet

# -----------------------------------------------------------------------------
# load_file_as_dataset: parquet
# -----------------------------------------------------------------------------

def test_load_file_as_dataset_parquet(tmp_path):
    """load_file_as_dataset loads parquet and returns Dataset with rows."""
    records = [sample_understanding(), sample_trajectory_one_turn()]
    path = tmp_path / "data.parquet"
    write_records_to_parquet(records, path, json_fields=("messages", "metadata"))

    ds = load_file_as_dataset(path)
    assert len(ds) == 2
    assert "messages" in ds.column_names
    assert "metadata" in ds.column_names
    # Parquet may store messages as JSON string
    row0 = ds[0]
    if isinstance(row0["messages"], str):
        parsed = json.loads(row0["messages"])
        assert len(parsed) >= 1
    else:
        assert len(row0["messages"]) >= 1

def test_load_file_as_dataset_parquet_single_column(tmp_path):
    """load_file_as_dataset loads parquet with single row (minimal columns)."""
    path = tmp_path / "single.parquet"
    metadata = LiteCUAMetadata(
        dims=("desktop", "understanding"),
        others={"id": "one", "source": "t"},
    ).to_dict()
    write_records_to_parquet(
        [{
            "messages": [],
            "tools": None,
            "images": [],
            "metadata": metadata,
        }],
        path,
        json_fields=("messages", "metadata"),
    )
    ds = load_file_as_dataset(path)
    assert len(ds) == 1
    assert json.loads(ds[0]["metadata"]) == metadata

# -----------------------------------------------------------------------------
# load_file_as_dataset: JSONL
# -----------------------------------------------------------------------------

def test_load_file_as_dataset_jsonl(tmp_path):
    """load_file_as_dataset loads JSONL (one JSON object per line)."""
    path = tmp_path / "data.jsonl"
    records = [sample_understanding(), sample_trajectory_one_turn()]
    dicts = [r.to_dict() for r in records]
    with open(path, "w", encoding="utf-8") as f:
        for d in dicts:
            f.write(json.dumps(d) + "\n")

    ds = load_file_as_dataset(path)
    assert len(ds) == 2
    row_meta = ds[0]["metadata"]
    if isinstance(row_meta, str):
        row_meta = json.loads(row_meta)
    assert row_meta["metadata_kind"] == dicts[0]["metadata"]["metadata_kind"]
    assert row_meta["dims"] == dicts[0]["metadata"]["dims"]
    messages = coerce_messages(ds[0]["messages"])
    assert len(messages) == len(dicts[0]["messages"])
    assert messages[0]["role"] == dicts[0]["messages"][0]["role"]


def test_load_file_as_dataset_jsonl_keeps_messages_and_metadata_opaque(tmp_path):
    """JSONL load must not let datasets infer nested tool-call/schema structs."""
    messages = [
        {
            "role": "assistant",
            "tool_calls": [make_tool_call(
                "computer",
                {"actions": [
                    {"action": "click", "coordinate": [1, 2]},
                    {"action": "type", "text": "hello"},
                ]},
                call_id="call_0000",
            )],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0000",
            "content": [{"type": "text", "text": "ok"}],
        },
    ]
    metadata = LiteCUAMetadata(
        dims=("desktop", "use"),
        extra_tool_schemas=[
            make_tool_schema(
                "bash",
                parameters={
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
            )
        ],
        valid_actions=None,
        others={},
    ).to_dict()
    path = tmp_path / "opaque.jsonl"
    path.write_text(json.dumps({"images": [], "messages": messages, "metadata": metadata}) + "\n")

    ds = load_file_as_dataset(path)
    row = ds[0]

    assert isinstance(row["messages"], str)
    assert isinstance(row["metadata"], str)
    assert json.loads(row["messages"]) == messages
    assert json.loads(row["metadata"]) == metadata


def test_load_file_as_dataset_jsonl_empty_file(tmp_path):
    """load_file_as_dataset on empty JSONL returns empty Dataset."""
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    ds = load_file_as_dataset(path)
    assert len(ds) == 0

def test_load_file_as_dataset_jsonl_skips_blank_lines(tmp_path):
    """load_file_as_dataset skips blank lines in JSONL."""
    path = tmp_path / "data.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample_understanding().to_dict()) + "\n")
        f.write("\n")
        f.write("   \n")
        f.write(json.dumps(sample_trajectory_one_turn().to_dict()) + "\n")
    ds = load_file_as_dataset(path)
    assert len(ds) == 2
