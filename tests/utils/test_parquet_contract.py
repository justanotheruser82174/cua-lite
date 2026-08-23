"""Parquet helper contract tests."""

from __future__ import annotations

import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from lite_samples import sample_grounding_action_minimal, sample_understanding

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools import make_tool_call
from lite.utils.parquet import _EMPTY_STRUCT_CHILD, write_records_to_parquet


def test_write_records_to_parquet_roundtrip(tmp_path):
    """Write records to parquet and read back; messages and meta are JSON strings in parquet."""
    records = [sample_understanding(), sample_grounding_action_minimal()]
    path = tmp_path / "out.parquet"
    write_records_to_parquet(records, path, json_fields=("messages", "metadata"))

    assert path.exists()
    df = pd.read_parquet(path)
    assert len(df) == 2
    assert "messages" in df.columns and "metadata" in df.columns
    row0 = df.iloc[0]
    messages = json.loads(row0["messages"])
    metadata = json.loads(row0["metadata"])
    assert messages == records[0].messages
    assert metadata == records[0].metadata.to_dict()


def test_write_records_to_parquet_empty_does_not_create_file(tmp_path):
    """write_records_to_parquet with empty list does not create file."""
    path = tmp_path / "empty.parquet"
    write_records_to_parquet([], path)
    assert not path.exists()


def test_write_records_to_parquet_json_fields_none(tmp_path):
    """write_records_to_parquet with json_fields=None does not serialize to string."""
    records = [sample_understanding()]
    path = tmp_path / "raw.parquet"
    write_records_to_parquet(records, path, json_fields=None)

    df = pd.read_parquet(path)
    # messages column may be stored as list/dict by PyArrow
    assert len(df) == 1


def test_binary_list_promoted_to_large_list_large_binary(tmp_path) -> None:
    # `processed_images: list[bytes]` must become large_list<large_binary> so
    # ParquetFile.iter_batches() stays valid past 2 GiB combined bytes.
    path = tmp_path / "binary.parquet"
    records = [
        {"processed_images": [b"\x89PNG\x00", b"abc"]},
        {"processed_images": [b"xyz"]},
    ]
    write_records_to_parquet(records, path)

    table = pq.read_table(path)
    field = table.schema.field("processed_images")
    assert pa.types.is_large_list(field.type)
    assert pa.types.is_large_binary(field.type.value_type)
    assert table.column("processed_images").to_pylist() == [
        [b"\x89PNG\x00", b"abc"],
        [b"xyz"],
    ]


def test_int_list_stays_normal_list(tmp_path) -> None:
    path = tmp_path / "nums.parquet"
    write_records_to_parquet([{"nums": [1, 2]}], path)

    table = pq.read_table(path)
    field = table.schema.field("nums")
    assert pa.types.is_list(field.type)
    assert not pa.types.is_large_list(field.type)
    assert pa.types.is_integer(field.type.value_type)
    assert table.column("nums").to_pylist() == [[1, 2]]


def test_empty_dict_survives_the_struct_limitation_as_a_dict(tmp_path) -> None:
    path = tmp_path / "empty_dict.parquet"
    write_records_to_parquet([{"task_params": {}}], path)

    table = pq.read_table(path)
    assert table.column("task_params").to_pylist() == [{_EMPTY_STRUCT_CHILD: None}]
    assert pa.types.is_struct(table.schema.field("task_params").type)


def test_empty_dict_beside_a_populated_sibling_needs_no_placeholder(tmp_path) -> None:
    path = tmp_path / "mixed_dict.parquet"
    write_records_to_parquet(
        [{"task_params": {}}, {"task_params": {"seed": 1}}],
        path,
    )

    values = pq.read_table(path).column("task_params").to_pylist()
    assert values == [{"seed": None}, {"seed": 1}]
    assert all(v is not None for v in values)
    assert _EMPTY_STRUCT_CHILD not in values[0]


def test_json_fields_do_not_receive_empty_struct_placeholder(tmp_path) -> None:
    path = tmp_path / "messages.parquet"
    messages = [
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [make_tool_call("back", {}, call_id="call_0000")],
        }
    ]
    write_records_to_parquet(
        [{"messages": messages, "metadata": LiteCUAMetadata().to_dict()}],
        path,
        json_fields=("messages", "metadata"),
    )

    row = pq.read_table(path).to_pylist()[0]
    assert _EMPTY_STRUCT_CHILD not in row["messages"]
    assert json.loads(row["messages"])[0]["tool_calls"] == [
        make_tool_call("back", {}, call_id="call_0000")
    ]
