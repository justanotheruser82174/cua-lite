from __future__ import annotations

import json
import sys

import pandas as pd

from lite.core import LiteCUAMetadata
from lite.data import split as split_mod
from lite.data.staging import write_partition
from lite.utils.parquet import write_records_to_parquet


def test_split_rewrites_metadata_split_per_output(tmp_path, monkeypatch) -> None:
    rows = [
        {
            "problem": f"Complete the task: task_{i}",
            "metadata": {
                "env_key": f"osworld_g@task_{i}",
                "split": "eval",
            },
        }
        for i in range(3)
    ]
    input_path = tmp_path / "all.parquet"
    train_path = tmp_path / "train.parquet"
    eval_path = tmp_path / "eval.parquet"
    write_records_to_parquet(rows, input_path)
    assert isinstance(pd.read_parquet(input_path).iloc[0]["metadata"], dict)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "split",
            "-i",
            str(input_path),
            "--eval-size",
            "1",
            "--train-output",
            str(train_path),
            "--eval-output",
            str(eval_path),
            "--seed",
            "1",
        ],
    )
    split_mod.main()

    train_rows = pd.read_parquet(train_path).to_dict(orient="records")
    eval_rows = pd.read_parquet(eval_path).to_dict(orient="records")
    assert len(train_rows) == 2 and len(eval_rows) == 1
    assert all(isinstance(r["metadata"], dict) for r in train_rows + eval_rows)
    assert {r["metadata"]["split"] for r in train_rows} == {"train"}
    assert {r["metadata"]["split"] for r in eval_rows} == {"eval"}
    assert {r["metadata"]["env_key"] for r in train_rows + eval_rows} == {
        r["metadata"]["env_key"] for r in rows
    }


def test_split_preserves_opaque_canonical_metadata(tmp_path, monkeypatch) -> None:
    rows = [
        {
            "images": [],
            "messages": [{"role": "user", "content": [{"type": "text", "text": f"task {i}"}]}],
            "metadata": LiteCUAMetadata(
                dims=("desktop", "use"),
                extra_tool_schemas=[],
                valid_actions=None,
                others={"env_key": f"osworld_g@task_{i}"},
            ).to_dict(),
        }
        for i in range(3)
    ]
    input_path = tmp_path / "canonical.parquet"
    train_path = tmp_path / "train.parquet"
    eval_path = tmp_path / "eval.parquet"
    write_partition(rows, input_path)
    assert isinstance(pd.read_parquet(input_path).iloc[0]["metadata"], str)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "split",
            "-i",
            str(input_path),
            "--eval-size",
            "1",
            "--train-output",
            str(train_path),
            "--eval-output",
            str(eval_path),
            "--seed",
            "1",
        ],
    )
    split_mod.main()

    output_rows = (
        pd.read_parquet(train_path).to_dict(orient="records")
        + pd.read_parquet(eval_path).to_dict(orient="records")
    )
    assert len(output_rows) == 3
    assert all(isinstance(row["metadata"], str) for row in output_rows)
    assert all(isinstance(row["messages"], str) for row in output_rows)
    assert all("split" not in json.loads(row["metadata"]) for row in output_rows)
    assert {
        json.loads(row["metadata"])["others"]["env_key"]
        for row in output_rows
    } == {f"osworld_g@task_{i}" for i in range(3)}
