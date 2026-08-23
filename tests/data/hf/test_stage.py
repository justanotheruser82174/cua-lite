from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from lite.core import LiteCUAMetadata
from lite.core.tools import make_tool_call
from lite.data.hf.stage import stage
from lite.utils.parquet import write_records_to_parquet


def _pre_migration_row() -> dict:
    return {
        "images": [],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "clicking"}],
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "computer",
                            "arguments": '{"actions": [{"action": "click", "coordinate": [1, 2]}]}',
                        },
                    }
                ],
            },
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={"task_id": "t1"},
        ).to_dict(),
    }


def _final_gui_action_row(*, task_id: str = "t1") -> dict:
    return {
        "images": [],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "click target"}]},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call(
                        "computer",
                        {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                        call_id="call_0000",
                    )
                ],
            },
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={"task_id": task_id},
        ).to_dict(),
    }


def _write_stage_input(log_root: Path, row: dict, *, task_id: str = "t1") -> None:
    sample = log_root / "train" / task_id / "sample_00"
    sample.mkdir(parents=True)
    write_records_to_parquet([row], sample / "trajectory.parquet")


def test_stage_still_rejects_pre_migration_unstamped_tool_call(tmp_path) -> None:
    log_root = tmp_path / "logroot"
    _write_stage_input(log_root, _pre_migration_row())

    with pytest.raises(ValueError, match="missing non-empty id"):
        stage([log_root], name="X", out_dir=tmp_path / "staged", filter_expr=None)


def test_stage_accepts_final_gui_action_without_tool_result(tmp_path) -> None:
    log_root = tmp_path / "logroot"
    _write_stage_input(log_root, _final_gui_action_row())

    out = tmp_path / "staged"
    stage([log_root], name="X", out_dir=out, filter_expr=None)

    written = list(out.rglob("*.parquet"))
    assert written, "stage produced no partition"
    messages = json.loads(pd.read_parquet(written[0]).iloc[0]["messages"])
    assert messages[-1]["tool_calls"][0]["id"] == "call_0000"


def test_stage_rejects_out_of_bounds_coordinates(tmp_path) -> None:
    row = {
        "images": [],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "click target"}]},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call(
                        "computer",
                        {"actions": [{"action": "click", "coordinate": [1001, 5]}]},
                        call_id="call_0000",
                    )
                ],
            },
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "grounding.action"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={"task_id": "t1"},
        ).to_dict(),
    }
    log_root = tmp_path / "logroot"
    sample = log_root / "train" / "t1" / "sample_00"
    sample.mkdir(parents=True)
    write_records_to_parquet([row], sample / "trajectory.parquet")

    out = tmp_path / "staged"
    with pytest.raises(ValueError, match="out-of-range coordinates"):
        stage([log_root], name="X", out_dir=out, filter_expr=None)

    assert not list(out.rglob("*.parquet"))


def test_stage_stats_count_filtered_rows(tmp_path):
    log_root = tmp_path / "logs"
    _write_stage_input(
        log_root,
        _final_gui_action_row(task_id="task_logger_path"),
        task_id="task_logger_path",
    )

    staged_dataset = tmp_path / "staged" / "cua-lite" / "LoggerStats"
    stage(
        [log_root],
        name="LoggerStats",
        out_dir=staged_dataset,
        filter_expr="lambda m: False",
    )

    stats = json.loads((staged_dataset / "stats.json").read_text())
    assert stats["rows_in"] == 1
    assert stats["rows_out"] == 0
    assert stats["rows_dropped"] == 1


def test_stage_refuses_stale_output_without_overwrite(tmp_path):
    log_root = tmp_path / "logs"
    _write_stage_input(
        log_root,
        _final_gui_action_row(task_id="task_logger_path"),
        task_id="task_logger_path",
    )
    staged_dataset = tmp_path / "staged" / "cua-lite" / "LoggerStale"
    staged_dataset.mkdir(parents=True)
    stale = staged_dataset / "stale.txt"
    stale.write_text("old")

    with pytest.raises(FileExistsError, match="--overwrite"):
        stage(
            [log_root],
            name="LoggerStale",
            out_dir=staged_dataset,
            filter_expr=None,
        )

    stage(
        [log_root],
        name="LoggerStale",
        out_dir=staged_dataset,
        filter_expr=None,
        overwrite=True,
    )
    assert not stale.exists()
    assert (staged_dataset / "stats.json").is_file()


def test_stage_refuses_output_overlapping_input_root(tmp_path):
    log_root = tmp_path / "logs"
    _write_stage_input(
        log_root,
        _final_gui_action_row(task_id="task_logger_path"),
        task_id="task_logger_path",
    )

    with pytest.raises(ValueError, match="must not overlap protected input root"):
        stage(
            [log_root],
            name="LoggerOverlap",
            out_dir=log_root / "annotated",
            filter_expr=None,
            overwrite=True,
        )


def test_stage_runs_canonical_content_gate_before_writing(tmp_path):
    task_id = "task_bad_content"
    sample_dir = tmp_path / "logs" / "train" / task_id / "sample_00"
    sample_dir.mkdir(parents=True)
    write_records_to_parquet(
        [
            {
                "images": [],
                "messages": [{"role": "assistant", "content": []}],
                "metadata": LiteCUAMetadata(
                    dims=("desktop", "use"),
                    extra_tool_schemas=[],
                    valid_actions=None,
                    others={"task_id": task_id},
                ).to_dict(),
            }
        ],
        sample_dir / "trajectory.parquet",
        json_fields=("messages", "metadata"),
    )
    staged_dataset = tmp_path / "staged" / "cua-lite" / "BadContent"

    with pytest.raises(ValueError, match="content-only final assistant turn"):
        stage(
            [tmp_path / "logs"],
            name="BadContent",
            out_dir=staged_dataset,
            filter_expr=None,
        )

    assert list(staged_dataset.rglob("*.parquet")) == []
