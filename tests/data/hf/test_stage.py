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


def test_stage_records_upstream_origin_urls(tmp_path) -> None:
    log_root = tmp_path / "logroot"
    _write_stage_input(log_root, _final_gui_action_row())

    out = tmp_path / "staged"
    stage(
        [log_root],
        name="X",
        out_dir=out,
        filter_expr=None,
        original_urls=["https://github.com/THUDM/SCALE-CUA", "https://arxiv.org/abs/2607.11185"],
        citation="See https://arxiv.org/abs/2607.11185",
    )

    repo = json.loads((out / "repo.json").read_text())
    assert repo["original_urls"] == [
        "https://github.com/THUDM/SCALE-CUA",
        "https://arxiv.org/abs/2607.11185",
    ]
    assert repo["citation"] == "See https://arxiv.org/abs/2607.11185"


def test_stage_seeds_card_fields_from_repo_dir(tmp_path) -> None:
    log_root = tmp_path / "logroot"
    _write_stage_input(log_root, _final_gui_action_row())
    repo_dir = tmp_path / "route"
    repo_dir.mkdir()
    (repo_dir / "repo.json").write_text(json.dumps({
        "description": "route-owned description",
        "original_urls": ["https://github.com/THUDM/SCALE-CUA"],
        "license": "other",
        "citation": "See https://arxiv.org/abs/2607.11185",
    }))

    out = tmp_path / "staged"
    stage([log_root], name="X", out_dir=out, filter_expr=None, repo_dir=repo_dir)

    repo = json.loads((out / "repo.json").read_text())
    assert repo["description"] == "route-owned description"
    assert repo["original_urls"] == ["https://github.com/THUDM/SCALE-CUA"]
    assert repo["citation"] == "See https://arxiv.org/abs/2607.11185"
    # the run-derived half is still this run's, not the file's
    assert "Staged via `lite.data.hf.stage`" in repo["extra_notes"]


def test_stage_flags_override_repo_dir(tmp_path) -> None:
    log_root = tmp_path / "logroot"
    _write_stage_input(log_root, _final_gui_action_row())
    repo_dir = tmp_path / "route"
    repo_dir.mkdir()
    (repo_dir / "repo.json").write_text(json.dumps({
        "description": "route-owned", "original_urls": ["https://example.com/route"],
        "license": "other", "citation": "route citation",
    }))

    out = tmp_path / "staged"
    stage([log_root], name="X", out_dir=out, filter_expr=None, repo_dir=repo_dir,
          description="ad-hoc", original_urls=["https://example.com/adhoc"])

    repo = json.loads((out / "repo.json").read_text())
    assert repo["description"] == "ad-hoc"
    assert repo["original_urls"] == ["https://example.com/adhoc"]
    assert repo["citation"] == "route citation"  # unset flag falls back to the file


def test_stage_takes_license_from_repo_dir_and_defaults_to_other(tmp_path) -> None:
    log_root = tmp_path / "logroot"
    _write_stage_input(log_root, _final_gui_action_row())
    repo_dir = tmp_path / "route"
    repo_dir.mkdir()
    (repo_dir / "repo.json").write_text(json.dumps({
        "description": "d", "original_urls": [], "license": "See upstream (MIT).", "citation": "c",
    }))

    stage([log_root], name="X", out_dir=tmp_path / "a", filter_expr=None, repo_dir=repo_dir)
    assert json.loads((tmp_path / "a" / "repo.json").read_text())["license"] == (
        "See upstream (MIT)."
    )

    stage([log_root], name="X", out_dir=tmp_path / "b", filter_expr=None)
    assert json.loads((tmp_path / "b" / "repo.json").read_text())["license"] == "other"


def test_stage_rejects_repo_dir_missing_a_required_key(tmp_path) -> None:
    log_root = tmp_path / "logroot"
    _write_stage_input(log_root, _final_gui_action_row())
    repo_dir = tmp_path / "route"
    repo_dir.mkdir()
    (repo_dir / "repo.json").write_text(json.dumps({"description": "d", "license": "other"}))

    with pytest.raises(KeyError, match="original_urls"):
        stage([log_root], name="X", out_dir=tmp_path / "staged", filter_expr=None,
              repo_dir=repo_dir)


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
