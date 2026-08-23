from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from lite.infer.debug.log_layout import (
    ACTIONS_JSON,
    PROMPT_IMAGES_ANNOTATED_DIR,
    PROMPT_IMAGES_DIR,
    RESPONSE_TXT,
    RESULTS_JSON,
)

_SCRIPT = Path(__file__).resolve().parents[1] / "vision_gate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("scalecua_vision_gate", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    return str(path)


def _queue_row(root: Path, queue_id: str, reward: float = 1.0) -> dict[str, Any]:
    log_dir = root / "logs" / "train" / queue_id / "sample_00"
    return {
        "queue_id": f"train/{queue_id}/sample_00",
        "sample_relpath": f"train/{queue_id}/sample_00",
        "split": "train",
        "task_id": queue_id,
        "sample_id": "sample_00",
        "domain": "chrome",
        "log_dir": str(log_dir),
        "reported_reward": reward,
        "reported_terminated": True,
        "reported_truncated": False,
    }


def _sidecar(root: Path, queue: dict[str, Any], label: str = "true_success") -> dict[str, Any]:
    qid = queue["queue_id"]
    sample = Path(queue["log_dir"])
    turn0 = sample / "turn_00"
    turn1 = sample / "turn_01"
    initial_image = _touch(turn0 / PROMPT_IMAGES_DIR / "0000_reset.png")
    final_image = _touch(turn1 / PROMPT_IMAGES_DIR / "0001_from_call_0000.png")
    _touch(turn1 / PROMPT_IMAGES_ANNOTATED_DIR / "0001_from_call_0000.png")
    visual_success = label in {"true_success", "false_failure"}
    reward_success = float(queue["reported_reward"]) > 0.5
    return {
        "schema_version": 1,
        "run_id": "run",
        "source_queue_path": str(root / "queue.jsonl"),
        "queue_id": qid,
        "sample_id": queue["sample_id"],
        "task_id": queue["task_id"],
        "split": queue["split"],
        "domain": queue["domain"],
        "log_dir": str(sample),
        "reported_reward": queue["reported_reward"],
        "reported_terminated": queue["reported_terminated"],
        "reported_truncated": queue["reported_truncated"],
        "setup_ok": True,
        "action_ok": True,
        "eval_ok": True,
        "visual_label": label,
        "visual_success": visual_success,
        "notes": "checked",
        "reviewer": "unit",
        "checked_at": "2026-07-20T00:00:00Z",
        "latest_turn": "turn_01",
        "result_json_path": _touch(turn1 / RESULTS_JSON),
        "actions_path": _touch(turn1 / ACTIONS_JSON),
        "response_path": _touch(turn1 / RESPONSE_TXT),
        "frame_paths": {
            "initial": initial_image,
            "final": final_image,
        },
        "artifact_paths": [],
        "probe_paths": [],
        "vision_done": visual_success,
        "reward_vision_disagree": reward_success != visual_success,
        "probe_status": "not_needed",
    }


def _write_sidecar(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")


def _run_gate(tmp_path: Path, queue_rows: list[dict[str, Any]], sidecars: list[dict[str, Any]]):
    mod = _load_module()
    queue_path = tmp_path / "queue.jsonl"
    sidecar_root = tmp_path / "sidecars"
    report_path = tmp_path / "report.json"
    _write_jsonl(queue_path, queue_rows)
    for idx, sidecar in enumerate(sidecars):
        _write_sidecar(sidecar_root / f"sidecar_{idx}.json", sidecar)
    report = mod.validate_full_census(
        queue_path=queue_path,
        sidecar_root=sidecar_root,
        report_path=report_path,
    )
    return report, json.loads(report_path.read_text(encoding="utf-8"))


def test_full_census_gate_passes_complete_queue(tmp_path):
    queues = [
        _queue_row(tmp_path, "task_1", 1.0),
        _queue_row(tmp_path, "task_2", 1.0),
        _queue_row(tmp_path, "task_3", 1.0),
    ]
    sidecars = [_sidecar(tmp_path, row) for row in queues]

    report, written = _run_gate(tmp_path, queues, sidecars)

    assert report["passed_gate"] is True
    assert written["total"] == 3
    assert written["passed"] == 3
    assert written["missing_sidecar"] == 0
    assert written["duplicate_sidecar"] == 0
    assert written["reward_mismatch"] == 0
    assert written["invalid_label"] == 0
    assert written["unresolved_labels"] == 0


def test_full_census_gate_fails_missing_sidecar(tmp_path):
    queues = [_queue_row(tmp_path, "task_1"), _queue_row(tmp_path, "task_2")]
    sidecars = [_sidecar(tmp_path, queues[0])]

    report, _ = _run_gate(tmp_path, queues, sidecars)

    assert report["passed_gate"] is False
    assert report["missing_sidecar"] == 1
    assert report["errors"]["missing_sidecar"][0]["queue_id"] == "train/task_2/sample_00"


def test_full_census_gate_fails_duplicate_sidecar(tmp_path):
    queues = [_queue_row(tmp_path, "task_1")]
    sidecar = _sidecar(tmp_path, queues[0])

    report, _ = _run_gate(tmp_path, queues, [sidecar, dict(sidecar)])

    assert report["passed_gate"] is False
    assert report["duplicate_sidecar"] == 1
    assert report["errors"]["duplicate_sidecar"][0]["queue_id"] == "train/task_1/sample_00"


def test_full_census_gate_fails_reward_mismatch(tmp_path):
    queue = _queue_row(tmp_path, "task_1", 0.0)
    sidecar = _sidecar(tmp_path, queue, label="true_failure")
    sidecar["reported_reward"] = 1.0

    report, _ = _run_gate(tmp_path, [queue], [sidecar])

    assert report["passed_gate"] is False
    assert report["reward_mismatch"] == 1
    mismatch = report["errors"]["reward_mismatch"][0]
    assert mismatch["queue_reward"] == 0.0
    assert mismatch["sidecar_reward"] == 1.0


def test_full_census_gate_fails_invalid_label(tmp_path):
    queue = _queue_row(tmp_path, "task_1")
    sidecar = _sidecar(tmp_path, queue)
    sidecar["visual_label"] = "maybe_done"

    report, _ = _run_gate(tmp_path, [queue], [sidecar])

    assert report["passed_gate"] is False
    assert report["invalid_label"] == 1
    assert report["errors"]["invalid_label"][0]["visual_label"] == "maybe_done"


def test_full_census_gate_fails_unresolved_labels(tmp_path):
    queue = _queue_row(tmp_path, "task_1", 0.0)
    sidecar = _sidecar(tmp_path, queue, label="ambiguous_needs_evaluator_probe")

    report, _ = _run_gate(tmp_path, [queue], [sidecar])

    assert report["passed_gate"] is False
    assert report["unresolved_labels"] == 1
    unresolved = report["errors"]["unresolved_labels"][0]
    assert unresolved["visual_label"] == "ambiguous_needs_evaluator_probe"


def test_full_census_gate_fails_empty_required_artifact_paths(tmp_path):
    queue = _queue_row(tmp_path, "task_1")
    sidecar = _sidecar(tmp_path, queue)
    sidecar["result_json_path"] = ""
    sidecar["actions_path"] = ""
    sidecar["response_path"] = ""

    report, written = _run_gate(tmp_path, [queue], [sidecar])

    assert report["passed_gate"] is False
    assert report["passed"] == 0
    assert report["missing_sidecar"] == 0
    assert report["path_missing"] == 1
    missing_paths = report["errors"]["path_missing"][0]["missing_paths"]
    assert "result_json_path=<empty>" in missing_paths
    assert "actions_path=<empty>" in missing_paths
    assert "response_path=<empty>" in missing_paths
    assert written["path_missing"] == 1
    assert written["passed_gate"] is False


def test_full_census_gate_fails_insufficient_frame_evidence(tmp_path):
    queue = _queue_row(tmp_path, "task_1")
    sidecar = _sidecar(tmp_path, queue)
    sidecar["frame_paths"] = {"initial": sidecar["frame_paths"]["initial"]}

    report, written = _run_gate(tmp_path, [queue], [sidecar])

    assert report["passed_gate"] is False
    assert report["passed"] == 0
    assert report["insufficient_frames"] == 1
    assert report["errors"]["insufficient_frames"][0]["queue_id"] == "train/task_1/sample_00"
    assert written["insufficient_frames"] == 1


def test_full_census_gate_writes_exact_summary_counts(tmp_path):
    queues = [
        _queue_row(tmp_path, "pass", 1.0),
        _queue_row(tmp_path, "missing", 1.0),
        _queue_row(tmp_path, "duplicate", 1.0),
        _queue_row(tmp_path, "mismatch", 0.0),
        _queue_row(tmp_path, "unresolved", 0.0),
    ]
    pass_sidecar = _sidecar(tmp_path, queues[0])
    duplicate_sidecar = _sidecar(tmp_path, queues[2])
    mismatch_sidecar = _sidecar(tmp_path, queues[3], label="true_failure")
    mismatch_sidecar["reported_reward"] = 1.0
    unresolved_sidecar = _sidecar(tmp_path, queues[4], label="not_visually_decidable")

    report, _ = _run_gate(
        tmp_path,
        queues,
        [
            pass_sidecar,
            duplicate_sidecar,
            dict(duplicate_sidecar),
            mismatch_sidecar,
            unresolved_sidecar,
        ],
    )

    assert report["total"] == 5
    assert report["passed"] == 1
    assert report["missing_sidecar"] == 1
    assert report["duplicate_sidecar"] == 1
    assert report["reward_mismatch"] == 1
    assert report["unresolved_labels"] == 1
    assert report["passed_gate"] is False


def test_build_queue_uses_sample_summary_and_parquet_images_without_turn_dirs(tmp_path):
    mod = _load_module()
    log_root = tmp_path / "rollout"
    sample_dir = log_root / "train" / "task_1" / "sample_00"
    sample_dir.mkdir(parents=True)
    (sample_dir / "summary.json").write_text(
        json.dumps({
            "n_turns": 2,
            "episode_return": 1.0,
            "terminated": True,
            "truncated": False,
        }),
        encoding="utf-8",
    )
    pd.DataFrame([{
        "images": [
            f"turn_0000/{PROMPT_IMAGES_DIR}/0000_reset.png",
            f"turn_0001/{PROMPT_IMAGES_DIR}/0001_from_call_0000.png",
        ],
        "messages": [],
        "metadata": {
            "others": {
                "env_id": "lite.scalecua",
                "task_id": "task_1",
                "episode_return": 0.0,
                "terminated": False,
                "truncated": True,
            },
        },
    }]).to_parquet(sample_dir / "trajectory.parquet")

    rows = mod.build_queue(
        log_root,
        catalog={
            "task_1": {
                "split": "train",
                "instruction": "Do the task",
                "domain": "chrome",
                "related_app_domain": "chrome",
            },
        },
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["queue_id"] == "train/task_1/sample_00"
    assert row["reported_reward"] == 1.0
    assert row["reported_terminated"] is True
    assert row["reported_truncated"] is False
    assert row["frame_paths"] == {
        "initial": f"turn_0000/{PROMPT_IMAGES_DIR}/0000_reset.png",
        "final": f"turn_0001/{PROMPT_IMAGES_DIR}/0001_from_call_0000.png",
    }


def test_build_queue_prefers_top_level_metadata_over_stale_others(tmp_path):
    mod = _load_module()
    log_root = tmp_path / "rollout"
    sample_dir = log_root / "train" / "path_task" / "sample_00"
    sample_dir.mkdir(parents=True)
    pd.DataFrame([{
        "images": [],
        "messages": [],
        "metadata": {
            "env_id": "lite.scalecua",
            "task_id": "task_current",
            "episode_return": 0.0,
            "terminated": False,
            "truncated": False,
            "others": {
                "env_id": "historical.env",
                "task_id": "task_stale",
                "episode_return": 1.0,
                "terminated": True,
                "truncated": True,
            },
        },
    }]).to_parquet(sample_dir / "trajectory.parquet")

    rows = mod.build_queue(
        log_root,
        catalog={
            "task_current": {
                "split": "train",
                "instruction": "Do the task",
                "domain": "chrome",
                "related_app_domain": "chrome",
            },
        },
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["env_id"] == "lite.scalecua"
    assert row["task_id"] == "task_current"
    assert row["reported_reward"] == 0.0
    assert row["reported_terminated"] is False
    assert row["reported_truncated"] is False
    assert row["domain"] == "chrome"


def test_build_queue_uses_historical_others_durable_metadata_when_top_level_absent(tmp_path):
    mod = _load_module()
    log_root = tmp_path / "rollout"
    sample_dir = log_root / "train" / "path_task" / "sample_00"
    sample_dir.mkdir(parents=True)
    pd.DataFrame([{
        "images": [],
        "messages": [],
        "metadata": {
            "others": {
                "env_id": "legacy.scalecua",
                "task_id": "task_legacy",
                "episode_return": 0.0,
                "terminated": False,
                "truncated": True,
            },
        },
    }]).to_parquet(sample_dir / "trajectory.parquet")

    rows = mod.build_queue(
        log_root,
        catalog={
            "task_legacy": {
                "split": "train",
                "instruction": "Do the task",
                "domain": "chrome",
                "related_app_domain": "chrome",
            },
        },
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["env_id"] == "legacy.scalecua"
    assert row["task_id"] == "task_legacy"
    assert row["reported_reward"] == 0.0
    assert row["reported_terminated"] is False
    assert row["reported_truncated"] is True
    assert row["domain"] == "chrome"


def test_build_queue_falls_back_to_image_bearing_turn_dirs_case_b(tmp_path):
    mod = _load_module()
    log_root = tmp_path / "rollout"
    sample_dir = log_root / "train" / "task_1" / "sample_00"
    turn0 = sample_dir / "turn_0000"
    turn1 = sample_dir / "turn_0001"
    turn0.mkdir(parents=True)
    turn1.mkdir(parents=True)
    (sample_dir / "summary.json").write_text(
        json.dumps({
            "episode_return": 1.0,
            "terminated": True,
            "truncated": False,
        }),
        encoding="utf-8",
    )
    (turn0 / RESULTS_JSON).write_text(json.dumps({"reward": None}), encoding="utf-8")
    (turn1 / RESULTS_JSON).write_text(json.dumps({"reward": 1.0}), encoding="utf-8")
    image = turn1 / PROMPT_IMAGES_DIR / "0000_from_call_0000.png"
    annotated = turn1 / PROMPT_IMAGES_ANNOTATED_DIR / "0000_from_call_0000.png"
    _touch(image)
    _touch(annotated)
    pd.DataFrame([{
        "images": [],
        "messages": [],
        "metadata": {
            "others": {
                "env_id": "lite.scalecua",
                "task_id": "task_1",
            },
        },
    }]).to_parquet(sample_dir / "trajectory.parquet")

    rows = mod.build_queue(
        log_root,
        catalog={
            "task_1": {
                "split": "train",
                "instruction": "Do the task",
                "domain": "chrome",
                "related_app_domain": "chrome",
            },
        },
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["latest_turn"] == "turn_0001"
    assert row["frame_paths"] == {
        "initial": str(image),
        "final": str(image),
        "final_annotated": str(annotated),
    }


def test_build_queue_reads_current_turn_result_and_prompt_image_layout(tmp_path):
    mod = _load_module()
    log_root = tmp_path / "rollout"
    sample_dir = log_root / "train" / "task_1" / "sample_00"
    turn0 = sample_dir / "turn_0000"
    turn1 = sample_dir / "turn_0001"
    turn0.mkdir(parents=True)
    turn1.mkdir(parents=True)
    (sample_dir / "summary.json").write_text(
        json.dumps({
            "episode_return": 1.0,
            "terminated": True,
            "truncated": False,
        }),
        encoding="utf-8",
    )
    (turn0 / RESULTS_JSON).write_text(json.dumps({"reward": None}), encoding="utf-8")
    (turn1 / RESULTS_JSON).write_text(json.dumps({"reward": 1.0}), encoding="utf-8")
    image = turn1 / PROMPT_IMAGES_DIR / "0001_from_call_0000.png"
    annotated = turn1 / PROMPT_IMAGES_ANNOTATED_DIR / "0001_from_call_0000.png"
    _touch(image)
    _touch(annotated)
    pd.DataFrame([{
        "images": [],
        "messages": [],
        "metadata": {
            "others": {
                "env_id": "lite.scalecua",
                "task_id": "task_1",
            },
        },
    }]).to_parquet(sample_dir / "trajectory.parquet")

    rows = mod.build_queue(
        log_root,
        catalog={
            "task_1": {
                "split": "train",
                "instruction": "Do the task",
                "domain": "chrome",
                "related_app_domain": "chrome",
            },
        },
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["latest_turn"] == "turn_0001"
    assert row["frame_paths"] == {
        "initial": str(image),
        "final": str(image),
        "final_annotated": str(annotated),
    }


def test_make_batch_prompt_data_forces_local_catalog_scope(monkeypatch):
    rollout_dir = Path(__file__).resolve().parents[1]
    script = rollout_dir / "make_batch_prompt_data.py"
    monkeypatch.syspath_prepend(str(rollout_dir))
    monkeypatch.delitem(sys.modules, "common", raising=False)
    monkeypatch.setenv("CUA_LITE_ENV_SERVER_URL", "http://remote.invalid")
    monkeypatch.setenv("CUA_LITE_ENV_SERVER_TOKEN", "secret")

    spec = importlib.util.spec_from_file_location(
        "scalecua_make_batch_prompt_data_under_test",
        script,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert "CUA_LITE_ENV_SERVER_URL" not in module.os.environ
    assert "CUA_LITE_ENV_SERVER_TOKEN" not in module.os.environ
