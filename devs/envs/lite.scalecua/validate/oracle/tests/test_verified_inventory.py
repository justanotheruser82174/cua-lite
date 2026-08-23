from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path


def _load_verified_inventory_module():
    path = Path(__file__).resolve().parents[1] / "verified_inventory.py"
    spec = importlib.util.spec_from_file_location("scalecua_verified_inventory", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_verified_oracle_inventory_counts_current_unique_fixture_evidence(tmp_path):
    verified_inventory = _load_verified_inventory_module()
    catalog = tmp_path / "catalog"
    fixtures_dir = tmp_path / "fixtures"
    reports_dir = tmp_path / "reports"

    _write_jsonl(
        catalog / "rl.jsonl",
        [
            {
                "task_id": "task_ok",
                "instruction": "do ok",
                "metadata": {"others": {"domain": "chrome"}},
            },
            {
                "task_id": "task_missing_artifact",
                "instruction": "do missing",
                "metadata": {"others": {"domain": "chrome"}},
            },
        ],
    )
    _write_jsonl(
        fixtures_dir / "fixtures.jsonl",
        [
            {
                "fixture_id": "fixture_ok",
                "task_id": "task_ok",
                "split": "rl",
                "expected_pre_reward": 0.0,
                "expected_reward": 1.0,
                "oracle_actions": [{"type": "execute", "parameters": {"command": "true"}}],
            },
            {
                "fixture_id": "fixture_missing_artifact",
                "task_id": "task_missing_artifact",
                "split": "rl",
                "oracle_actions": [{"type": "execute", "parameters": {"command": "true"}}],
            },
        ],
    )

    valid_ok = {
        "fixture_id": "fixture_ok",
        "task_id": "task_ok",
        "passed": True,
        "precheck": {"passed": True, "reward": 0.0, "split": "rl", "task_id": "task_ok"},
        "replay": {
            "passed": True,
            "reward": 1.0,
            "expected_reward": 1.0,
            "split": "rl",
            "task_id": "task_ok",
            "trace": [{"kind": "oracle_action"}],
            "eval_debug": {"scores": [1.0]},
        },
    }
    valid_missing_artifact = copy.deepcopy(valid_ok)
    valid_missing_artifact["fixture_id"] = "fixture_missing_artifact"
    valid_missing_artifact["task_id"] = "task_missing_artifact"
    valid_missing_artifact["precheck"]["task_id"] = "task_missing_artifact"
    valid_missing_artifact["replay"]["task_id"] = "task_missing_artifact"
    stale = copy.deepcopy(valid_ok)
    stale["fixture_id"] = "fixture_stale"
    stale["task_id"] = "task_stale"
    stale["precheck"]["task_id"] = "task_stale"
    stale["replay"]["task_id"] = "task_stale"
    _write_jsonl(
        reports_dir / "run.report.jsonl",
        [valid_ok, copy.deepcopy(valid_ok), valid_missing_artifact, stale],
    )
    artifact_dir = reports_dir / "run" / "fixture_ok"
    artifact_dir.mkdir(parents=True)
    for name in verified_inventory.REQUIRED_ARTIFACT_FILES:
        (artifact_dir / name).write_bytes(b"x")
    (artifact_dir / "result.json").write_text(
        json.dumps(
            {
                "fixture_id": "fixture_ok",
                "task_id": "task_ok",
                "passed": True,
            }
        ),
        encoding="utf-8",
    )

    report = verified_inventory.build_report(
        [fixtures_dir],
        [reports_dir],
        splits=("rl",),
        catalog_dir=catalog,
        require_artifacts=True,
        top=10,
    )
    summary = report["summary"]
    assert summary["current_fixture_rows"] == 2
    assert summary["strict_valid_report_rows"] == 3
    assert summary["strict_valid_unique_fixtures_ignoring_artifacts"] == 2
    assert summary["verified_fixture_rows"] == 1
    assert summary["duplicate_strict_valid_report_rows"] == 1
    assert summary["stale_report_rows"] == 1
    assert summary["artifact_missing_strict_valid_report_rows"] == 1

    relaxed = verified_inventory.build_report(
        [fixtures_dir],
        [reports_dir],
        splits=("rl",),
        catalog_dir=catalog,
        require_artifacts=False,
        top=10,
    )
    assert relaxed["summary"]["verified_fixture_rows"] == 2


def test_verified_oracle_inventory_finds_artifacts_from_explicit_artifact_root(tmp_path):
    verified_inventory = _load_verified_inventory_module()
    catalog = tmp_path / "catalog"
    fixtures_dir = tmp_path / "fixtures"
    reports_dir = tmp_path / "reports"

    _write_jsonl(
        catalog / "rl.jsonl",
        [
            {
                "task_id": "task_ok",
                "instruction": "do ok",
                "metadata": {"others": {"domain": "os"}},
            },
        ],
    )
    _write_jsonl(
        fixtures_dir / "fixtures.jsonl",
        [
            {
                "fixture_id": "fixture_ok",
                "task_id": "task_ok",
                "split": "rl",
                "oracle_actions": [{"type": "execute", "parameters": {"command": "true"}}],
            },
        ],
    )
    _write_jsonl(
        reports_dir / "run.remaining-after-c8.c16.report.jsonl",
        [
            {
                "fixture_id": "fixture_ok",
                "task_id": "task_ok",
                "passed": True,
                "precheck": {
                    "passed": True,
                    "reward": 0.0,
                    "split": "rl",
                    "task_id": "task_ok",
                },
                "replay": {
                    "passed": True,
                    "reward": 1.0,
                    "expected_reward": 1.0,
                    "split": "rl",
                    "task_id": "task_ok",
                    "trace": [{"kind": "oracle_action"}],
                    "eval_debug": {"scores": [1.0]},
                },
            },
        ],
    )

    artifact_dir = reports_dir / "run-c16" / "fixture_ok"
    artifact_dir.mkdir(parents=True)
    for name in verified_inventory.REQUIRED_ARTIFACT_FILES:
        (artifact_dir / name).write_bytes(b"x")
    (artifact_dir / "result.json").write_text(
        json.dumps(
            {
                "fixture_id": "fixture_ok",
                "task_id": "task_ok",
                "passed": True,
            }
        ),
        encoding="utf-8",
    )

    report = verified_inventory.build_report(
        [fixtures_dir],
        [reports_dir],
        splits=("rl",),
        catalog_dir=catalog,
        require_artifacts=True,
        top=10,
    )

    assert report["summary"]["strict_valid_unique_fixtures_ignoring_artifacts"] == 1
    assert report["summary"]["verified_fixture_rows"] == 1
    assert report["verified_fixtures"][0]["artifact_dir"] == str(artifact_dir)
