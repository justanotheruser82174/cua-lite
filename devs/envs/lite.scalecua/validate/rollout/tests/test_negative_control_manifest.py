from __future__ import annotations

import json
from pathlib import Path

import pytest

from lite.gym.envs.lite.scalecua.src.utils import dataset


def test_scalecua_negative_control_corpus_is_pinned_and_current():
    path = Path(__file__).resolve().parents[2] / "samples/negative_control_corpus.jsonl"
    required = {
        "g1_7_cf",
        "g1_7_formula",
        "g1_7_wshd",
        "g1_8_scope_argv",
        "g1_8_ext_path",
        "g1_8_string",
        "flush_vlc",
        "flush_thunderbird",
    }

    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            missing = {
                "loosening",
                "task_id",
                "split",
                "domain",
                "artifact_source",
                "gate_eligible",
                "expected_negative_reward",
            } - set(row)
            assert not missing, f"{path}:{line_number}: missing {sorted(missing)}"
            assert row["expected_negative_reward"] == 0.0
            if row["gate_eligible"]:
                _validate_collected_negative_control_row(row, source_path=path)
                rows.append(row)
                continue

            missing = {
                "fixture_id",
                "coverage_result",
                "coverage_func",
                "negative_source",
                "expected_positive_reward",
                "evidence_result_json",
                "evidence_summary",
            } - set(row)
            assert not missing, f"{path}:{line_number}: missing {sorted(missing)}"
            assert row["negative_source"] == "oracle_noop_precheck"
            assert row["artifact_source"] == "oracle_noop_scaffold"
            assert row["gate_eligible"] is False
            assert row["expected_positive_reward"] == 1.0
            assert isinstance(row["coverage_result"], list)
            assert isinstance(row["coverage_func"], list)
            summary = row["evidence_summary"]
            assert summary == {
                "passed": True,
                "fixture_id": row["fixture_id"],
                "task_id": row["task_id"],
                "precheck_reward": row["expected_negative_reward"],
                "replay_reward": row["expected_positive_reward"],
            }
            evidence = Path(row["evidence_result_json"])
            if evidence.is_file():
                result = json.loads(evidence.read_text(encoding="utf-8"))
                assert result["fixture_id"] == row["fixture_id"]
                assert result["task_id"] == row["task_id"]
                assert result["passed"] is True
                assert result["precheck"]["reward"] == row["expected_negative_reward"]
                assert result["replay"]["reward"] == row["expected_positive_reward"]
            rows.append(row)

    assert rows, "negative-control corpus must not be empty"
    counts = {name: 0 for name in required}
    seen = set()
    for row in rows:
        assert row["loosening"] in required
        key = (
            row["loosening"],
            row["task_id"],
            row.get("fixture_id") or row.get("collected_log_dir"),
        )
        assert key not in seen
        seen.add(key)
        counts[row["loosening"]] += 1

    assert {name: count for name, count in counts.items() if count < 3} == {}

    catalog_ids: dict[str, set[str]] = {}
    for split in dataset.RUNTIME_SPLITS:
        catalog_ids[split] = {
            row["task_id"] for _, row in dataset.iter_jsonl(dataset.catalog_path(split))
        }
    for row in rows:
        assert row["task_id"] in catalog_ids[row["split"]]


def _validate_collected_negative_control_row(row: dict, *, source_path: Path) -> None:
    required = {
        "collected_log_dir",
        "sample_summary_json",
        "turn_frame_paths",
        "actions_path",
        "results_path",
        "feature_absent_review",
        "reviewer",
        "checked_at",
    }
    missing = required - set(row)
    assert not missing, f"{source_path}: gate-eligible row missing {sorted(missing)}"
    assert row["artifact_source"] == "collected_rollout"
    assert row["gate_eligible"] is True
    assert isinstance(row["turn_frame_paths"], list) and len(row["turn_frame_paths"]) >= 3
    assert row["feature_absent_review"] == {
        "initial_frame": "feature_absent",
        "key_mutation_frame": "feature_absent",
        "final_frame": "feature_absent",
    }
    for key in ("collected_log_dir", "sample_summary_json", "actions_path", "results_path"):
        assert Path(row[key]).exists(), f"{source_path}: missing {key} path {row[key]}"
    for frame in row["turn_frame_paths"]:
        assert Path(frame).is_file(), f"{source_path}: missing frame path {frame}"


def test_scalecua_negative_control_gate_eligible_rows_require_collected_evidence(tmp_path):
    scaffold = {
        "artifact_source": "oracle_noop_scaffold",
        "gate_eligible": False,
    }
    assert scaffold["gate_eligible"] is False

    with pytest.raises(AssertionError, match="gate-eligible row missing"):
        _validate_collected_negative_control_row(
            {
                "artifact_source": "collected_rollout",
                "gate_eligible": True,
                "turn_frame_paths": [],
            },
            source_path=tmp_path / "bad.jsonl",
        )

    sample_dir = tmp_path / "sample_0001"
    sample_dir.mkdir()
    summary = sample_dir / "summary.json"
    actions = sample_dir / "03_actions.json"
    results = sample_dir / "05_results.json"
    frames = [sample_dir / f"frame_{idx}.png" for idx in range(3)]
    summary.write_text("{}", encoding="utf-8")
    actions.write_text("[]", encoding="utf-8")
    results.write_text("{}", encoding="utf-8")
    for frame in frames:
        frame.write_bytes(b"png")

    _validate_collected_negative_control_row(
        {
            "artifact_source": "collected_rollout",
            "gate_eligible": True,
            "collected_log_dir": str(sample_dir),
            "sample_summary_json": str(summary),
            "turn_frame_paths": [str(frame) for frame in frames],
            "actions_path": str(actions),
            "results_path": str(results),
            "feature_absent_review": {
                "initial_frame": "feature_absent",
                "key_mutation_frame": "feature_absent",
                "final_frame": "feature_absent",
            },
            "reviewer": "unit-test",
            "checked_at": "2026-07-20T00:00:00Z",
        },
        source_path=tmp_path / "ok.jsonl",
    )

