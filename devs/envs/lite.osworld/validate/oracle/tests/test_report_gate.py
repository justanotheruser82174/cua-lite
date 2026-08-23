from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_gate_module():
    path = Path(__file__).resolve().parents[1] / "report_gate.py"
    spec = importlib.util.spec_from_file_location("_lite_osworld_report_gate_under_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _task(task_id: str, *, oracle: bool = True, excluded: bool = False) -> dict:
    others = {}
    if oracle:
        others["oracle_actions"] = [{"type": "execute", "parameters": {"command": "true"}}]
    if excluded:
        others["exclude_reason"] = "infeasible"
    return {"task_id": task_id, "metadata": {"others": others}}


def _record(task_id: str, *, passed: bool = True, message: str = "ok") -> dict:
    return {
        "task_id": task_id,
        "passed": passed,
        "trivial_pass": message.startswith("trivial_pass:"),
        "message": message,
    }


def test_osworld_report_gate_passes_complete_solvable_report(tmp_path):
    mod = _load_gate_module()
    data = tmp_path / "data.jsonl"
    report = tmp_path / "report.jsonl"
    _write_jsonl(
        data,
        [
            _task("a"),
            _task("b"),
            _task("excluded", excluded=True),
            _task("no_oracle", oracle=False),
        ],
    )
    _write_jsonl(report, [_record("a"), _record("b")])

    summary = mod.summarize(data, report)

    assert summary["ok"] is True
    assert summary["expected"] == 2
    assert summary["report_rows"] == 2
    assert summary["passed"] == 2


def test_osworld_report_gate_rejects_missing_duplicate_unknown_and_failed(tmp_path):
    mod = _load_gate_module()
    data = tmp_path / "data.jsonl"
    report = tmp_path / "report.jsonl"
    _write_jsonl(data, [_task("a"), _task("b"), _task("c")])
    _write_jsonl(
        report,
        [
            _record("a"),
            _record("a"),
            _record("unknown"),
            _record("b", passed=False, message="trivial_pass: eval returns 1.0"),
        ],
    )

    summary = mod.summarize(data, report)

    assert summary["ok"] is False
    assert summary["missing"] == ["c"]
    assert summary["duplicates"] == [("a", 2)]
    assert summary["unknown"] == ["unknown"]
    assert summary["failure_classes"] == {"trivial_pass": ["b"]}
