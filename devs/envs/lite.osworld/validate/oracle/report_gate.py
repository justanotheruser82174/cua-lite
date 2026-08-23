"""Completeness gate for lite.osworld oracle-validation reports.

The validator writes one JSON object per validated task. This gate checks that a
finished report covers exactly the solvable rows from the source JSONL: no
missing rows, no duplicates, no unknown rows, and no failed records.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def _is_solvable(row: dict[str, Any]) -> bool:
    others = (row.get("metadata") or {}).get("others") or {}
    has_oracle = bool(others.get("oracle_actions"))
    return has_oracle and not others.get("exclude_reason")


def expected_task_ids(data_path: Path) -> dict[str, int]:
    expected: dict[str, int] = {}
    with data_path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if _is_solvable(row):
                expected[row["task_id"]] = line_no
    return expected


def load_report(report_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not report_path.exists():
        return rows, [f"missing report: {report_path}"]
    with report_path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"bad json line {line_no}: {exc}")
    return rows, errors


def classify_failure(row: dict[str, Any]) -> str:
    message = str(row.get("message") or "")
    if row.get("trivial_pass") or "trivial_pass:" in message:
        return "trivial_pass"
    if "Oracle scored" in message:
        return "oracle_scored_0"
    if "timed out" in message:
        return "timeout"
    if "Exception:" in message:
        return "exception"
    if "No oracle_actions" in message:
        return "oracle_missing"
    return "other"


def summarize(data_path: Path, report_path: Path) -> dict[str, Any]:
    expected = expected_task_ids(data_path)
    rows, errors = load_report(report_path)

    ids = [row.get("task_id") for row in rows]
    counts = collections.Counter(ids)
    duplicate_ids = sorted((task_id, count) for task_id, count in counts.items() if count > 1)
    unknown_ids = sorted(task_id for task_id in counts if task_id not in expected)
    missing_ids = sorted(set(expected) - set(counts))
    failures = [row for row in rows if row.get("passed") is False]

    bad_shape: list[str] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row.get("task_id"), str):
            bad_shape.append(f"report line {index}: task_id is not a string")
        if not isinstance(row.get("passed"), bool):
            bad_shape.append(f"report line {index}: passed is not a bool")
        if not isinstance(row.get("trivial_pass"), bool):
            bad_shape.append(f"report line {index}: trivial_pass is not a bool")
        if not isinstance(row.get("message"), str):
            bad_shape.append(f"report line {index}: message is not a string")

    failure_classes: dict[str, list[str]] = collections.defaultdict(list)
    for row in failures:
        failure_classes[classify_failure(row)].append(str(row.get("task_id")))

    gate_errors = list(errors)
    if duplicate_ids:
        gate_errors.append(f"duplicate task_ids: {duplicate_ids[:20]}")
    if unknown_ids:
        gate_errors.append(f"unknown task_ids: {unknown_ids[:20]}")
    if missing_ids:
        gate_errors.append(f"missing {len(missing_ids)} task_ids; first={missing_ids[:20]}")
    if bad_shape:
        gate_errors.append(f"bad record shape: {bad_shape[:20]}")
    if failures:
        gate_errors.append(f"failed records: {len(failures)}")

    return {
        "data": str(data_path),
        "report": str(report_path),
        "expected": len(expected),
        "report_rows": len(rows),
        "unique": len(counts),
        "passed": sum(row.get("passed") is True for row in rows),
        "failed": len(failures),
        "duplicates": duplicate_ids,
        "unknown": unknown_ids,
        "missing": missing_ids,
        "bad_shape": bad_shape,
        "failure_classes": dict(failure_classes),
        "ok": not gate_errors,
        "errors": gate_errors,
    }


def _print_text(summary: dict[str, Any]) -> None:
    print(
        f"expected={summary['expected']} report_rows={summary['report_rows']} "
        f"unique={summary['unique']} passed={summary['passed']} "
        f"failed={summary['failed']}"
    )
    if summary["duplicates"]:
        print(f"duplicates={len(summary['duplicates'])}")
    if summary["unknown"]:
        print(f"unknown={len(summary['unknown'])}")
    if summary["missing"]:
        print(f"missing={len(summary['missing'])}")
    if summary["bad_shape"]:
        print(f"bad_shape={len(summary['bad_shape'])}")
    for name, task_ids in sorted(summary["failure_classes"].items()):
        print(f"{name}={len(task_ids)}")
        for task_id in task_ids[:50]:
            print(f"  {task_id}")
    if summary["ok"]:
        print("GATE: PASS")
        return
    print("GATE: FAIL")
    for error in summary["errors"]:
        print(f" - {error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True,
                        help="Same task-rows JSONL passed to validate.py --fixtures.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary.")
    args = parser.parse_args(argv)

    summary = summarize(args.fixtures, args.report)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_text(summary)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
