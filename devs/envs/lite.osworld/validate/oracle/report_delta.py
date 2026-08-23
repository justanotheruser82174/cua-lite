"""P0-vs-P3 baseline delta gate for oracle-validation reports.

``report_gate.py`` fails on ANY red row, so the handful of pre-existing
scalecua/corpus reds fail *both* the P0 baseline and the P3 candidate. The
floor is instead **0 NEW failures vs P0**: a candidate red that was already red
in the baseline is not a regression. This tool diffs two report JSONLs and fails
only when the candidate introduces a red row that was GREEN (or absent) in the
baseline.

Report-format-agnostic: it auto-detects the per-row key field —
``task_id`` (lite.osworld reports) or ``fixture_id`` (lite.scalecua reports) —
and keys on whichever the report carries (``fixture_id`` wins when both exist,
matching the scalecua schema). Both schemas use a ``passed`` bool per row.

Run:
    uv run python devs/envs/lite.osworld/validate/oracle/report_delta.py \\
        --baseline /tmp/p0_train_synth.report.jsonl \\
        --candidate /tmp/p3_train_synth.report.jsonl

    # machine-readable summary
    uv run python devs/envs/lite.osworld/validate/oracle/report_delta.py \\
        --baseline p0.jsonl --candidate p3.jsonl --json

Exit code: 0 if the candidate adds no new failures (pre-existing reds are
excluded), 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Preference order: scalecua rows carry BOTH fixture_id and task_id, so
# fixture_id (the stable per-fixture identity) is checked first.
_KEY_FIELDS = ("fixture_id", "task_id")


def load_report(report_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read a report JSONL into rows + parse errors (mirrors report_gate)."""
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


def detect_key_field(rows: list[dict[str, Any]]) -> str | None:
    """Auto-detect the row key field. ``fixture_id`` wins over ``task_id``."""
    for field in _KEY_FIELDS:
        if any(isinstance(row.get(field), str) for row in rows):
            return field
    return None


def index_by_key(rows: list[dict[str, Any]], key_field: str) -> dict[str, bool]:
    """Map key -> passed (bool). Last occurrence wins on duplicates.

    A row is GREEN iff ``passed is True``; anything else (False / missing) is
    treated as red so a malformed candidate row can never masquerade as a pass.
    """
    index: dict[str, bool] = {}
    for row in rows:
        key = row.get(key_field)
        if isinstance(key, str):
            index[key] = row.get("passed") is True
    return index


def compute_delta(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    baseline_rows, baseline_errors = load_report(baseline_path)
    candidate_rows, candidate_errors = load_report(candidate_path)

    errors = list(candidate_errors)
    # A malformed/missing baseline can't be trusted to exonerate a candidate red
    # (every red would look "pre-existing"), so surface it as a gate error.
    errors += [f"baseline: {e}" for e in baseline_errors]

    baseline_key = detect_key_field(baseline_rows)
    candidate_key = detect_key_field(candidate_rows)
    key_field = candidate_key or baseline_key

    if key_field is None:
        errors.append("could not detect a key field (task_id / fixture_id) in either report")
        return {
            "baseline": str(baseline_path),
            "candidate": str(candidate_path),
            "key_field": None,
            "new_failures": [],
            "preexisting_reds": [],
            "candidate_reds": [],
            "ok": False,
            "errors": errors,
        }
    if baseline_key and candidate_key and baseline_key != candidate_key:
        errors.append(
            f"baseline keyed on {baseline_key!r} but candidate on {candidate_key!r} "
            "— comparing incompatible report schemas"
        )

    baseline_pass = index_by_key(baseline_rows, key_field)
    candidate_pass = index_by_key(candidate_rows, key_field)

    candidate_reds = sorted(k for k, ok in candidate_pass.items() if not ok)
    # A candidate red is a NEW failure unless it was ALSO red in the baseline.
    # baseline_pass.get(k): True=green, False=red, None=absent. Green or absent
    # (``is not False``) ⇒ regression; red (``is False``) ⇒ pre-existing.
    new_failures = sorted(
        k for k in candidate_reds if baseline_pass.get(k) is not False
    )
    preexisting_reds = sorted(
        k for k in candidate_reds if baseline_pass.get(k) is False
    )

    ok = not new_failures and not errors
    return {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "key_field": key_field,
        "baseline_rows": len(baseline_pass),
        "candidate_rows": len(candidate_pass),
        "candidate_reds": candidate_reds,
        "preexisting_reds": preexisting_reds,
        "new_failures": new_failures,
        "ok": ok,
        "errors": errors,
    }


def _print_text(delta: dict[str, Any]) -> None:
    print(
        f"key_field={delta['key_field']} "
        f"baseline_rows={delta.get('baseline_rows', 0)} "
        f"candidate_rows={delta.get('candidate_rows', 0)} "
        f"candidate_reds={len(delta['candidate_reds'])} "
        f"preexisting_reds={len(delta['preexisting_reds'])} "
        f"new_failures={len(delta['new_failures'])}"
    )
    if delta["new_failures"]:
        print("NEW FAILURES (red in candidate, green/absent in baseline):")
        for key in delta["new_failures"]:
            print(f"  {key}")
    if delta["errors"]:
        for error in delta["errors"]:
            print(f" - {error}")
    print("DELTA GATE: PASS" if delta["ok"] else "DELTA GATE: FAIL")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="P0 baseline report JSONL.")
    parser.add_argument("--candidate", type=Path, required=True,
                        help="P3 candidate report JSONL to diff against the baseline.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary.")
    args = parser.parse_args(argv)

    delta = compute_delta(args.baseline, args.candidate)
    if args.json:
        print(json.dumps(delta, indent=2, sort_keys=True))
    else:
        _print_text(delta)
    return 0 if delta["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
