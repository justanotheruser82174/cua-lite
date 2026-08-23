"""V1 static checks for train.perturb.jsonl.

Runs without launching containers (cf. validate/oracle/validate.py = V2 oracle smoke).

Checks:
  1. Required fields present (task_id, instruction, metadata)
  2. Unique task_ids
  3. No "save the file" instruction leak
  4. Non-empty oracle_actions
  5. No exact-instruction leak vs eval split
  6. (D6 NEW, warning) No (evaluator.func + evaluator.expected) tuple identical
     to any eval row — flagged as warning, not fatal, since paraphrase-only
     perturbations legitimately share evaluator+expected with their eval source.
  7. (NEW) No infeasible row — train.perturb should never claim infeasible.

Usage:
    uv run python devs/envs/lite.osworld/validate/static.py \\
        --data lite/gym/envs/lite/osworld/data/train.perturb.jsonl \\
        --eval lite/gym/envs/lite/osworld/data/eval.jsonl

Exit codes:
    0 — all hard checks pass (warnings only)
    1 — at least one hard check failed
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _evaluator_sig(ev: dict) -> tuple:
    """Return (func_canonical, expected_json_canonical) tuple for leak compare."""
    func = ev.get("func", "?")
    if isinstance(func, list):
        func = "+".join(sorted(func))
    expected = ev.get("expected", {})
    return (func, json.dumps(expected, sort_keys=True, default=str))


def check(perturb_path: Path, eval_path: Path | None = None) -> tuple[int, dict]:
    """Run V1 checks. Returns (exit_code, summary_dict)."""
    rows = []
    with perturb_path.open() as f:
        for line in f:
            rows.append(json.loads(line))

    eval_rows: list[dict] = []
    eval_instr: set[str] = set()
    eval_sigs: dict[tuple, str] = {}
    if eval_path is not None and eval_path.exists():
        with eval_path.open() as f:
            for line in f:
                er = json.loads(line)
                eval_rows.append(er)
                eval_instr.add(er["instruction"])
                ev = er["metadata"]["evaluator"]
                eval_sigs[_evaluator_sig(ev)] = er["task_id"]

    summary = {
        "total_rows": len(rows),
        "missing_fields": [],
        "dup_task_ids": [],
        "save_the_file_leaks": [],
        "empty_oracle_actions": [],
        "exact_instruction_leaks": [],
        "semantic_leaks_warning": [],
        "infeasible_in_train": [],
    }

    required = ["task_id", "instruction", "metadata"]
    ids: list[str] = []
    for r in rows:
        for k in required:
            if k not in r:
                summary["missing_fields"].append((r.get("task_id", "?"), k))
        ids.append(r["task_id"])
        meta = r.get("metadata", {})
        others = meta.get("others") or {}
        if "save the file" in r.get("instruction", "").lower():
            summary["save_the_file_leaks"].append(r["task_id"])
        if not others.get("oracle_actions"):
            summary["empty_oracle_actions"].append(r["task_id"])
        if r.get("instruction", "") in eval_instr:
            summary["exact_instruction_leaks"].append(r["task_id"])
        ev = meta.get("evaluator", {})
        # D6: warning-level semantic leak
        sig = _evaluator_sig(ev)
        if sig in eval_sigs:
            summary["semantic_leaks_warning"].append((r["task_id"], eval_sigs[sig]))
        # Infeasible in train is a bug
        func = ev.get("func", "")
        if func == "infeasible":
            summary["infeasible_in_train"].append(r["task_id"])

    for tid, count in Counter(ids).items():
        if count > 1:
            summary["dup_task_ids"].append((tid, count))

    # Hard fails
    hard_fail_keys = (
        "missing_fields",
        "dup_task_ids",
        "save_the_file_leaks",
        "empty_oracle_actions",
        "exact_instruction_leaks",
        "infeasible_in_train",
    )
    n_hard = sum(len(summary[k]) for k in hard_fail_keys)
    return (1 if n_hard else 0), summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="train.perturb.jsonl path")
    ap.add_argument("--eval", default=None, help="eval.jsonl path (for leak compare)")
    args = ap.parse_args()

    rc, summary = check(Path(args.data), Path(args.eval) if args.eval else None)

    print(f"V1 static checks on {args.data}")
    print(f"  total rows: {summary['total_rows']}")
    print()
    print("=== Hard checks ===")
    print(f"  missing required fields:    {len(summary['missing_fields'])}")
    print(f"  duplicate task_ids:         {len(summary['dup_task_ids'])}")
    print(f"  'save the file' leaks:      {len(summary['save_the_file_leaks'])}")
    print(f"  empty oracle_actions:       {len(summary['empty_oracle_actions'])}")
    print(f"  exact instruction leaks:    {len(summary['exact_instruction_leaks'])}")
    print(f"  infeasible rows in train:   {len(summary['infeasible_in_train'])}")
    print()
    print("=== Soft checks (warning) ===")
    n_sem = len(summary["semantic_leaks_warning"])
    print(f"  semantic leaks (D6):        {n_sem}"
          f"{'  ⚠️ same (func, expected) as some eval row — paraphrase-only is OK, config-clone is NOT' if n_sem else ''}")
    if n_sem:
        # Group by domain
        by_dom: dict[str, int] = defaultdict(int)
        for ptid, _etid in summary["semantic_leaks_warning"]:
            # domain is in task_id like perturb_osworld_<domain>_xxx
            parts = ptid.split("_")
            if len(parts) >= 3 and parts[0] == "perturb" and parts[1] == "osworld":
                dom = "_".join(parts[2:-2]) if len(parts) > 4 else parts[2]
                by_dom[dom] += 1
        for dom, c in sorted(by_dom.items(), key=lambda x: -x[1]):
            print(f"      {dom}: {c}")

    if rc:
        print("\nFAIL: hard checks have failures.")
        # Show first 5 of each non-empty failure
        for k in ("missing_fields", "dup_task_ids", "save_the_file_leaks",
                  "empty_oracle_actions", "exact_instruction_leaks", "infeasible_in_train"):
            v = summary[k]
            if v:
                print(f"\n  {k}:")
                for item in v[:5]:
                    print(f"    {item}")
    else:
        print("\nPASS: all hard checks ✓ (warnings if any: see above)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
