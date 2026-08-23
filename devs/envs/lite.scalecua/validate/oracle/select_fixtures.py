"""Select lite.scalecua oracle fixture candidates.

This script does not invent oracle actions. It selects task ids whose setup and
eval families should be covered by curated oracle fixtures, then emits candidate
rows for a human to fill with `oracle_actions`.

Run:
    uv run python devs/envs/lite.scalecua/validate/oracle/select_fixtures.py \
        --coverage-target all-tasks --splits rl \
        --output .exps/validate/lite.scalecua/oracle/candidates/rl_full.candidates.current.jsonl \
        --report .exps/validate/lite.scalecua/oracle/candidates/rl_full.coverage.current.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"pyproject.toml not found above {__file__}")


REPO_ROOT = _repo_root()
DEFAULT_CATALOG_DIR = REPO_ROOT / "lite" / "gym" / "envs" / "lite" / "scalecua" / "data"


DOMAIN_BUDGETS = {
    "smoke": {
        "chrome": 32,
        "gimp": 16,
        "libreoffice_calc": 32,
        "libreoffice_impress": 40,
        "libreoffice_writer": 24,
        "multi_apps": 20,
        "os": 20,
        "thunderbird": 20,
        "vlc": 20,
        "vs_code": 24,
    },
    "recommended": {
        "chrome": 80,
        "gimp": 48,
        "libreoffice_calc": 96,
        "libreoffice_impress": 112,
        "libreoffice_writer": 72,
        "multi_apps": 48,
        "os": 48,
        "thunderbird": 40,
        "vlc": 48,
        "vs_code": 56,
    },
    "full": {
        "chrome": 224,
        "gimp": 112,
        "libreoffice_calc": 512,
        "libreoffice_impress": 352,
        "libreoffice_writer": 288,
        "multi_apps": 112,
        "os": 112,
        "thunderbird": 72,
        "vlc": 96,
        "vs_code": 128,
    },
}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def family_name(value: Any) -> str:
    text = str(value or "none")
    text = re.sub(r"__qw35sft2_[0-9a-f]+$", "", text)
    text = re.sub(r"_qw35sft2_[0-9a-f]+$", "", text)
    text = re.sub(r"__[0-9a-f]{6,}.*$", "", text)
    text = re.sub(r"_[0-9a-f]{16,}$", "", text)
    text = re.sub(r"__[A-Za-z0-9_-]*(task|traj)_verify.*$", "", text)
    if text == "is_utc_0":
        return "is_utc"
    return text


def _cfg_type(value: Any, index: int) -> str:
    values = _as_list(value)
    if not values:
        return "none"
    cfg = values[index] if index < len(values) else values[-1]
    if not isinstance(cfg, dict):
        return "none"
    return family_name(cfg.get("type"))


def _task_coverage(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata", {})
    evaluator = metadata.get("evaluator", {})
    setup = sorted(
        {
            str(action.get("type"))
            for action in metadata.get("config", []) or []
            if isinstance(action, dict) and action.get("type")
        }
    )
    postconfig = sorted(
        {
            str(action.get("type"))
            for action in evaluator.get("postconfig", []) or []
            if isinstance(action, dict) and action.get("type")
        }
    )
    funcs = [family_name(fn) for fn in _as_list(evaluator.get("func", "none"))] or ["none"]
    result = []
    expected = []
    combos = []
    for index, func in enumerate(funcs):
        rtype = _cfg_type(evaluator.get("result", {}), index)
        etype = _cfg_type(evaluator.get("expected", {}), index)
        result.append(rtype)
        expected.append(etype)
        combos.append([rtype, etype, func])
    return {
        "setup": setup,
        "postconfig": postconfig,
        "result": sorted(set(result)),
        "expected": sorted(set(expected)),
        "func": sorted(set(funcs)),
        "combo": combos,
    }


def _elements(coverage: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for key in ("setup", "postconfig", "result", "expected", "func"):
        for value in coverage.get(key, []):
            out.add(f"{key}:{value}")
    for combo in coverage.get("combo", []):
        out.add("combo:" + "\t".join(combo))
    return out


def _weight_elements(rows: list[dict[str, Any]]) -> Counter[str]:
    weights: Counter[str] = Counter()
    for row in rows:
        for elem in _elements(row["coverage"]):
            if elem.startswith("combo:"):
                weights[elem] += 8
            elif elem.startswith(("setup:", "postconfig:")):
                weights[elem] += 4
            else:
                weights[elem] += 2
    return weights


def _select_domain(rows: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    weights = _weight_elements(rows)
    uncovered = set(weights)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    candidates = sorted(rows, key=lambda row: row["task_id"])

    while len(selected) < budget and uncovered:
        best: dict[str, Any] | None = None
        best_gain: set[str] = set()
        best_score = -1
        for row in candidates:
            if row["task_id"] in selected_ids:
                continue
            gain = _elements(row["coverage"]) & uncovered
            score = sum(weights[elem] for elem in gain)
            if score > best_score or (score == best_score and len(gain) > len(best_gain)):
                best = row
                best_gain = gain
                best_score = score
        if best is None or not best_gain:
            break
        selected.append(best)
        selected_ids.add(best["task_id"])
        uncovered -= best_gain

    if len(selected) < budget:
        for row in candidates:
            if row["task_id"] in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row["task_id"])
            if len(selected) >= budget:
                break
    return selected


def _coverage_keys(row: dict[str, Any], *, include_postconfig: bool) -> set[tuple[Any, ...]]:
    coverage = row["coverage"]
    setup = tuple(coverage.get("setup") or ())
    postconfig = tuple(coverage.get("postconfig") or ())
    keys = set()
    for combo in coverage.get("combo", []) or [["none", "none", "none"]]:
        combo_key = tuple(combo[:3]) if isinstance(combo, list) else (str(combo),)
        key: tuple[Any, ...] = (row["split"], row["domain"], setup, combo_key)
        if include_postconfig:
            key += (postconfig,)
        keys.add(key)
    return keys


def _select_all_coverage(
    rows: list[dict[str, Any]],
    *,
    include_postconfig: bool,
) -> list[dict[str, Any]]:
    """Greedily select task ids until every setup/eval key has a fixture candidate."""

    row_keys = {
        row["task_id"]: _coverage_keys(row, include_postconfig=include_postconfig)
        for row in rows
    }
    uncovered = set().union(*row_keys.values()) if row_keys else set()
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    candidates = sorted(rows, key=lambda row: (row["domain"], row["split"], row["task_id"]))

    while uncovered:
        best: dict[str, Any] | None = None
        best_gain: set[tuple[Any, ...]] = set()
        for row in candidates:
            task_id = row["task_id"]
            if task_id in selected_ids:
                continue
            gain = row_keys[task_id] & uncovered
            if len(gain) > len(best_gain):
                best = row
                best_gain = gain
        if best is None or not best_gain:
            break
        selected.append(best)
        selected_ids.add(best["task_id"])
        uncovered -= best_gain
    return selected


def _catalog_path(split: str, catalog_dir: Path) -> Path:
    return catalog_dir / f"{split}.jsonl"


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def _load_rows(
    splits: tuple[str, ...],
    catalog_dir: Path = DEFAULT_CATALOG_DIR,
) -> list[dict[str, Any]]:
    """Load runnable task rows from the `catalog_dir` split JSONL catalogs.

    The oracle sidecar should keep working even when the lite.scalecua package is
    not importable from a file-executed script, so avoid depending on the runtime
    dataset module here.
    """

    rows: list[dict[str, Any]] = []
    for split in splits:
        path = _catalog_path(split, catalog_dir)
        if not path.exists():
            raise FileNotFoundError(f"catalog not found for split {split!r}: {path}")
        for _, row in _iter_jsonl(path):
            metadata = row.get("metadata", {})
            others = metadata.get("others", {})
            if others.get("exclude_reason"):
                continue
            domain = others.get("domain", "unknown")
            rows.append(
                {
                    "split": split,
                    "task_id": row["task_id"],
                    "domain": domain,
                    "instruction": row.get("instruction", ""),
                    "coverage": _task_coverage(row),
                }
            )
    return rows


def _candidate_fixture(row: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "fixture_id": f"oracle_{row['domain']}_{index:04d}",
        "split": row["split"],
        "task_id": row["task_id"],
        "domain": row["domain"],
        "coverage": row["coverage"],
        "oracle_actions": [],
        "notes": "candidate selected by oracle/select_fixtures.py; fill oracle before committing",
    }


def _coverage_report(
    rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    splits: tuple[str, ...],
) -> dict[str, Any]:
    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        elements = Counter()
        combos = Counter()
        setup = Counter()
        postconfig = Counter()
        for row in items:
            cov = row["coverage"]
            for elem in _elements(cov):
                elements[elem] += 1
            for combo in cov.get("combo", []):
                combos[tuple(combo)] += 1
            setup.update(cov.get("setup", []))
            postconfig.update(cov.get("postconfig", []))
        return {
            "rows": len(items),
            "elements": len(elements),
            "combo_families": len(combos),
            "combo_ge_10": sum(1 for count in combos.values() if count >= 10),
            "combo_ge_4": sum(1 for count in combos.values() if count >= 4),
            "setup_types": sorted(setup),
            "postconfig_types": sorted(postconfig),
        }

    by_domain = {}
    all_by_domain: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_by_domain: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        all_by_domain[row["domain"]].append(row)
    for row in selected:
        selected_by_domain[row["domain"]].append(row)
    for domain in sorted(all_by_domain):
        all_summary = summarize(all_by_domain[domain])
        selected_summary = summarize(selected_by_domain[domain])
        by_domain[domain] = {
            "available": all_summary,
            "selected": selected_summary,
        }
    def count_keys(items: list[dict[str, Any]], *, include_postconfig: bool) -> int:
        keys = set()
        for item in items:
            keys.update(_coverage_keys(item, include_postconfig=include_postconfig))
        return len(keys)

    return {
        "splits": list(splits),
        "domains": by_domain,
        "selected_total": len(selected),
        "coverage_keys": {
            "eval_setup_available": count_keys(rows, include_postconfig=False),
            "eval_setup_selected": count_keys(selected, include_postconfig=False),
            "full_available": count_keys(rows, include_postconfig=True),
            "full_selected": count_keys(selected, include_postconfig=True),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=sorted(DOMAIN_BUDGETS), default="smoke")
    parser.add_argument(
        "--coverage-target",
        choices=("tier-budget", "all-eval-setup", "all-full", "all-tasks"),
        default="tier-budget",
        help=(
            "tier-budget selects the configured per-domain count. all-eval-setup selects "
            "one or more candidates covering every split/domain/setup/eval combo. "
            "all-full additionally distinguishes evaluator postconfig sets. "
            "all-tasks selects every runnable task in the requested splits."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "rl"),
        default=["train", "rl"],
        help="Runtime splits to include in the candidate universe.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=DEFAULT_CATALOG_DIR,
        help="Directory containing rl.jsonl/train.jsonl task catalogs.",
    )
    args = parser.parse_args()

    splits = tuple(args.splits)
    rows = _load_rows(splits, args.catalog_dir)
    if args.coverage_target == "all-eval-setup":
        selected = _select_all_coverage(rows, include_postconfig=False)
    elif args.coverage_target == "all-full":
        selected = _select_all_coverage(rows, include_postconfig=True)
    elif args.coverage_target == "all-tasks":
        selected = sorted(rows, key=lambda row: (row["split"], row["domain"], row["task_id"]))
    else:
        by_domain: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_domain[row["domain"]].append(row)

        selected = []
        for domain, budget in DOMAIN_BUDGETS[args.tier].items():
            selected.extend(_select_domain(by_domain[domain], budget))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for index, row in enumerate(selected, 1):
            f.write(json.dumps(_candidate_fixture(row, index), ensure_ascii=False) + "\n")

    report = _coverage_report(rows, selected, splits)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
