"""Audit lite.scalecua oracle fixture coverage against runnable train/rl tasks.

Run:
    uv run python devs/envs/lite.scalecua/validate/oracle/coverage_inventory.py \
        --fixtures lite/gym/envs/lite/scalecua/data/oracle --splits rl --require-clean \
        --output .exps/validate/lite.scalecua/oracle/coverage/rl.coverage.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"pyproject.toml not found above {__file__}")


REPO_ROOT = _repo_root()
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CATALOG_DIR = REPO_ROOT / "lite" / "gym" / "envs" / "lite" / "scalecua" / "data"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from select_fixtures import DOMAIN_BUDGETS, _elements, _task_coverage  # noqa: E402


def _load_catalog_rows(
    splits: tuple[str, ...],
    catalog_dir: Path = DEFAULT_CATALOG_DIR,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    runnable: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    all_by_task_id: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for split in splits:
        path = catalog_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"catalog not found for split {split!r}: {path}")
        for _, row in _read_jsonl(path):
            metadata = row.get("metadata", {})
            others = metadata.get("others", {})
            domain = others.get("domain", "unknown")
            normalized = {
                "split": split,
                "task_id": row["task_id"],
                "domain": domain,
                "instruction": row.get("instruction", ""),
                "exclude_reason": others.get("exclude_reason"),
                "coverage": _task_coverage(row),
            }
            all_by_task_id[row["task_id"]].append(normalized)
            if normalized["exclude_reason"]:
                continue
            runnable.append(normalized)
            by_key[(split, row["task_id"])] = normalized

    return runnable, by_key, all_by_task_id


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            rows.append((line_no, json.loads(line)))
    return rows


def _fixture_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.jsonl")))
        elif not path.exists():
            continue
        else:
            files.append(path)
    return sorted(dict.fromkeys(files))


def _tuplify(values: Any) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(str(value) for value in values)


def _combo_tuple(combo: Any) -> tuple[str, str, str]:
    if isinstance(combo, (list, tuple)) and len(combo) >= 3:
        return (str(combo[0]), str(combo[1]), str(combo[2]))
    return ("none", "none", str(combo or "none"))


def _coverage_combos(row: dict[str, Any]) -> list[dict[str, Any]]:
    coverage = row["coverage"]
    setup = _tuplify(coverage.get("setup"))
    postconfig = _tuplify(coverage.get("postconfig"))
    combos = coverage.get("combo") or [["none", "none", "none"]]
    out = []
    for combo in combos:
        result, expected, func = _combo_tuple(combo)
        out.append(
            {
                "split": row["split"],
                "domain": row["domain"],
                "setup": setup,
                "postconfig": postconfig,
                "result": result,
                "expected": expected,
                "func": func,
            }
        )
    return out


def _combo_key(combo: dict[str, Any], *, include_postconfig: bool) -> tuple[Any, ...]:
    key: tuple[Any, ...] = (
        combo["split"],
        combo["domain"],
        combo["setup"],
        combo["result"],
        combo["expected"],
        combo["func"],
    )
    if include_postconfig:
        key += (combo["postconfig"],)
    return key


def _combo_record(key: tuple[Any, ...], count: int, *, include_postconfig: bool) -> dict[str, Any]:
    split, domain, setup, result, expected, func, *rest = key
    record = {
        "count": count,
        "split": split,
        "domain": domain,
        "setup": list(setup),
        "eval_combo": [result, expected, func],
    }
    if include_postconfig:
        record["postconfig"] = list(rest[0])
    return record


def _load_fixtures(
    files: list[Path],
    by_key: dict[tuple[str, str], dict[str, Any]],
    all_by_task_id: dict[str, list[dict[str, Any]]],
    splits: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    matched: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    out_of_scope = 0
    split_set = set(splits)

    for file_path in files:
        for line_no, fixture in _read_jsonl(file_path):
            split = fixture.get("split")
            task_id = fixture.get("task_id")
            if split not in split_set:
                out_of_scope += 1
                continue
            row = by_key.get((split, task_id))
            problem: dict[str, Any] | None = None

            if row is None:
                candidates = all_by_task_id.get(str(task_id), [])
                excluded = [
                    candidate for candidate in candidates if candidate.get("exclude_reason")
                ]
                if excluded:
                    problem = {
                        "file": str(file_path),
                        "line": line_no,
                        "fixture_id": fixture.get("fixture_id"),
                        "split": split,
                        "task_id": task_id,
                        "problem": "fixture_points_to_excluded_task",
                        "exclude_reasons": sorted(
                            {str(item["exclude_reason"]) for item in excluded}
                        ),
                    }
                elif candidates:
                    problem = {
                        "file": str(file_path),
                        "line": line_no,
                        "fixture_id": fixture.get("fixture_id"),
                        "split": split,
                        "task_id": task_id,
                        "problem": "fixture_split_mismatch",
                        "candidate_splits": sorted({str(item["split"]) for item in candidates}),
                    }
                else:
                    problem = {
                        "file": str(file_path),
                        "line": line_no,
                        "fixture_id": fixture.get("fixture_id"),
                        "split": split,
                        "task_id": task_id,
                        "problem": "fixture_task_not_in_catalog",
                    }
                problems.append(problem)
                continue

            matched.append(
                {
                    "file": str(file_path),
                    "line": line_no,
                    "fixture_id": fixture.get("fixture_id"),
                    "split": split,
                    "task_id": task_id,
                    "domain": row["domain"],
                    "declared_domain": fixture.get("domain"),
                    "explicit_coverage": bool(fixture.get("coverage")),
                    "coverage": row["coverage"],
                    "action_kinds": _fixture_action_kinds(fixture),
                }
            )

    return matched, problems, out_of_scope


def _fixture_action_kinds(fixture: dict[str, Any]) -> list[str]:
    kinds = []
    for action in fixture.get("oracle_actions") or []:
        if isinstance(action, dict):
            kinds.append(str(action.get("type", "unknown")))
    return sorted(set(kinds)) or ["none"]


def _counter_for(
    rows: list[dict[str, Any]],
    *,
    include_postconfig: bool,
) -> Counter[tuple[Any, ...]]:
    counter: Counter[tuple[Any, ...]] = Counter()
    for row in rows:
        for combo in _coverage_combos(row):
            counter[_combo_key(combo, include_postconfig=include_postconfig)] += 1
    return counter


def _row_combo_keys(row: dict[str, Any], *, include_postconfig: bool) -> set[tuple[Any, ...]]:
    return {
        _combo_key(combo, include_postconfig=include_postconfig)
        for combo in _coverage_combos(row)
    }


def _type_counter(rows: list[dict[str, Any]], field: str) -> Counter[tuple[str, str, str]]:
    counter: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        values = row["coverage"].get(field) or []
        if not values:
            counter[(row["split"], row["domain"], "none")] += 1
            continue
        for value in values:
            counter[(row["split"], row["domain"], str(value))] += 1
    return counter


def _domain_summary(
    runnable: list[dict[str, Any]],
    fixtures: list[dict[str, Any]],
    budgets: dict[str, int],
) -> dict[str, dict[str, Any]]:
    domains = sorted({row["domain"] for row in runnable} | set(budgets))
    report: dict[str, dict[str, Any]] = {}
    for domain in domains:
        domain_rows = [row for row in runnable if row["domain"] == domain]
        domain_fixtures = [row for row in fixtures if row["domain"] == domain]
        available_eval_setup = set(_counter_for(domain_rows, include_postconfig=False))
        covered_eval_setup = set(_counter_for(domain_fixtures, include_postconfig=False))
        available_full = set(_counter_for(domain_rows, include_postconfig=True))
        covered_full = set(_counter_for(domain_fixtures, include_postconfig=True))
        setup_available = {key[2] for key in _type_counter(domain_rows, "setup")}
        setup_covered = {key[2] for key in _type_counter(domain_fixtures, "setup")}
        post_available = {key[2] for key in _type_counter(domain_rows, "postconfig")}
        post_covered = {key[2] for key in _type_counter(domain_fixtures, "postconfig")}
        fixture_ids = {(row["split"], row["task_id"]) for row in domain_fixtures}
        budget = budgets.get(domain, 0)
        report[domain] = {
            "runnable_rows": len(domain_rows),
            "fixture_rows": len(domain_fixtures),
            "unique_fixture_tasks": len(fixture_ids),
            "recommended_budget": budget,
            "recommended_add_needed_by_count": max(budget - len(fixture_ids), 0),
            "eval_setup_combos": {
                "available": len(available_eval_setup),
                "covered": len(covered_eval_setup & available_eval_setup),
                "missing": len(available_eval_setup - covered_eval_setup),
            },
            "full_setup_eval_postconfig_combos": {
                "available": len(available_full),
                "covered": len(covered_full & available_full),
                "missing": len(available_full - covered_full),
            },
            "setup_types": {
                "available": sorted(setup_available),
                "covered": sorted(setup_covered),
                "missing": sorted(setup_available - setup_covered),
            },
            "postconfig_types": {
                "available": sorted(post_available),
                "covered": sorted(post_covered),
                "missing": sorted(post_available - post_covered),
            },
        }
    return report


def _top_missing(
    available: Counter[tuple[Any, ...]],
    covered: Counter[tuple[Any, ...]],
    *,
    include_postconfig: bool,
    limit: int,
) -> list[dict[str, Any]]:
    missing = available.copy()
    for key in covered:
        missing.pop(key, None)
    return [
        _combo_record(key, count, include_postconfig=include_postconfig)
        for key, count in missing.most_common(limit)
    ]


def _duplicates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[(row["split"], row["task_id"])].append(row)
    out = []
    for (split, task_id), items in sorted(by_task.items()):
        if len(items) <= 1:
            continue
        out.append(
            {
                "split": split,
                "task_id": task_id,
                "fixture_ids": [str(item["fixture_id"]) for item in items],
                "files": sorted({item["file"] for item in items}),
            }
        )
    return out


def _candidate_priority(
    runnable: list[dict[str, Any]],
    fixtures: list[dict[str, Any]],
    *,
    candidate_split: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Greedily rank runnable RL rows that add currently missing oracle coverage."""

    fixture_task_ids = {(row["split"], row["task_id"]) for row in fixtures}
    available_eval_setup = _counter_for(runnable, include_postconfig=False)
    covered_eval_setup = _counter_for(fixtures, include_postconfig=False)
    available_full = _counter_for(runnable, include_postconfig=True)
    covered_full = _counter_for(fixtures, include_postconfig=True)
    missing_eval_setup = set(available_eval_setup) - set(covered_eval_setup)
    missing_full = set(available_full) - set(covered_full)
    domain_report = _domain_summary(
        [row for row in runnable if row["split"] == candidate_split],
        [row for row in fixtures if row["split"] == candidate_split],
        DOMAIN_BUDGETS["recommended"],
    )

    candidates = sorted(
        (
            row
            for row in runnable
            if row["split"] == candidate_split
            and (row["split"], row["task_id"]) not in fixture_task_ids
        ),
        key=lambda row: (row["domain"], row["task_id"]),
    )
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    while len(selected) < limit:
        best: dict[str, Any] | None = None
        best_gain_full: set[tuple[Any, ...]] = set()
        best_gain_eval: set[tuple[Any, ...]] = set()
        best_score: tuple[int, int, int, int] | None = None
        for row in candidates:
            if row["task_id"] in selected_ids:
                continue
            full_gain = _row_combo_keys(row, include_postconfig=True) & missing_full
            eval_gain = _row_combo_keys(row, include_postconfig=False) & missing_eval_setup
            if not full_gain and not eval_gain:
                continue
            domain_gap = domain_report.get(row["domain"], {}).get(
                "recommended_add_needed_by_count",
                0,
            )
            full_weight = sum(available_full[key] for key in full_gain)
            eval_weight = sum(available_eval_setup[key] for key in eval_gain)
            score = (
                len(full_gain),
                len(eval_gain),
                full_weight + eval_weight,
                domain_gap,
            )
            if best_score is None or score > best_score:
                best = row
                best_gain_full = full_gain
                best_gain_eval = eval_gain
                best_score = score
        if best is None or best_score is None:
            break
        selected_ids.add(best["task_id"])
        missing_full -= best_gain_full
        missing_eval_setup -= best_gain_eval
        selected.append(
            {
                "rank": len(selected) + 1,
                "split": best["split"],
                "task_id": best["task_id"],
                "domain": best["domain"],
                "instruction": best["instruction"],
                "new_full_combo_count": len(best_gain_full),
                "new_eval_setup_combo_count": len(best_gain_eval),
                "domain_recommended_add_remaining_before_selection": domain_report.get(
                    best["domain"], {}
                ).get("recommended_add_needed_by_count", 0),
                "coverage": best["coverage"],
                "coverage_elements": sorted(_elements(best["coverage"])),
            }
        )
    return selected


def build_report(
    fixture_paths: list[Path],
    top: int,
    splits: tuple[str, ...],
    *,
    catalog_dir: Path = DEFAULT_CATALOG_DIR,
    candidate_split: str = "rl",
    candidate_limit: int = 40,
) -> dict[str, Any]:
    runnable, by_key, all_by_task_id = _load_catalog_rows(splits, catalog_dir)
    files = _fixture_files(fixture_paths)
    fixtures, problems, out_of_scope = _load_fixtures(files, by_key, all_by_task_id, splits)

    available_eval_setup = _counter_for(runnable, include_postconfig=False)
    covered_eval_setup = _counter_for(fixtures, include_postconfig=False)
    available_full = _counter_for(runnable, include_postconfig=True)
    covered_full = _counter_for(fixtures, include_postconfig=True)
    fixture_task_ids = {(row["split"], row["task_id"]) for row in fixtures}
    action_kinds = Counter(kind for row in fixtures for kind in row["action_kinds"])

    report = {
        "summary": {
            "splits": list(splits),
            "catalog_dir": str(catalog_dir),
            "fixture_files": [str(path) for path in files],
            "runnable_rows": len(runnable),
            "fixture_rows": len(fixtures),
            "out_of_scope_fixture_rows": out_of_scope,
            "unique_fixture_tasks": len(fixture_task_ids),
            "explicit_coverage_rows": sum(1 for row in fixtures if row["explicit_coverage"]),
            "fixture_problem_rows": len(problems),
            "duplicate_fixture_tasks": len(_duplicates(fixtures)),
            "eval_setup_combos_available": len(available_eval_setup),
            "eval_setup_combos_covered": len(set(covered_eval_setup) & set(available_eval_setup)),
            "eval_setup_combos_missing": len(set(available_eval_setup) - set(covered_eval_setup)),
            "full_setup_eval_postconfig_combos_available": len(available_full),
            "full_setup_eval_postconfig_combos_covered": len(
                set(covered_full) & set(available_full)
            ),
            "full_setup_eval_postconfig_combos_missing": len(
                set(available_full) - set(covered_full)
            ),
            "oracle_action_kinds": dict(sorted(action_kinds.items())),
        },
        "domains": _domain_summary(runnable, fixtures, DOMAIN_BUDGETS["recommended"]),
        "top_missing_eval_setup_combos": _top_missing(
            available_eval_setup,
            covered_eval_setup,
            include_postconfig=False,
            limit=top,
        ),
        "top_missing_full_setup_eval_postconfig_combos": _top_missing(
            available_full,
            covered_full,
            include_postconfig=True,
            limit=top,
        ),
        "top_rl_oracle_candidates": _candidate_priority(
            runnable,
            fixtures,
            candidate_split=candidate_split,
            limit=candidate_limit,
        ),
        "fixture_problems": problems,
        "duplicate_fixture_tasks": _duplicates(fixtures),
    }
    return report


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    split_label = ", ".join(summary["splits"])
    lines = [
        "# lite.scalecua Oracle Coverage Inventory",
        "",
        "This report maps committed oracle fixtures back to the current runnable",
        f"`{split_label}` catalog set. Missing `coverage` fields in fixture rows are",
        "filled from the catalog during this audit; committed fixtures should still",
        "add explicit coverage when they are promoted.",
        "",
        "## Summary",
        "",
        f"- Splits: {split_label}",
        f"- Runnable rows: {summary['runnable_rows']}",
        f"- Fixture rows matched to runnable tasks: {summary['fixture_rows']}",
        f"- Out-of-scope fixture rows skipped: {summary['out_of_scope_fixture_rows']}",
        f"- Unique fixture tasks: {summary['unique_fixture_tasks']}",
        f"- Fixture rows with explicit coverage: {summary['explicit_coverage_rows']}",
        f"- Fixture problem rows: {summary['fixture_problem_rows']}",
        f"- Duplicate fixture task ids: {summary['duplicate_fixture_tasks']}",
        "- Eval x setup combos: "
        f"{summary['eval_setup_combos_covered']} / "
        f"{summary['eval_setup_combos_available']} covered",
        "- Eval x setup x postconfig combos: "
        f"{summary['full_setup_eval_postconfig_combos_covered']} / "
        f"{summary['full_setup_eval_postconfig_combos_available']} covered",
        f"- Oracle action kinds: {json.dumps(summary['oracle_action_kinds'], sort_keys=True)}",
        "",
        "## Domain Matrix",
        "",
        "| Domain | Fixtures | Recommended add | Eval x setup | Full combos | "
        "Missing setup | Missing postconfig |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for domain, item in report["domains"].items():
        eval_setup = item["eval_setup_combos"]
        full = item["full_setup_eval_postconfig_combos"]
        missing_setup = ", ".join(item["setup_types"]["missing"]) or "-"
        missing_post = ", ".join(item["postconfig_types"]["missing"]) or "-"
        lines.append(
            (
                "| {domain} | {fixtures} | {add} | {eval_cov}/{eval_total} | "
                "{full_cov}/{full_total} | {setup} | {post} |"
            ).format(
                domain=domain,
                fixtures=item["unique_fixture_tasks"],
                add=item["recommended_add_needed_by_count"],
                eval_cov=eval_setup["covered"],
                eval_total=eval_setup["available"],
                full_cov=full["covered"],
                full_total=full["available"],
                setup=missing_setup,
                post=missing_post,
            )
        )

    lines.extend(
        [
            "",
            "## Top RL Oracle Candidates",
            "",
            "| Rank | Domain | Task id | New full combos | New eval x setup combos |",
            "| ---: | --- | --- | ---: | ---: |",
        ]
    )
    for item in report["top_rl_oracle_candidates"]:
        lines.append(
            "| {rank} | {domain} | `{task_id}` | {full} | {eval_setup} |".format(
                rank=item["rank"],
                domain=item["domain"],
                task_id=item["task_id"],
                full=item["new_full_combo_count"],
                eval_setup=item["new_eval_setup_combo_count"],
            )
        )

    lines.extend(
        [
            "",
            "## Top Missing Eval x Setup Combos",
            "",
            "| Count | Split | Domain | Setup | Eval combo |",
            "| ---: | --- | --- | --- | --- |",
        ]
    )
    for item in report["top_missing_eval_setup_combos"]:
        lines.append(
            "| {count} | {split} | {domain} | {setup} | {combo} |".format(
                count=item["count"],
                split=item["split"],
                domain=item["domain"],
                setup=", ".join(item["setup"]) or "none",
                combo=" / ".join(item["eval_combo"]),
            )
        )

    lines.extend(
        [
            "",
            "## Top Missing Eval x Setup x Postconfig Combos",
            "",
            "| Count | Split | Domain | Setup | Eval combo | Postconfig |",
            "| ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for item in report["top_missing_full_setup_eval_postconfig_combos"]:
        lines.append(
            "| {count} | {split} | {domain} | {setup} | {combo} | {post} |".format(
                count=item["count"],
                split=item["split"],
                domain=item["domain"],
                setup=", ".join(item["setup"]) or "none",
                combo=" / ".join(item["eval_combo"]),
                post=", ".join(item.get("postconfig", [])) or "none",
            )
        )

    if report["fixture_problems"]:
        lines.extend(["", "## Fixture Problems", ""])
        for item in report["fixture_problems"]:
            lines.append(f"- {json.dumps(item, sort_keys=True)}")

    if report["duplicate_fixture_tasks"]:
        lines.extend(["", "## Duplicate Fixture Tasks", ""])
        for item in report["duplicate_fixture_tasks"]:
            lines.append(f"- {json.dumps(item, sort_keys=True)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixtures",
        type=Path,
        nargs="+",
        default=[Path("lite/gym/envs/lite/scalecua/data/oracle")],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=40,
        help="Number of prioritized RL oracle candidate task ids to include.",
    )
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=DEFAULT_CATALOG_DIR,
        help="Directory containing rl.jsonl/train.jsonl task catalogs.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "rl"),
        default=["train", "rl"],
        help="Runtime splits to include in the coverage universe.",
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help=(
            "Exit non-zero if any fixture points outside the runnable catalog "
            "or duplicates a task."
        ),
    )
    args = parser.parse_args()

    report = build_report(
        args.fixtures,
        args.top,
        tuple(args.splits),
        catalog_dir=args.catalog_dir,
        candidate_limit=args.candidate_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.markdown:
        write_markdown(report, args.markdown)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    if args.require_clean:
        summary = report["summary"]
        if summary["fixture_problem_rows"] or summary["duplicate_fixture_tasks"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
