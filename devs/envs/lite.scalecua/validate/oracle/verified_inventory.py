"""Audit verified lite.scalecua oracle replay evidence.

`coverage_inventory.py` answers "which current catalog rows have fixture rows?".
This script answers the stricter release question: "which current fixture rows
have no-op-negative plus oracle-positive replay evidence from validate.py?".

Run:
    uv run python devs/envs/lite.scalecua/validate/oracle/verified_inventory.py \
        --fixtures lite/gym/envs/lite/scalecua/data/oracle \
        --reports .exps/validate/lite.scalecua/oracle \
        --splits rl --require-artifacts \
        --output .exps/validate/lite.scalecua/oracle/coverage/rl.verified.json
"""

from __future__ import annotations

import argparse
import json
import math
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
DEFAULT_CATALOG_DIR = REPO_ROOT / "lite" / "gym" / "envs" / "lite" / "scalecua" / "data"
DEFAULT_FIXTURE_DIR = REPO_ROOT / "lite" / "gym" / "envs" / "lite" / "scalecua" / "data" / "oracle"
DEFAULT_REPORT_ROOT = REPO_ROOT / ".exps" / "validate" / "lite.scalecua" / "oracle"
REQUIRED_ARTIFACT_FILES = (
    "result.json",
    "00_noop_reset.png",
    "01_noop_final.png",
    "10_oracle_reset.png",
    "11_oracle_after_actions.png",
)


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            rows.append((line_no, json.loads(line)))
    return rows


def _files(paths: list[Path], pattern: str, *, recursive: bool = False) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob(pattern) if recursive else path.glob(pattern)))
        elif path.exists():
            files.append(path)
    return sorted(dict.fromkeys(files))


def _report_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.report.jsonl")))
            files.extend(sorted(path.rglob("report.jsonl")))
        elif path.exists():
            files.append(path)
    return sorted(dict.fromkeys(files))


def _load_catalog(
    splits: tuple[str, ...],
    catalog_dir: Path,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], str]]:
    runnable: dict[tuple[str, str], dict[str, Any]] = {}
    excluded: dict[tuple[str, str], str] = {}
    for split in splits:
        path = catalog_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"catalog not found for split {split!r}: {path}")
        for _, row in _read_jsonl(path):
            metadata = row.get("metadata", {})
            others = metadata.get("others", {})
            key = (split, row["task_id"])
            reason = others.get("exclude_reason")
            if reason:
                excluded[key] = str(reason)
                continue
            runnable[key] = {
                "split": split,
                "task_id": row["task_id"],
                "domain": others.get("domain", "unknown"),
                "instruction": row.get("instruction", ""),
            }
    return runnable, excluded


def _load_fixtures(
    fixture_files: list[Path],
    *,
    splits: tuple[str, ...],
    catalog: dict[tuple[str, str], dict[str, Any]],
    excluded: dict[tuple[str, str], str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], int]:
    fixtures: dict[str, dict[str, Any]] = {}
    problems: list[dict[str, Any]] = []
    out_of_scope = 0
    split_set = set(splits)
    seen_ids: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_tasks: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for file_path in fixture_files:
        for line_no, fixture in _read_jsonl(file_path):
            split = fixture.get("split")
            task_id = fixture.get("task_id")
            fixture_id = fixture.get("fixture_id")
            location = {
                "file": str(file_path),
                "line": line_no,
                "fixture_id": fixture_id,
                "split": split,
                "task_id": task_id,
            }
            if split not in split_set:
                out_of_scope += 1
                continue
            if not fixture_id or not task_id:
                problems.append({**location, "problem": "fixture_missing_id_or_task"})
                continue

            key = (str(split), str(task_id))
            if key in excluded:
                problems.append(
                    {
                        **location,
                        "problem": "fixture_points_to_excluded_task",
                        "exclude_reason": excluded[key],
                    }
                )
                continue
            if key not in catalog:
                problems.append({**location, "problem": "fixture_task_not_in_catalog"})
                continue

            record = {
                **location,
                "fixture": fixture,
                "catalog_domain": catalog[key]["domain"],
                "declared_domain": fixture.get("domain"),
                "expected_pre_reward": fixture.get("expected_pre_reward", 0.0),
                "expected_reward": fixture.get("expected_reward", 1.0),
            }
            fixtures[str(fixture_id)] = record
            seen_ids[str(fixture_id)].append(record)
            seen_tasks[key].append(record)

    for fixture_id, items in sorted(seen_ids.items()):
        if len(items) > 1:
            problems.append(
                {
                    "problem": "duplicate_fixture_id",
                    "fixture_id": fixture_id,
                    "locations": [
                        {"file": item["file"], "line": item["line"], "task_id": item["task_id"]}
                        for item in items
                    ],
                }
            )
    for (split, task_id), items in sorted(seen_tasks.items()):
        if len(items) > 1:
            problems.append(
                {
                    "problem": "duplicate_fixture_task",
                    "split": split,
                    "task_id": task_id,
                    "fixture_ids": [item["fixture_id"] for item in items],
                    "files": sorted({item["file"] for item in items}),
                }
            )

    duplicate_ids = {fixture_id for fixture_id, items in seen_ids.items() if len(items) > 1}
    if duplicate_ids:
        fixtures = {
            fixture_id: record
            for fixture_id, record in fixtures.items()
            if fixture_id not in duplicate_ids
        }
    return fixtures, problems, out_of_scope


def _num_equal(actual: Any, expected: Any) -> bool:
    try:
        return math.isclose(float(actual), float(expected), abs_tol=1e-9)
    except (TypeError, ValueError):
        return actual == expected


def _report_base_name(path: Path) -> str:
    suffix = ".report.jsonl"
    if path.name.endswith(suffix):
        return path.name[: -len(suffix)]
    return path.stem


def _artifact_candidates(report_path: Path, fixture_id: str) -> list[Path]:
    base = _report_base_name(report_path)
    names = [base, base.replace(".", "-"), base.replace("_", "-")]
    candidates = [report_path.parent / name / fixture_id for name in dict.fromkeys(names)]
    candidates.append(report_path.parent / fixture_id)
    candidates.extend(sorted(report_path.parent.glob(f"*/{fixture_id}")))
    return candidates


def _artifact_result_matches(
    candidate: Path,
    fixture_id: str,
    report: dict[str, Any],
) -> bool:
    result_path = candidate / "result.json"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if result.get("fixture_id") != fixture_id:
        return False
    if result.get("task_id") != report.get("task_id"):
        return False
    return result.get("passed") is True


def _artifact_status(
    report_path: Path,
    fixture_id: str,
    report: dict[str, Any],
) -> tuple[bool, str | None, list[str]]:
    for candidate in _artifact_candidates(report_path, fixture_id):
        missing = [name for name in REQUIRED_ARTIFACT_FILES if not (candidate / name).is_file()]
        if not missing and _artifact_result_matches(candidate, fixture_id, report):
            return True, str(candidate), []
    first = _artifact_candidates(report_path, fixture_id)[0]
    missing = [name for name in REQUIRED_ARTIFACT_FILES if not (first / name).is_file()]
    return False, None, missing


def _failure_reasons(
    report: dict[str, Any],
    fixture_record: dict[str, Any] | None,
) -> list[str]:
    reasons: list[str] = []
    if fixture_record is None:
        return ["report_not_current_fixture"]

    fixture = fixture_record["fixture"]
    expected_pre = fixture.get("expected_pre_reward", 0.0)
    expected = fixture.get("expected_reward", 1.0)
    split = fixture.get("split")
    task_id = fixture.get("task_id")

    if report.get("passed") is not True:
        reasons.append("top_level_not_passed")
    if report.get("task_id") != task_id:
        reasons.append("top_level_task_mismatch")

    precheck = report.get("precheck")
    if not isinstance(precheck, dict):
        reasons.append("missing_precheck")
    else:
        if precheck.get("passed") is not True:
            reasons.append("precheck_not_passed")
        if not _num_equal(precheck.get("reward"), expected_pre):
            reasons.append("precheck_reward_mismatch")
        if precheck.get("task_id") != task_id:
            reasons.append("precheck_task_mismatch")
        if precheck.get("split") != split:
            reasons.append("precheck_split_mismatch")

    replay = report.get("replay")
    if not isinstance(replay, dict):
        reasons.append("missing_replay")
    else:
        if replay.get("passed") is not True:
            reasons.append("replay_not_passed")
        if not _num_equal(replay.get("expected_reward"), expected):
            reasons.append("replay_expected_reward_mismatch")
        if not _num_equal(replay.get("reward"), expected):
            reasons.append("replay_reward_mismatch")
        if replay.get("task_id") != task_id:
            reasons.append("replay_task_mismatch")
        if replay.get("split") != split:
            reasons.append("replay_split_mismatch")
        if not replay.get("trace"):
            reasons.append("missing_replay_trace")
        if "eval_debug" not in replay:
            reasons.append("missing_eval_debug")

    return reasons


def build_report(
    fixture_paths: list[Path],
    report_paths: list[Path],
    *,
    splits: tuple[str, ...],
    catalog_dir: Path,
    require_artifacts: bool,
    top: int,
) -> dict[str, Any]:
    catalog, excluded = _load_catalog(splits, catalog_dir)
    fixture_files = _files(fixture_paths, "*.jsonl")
    report_files = _report_files(report_paths)
    fixtures, fixture_problems, out_of_scope = _load_fixtures(
        fixture_files,
        splits=splits,
        catalog=catalog,
        excluded=excluded,
    )

    valid_report_rows: list[dict[str, Any]] = []
    verified_by_fixture: dict[str, dict[str, Any]] = {}
    report_problem_counter: Counter[str] = Counter()
    stale_rows: list[dict[str, Any]] = []
    artifact_missing_rows: list[dict[str, Any]] = []

    for report_file in report_files:
        for line_no, row in _read_jsonl(report_file):
            fixture_id = row.get("fixture_id")
            fixture_record = fixtures.get(str(fixture_id)) if fixture_id is not None else None
            reasons = _failure_reasons(row, fixture_record)
            if reasons == ["report_not_current_fixture"]:
                stale_rows.append(
                    {
                        "report_file": str(report_file),
                        "line": line_no,
                        "fixture_id": fixture_id,
                        "task_id": row.get("task_id"),
                        "passed": row.get("passed"),
                    }
                )
                report_problem_counter.update(reasons)
                continue
            if reasons:
                report_problem_counter.update(reasons)
                continue

            artifact_ok, artifact_dir, missing_artifacts = _artifact_status(
                report_file,
                str(fixture_id),
                row,
            )
            evidence = {
                "fixture_id": str(fixture_id),
                "task_id": row.get("task_id"),
                "split": fixture_record["fixture"].get("split") if fixture_record else None,
                "domain": fixture_record["catalog_domain"] if fixture_record else None,
                "report_file": str(report_file),
                "report_line": line_no,
                "artifact_dir": artifact_dir,
                "artifact_ok": artifact_ok,
            }
            valid_report_rows.append(evidence)
            if not artifact_ok:
                artifact_missing_rows.append(
                    {**evidence, "missing_artifacts": missing_artifacts}
                )
                if require_artifacts:
                    continue
            if str(fixture_id) not in verified_by_fixture:
                verified_by_fixture[str(fixture_id)] = evidence

    valid_by_fixture: dict[str, dict[str, Any]] = {}
    for row in valid_report_rows:
        valid_by_fixture.setdefault(row["fixture_id"], row)
    valid_fixture_ids = set(valid_by_fixture)
    verified_fixture_ids = set(verified_by_fixture)
    fixtures_by_split = Counter(record["fixture"]["split"] for record in fixtures.values())
    verified_by_split = Counter(
        evidence["split"] for evidence in verified_by_fixture.values()
    )
    valid_by_split = Counter(
        fixtures[fixture_id]["fixture"]["split"]
        for fixture_id in valid_by_fixture
        if fixture_id in fixtures
    )
    fixtures_by_domain = Counter(record["catalog_domain"] for record in fixtures.values())
    verified_by_domain = Counter(
        evidence["domain"] for evidence in verified_by_fixture.values()
    )
    verified_catalog_keys = {
        (
            str(fixtures[fixture_id]["fixture"].get("split")),
            str(fixtures[fixture_id]["fixture"].get("task_id")),
        )
        for fixture_id in verified_fixture_ids
        if fixture_id in fixtures
    }
    unverified_catalog = [
        {
            "split": split,
            "task_id": task_id,
            "domain": row["domain"],
            "instruction": row["instruction"],
        }
        for (split, task_id), row in sorted(catalog.items())
        if (split, task_id) not in verified_catalog_keys
    ]
    duplicate_valid_reports = len(valid_report_rows) - len(valid_fixture_ids)

    unverified = [
        {
            "fixture_id": record["fixture_id"],
            "split": record["fixture"].get("split"),
            "task_id": record["fixture"].get("task_id"),
            "domain": record["catalog_domain"],
            "file": record["file"],
            "line": record["line"],
        }
        for fixture_id, record in sorted(fixtures.items())
        if fixture_id not in verified_fixture_ids
    ]

    summary = {
        "splits": list(splits),
        "catalog_dir": str(catalog_dir),
        "fixture_files": [str(path) for path in fixture_files],
        "report_files": [str(path) for path in report_files],
        "require_artifacts": require_artifacts,
        "current_fixture_rows": len(fixtures),
        "current_fixture_rows_by_split": dict(sorted(fixtures_by_split.items())),
        "out_of_scope_fixture_rows": out_of_scope,
        "fixture_problem_rows": len(fixture_problems),
        "strict_valid_report_rows": len(valid_report_rows),
        "strict_valid_unique_fixtures_ignoring_artifacts": len(valid_fixture_ids),
        "verified_fixture_rows": len(verified_fixture_ids),
        "verified_fixture_rows_by_split": dict(sorted(verified_by_split.items())),
        "catalog_runnable_rows": len(catalog),
        "verified_catalog_rows": len(verified_catalog_keys),
        "unverified_catalog_rows": len(unverified_catalog),
        "strict_valid_unique_fixtures_by_split_ignoring_artifacts": dict(
            sorted(valid_by_split.items())
        ),
        "unverified_fixture_rows": len(unverified),
        "duplicate_strict_valid_report_rows": duplicate_valid_reports,
        "stale_report_rows": len(stale_rows),
        "artifact_missing_strict_valid_report_rows": len(artifact_missing_rows),
        "report_problem_counts": dict(sorted(report_problem_counter.items())),
    }

    return {
        "summary": summary,
        "domains": {
            domain: {
                "fixture_rows": fixtures_by_domain[domain],
                "verified_fixture_rows": verified_by_domain[domain],
                "unverified_fixture_rows": fixtures_by_domain[domain]
                - verified_by_domain[domain],
            }
            for domain in sorted(fixtures_by_domain)
        },
        "verified_fixtures": sorted(
            verified_by_fixture.values(),
            key=lambda item: (str(item["split"]), str(item["domain"]), str(item["fixture_id"])),
        ),
        "unverified_fixtures_sample": unverified[:top],
        "unverified_catalog_rows_sample": unverified_catalog[:top],
        "fixture_problems": fixture_problems[:top],
        "stale_report_rows_sample": stale_rows[:top],
        "artifact_missing_rows_sample": artifact_missing_rows[:top],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    split_label = ", ".join(summary["splits"])
    lines = [
        "# lite.scalecua Verified Oracle Inventory",
        "",
        "This report maps current oracle fixture rows to full no-op-negative plus",
        "oracle-positive replay reports from `validate.py`. It is stricter than",
        "`coverage_inventory.py`: fixture coverage alone is not release evidence.",
        "",
        "## Summary",
        "",
        f"- Splits: {split_label}",
        f"- Require artifact files: {summary['require_artifacts']}",
        f"- Current fixture rows: {summary['current_fixture_rows']}",
        f"- Fixture problem rows: {summary['fixture_problem_rows']}",
        f"- Strict valid report rows: {summary['strict_valid_report_rows']}",
        "- Strict valid unique fixtures ignoring artifact files: "
        f"{summary['strict_valid_unique_fixtures_ignoring_artifacts']}",
        f"- Verified fixture rows: {summary['verified_fixture_rows']}",
        f"- Catalog runnable rows: {summary['catalog_runnable_rows']}",
        f"- Verified catalog rows: {summary['verified_catalog_rows']}",
        f"- Unverified catalog rows: {summary['unverified_catalog_rows']}",
        f"- Unverified fixture rows: {summary['unverified_fixture_rows']}",
        f"- Duplicate strict valid report rows: {summary['duplicate_strict_valid_report_rows']}",
        f"- Stale report rows: {summary['stale_report_rows']}",
        "- Artifact-missing strict valid report rows: "
        f"{summary['artifact_missing_strict_valid_report_rows']}",
        "- Verified by split: "
        f"{json.dumps(summary['verified_fixture_rows_by_split'], sort_keys=True)}",
        "",
        "## Domain Matrix",
        "",
        "| Domain | Fixtures | Verified | Unverified |",
        "| --- | ---: | ---: | ---: |",
    ]
    for domain, item in report["domains"].items():
        lines.append(
            f"| {domain} | {item['fixture_rows']} | "
            f"{item['verified_fixture_rows']} | {item['unverified_fixture_rows']} |"
        )

    lines.extend(
        [
            "",
            "## Report Problem Counts",
            "",
            "```json",
            json.dumps(summary["report_problem_counts"], indent=2, sort_keys=True),
            "```",
        ]
    )

    if report["unverified_fixtures_sample"]:
        lines.extend(["", "## Unverified Fixture Sample", ""])
        for item in report["unverified_fixtures_sample"]:
            lines.append(f"- {json.dumps(item, sort_keys=True)}")

    if report["unverified_catalog_rows_sample"]:
        lines.extend(["", "## Unverified Catalog Row Sample", ""])
        for item in report["unverified_catalog_rows_sample"]:
            lines.append(f"- {json.dumps(item, sort_keys=True)}")

    if report["fixture_problems"]:
        lines.extend(["", "## Fixture Problem Sample", ""])
        for item in report["fixture_problems"]:
            lines.append(f"- {json.dumps(item, sort_keys=True)}")

    if report["stale_report_rows_sample"]:
        lines.extend(["", "## Stale Report Row Sample", ""])
        for item in report["stale_report_rows_sample"]:
            lines.append(f"- {json.dumps(item, sort_keys=True)}")

    if report["artifact_missing_rows_sample"]:
        lines.extend(["", "## Artifact-Missing Strict Pass Sample", ""])
        for item in report["artifact_missing_rows_sample"]:
            lines.append(f"- {json.dumps(item, sort_keys=True)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixtures",
        type=Path,
        nargs="+",
        default=[DEFAULT_FIXTURE_DIR],
        help="Fixture JSONL files or directories.",
    )
    parser.add_argument(
        "--reports",
        type=Path,
        nargs="+",
        default=[DEFAULT_REPORT_ROOT],
        help="Report JSONL files or directories to scan recursively.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=DEFAULT_CATALOG_DIR,
        help="Directory containing train.jsonl and rl.jsonl catalogs.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "rl"),
        default=["train", "rl"],
        help="Runtime splits to include.",
    )
    parser.add_argument(
        "--require-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require result.json and key screenshots next to each passing report row.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit non-zero unless every selected current fixture row is verified.",
    )
    parser.add_argument(
        "--require-catalog-complete",
        action="store_true",
        help="Exit non-zero unless every selected non-excluded catalog row has verified evidence.",
    )
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    report = build_report(
        args.fixtures,
        args.reports,
        splits=tuple(args.splits),
        catalog_dir=args.catalog_dir,
        require_artifacts=args.require_artifacts,
        top=args.top,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.markdown:
        write_markdown(report, args.markdown)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    if args.require_complete:
        summary = report["summary"]
        if summary["fixture_problem_rows"] or summary["unverified_fixture_rows"]:
            return 1
    if args.require_catalog_complete:
        summary = report["summary"]
        if summary["fixture_problem_rows"] or summary["unverified_catalog_rows"]:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
