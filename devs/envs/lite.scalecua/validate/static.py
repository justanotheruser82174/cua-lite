#!/usr/bin/env python3
"""Static validation for lite.scalecua catalogs.

Run:
    uv run python devs/envs/lite.scalecua/validate/static.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lite.gym.envs.lite.scalecua.src.utils import assets, dataset  # noqa: E402


def _load_json(path: Path):
    return json.loads(path.read_text())


def _rows(path: Path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _failures(args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    cache = Path(args.cache)
    catalog = Path(args.catalog)

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    try:
        dataset.validate_catalog_lock()
        dataset.validate_runtime_cache()
        dataset.validate_all(catalog, strict_counts=True)
    except Exception as exc:
        failures.append(f"catalog strict validation failed: {exc}")
        return failures

    check(
        sorted(p.name for p in catalog.glob("*.jsonl"))
        == ["rl.jsonl", "train.jsonl"],
        "catalog filenames must be exactly train/rl.jsonl",
    )
    check(not list(catalog.glob("osworld_*.jsonl")), "legacy osworld_* catalog files present")
    check(not list(cache.glob("*.jsonl")), "legacy cache-root catalog files present")

    all_ids: set[str] = set()
    for split in dataset.RUNTIME_SPLITS:
        for row in _rows(catalog / f"{split}.jsonl"):
            task_id = row["task_id"]
            check(task_id not in all_ids, f"duplicate task_id across splits: {task_id}")
            all_ids.add(task_id)
            others = row["metadata"]["others"]
            reason = others.get("exclude_reason")
            if reason is None:
                check(
                    "resolve/main" not in json.dumps(row, ensure_ascii=False),
                    f"{task_id}: runnable row contains resolve/main URL",
                )
            else:
                check(isinstance(reason, str) and bool(reason), f"{task_id}: bad exclude_reason")

    report_path = cache / "import_report.json"
    check(report_path.is_file(), "missing import_report.json")
    if report_path.is_file():
        report = _load_json(report_path)
        for key in (
            "asset_snapshot",
            "asset_identity",
            "splits",
            "action_type_counts_before",
            "action_type_counts_after",
            "excluded_count_by_reason",
            "url_rewrite_count",
        ):
            check(key in report, f"import_report missing {key}")
        check(
            sorted(report.get("splits", {})) == ["rl", "train"],
            "import_report splits must be exactly train/rl",
        )

    for split in ("train", "rl"):
        root = cache / "judge_functions" / split
        check((root / "getters.py").is_file(), f"{split}: missing getters.py overlay")
        check((root / "metrics.py").is_file(), f"{split}: missing metrics.py overlay")
        check((root / "verigen_getters").is_dir(), f"{split}: missing verigen_getters")
        check((root / "verigen_metrics").is_dir(), f"{split}: missing verigen_metrics")

    stamp = cache / ".asset_identity"
    check(stamp.is_file(), "missing .asset_identity")
    check((cache / ".complete").is_file(), "missing .complete")
    if stamp.is_file():
        check(stamp.read_text().strip() == assets.asset_identity(), ".asset_identity mismatch")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(assets.CACHE_DIR))
    parser.add_argument("--catalog", default=str(assets.CATALOG_DIR))
    args = parser.parse_args()
    failures = _failures(args)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("lite.scalecua static validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
