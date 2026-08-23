"""Summarize in-progress lite.scalecua batch rollouts for visual audit.

``metadata.others.domain`` is the canonical domain field, matching
``lite.osworld``. Some ScaleCUA generated tasks also carry a different
``metadata.scalecua.related_apps`` value; this script reports that only as an
audit hint and never as the sampling/filtering domain.

Run:
    uv run python devs/envs/lite.scalecua/validate/rollout/audit_queue.py \
        --log-root .exps/validate/lite.scalecua/batch/<run-id>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from lite.infer.debug.log_layout import (
    RESPONSE_TXT,
    turn_dirs,
    turn_images,
    turn_results_path,
)

DEFAULT_CATALOG_ROOT = Path("lite/gym/envs/lite/scalecua/data")


def _load_catalog(catalog_root: Path) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for split in ("train", "rl"):
        path = catalog_root / f"{split}.jsonl"
        if not path.exists():
            continue
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                metadata = row.get("metadata") or {}
                others = metadata.get("others") or {}
                scalecua = metadata.get("scalecua") or {}
                by_id[row["task_id"]] = {
                    "split": split,
                    "instruction": row.get("instruction", ""),
                    "domain": str(others.get("domain") or "unknown"),
                    "related_app_domain": _related_app_domain(scalecua, others),
                    "exclude_reason": others.get("exclude_reason"),
                }
    return by_id


def _normalize_domain(value: Any) -> str:
    text = str(value or "").strip()
    aliases = {
        "base_setup": "os",
        "calc": "libreoffice_calc",
        "libreoffice calc": "libreoffice_calc",
        "multiapps": "multi_apps",
        "vscode": "vs_code",
    }
    return aliases.get(text, text or "unknown")


def _related_app_domain(scalecua: dict[str, Any], others: dict[str, Any]) -> str:
    related = scalecua.get("related_apps") or []
    if len(related) > 1:
        return "multi_apps"
    if len(related) == 1:
        return _normalize_domain(related[0])
    return _normalize_domain(scalecua.get("snapshot") or others.get("domain"))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"_read_error": str(exc)}


def _iter_trajectories(log_root: Path, catalog: dict[str, dict[str, Any]]):
    for split_dir in sorted(p for p in log_root.iterdir() if p.is_dir()):
        split = split_dir.name
        for task_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            sample_dir = task_dir / "sample_00"
            if not sample_dir.is_dir():
                continue
            result_paths = [
                result_path
                for path in turn_dirs(sample_dir)
                if (result_path := turn_results_path(path)) is not None
            ]
            if not result_paths:
                continue
            latest_result_path = result_paths[-1]
            result = _read_json(latest_result_path)
            latest_turn_dir = latest_result_path.parent
            images = turn_images(sample_dir)
            screenshot = images[-1][1] if images else None
            task_id = task_dir.name
            meta = catalog.get(task_id, {})
            yield {
                "split": meta.get("split", split),
                "task_id": task_id,
                "domain": meta.get("domain", "unknown"),
                "related_app_domain": meta.get("related_app_domain", "unknown"),
                "instruction": meta.get("instruction", ""),
                "turns": len(result_paths),
                "latest_turn": latest_turn_dir.name,
                "reward": result.get("reward"),
                "terminated": result.get("terminated"),
                "truncated": result.get("truncated"),
                "log_dir": str(sample_dir),
                "screenshot": str(screenshot) if screenshot is not None else "",
                "response": str(latest_turn_dir / RESPONSE_TXT),
            }


def _sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["split"]), str(row["domain"]), str(row["task_id"]))


def _print_counter(title: str, counter: Counter[tuple[str, str]]) -> None:
    print(title)
    for (split, domain), count in sorted(counter.items()):
        print(f"  {split}/{domain}: {count}")


def _reward_value(row: dict[str, Any]) -> float:
    value = float(row["reward"])
    if value >= 1.0 - 1e-9:
        return 1.0
    if value <= 1e-9:
        return 0.0
    return min(1.0, max(0.0, value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--catalog-root", type=Path, default=DEFAULT_CATALOG_ROOT)
    parser.add_argument("--queue-size", type=int, default=12)
    parser.add_argument("--jsonl", type=Path, help="Optional path to write all scanned rows.")
    args = parser.parse_args()

    catalog = _load_catalog(args.catalog_root)
    rows = sorted(_iter_trajectories(args.log_root, catalog), key=_sort_key)
    completed = [row for row in rows if row["reward"] is not None]
    in_progress = [row for row in rows if row["reward"] is None]

    print(f"scanned={len(rows)} completed={len(completed)} in_progress={len(in_progress)}")
    print(f"reward=1 {sum(1 for row in completed if _reward_value(row) == 1.0)}")
    print(f"reward=0 {sum(1 for row in completed if _reward_value(row) == 0.0)}")
    print(
        "reward_partial "
        f"{sum(1 for row in completed if 0.0 < _reward_value(row) < 1.0)}"
    )
    _print_counter(
        "completed by metadata.others.domain",
        Counter((row["split"], row["domain"]) for row in completed),
    )
    _print_counter(
        "completed by related_app_domain",
        Counter((row["split"], row["related_app_domain"]) for row in completed),
    )

    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"wrote {args.jsonl}")

    queues = (
        ("reward=1", lambda row: _reward_value(row) == 1.0),
        ("reward=0", lambda row: _reward_value(row) == 0.0),
        ("reward_partial", lambda row: 0.0 < _reward_value(row) < 1.0),
    )
    for label, predicate in queues:
        queue = [row for row in completed if predicate(row)]
        queue = sorted(
            queue,
            key=lambda r: (r["split"], r["domain"], r["related_app_domain"], r["task_id"]),
        )
        print(f"\n{label} audit queue")
        for row in queue[: args.queue_size]:
            instruction = row["instruction"].replace("\n", " ")[:120]
            print(
                f"- {row['split']} domain={row['domain']} "
                f"related_app_domain={row['related_app_domain']} "
                f"turns={row['turns']} task={row['task_id']}\n"
                f"  screenshot={row['screenshot']}\n"
                f"  instruction={instruction}"
            )


if __name__ == "__main__":
    main()
