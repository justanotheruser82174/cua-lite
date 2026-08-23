"""Rebuild rollout root ``summary.json`` from per-task summaries.

Some evals are run in multiple passes against the same slug directory, for
example WebHarbor's non-mutating pass followed by mutating tasks. The second
pass overwrites ``<slug>/summary.json`` with only that pass, while the per-task
summaries from both passes remain under ``<slug>/eval/<task>/``.

This script reconstructs the slug-level summary from:

* ``eval/<task>/sample_*/summary.json`` when sample summaries exist.
* ``eval/<task>/summary.json`` as a fallback for older or manually restored
  logs.

Usage:

    # Rebuild every slug under one run dir.
    uv run python devs/exps/eval/utils/update_summery.py \\
        .exps/eval/webharbor.webvoyager/2026-07-08T01-38_c1001a613c/run_0

    # Rebuild one slug only.
    uv run python devs/exps/eval/utils/update_summery.py \\
        .exps/eval/webharbor.webvoyager/2026-07-08T01-38_c1001a613c/run_0/Qwen_Qwen3.5-4B__som

    # Inspect without writing.
    uv run python devs/exps/eval/utils/update_summery.py --dry-run <path>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]


class _CompactListEncoder:
    """Match rollout.py's compact summary style closely enough for readability."""

    def encode(self, obj: Any) -> str:
        return self._encode(obj, indent_level=0)

    def _encode(self, obj: Any, indent_level: int) -> str:
        if isinstance(obj, dict):
            if not obj:
                return "{}"
            indent = "  " * (indent_level + 1)
            closing = "  " * indent_level
            items = [
                f"{indent}{json.dumps(k, ensure_ascii=False)}: "
                f"{self._encode(v, indent_level + 1)}"
                for k, v in obj.items()
            ]
            return "{\n" + ",\n".join(items) + f"\n{closing}}}"

        if isinstance(obj, list):
            if all(not isinstance(x, (dict, list)) for x in obj):
                return "[" + ", ".join(
                    json.dumps(x, ensure_ascii=False, default=str) for x in obj
                ) + "]"
            indent = "  " * (indent_level + 1)
            closing = "  " * indent_level
            items = [f"{indent}{self._encode(x, indent_level + 1)}" for x in obj]
            return "[\n" + ",\n".join(items) + f"\n{closing}]"

        return json.dumps(obj, ensure_ascii=False, default=str)


def compact_json(obj: Any) -> str:
    return _CompactListEncoder().encode(obj) + "\n"


def relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def resolve_path(path: Path) -> Path:
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def discover_slug_dirs(path: Path) -> list[Path]:
    """Return slug dirs to update.

    ``path`` may be a slug dir, a run dir containing slug dirs, or an existing
    slug-level ``summary.json``.
    """
    if path.name == "summary.json":
        path = path.parent

    if (path / "eval").is_dir():
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"path not found: {path}")

    slugs = [child for child in sorted(path.iterdir()) if (child / "eval").is_dir()]
    if not slugs:
        raise FileNotFoundError(f"no slug dirs with eval/ found under {path}")
    return slugs


def sample_index(sample_dir: Path) -> int:
    try:
        return int(sample_dir.name.split("_", 1)[1])
    except Exception:
        return 0


def sample_result(sample_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    error = summary.get("error")
    has_valid_return = error is None and summary.get("episode_return") is not None
    return {
        "sample_idx": sample_index(sample_dir),
        "turns": int(summary.get("n_turns", 0) or 0),
        "episode_return": float(summary.get("episode_return", 0.0) or 0.0),
        "terminated": bool(summary.get("terminated", False)),
        "truncated": bool(summary.get("truncated", False)),
        "error": error,
        "valid": has_valid_return,
    }


def load_sample_results(task_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for sample_dir in sorted(task_dir.glob("sample_*"), key=sample_index):
        summary_path = sample_dir / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            results.append(sample_result(sample_dir, load_json(summary_path)))
        except Exception as exc:
            print(f"  WARN: skipping {relpath(summary_path)}: {exc}", file=sys.stderr)
    return results


def fallback_task_summary(task_dir: Path, env_id: str) -> dict[str, Any] | None:
    summary_path = task_dir / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        data = load_json(summary_path)
    except Exception as exc:
        print(f"  WARN: skipping {relpath(summary_path)}: {exc}", file=sys.stderr)
        return None

    values = data.get("episode_returns")
    if values is None and "episode_return" in data:
        values = [data.get("episode_return")]
    if values is None:
        return None

    returns = [float(v) for v in values if v is not None]
    num_samples = int(data.get("num_samples", len(values)) or 0)
    num_valid = int(data.get("num_valid", len(returns)) or 0)
    if num_valid <= 0:
        returns = []

    return {
        "task": str(data.get("task") or task_dir.name),
        "env_id": str(data.get("env_id") or env_id),
        "num_samples": num_samples,
        "num_valid": len(returns),
        "episode_returns": returns,
        "mean_episode_return": sum(returns) / len(returns) if returns else 0.0,
        "has_variance": len(set(returns)) > 1,
    }


def summarize_task(task_dir: Path, default_env_id: str) -> dict[str, Any] | None:
    sample_results = load_sample_results(task_dir)
    if not sample_results:
        return fallback_task_summary(task_dir, default_env_id)

    returns = [r["episode_return"] for r in sample_results if r["valid"]]
    return {
        "task": task_dir.name,
        "env_id": default_env_id,
        "num_samples": len(sample_results),
        "num_valid": len(returns),
        "episode_returns": returns,
        "mean_episode_return": sum(returns) / len(returns) if returns else 0.0,
        "has_variance": len(set(returns)) > 1,
    }


def infer_env_id(slug_dir: Path, existing_summary: dict[str, Any] | None) -> str:
    config = (existing_summary or {}).get("config") or {}
    if isinstance(config, dict) and config.get("env_id"):
        return str(config["env_id"])

    # Expected layout: .exps/eval/<env>/<commit>/<run>/<slug>
    parts = slug_dir.resolve().parts
    for i, part in enumerate(parts):
        if part == "eval" and i + 1 < len(parts):
            return parts[i + 1]
    return ""


def build_slug_summary(slug_dir: Path) -> dict[str, Any]:
    existing_path = slug_dir / "summary.json"
    existing = load_json(existing_path) if existing_path.is_file() else None
    env_id = infer_env_id(slug_dir, existing)

    task_summaries: list[dict[str, Any]] = []
    eval_root = slug_dir / "eval"
    for task_dir in sorted(eval_root.iterdir()):
        if not task_dir.is_dir():
            continue
        task_summary = summarize_task(task_dir, env_id)
        if task_summary is not None:
            task_summaries.append(task_summary)

    if not task_summaries:
        raise ValueError(f"no usable task summaries under {eval_root}")

    all_returns = [
        value
        for task in task_summaries
        for value in task.get("episode_returns", [])
        if value is not None
    ]
    num_samples = sum(int(task.get("num_samples", 0) or 0) for task in task_summaries)
    num_finished = len(task_summaries)
    stats = {
        "num_tasks": len(task_summaries),
        "num_samples": num_samples,
        "num_finished": num_finished,
        "num_valid": len(all_returns),
        "mean_episode_return": sum(all_returns) / len(all_returns) if all_returns else 0.0,
        "groups_with_variance": sum(1 for task in task_summaries if task.get("has_variance")),
    }

    config = {}
    if existing is not None and isinstance(existing.get("config"), dict):
        config = existing["config"]

    return {"config": config, "stats": stats, "tasks": task_summaries}


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(compact_json(payload), encoding="utf-8")
    tmp.replace(path)


def update_slug(slug_dir: Path, dry_run: bool = False) -> tuple[int, int, int, float]:
    payload = build_slug_summary(slug_dir)
    stats = payload["stats"]
    if not dry_run:
        write_json_atomic(slug_dir / "summary.json", payload)
        # Keep per-task aggregate summaries consistent with the rebuilt root.
        for task in payload["tasks"]:
            task_dir = slug_dir / "eval" / str(task["task"])
            write_json_atomic(task_dir / "summary.json", task)

    return (
        int(stats["num_finished"]),
        int(stats["num_valid"]),
        int(stats["num_tasks"]),
        float(stats["mean_episode_return"]),
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild slug-level summary.json from eval/<task>/ summaries."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Slug dir, run dir, or slug-level summary.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print rebuilt stats without writing summary.json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    slugs = discover_slug_dirs(resolve_path(args.path))

    failures = 0
    for slug_dir in slugs:
        try:
            num_finished, num_valid, num_tasks, mer = update_slug(slug_dir, dry_run=args.dry_run)
        except Exception as exc:
            failures += 1
            print(f"[update_summery.py] ERROR {relpath(slug_dir)}: {exc}", file=sys.stderr)
            continue

        verb = "would write" if args.dry_run else "wrote"
        print(
            f"[update_summery.py] {verb} {relpath(slug_dir / 'summary.json')} "
            f"(finished={num_finished}/{num_tasks}, valid={num_valid}/{num_tasks}, "
            f"mean_episode_return={mer:.4f})"
        )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
