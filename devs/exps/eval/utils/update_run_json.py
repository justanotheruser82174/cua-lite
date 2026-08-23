"""Aggregate one eval run's model summaries into a committed JSON snapshot.

The rollout writes raw metrics to:

    .exps/eval/<env>/<commit-dir>/<run_id>/<slug>/summary.json

This script writes the compact, committed snapshot used by eval README
renderers:

    devs/exps/eval/<env>/logs/<commit-dir>/<run_id>.json

Usage:

    uv run python devs/exps/eval/utils/update_run_json.py \
        .exps/eval/osworld_g/2026-07-06T20-34_0180ceefd8/run_0

    uv run python devs/exps/eval/utils/update_run_json.py \
        --env osworld_g \
        --commit-dir .exps/eval/osworld_g/2026-07-06T20-34_0180ceefd8 \
        --run-id run_0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
COMMIT_DIR_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}-[0-9]{2}_[0-9a-f]{7,}$")
RUN_ID_CONFIG_RE = re.compile(
    r"^run_[0-9]+(?:_(default|som|mixed|text_only|textonly|think_on|think_off))?$"
)


def slugify_model(model: str) -> str:
    return model.replace("/", "_")


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


def config_id_from_run_id(run_id: str) -> str | None:
    match = RUN_ID_CONFIG_RE.fullmatch(run_id)
    if not match:
        return None
    return match.group(1)


def resolve_run_dir(args: argparse.Namespace) -> tuple[Path, str, str, str]:
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
        run_dir = run_dir.resolve()
        run_id = args.run_id or run_dir.name
        commit_name = args.commit_dir or run_dir.parent.name
        env_name = args.env_name or run_dir.parent.parent.name
        return run_dir, env_name, Path(commit_name).name, run_id

    if not args.env_name or not args.commit_dir or not args.run_id:
        raise SystemExit("provide either RUN_DIR or all of --env, --commit-dir, and --run-id")

    commit_path = Path(args.commit_dir)
    if commit_path.is_absolute() or commit_path.parts[:2] == (".exps", "eval"):
        if not commit_path.is_absolute():
            commit_path = REPO_ROOT / commit_path
        commit_path = commit_path.resolve()
    else:
        commit_path = REPO_ROOT / ".exps/eval" / args.env_name / args.commit_dir

    env_name = args.env_name
    if commit_path.parent.name != env_name and commit_path.parent.parent.name == "eval":
        env_name = commit_path.parent.name

    run_dir = commit_path / args.run_id
    return run_dir.resolve(), env_name, commit_path.name, args.run_id


def result_from_summary(
    slug_dir: Path, summary_path: Path, *, run_id: str | None = None
) -> dict[str, Any]:
    data = load_json(summary_path)
    config = data.get("config") or {}
    stats = data.get("stats") or {}
    if not isinstance(config, dict) or not isinstance(stats, dict):
        raise ValueError(f"{summary_path} missing object-valued config/stats")

    model = str(config.get("model") or config.get("model_path") or slug_dir.name)
    base_slug = slugify_model(model)
    config_id = "default"
    if slug_dir.name.startswith(base_slug + "__"):
        config_id = slug_dir.name[len(base_slug) + 2 :]
    elif run_id_config := config_id_from_run_id(run_id or ""):
        config_id = run_id_config

    num_valid = int(stats.get("num_valid", 0) or 0)
    num_tasks = int(stats.get("num_tasks", 0) or 0)
    num_samples = int(stats.get("num_samples", 0) or 0)
    mer_raw = stats.get("mean_episode_return")
    mean_episode_return = float(mer_raw) if mer_raw is not None else None

    if num_tasks > 0 and num_valid >= num_tasks:
        status = "complete"
    elif num_valid > 0:
        status = "partial"
    else:
        status = "empty"

    return {
        "model": model,
        "slug": slug_dir.name,
        "base_slug": base_slug,
        "config_id": config_id,
        "status": status,
        "num_valid": num_valid,
        "num_tasks": num_tasks,
        "num_samples": num_samples,
        "mean_episode_return": mean_episode_return,
        "summary_path": relpath(summary_path),
        "artifact_path": relpath(slug_dir),
    }


def sort_key(row: dict[str, Any]) -> tuple[int, float, str]:
    status_rank = {"complete": 0, "partial": 1, "empty": 2}.get(row["status"], 3)
    mer = row.get("mean_episode_return")
    mer_key = -float(mer) if mer is not None else float("inf")
    return status_rank, mer_key, row["slug"]


def collect_results(
    run_dir: Path, *, run_id: str | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    if not run_dir.is_dir():
        raise FileNotFoundError(f"run dir not found: {run_dir}")

    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        summary_path = child / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            results.append(result_from_summary(child, summary_path, run_id=run_id))
        except Exception as exc:  # keep one bad summary from hiding the rest
            errors.append(f"{relpath(summary_path)}: {exc}")

    results.sort(key=sort_key)
    return results, errors


def _runtime_stamp(run_dir: Path) -> dict[str, Any]:
    stamp: dict[str, Any] = {}
    info_path = run_dir / "run_info.txt"
    if info_path.is_file():
        for line in info_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
            elif "=" in line:
                key, value = line.split("=", 1)
            else:
                continue
            key = key.strip().lower().replace("-", "_").replace(" ", "_")
            value = value.strip()
            if key and value:
                stamp[key] = value

    env_mode = os.environ.get("CUA_LITE_EVAL_RUNTIME_MODE")
    if env_mode:
        stamp.setdefault("runtime_mode", env_mode)
    session_id = os.environ.get("SESSION_ID")
    if session_id:
        stamp.setdefault("session_id", session_id)
    if "env_server_url_present" not in stamp:
        stamp["env_server_url_present"] = bool(os.environ.get("CUA_LITE_ENV_SERVER_URL"))
    if "env_server_token_present" not in stamp:
        stamp["env_server_token_present"] = bool(os.environ.get("CUA_LITE_ENV_SERVER_TOKEN"))
    return stamp


def commit_info(commit_name: str) -> dict[str, str | None]:
    sha = commit_name.split("_")[-1]
    info: dict[str, str | None] = {"short_sha": sha, "subject": None}
    if not re.fullmatch(r"[0-9a-f]{7,40}", sha):
        return info

    try:
        import subprocess

        subject = subprocess.check_output(
            ["git", "show", "-s", "--format=%s", sha],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        subject = ""
    info["subject"] = subject or None
    return info


def build_payload(run_dir: Path, env_name: str, commit_name: str, run_id: str) -> dict[str, Any]:
    results, errors = collect_results(run_dir, run_id=run_id)
    completed = sum(1 for r in results if r["status"] == "complete")
    partial = sum(1 for r in results if r["status"] == "partial")
    empty = sum(1 for r in results if r["status"] == "empty")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "env": env_name,
        "commit_dir": commit_name,
        "run_id": run_id,
        "commit": commit_info(commit_name),
        "artifact_root": relpath(run_dir),
        "result_count": len(results),
        "status_counts": {
            "complete": completed,
            "partial": partial,
            "empty": empty,
        },
        "runtime": _runtime_stamp(run_dir),
        "results": results,
    }
    if errors:
        payload["errors"] = errors
    return payload


def default_output_path(env_name: str, commit_name: str, run_id: str) -> Path:
    return REPO_ROOT / "devs/exps/eval" / env_name / "logs" / commit_name / f"{run_id}.json"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write devs/exps/eval/<env>/logs/<commit>/<run_id>.json from "
            "run summary.json files."
        )
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        help="Path to .exps/eval/<env>/<commit-dir>/<run_id>.",
    )
    parser.add_argument("--env", dest="env_name", help="Eval env name, e.g. lite.osworld.")
    parser.add_argument(
        "--commit-dir",
        help="Commit dir basename or path to .exps/eval/<env>/<commit-dir>.",
    )
    parser.add_argument("--run-id", help="Run id, e.g. run_0 or run_1_textonly.")
    parser.add_argument("--output", type=Path, help="Override output JSON path.")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write JSON even if no model-level summary.json files are found.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run_dir, env_name, commit_name, run_id = resolve_run_dir(args)

    if not COMMIT_DIR_RE.match(commit_name):
        print(
            f"[update_run_json.py] warning: unusual commit dir name: {commit_name}",
            file=sys.stderr,
        )

    payload = build_payload(run_dir, env_name, commit_name, run_id)
    if payload["result_count"] == 0 and not args.allow_empty:
        message = (
            "[update_run_json.py] no model-level summary.json files found under "
            f"{relpath(run_dir)}"
        )
        print(
            message,
            file=sys.stderr,
        )
        return 1

    output = (
        args.output.resolve()
        if args.output
        else default_output_path(env_name, commit_name, run_id)
    )
    write_json_atomic(output, payload)
    print(
        f"[update_run_json.py] wrote {relpath(output)} "
        f"({payload['result_count']} results; {payload['status_counts']['complete']} complete)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
