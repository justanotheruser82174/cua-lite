#!/bin/bash
# Dump mobilegym's static task metadata from the docker image into a
# checked-in JSON file the host reads at module-import time. Lets the
# host enumerate task ids + register them with gym.registry WITHOUT
# importing the `bench_env` package (which would force the bench tree +
# playwright onto every env-server host). Mirrors androidworld's
# scripts/utils/tasks.sh.
#
# Re-run this whenever the mobilegym pin in docker/Dockerfile changes
# (tasks added/removed, difficulty/apps/answer_fields changed, splits
# edited). Check the result in. WS-D diffs a freshly-generated copy
# against the committed file and fails on drift.
#
# Usage:
#   bash lite/gym/envs/mobilegym/scripts/utils/tasks.sh
#
# Pre-req: cua-lite/mobilegym:latest image must exist (run scripts/install.sh first).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OUT="$SCRIPT_DIR/../../data/tasks.json"

if ! docker image inspect cua-lite/mobilegym:latest >/dev/null 2>&1; then
    echo "ERROR: cua-lite/mobilegym:latest not found. Run install.sh first." >&2
    exit 1
fi

DUMP_PY='
import json, sys
from pathlib import Path
import bench_env
from bench_env.task.registry import TaskRegistry

registry = TaskRegistry()
splits_dir = Path(bench_env.__file__).parent / "splits"

_DIFFICULTY_MAX_STEPS = {"L1": 15, "L2": 30, "L3": 45, "L4": 60}

def resolve_base_max_steps(cls):
    # Mirror main.py / server.py _resolve_task_max_steps: explicit
    # cls.max_steps, else difficulty default, else 30. Text-mode base
    # (the grounded +15 AnswerSheet bonus is added per-instance host-side).
    raw = getattr(cls, "max_steps", None)
    if raw is not None:
        return int(raw)
    return _DIFFICULTY_MAX_STEPS.get(getattr(cls, "difficulty", "L1"), 30)

def load_split(name):
    f = splits_dir / f"{name}.txt"
    if not f.is_file():
        return []
    return [ln.strip() for ln in f.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")]

# Mirror main.py _register_tasks: ("test","eval",42), ("train","train",None).
out = {}
for split_name, cua_split, seed in [("test", "eval", 42), ("train", "train", None)]:
    for task_id in load_split(split_name):
        try:
            cls = registry.get_by_id(task_id)
        except Exception as e:
            print(f"skip {task_id}: {e}", file=sys.stderr)
            continue
        out[task_id] = {
            "split": cua_split,
            "seed": seed,
            "difficulty": getattr(cls, "difficulty", "L1"),
            "apps": list(getattr(cls, "apps", []) or []),
            "objective": getattr(cls, "objective", "operate"),
            "scope": getattr(cls, "scope", "S1"),
            "composition": getattr(cls, "composition", "atomic"),
            "capabilities": list(getattr(cls, "capabilities", []) or []),
            "answer_fields": getattr(cls, "answer_fields", None) is not None,
            "max_steps": resolve_base_max_steps(cls),
        }

with open("/tmp/tasks.json", "w") as f:
    json.dump(out, f, indent=2, sort_keys=True)
print(f"wrote {len(out)} tasks to /tmp/tasks.json", file=sys.stderr)
'

CNAME="mobilegym-tasks-dump-$$"
docker run --name "$CNAME" --entrypoint python cua-lite/mobilegym:latest -c "$DUMP_PY"
mkdir -p "$(dirname "$OUT")"
docker cp "$CNAME:/tmp/tasks.json" "$OUT"
docker rm -fv "$CNAME" >/dev/null 2>&1 || true
echo "wrote $OUT"
