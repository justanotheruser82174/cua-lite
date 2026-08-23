#!/bin/bash
# Dump androidlab's static task list (138 entries) from the docker
# image into a checked-in tasks.json that the host reads at module
# import time. Lets the host register tasks with gym.registry WITHOUT
# importing the androidlab Python package.
#
# Re-run this script whenever the android-lab pin in docker/Dockerfile
# changes. Check the result in. CI should diff tasks.json against a
# freshly-generated copy and fail on drift.
#
# Usage:
#   bash lite/gym/envs/androidlab/scripts/utils/tasks.sh
#
# Pre-req: cua-lite/androidlab:latest image must exist
# (run scripts/install.sh first).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OUT="$SCRIPT_DIR/../../data/tasks.json"

if ! docker image inspect cua-lite/androidlab:latest >/dev/null 2>&1; then
    echo "ERROR: cua-lite/androidlab:latest not found. Run install.sh first." >&2
    exit 1
fi

DUMP_PY='
import importlib
import json
import sys
from pathlib import Path
import yaml as _yaml
import android_lab.evaluation

config_dir = Path(android_lab.evaluation.__file__).parent / "config"
out = {}
for yaml_path in sorted(config_dir.glob("*.yaml")):
    with open(yaml_path) as f:
        data = _yaml.safe_load(f)
    app_name = data.get("APP", "")
    package = data.get("package", "")
    for task in data.get("tasks", []):
        task_id = task.get("task_id")
        if not task_id:
            continue
        metric_func = task.get("metric_func", "")
        app_module_name = metric_func.rsplit(".", 1)[-1]
        # Validate the judge actually resolves (skip if not).
        try:
            module = importlib.import_module(
                f"android_lab.evaluation.tasks.{app_module_name}"
            )
            judge_cls = module.function_map.get(task_id)
        except Exception:
            continue
        if judge_cls is None:
            continue
        out[task_id] = {
            "app": app_name,
            "package": package,
            "category": task.get("category", ""),
            "metric_type": task.get("metric_type", ""),
            "adb_query": task.get("adb_query"),
            "instruction": (task.get("task") or "").strip(),
            # Container-side lookup keys used by /task/load:
            "judge_class_name": task_id,        # function_map key (==task_id)
            "app_module_name": app_module_name, # android_lab.evaluation.tasks.<this>
        }

with open("/tmp/tasks.json", "w") as f:
    json.dump(out, f, indent=2, sort_keys=True)
print(f"wrote {len(out)} tasks to /tmp/tasks.json", file=sys.stderr)
'

NAME="lite-al-tasks-dump-$$"
docker run -d --rm --name "$NAME" cua-lite/androidlab:latest sleep 60 >/dev/null
docker exec "$NAME" python -c "$DUMP_PY"
docker cp "$NAME:/tmp/tasks.json" "$OUT"
docker rm -fv "$NAME" >/dev/null

echo "[tasks] wrote $OUT ($(wc -l < "$OUT") lines)"
