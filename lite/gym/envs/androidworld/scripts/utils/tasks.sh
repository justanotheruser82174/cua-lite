#!/bin/bash
# Dump androidworld's static task metadata from the docker image into a
# checked-in JSON file the host reads at module-import time. Lets the
# host enumerate task names + register them with gym.registry WITHOUT
# importing the androidworld Python package.
#
# Re-run this script whenever the androidworld pin in
# docker/Dockerfile changes (new tasks added / removed, complexity
# changed, etc.). Check the result in. CI should diff
# tasks.json against a freshly-generated copy and fail on drift.
#
# Usage:
#   bash lite/gym/envs/androidworld/scripts/utils/tasks.sh
#
# Pre-req: cua-lite/androidworld:latest image must exist
# (run scripts/install.sh first).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OUT="$SCRIPT_DIR/../../data/tasks.json"

if ! docker image inspect cua-lite/androidworld:latest >/dev/null 2>&1; then
    echo "ERROR: cua-lite/androidworld:latest not found. Run install.sh first." >&2
    exit 1
fi

DUMP_PY='
import json, sys
from pathlib import Path
import android_world
from android_world import registry as aw_registry

tasks = aw_registry.TaskRegistry().get_registry("android_world")
metadata_path = Path(android_world.__file__).parent / "task_metadata.json"
upstream = {}
if metadata_path.is_file():
    with open(metadata_path) as f:
        for item in json.load(f):
            upstream[item["task_name"]] = item

out = {}
for name, cls in tasks.items():
    m = upstream.get(name, {})
    out[name] = {
        "complexity": getattr(cls, "complexity", None),
        "app_names": list(getattr(cls, "app_names", ())),
        "instruction_template": m.get("task_template", f"Complete the {name!r} task on Android."),
        "difficulty": m.get("difficulty", "unknown"),
        "tags": m.get("tags", []),
        "optimal_steps": m.get("optimal_steps", ""),
    }

# Write directly to a known path inside the container so stdout
# can carry pip / pydub warnings without corrupting the JSON.
with open("/tmp/tasks.json", "w") as f:
    json.dump(out, f, indent=2, sort_keys=True)
print(f"wrote {len(out)} tasks to /tmp/tasks.json", file=sys.stderr)
'

# Run inside a named container so we can docker-cp the JSON out cleanly.
NAME="lite-aw-tasks-dump-$$"
docker run -d --rm --name "$NAME" cua-lite/androidworld:latest sleep 60 >/dev/null
docker exec "$NAME" python -c "$DUMP_PY"
docker cp "$NAME:/tmp/tasks.json" "$OUT"
docker rm -fv "$NAME" >/dev/null

echo "[tasks] wrote $OUT ($(wc -l < "$OUT") lines)"
