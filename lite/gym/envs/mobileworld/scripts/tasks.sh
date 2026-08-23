#!/bin/bash
# Dump MobileWorld's static task metadata from the docker image into a
# checked-in JSON file the host reads at module-import time. Lets the
# host enumerate task names + register them with gym.registry WITHOUT
# importing the mobile_world Python package (its deps live only in the image).
#
# The dump runs against the PINNED source baked into cua-lite/mobileworld:latest
# (cloned at build time from the upstream repo at a fixed SHA — never from a
# local checkout), with the container entrypoint overridden: enumerating the
# task registry only imports/instantiates task classes, so no emulator, KVM,
# or nested dockerd is needed.
#
# Re-run this script whenever the MOBILEWORLD_SHA pin in docker/Dockerfile
# changes (tasks added / removed, goals or tags changed, etc.) and check the
# result in.
#
# Usage:
#   bash lite/gym/envs/mobileworld/scripts/tasks.sh
#
# Pre-req: cua-lite/mobileworld:latest image must exist
# (run scripts/install.sh first).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OUT_DIR="$SCRIPT_DIR/../data"
OUT="$OUT_DIR/tasks.json"

IMG=cua-lite/mobileworld:latest

if ! docker image inspect "$IMG" >/dev/null 2>&1; then
    echo "ERROR: $IMG not found. Run install.sh first." >&2
    exit 1
fi

DUMP_PY='
import json, sys
from mobile_world.tasks.registry import TaskRegistry

reg = TaskRegistry()
out = {}
for name, task in reg.tasks.items():
    out[name] = {
        "goal": task.goal,
        "tags": sorted(task.task_tags),
        "apps": sorted(task.app_names),
        "snapshot_tag": task.snapshot_tag,
    }

# Write directly to a known path inside the container so stdout/stderr can
# carry loguru registry-scan noise without corrupting the JSON.
with open("/tmp/tasks.json", "w") as f:
    json.dump(out, f, indent=2, sort_keys=True, ensure_ascii=False)
print(f"wrote {len(out)} tasks to /tmp/tasks.json", file=sys.stderr)
'

mkdir -p "$OUT_DIR"

# Run inside a named container (entrypoint overridden — no emulator / nested
# dockerd boot) so we can docker-cp the JSON out cleanly.
NAME="lite-mw-tasks-dump-$$"
SCAN_LOG=$(mktemp)
trap 'docker rm -fv "$NAME" >/dev/null 2>&1 || true; rm -f "$SCAN_LOG"' EXIT

docker run -d --rm --name "$NAME" --entrypoint sleep "$IMG" 600 >/dev/null
docker exec -w /app/service "$NAME" uv run --no-sync python -c "$DUMP_PY" 2>&1 \
    | tee "$SCAN_LOG" >&2

# The registry swallows per-task instantiation failures (logs + skips), which
# would silently drop tasks from the dump — fail loudly instead.
if grep -q "Error instantiating\|Error loading tasks" "$SCAN_LOG"; then
    echo "ERROR: task registry scan dropped tasks (see log above); not writing $OUT" >&2
    exit 1
fi

docker cp "$NAME:/tmp/tasks.json" "$OUT"

echo "[tasks] wrote $OUT ($(python -c "import json;print(len(json.load(open('$OUT'))))") tasks)"
