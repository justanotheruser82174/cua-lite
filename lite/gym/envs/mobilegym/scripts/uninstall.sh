#!/bin/bash
# Remove the cua-lite/mobilegym Docker image (container-only env; nothing on the host
# to clean — steady-state container reap is `docker rm -fv mobilegym-<port>`).
#
# Usage:
#   bash lite/gym/envs/mobilegym/scripts/uninstall.sh

set -euo pipefail

log() { echo "[mobilegym uninstall] $*" >&2; }

IMAGE="cua-lite/mobilegym:latest"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker image rm "$IMAGE"
    log "removed $IMAGE"
else
    log "nothing to remove ($IMAGE absent)"
fi

log "done"
