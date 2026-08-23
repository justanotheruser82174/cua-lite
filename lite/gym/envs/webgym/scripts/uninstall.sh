#!/bin/bash
# Remove the cua-lite/webgym Docker image (and any stray build cache).
#
# Usage:
#   bash lite/gym/envs/webgym/scripts/uninstall.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE="cua-lite/webgym:latest"

log() { echo "[webgym uninstall] $*" >&2; }

if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker image rm "$IMAGE"
    log "removed $IMAGE"
else
    log "$IMAGE not present"
fi

# Steady-state reap of running containers is `docker rm -fv webgym-<port>` (done
# by WebGymContainerServices); nothing to pkill. Remove any stray build cache.
for d in "$ENV_DIR/docker/.cache" "$ENV_DIR/.cache"; do
    if [ -d "$d" ]; then
        rm -rf "$d"
        log "removed $d"
    fi
done

log "done"
