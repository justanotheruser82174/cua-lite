#!/bin/bash
# Remove the cua-lite/webharbor.webvoyager Docker image (container-only env; nothing on the host
# to clean — steady-state container reap is `docker rm -fv webharbor.webvoyager-<port>`, done
# by WebVoyagerContainerServices).
#
# Usage:
#   uv run --no-sync bash lite/gym/envs/webharbor/webvoyager/scripts/uninstall.sh

set -euo pipefail

log() { echo "[webharbor.webvoyager uninstall] $*" >&2; }

IMAGE="cua-lite/webharbor.webvoyager:latest"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker image rm "$IMAGE"
    log "removed $IMAGE"
else
    log "nothing to remove ($IMAGE absent)"
fi

log "done"
