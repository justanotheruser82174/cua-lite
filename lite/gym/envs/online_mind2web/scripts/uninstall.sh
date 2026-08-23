#!/bin/bash
# Remove the cua-lite/online_mind2web Docker image (container-only env; nothing on the host
# to clean — steady-state container reap is `docker rm -fv online_mind2web-<port>`, done
# by OnlineMind2WebContainerServices).
#
# Usage:
#   uv run --no-sync bash lite/gym/envs/online_mind2web/scripts/uninstall.sh

set -euo pipefail

log() { echo "[online_mind2web uninstall] $*" >&2; }

IMAGE="cua-lite/online_mind2web:latest"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker image rm "$IMAGE"
    log "removed $IMAGE"
else
    log "nothing to remove ($IMAGE absent)"
fi

log "done"
