#!/bin/bash
# Remove androidworld Docker images.
#
# Usage:
#   bash lite/gym/envs/androidworld/scripts/uninstall.sh

set -euo pipefail

log() { echo "[androidworld uninstall] $*" >&2; }

for img in cua-lite/androidworld:latest cua-lite/androidworld:base; do
    if docker image inspect "$img" >/dev/null 2>&1; then
        docker image rm "$img"
        log "removed $img"
    fi
done

log "done"
