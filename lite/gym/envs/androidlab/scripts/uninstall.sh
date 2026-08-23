#!/bin/bash
# Remove androidlab Docker image and cached downloads.
#
# Usage:
#   bash lite/gym/envs/androidlab/scripts/uninstall.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_DIR="${ANDROID_LAB_CACHE:-$ENV_DIR/.cache}"

log() { echo "[androidlab uninstall] $*" >&2; }

if docker image inspect cua-lite/androidlab:latest >/dev/null 2>&1; then
    docker image rm cua-lite/androidlab:latest
    log "removed cua-lite/androidlab:latest"
fi

if [ -d "$CACHE_DIR" ]; then
    rm -rf "$CACHE_DIR"
    log "removed $CACHE_DIR"
fi

log "done"
