#!/bin/bash
# Remove the downloaded cua-bench datasets (.cache/). Mirrors osworld_g/screenspot_pro.
#
# Usage:
#   bash lite/gym/envs/cua/scripts/uninstall.sh
#   bash lite/gym/envs/cua/scripts/uninstall.sh --image   # also drop the trycua/cua-xfce base image

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_DIR="$ENV_DIR/.cache"
log() { echo "[cua uninstall] $*" >&2; }

if [ -d "$CACHE_DIR" ]; then
    rm -rf "$CACHE_DIR"
    log "removed $CACHE_DIR"
else
    log "no .cache to remove"
fi

if [ "${1:-}" = "--image" ] && command -v docker >/dev/null 2>&1; then
    docker rmi trycua/cua-xfce:latest 2>/dev/null && log "removed trycua/cua-xfce:latest" || true
fi

log "done"
