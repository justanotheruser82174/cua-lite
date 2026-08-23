#!/bin/bash
# Remove OSWorld-G cloned benchmark data.
#
# Usage:
#   bash lite/gym/envs/osworld_g/scripts/uninstall.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_DIR="$ENV_DIR/.cache"

log() { echo "[osworld_g uninstall] $*" >&2; }

if [ -d "$CACHE_DIR" ]; then
    rm -rf "$CACHE_DIR"
    log "removed $CACHE_DIR"
else
    log "nothing to remove"
fi

log "done"
