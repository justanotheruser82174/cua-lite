#!/bin/bash
# Remove ScreenSpot-Pro cached dataset from HuggingFace cache.
#
# Usage:
#   bash lite/gym/envs/screenspot_pro/scripts/uninstall.sh

set -euo pipefail

log() { echo "[screenspot_pro uninstall] $*" >&2; }

# HuggingFace caches under ~/.cache/huggingface/hub/datasets--likaixin--ScreenSpot-Pro
CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub/datasets--likaixin--ScreenSpot-Pro"

if [ -d "$CACHE_DIR" ]; then
    rm -rf "$CACHE_DIR"
    log "removed $CACHE_DIR"
else
    log "nothing to remove"
fi

log "done"
