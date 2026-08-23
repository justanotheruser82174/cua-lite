#!/bin/bash
# Remove captcha env downloads: .cache/ assets (after stopping leftover procs).
#
# Leaves the venv's Python packages and the shared ~/.cache/ms-playwright
# Chromium untouched (both are used by other envs — mobilegym / browsergym).
#
# Usage:
#   bash lite/gym/envs/captcha/scripts/uninstall.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_DIR="$ENV_DIR/.cache"

log() { echo "[captcha uninstall] $*" >&2; }

# Stop leftover processes + /tmp files first.
bash "$SCRIPT_DIR/cleanup.sh" 2>/dev/null || true

if [ -d "$CACHE_DIR" ]; then
    rm -rf "$CACHE_DIR"
    log "removed $CACHE_DIR"
else
    log "nothing to remove"
fi

log "done"
