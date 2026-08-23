#!/bin/bash
# Remove the CUA-Lite-built mobileworld Docker image.
#
# Usage:
#   bash lite/gym/envs/mobileworld/scripts/uninstall.sh

set -euo pipefail

log() { echo "[mobileworld uninstall] $*" >&2; }

IMG=cua-lite/mobileworld:latest

if docker image inspect "$IMG" >/dev/null 2>&1; then
    docker image rm "$IMG"
    log "removed $IMG"
fi

log "done"
