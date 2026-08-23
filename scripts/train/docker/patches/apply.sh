#!/bin/bash
# Apply patches to dependencies installed in the Slime container.
# Usage: bash scripts/train/docker/patches/apply.sh
# Safe to run multiple times (skips already-applied patches).
# Auto-invoked by scripts/train/slime/init.sh after the editable installs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- megatron: git patches against /root/Megatron-LM source ---
# Why patch at all: slime's Dockerfile PINS a Megatron commit
# (slime/docker/Dockerfile `ARG MEGATRON_COMMIT`) instead of tracking upstream,
# and that pin does not move when slime itself is released — v0.3.0 was built
# 2026-05-31 but carries the 2026-02-14 commit 1dcf0dafa. So upstream fixes
# merged after the pinned commit never arrive, no matter how new the image is.
# slime's own docker/patch/latest/megatron.patch does not cover these files.
# megatron-core is an editable install pointing at this checkout, so patching
# the source here is what actually takes effect at import time.
MEGATRON_DIR="/root/Megatron-LM"
if [ -d "$MEGATRON_DIR" ]; then
    for patch in "$SCRIPT_DIR"/megatron/*.patch; do
        [ -f "$patch" ] || continue
        name="$(basename "$patch")"
        if cd "$MEGATRON_DIR" && git apply --check "$patch" 2>/dev/null; then
            git apply "$patch"
            echo "[patches] applied $name"
        else
            echo "[patches] skipped $name (already applied or conflict)"
        fi
    done
else
    echo "[patches] megatron not found at $MEGATRON_DIR, skipping megatron patches."
fi
