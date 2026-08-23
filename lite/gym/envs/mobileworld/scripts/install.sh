#!/bin/bash
# Build the cua-lite/mobileworld Docker image.
#
# Single-stage build: `docker build` pulls the upstream prebuilt base
# (ghcr.io/tongyi-mai/mobile_world pinned by digest, ~10.5 GB compressed — the AVD
# snapshot + app-backend images it contains are not in the upstream git repo,
# so it cannot be built from source) and overlays the upstream source pinned
# at a fixed SHA. Unlike androidworld there is NO KVM-requiring stage-2: the
# device snapshot is already baked into the base image. KVM (+ --privileged
# + /dev/kvm) is a RUN-time requirement only.
#
# Usage:
#   uv run --no-sync bash lite/gym/envs/mobileworld/scripts/install.sh           # build if missing/stale
#   uv run --no-sync bash lite/gym/envs/mobileworld/scripts/install.sh rebuild   # force rebuild
#   uv run --no-sync bash lite/gym/envs/mobileworld/scripts/install.sh status    # image freshness/resources
#   uv run --no-sync bash lite/gym/envs/mobileworld/scripts/install.sh pull|push # GHCR distribution (src_hash-gated)
set -euo pipefail

export DOCKER_BUILDKIT=1

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
DOCKER_DIR=$(cd "$SCRIPT_DIR/../docker" && pwd)

# Build-source hashing + stale-skip live in the shared helper (one impl for all
# envs). mobileworld is at lite/gym/envs/mobileworld → scripts is ../../../scripts.
source "$SCRIPT_DIR/../../../scripts/image_build.sh"

IMG=cua-lite/mobileworld:latest
BASE_IMG=ghcr.io/tongyi-mai/mobile_world@sha256:b680380eac98a7ad064707f9653772af18554d201a3e6e7cf8f15d58cdc73240

log() { echo "[install] $*" >&2; }

image_exists() { docker image inspect "$1" >/dev/null 2>&1; }

# KVM is a run-time (not build-time) requirement — warn loudly, don't fail.
warn_kvm() {
    if [ ! -e /dev/kvm ]; then
        log "WARNING: /dev/kvm does not exist on this host."
        log "  The image will build, but containers will NOT run here."
        log "  Install the KVM kernel module + grant access:"
        log "    sudo usermod -aG kvm \$(whoami) && newgrp kvm"
    elif [ ! -r /dev/kvm ] || [ ! -w /dev/kvm ]; then
        log "WARNING: /dev/kvm exists but is not accessible by $(whoami)."
        log "  The image will build, but containers will NOT run here. Fix (one-time):"
        log "    sudo usermod -aG kvm \$(whoami) && newgrp kvm   # persistent"
        log "    sudo setfacl -m u:\$(id -u):rw /dev/kvm         # per-boot"
    fi
}

build() {
    if image_is_fresh "$IMG" mobileworld; then
        log "$IMG up to date (build sources unchanged); skipping build."
        return
    fi
    warn_kvm
    log "docker build $IMG (from $BASE_IMG + pinned upstream source overlay;"
    log "  first build pulls the base into the BuildKit cache — ~10.5 GB compressed)"
    docker build -t "$IMG" --label "$(src_label mobileworld)" "$DOCKER_DIR"
    log "built $IMG"
}

rebuild() {
    image_rm "$IMG"
    build
}

status() {
    echo "Dockerfile : $DOCKER_DIR/Dockerfile"
    if [ -w /dev/kvm ]; then
        echo "/dev/kvm   : accessible"
    else
        echo "/dev/kvm   : NOT accessible (build OK, but containers cannot run here)"
    fi
    # The base image is pulled into BuildKit's cache during ``docker build``,
    # not tagged in the local image store — so only the final image is listed.
    image_status_line "image     " "$IMG" mobileworld || true
}

case "${1:-build}" in
    build)    build ;;
    rebuild)  rebuild ;;
    pull)     do_pull "$IMG" mobileworld ;;
    push)     do_push "$IMG" mobileworld ;;
    status)   status ;;
    *)        echo "Usage: $0 [build|rebuild|pull|push|status]" >&2; exit 1 ;;
esac
