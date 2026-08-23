#!/bin/bash
# Build the cua-lite/mobilegym Docker image.
#
# Container-only (SINGLETON): the whole task lifecycle (browser
# pool + vite UI + RPC server) runs INSIDE the image. The Dockerfile is
# self-contained (D4/D5a) — it clones UPSTREAM Purewhiter/mobilegym at a pinned
# full SHA and builds the vite dist/ in a node stage, so there is NO host clone /
# npm / playwright / dataset download here. Build context is docker/.
#
# Safe to run repeatedly — skips only when the image is fresh; auto-rebuilds when
# docker/ sources changed. Force a fresh build
# with `install.sh rebuild`.
#
# Usage:
#   uv run --no-sync bash lite/gym/envs/mobilegym/scripts/install.sh           # build if missing/stale
#   uv run --no-sync bash lite/gym/envs/mobilegym/scripts/install.sh rebuild   # force rebuild
#   uv run --no-sync bash lite/gym/envs/mobilegym/scripts/install.sh status
#   uv run --no-sync bash lite/gym/envs/mobilegym/scripts/install.sh pull|push # GHCR distribution (src_hash-gated)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_DIR="$ENV_DIR/docker"
IMAGE="cua-lite/mobilegym:latest"

log() { echo "[mobilegym install] $*" >&2; }

image_exists() { docker image inspect "$1" >/dev/null 2>&1; }

# Build-source hashing + stale-skip live in the shared helper (one impl for all
# container envs); it stamps the same ``lite.src_hash`` the runtime freshness gate
# reads, via the same python hash, so build and launch can't drift.
source "$ENV_DIR/../../scripts/image_build.sh"

# No-arg build: auto-rebuilds when the Dockerfile/sources changed since the
# existing image was built; up-to-date images are skipped (fast no-op).
build() {
    if image_is_fresh "$IMAGE" mobilegym; then
        log "$IMAGE up to date (build sources unchanged); skipping build."
        return
    fi
    image_rm "$IMAGE"
    log "docker build -t $IMAGE (clones + builds mobilegym inside the image) ..."
    docker build -t "$IMAGE" --label "$(src_label mobilegym)" \
        -f "$DOCKER_DIR/Dockerfile" "$DOCKER_DIR"
    log "built $IMAGE"
}

rebuild() {
    image_rm "$IMAGE"
    build
}

status() {
    if image_exists "$IMAGE"; then
        local sz
        sz="$(docker image inspect --format='{{.Size}}' "$IMAGE" | awk '{printf "%.1fGB", $1/1024/1024/1024}')"
        if image_is_fresh "$IMAGE" mobilegym; then
            echo "image $IMAGE : PRESENT ($sz, up to date)"
        else
            echo "image $IMAGE : PRESENT ($sz, STALE — sources changed; run install.sh to rebuild)"
        fi
    else
        echo "image $IMAGE : MISSING"
    fi
}

case "${1:-build}" in
    build)    build ;;
    rebuild)  rebuild ;;
    pull)     do_pull "$IMAGE" mobilegym ;;
    push)     do_push "$IMAGE" mobilegym ;;
    status)   status ;;
    *)        echo "Usage: $0 [build|rebuild|pull|push|status]" >&2; exit 1 ;;
esac
