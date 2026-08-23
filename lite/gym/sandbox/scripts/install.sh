#!/usr/bin/env bash
# Build cua-lite/sandbox.linux — the sandbox-family desktop base image the lite.demo
# env runs. Stamps the build-source hash label (via the shared helper) so the
# runtime freshness gate (lite.gym.utils.backend.freshness) accepts it, and
# auto-rebuilds when the Dockerfile / exec-stdio server change.
#
# Usage:
#   uv run --no-sync bash lite/gym/sandbox/scripts/install.sh           # build if missing/stale
#   uv run --no-sync bash lite/gym/sandbox/scripts/install.sh rebuild   # force rebuild
#   uv run --no-sync bash lite/gym/sandbox/scripts/install.sh pull      # adopt matching GHCR image
#   uv run --no-sync bash lite/gym/sandbox/scripts/install.sh push      # publish fresh local image
#   uv run --no-sync bash lite/gym/sandbox/scripts/install.sh status    # image freshness
set -euo pipefail

SANDBOX_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"   # lite/gym/sandbox
source "$SANDBOX_DIR/../scripts/image_build.sh"
IMAGE="cua-lite/sandbox.linux:latest"

log() { echo "[sandbox install] $*" >&2; }

build() {
    if image_is_fresh "$IMAGE" lite.demo; then
        log "$IMAGE up to date (build sources unchanged); skipping build."
        return
    fi
    image_rm "$IMAGE"
    log "docker build -t $IMAGE (context: $SANDBOX_DIR) ..."
    DOCKER_BUILDKIT=1 docker build -t "$IMAGE" --label "$(src_label lite.demo)" \
        -f "$SANDBOX_DIR/docker/Dockerfile.linux" "$SANDBOX_DIR"
    log "built $IMAGE"
}

rebuild() {
    image_rm "$IMAGE"
    build
}

status() {
    image_status_line "image        " "$IMAGE" lite.demo || true
}

case "${1:-build}" in
    build)   build ;;
    rebuild) rebuild ;;
    status)  status ;;
    pull)    do_pull "$IMAGE" lite.demo ;;
    push)    do_push "$IMAGE" lite.demo ;;
    *)       echo "Usage: $0 [build|rebuild|pull|push|status]" >&2; exit 1 ;;
esac
