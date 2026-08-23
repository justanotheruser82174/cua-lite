#!/bin/bash
# Build the cua-lite/webharbor.webvoyager Docker image.
#
# Container-only (SINGLETON): WebHarbor's local site mirrors, the browser pool,
# and task lifecycle all run INSIDE one shared image. The Dockerfile is
# self-contained (D4/D5a): it clones pinned WebVoyager + WebHarbor SHAs and
# fetches WebHarbor's HF-managed assets at build time. Build context is docker/.
#
# Safe to run repeatedly — the no-arg build AUTO-DETECTS staleness: it rebuilds
# only when docker/ sources changed since the image was built (a content hash
# stored as the ``lite.src_hash`` image label) and skips otherwise. Editing the
# Dockerfile and re-running rebuilds; no stale image silently survives.
# ``rebuild`` forces a fresh image unconditionally.
#
# Usage:
#   uv run --no-sync bash lite/gym/envs/webharbor/webvoyager/scripts/install.sh           # build (auto-rebuild if stale)
#   uv run --no-sync bash lite/gym/envs/webharbor/webvoyager/scripts/install.sh rebuild   # force rebuild
#   uv run --no-sync bash lite/gym/envs/webharbor/webvoyager/scripts/install.sh status
#   uv run --no-sync bash lite/gym/envs/webharbor/webvoyager/scripts/install.sh health
#   uv run --no-sync bash lite/gym/envs/webharbor/webvoyager/scripts/install.sh pull|push # GHCR distribution (src_hash-gated)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_DIR="$ENV_DIR/docker"
IMAGE="cua-lite/webharbor.webvoyager:latest"

log() { echo "[webharbor.webvoyager install] $*" >&2; }

image_exists() { docker image inspect "$1" >/dev/null 2>&1; }

# Build-source hashing + stale-skip live in the shared helper (one impl for all
# container envs); it stamps the same ``lite.src_hash`` label the image
# freshness check reads, so editing the Dockerfile and re-running rebuilds instead of silently
# keeping the stale image (D4).
source "$ENV_DIR/../../../scripts/image_build.sh"

build() {
    if image_is_fresh "$IMAGE" webharbor.webvoyager; then
        log "$IMAGE up to date (build sources unchanged); skipping build."
        return
    fi
    image_rm "$IMAGE"
    log "docker build -t $IMAGE (clones WebVoyager + WebHarbor and fetches assets) ..."
    DOCKER_BUILDKIT=1 docker build -t "$IMAGE" --label "$(src_label webharbor.webvoyager)" \
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
        if image_is_fresh "$IMAGE" webharbor.webvoyager; then
            echo "image $IMAGE : PRESENT ($sz, up to date)"
        else
            echo "image $IMAGE : PRESENT ($sz, STALE — sources changed; run install.sh to rebuild)"
        fi
    else
        echo "image $IMAGE : MISSING"
    fi
}

health() {
    build
    local name="webharbor.webvoyager-health-$$"
    docker rm -fv "$name" >/dev/null 2>&1 || true
    log "starting temporary health container $name ..."
    docker run -d --name "$name" --init "$IMAGE" >/dev/null
    trap "docker rm -fv '$name' >/dev/null 2>&1 || true" EXIT

    local deadline=$((SECONDS + 180))
    while (( SECONDS < deadline )); do
        if docker exec "$name" python -c "$(cat "$DOCKER_DIR/healthcheck.py")" \
                2>/dev/null | grep -q True; then
            echo "health $IMAGE : OK"
            docker rm -fv "$name" >/dev/null 2>&1 || true
            trap - EXIT
            return
        fi
        sleep 2
    done

    log "health check failed; recent logs:"
    docker logs --tail 80 "$name" >&2 || true
    docker rm -fv "$name" >/dev/null 2>&1 || true
    trap - EXIT
    return 1
}

case "${1:-build}" in
    build)    build ;;
    rebuild)  rebuild ;;
    pull)     do_pull "$IMAGE" webharbor.webvoyager ;;
    push)     do_push "$IMAGE" webharbor.webvoyager ;;
    status)   status ;;
    health)   health ;;
    *)        echo "Usage: $0 [build|rebuild|pull|push|status|health]" >&2; exit 1 ;;
esac
