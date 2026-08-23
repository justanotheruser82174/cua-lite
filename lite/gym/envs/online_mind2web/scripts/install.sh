#!/bin/bash
# Build the cua-lite/online_mind2web Docker image; health is a separate verb.
#
# Usage:
#   uv run --no-sync bash lite/gym/envs/online_mind2web/scripts/install.sh
#   uv run --no-sync bash lite/gym/envs/online_mind2web/scripts/install.sh rebuild
#   uv run --no-sync bash lite/gym/envs/online_mind2web/scripts/install.sh status
#   uv run --no-sync bash lite/gym/envs/online_mind2web/scripts/install.sh health
#   uv run --no-sync bash lite/gym/envs/online_mind2web/scripts/install.sh pull|push

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_DIR="$ENV_DIR/docker"
IMAGE="cua-lite/online_mind2web:latest"

log() { echo "[online_mind2web install] $*" >&2; }

image_exists() { docker image inspect "$1" >/dev/null 2>&1; }

# Build-source hashing + stale-skip live in the shared helper (one impl for all
# container envs); it stamps the same ``lite.src_hash`` label the image
# freshness check reads, so editing the Dockerfile and re-running rebuilds instead of silently
# keeping the stale image (D4).
source "$ENV_DIR/../../scripts/image_build.sh"

# No-arg build: auto-rebuilds when docker/ sources changed since the existing
# image was built; up-to-date images are skipped (fast no-op).
build() {
    if image_is_fresh "$IMAGE" online_mind2web; then
        log "$IMAGE up to date (build sources unchanged); skipping build."
        return
    fi
    image_rm "$IMAGE"
    log "docker build -t $IMAGE (context: $DOCKER_DIR) ..."
    DOCKER_BUILDKIT=1 docker build -t "$IMAGE" --label "$(src_label online_mind2web)" \
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
        sz="$(
            docker image inspect --format='{{.Size}}' "$IMAGE" \
                | awk '{printf "%.1fGB", $1/1024/1024/1024}'
        )"
        if image_is_fresh "$IMAGE" online_mind2web; then
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
    local name="online_mind2web-health-$$"
    docker rm -fv "$name" >/dev/null 2>&1 || true
    log "starting temporary health container $name ..."
    docker run -d --name "$name" --init "$IMAGE" >/dev/null
    trap "docker rm -fv '$name' >/dev/null 2>&1 || true" EXIT

    local deadline=$((SECONDS + 90))
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
    pull)     do_pull "$IMAGE" online_mind2web ;;
    push)     do_push "$IMAGE" online_mind2web ;;
    status)   status ;;
    health)   health ;;
    *)        echo "Usage: $0 [build|rebuild|pull|push|status|health]" >&2; exit 1 ;;
esac
