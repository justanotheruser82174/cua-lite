#!/bin/bash
# Build the cua-lite/androidworld Docker image.
#
# Two-stage build (buildx's security.insecure entitlement does NOT pass
# /dev/kvm into the buildkit step, so we can't install apps during
# `docker build`):
#
#   1. `docker build` the Dockerfile → cua-lite/androidworld:base. This
#      stage has Python + JDK + SDK + AVD config + the staged
#      apps.sh, but the benchmark apps are NOT yet installed.
#
#   2. `docker run --device /dev/kvm --privileged` a builder container
#      from :base, `docker exec` it to invoke apps.sh (boots the
#      emulator, installs all benchmark APKs, grants permissions, then
#      gracefully halts the emulator), and `docker commit` the result
#      as cua-lite/androidworld:latest. The :base tag is left in place
#      (cheap, mostly shared layers) for fast rebuilds.
#
# Build host requirements:
#   - Docker (no buildx required)
#   - /dev/kvm readable + writable by the calling user
#
# Usage:
#   uv run --no-sync bash lite/gym/envs/androidworld/scripts/install.sh           # build if missing/stale
#   uv run --no-sync bash lite/gym/envs/androidworld/scripts/install.sh rebuild   # force rebuild
#   uv run --no-sync bash lite/gym/envs/androidworld/scripts/install.sh status    # image freshness/resources
#   uv run --no-sync bash lite/gym/envs/androidworld/scripts/install.sh pull|push # GHCR distribution (src_hash-gated)
set -euo pipefail

# Prefer BuildKit. The a11y patch is now a plain ``COPY patches/`` +
# ``RUN python3 docker/patches/a11y_patch.py`` (no heredoc), so legacy
# builder would also work, but BuildKit gives faster, cache-friendly
# builds. Idempotent on newer Docker (BuildKit is already the default
# on daemons >= 23.x).
export DOCKER_BUILDKIT=1

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
DOCKER_DIR=$(cd "$SCRIPT_DIR/../docker" && pwd)

# Build-source hashing + stale-skip live in the shared helper (one impl for all
# envs). androidworld is at lite/gym/envs/androidworld → utils is ../../utils.
source "$SCRIPT_DIR/../../../scripts/image_build.sh"

IMG=cua-lite/androidworld:latest
BASE_IMG=cua-lite/androidworld:base
BUILDER_CONTAINER=lite-android-world-apps-installer

log() { echo "[install] $*" >&2; }

image_exists() { docker image inspect "$1" >/dev/null 2>&1; }

ensure_kvm() {
    if [ ! -e /dev/kvm ]; then
        echo "ERROR: /dev/kvm does not exist on this host." >&2
        echo "  Cannot build the image — the apps-install step needs KVM." >&2
        echo "  Install KVM kernel module + ensure your user has access:" >&2
        echo "    sudo usermod -aG kvm \$(whoami) && newgrp kvm" >&2
        exit 1
    fi
    if [ ! -w /dev/kvm ]; then
        echo "ERROR: /dev/kvm exists but is not writable by \$(whoami)." >&2
        echo "  Pick one (one-time, sysadmin):" >&2
        echo "    sudo usermod -aG kvm \$(whoami) && newgrp kvm   # persistent" >&2
        echo "    sudo setfacl -m u:\$(id -u):rw /dev/kvm         # per-boot" >&2
        exit 1
    fi
}

cleanup_builder() {
    docker rm -fv "$BUILDER_CONTAINER" >/dev/null 2>&1 || true
}

build_base() {
    log "Stage 1: docker build $BASE_IMG"
    # docker/ is the build context — Dockerfile, apps.sh and
    # server.py all live there. Nothing outside this
    # directory needs to ship with the image.
    docker build -t "$BASE_IMG" --label "$(src_label androidworld)" "$DOCKER_DIR"
    log "  built $BASE_IMG"
}

install_apps_and_commit() {
    log "Stage 2: spawn privileged builder container with /dev/kvm passthrough"
    cleanup_builder

    docker run -d \
        --name "$BUILDER_CONTAINER" \
        --device /dev/kvm \
        --privileged \
        --entrypoint sleep \
        "$BASE_IMG" \
        infinity >/dev/null

    log "  exec apps.sh inside $BUILDER_CONTAINER (~5-10 min: boot + APK install)"
    # We want full output streamed to the build log, so no `-q` / no `&>`.
    if ! docker exec "$BUILDER_CONTAINER" bash /usr/local/bin/apps.sh; then
        log "ERROR: apps.sh failed; leaving $BUILDER_CONTAINER for inspection"
        log "       (run 'docker logs $BUILDER_CONTAINER' or 'docker exec -it $BUILDER_CONTAINER bash')"
        log "       remove with: docker rm -fv $BUILDER_CONTAINER"
        exit 1
    fi

    log "  commit $BUILDER_CONTAINER → $IMG"
    docker commit "$BUILDER_CONTAINER" "${IMG}-stage2-raw" >/dev/null
    cleanup_builder

    # Reset Entrypoint/Cmd. The builder container was created with
    # ``--entrypoint sleep infinity`` to keep it alive for ``docker exec``
    # to apps.sh; the committed image inherits that entrypoint, which
    # would silently break downstream ``docker run cua-lite/androidworld:latest
    # sleep 86400`` (executes as ``sleep sleep 86400`` → invalid arg).
    #
    # ``docker commit --change 'ENTRYPOINT []'`` does NOT reliably clear
    # the entrypoint (observed empirically on docker 28.5.1: inspect
    # shows ENTRYPOINT=[sleep] still). Wrapping with a one-line build is
    # the reliable way.
    log "  wrapping :stage2-raw → :latest (reset entrypoint)"
    WRAP_DIR=$(mktemp -d)
    cat > "$WRAP_DIR/Dockerfile" <<EOF
FROM ${IMG}-stage2-raw
ENTRYPOINT []
CMD ["/bin/bash"]
EOF
    docker build -t "$IMG" --label "$(src_label androidworld)" "$WRAP_DIR" >/dev/null
    rm -rf "$WRAP_DIR"
    image_rm "${IMG}-stage2-raw"

    log "built $IMG"
}

build() {
    # Skip BOTH stages when the final image is present AND fresh (build sources
    # under docker/ unchanged since it was built). A stale/missing image falls
    # through to a full rebuild.
    if image_is_fresh "$IMG" androidworld; then
        log "$IMG up to date (build sources unchanged); skipping build."
        return
    fi
    ensure_kvm
    # We're rebuilding because the final image is stale/missing. The base is built
    # from the same hashed docker/ sources, so a stale final image invalidates it
    # too — drop and rebuild it (else a base-source edit ships inside a "fresh"-
    # labeled final image, defeating the freshness gate).
    image_rm "$BASE_IMG"
    build_base
    install_apps_and_commit
}

rebuild() {
    cleanup_builder
    image_rm "$IMG"
    image_rm "$BASE_IMG"
    build
}

status() {
    echo "Dockerfile     : $DOCKER_DIR/Dockerfile"
    echo "apps.sh  : $DOCKER_DIR/apps.sh"
    if [ -w /dev/kvm ]; then
        echo "/dev/kvm       : accessible"
    else
        echo "/dev/kvm       : NOT accessible (cannot build until fixed)"
    fi
    image_status_line "build cache base" "$BASE_IMG" androidworld || true
    image_status_line "image          " "$IMG" androidworld || true
    if docker ps -a --format '{{.Names}}' | grep -q "^${BUILDER_CONTAINER}$"; then
        echo "builder ctr    : $BUILDER_CONTAINER (lingering — run '$0 rebuild' to clean up)"
    fi
}

case "${1:-build}" in
    build)    build ;;
    rebuild)  rebuild ;;
    pull)     do_pull "$IMG" androidworld ;;
    push)     do_push "$IMG" androidworld ;;
    status)   status ;;
    *)        echo "Usage: $0 [build|rebuild|pull|push|status]" >&2; exit 1 ;;
esac
