#!/bin/bash
# Build the cua-lite/webgym Docker image + install the host-side VLM-judge reference.
#
# webgym has TWO halves:
#   1. The OmniBoxes pool (redis + master + node + M instance-server Chromiums)
#      runs INSIDE one self-contained image — the image clones the pinned
#      OmniBoxes SHA and applies docker/patches/ at build time (docker/Dockerfile);
#      no host clone/patch/npm/redis, no dataset download.
#   2. The episode-end VLM judge (``evaluator.get_verifiable_reward``) runs
#      HOST-side in the env-server's python (NOT in the container — it just calls
#      the judge API). So the host venv needs the ``webgym`` package importable,
#      or the judge is disabled and EVERY reward is 0.0 (reset hard-fails unless
#      skip_eval=True). This script installs BOTH halves so a fresh checkout runs
#      experiments with NO extra manual step — pinned to the SAME upstream SHA the
#      Dockerfile uses (single source of truth: parsed from docker/Dockerfile).
#
# Safe to run repeatedly — the no-arg build AUTO-DETECTS staleness: it rebuilds
# only when the Dockerfile/patches changed since the image was built (a content
# hash of the build sources, stored as the ``lite.src_hash`` image label), and
# skips otherwise. So editing a patch and re-running install.sh (no arg) rebuilds;
# no stale image silently survives. The host install exits early if the evaluator
# already imports. ``rebuild`` forces a fresh image unconditionally.
#
# Usage:
#   uv run --no-sync bash lite/gym/envs/webgym/scripts/install.sh           # build (auto-rebuild if stale) + host judge
#   uv run --no-sync bash lite/gym/envs/webgym/scripts/install.sh rebuild   # force image rebuild unconditionally
#   uv run --no-sync bash lite/gym/envs/webgym/scripts/install.sh status    # show image present/stale + host judge
#   uv run --no-sync bash lite/gym/envs/webgym/scripts/install.sh pull|push # GHCR distribution (src_hash-gated)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_DIR="$ENV_DIR/docker"
IMAGE="cua-lite/webgym:latest"

log() { echo "[webgym install] $*" >&2; }

image_exists() { docker image inspect "$1" >/dev/null 2>&1; }

# Build-source hashing + stale-skip live in the shared helper (one impl for all
# container envs); it stamps the same ``lite.src_hash`` the runtime freshness gate
# reads, via the same python hash, so build and launch can't drift. It also owns
# ``require_uv`` and the `uv run --no-sync bash` invocation contract these scripts
# share (for webgym, a bare-pip fallback would leave the host judge missing and
# every reward silently 0.0).
source "$ENV_DIR/../../scripts/image_build.sh"

# No-arg build: auto-rebuilds when the Dockerfile/patches/sources changed since the
# existing image was built; up-to-date images are skipped (fast no-op).
build() {
    if image_is_fresh "$IMAGE" webgym; then
        log "$IMAGE up to date (build sources unchanged); skipping build."
        return
    fi
    image_rm "$IMAGE"
    log "docker build -t $IMAGE (context: $DOCKER_DIR) ..."
    DOCKER_BUILDKIT=1 docker build -t "$IMAGE" --label "$(src_label webgym)" \
        -f "$DOCKER_DIR/Dockerfile" "$DOCKER_DIR"
    log "built $IMAGE"
}

rebuild() {
    image_rm "$IMAGE"
    build
}

# The pinned upstream the Dockerfile clones — parsed from the Dockerfile's
# ``ARG WEBGYM_REPO=`` / ``ARG WEBGYM_SHA=`` so host + container never drift.
webgym_pin() {
    local repo sha
    # ``| sed 's/[[:space:]].*//'`` strips any trailing inline comment / quote so a
    # future ``ARG WEBGYM_SHA=abc  # bump`` can't leak junk into the git+ URL.
    repo="$(sed -n 's/^ARG WEBGYM_REPO=//p' "$DOCKER_DIR/Dockerfile" | head -1 | sed 's/[[:space:]].*//')"
    sha="$(sed -n 's/^ARG WEBGYM_SHA=//p' "$DOCKER_DIR/Dockerfile" | head -1 | sed 's/[[:space:]].*//')"
    echo "${repo}@${sha}"
}

# Install the host-side judge reference. The container-only pool extras
# (``[omnibox]``: redis/playwright/...) are NOT needed host-side — the judge only
# imports ``webgym.models.evaluator`` + calls the configured judge endpoint — so
# we install the BASE package (no extra) to keep the host venv light.
# NOTE (durability): ``webgym`` is NOT a pyproject dependency, so a later
# ``uv sync`` can evict this manual install and silently drop rewards to 0.0.
# ``status`` surfaces it (host judge: MISSING); re-run this script to restore.
install_host_evaluator() {
    if python -c "import webgym.models.evaluator" >/dev/null 2>&1; then
        log "host webgym judge already importable; skipping pip install."
        return
    fi
    local pin; pin="$(webgym_pin)"
    log "pip-installing host-side webgym judge reference (git+${pin}) ..."
    require_uv
    uv pip install "git+${pin}"
    if python -c "import webgym.models.evaluator" >/dev/null 2>&1; then
        log "host webgym judge installed + importable."
    else
        log "WARN host webgym judge still NOT importable after install — rewards will be 0.0." >&2
    fi
}

status() {
    if image_exists "$IMAGE"; then
        local sz
        sz="$(docker image inspect --format='{{.Size}}' "$IMAGE" | awk '{printf "%.1fGB", $1/1024/1024/1024}')"
        if image_is_fresh "$IMAGE" webgym; then
            echo "image $IMAGE : PRESENT ($sz, up to date)"
        else
            echo "image $IMAGE : PRESENT ($sz, STALE — sources changed; run install.sh to rebuild)"
        fi
    else
        echo "image $IMAGE : MISSING"
    fi
    if python -c "import webgym.models.evaluator" >/dev/null 2>&1; then
        echo "host judge        : PRESENT (webgym.models.evaluator importable)"
    else
        echo "host judge        : MISSING (rewards 0.0 — run install.sh)"
    fi
}

case "${1:-build}" in
    build)    build; install_host_evaluator ;;
    rebuild)  rebuild; install_host_evaluator ;;
    pull)     do_pull "$IMAGE" webgym; install_host_evaluator ;;
    push)     do_push "$IMAGE" webgym ;;
    status)   status ;;
    *)        echo "Usage: $0 [build|rebuild|pull|push|status]" >&2; exit 1 ;;
esac
