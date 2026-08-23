#!/usr/bin/env bash
# Install lite.scalecua task catalogs. This env does not define a Docker image;
# it runs on a lite.osworld base image.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ENV_DIR/../../../../.." && pwd)"
OSWORLD_IMAGE="cua-lite/lite.osworld:latest"
OSWORLD_INSTALL="$ENV_DIR/../osworld/scripts/install.sh"
TASKS_HELPER="$SCRIPT_DIR/utils/tasks.sh"

source "$REPO_ROOT/lite/gym/scripts/image_build.sh"

log() { echo "[lite.scalecua install] $*" >&2; }

install_python_deps() {
    log "installing Python deps for ScaleCUA import ..."
    # `uv pip`, never a bare-pip fallback — the shared rule + its rationale live in
    # image_build.sh's require_uv (sourced above). Here the old fallback failed LOUD
    # only by accident: `requires-python = ">=3.12,<3.13"` makes a 3.10 pip abort on
    # the editable install below; any change to that pin would turn it silent.
    require_uv
    # ``-e`` — install THIS worktree editable, never a site-packages copy; see
    # the same note in ../../osworld/scripts/install.sh.
    uv pip install -q -e "$REPO_ROOT[gym]" huggingface_hub pyyaml xlrd
}

import_tasks() {
    log "importing ScaleCUA OSWorld catalogs ..."
    bash "$TASKS_HELPER" generate "$@"
}

check_tasks() {
    bash "$TASKS_HELPER" check
}

# Docker-free prerequisites: python deps + task catalogs (no image — this env
# reuses lite.osworld's). Split out so a fresh git worktree can be provisioned on
# its own, without the base-image stage that could touch a co-tenant's shared
# image. build/rebuild call this directly; pull delegates to lite.osworld pull
# first so its host deps/assets/catalogs and image are present. The `provision`
# verb runs just this.
# ``$@`` forwards to import_tasks (e.g. ``--force-download``).
provision() {
    install_python_deps
    import_tasks "$@"
    check_tasks
}

ensure_base_image() {
    log "ensuring base image $OSWORLD_IMAGE via lite.osworld installer ..."
    bash "$OSWORLD_INSTALL"
}

build() {
    ensure_base_image
    provision
    log "install complete. Try: gym.registry.task_ids('lite.scalecua', split='rl')"
}

rebuild() {
    ensure_base_image
    provision --force-download
}

status() {
    bash "$TASKS_HELPER" status || true
    if image_is_fresh "$OSWORLD_IMAGE" lite.osworld; then
        echo "base image   : FRESH ($OSWORLD_IMAGE)"
    elif docker image inspect "$OSWORLD_IMAGE" >/dev/null 2>&1; then
        echo "base image   : STALE ($OSWORLD_IMAGE)"
    else
        echo "base image   : MISSING ($OSWORLD_IMAGE)"
    fi
}

pull() {
    bash "$OSWORLD_INSTALL" pull
    provision
}

push() {
    echo "lite.scalecua has no image to push; it reuses $OSWORLD_IMAGE" >&2
    exit 2
}

case "${1:-build}" in
    build)     build ;;
    rebuild)   rebuild ;;
    provision) shift; provision "$@" ;;   # docker-free prerequisites (deps + catalogs), no image
    import)    shift; provision "$@" ;;   # alias for provision (kept for back-compat)
    status)    status ;;
    pull)      pull ;;
    push)      push ;;
    *)
        echo "Usage: $0 [build|rebuild|provision|import|status|pull|push]" >&2
        exit 1
        ;;
esac
