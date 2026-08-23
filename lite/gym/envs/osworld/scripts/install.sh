#!/bin/bash
# Install osworld (v1) — self-contained + idempotent. EVAL-IN-CONTAINER: v1's `desktop_env`
# is baked into a DERIVED image (docker build), NOT installed on the host — so osworld +
# osworld_2 add NO desktop_env to the host uv. (lite.osworld still host-installs the v1
# `osworld` dist until it too is migrated; it shares v1's pin so there's no conflict — the
# Dockerfile's pin must stay byte-identical to lite.osworld's.) Two things:
#   image → docker build docker/Dockerfile → cua-lite/osworld:latest
#           (FROM happysixd/osworld-docker + Python venv + v1 desktop_env + osworld_evaluation_examples + docker/server.py)
#   qcow2 → provision <env>/.cache/Ubuntu.qcow2 (symlink a repo-root docker_vm_data/Ubuntu.qcow2 if present, else HF download)
# The qcow2 is downloaded on the HOST and bind-mounted into each container at run time.
#
# Usage:
#   uv run --no-sync bash lite/gym/envs/osworld/scripts/install.sh          # full install (idempotent; rebuilds if image sources changed)
#   uv run --no-sync bash lite/gym/envs/osworld/scripts/install.sh status   # image freshness / qcow2 / KVM
#   uv run --no-sync bash lite/gym/envs/osworld/scripts/install.sh rebuild  # force-rebuild the image
#   uv run --no-sync bash lite/gym/envs/osworld/scripts/install.sh pull|push # GHCR distribution (src_hash-gated)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ENV_DIR/../../../.." && pwd)"
CACHE_DIR="$ENV_DIR/.cache"
IMAGE="cua-lite/osworld:latest"
DOCKER_DIR="$ENV_DIR/docker"

# Shared image-build helpers (image_is_fresh / src_label / do_pull / do_push). The gate stamps a
# lite.src_hash label at build; the runtime preflight (image_for("osworld")) re-checks it, so
# editing docker/Dockerfile auto-triggers a rebuild here. docker/server.py is bind-mounted over
# the baked copy at run time, so it is EXCLUDED from the hash — its edits hot-reload without a rebuild.
source "$REPO_ROOT/lite/gym/scripts/image_build.sh"

read -r QCOW2_FILENAME QCOW2_SIZE QCOW2_URL < <(
    python - "$ENV_DIR/data/release.json" <<'PY'
import json
import sys

release = json.load(open(sys.argv[1], encoding="utf-8"))
qcow2 = release["qcow2"]
print(qcow2["filename"], qcow2["size"], qcow2["url"])
PY
)
QCOW2="$CACHE_DIR/$QCOW2_FILENAME"
DVM="$REPO_ROOT/docker_vm_data/$QCOW2_FILENAME"

build_image() {
    if [ "${1:-}" != "force" ] && image_is_fresh "$IMAGE" osworld; then
        echo "[install] image $IMAGE up-to-date (src_hash match); skipping build (pass 'rebuild' to force)." >&2
        return 0
    fi
    image_rm "$IMAGE"   # drop the stale tag so the rebuild can't ride the old label
    echo "[install] docker build $IMAGE (bakes v1 desktop_env from the git fork — needs network) ..." >&2
    docker build -t "$IMAGE" --label "$(src_label osworld)" "$DOCKER_DIR"
}

provision_qcow2() {
    mkdir -p "$CACHE_DIR"
    if [ -e "$QCOW2" ]; then
        local sz; sz="$(stat -Lc%s "$QCOW2" 2>/dev/null || echo 0)"
        if [ "$sz" = "$QCOW2_SIZE" ]; then echo "[install] qcow2 present ($sz bytes); skipping." >&2; return 0; fi
        echo "[install] qcow2 present but wrong size ($sz != $QCOW2_SIZE); reprovisioning." >&2
        rm -f "$QCOW2"
    fi
    if [ -f "$DVM" ]; then
        echo "[install] symlinking existing $DVM -> $QCOW2 (no re-download)." >&2
        ln -sf "$DVM" "$QCOW2"
    else
        echo "[install] downloading qcow2 (22.8 GiB) from $QCOW2_URL ..." >&2
        curl -L --fail -o "$CACHE_DIR/$QCOW2_FILENAME.zip" "$QCOW2_URL"
        ( cd "$CACHE_DIR" && unzip -o "$QCOW2_FILENAME.zip" && rm -f "$QCOW2_FILENAME.zip" )
    fi
    local sz; sz="$(stat -Lc%s "$QCOW2" 2>/dev/null || echo 0)"
    if [ "$sz" != "$QCOW2_SIZE" ]; then
        echo "[install] ERROR: qcow2 size $sz != expected $QCOW2_SIZE (truncated/corrupt)." >&2
        exit 1
    fi
    echo "[install] qcow2 ready ($sz bytes)." >&2
}

# The docker-free prerequisites: provision the host qcow2 (symlink or HF download)
# the runtime bind-mounts into each container. NO Docker image is touched. Split
# out so a fresh git worktree (shares the host docker daemon but has its own
# gitignored .cache/) can be provisioned on its own — without the image stage that
# rebuilds the SHARED cua-lite/osworld:latest a co-tenant may be using. install
# calls this after the image; pull gates the published image first, then
# provisions. The standalone `provision` verb runs just this.
provision() {
    provision_qcow2
}

status() {
    echo "image:  $(image_is_fresh "$IMAGE" osworld && echo fresh || { docker image inspect "$IMAGE" >/dev/null 2>&1 && echo 'STALE (run rebuild)' || echo MISSING; })  ($IMAGE)"
    if [ -e "$QCOW2" ]; then echo "qcow2:  $(stat -Lc%s "$QCOW2" 2>/dev/null) bytes  ($QCOW2)"; else echo "qcow2:  MISSING  ($QCOW2)"; fi
    echo "kvm:    $([ -e /dev/kvm ] && echo present || echo MISSING)  /dev/kvm"
    echo "tun:    $([ -e /dev/net/tun ] && echo present || echo MISSING)  /dev/net/tun"
    echo "(this env's desktop_env is in its image; note: lite.osworld still host-installs a v1 desktop_env until it migrates)"
}

case "${1:-install}" in
    status)    status ;;
    build)     build_image; provision; echo "[install] done ✓" >&2 ;;
    rebuild)   build_image force; provision ;;
    provision) provision ;;   # docker-free prerequisites only (host qcow2); safe on a shared host
    pull)      do_pull "$IMAGE" osworld; provision ;;
    push)      do_push "$IMAGE" osworld ;;
    install)   build_image; provision; echo "[install] done ✓" >&2 ;;
    *) echo "usage: install.sh [install|build|status|rebuild|provision|pull|push]   (no arg = full idempotent install)" >&2; exit 1 ;;
esac
