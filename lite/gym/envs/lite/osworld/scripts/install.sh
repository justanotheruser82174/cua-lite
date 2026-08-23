#!/bin/bash
# Build the lite.osworld Docker image (two-stage: base + additive).
# Skips the build only when the image is FRESH (build sources unchanged);
# auto-rebuilds the osworld additive layer when a Dockerfile/source changed.
# Force a fresh build with ``install.sh rebuild``.
#
# Stage 1 (cua-lite/sandbox.linux:latest): the SHARED GNOME-Shell (Ubuntu session) desktop
#   base — built by the sandbox installer (lite/gym/sandbox/scripts/install.sh)
#   from lite/gym/sandbox/docker/Dockerfile.linux. Shared with lite.demo.
# Stage 2 (cua-lite/lite.osworld:latest): LibreOffice/Chrome/VLC/etc + OSWorld Flask
#   server + the osworld appearance layer, FROM cua-lite/sandbox.linux:latest.
#   Context = the env's osworld/ root because the Dockerfile does `COPY docker/server/ ...`.
#
# Also pulls the synth task-asset bundle (pdf/html/photos referenced by
# train.synth tasks) from HF (cua-lite/lite.osworld-assets, pinned) into
# <env>/.cache/assets/pulled/synth/ — the bundle is NOT in git (see
# src/utils/assets.py).
#
# Usage:
#   uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/install.sh           # assets + build if missing/stale
#   uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/install.sh provision # deps + assets + catalogs only
#   uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/install.sh pull      # pull image, then provision
#   uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/install.sh rebuild   # force image rebuild
#   uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/install.sh status    # freshness/assets check
#
# Works both from the repo root (host) and inside a Slime container
# (where the repo is mounted at /workspaces/cua-lite).

set -euo pipefail

# Force BuildKit. Required because the Dockerfile (and the shared base) use
# ``RUN cat > /path <<'EOF' ... EOF`` heredocs
# which legacy builder doesn't parse — RUN terminates at ``<<'EOF'``
# and the body silently disappears, leaving startup scripts and
# supervisord configs empty (build succeeds, container fails to
# boot at runtime). Both Dockerfiles carry a ``# syntax=docker/dockerfile:1.x``
# directive to pin the frontend, but that directive is
# **ignored** when legacy builder is active. Setting this here covers
# Docker daemons (< 23.x) where legacy is still the default.
export DOCKER_BUILDKIT=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OSWORLD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$OSWORLD_DIR/../../../../.." && pwd)"
DOCKER_DIR="$OSWORLD_DIR/docker"
ASSETS_HELPER="$OSWORLD_DIR/scripts/utils/assets.sh"
TASKS_HELPER="$OSWORLD_DIR/scripts/utils/tasks.sh"

BASE_TAG="cua-lite/sandbox.linux:latest"   # the shared desktop base (was lite.osworld.base)
SANDBOX_DIR="$(cd "$OSWORLD_DIR/../../../sandbox" && pwd)"
ADDITIVE_TAG="cua-lite/lite.osworld:latest"
IMAGE="$ADDITIVE_TAG"

# Build-source hashing + stale-skip live in the shared helper (one impl for all
# envs). osworld sits one level deeper than other envs, so reach up three dirs.
source "$OSWORLD_DIR/../../../scripts/image_build.sh"

# Python deps (osworld for evaluators, pandas/requests for metrics). cua-computer
# was retired alongside the cua computer-server when host↔container moved to
# the exec-stdio transport (lite/gym/sandbox/exec_stdio/); the eval-side
# osworld package still provides ``desktop_env.evaluators.metrics.*``.
install_python_deps() {
    echo "[install] Installing Python dependencies..." >&2
    # `uv pip`, never a bare-pip fallback — the shared rule + its rationale live in
    # image_build.sh's require_uv (sourced above). Here the old fallback failed LOUD
    # only by accident: `requires-python = ">=3.12,<3.13"` makes a 3.10 pip abort on
    # this install; any change to that pin would turn it silent.
    require_uv
    # ``-e``: the repo installs itself EDITABLE, so ``import lite`` resolves to
    # THIS worktree. A non-editable ``"$REPO_ROOT[gym]"`` copies the tree into
    # site-packages, and every later import silently reads that snapshot instead
    # of the checkout — a fix you land does nothing, and generators that resolve
    # their output dir from ``lite.__file__`` write into site-packages (this is
    # exactly how cuagym's task import landed its train.jsonl somewhere the
    # runtime registration could not see it). Editable is what makes a fresh
    # worktree provision AND run itself.
    uv pip install -q \
        -e "$REPO_ROOT[gym]" \
        "osworld @ git+https://github.com/cua-lite/OSWorld.git@3b33421192da4de61abe52c8da5f7a8432633837" \
        "pandas>=2.0" \
        "requests>=2.28" \
        "huggingface_hub>=0.24" \
        "pyyaml>=6.0"
}

pull_assets() {
    bash "$ASSETS_HELPER" pull
}

ensure_catalogs() {
    bash "$TASKS_HELPER" ensure
}

# The docker-free prerequisites: python deps + synth assets + task catalogs.
# Everything needed to REGISTER and RUN the env, WITHOUT touching any Docker
# image. Split out so a fresh git worktree (which shares the host docker daemon
# but has its own working dir + gitignored data/.cache) can be fully provisioned
# on its own — without the image stage that rebuilds the SHARED
# cua-lite/sandbox.linux:latest / :latest a co-tenant may be using.
# build/rebuild/install call this; pull runs the remote hash gate first, then
# provisions host-side assets only after the image pull is accepted.
provision() {
    install_python_deps
    pull_assets
    ensure_catalogs
}

status() {
    echo "Base (shared)   : $BASE_TAG  (lite/gym/sandbox/docker/Dockerfile.linux)"
    echo "Dockerfile      : $DOCKER_DIR/Dockerfile"
    image_status_line "shared base   " "$BASE_TAG" lite.demo || true
    image_status_line "image         " "$ADDITIVE_TAG" lite.osworld || true
    bash "$ASSETS_HELPER" status || true
    bash "$TASKS_HELPER" status || true
}

# Two-stage image build, called by build/rebuild only. Stage 1 is the SHARED base,
# whose freshness (rebuild-on-source-change) is the sandbox installer's own job;
# stage 2 is the osworld additive layer.
build_image() {
    # Stage 1: the SHARED sandbox base (GNOME-Shell/Ubuntu session + Xvnc/noVNC).
    # Delegate to the sandbox installer — it freshness-checks + stamps the
    # src-hash label the image freshness check reads, and rebuilds only when the base
    # Dockerfile / exec-stdio server changed. Shared with lite.demo, so we never
    # force-drop it here.
    echo "[install] Ensuring $BASE_TAG (shared desktop base)" >&2
    bash "$SANDBOX_DIR/scripts/install.sh"

    # Stage 2: additive (apps + OSWorld server), FROM the base. Context = osworld/ root
    # because the Dockerfile copies docker/server/. The exec-stdio in-container server
    # (/opt/lite/stdio_server.py) is baked by the SHARED base, so nothing to stage here.
    echo "[install] Building $ADDITIVE_TAG (stage 2/2 — apps + OSWorld server)" >&2
    docker build -t "$ADDITIVE_TAG" \
      --label "$(src_label lite.osworld)" \
      -f "$DOCKER_DIR/Dockerfile" \
      "$OSWORLD_DIR"
}

# One function per verb (matches every other env's install.sh).
build() {
    provision
    if image_is_fresh "$IMAGE" lite.osworld; then
        echo "[install] $IMAGE up to date (build sources unchanged); skipping both stages. Run 'install.sh rebuild' to force." >&2
        return 0
    fi
    # stale/missing final image → rebuild the osworld layer. The shared base is
    # managed by the sandbox installer's own freshness check (do NOT drop it — it
    # is shared with lite.demo).
    build_image
}

rebuild() {
    provision
    # Drop only the osworld layer; the shared base is rebuilt (if stale) by the
    # sandbox installer inside build_image.
    image_rm "$ADDITIVE_TAG"
    build_image
}

# Dispatch: terminal case, one function per verb (no fall-through after esac).
case "${1:-build}" in
    build)     build ;;
    rebuild)   rebuild ;;
    provision) provision ;;   # docker-free prerequisites only (deps + assets + catalogs); safe on a shared host
    pull)      do_pull "$IMAGE" lite.osworld; provision ;;
    push)
        # image only — assets are published SEPARATELY via `assets.sh push`
        # (produces a new HF commit → manual assets.lock.yaml revision bump), on
        # their own cadence, not per image push.
        if ! image_is_fresh "$IMAGE" lite.osworld; then
            echo "[install] Refusing to push missing/stale $IMAGE; run install.sh build first." >&2
            exit 1
        fi
        do_push "$IMAGE" lite.osworld
        ;;
    status)    status ;;
    *)
        echo "Usage: $0 [build|rebuild|provision|pull|push|status]" >&2
        exit 1
        ;;
esac
