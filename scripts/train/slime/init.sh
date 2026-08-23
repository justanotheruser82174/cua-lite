#!/usr/bin/env bash
# Per-container runtime init. Auto-run by slime/launch.sh.
#
# We use the upstream slimerl/slime image as-is (no custom Dockerfile). This
# script makes the bind-mounted cua-lite worktree the runtime source of truth:
#
#   0. Install iproute2 when possible for fast Ray port probing
#   1. HF auth via $HF_TOKEN (gated model downloads)
#   2. Slime submodule preflight check
#   3. wandb version floor
#   4. cuDNN force-pin
#   5. Apply dependency patches (scripts/train/docker/patches/)
#   6. Drop /root/slime so our submodule's editable install isn't shadowed
#   7. Editable-install `lite` and `slime` from the bind-mounted source
#   8. Bridge the shared HF cache into /root/models
#   9. Remove uv so env install helpers use system pip in this container
#
# Env-side dependencies (env docker images, Android apps, Redis, Chromium, ...)
# live on the env-server host behind `serve_env.py`.
#
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
CUA_LITE_ROOT="${CUA_LITE_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)}"

# =============================================================================================== #
#                                      Bootstrap Helpers                                          #
# =============================================================================================== #

# Retry pip install up to 3 times on transient network failures (pip index
# flakes, cuDNN ~500MB download interrupted). Last attempt drops `-q` so the
# real error surfaces if it still fails.
pip_with_retry() {
  for i in 1 2; do
    if pip install -q "$@"; then return 0; fi
    echo "[init] pip install failed (attempt $i/3) — retrying in 5s..." >&2
    sleep 5
  done
  pip install "$@"  # final attempt, verbose
}

# =============================================================================================== #
#                                    System Preconditions                                         #
# =============================================================================================== #

# 0. System packages: iproute2 provides ``ss`` for fast port scans.
#    ``scripts/train/utils/ray.sh`` falls back to Python/socket probing when
#    ``ss`` is unavailable, so this is an optimization rather than a hard
#    runtime dependency.
if ! command -v ss >/dev/null 2>&1; then
  apt-get update -qq >/dev/null && apt-get install -y -qq iproute2 >/dev/null \
    || echo "[init] WARNING: apt-get iproute2 failed; using Python port probe fallback" >&2
fi

# =============================================================================================== #
#                                  Auth, Dependencies, Patches                                    #
# =============================================================================================== #

# 1. HuggingFace login (no-op if HF_TOKEN unset)
[[ -n "${HF_TOKEN:-}" ]] && hf auth login --token "$HF_TOKEN" 2>/dev/null || true

# 2. Slime submodule preflight: the repo is bind-mounted, so the submodule
# must be initialized on the host first.
if [ ! -f "${CUA_LITE_ROOT}/slime/pyproject.toml" ]; then
  echo "error: slime submodule not initialized at ${CUA_LITE_ROOT}/slime/" >&2
  echo "  run on host: git submodule update --init --recursive" >&2
  exit 1
fi

# 3. wandb: keep container runs on the supported upload path.
pip_with_retry --upgrade 'wandb>=0.25'

# 4. cuDNN: force-pin last so later editable installs cannot override it.
pip_with_retry nvidia-cudnn-cu12==9.16.0.29

# 5. Apply dependency patches (currently megatron only). Idempotent — skips
# already-applied patches, so re-running init on a live container is safe.
bash "${CUA_LITE_ROOT}/scripts/train/docker/patches/apply.sh"

# =============================================================================================== #
#                               Bind-Mounted Source Takes Priority                                #
# =============================================================================================== #

# 6. Drop the slime base image's checkout so a bare `import slime` resolves to
# the bind-mounted submodule installed below.
rm -rf /root/slime

# 7. Editable installs against the bind-mounted source. --no-deps because
# all transitive deps are in the slime base image; cua-lite's framework
# has no runtime deps of its own (env-specific extras live on env nodes).
pip install -q --no-deps -e "${CUA_LITE_ROOT}"
pip install -q --no-deps -e "${CUA_LITE_ROOT}/slime"

# =============================================================================================== #
#                                      Shared Model Cache                                         #
# =============================================================================================== #

# 8. Bridge the shared HF hub cache into the /root/models layout the trainers read.
# slime/launch.sh mounts HF_SHARED_HUB_CACHE (read-only) at the standard cache path
# /root/.cache/huggingface/hub, where models live as `models--<org>--<name>/snapshots/<hash>/`.
# But megatron (--hf-checkpoint) and sglang (--model-path) read `/root/models/<org>/<name>/`
# — a different layout — so the bare mount alone is invisible to them. Symlink each cached
# snapshot to its /root/models/<org>/<name> path so staged-in models resolve without a
# re-download. An explicitly-staged checkpoint (a real dir already at the dst) always wins.
HUB=/root/.cache/huggingface/hub
if [ -d "$HUB" ]; then
  for d in "$HUB"/models--*/; do
    [ -d "$d" ] || continue                       # no cached models → skip
    rest=$(basename "$d"); rest=${rest#models--}
    org=${rest%%--*}; repo=${rest#*--}            # split on FIRST -- (org may contain dashes)
    ref="$d/refs/main"
    if [ -f "$ref" ]; then snap="$d/snapshots/$(cat "$ref")";
    else snap=$(find "$d/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null || true); fi
    [ -n "$snap" ] && [ -d "$snap" ] || continue  # incomplete cache entry → skip
    dst="/root/models/$org/$repo"
    [ -e "$dst" ] && continue                     # already staged (real dir or prior link) → keep it
    mkdir -p "$(dirname "$dst")"
    ln -sfn "${snap%/}" "$dst"
    echo "[init] shared cache: linked $org/$repo -> ${snap%/}"
  done
fi

# =============================================================================================== #
#                                      Runtime Guardrails                                         #
# =============================================================================================== #

# 9. Ensure uv is NOT available inside the container. Env install.sh
# scripts use a _pip() helper that picks `uv pip install` when uv is
# present; inside the slime container we always want plain `pip` so
# packages land in the system site-packages (no venv).
if command -v uv >/dev/null 2>&1; then
  pip uninstall -y uv 2>/dev/null || rm -f "$(command -v uv)" 2>/dev/null || true
fi
