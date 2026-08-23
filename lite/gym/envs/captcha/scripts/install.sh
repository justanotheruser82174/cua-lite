#!/bin/bash
# Install the captcha env: Python deps, Playwright Chromium, HF assets.
#
# Everything downloaded lands in a cache and re-runs are no-ops:
#   $ENV_DIR/.cache/assets/      — challenge images + JSON configs
#                                  (HF dataset OnAnOrange/captcha-assets, ~75 MB)
#   ~/.cache/ms-playwright/      — Chromium binaries (playwright default)
#
# Python deps (flask / pillow / playwright / huggingface_hub) are installed
# into the active venv here — intentionally NOT a pyproject extra, matching
# mobilegym / browsergym: env-specific deps stay out of the core lockfile.
#
# Usage:
#   uv run --no-sync bash lite/gym/envs/captcha/scripts/install.sh           # deps + Chromium + assets
#   uv run --no-sync bash lite/gym/envs/captcha/scripts/install.sh rebuild    # force re-download
#   uv run --no-sync bash lite/gym/envs/captcha/scripts/install.sh status     # check state

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_DIR="$ENV_DIR/.cache"
ASSETS_DIR="$CACHE_DIR/assets"
# Same sentinel main.py's _register_tasks() checks (fails loud if absent).
ASSETS_SENTINEL="$ASSETS_DIR/icon_click/train_eval.json"

log() { echo "[captcha install] $*" >&2; }

# No image here, but the `uv run --no-sync bash` invocation contract (bare `python`
# + `uv pip`, never bare pip) is the same one every install.sh signs; it and
# ``require_uv`` live in the shared helper.
source "$SCRIPT_DIR/../../../scripts/image_build.sh"

# ---------------------------------------------------------------------------
# Install steps
# ---------------------------------------------------------------------------

do_python_deps() {
    log "installing Python dependencies..."
    require_uv
    uv pip install -q "flask>=3.0" "pillow>=10.0" "playwright>=1.40" "huggingface_hub>=0.20"
}

do_chromium() {
    # `playwright install` is itself idempotent (checksums the installed
    # revision and skips the download when present).
    log "playwright install chromium..."
    python -m playwright install chromium
}

do_chromium_system_deps() {
    # Chromium needs system shared libs (libatk, libnss, libgbm, etc.) that
    # `playwright install chromium` does NOT install. Fresh Ubuntu Server
    # images don't have them. Idempotent — apt-get install of already-present
    # packages is a no-op. Same logic as mobilegym's install.sh.
    log "installing Chromium system libs..."
    if [ "$(id -u)" = "0" ]; then
        python -m playwright install-deps chromium
    elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        sudo env PATH="$PATH" python -m playwright install-deps chromium
    else
        echo "WARNING: passwordless sudo unavailable; skipping system libs." >&2
        echo "  Run manually:  sudo python -m playwright install-deps chromium" >&2
    fi
}

do_assets() {
    if [ -f "$ASSETS_SENTINEL" ]; then
        log "assets: $ASSETS_DIR exists, skipping download"
        return
    fi
    log "downloading captcha assets (~75 MB) to $ASSETS_DIR ..."
    python "$SCRIPT_DIR/utils/download_assets.py"
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

build() {
    do_python_deps
    do_chromium
    do_chromium_system_deps
    do_assets
    log "install complete ✓"
}

rebuild() {
    log "rebuilding (removing $ASSETS_DIR)..."
    rm -rf "$ASSETS_DIR"
    build
}

status() {
    echo "assets           : $([ -f "$ASSETS_SENTINEL" ] && echo "PRESENT ($ASSETS_DIR)" || echo MISSING)"
    echo "flask            : $(python -c 'import flask; print(flask.__version__)' 2>/dev/null || echo MISSING)"
    echo "pillow           : $(python -c 'import PIL; print(PIL.__version__)' 2>/dev/null || echo MISSING)"
    echo "huggingface_hub  : $(python -c 'import huggingface_hub; print(huggingface_hub.__version__)' 2>/dev/null || echo MISSING)"
    echo "playwright       : $(python -m playwright --version 2>/dev/null || echo MISSING)"
    echo "chromium         : $(python - <<'EOF' 2>/dev/null || echo MISSING
import os
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    path = p.chromium.executable_path
    print(f"PRESENT ({path})" if os.path.exists(path) else "MISSING")
EOF
)"
}

case "${1:-build}" in
    build)    build ;;
    rebuild)  rebuild ;;
    status)   status ;;
    *)        echo "Usage: $0 [build|rebuild|status]" >&2; exit 1 ;;
esac
