#!/bin/bash
# Install lite.cuagym — build ONE Docker image
# (cua-lite/lite.cuagym:latest) that serves browser-side upstream web/cross_app
# rows plus desktop-shaped CUA-Gym tasks, and import both task sets.
#
# Steps: fetch mirrored HF assets -> import upstream web/cross_app + desktop tasks (train.jsonl per
# backend) -> if a source build is needed, bootstrap Node, npm-build referenced
# mocks, stage mocks + union deps into the image context, and docker build (FROM
# cua-lite/lite.osworld:latest by default; override with LITE_CUAGYM_BASE_IMAGE
# for private throwaway validation; adds desktop doc libs + Node + mocks + chrome
# wrapper).
#
# Usage:
#   uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh           # provision + build if missing/stale
#   uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh rebuild   # force local dist + image rebuild
#   uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh provision # deps + assets + catalogs only
#   uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh pull      # pull fresh GHCR image
#   uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh push      # publish fresh local image
#   uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh assets    # maintainer: force-refresh mirrored HF assets
#   uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh status    # read-only state

set -euo pipefail
export DOCKER_BUILDKIT=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ENV_DIR/../../../../.." && pwd)"
CACHE_DIR="$ENV_DIR/.cache"
WEB_CACHE="$CACHE_DIR/web"
DESKTOP_CACHE="$CACHE_DIR/desktop"
HUB_DIR="$WEB_CACHE/cua-gym-hub"
CTX_DIR="$CACHE_DIR/_imgctx"
DOCKER_DIR="$ENV_DIR/docker"

# Build-source hashing + stale-skip live in the shared helper. Source this before
# any Python call so a bare-shell invocation fails with the shared uv/no-sync hint.
source "$ENV_DIR/../../../scripts/image_build.sh"
cd "$REPO_ROOT"

DEFAULT_IMAGE="$(python - "$ENV_DIR" <<'PY'
import sys
from lite.gym.utils import config as env_config
cfg = env_config.load(sys.argv[1])
print(cfg.env_kwargs["computer"]["image"])
PY
)"
IMAGE="${LITE_CUAGYM_IMAGE:-$DEFAULT_IMAGE}"
BASE="${LITE_CUAGYM_BASE_IMAGE:-cua-lite/lite.osworld:latest}"
OSWORLD_INSTALL="$ENV_DIR/../osworld/scripts/install.sh"

log() { echo "[lite.cuagym install] $*" >&2; }

check_node() {
    local fnm_dir="$HOME/.local/share/fnm" need=0
    if ! command -v node >/dev/null 2>&1; then need=1
    else local ver; ver="$(node --version | sed 's/^v//' | cut -d. -f1)"; [ "$ver" -lt 20 ] && need=1; fi
    if [ "$need" = "1" ]; then
        if [ ! -x "$fnm_dir/fnm" ]; then
            log "Node missing/<20; bootstrapping fnm into $fnm_dir ..."
            curl -fsSL https://fnm.vercel.app/install | bash -s -- --skip-shell
        fi
        export PATH="$fnm_dir:$PATH"
        eval "$("$fnm_dir/fnm" env --shell bash)"
        "$fnm_dir/fnm" install 20 && "$fnm_dir/fnm" use 20
    fi
    log "Node $(node --version) ✓"
}

fetch_hub() {
    local refresh="${1:-0}"
    local revision
    revision="$(python - <<'PY'
from lite.gym.envs.lite.cuagym.src.utils import dataset
print(dataset.asset_identity())
PY
)"
    if [ "$refresh" = "0" ] && [ -d "$HUB_DIR/websites" ] && \
       [ "$(cat "$HUB_DIR/.asset_revision" 2>/dev/null || true)" = "$revision" ]; then
        log "Hub mirror present at $revision, skipping fetch"
        return
    fi
    refresh=1
    log "fetching CUA-Gym-Hub mirror from HF assets ..."
    python - "$WEB_CACHE" "$HUB_DIR" "$refresh" <<'PY'
import sys
from pathlib import Path
from lite.gym.envs.lite.cuagym.src.utils import dataset

cache, hub, refresh = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == "1"
tar = dataset.download_hub(cache, force_download=refresh)
dataset.extract_tar_zst(tar, hub, refresh=refresh)
(hub / ".asset_revision").write_text(dataset.asset_identity() + "\n")
PY
}

_apps() {
    local apps_file="$WEB_CACHE/lite.cuagym_tasks/apps.txt"
    [ -f "$apps_file" ] && grep -v '^[[:space:]]*$' "$apps_file"
}

app_build_stamp() {
    python - "$1" <<'PY'
import hashlib
import sys
from pathlib import Path

from lite.gym.envs.lite.cuagym.src.utils import dataset

root = Path(sys.argv[1])
h = hashlib.sha256()
h.update(dataset.asset_identity().encode())
h.update(b"\0")
inputs = [
    root / "index.html",
    root / "package.json",
    root / "package-lock.json",
]
for pattern in ("*.css", "*.html", "*.js", "*.json", "*.mjs", "*.ts"):
    inputs.extend(sorted(root.glob(pattern)))
for dirname in ("public", "src"):
    tree = root / dirname
    if tree.is_dir():
        inputs.extend(sorted(p for p in tree.rglob("*") if p.is_file()))
for path in inputs:
    if not path.is_file():
        continue
    h.update(path.relative_to(root).as_posix().encode())
    h.update(b"\0")
    h.update(path.read_bytes())
    h.update(b"\0")
print(h.hexdigest())
PY
}

build_apps() {
    local force="${1:-0}"
    while read -r app; do
        [ -z "$app" ] && continue
        local d="$HUB_DIR/websites/$app"
        [ -d "$d" ] || { log "WARN: $app not in Hub, skipping"; continue; }
        local stamp stamp_file
        stamp="$(app_build_stamp "$d")"
        stamp_file="$d/.lite-build-stamp"
        if [ "$force" != "1" ] && [ -f "$d/dist/index.html" ] && [ "$(cat "$stamp_file" 2>/dev/null || true)" = "$stamp" ]; then
            log "$app: existing pinned dist, skipping build"
            continue
        fi
        log "$app: npm install + build ..."
        rm -rf "$d/dist"
        (
            cd "$d"
            npm install --no-audit --no-fund
            npm run build
        )
        # npm install creates package-lock.json for this HF snapshot. Stamp after
        # install/build so a clean first build is immediately considered fresh.
        stamp="$(app_build_stamp "$d")"
        printf '%s\n' "$stamp" > "$stamp_file"
    done < <(_apps)
    return 0
}

# require_uv comes from the shared image_build.sh sourced above, along with the
# rationale for why a bare-pip fallback is forbidden here.

import_tasks() {
    log "importing CUA-Gym upstream web/cross_app + desktop tasks (download dataset + extract bundles) ..."
    python -m lite.gym.envs.lite.cuagym.scripts.utils.import_tasks --backend web "$@"
    python -m lite.gym.envs.lite.cuagym.scripts.utils.import_tasks --backend desktop "$@"
}

fetch_assets() {
    local refresh="${1:-0}"
    log "installing python deps for task import ..."
    require_uv
    # This script is the ONLY owner of these four — none is declared in pyproject,
    # by design: env deps live in the env's install.sh so any worktree provisions
    # itself. huggingface_hub is the rollout-path one (module-top in
    # src/utils/dataset.py, reached from main.py's _register_tasks); pyarrow/pandas
    # are install-time only (function-local in read_tasks(), whose sole callers are
    # scripts/utils/import_{web,desktop}_tasks.py). Runs inside fetch_assets(), which
    # build() calls BEFORE the freshness gate, so a fresh image + fresh venv still
    # gets them.
    uv pip install -q pyarrow pandas zstandard huggingface_hub
    fetch_hub "$refresh"
    if [ "$refresh" = "1" ]; then
        import_tasks --force-download --refresh
    else
        import_tasks
    fi
}

stage_context() {
    rm -rf "$CTX_DIR"; mkdir -p "$CTX_DIR/mocks"
    cp \
        "$DOCKER_DIR/Dockerfile" \
        "$DOCKER_DIR/google-chrome" \
        "$DOCKER_DIR/page_health.py" \
        "$CTX_DIR/"
    local n=0
    while read -r app; do
        [ -z "$app" ] && continue
        local d="$HUB_DIR/websites/$app"
        [ -f "$d/dist/index.html" ] || { log "WARN: $app not built, skipping"; continue; }
        mkdir -p "$CTX_DIR/mocks/$app"
        cp -r "$d/dist" "$d/src" "$d/package.json" "$CTX_DIR/mocks/$app/" 2>/dev/null
        # vite preview serves the /state,/post,/go state API from the mock's vite
        # config, so copy every top-level .js/.ts — not just vite.config.js: some
        # mocks ship only vite.config.ts (asana/jira/google_drive), and some import
        # a sibling helper from the config (wechat's defaultState.js). Without these
        # the preview config fails to load / falls back to static serving, so setup
        # state injection 404s or the mock never comes up. (config imports under
        # src/ are already covered by the src/ copy above.)
        local config_files=()
        shopt -s nullglob
        config_files=("$d"/*.js "$d"/*.ts)
        shopt -u nullglob
        [ "${#config_files[@]}" -gt 0 ] && cp "${config_files[@]}" "$CTX_DIR/mocks/$app/"
        # The pinned Hub imports its optional hardened API plugin from a shared
        # directory. CUA-Lite uses the original in-container mock API, so remove
        # that optional import/registration from the staged runtime config.
        local staged_configs=()
        shopt -s nullglob
        staged_configs=("$CTX_DIR/mocks/$app"/vite.config.*)
        shopt -u nullglob
        if [ "${#staged_configs[@]}" -gt 0 ]; then
            sed -i \
                -e '/secureMockApiPlugin.*from/d' \
                -e 's/secureMockApiPlugin(),[[:space:]]*//g' \
                "${staged_configs[@]}"
        fi
        n=$((n + 1))
    done < <(_apps)
    # The caller may use a restrictive umask (for example 0077). Normalize the
    # staged tree so Docker COPY never bakes non-traversable top-level app dirs.
    # The image assigns ownership to `user`, which also lets Vite create each
    # mock's per-session .mock-states directory at runtime.
    chmod -R u+rwX,go+rX "$CTX_DIR/mocks"
    # union of every staged mock's deps (minus test-only tooling) for the shared
    # /opt/mocks/node_modules.
    python - "$CTX_DIR" <<'PY'
import json, os, sys
ctx = sys.argv[1]
deps = {}
for app in sorted(os.listdir(f"{ctx}/mocks")):
    pj = f"{ctx}/mocks/{app}/package.json"
    if not os.path.exists(pj):
        continue
    d = json.load(open(pj))
    for sect in ("dependencies", "devDependencies"):
        for k, v in (d.get(sect) or {}).items():
            deps.setdefault(k, v)
for junk in ("@playwright/test", "playwright", "eslint", "eslint-plugin-react",
             "eslint-plugin-react-hooks", "eslint-plugin-react-refresh",
             "@eslint/js", "globals"):
    deps.pop(junk, None)
pkg = {"name": "cua-gym-mocks", "private": True, "type": "module", "dependencies": deps}
open(f"{ctx}/mocks-package.json", "w").write(json.dumps(pkg, indent=2))
PY
    log "staged $n mocks into image context"
}

build_image() {
    log "ensuring base image $BASE ..."
    if [ "$BASE" = "cua-lite/lite.osworld:latest" ]; then
        bash "$OSWORLD_INSTALL"
    else
        docker image inspect "$BASE" >/dev/null
    fi
    log "building $IMAGE (FROM $BASE) ..."
    docker build -t "$IMAGE" \
      --build-arg "BASE_IMAGE=$BASE" \
      --label "$(src_label lite.cuagym)" \
      -f "$CTX_DIR/Dockerfile" "$CTX_DIR"
}

provision() {
    fetch_assets 0    # task catalogs + apps.txt for both runtime sides
}

build() {
    local force_apps="${1:-0}"
    provision
    if [ "$BASE" = "cua-lite/lite.osworld:latest" ] && image_is_fresh "$IMAGE" lite.cuagym; then
        log "$IMAGE up to date (build sources unchanged); skipping image rebuild. Run 'install.sh rebuild' to force."
        return
    fi
    check_node
    if [ "$BASE" != "cua-lite/lite.osworld:latest" ]; then
        log "private base $BASE requested; rebuilding $IMAGE to avoid stale parent layers"
    fi
    image_rm "$IMAGE"
    build_apps "$force_apps"        # builds dist/ for every web app in apps.txt
    stage_context
    build_image
    log "install complete ✓  Try: gym.make('lite.cuagym@<task_uuid>')"
}

status() {
    echo "Hub repo     : $([ -d "$HUB_DIR/websites" ] && echo PRESENT || echo MISSING)"
    local catalogs wj dj
    mapfile -t catalogs < <(python - "$WEB_CACHE" "$DESKTOP_CACHE" <<'PY'
import sys
from pathlib import Path

print(Path(sys.argv[1]) / "lite.cuagym_tasks" / "train.jsonl")
print(Path(sys.argv[2]) / "lite.cuagym_desktop_tasks" / "train.jsonl")
PY
)
    wj="${catalogs[0]}"
    dj="${catalogs[1]}"
    echo "web tasks    : $([ -f "$wj" ] && wc -l < "$wj" || echo 0)"
    echo "desktop tasks: $([ -f "$dj" ] && wc -l < "$dj" || echo 0)"
    local built=0 total=0
    while read -r app; do
        [ -z "$app" ] && continue
        total=$((total + 1))
        local d="$HUB_DIR/websites/$app" stamp
        stamp="$(app_build_stamp "$d" 2>/dev/null || true)"
        [ -n "$stamp" ] \
            && [ -f "$d/dist/index.html" ] \
            && [ "$(cat "$d/.lite-build-stamp" 2>/dev/null || true)" = "$stamp" ] \
            && built=$((built + 1))
    done < <(_apps)
    echo "local dists  : $built / $total (source-build cache; fresh pulled images may already contain mocks)"
    if [ "$BASE" = "cua-lite/lite.osworld:latest" ]; then
        echo "source base  : $(image_is_fresh "$BASE" lite.osworld && echo FRESH || { docker image inspect "$BASE" >/dev/null 2>&1 && echo STALE || echo MISSING; }) ($BASE; only needed for local source builds)"
    else
        echo "source base  : $(docker image inspect "$BASE" >/dev/null 2>&1 && echo PRESENT || echo MISSING) ($BASE; private override, only needed for local source builds)"
    fi
    echo "image        : $(image_is_fresh "$IMAGE" lite.cuagym && echo FRESH || { docker image inspect "$IMAGE" >/dev/null 2>&1 && echo STALE || echo MISSING; }) ($IMAGE)"
}

pull() {
    do_pull "$IMAGE" lite.cuagym
    provision
}

push() {
    if [ "$BASE" != "cua-lite/lite.osworld:latest" ]; then
        echo "ERROR: refusing to push $IMAGE built against private base $BASE" >&2
        echo "       Unset LITE_CUAGYM_BASE_IMAGE and rebuild before publishing." >&2
        exit 1
    fi
    do_push "$IMAGE" lite.cuagym
}

case "${1:-build}" in
    build)     build ;;
    rebuild)   image_rm "$IMAGE"; build 1 ;;
    provision) provision ;;
    pull)      pull ;;
    push)      push ;;
    assets)    fetch_assets 1 ;;
    status)    status ;;
    *)         echo "Usage: $0 [build|rebuild|provision|pull|push|assets|status]" >&2; exit 1 ;;
esac
