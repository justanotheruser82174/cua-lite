#!/usr/bin/env bash
# Lifecycle for a gym-anything software image cua-lite/lite.cuaworld.<software>:latest,
# built on the shared sandbox base with the `ga` desktop user (USER=ga build-arg —
# the upstream assets hardcode `ga` in prompts/verifiers/scripts, so we match the
# container to the assets rather than rewrite thousands of refs). Stages the
# software's locked materials into the image build context and applies only
# CUA-Lite-owned integration fixups there; the cached materials remain pristine.
#
# Subcommands mirror every other env's install.sh. The canonical form puts the
# verb first, then the software, because CUAWorld has one image per upstream
# software:
#   uv run --no-sync bash .../cuaworld/scripts/install.sh [build|rebuild|provision|pull|push|status] <software>
# A software-first compatibility alias remains accepted:
#   uv run --no-sync bash .../cuaworld/scripts/install.sh <software> [build|rebuild|provision|pull|push|status]
#     build     (default) — provision + build locally if missing/stale (marker-gated)
#     rebuild             — force an image rebuild; materials refresh if stale/incomplete
#     provision           — docker-free prerequisites only (materials + host verifier deps)
#     pull                — adopt the GHCR image iff src_hash matches, then provision
#     push                — publish fresh local images to GHCR (needs write:packages)
#     status              — read-only state check
#
# Materials come from the public HF dataset in data/assets.lock.yaml unless
# LITE_CUAWORLD_MATERIALS_REPO points at a local checkout. The locked public
# revision, or the selected local subtree identity, is folded into the image
# freshness hash.
set -euo pipefail

_MODES=" build rebuild provision pull push status "
usage() {
  echo "Usage: $0 [build|rebuild|provision|pull|push|status] <software>" >&2
  echo "       $0 <software> [build|rebuild|provision|pull|push|status]" >&2
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SANDBOX="$(cd "${ENV_DIR}/../../../sandbox" && pwd)"

[ "$#" -ge 1 ] || { usage; exit 2; }
if [[ "${_MODES}" == *" ${1} "* ]]; then
  MODE="${1}"
  SOFTWARE="${2:-}"
  [ "$#" -eq 2 ] || { usage; exit 2; }
else
  SOFTWARE="${1}"
  MODE="${2:-build}"
  [ "$#" -le 2 ] || { usage; exit 2; }
  [[ "${_MODES}" == *" ${MODE} "* ]] || {
    echo "FATAL: unknown mode '${MODE}' (${_MODES})" >&2
    usage
    exit 2
  }
fi
[ -n "${SOFTWARE}" ] || { usage; exit 2; }

# Source the shared invocation guard before the first Python call.
source "${ENV_DIR}/../../../scripts/image_build.sh"   # src_label / image_is_fresh / require_uv / do_pull / do_push

UPSTREAM_ENV="$(python -m lite.gym.envs.lite.cuaworld.src.image_spec upstream-env "${SOFTWARE}")" || {
  echo "FATAL: unknown lite.cuaworld software '${SOFTWARE}' (not registered in lite/gym/envs/lite/cuaworld/main.py)" >&2
  exit 2
}
REBUILD=0; [ "${MODE}" = "rebuild" ] && REBUILD=1

# Validate names — they interpolate into fs paths, docker tags, HF allow_patterns.
for a in "${SOFTWARE}" "${UPSTREAM_ENV}"; do
  [[ "$a" =~ ^[a-z0-9_]+$ ]] || { echo "FATAL: invalid name '$a' (want ^[a-z0-9_]+$)" >&2; exit 2; }
done

BASE_IMAGE="cua-lite/lite.cuaworld.base:latest"
IMAGE="cua-lite/lite.cuaworld.${SOFTWARE}:latest"
ENV_ID="lite.cuaworld.${SOFTWARE}"
CACHE="${ENV_DIR}/.cache/${SOFTWARE}/${UPSTREAM_ENV}"
MATERIALS_REPO="${LITE_CUAWORLD_MATERIALS_REPO:-}"
IFS=$'\t' read -r MATERIALS_REPO_ID MATERIALS_REF < <(
  python - "$ENV_DIR" <<'PY'
import sys
from pathlib import Path
from lite.gym.utils.config.manifest import load_asset_lock
pin = load_asset_lock(Path(sys.argv[1]))
print(
    pin.repo,
    pin.revision,
    sep="\t",
)
PY
)
MATERIALS_IDENTITY="$(python -m lite.gym.envs.lite.cuaworld.src.image_spec locked-materials-identity "${UPSTREAM_ENV}")"

local_materials_identity() {
  python -m lite.gym.envs.lite.cuaworld.src.image_spec \
    local-materials-identity "${UPSTREAM_ENV}" 2>/dev/null
}

materials_complete_at() {
  local root="$1"
  [ -f "${root}/env.json" ] && \
  [ -d "${root}/scripts" ] && \
  [ -d "${root}/tasks" ] && \
  [ -f "${root}/registered.json" ]
}

materials_complete() {
  materials_complete_at "${CACHE}"
}

materials_missing_at() {
  local root="$1" missing=()
  [ -f "${root}/env.json" ] || missing+=("env.json")
  [ -d "${root}/scripts" ] || missing+=("scripts/")
  [ -d "${root}/tasks" ] || missing+=("tasks/")
  [ -f "${root}/registered.json" ] || missing+=("registered.json")
  printf '%s ' "${missing[@]}"
}

materials_digest_at() {
  python -m lite.gym.envs.lite.cuaworld.src.image_spec cache-materials-digest "$1"
}

validate_materials_at() {
  local root="$1" missing
  if ! materials_complete_at "${root}"; then
    missing="$(materials_missing_at "${root}")"
    echo "FATAL: ${UPSTREAM_ENV} materials in ${root} are incomplete; missing: ${missing}" >&2
    exit 1
  fi
}

# --- materials fetch (only this env's subtree, at the pinned revision) ----------
fetch_materials() {  # $1=upstream_env  $2=dest(.cache/<software>/<upstream_env>)
  local up="$1" dest="$2" allow_pattern source_dir
  source_dir="$(python -m lite.gym.envs.lite.cuaworld.src.image_spec component-path "${up}")"
  allow_pattern="${source_dir}/**"
  mkdir -p "$dest"
  if [ -n "${MATERIALS_REPO}" ]; then
    local src="${MATERIALS_REPO}/${source_dir}"
    [ -d "$src" ] || {
      echo "FATAL: ${source_dir} not in \$LITE_CUAWORLD_MATERIALS_REPO (${src})" >&2
      exit 1
    }
    cp -r "${src}/." "$dest/"
    find "$dest" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
    validate_materials_at "$dest"
    python -m lite.gym.envs.lite.cuaworld.src.image_spec \
      local-materials-identity "${up}" > "${dest}/.materials_revision"
    materials_digest_at "$dest" > "${dest}/.materials_digest"
    : > "${dest}/.complete"
    return
  fi
  require_uv
  uv pip install -q huggingface_hub
  local tmp; tmp="$(mktemp -d)"
  python - \
    "$MATERIALS_REPO_ID" "$allow_pattern" "$source_dir" \
    "$tmp" "$MATERIALS_REF" <<'PYEOF' || {
import sys
from huggingface_hub import snapshot_download
repo, pattern, source_dir, tmp, rev = sys.argv[1:]
snapshot_download(repo_id=repo, repo_type="dataset", revision=rev,
                  allow_patterns=[pattern], local_dir=tmp)
PYEOF
    echo "FATAL: HF fetch of ${up}@${MATERIALS_REF} from ${MATERIALS_REPO_ID} failed (need huggingface_hub + network access)" >&2
    rm -rf "$tmp"; exit 1; }
  [ -d "${tmp}/${source_dir}" ] || { echo "FATAL: ${source_dir} not in materials repo ${MATERIALS_REPO_ID}@${MATERIALS_REF}" >&2; rm -rf "$tmp"; exit 1; }
  cp -r "${tmp}/${source_dir}/." "$dest/"
  rm -rf "$tmp"
  find "$dest" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
  validate_materials_at "$dest"
  # Completion sentinel written LAST, so an interrupted fetch (partial `scripts/`
  # but no `.complete`) is re-fetched next run instead of treated as done.
  printf '%s\n' "${MATERIALS_IDENTITY}" > "${dest}/.materials_revision"
  materials_digest_at "$dest" > "${dest}/.materials_digest"
  : > "${dest}/.complete"
}

ensure_materials() {
  local expected="${MATERIALS_IDENTITY}"
  if [ -n "${MATERIALS_REPO}" ]; then
    expected="$(local_materials_identity)" || {
      echo "FATAL: ${UPSTREAM_ENV} materials subtree is missing or unreadable under \$LITE_CUAWORLD_MATERIALS_REPO (${MATERIALS_REPO})" >&2
      exit 1
    }
  fi
  local cached_revision cached_digest actual_digest
  cached_revision="$(cat "${CACHE}/.materials_revision" 2>/dev/null || true)"
  cached_digest="$(cat "${CACHE}/.materials_digest" 2>/dev/null || true)"
  actual_digest=""
  if [ -f "${CACHE}/.complete" ] && materials_complete; then
    actual_digest="$(materials_digest_at "${CACHE}" 2>/dev/null || true)"
  fi
  if [ ! -f "${CACHE}/.complete" ] || \
     [ "${cached_revision}" != "${expected}" ] || \
     ! materials_complete || \
     [ -z "${cached_digest}" ] || \
     [ -z "${actual_digest}" ] || \
     [ "${cached_digest}" != "${actual_digest}" ]; then
    rm -rf "${CACHE}"
    fetch_materials "${UPSTREAM_ENV}" "${CACHE}"
  fi
}

# Host-side verifier dependencies. cuaworld verifiers run in a HOST process (an
# adapter subprocess), not in the container, so their imports resolve against this
# repo's venv — the same reason lite.osworld's install.sh pip-installs the
# `osworld` package for its host-side evaluators. Each library below IS the parser
# a verifier needs to inspect the artifact the agent produced.
# Measured on the pinned materials: 141 live tasks (registered ∧ non-excluded).
# Counts are tasks affected, same basis.
#
# What upstream does when the library is missing is NOT uniform, so do not rely on
# any one failure mode. Population, ON-DISK basis: 78 host-side .py files import one
# of the 11 libraries below (77 task `verifier.py` + libreoffice_calc/utils/
# calc_verification_utils.py); 56 of them sit in live tasks. Of the 78 —
#   * 30 (14 live) `pip install` the library inside their own `except ImportError`,
#     i.e. they self-heal mid-rollout at the cost of a network round trip. That is
#     exactly why 13 tasks carry `verifier_runtime_pip` in
#     data/validation_excludes.json (gvsig_desktop 7, librecad 4, slicer3d 2).
#   * 46 (41 live) set an availability flag and continue with a degraded check —
#     which for libreoffice_calc IS the scored failure described below.
#   * 2 (1 live — pycharm/fix_excel_sales_report) have NO handler at all: the
#     module-level import makes the verifier itself unloadable.
# The literal "Verifier Error: <lib> not installed on host" message occurs in
# exactly ONE file, librecad/compass_rose_symbol/verifier.py:97 — do not
# generalize it.
#
# Called from BOTH the build path and the pull path — `pull` is how a fresh host
# gets its images and it never enters build(), so hanging this off build() alone
# left every pull-provisioned host scoring those 141 tasks 0.
#
# It PROBES before it installs, which is what lets it sit ABOVE the freshness gate.
# Gating it on staleness instead looks safer but is wrong in the common case: a
# fresh checkout whose images are already fresh would then install nothing and score
# those 141 tasks 0 or near-0. Probing keeps the "already fresh" path
# network-free (the reason for the ordering worry) while still being correct on a
# venv that is missing a library. Deliberately NOT `|| true`: these are required at
# ROLLOUT time, so a loud install failure beats a silent scored-failure later.
install_python_deps() {
  # import name -> pip requirement (they differ for pyshp/pynrrd/biopython/odfpy).
  # openpyxl and odfpy dominate: EVERY libreoffice_calc verifier routes through
  # `libreoffice_calc_env/utils/calc_verification_utils.py`, which soft-fails to
  # OPENPYXL_AVAILABLE/ODF_AVAILABLE=False and returns {"error": "<lib> not
  # available"} — a scored failure, not a crash. pycharm/fix_excel_sales_report is
  # worse: a module-level `import openpyxl` makes the verifier itself unloadable.
  local -a probes=(
      "openpyxl:openpyxl>=3.1"        # libreoffice_calc 87, pycharm 1 (unguarded)
      "odf:odfpy>=1.4"                # libreoffice_calc 87 (.ods side of the same helper)
      "ezdxf:ezdxf>=1.1"              # librecad 24 — the DXF parser IS the check
      "nibabel:nibabel>=5.2"          # slicer3d 16
      "shapefile:pyshp>=2.3"          # gvsig_desktop 7
      "astropy:astropy>=6.0"          # kstars_sim 2
      "statsmodels:statsmodels>=0.14" # gretl 1
      "dbfread:dbfread>=2.0"          # gvsig_desktop 1
      "nrrd:pynrrd>=1.0"              # slicer3d 1
      "Bio:biopython>=1.83"           # ugene 1
      "pdfminer:pdfminer.six"         # openrocket 1 — else it pip-installs mid-rollout
      # Never probed before. They were present on dev hosts only as `quick-start`
      # transitives (scipy/cv2/skimage/trimesh <- sglang[all], lxml <- blobfile,
      # pandas <- datasets), so `uv sync --extra gym` alone had NONE of them and a
      # cuaworld-only host got nothing: cuaworld builds its base from the shared
      # sandbox Dockerfile and never runs lite.osworld's installer.
      "pandas:pandas>=2.0"            # dbeaver 3, hec_ras 6, gretl 2, pycharm 1 — 12 unguarded
      "scipy:scipy>=1.11"             # slicer3d 6, vlc 2, gretl 1, imagej 1 — 4 unguarded
      "cv2:opencv-python-headless>=4.8"  # vlc 7, imagej 1 — 5 unguarded. Keyed on the IMPORT
                                      #   name so an existing opencv-python (which
                                      #   lite.osworld's installer brings) satisfies it
                                      #   instead of installing a second, conflicting build.
      "skimage:scikit-image>=0.22"    # imagej 1, slicer3d 1 (both guarded)
      "lxml:lxml>=5.0"                # diagrams_net 1 — unguarded
      "trimesh:trimesh>=4.0"          # solvespace 1 — base extra only; it otherwise
                                      #   self-heals with a bare `pip install` mid-rollout
      "dateutil:python-dateutil>=2.8" # libreoffice_calc 2 (arrives free with pandas)
      "litellm:litellm>=1.72"         # cuaworld VLM-judged verifiers call vlm.py;
                                      #   declared only in `[quick-start]`, so a `[gym]`
                                      #   venv had none and every judged task scored 0
  )
  # NOT probed on purpose: shapely, rtree, networkx. A census of all 2962 verifiers
  # finds ZERO host-side imports of any of them — they appear only inside in-container
  # setup_task.sh hooks, as literal strings in solvespace's self-heal pip list, and (for
  # networkx) in a verifier that greps the AGENT's script text for "import networkx".
  local -a missing=()
  local entry
  for entry in "${probes[@]}"; do
    python -c "import ${entry%%:*}" >/dev/null 2>&1 || missing+=("${entry#*:}")
  done
  if [ "${#missing[@]}" -eq 0 ]; then
    echo "==> host-side verifier deps already present — nothing to do"
    return 0
  fi
  echo "==> ensure host-side verifier deps (runtime verifier/VLM support): ${missing[*]}"
  require_uv          # shared: image_build.sh (the `uv pip`, never bare-pip rule)
  uv pip install -q "${missing[@]}"
}

provision() {
  echo "==> ensure materials (${UPSTREAM_ENV}@${MATERIALS_REF}) in .cache/${SOFTWARE}/${UPSTREAM_ENV}"
  ensure_materials
  install_python_deps
}

build() {
  provision
  echo "==> ensure ${BASE_IMAGE} (shared sandbox Dockerfile, USER=ga)"
  # Build the shared base with the ga desktop user, stamping the same lite.src_hash
  # freshness label every container env uses — via ``src_label``, like all 14 other
  # build sites. This used to hand-roll ``--label "lite.src_hash=$(... hash ...)"``,
  # which is precisely the shape src_label exists to stop: an empty hash would have
  # stamped a well-formed-looking ``lite.src_hash=`` that the image freshness check reads as
  # STALE forever, on the base plus all 40 software images built on it.
  #
  # ``--build-arg USER=ga`` is the ONLY copy of that value; lite.cuaworld.base's
  # spec hashes THIS script for it (backend.freshness.image_for), so editing the arg
  # re-stales the base instead of silently leaving it FRESH.
  if [ "${REBUILD}" = "1" ] || ! image_is_fresh "${BASE_IMAGE}" lite.cuaworld.base; then
    image_rm "${BASE_IMAGE}"
    DOCKER_BUILDKIT=1 docker build -t "${BASE_IMAGE}" --build-arg USER=ga \
      --label "$(src_label lite.cuaworld.base)" \
      -f "${SANDBOX}/docker/Dockerfile.linux" "${SANDBOX}"
  fi

  # marker gate: fresh image = no-op unless rebuild. Keep this AFTER the base
  # check: a stale base must be rebuilt even when the child label still matches.
  if [ "${REBUILD}" != "1" ] && image_is_fresh "${IMAGE}" "${ENV_ID}"; then
    echo "==> ${IMAGE} already fresh (build sources unchanged); skipping image rebuild. Run 'install.sh rebuild ${SOFTWARE}' to force."; return 0
  fi

  echo "==> assemble build context"
  local ctx; ctx="$(mktemp -d)"; trap 'rm -rf "${ctx}"' RETURN
  cp -r "${CACHE}/scripts" "${ctx}/scripts"
  if [ "${SOFTWARE}" = "openemr" ]; then
    python - "${ctx}/scripts/setup_openemr.sh" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
schema_marker = "MODIFY pc_multiple int(11) NOT NULL DEFAULT 0"
registration_marker = "CUA-Lite: suppress OpenEMR Product Registration modal"
schema_block = """
# CUA-Lite runs OpenEMR natively instead of upstream's openemr/openemr +
# mariadb:10.11 compose pair. Normalize the appointment schema so upstream task
# setup SQL that omits pc_multiple still works under native MariaDB strict mode.
mysql -uopenemr -popenemr openemr <<'SQL'
ALTER TABLE openemr_postcalendar_events
  MODIFY pc_multiple int(11) NOT NULL DEFAULT 0;
SQL
pc_multiple_default="$(mysql -uopenemr -popenemr -N -B openemr -e "SELECT COALESCE(COLUMN_DEFAULT, '<NULL>') FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA='openemr' AND TABLE_NAME='openemr_postcalendar_events' AND COLUMN_NAME='pc_multiple';")"
if [ "$pc_multiple_default" != "0" ]; then
    echo "ERROR: OpenEMR pc_multiple default normalization failed: ${pc_multiple_default}" >&2
    exit 1
fi
"""
registration_block = """
# CUA-Lite: suppress OpenEMR Product Registration modal before first screenshot.
mysql -uopenemr -popenemr openemr <<'SQL' || true
INSERT INTO product_registration (email, opt_out)
SELECT NULL, 1
WHERE NOT EXISTS (SELECT 1 FROM product_registration);
SQL
"""
insertion = ""
if schema_marker not in text:
    insertion += schema_block
if registration_marker not in text:
    insertion += registration_block
if insertion:
    anchor = "chown -R www-data:www-data /var/www/html/openemr/sites\n"
    index = text.rfind(anchor)
    if index < 0:
        raise SystemExit(f"OpenEMR setup normalization anchor missing in {path}")
    index += len(anchor)
    text = text[:index] + insertion + text[index:]
    path.write_text(text)
PY
  fi
  if [ "${SOFTWARE}" = "astroimagej" ]; then
    python - "${ctx}/scripts/setup_astroimagej.sh" "${ctx}/scripts/task_utils.sh" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
utils_path = Path(sys.argv[2])
text = path.read_text()
pref_marker = "CUA-Lite: suppress AstroImageJ updater preferences"
launch_marker = "CUA-Lite: close AstroImageJ updater dialog"
wrapper_marker = "CUA-Lite: wrap /usr/local/bin/aij with updater suppression"
utils_marker = "CUA-Lite: task_utils close AstroImageJ updater"
pref_block = """
    # CUA-Lite: suppress AstroImageJ updater preferences before first screenshot.
    # AstroImageJ's updater reads ij.Prefs keys with the `.aij.update` prefix; the
    # upstream `.astroimagej.prefs.updateCheckOnStartup` key is not sufficient.
    sudo -u $username mkdir -p "$home_dir/.imagej"
    cat >> "$home_dir/.astroimagej/IJ_Prefs.txt" << 'PREFEOF'
.aij.update=false
.aij.update.updateCheck=false
.aij.update.updateCheckOnStartup=false
.aij.updateCheckOnStartup=false
PREFEOF
    cp "$home_dir/.astroimagej/IJ_Prefs.txt" "$home_dir/.imagej/IJ_Prefs.txt"
    chown -R $username:$username "$home_dir/.imagej"
"""
launch_block = """
# CUA-Lite: close AstroImageJ updater dialog before setup returns.
cua_lite_close_aij_updater() {
  local quiet=0
  local poll=0
  local min_polls="${CUA_LITE_AIJ_UPDATER_MIN_POLLS:-20}"
  local max_polls="${CUA_LITE_AIJ_UPDATER_MAX_POLLS:-90}"
  local quiet_polls="${CUA_LITE_AIJ_UPDATER_QUIET_POLLS:-6}"
  for poll in $(seq 1 "$max_polls"); do
    local ids
    ids=$(DISPLAY=${DISPLAY:-:1} xdotool search --name "AstroImageJ Updater" 2>/dev/null || true)
    if [ -n "$ids" ]; then
      quiet=0
      for wid in $ids; do
        DISPLAY=${DISPLAY:-:1} xdotool windowactivate --sync "$wid" 2>/dev/null || true
        DISPLAY=${DISPLAY:-:1} xdotool key Escape 2>/dev/null || true
        DISPLAY=${DISPLAY:-:1} xdotool windowclose "$wid" 2>/dev/null || true
      done
    elif [ "$poll" -ge "$min_polls" ]; then
      quiet=$((quiet + 1))
      [ "$quiet" -ge "$quiet_polls" ] && return 0
    else
      quiet=0
    fi
    sleep 0.5
  done
}
cua_lite_close_aij_updater >/tmp/aij_updater_suppression_${USER}.log 2>&1 || true
"""
wrapper_block = """
    # CUA-Lite: wrap /usr/local/bin/aij with updater suppression for task hooks
    # that launch AstroImageJ directly instead of ~/launch_astroimagej.sh.
    if [ -x /usr/local/bin/aij ] && [ ! -e /usr/local/bin/aij.cua-lite-real ]; then
        AIJ_REAL="$(readlink -f /usr/local/bin/aij)"
        ln -sf "$AIJ_REAL" /usr/local/bin/aij.cua-lite-real
        rm -f /usr/local/bin/aij
        cat > /usr/local/bin/aij <<'AIJWRAP'
#!/usr/bin/env bash
set -u
export DISPLAY=${DISPLAY:-:1}
REAL="/usr/local/bin/aij.cua-lite-real"
if [ ! -x "$REAL" ]; then
  echo "AstroImageJ real launcher missing: $REAL" >&2
  exit 1
fi
"$REAL" "$@" &
aij_pid=$!
cua_lite_close_aij_updater() {
  local quiet=0
  local poll=0
  local min_polls="${CUA_LITE_AIJ_UPDATER_MIN_POLLS:-20}"
  local max_polls="${CUA_LITE_AIJ_UPDATER_MAX_POLLS:-90}"
  local quiet_polls="${CUA_LITE_AIJ_UPDATER_QUIET_POLLS:-6}"
  for poll in $(seq 1 "$max_polls"); do
    local ids
    ids=$(DISPLAY=${DISPLAY:-:1} xdotool search --name "AstroImageJ Updater" 2>/dev/null || true)
    if [ -n "$ids" ]; then
      quiet=0
      for wid in $ids; do
        DISPLAY=${DISPLAY:-:1} xdotool windowactivate --sync "$wid" 2>/dev/null || true
        DISPLAY=${DISPLAY:-:1} xdotool key Escape 2>/dev/null || true
        DISPLAY=${DISPLAY:-:1} xdotool windowclose "$wid" 2>/dev/null || true
      done
    elif [ "$poll" -ge "$min_polls" ]; then
      quiet=$((quiet + 1))
      [ "$quiet" -ge "$quiet_polls" ] && break
    else
      quiet=0
    fi
    sleep 0.5
  done
}
cua_lite_close_aij_updater >/tmp/aij_updater_suppression_${USER:-ga}.log 2>&1 || true
echo "AstroImageJ started (PID: $aij_pid)"
exit 0
AIJWRAP
        chmod +x /usr/local/bin/aij
    fi
"""
if pref_marker not in text:
    anchor = '    chown $username:$username "$home_dir/.astroimagej/IJ_Prefs.txt"\n'
    index = text.find(anchor)
    if index < 0:
        raise SystemExit(f"AstroImageJ prefs anchor missing in {path}")
    text = text[:index] + pref_block + text[index:]
if launch_marker not in text:
    anchor = '"$AIJ_PATH" "$@" > /tmp/astroimagej_$USER.log 2>&1 &\n'
    index = text.find(anchor)
    if index < 0:
        raise SystemExit(f"AstroImageJ launch anchor missing in {path}")
    index += len(anchor)
    text = text[:index] + launch_block + text[index:]
if wrapper_marker not in text:
    anchor = '    echo "  - Created launch script"\n'
    index = text.find(anchor)
    if index < 0:
        raise SystemExit(f"AstroImageJ wrapper anchor missing in {path}")
    index += len(anchor)
    text = text[:index] + wrapper_block + text[index:]
path.write_text(text)

utils = utils_path.read_text()
utils_block = """
# CUA-Lite: task_utils close AstroImageJ updater before first screenshot.
cua_lite_close_aij_updater() {
    local quiet=0
    local poll=0
    local min_polls="${CUA_LITE_AIJ_UPDATER_MIN_POLLS:-20}"
    local max_polls="${CUA_LITE_AIJ_UPDATER_MAX_POLLS:-90}"
    local quiet_polls="${CUA_LITE_AIJ_UPDATER_QUIET_POLLS:-6}"
    for poll in $(seq 1 "$max_polls"); do
        local ids
        ids=$(DISPLAY=:1 sudo -u ga xdotool search --name "AstroImageJ Updater" 2>/dev/null || true)
        if [ -n "$ids" ]; then
            quiet=0
            for wid in $ids; do
                DISPLAY=:1 sudo -u ga xdotool windowactivate --sync "$wid" 2>/dev/null || true
                DISPLAY=:1 sudo -u ga xdotool key Escape 2>/dev/null || true
                DISPLAY=:1 sudo -u ga xdotool windowclose "$wid" 2>/dev/null || true
            done
        elif [ "$poll" -ge "$min_polls" ]; then
            quiet=$((quiet + 1))
            [ "$quiet" -ge "$quiet_polls" ] && return 0
        else
            quiet=0
        fi
        sleep 0.5
    done
}

"""
if utils_marker not in utils:
    anchor = "# Launch AstroImageJ and wait for it to start\n"
    index = utils.find(anchor)
    if index < 0:
        raise SystemExit(f"AstroImageJ task_utils anchor missing in {utils_path}")
    utils = utils[:index] + utils_block + utils[index:]
utils = utils.replace(
    "        focus_aij_window\n        return 0\n",
    "        focus_aij_window\n        cua_lite_close_aij_updater\n        return 0\n",
)
utils = utils.replace(
    "    focus_aij_window\n\n    # Check if actually running\n",
    "    focus_aij_window\n    cua_lite_close_aij_updater\n\n    # Check if actually running\n",
)
utils_path.write_text(utils)
PY
  fi
  if [ -d "${CACHE}/data" ]; then cp -r "${CACHE}/data" "${ctx}/data"; else mkdir "${ctx}/data"; fi
  # Stage every env-root payload dir the hooks read from /workspace. Enumerating these by hand is
  # exactly how we lost them before: `config/` was absent from the import spec AND from this list, so
  # kstars' setup (`set -e`, unguarded `cp /workspace/config/kstarsrc`) aborted and baked a broken
  # image; `assets/` was likewise never staged, so ugene's 19 asset-copying task setups produced
  # tasks with no input files. Keep this driven by what upstream actually ships, not by a guess.
  #   config/ — curated app prefs (kstars kstarsrc, gpredict .qth, vscode settings.json, ...)
  #   assets/ — task input files (ugene genomes, ...)
  # `tasks/` is deliberately NOT staged: it is host-side (setup/verify read it from .cache).
  for payload in config assets; do
    if [ -d "${CACHE}/${payload}" ]; then cp -r "${CACHE}/${payload}" "${ctx}/${payload}"
    else mkdir "${ctx}/${payload}"; fi
  done
  cp "${CACHE}/env.json" "${ctx}/env.json"
  [ -f "${CACHE}/post_build.sh" ] && cp "${CACHE}/post_build.sh" "${ctx}/post_build.sh" || true
  cp "${ENV_DIR}/docker/run_hooks.sh" "${ctx}/run_hooks.sh"

  echo "==> docker build ${IMAGE}"
  # Image freshness hashes assets.lock.yaml directly (plus local mode when set).
  local img_label; img_label="$(src_label "${ENV_ID}")"
  image_rm "${IMAGE}"
  docker build -t "${IMAGE}" \
    -f "${ENV_DIR}/docker/Dockerfile" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg SOFTWARE="${SOFTWARE}" \
    --label "${img_label}" \
    "${ctx}"
  local img_hash="${img_label#lite.src_hash=}"
  echo "Built ${IMAGE} (materials ${UPSTREAM_ENV}@${MATERIALS_REF}, src_hash ${img_hash:0:12})"
}

materials_status() {
  local expected="${MATERIALS_IDENTITY}"
  if [ -n "${MATERIALS_REPO}" ]; then
    expected="$(local_materials_identity)" || {
      echo "materials    : ERROR (${UPSTREAM_ENV}; missing local subtree under \$LITE_CUAWORLD_MATERIALS_REPO=${MATERIALS_REPO})"
      return 1
    }
  fi
  if [ ! -d "${CACHE}" ]; then
    echo "materials    : MISSING (${UPSTREAM_ENV})"
    return 1
  fi
  if [ ! -f "${CACHE}/.complete" ]; then
    echo "materials    : INCOMPLETE (${UPSTREAM_ENV})"
    return 1
  fi
  if [ -z "${expected}" ] || \
     [ "$(cat "${CACHE}/.materials_revision" 2>/dev/null || true)" != "${expected}" ]; then
    echo "materials    : STALE (${UPSTREAM_ENV})"
    return 1
  fi
  if ! materials_complete; then
    echo "materials    : INCOMPLETE (${UPSTREAM_ENV}; missing env.json/scripts/tasks/registered.json)"
    return 1
  fi
  local cached_digest actual_digest
  cached_digest="$(cat "${CACHE}/.materials_digest" 2>/dev/null || true)"
  actual_digest="$(materials_digest_at "${CACHE}" 2>/dev/null || true)"
  if [ -z "${cached_digest}" ] || [ -z "${actual_digest}" ] || \
     [ "${cached_digest}" != "${actual_digest}" ]; then
    echo "materials    : STALE (${UPSTREAM_ENV}; cache digest mismatch)"
    return 1
  fi
  echo "materials    : FRESH (${UPSTREAM_ENV})"
}

image_status() {
  local label="$1" image="$2" env_id="$3"
  image_status_line "${label}" "${image}" "${env_id}"
}

status() {
  materials_status || true
  echo "source base  : $(image_is_fresh "${BASE_IMAGE}" lite.cuaworld.base && echo FRESH || { docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1 && echo STALE || echo MISSING; }) (${BASE_IMAGE}; only needed for local source builds)"
  image_status "image        " "${IMAGE}" "${ENV_ID}" || true
}

case "${MODE}" in
  build|rebuild) build ;;
  provision) provision ;;
  pull)
    [ -z "${MATERIALS_REPO}" ] || {
      echo "FATAL: pull is disabled with LITE_CUAWORLD_MATERIALS_REPO; build local materials instead" >&2
      exit 2
    }
    do_pull "${IMAGE}" "${ENV_ID}"
    provision
    ;;
  push)
    [ -z "${MATERIALS_REPO}" ] || {
      echo "FATAL: refusing to publish an image built from local materials" >&2
      exit 2
    }
    image_is_fresh "${IMAGE}" "${ENV_ID}" || {
      echo "FATAL: refusing to publish missing/stale ${IMAGE}; run 'install.sh build ${SOFTWARE}' first" >&2
      exit 1
    }
    image_is_fresh "${BASE_IMAGE}" lite.cuaworld.base || {
      echo "FATAL: refusing to publish missing/stale ${BASE_IMAGE}; run 'install.sh build ${SOFTWARE}' first" >&2
      exit 1
    }
    do_push "${BASE_IMAGE}" lite.cuaworld.base
    do_push "${IMAGE}" "${ENV_ID}"
    ;;
  status)        status ;;
esac
