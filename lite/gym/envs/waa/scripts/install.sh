#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ENV_DIR/../../../.." && pwd)"
source "$ENV_DIR/../../scripts/image_build.sh"

DEFAULT_PREP_BASE_IMAGE="windowsarena/winarena@sha256:869afffe384f0e0356dbfcdf78486611e31d62e692544cbfd9a6c08606f07440"

usage() {
  cat <<EOF
Usage: $0 [build|rebuild|provision|pull|push|status]

Prepares the local WindowsAgentArena resources: runner image, pinned task assets,
official Windows 11 evaluation ISO, prepared qcow2, and ready snapshot.

Modes:
  build     (default) Install the fastest way: prefer the published, source-
            matched runner + prepared qcow2 from GHCR — skipping the ~1h Windows
            install — and fall back to a local build/prep when they are
            unavailable. Task assets and the ready snapshot are always local.
  pull      Alias for the default fast path (pull-first, build-fallback).
  rebuild   Force the runner and prepared qcow2 from source (ignore published).
  provision Docker-free-of-a-runner-rebuild prep only: task assets + prepared
            qcow2 (from-scratch, or a host copy) + ready snapshot. Assumes the
            runner image already exists (qemu-img validation runs it) and never
            builds/pulls the SHARED cua-lite/waa:latest — so a fresh git worktree
            can prep its host-shared qcow2/asset cache without rebuilding a runner
            a co-tenant may be using.
  push      Publish the runner + prepared qcow2 to GHCR (maintainers only).
  status    Report runner / ISO / qcow2 / provenance / task-asset status.

Paths and resources come from configs/default.yaml (override the whole file with
WAA_CONFIG=/path). The Windows ISO is Microsoft's freely downloadable evaluation
image, used under Microsoft's evaluation license. Verify separately with
utils/smoke_test.py or utils/verify_all_tasks.py.
EOF
}

MODE="${1:-build}"
case "$MODE" in
  build|rebuild|provision|pull|push|status) ;;
  -h|--help) usage; exit 0 ;;
  *) echo "Unknown mode: $MODE" >&2; usage >&2; exit 2 ;;
esac

command -v docker >/dev/null || {
  echo "WindowsAgentArena installation requires Docker." >&2
  exit 2
}
command -v python >/dev/null || {
  echo "Run this script through: uv run --no-sync bash $0" >&2
  exit 2
}

IMAGE="$(python "$SCRIPT_DIR/utils/config_value.py" env_kwargs runner_image)"
DISK_IMAGE="${IMAGE%:*}-disk:${IMAGE##*:}"   # cua-lite/waa:latest -> cua-lite/waa-disk:latest
OUTPUT="$(python "$SCRIPT_DIR/utils/config_value.py" env_kwargs base_disk)"
ASSETS_DIR="$(python "$SCRIPT_DIR/utils/config_value.py" env_kwargs assets_dir)"
ISO="$HOME/.cache/cua-lite/waa/iso/windows11-enterprise-eval-en-us-x64.iso"
OUTPUT="$(realpath -m "${OUTPUT/#\~/$HOME}")"
ASSETS_DIR="$(realpath -m "${ASSETS_DIR/#\~/$HOME}")"
ISO="$(realpath -m "${ISO/#\~/$HOME}")"
PROVENANCE_PREP_IMAGE="${WAA_PREP_IMAGE:-${WAA_PREP_BASE_IMAGE:-$DEFAULT_PREP_BASE_IMAGE}}"
provenance_args=(--prep-image "$PROVENANCE_PREP_IMAGE")

build_runner() {
  if [[ "$MODE" == rebuild ]]; then
    image_rm "$IMAGE"
  elif image_is_fresh "$IMAGE" waa; then
    echo "[install] $IMAGE is up to date" >&2
    return
  fi
  docker build \
    -f "$ENV_DIR/docker/Dockerfile" \
    -t "$IMAGE" \
    --label "$(src_label waa)" \
    "$ENV_DIR"
}

status() {
  if image_is_fresh "$IMAGE" waa; then
    echo "$IMAGE: CURRENT"
  elif docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "$IMAGE: STALE"
  else
    echo "$IMAGE: MISSING"
  fi
  [[ -s "$ISO" ]] && echo "ISO: PRESENT ($ISO)" || echo "ISO: MISSING ($ISO)"
  [[ -s "$OUTPUT" ]] \
    && echo "qcow2: PRESENT ($OUTPUT)" || echo "qcow2: MISSING ($OUTPUT)"
  if [[ -s "$ISO" ]] && python "$SCRIPT_DIR/utils/image_provenance.py" verify \
      --image "$OUTPUT" --iso "$ISO" "${provenance_args[@]}"; then
    echo "qcow2 provenance: CURRENT"
  else
    echo "qcow2 provenance: MISSING/STALE"
  fi
  if python "$SCRIPT_DIR/utils/assets.py" verify --cache-dir "$ASSETS_DIR" >/dev/null 2>&1; then
    echo "task assets: VERIFIED ($ASSETS_DIR)"
  else
    echo "task assets: MISSING/STALE ($ASSETS_DIR)"
  fi
}

# Publish the prepared qcow2 as cua-lite/waa-disk (data-only image) so others can
# `pull` it instead of the ~1h local ISO install. The snapshot is NOT shipped — it
# is host/CPU-specific and cheap, rebuilt locally after the disk lands.
publish_disk() {
  local sidecar="$OUTPUT.provenance.json"
  [[ -s "$OUTPUT" && -s "$sidecar" ]] || {
    echo "No prepared qcow2 + provenance at $OUTPUT to publish; run install first." >&2
    return 1
  }
  if [[ "$(python "$SCRIPT_DIR/utils/disk_id.py")" \
        != "$(python "$SCRIPT_DIR/utils/disk_id.py" --from-sidecar "$sidecar")" ]]; then
    echo "Local qcow2 is stale vs this checkout (disk_id mismatch); rebuild before publishing." >&2
    return 1
  fi
  local did ctx; did="$(python "$SCRIPT_DIR/utils/disk_id.py")"
  ctx="$(mktemp -d "$(dirname "$OUTPUT")/.waa-disk-ctx.XXXXXX")"
  trap 'rm -rf "$ctx"' RETURN
  ln "$OUTPUT" "$ctx/windows11-waa.qcow2"   # hardlink (same fs) — no 16 GiB copy
  DOCKER_BUILDKIT=0 docker build -f "$ENV_DIR/docker/Dockerfile.disk" \
    --build-arg "WAA_DISK_ID=$did" -t "$DISK_IMAGE" "$ctx"
  do_push_unchecked "$DISK_IMAGE"
}

# Adopt a published cua-lite/waa-disk matching THIS checkout: pull it, extract the
# qcow2, stamp provenance. Returns 0 (disk in place) or 1 (caller builds locally).
adopt_pulled_disk() {
  local remote want have cid
  remote="$(_ghcr_ref "$DISK_IMAGE")"
  want="$(python "$SCRIPT_DIR/utils/disk_id.py")"
  # Idempotent: a local qcow2 already matching this checkout needs no re-pull.
  if [[ -s "$OUTPUT" && -s "$OUTPUT.provenance.json" \
        && "$(python "$SCRIPT_DIR/utils/disk_id.py" --from-sidecar "$OUTPUT.provenance.json")" == "$want" ]]; then
    echo "[install] prepared qcow2 already current (disk_id ${want:0:12})." >&2
    return 0
  fi
  have="$(docker buildx imagetools inspect "$remote" \
            --format '{{ index .Image.Config.Labels "lite.waa_disk_id" }}' 2>/dev/null || true)"
  if [[ -n "$have" && "$have" != "$want" ]]; then
    echo "[install] published disk (${have:0:12}) != your checkout (${want:0:12}); building locally." >&2
    return 1
  fi
  docker pull "$remote" >/dev/null 2>&1 || {
    echo "[install] no matching published disk (or need docker login ghcr.io); building locally." >&2
    return 1
  }
  have="$(docker image inspect --format '{{index .Config.Labels "lite.waa_disk_id"}}' "$remote" 2>/dev/null)"
  [[ "$have" == "$want" ]] || { image_rm "$remote"; return 1; }
  mkdir -p "$(dirname "$OUTPUT")"
  # `docker create` requires a recorded command; a FROM-scratch image has none,
  # so pass a placeholder. It is never executed — we only `docker cp` the disk out.
  cid="$(docker create "$remote" /noop)"
  docker cp "$cid:/waa/windows11-waa.qcow2" "$OUTPUT"
  docker rm -v "$cid" >/dev/null
  # The ~16 GB disk image is a delivery vehicle only — the runtime uses the
  # extracted qcow2, so free it (a re-run re-pulls, but the guard above skips that).
  image_rm "$remote"
  python "$SCRIPT_DIR/utils/disk_id.py" --write-sidecar "$OUTPUT" \
    --prep-image "$PROVENANCE_PREP_IMAGE" >/dev/null
  echo "[install] adopted published qcow2 from $remote (disk_id ${want:0:12}) — skipped ~1h install." >&2
}

# Set by build() when a published qcow2 was adopted (skips the ~1h local prep in
# provision). Defaults to 0 for the standalone `provision` verb (from-scratch).
DISK_ADOPTED=0

# The prep that does NOT rebuild/pull the SHARED runner (cua-lite/waa:latest):
# pinned task assets + the prepared Windows qcow2 (from-scratch ISO install unless
# DISK_ADOPTED, else validated in place) + the ready snapshot. The qemu-img checks
# `docker run` the EXISTING runner (validation, not a rebuild); acquiring the
# runner itself is the image step, kept out here (see build()). Split out so a
# fresh git worktree can prep its host-shared qcow2/asset cache without rebuilding
# a runner a co-tenant may be using. build/rebuild/pull call this after runner
# acquisition; the standalone `provision` verb runs just this.
provision() {
  python "$SCRIPT_DIR/utils/assets.py" install \
    --cache-dir "$ASSETS_DIR" \
    --workers 8

  # When a published disk was adopted, the qcow2 + provenance are already in place;
  # skip the Windows ISO download and the ~1h local prep entirely.
  if (( ! DISK_ADOPTED )); then
    ISO="$("$SCRIPT_DIR/utils/download_windows_iso.sh" "$ISO" | tail -n 1)"

    needs_image=0
    if [[ "$MODE" == rebuild ]]; then
      needs_image=1
    elif python "$SCRIPT_DIR/utils/image_provenance.py" verify \
        --image "$OUTPUT" --iso "$ISO" "${provenance_args[@]}"; then
      echo "[install] prepared WAA disk is current: $OUTPUT" >&2
    else
      needs_image=1
    fi

    if (( needs_image )); then
      mkdir -p "$(dirname "$OUTPUT")"
      available_kb="$(df -Pk "$(dirname "$OUTPUT")" | awk 'NR == 2 {print $4}')"
      required_kb=$((80 * 1024 * 1024))
      if (( available_kb < required_kb )); then
        echo "Preparing the WAA disk requires at least 80 GiB free disk space." >&2
        exit 2
      fi
      "$SCRIPT_DIR/utils/prepare_image.sh" "$ISO" "$OUTPUT"
      # Validate the freshly prepared disk BEFORE stamping provenance. Otherwise a
      # corrupt qcow2 could be left behind with a valid-looking sidecar, which a later
      # `verify` would trust and skip the rebuild, booting a bad disk at runtime.
      if ! docker run --rm \
          --entrypoint qemu-img \
          --mount "type=bind,src=$OUTPUT,dst=/image.qcow2,readonly" \
          "$IMAGE" \
          check /image.qcow2; then
        rm -f "$OUTPUT" "$OUTPUT.provenance.json"
        echo "[install] prepared WAA disk failed qemu-img check; removed it. Re-run install." >&2
        exit 1
      fi
      python "$SCRIPT_DIR/utils/image_provenance.py" write \
        --image "$OUTPUT" --iso "$ISO" "${provenance_args[@]}"
    fi
  fi

  docker run --rm \
    --entrypoint qemu-img \
    --mount "type=bind,src=$OUTPUT,dst=/image.qcow2,readonly" \
    "$IMAGE" \
    check /image.qcow2

  # Build the "ready" snapshot so runtime VMs restore in ~15-50s instead of cold-booting
  # Windows (~60-90s). Used automatically when present; non-fatal if it fails.
  (
    cd "$REPO_ROOT"
    python lite/gym/envs/waa/scripts/utils/prepare_snapshot.py \
      --base-disk "$OUTPUT" --assets-dir "$ASSETS_DIR"
  ) || echo "[install] snapshot build failed; runtime will cold-boot (rerun prepare_snapshot.py later)" >&2
}

# Full install (default / pull / rebuild): acquire the runner image (the SHARED
# cua-lite/waa:latest — pull-first, build-fallback; rebuild forces from source),
# adopt a published qcow2 when possible, then provision the rest.
build() {
  # Fast path by default: prefer the published, source-matched runner + prepared
  # qcow2 (skips the ~1h Windows install), falling back to a local build/prep when
  # they are not published / not reachable / don't match this checkout. `rebuild`
  # forces everything from source.
  if [[ "$MODE" == rebuild ]]; then
    build_runner
  else
    do_pull "$IMAGE" waa || build_runner
    adopt_pulled_disk && DISK_ADOPTED=1 || true
  fi

  provision

  echo "WindowsAgentArena installation complete."
  echo "Base disk: $OUTPUT"
  echo "Task assets: $ASSETS_DIR"
  echo "Verify (optional): utils/smoke_test.py (one task) or utils/verify_all_tasks.py (every task)."
}

if [[ "$MODE" == status ]]; then
  status
  exit 0
fi
if [[ "$MODE" == push ]]; then
  if ! image_is_fresh "$IMAGE" waa; then
    echo "Refusing to publish missing/stale runner: $IMAGE" >&2
    echo "Run $0 build first." >&2
    exit 1
  fi
  do_push "$IMAGE" waa
  publish_disk || echo "[install] runner published; qcow2 not published (see above)." >&2
  exit 0
fi

[[ -c /dev/kvm ]] || {
  echo "WindowsAgentArena requires /dev/kvm." >&2
  exit 2
}

# build|rebuild|pull all go through build() (runner acquisition + provision);
# provision runs the docker-free-of-a-runner-rebuild prep only (assumes the
# runner already exists — never builds/pulls the SHARED cua-lite/waa:latest).
case "$MODE" in
  provision) provision ;;
  *)         build ;;   # build|rebuild|pull
esac
