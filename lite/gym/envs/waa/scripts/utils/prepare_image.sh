#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 /path/to/windows.iso [output.qcow2]" >&2
  exit 2
fi

[[ -f "$1" ]] || { echo "ISO not found: $1" >&2; exit 2; }
ISO="$(realpath "$1")"
OUTPUT="${2:-$HOME/.cache/cua-lite/waa/images/windows11-waa.qcow2}"
OUTPUT="$(realpath -m "$OUTPUT")"
OUTPUT_TMP="${OUTPUT}.tmp"
OUTPUT_KEY="$(printf '%s' "$OUTPUT" | sha256sum | cut -c1-16)"
WORK_DIR="${WAA_IMAGE_WORK_DIR:-$HOME/.cache/cua-lite/waa/image-build/$OUTPUT_KEY}"
UPSTREAM_WAA_IMAGE="${WAA_PREP_BASE_IMAGE:-windowsarena/winarena@sha256:869afffe384f0e0356dbfcdf78486611e31d62e692544cbfd9a6c08606f07440}"
PATCHED_WAA_IMAGE="${WAA_PATCHED_PREP_IMAGE:-cua-lite/waa-prep:latest}"
RUNNER_IMAGE="${WAA_RUNNER_IMAGE:-cua-lite/waa:latest}"
# noVNC (container port 8006) is only a human-watch convenience during the long
# prep. Hardcoding a host port fails the whole install on a shared host where it
# is taken (as 8006 routinely is). Host-side probing is also unreliable under
# rootless Docker — published ports bind inside the rootlesskit network namespace,
# invisible to host `ss` — so publish to an ephemeral host port by default
# (collision-proof, same as the runtime container). A caller who needs a fixed
# port can still set WAA_PREP_NOVNC_PORT.
if [[ -n "${WAA_PREP_NOVNC_PORT:-}" ]]; then
  NOVNC_PUBLISH="127.0.0.1:${WAA_PREP_NOVNC_PORT}:8006"
else
  NOVNC_PUBLISH="127.0.0.1:0:8006"
fi
PREP_CONTAINER="waa-prep-${OUTPUT_KEY}"
# Per-boot wall-clock cap so a wedged guest fails loudly instead of hanging forever
# (Windows install can be slow on a loaded shared host, so default generously).
BOOT_TIMEOUT_S="${WAA_PREP_BOOT_TIMEOUT_S:-7200}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

[[ -c /dev/kvm ]] || {
  echo "Image preparation requires /dev/kvm" >&2
  exit 2
}
command -v docker >/dev/null || { echo "Image preparation requires Docker" >&2; exit 2; }
command -v flock >/dev/null || { echo "Image preparation requires flock" >&2; exit 2; }
mkdir -p "$WORK_DIR" "$(dirname "$OUTPUT")"
exec 9>"$WORK_DIR/build.lock"
flock -n 9 || {
  echo "Another WAA image build is using $WORK_DIR" >&2
  exit 75
}
trap 'rm -f "$OUTPUT_TMP"' EXIT
docker image inspect "$RUNNER_IMAGE" >/dev/null 2>&1 || {
  echo "Runner image not found: $RUNNER_IMAGE" >&2
  echo "Build it with scripts/install.sh first." >&2
  exit 2
}
if [[ -n "${WAA_PREP_IMAGE:-}" ]]; then
  WAA_IMAGE="$WAA_PREP_IMAGE"
else
  docker build \
    --platform linux/amd64 \
    --build-arg "BASE_IMAGE=$UPSTREAM_WAA_IMAGE" \
    -t "$PATCHED_WAA_IMAGE" \
    "$ENV_DIR/docker/prep"
  WAA_IMAGE="$PATCHED_WAA_IMAGE"
fi
mkdir -p "$WORK_DIR/storage"
rm -f \
  "$WORK_DIR/storage/data.img" \
  "$WORK_DIR/storage/data.qcow2" \
  "$WORK_DIR/storage/windows.boot" \
  "$WORK_DIR/storage/windows.base" \
  "$WORK_DIR/storage/windows.mac" \
  "$WORK_DIR/storage/windows.mode" \
  "$WORK_DIR/storage/windows.ver" \
  "$WORK_DIR/storage/windows_secure.rom" \
  "$WORK_DIR/storage/windows_secure.tpm" \
  "$WORK_DIR/storage/windows_secure.vars" \
  "$WORK_DIR/storage"/windows.*.iso
rm -f "$OUTPUT_TMP"

docker rm -fv "$PREP_CONTAINER" >/dev/null 2>&1 || true
echo "noVNC: run 'docker port $PREP_CONTAINER 8006' to get the host port and watch the guest." >&2
prepare_status=75
for boot_attempt in {1..6}; do
  echo "Starting WAA image preparation boot ${boot_attempt}/6..."
  set +e
  # Heartbeat: the guest install phase is otherwise silent for many minutes, so
  # emit a progress line every 2 minutes until this boot's docker run returns.
  ( m=0; while sleep 120; do m=$((m + 2)); \
      echo "[prep] boot ${boot_attempt}/6 still running (~${m}m into this boot; timeout ${BOOT_TIMEOUT_S}s)" >&2; \
    done ) &
  heartbeat_pid=$!
  timeout --signal=TERM "$BOOT_TIMEOUT_S" \
  docker run --rm \
    --name "$PREP_CONTAINER" \
    --platform linux/amd64 \
    --device=/dev/kvm \
    --device=/dev/net/tun \
    --cap-add NET_ADMIN \
    --sysctl net.ipv4.ip_forward=1 \
    --stop-timeout 120 \
    --publish "$NOVNC_PUBLISH" \
    -v "$ISO:/custom.iso:ro" \
    -v "$WORK_DIR/storage:/storage" \
    --entrypoint /bin/bash \
    "$WAA_IMAGE" \
    -lc '
      set -u
      /entry.sh --prepare-image true --start-client false &
      entry_pid=$!
      qemu_seen=0
      while kill -0 "$entry_pid" 2>/dev/null; do
        if pgrep -f "^qemu-system-x86_64 " >/dev/null; then
          qemu_seen=1
        elif (( qemu_seen )); then
          # Dockur writes windows.boot immediately after QEMU exits. Give its
          # power-management child time to persist that clean-shutdown marker
          # before stopping the upstream entrypoint during its 180-second sleep.
          for _ in {1..30}; do
            [[ -e /storage/windows.boot ]] && break
            sleep 1
          done
          kill "$entry_pid" 2>/dev/null || true
          wait "$entry_pid" 2>/dev/null || true
          if [[ -e /storage/windows.boot ]]; then
            exit 0
          fi
          # Windows Setup performs full shutdowns between installation phases.
          # Ask the host wrapper to boot the same disk again.
          exit 75
        fi
        sleep 5
      done
      wait "$entry_pid"
    '
  prepare_status=$?
  kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
  set -e

  if (( prepare_status == 0 )) && [[ -e "$WORK_DIR/storage/windows.boot" ]]; then
    break
  fi
  if (( prepare_status == 124 )); then
    echo "WAA image preparation timed out after ${BOOT_TIMEOUT_S}s on boot ${boot_attempt}/6." >&2
    echo "The Windows guest did not seal/shut down in time. On a heavily loaded host," \
         "raise WAA_PREP_BOOT_TIMEOUT_S; otherwise inspect the guest via noVNC." >&2
    exit 124
  fi
  # status 75 = reboot-and-continue; status 0 without windows.boot = booted but
  # not sealed yet, also continue. Any other status is a genuine failure. (Never
  # `exit 0` here: that would mask a failed prep as success.)
  if (( prepare_status != 0 && prepare_status != 75 )); then
    echo "WAA image preparation failed with status $prepare_status" >&2
    exit "$prepare_status"
  fi
done

if (( prepare_status != 0 )) || [[ ! -e "$WORK_DIR/storage/windows.boot" ]]; then
  echo "WAA image preparation did not finish after 6 Windows boots." >&2
  exit 1
fi

[[ -s "$WORK_DIR/storage/data.img" ]] || {
  echo "WAA image preparation did not produce $WORK_DIR/storage/data.img" >&2
  exit 1
}
[[ -e "$WORK_DIR/storage/windows.boot" ]] || {
  echo "WAA image preparation did not finish a clean Windows shutdown." >&2
  exit 1
}

docker run --rm --entrypoint qemu-img \
  -v "$WORK_DIR/storage:/input:ro" \
  -v "$(dirname "$OUTPUT"):/output" \
  "$RUNNER_IMAGE" \
  convert -p -f raw -O qcow2 /input/data.img "/output/$(basename "$OUTPUT_TMP")"

mv -f "$OUTPUT_TMP" "$OUTPUT"
if [[ "${WAA_KEEP_RAW:-0}" != "1" ]]; then
  rm -f \
    "$WORK_DIR/storage/data.img" \
    "$WORK_DIR/storage/windows.boot" \
    "$WORK_DIR/storage/windows.base" \
    "$WORK_DIR/storage/windows.mac" \
    "$WORK_DIR/storage/windows.mode" \
    "$WORK_DIR/storage/windows.ver" \
    "$WORK_DIR/storage/windows_secure.rom" \
    "$WORK_DIR/storage/windows_secure.tpm" \
    "$WORK_DIR/storage/windows_secure.vars" \
    "$WORK_DIR/storage"/windows.*.iso
fi

echo "Prepared WindowsAgentArena base disk: $OUTPUT"
