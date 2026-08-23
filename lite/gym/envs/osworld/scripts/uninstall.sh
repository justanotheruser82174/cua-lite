#!/bin/bash
# Reverse install.sh. Removes the provisioned qcow2 and (by default) the derived image. There
# is NO host `desktop_env` to worry about — it lives only inside the image (eval-in-container),
# so uninstalling osworld can't affect osworld_2 or lite.osworld.
#
# Usage:
#   uv run --no-sync bash lite/gym/envs/osworld/scripts/uninstall.sh          # qcow2 + image
#   uv run --no-sync bash lite/gym/envs/osworld/scripts/uninstall.sh keep-image   # qcow2 only

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
QCOW2="$ENV_DIR/.cache/Ubuntu.qcow2"
IMAGE="cua-lite/osworld:latest"

if [ -e "$QCOW2" ]; then rm -f "$QCOW2" && echo "[uninstall] removed $QCOW2" >&2; else echo "[uninstall] qcow2 already absent." >&2; fi
if [ "${1:-}" != "keep-image" ]; then
    docker rmi "$IMAGE" >/dev/null 2>&1 && echo "[uninstall] removed image $IMAGE" >&2 || echo "[uninstall] image $IMAGE absent/in-use." >&2
fi
echo "[uninstall] done ✓ (no host desktop_env involved — osworld_2 + lite.osworld unaffected)" >&2
