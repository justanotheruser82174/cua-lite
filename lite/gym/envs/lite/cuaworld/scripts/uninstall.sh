#!/usr/bin/env bash
# One-time teardown for a gym-anything software image.
# Removes cua-lite/lite.cuaworld.<software>:latest and its fetched materials cache. Idempotent —
# exits 0 when there is nothing to remove. The shared cua-lite/lite.cuaworld.base base image
# (built from the sandbox Dockerfile with USER=ga) is left alone (other
# lite.cuaworld.<software> images depend on it).
#
# Usage:
#   uv run --no-sync bash lite/gym/envs/lite/cuaworld/scripts/uninstall.sh <software>
set -euo pipefail

SOFTWARE="${1:?usage: uninstall.sh <software>}"
[ "$#" -eq 1 ] || { echo "FATAL: usage: uninstall.sh <software>" >&2; exit 2; }
# Validate before it lands in an image tag + an `rm -rf` path (symmetry with install.sh).
[[ "${SOFTWARE}" =~ ^[a-z0-9_]+$ ]] || { echo "FATAL: invalid software name '${SOFTWARE}'" >&2; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE="cua-lite/lite.cuaworld.${SOFTWARE}:latest"

if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  docker rmi "${IMAGE}" >/dev/null 2>&1 && echo "removed image ${IMAGE}" \
    || echo "kept image ${IMAGE} — a container still references it"
else
  echo "image ${IMAGE} not present — nothing to remove"
fi

CACHE_DIR="${ENV_DIR}/.cache/${SOFTWARE}"
[ -d "${CACHE_DIR}" ] && { rm -rf "${CACHE_DIR}"; echo "removed materials cache ${CACHE_DIR}"; } || true
echo "uninstall done for ${SOFTWARE}"
