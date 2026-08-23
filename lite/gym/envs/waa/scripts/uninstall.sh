#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
CACHE_ROOT="$HOME/.cache/cua-lite/waa"
RUNNER_IMAGE="$(python "$SCRIPT_DIR/utils/config_value.py" env_kwargs runner_image)"
PREP_IMAGE="cua-lite/waa-prep:latest"

bash "$SCRIPT_DIR/cleanup.sh"
for image in "$RUNNER_IMAGE" "$PREP_IMAGE"; do
  if docker image inspect "$image" >/dev/null 2>&1; then
    docker image rm "$image"
  fi
done
rm -rf "$CACHE_ROOT"
echo "WindowsAgentArena resources removed."
