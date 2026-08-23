#!/bin/bash
# Install ScreenSpot-Pro: pre-download dataset from HuggingFace.
#
# Data goes into ~/.cache/huggingface/hub/ (HuggingFace default cache).
# Idempotent: re-run is a fast no-op if already cached.
#
# Usage:
#   uv run --no-sync bash lite/gym/envs/screenspot_pro/scripts/install.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# No image here, but the `uv run --no-sync bash` invocation contract (bare `python`
# + `uv pip`, never bare pip) is the same one every install.sh signs; it and
# ``require_uv`` live in the shared helper.
source "$SCRIPT_DIR/../../../scripts/image_build.sh"

echo "[screenspot_pro install] installing huggingface_hub ..."
require_uv
uv pip install -q huggingface_hub

echo "[screenspot_pro install] Pre-downloading dataset from HuggingFace..."
python "$SCRIPT_DIR/utils/download_tasks.py"
echo "[screenspot_pro install] done ✓"
