#!/bin/bash
# Install OSWorld-G: clone benchmark repo into .cache/.
#
# Data goes into $ENV_DIR/.cache/OSWorld-G/ (git-ignored).
# Idempotent: re-run is a fast no-op if already cloned.
#
# Usage:
#   uv run --no-sync bash lite/gym/envs/osworld_g/scripts/install.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

echo "[osworld_g install] Cloning OSWorld-G benchmark data..."
python "$SCRIPT_DIR/utils/download_tasks.py"
echo "[osworld_g install] done ✓"
