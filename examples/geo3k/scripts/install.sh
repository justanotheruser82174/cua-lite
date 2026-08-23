#!/bin/bash
# Download Geo3K example parquet data into examples/geo3k/.cache/.
#
# Usage:
#   uv run bash examples/geo3k/scripts/install.sh  # host
#   bash examples/geo3k/scripts/install.sh         # Slime container

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${EXAMPLE_DIR}/.cache/geo3k_imgurl"

mkdir -p "$DATA_DIR"

python - "$DATA_DIR" <<'PY'
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise SystemExit(
        "huggingface_hub is required; install project dependencies with `uv sync --all-extras`"
    ) from exc

snapshot_download(
    repo_id="chenhegu/geo3k_imgurl",
    repo_type="dataset",
    allow_patterns=["*.parquet"],
    local_dir=Path(sys.argv[1]),
)
PY

if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "Error: expected ${DATA_DIR}/train.parquet after download" >&2
    exit 1
fi

echo "[geo3k install] data ready: ${DATA_DIR}/train.parquet"
