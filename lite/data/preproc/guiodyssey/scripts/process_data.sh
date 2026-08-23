#!/bin/bash
#
# Run the GUIOdyssey Python preprocessing pipeline.
# Usage: ./process_data.sh [--verbose] [--dry-run] [--head N]
#
# Runs: use.py, understanding.py
# Requires: CUA_LITE_RAW_DATASETS_ROOT, plus CUA_LITE_DATASETS_ROOT when writing.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUIODYSSEY_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT="$(cd "${GUIODYSSEY_DIR}/../../../.." && pwd)"

ARGS=()
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        -h|--help)
            echo "Usage: $0 [--verbose] [--dry-run] [--head N]"
            echo ""
            echo "Runs: use.py, understanding.py"
            echo "Requires: CUA_LITE_RAW_DATASETS_ROOT, plus CUA_LITE_DATASETS_ROOT unless --dry-run"
            exit 0
            ;;
        --dry-run) DRY_RUN=true; ARGS+=("$arg") ;;
        *) ARGS+=("$arg") ;;
    esac
done

if [ -z "${CUA_LITE_RAW_DATASETS_ROOT}" ]; then
    echo "Error: CUA_LITE_RAW_DATASETS_ROOT is not set."
    exit 1
fi
if [ "$DRY_RUN" = false ] && [ -z "${CUA_LITE_DATASETS_ROOT}" ]; then
    echo "Error: CUA_LITE_DATASETS_ROOT is not set."
    exit 1
fi

cd "${REPO_ROOT}"
# Ensure the worktree's lite/ shadows any editable install pointing elsewhere.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

uv run python "${GUIODYSSEY_DIR}/use.py" "${ARGS[@]}"
uv run python "${GUIODYSSEY_DIR}/understanding.py" "${ARGS[@]}"
if [ "$DRY_RUN" = false ]; then echo "Done. Output under \${CUA_LITE_DATASETS_ROOT}/cua-lite/GUIOdyssey/."; fi
