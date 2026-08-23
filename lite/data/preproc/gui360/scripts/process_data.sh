#!/bin/bash
#
# Run the GUI-360 Python preprocessing pipeline.
# Usage: ./process_data.sh [--verbose] [--dry-run] [--head N]
#
# Runs: grounding-point.py, understanding.py, use.py
# Requires: CUA_LITE_RAW_DATASETS_ROOT, plus CUA_LITE_DATASETS_ROOT when writing.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUI360_DIR="$(dirname "${SCRIPT_DIR}")"
# Directory containing lite/ (repo root for running uv run python)
REPO_ROOT="$(cd "${GUI360_DIR}/../../../.." && pwd)"

ARGS=()
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true; ARGS+=("$arg") ;;
        -h|--help)
            echo "Usage: $0 [--verbose] [--dry-run] [--head N]"
            echo ""
            echo "Runs: grounding-point.py, understanding.py, use.py"
            echo "Requires: CUA_LITE_RAW_DATASETS_ROOT, plus CUA_LITE_DATASETS_ROOT unless --dry-run"
            exit 0
            ;;
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

uv run python "${GUI360_DIR}/grounding-point.py" "${ARGS[@]}"
uv run python "${GUI360_DIR}/understanding.py" "${ARGS[@]}"
uv run python "${GUI360_DIR}/use.py" "${ARGS[@]}"
if [ "$DRY_RUN" = false ]; then echo "Done. Output under \${CUA_LITE_DATASETS_ROOT}/cua-lite/GUI-360/."; fi
