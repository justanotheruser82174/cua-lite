#!/bin/bash
#
# Run ScaleCUA Python preprocessing pipeline (understanding, grounding, use).
# Usage: ./process_data.sh [--verbose] [--dry-run] [--head N] [--head-entries N]
#
# Requires: CUA_LITE_RAW_DATASETS_ROOT, plus CUA_LITE_DATASETS_ROOT when writing.
# Run from repo root or ensure Python can import lite.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCALECUA_DIR="$(dirname "${SCRIPT_DIR}")"
# Directory containing lite/ (repo root for running uv run python)
REPO_ROOT="$(cd "${SCALECUA_DIR}/../../../.." && pwd)"

ARGS=("$@")
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        -h|--help)
            echo "Usage: $0 [--verbose] [--dry-run] [--head N] [--head-entries N]"
            echo ""
            echo "Runs: understanding.py, grounding-action.py, grounding-point.py, grounding-bbox.py, use.py"
            echo "Requires: CUA_LITE_RAW_DATASETS_ROOT, plus CUA_LITE_DATASETS_ROOT unless --dry-run"
            exit 0
            ;;
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

uv run python "${SCALECUA_DIR}/understanding.py" "${ARGS[@]}"
uv run python "${SCALECUA_DIR}/grounding-action.py" "${ARGS[@]}"
uv run python "${SCALECUA_DIR}/grounding-point.py" "${ARGS[@]}"
uv run python "${SCALECUA_DIR}/grounding-bbox.py" "${ARGS[@]}"
uv run python "${SCALECUA_DIR}/use.py" "${ARGS[@]}"
if [ "$DRY_RUN" = false ]; then echo "Done. Output under \${CUA_LITE_DATASETS_ROOT}/cua-lite/ScaleCUA/."; fi
