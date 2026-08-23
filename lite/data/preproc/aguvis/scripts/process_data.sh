#!/bin/bash
#
# Run the Aguvis Python preprocessing pipeline.
# Usage: ./process_data.sh [--verbose] [--dry-run] [--head N]
#
# Runs: grounding-action.py (Stage 1), use.py (Stage 2)
# Requires: CUA_LITE_RAW_DATASETS_ROOT, plus CUA_LITE_DATASETS_ROOT when writing.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGUVIS_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT="$(cd "${AGUVIS_DIR}/../../../.." && pwd)"

ARGS=()
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        -h|--help)
            echo "Usage: $0 [--verbose] [--dry-run] [--head N]"
            echo ""
            echo "Runs: grounding-action.py (Stage 1), use.py (Stage 2)"
            echo "Requires: CUA_LITE_RAW_DATASETS_ROOT, plus CUA_LITE_DATASETS_ROOT unless --dry-run"
            exit 0
            ;;
        --subset|--subset=*)
            echo "Error: Stage 1 and Stage 2 have disjoint subsets; run the task Python module directly."
            exit 2
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
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

uv run python "${AGUVIS_DIR}/grounding-action.py" "${ARGS[@]}"
uv run python "${AGUVIS_DIR}/use.py" "${ARGS[@]}"
if [ "$DRY_RUN" = false ]; then echo "Done. Output under \${CUA_LITE_DATASETS_ROOT}/cua-lite/Aguvis/."; fi
