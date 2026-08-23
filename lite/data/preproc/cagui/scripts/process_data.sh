#!/bin/bash
#
# Run CAGUI Python preprocessing pipeline (understanding + use).
# Usage: ./process_data.sh [--verbose] [--dry-run] [--head N]
#
# Forwards common CLI args to both Python modules. Their subset namespaces are
# disjoint, so --subset is rejected here; invoke the task module directly.
#
# Requires: CUA_LITE_RAW_DATASETS_ROOT, plus CUA_LITE_DATASETS_ROOT when writing.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAGUI_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT="$(cd "${CAGUI_DIR}/../../../.." && pwd)"

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        -h|--help)
            echo "Usage: $0 [--verbose] [--dry-run] [--head N]"
            echo ""
            echo "Runs: lite.data.preproc.cagui.understanding, lite.data.preproc.cagui.use"
            echo "Requires: CUA_LITE_RAW_DATASETS_ROOT, plus CUA_LITE_DATASETS_ROOT unless --dry-run"
            exit 0
            ;;
        --subset|--subset=*)
            echo "Error: understanding and use have disjoint subsets; run the task Python module directly."
            exit 2
            ;;
        --dry-run) DRY_RUN=true ;;
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
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"   # worktree's lite/ shadows editable install

uv run python -m lite.data.preproc.cagui.understanding "$@"
uv run python -m lite.data.preproc.cagui.use "$@"
if [ "$DRY_RUN" = false ]; then echo "Done. Output under \${CUA_LITE_DATASETS_ROOT}/cua-lite/CAGUI/."; fi
