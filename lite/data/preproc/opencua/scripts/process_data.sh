#!/bin/bash
#
# Run OpenCUA (AgentNet) Python preprocessing (use).
# Usage: ./process_data.sh [--verbose] [--dry-run] [--head N]
#
# Requires: CUA_LITE_RAW_DATASETS_ROOT, plus CUA_LITE_DATASETS_ROOT when writing.
# Output:   ${CUA_LITE_DATASETS_ROOT}/cua-lite/OpenCUA/
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPROC_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT="$(cd "${PREPROC_DIR}/../../../.." && pwd)"

ARGS=()
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true; ARGS+=("$arg") ;;
        -h|--help)
            echo "Usage: $0 [--verbose] [--dry-run] [--head N]"
            echo ""
            echo "Runs: use.py"
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
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

uv run python "${PREPROC_DIR}/use.py" "${ARGS[@]}"

if [ "$DRY_RUN" = false ]; then echo "Done. Output under \${CUA_LITE_DATASETS_ROOT}/cua-lite/OpenCUA/."; fi
