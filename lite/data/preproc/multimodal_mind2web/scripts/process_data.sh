#!/bin/bash
# Run the Multimodal-Mind2Web preprocessing pipeline (train split only).
# Usage: ./process_data.sh [--verbose] [--dry-run] [--head N]
# Requires: CUA_LITE_RAW_DATASETS_ROOT, plus CUA_LITE_DATASETS_ROOT when writing.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M2W_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT="$(cd "${M2W_DIR}/../../../.." && pwd)"
ARGS=()
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        -h|--help) echo "Usage: $0 [--verbose] [--dry-run] [--head N]"; exit 0 ;;
        --dry-run) DRY_RUN=true; ARGS+=("$arg") ;;
        *) ARGS+=("$arg") ;;
    esac
done
if [ -z "${CUA_LITE_RAW_DATASETS_ROOT}" ]; then echo "Error: CUA_LITE_RAW_DATASETS_ROOT is not set."; exit 1; fi
if [ "$DRY_RUN" = false ] && [ -z "${CUA_LITE_DATASETS_ROOT}" ]; then echo "Error: CUA_LITE_DATASETS_ROOT is not set."; exit 1; fi
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
uv run python "${M2W_DIR}/use.py" "${ARGS[@]}"
if [ "$DRY_RUN" = false ]; then echo "Done. Output under \${CUA_LITE_DATASETS_ROOT}/cua-lite/Multimodal-Mind2Web/."; fi
