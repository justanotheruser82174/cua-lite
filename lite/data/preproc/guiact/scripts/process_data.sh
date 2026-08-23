#!/bin/bash
#
# Run the GUIAct Python preprocessing pipeline (grounding.action + use).
# Usage: ./process_data.sh [--verbose] [--dry-run] [--head N]
#
# Forwarded flags apply to both task scripts. The use-only --subset
# flag is not supported here; call use.py directly for that.
#
# Requires: CUA_LITE_RAW_DATASETS_ROOT, plus CUA_LITE_DATASETS_ROOT when writing.
# Images must be extracted first via process_raw_data.sh.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUIACT_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT="$(cd "${GUIACT_DIR}/../../../.." && pwd)"

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        -h|--help)
            echo "Usage: $0 [--verbose] [--dry-run] [--head N]"
            echo ""
            echo "Runs: grounding-action.py, use.py"
            echo "Requires: CUA_LITE_RAW_DATASETS_ROOT, plus CUA_LITE_DATASETS_ROOT unless --dry-run"
            exit 0
            ;;
        --subset|--subset=*)
            echo "Error: --subset is use-only; run use.py directly."
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
# Ensure the worktree's lite/ shadows any editable install pointing elsewhere.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

uv run python "${GUIACT_DIR}/grounding-action.py" "$@"   # web-single → browser/grounding.action
uv run python "${GUIACT_DIR}/use.py" "$@"         # web-multi + smartphone → use
if [ "$DRY_RUN" = false ]; then echo "Done. Output under \${CUA_LITE_DATASETS_ROOT}/cua-lite/GUIAct/."; fi
