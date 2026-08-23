#!/bin/bash
#
# Download the parts of vyokky/GUI-360 that the cua-lite adapters consume.
# Usage: ./download_raw_data.sh [--dry-run] [--all]
#
# By default downloads only what the adapters need (skips the ~268 GB `fail/`
# split and the unused a11y / action_prediction subsets):
#   - processed_data/grounding_resize/       (grounding.point)
#   - processed_data/screen_parsing_train_resize/  (understanding)
#   - train/                                 (use trajectories + images)
# Pass --all to mirror the entire repo instead.
#
# Requires: CUA_LITE_RAW_DATASETS_ROOT, and 'hf' or 'huggingface-cli' on PATH.
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

DRY_RUN=false
ALL=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --all) ALL=true ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--all]"
            echo ""
            echo "Downloads vyokky/GUI-360 -> \${CUA_LITE_RAW_DATASETS_ROOT}/vyokky/GUI-360"
            echo "Default: only grounding_resize, screen_parsing_train_resize, and train/."
            echo "--all: mirror the entire repository (large)."
            exit 0
            ;;
        *) echo -e "${RED}Error: unknown option: $arg${NC}" >&2; exit 2 ;;
    esac
done

if [ -z "${CUA_LITE_RAW_DATASETS_ROOT}" ]; then
    echo -e "${RED}Error: CUA_LITE_RAW_DATASETS_ROOT is not set.${NC}"
    exit 1
fi

if command -v hf &>/dev/null; then
    HF_CMD=hf
elif command -v huggingface-cli &>/dev/null; then
    HF_CMD=huggingface-cli
else
    echo -e "${RED}Error: Neither 'hf' nor 'huggingface-cli' found on PATH.${NC}"
    exit 1
fi

LOCAL_DIR="${CUA_LITE_RAW_DATASETS_ROOT}/vyokky/GUI-360"
INCLUDE_ARGS=()
if [ "$ALL" = false ]; then
    INCLUDE_ARGS=(
        --include "processed_data/grounding_resize/*"
        --include "processed_data/screen_parsing_train_resize/*"
        --include "train/*"
    )
fi

echo "============================================================"
echo "Downloading GUI-360 raw dataset ($([ "$ALL" = true ] && echo "full" || echo "adapter subset"))..."
echo "CUA_LITE_RAW_DATASETS_ROOT: ${CUA_LITE_RAW_DATASETS_ROOT}"
echo "============================================================"

CMD=("${HF_CMD}" download vyokky/GUI-360 --repo-type dataset --local-dir "${LOCAL_DIR}" "${INCLUDE_ARGS[@]}")
if [ "$DRY_RUN" = true ]; then
    echo -e "${BLUE}[DRY RUN] ${CMD[*]}${NC}"
else
    "${CMD[@]}"
    echo -e "${GREEN}✓ vyokky/GUI-360${NC}"
fi

echo "============================================================"
echo "Download complete. Next: bash lite/data/preproc/gui360/scripts/process_raw_data.sh"
echo "============================================================"
