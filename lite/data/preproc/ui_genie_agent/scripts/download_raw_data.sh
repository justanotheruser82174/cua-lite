#!/bin/bash
#
# Download raw data for both UI-Genie-Agent subsets:
#   - HanXiao1999/UI-Genie-Agent-16k (annotations + ui_genie subset screenshots, ~6 GB)
#   - Yuxiang007/AMEX (screenshot.z01-z08 + screenshot.zip, ~94 GB) — needed for amex subset
#
# Usage: ./download_raw_data.sh [--dry-run]
#
# Requires: CUA_LITE_RAW_DATASETS_ROOT, and 'hf' or 'huggingface-cli' on PATH.
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        -h|--help)
            echo "Usage: $0 [--dry-run]"
            echo ""
            echo "Downloads:"
            echo "  HanXiao1999/UI-Genie-Agent-16k -> \${CUA_LITE_RAW_DATASETS_ROOT}/HanXiao1999/UI-Genie-Agent-16k"
            echo "  Yuxiang007/AMEX (screenshot only) -> \${CUA_LITE_RAW_DATASETS_ROOT}/Yuxiang007/AMEX"
            echo ""
            echo "Requires CUA_LITE_RAW_DATASETS_ROOT to be set."
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

run() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY RUN] $*${NC}"
    else
        "$@"
    fi
}

echo "============================================================"
echo "Step 1/2: HanXiao1999/UI-Genie-Agent-16k (~6 GB)"
echo "============================================================"
run "${HF_CMD}" download HanXiao1999/UI-Genie-Agent-16k --repo-type dataset \
    --local-dir "${CUA_LITE_RAW_DATASETS_ROOT}/HanXiao1999/UI-Genie-Agent-16k"
echo -e "${GREEN}✓ HanXiao1999/UI-Genie-Agent-16k${NC}"

echo ""
echo "============================================================"
echo "Step 2/2: Yuxiang007/AMEX (screenshot zip parts only, ~94 GB)"
echo "============================================================"
echo "(Needed for the 'amex' subset only. Skip step 2 if you only"
echo " care about the default 'ui_genie' subset.)"
echo ""
run "${HF_CMD}" download Yuxiang007/AMEX --repo-type dataset \
    --include 'AMEX/screenshot.*' \
    --local-dir "${CUA_LITE_RAW_DATASETS_ROOT}/Yuxiang007/AMEX"
echo -e "${GREEN}✓ Yuxiang007/AMEX (screenshot parts)${NC}"

echo ""
echo "============================================================"
echo "Download complete. Next: run process_raw_data.sh to merge"
echo "the multi-part AMEX zip and unpack the screenshots."
echo "============================================================"
