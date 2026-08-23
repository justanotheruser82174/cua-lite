#!/bin/bash
#
# Download raw GUIOdyssey dataset from Hugging Face into CUA_LITE_RAW_DATASETS_ROOT.
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
            echo "Downloads: hflqf88888/GUIOdyssey -> \${CUA_LITE_RAW_DATASETS_ROOT}/hflqf88888/GUIOdyssey"
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
echo "Downloading GUIOdyssey raw dataset..."
echo "CUA_LITE_RAW_DATASETS_ROOT: ${CUA_LITE_RAW_DATASETS_ROOT}"
echo "Note: This is an 87 GB dataset (mostly screenshots). Expect a long download."
echo "============================================================"

run "${HF_CMD}" download hflqf88888/GUIOdyssey --repo-type dataset --local-dir "${CUA_LITE_RAW_DATASETS_ROOT}/hflqf88888/GUIOdyssey"
echo -e "${GREEN}✓ hflqf88888/GUIOdyssey${NC}"

echo "============================================================"
echo "Download complete."
echo "============================================================"
