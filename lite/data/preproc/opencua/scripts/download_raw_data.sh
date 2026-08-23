#!/bin/bash
#
# Download raw AgentNet (OpenCUA) dataset from Hugging Face into CUA_LITE_RAW_DATASETS_ROOT.
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
            echo "Downloads: xlangai/AgentNet -> \${CUA_LITE_RAW_DATASETS_ROOT}/xlangai/AgentNet"
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
echo "Downloading AgentNet (OpenCUA) raw dataset..."
echo "CUA_LITE_RAW_DATASETS_ROOT: ${CUA_LITE_RAW_DATASETS_ROOT}"
echo "============================================================"

run "${HF_CMD}" download xlangai/AgentNet --repo-type dataset --local-dir "${CUA_LITE_RAW_DATASETS_ROOT}/xlangai/AgentNet"
echo -e "${GREEN}✓ xlangai/AgentNet${NC}"

echo "============================================================"
echo "Download complete."
echo "============================================================"
