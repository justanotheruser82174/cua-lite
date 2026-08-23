#!/bin/bash
#
# Verify the extracted GUIOdyssey raw-data layout. The upstream screenshots
# arrive as a multipart zip; extraction is documented in ../README.md. This
# script validates the resulting tree and never rewrites the ~88 GiB payload.
#
# Usage: ./process_raw_data.sh [--dry-run]
#
# Requires: CUA_LITE_RAW_DATASETS_ROOT.
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
            echo "Verifies GUIOdyssey raw data layout under \${CUA_LITE_RAW_DATASETS_ROOT}/hflqf88888/GUIOdyssey."
            echo "Requires CUA_LITE_RAW_DATASETS_ROOT to be set."
            exit 0
            ;;
        *) echo -e "${RED}Error: unknown option: $arg${NC}"; exit 2 ;;
    esac
done

if [ -z "${CUA_LITE_RAW_DATASETS_ROOT}" ]; then
    echo -e "${RED}Error: CUA_LITE_RAW_DATASETS_ROOT is not set.${NC}"
    exit 1
fi

BASE="${CUA_LITE_RAW_DATASETS_ROOT}/hflqf88888/GUIOdyssey"

if [ ! -d "${BASE}" ]; then
    echo -e "${RED}Error: ${BASE} does not exist. Run download_raw_data.sh first.${NC}"
    exit 1
fi

echo "============================================================"
echo "Verifying GUIOdyssey layout under ${BASE}"
echo "============================================================"

required_paths=(
    "annotations"
    "screenshots"
    "splits"
    "all_annot.json"
)

missing=0
for rel in "${required_paths[@]}"; do
    p="${BASE}/${rel}"
    if [ -e "${p}" ]; then
        echo -e "${GREEN}✓${NC} ${rel}"
    else
        echo -e "${RED}✗${NC} ${rel} (missing)"
        missing=$((missing + 1))
    fi
done

if [ "${missing}" -gt 0 ]; then
    echo -e "${RED}Error: ${missing} required path(s) missing.${NC}"
    exit 1
fi

if [ "$DRY_RUN" = true ]; then
    echo -e "${BLUE}[DRY RUN] Would print counts for annotations/, screenshots/, splits/${NC}"
    exit 0
fi

n_annot=$(find "${BASE}/annotations" -maxdepth 1 -type f -name '*.json' | wc -l)
n_screens=$(find "${BASE}/screenshots" -maxdepth 1 -type f -name '*.png' | wc -l)
n_splits=$(find "${BASE}/splits" -maxdepth 1 -type f -name '*.json' | wc -l)

echo ""
echo "============================================================"
echo "GUIOdyssey counts (expected: 8334 / 127893 / 4)"
echo "============================================================"
printf "  annotations/*.json:     %s\n" "${n_annot}"
printf "  screenshots/*.png:      %s\n" "${n_screens}"
printf "  splits/*.json:          %s\n" "${n_splits}"
echo "============================================================"
if [ "${n_annot}" -ne 8334 ] || [ "${n_screens}" -ne 127893 ] || [ "${n_splits}" -ne 4 ]; then
    echo -e "${RED}Error: extracted GUIOdyssey counts do not match the expected corpus.${NC}"
    exit 1
fi
echo -e "${GREEN}Layout OK.${NC}"
