#!/bin/bash
#
# Verify CAGUI raw data layout. CAGUI is delivered pre-extracted (no archives
# to merge), so this script only validates that the expected directories and
# files exist and prints summary counts. Idempotent.
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
            echo "Verifies CAGUI raw data layout under \${CUA_LITE_RAW_DATASETS_ROOT}/OpenBMB/CAGUI."
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

BASE="${CUA_LITE_RAW_DATASETS_ROOT}/OpenBMB/CAGUI"

if [ ! -d "${BASE}" ]; then
    echo -e "${RED}Error: ${BASE} does not exist. Run download_raw_data.sh first.${NC}"
    exit 1
fi

echo "============================================================"
echo "Verifying CAGUI layout under ${BASE}"
echo "============================================================"

required_paths=(
    "CAGUI_grounding/code/cap.jsonl"
    "CAGUI_grounding/code/ocr.jsonl"
    "CAGUI_grounding/images/cap"
    "CAGUI_grounding/images/ocr"
    "CAGUI_agent/domestic"
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
    echo -e "${BLUE}[DRY RUN] Would print counts for: cap.jsonl, ocr.jsonl, image dirs, episode dirs${NC}"
    exit 0
fi

cap_lines=$(wc -l < "${BASE}/CAGUI_grounding/code/cap.jsonl")
ocr_lines=$(wc -l < "${BASE}/CAGUI_grounding/code/ocr.jsonl")
cap_images=$(find "${BASE}/CAGUI_grounding/images/cap" -maxdepth 1 -type f | wc -l)
ocr_images=$(find "${BASE}/CAGUI_grounding/images/ocr" -maxdepth 1 -type f | wc -l)
n_episodes=$(find "${BASE}/CAGUI_agent/domestic" -mindepth 1 -maxdepth 1 -type d | wc -l)
n_episode_json=0
for episode_dir in "${BASE}/CAGUI_agent/domestic"/*; do
    [ -d "${episode_dir}" ] || continue
    episode_id=${episode_dir##*/}
    [ -f "${episode_dir}/${episode_id}.json" ] && n_episode_json=$((n_episode_json + 1))
done

echo ""
echo "============================================================"
echo "CAGUI counts (expected: 1500 / 1500 / 1500 / 1500 / 600 / 600)"
echo "============================================================"
printf "  cap.jsonl records:     %s\n" "${cap_lines}"
printf "  ocr.jsonl records:     %s\n" "${ocr_lines}"
printf "  cap image files:       %s\n" "${cap_images}"
printf "  ocr image files:       %s\n" "${ocr_images}"
printf "  agent/domestic episodes: %s\n" "${n_episodes}"
printf "  matching episode JSON:    %s\n" "${n_episode_json}"
echo "============================================================"
if [ "${cap_lines}" -ne 1500 ] || [ "${ocr_lines}" -ne 1500 ] \
   || [ "${cap_images}" -ne 1500 ] || [ "${ocr_images}" -ne 1500 ] \
   || [ "${n_episodes}" -ne 600 ] || [ "${n_episode_json}" -ne 600 ]; then
    echo -e "${RED}Error: CAGUI raw-data counts do not match the expected corpus.${NC}"
    exit 1
fi
echo -e "${GREEN}Layout OK.${NC}"
