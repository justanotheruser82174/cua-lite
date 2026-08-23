#!/bin/bash
#
# Extract the image archives GUI-360 ships as .tar.gz.
# Usage: ./process_raw_data.sh [--dry-run]
#
# Extracts:
#   - processed_data/grounding_resize/images.tar.gz        -> .../images/   (grounding.point)
#   - processed_data/screen_parsing_train_resize/images.tar.gz -> .../images/   (understanding)
#   - train/image.tar.gz                                   -> train/image/  (use)
# A marker is written only after tar exits successfully, so an interrupted
# extraction is retried instead of mistaken for a complete directory.
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
            echo "Extracts GUI-360 image archives (processed subsets + train trajectories)."
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

ROOT="${CUA_LITE_RAW_DATASETS_ROOT}/vyokky/GUI-360"

# extract <archive> <dest_dir> <sentinel_subdir>
extract() {
    local archive="$1" dest="$2" sentinel="$3"
    local marker="${archive}.extracted"
    if [ ! -f "${archive}" ]; then
        echo -e "${BLUE}Skipping (no archive): ${archive}${NC}"
        return
    fi
    if [ -f "${marker}" ]; then
        echo -e "${BLUE}Skipping (complete): ${sentinel}${NC}"
        return
    fi
    echo "Extracting ${archive} -> ${dest}/ ..."
    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY RUN] tar --no-same-owner --no-same-permissions --touch --no-overwrite-dir -xzf ${archive} -C ${dest}${NC}"
    else
        mkdir -p "${dest}"
        ( umask 000 && tar --no-same-owner --no-same-permissions --touch \
            --no-overwrite-dir -xzf "${archive}" -C "${dest}" )
        touch "${marker}"
        echo -e "${GREEN}✓ ${archive}${NC}"
    fi
}

# Processed subsets: archive sits next to its images/ destination.
extract "${ROOT}/processed_data/grounding_resize/images.tar.gz" \
        "${ROOT}/processed_data/grounding_resize" \
        "${ROOT}/processed_data/grounding_resize/images"
extract "${ROOT}/processed_data/screen_parsing_train_resize/images.tar.gz" \
        "${ROOT}/processed_data/screen_parsing_train_resize" \
        "${ROOT}/processed_data/screen_parsing_train_resize/images"

# Train trajectory screenshots: tar root is image/, so extract into train/.
extract "${ROOT}/train/image.tar.gz" \
        "${ROOT}/train" \
        "${ROOT}/train/image"

echo "============================================================"
echo "Done. Next: bash lite/data/preproc/gui360/scripts/process_data.sh"
echo "============================================================"
