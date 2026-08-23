#!/bin/bash
#
# Merge split zip archives and extract images for AgentNet (OpenCUA).
# Usage: ./process_raw_data.sh [--dry-run]
#
# Requires: CUA_LITE_RAW_DATASETS_ROOT. Expects ubuntu_images and win_mac_images
# under ${CUA_LITE_RAW_DATASETS_ROOT}/xlangai/AgentNet/ with split images.zip.
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
            echo "Merges split images.zip in ubuntu_images/ and win_mac_images/ and extracts."
            echo "Requires CUA_LITE_RAW_DATASETS_ROOT to be set."
            exit 0
            ;;
        *)
            echo -e "${RED}Error: unknown option: $arg${NC}"
            exit 2
            ;;
    esac
done

if [ -z "${CUA_LITE_RAW_DATASETS_ROOT}" ]; then
    echo -e "${RED}Error: CUA_LITE_RAW_DATASETS_ROOT is not set.${NC}"
    exit 1
fi

BASE="${CUA_LITE_RAW_DATASETS_ROOT}/xlangai/AgentNet"
for subdir in ubuntu_images win_mac_images; do
    dir="${BASE}/${subdir}"
    if [ ! -d "${dir}" ]; then
        echo -e "${BLUE}Skipping ${subdir}/ (directory not found)${NC}"
        continue
    fi
    if [ ! -f "${dir}/images.zip" ]; then
        echo -e "${BLUE}Skipping ${subdir}/ (no images.zip)${NC}"
        continue
    fi
    echo "============================================================"
    echo "Processing ${subdir}/..."
    echo "============================================================"
    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY RUN] Would merge and unzip in ${dir}${NC}"
    else
        archive="${dir}/images-full.zip"
        marker="${archive}.extracted"
        if [ ! -f "${marker}" ]; then
            if [ ! -f "${archive}" ] || ! unzip -tq "${archive}" >/dev/null; then
                tmp_dir=$(mktemp -d "${dir}/.images-full.XXXXXX")
                tmp="${tmp_dir}/images-full.zip"
                if (cd "${dir}" && zip -s 0 images.zip --out "${tmp}" >/dev/null) \
                   && unzip -tq "${tmp}" >/dev/null; then
                    mv "${tmp}" "${archive}"
                    rmdir "${tmp_dir}"
                else
                    rm -f "${tmp}"
                    rmdir "${tmp_dir}"
                    echo -e "${RED}Error: failed to build a valid ${subdir}/images-full.zip${NC}"
                    exit 1
                fi
            fi
            (cd "${dir}" && unzip -q -o images-full.zip -d .)
            touch "${marker}"
        fi
        echo -e "${GREEN}✓ ${subdir}/${NC}"
    fi
done
echo "============================================================"
echo "Done."
echo "============================================================"
