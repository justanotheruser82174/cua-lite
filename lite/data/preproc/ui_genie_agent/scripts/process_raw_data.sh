#!/bin/bash
#
# Prepare raw data for both UI-Genie-Agent subsets.
#
#   ui_genie subset: pre-extracted by the HF repo, only verifies layout.
#   amex subset:    merges the multi-part zip from Yuxiang007/AMEX
#                   (screenshot.z01-z08 + screenshot.zip), unzips the result
#                   into Yuxiang007/AMEX/screenshot/, and symlinks it under
#                   HanXiao1999/UI-Genie-Agent-16k/AMEX/screenshot/.
#
# Idempotent: re-running skips any step that's already complete.
#
# Usage: ./process_raw_data.sh [--dry-run]
#
# Requires: CUA_LITE_RAW_DATASETS_ROOT, plus `zip` and `unzip` on PATH.
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        -h|--help)
            echo "Usage: $0 [--dry-run]"
            echo ""
            echo "Verifies UI-Genie layout, merges + unzips Yuxiang007/AMEX screenshots,"
            echo "and symlinks them under HanXiao1999/UI-Genie-Agent-16k/AMEX/screenshot/."
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
for cmd in zip unzip; do
    if ! command -v "$cmd" &>/dev/null; then
        echo -e "${RED}Error: '$cmd' is not on PATH.${NC}"
        exit 1
    fi
done

UI_GENIE_BASE="${CUA_LITE_RAW_DATASETS_ROOT}/HanXiao1999/UI-Genie-Agent-16k"
AMEX_BASE="${CUA_LITE_RAW_DATASETS_ROOT}/Yuxiang007/AMEX"

# ---------------------------------------------------------------------------
# Step 1: ui_genie subset layout check
# ---------------------------------------------------------------------------

echo "============================================================"
echo "Step 1/3: Verify ui_genie subset layout"
echo "============================================================"

if [ ! -d "${UI_GENIE_BASE}" ]; then
    echo -e "${RED}Error: ${UI_GENIE_BASE} does not exist. Run download_raw_data.sh first.${NC}"
    exit 1
fi

required_paths=(
    "ui_genie_agent_16k.jsonl"
    "data/screenshots"
)

missing=0
for rel in "${required_paths[@]}"; do
    p="${UI_GENIE_BASE}/${rel}"
    if [ -e "${p}" ]; then
        echo -e "${GREEN}✓${NC} ${rel}"
    else
        echo -e "${RED}✗${NC} ${rel} (missing — required for ui_genie subset)"
        missing=$((missing + 1))
    fi
done
if [ "${missing}" -gt 0 ]; then
    echo -e "${RED}Error: ${missing} required path(s) missing.${NC}"
    exit 1
fi

if [ -f "${UI_GENIE_BASE}/AMEX_Agent_34K.jsonl" ]; then
    echo -e "${GREEN}✓${NC} AMEX_Agent_34K.jsonl (annotations for amex subset)"
else
    echo -e "${YELLOW}-${NC} AMEX_Agent_34K.jsonl (missing — amex subset will be unavailable)"
fi

ui_lines=$(wc -l < "${UI_GENIE_BASE}/ui_genie_agent_16k.jsonl")
ui_uids=$(ls "${UI_GENIE_BASE}/data/screenshots" | wc -l)
printf "  ui_genie_agent_16k.jsonl records:  %s\n" "${ui_lines}"
printf "  data/screenshots/ uid dirs:        %s\n" "${ui_uids}"

# ---------------------------------------------------------------------------
# Step 2: AMEX merge + unzip
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "Step 2/3: Prepare AMEX screenshots (Yuxiang007/AMEX)"
echo "============================================================"

if [ ! -d "${AMEX_BASE}/AMEX" ]; then
    echo -e "${YELLOW}- ${AMEX_BASE}/AMEX is missing. Skipping amex prep.${NC}"
    echo -e "${YELLOW}  (Run download_raw_data.sh to fetch the AMEX zip parts if you want the amex subset.)${NC}"
    SKIP_AMEX=true
else
    SKIP_AMEX=false
    cd "${AMEX_BASE}/AMEX"

    # 2a. Verify all expected zip parts are present.
    required_zips=(screenshot.zip screenshot.z01 screenshot.z02 screenshot.z03 \
                   screenshot.z04 screenshot.z05 screenshot.z06 screenshot.z07 screenshot.z08)
    missing=0
    for f in "${required_zips[@]}"; do
        if [ -f "${f}" ]; then
            echo -e "${GREEN}✓${NC} ${f}"
        else
            echo -e "${RED}✗${NC} ${f} (missing)"
            missing=$((missing + 1))
        fi
    done
    if [ "${missing}" -gt 0 ]; then
        echo -e "${RED}Error: ${missing} AMEX zip part(s) missing. Re-run download_raw_data.sh.${NC}"
        exit 1
    fi

    # 2b. Build the merged archive atomically. Mere existence is not proof of
    # completion: an interrupted merge leaves a truncated output behind.
    UNZIP_MARKER="${PWD}/screenshot_merged.zip.extracted"
    if [ -f "${UNZIP_MARKER}" ]; then
        echo -e "${GREEN}✓${NC} screenshot_merged.zip already merged and extracted"
    elif [ -f "screenshot_merged.zip" ] && unzip -tq screenshot_merged.zip >/dev/null; then
        echo -e "${GREEN}✓${NC} screenshot_merged.zip is valid, skipping merge"
    else
        echo "Merging multi-part zip with 'zip --fix' (~5 min)..."
        if [ "$DRY_RUN" = true ]; then
            echo -e "${BLUE}[DRY RUN] Would atomically rebuild screenshot_merged.zip${NC}"
        else
            tmp_dir=$(mktemp -d "${PWD}/.screenshot_merged.XXXXXX")
            tmp="${tmp_dir}/screenshot_merged.zip"
            if zip --fix screenshot.zip --out "${tmp}" >/dev/null \
               && unzip -tq "${tmp}" >/dev/null; then
                mv "${tmp}" screenshot_merged.zip
                rmdir "${tmp_dir}"
            else
                rm -f "${tmp}"
                rmdir "${tmp_dir}"
                echo -e "${RED}Error: failed to build a valid screenshot_merged.zip${NC}"
                exit 1
            fi
        fi
        echo -e "${GREEN}✓${NC} screenshot_merged.zip"
    fi

    # 2c. Unzip into ../screenshot/ if not already done.
    UNZIP_DIR="${AMEX_BASE}/screenshot"
    if [ -f "${UNZIP_MARKER}" ]; then
        echo -e "${GREEN}✓${NC} ${UNZIP_DIR} extraction complete, skipping unzip"
    else
        echo "Unzipping screenshot_merged.zip to ${UNZIP_DIR} (~30 min, ~104675 files / ~100 GB)..."
        if [ "$DRY_RUN" = true ]; then
            echo -e "${BLUE}[DRY RUN] umask 000 && mkdir -p ${UNZIP_DIR} && unzip -j -q -o screenshot_merged.zip -d ${UNZIP_DIR}${NC}"
        else
            umask 000
            mkdir -p "${UNZIP_DIR}"
            unzip -j -q -o screenshot_merged.zip -d "${UNZIP_DIR}"
            touch "${UNZIP_MARKER}"
        fi
        echo -e "${GREEN}✓${NC} unzipped to ${UNZIP_DIR}"
    fi
fi

# ---------------------------------------------------------------------------
# Step 3: Symlink AMEX screenshots into the path the preprocessor expects
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "Step 3/3: Symlink AMEX screenshots under UI-Genie repo"
echo "============================================================"

LINK_TARGET="${UI_GENIE_BASE}/AMEX/screenshot"
LINK_SOURCE="${AMEX_BASE}/screenshot"

if [ "$SKIP_AMEX" = true ]; then
    echo -e "${YELLOW}- amex subset skipped (no Yuxiang007/AMEX on disk).${NC}"
elif [ "$DRY_RUN" = true ]; then
    echo -e "${BLUE}[DRY RUN] mkdir -p ${UI_GENIE_BASE}/AMEX && ln -sfn ${LINK_SOURCE} ${LINK_TARGET}${NC}"
else
    mkdir -p "${UI_GENIE_BASE}/AMEX"
    ln -sfn "${LINK_SOURCE}" "${LINK_TARGET}"
    n_files=$(ls "${LINK_TARGET}/" | wc -l)
    echo -e "${GREEN}✓${NC} ${LINK_TARGET} -> ${LINK_SOURCE} (${n_files} files)"
fi

echo ""
echo "============================================================"
echo -e "${GREEN}Layout OK.${NC}"
echo "============================================================"
