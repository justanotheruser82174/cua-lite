#!/bin/bash
#
# Merge and extract split .tar.gz archives (raw data unpack step).
# Usage: ./process_raw_data.sh --dataset-dir DIR [--keep-parts] [--no-extract] [--dry-run]
#

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
KEEP_PARTS=false
NO_EXTRACT=false
DRY_RUN=false
DATASET_DIR=""

# Parse command line arguments
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dataset-dir PATH  Path to dataset root (e.g. \${CUA_LITE_RAW_DATASETS_ROOT}/OpenGVLab/ScaleCUA-Data); must contain a 'data' subdir with data_*/"
    echo "  --keep-parts        Keep the part files after merging"
    echo "  --no-extract        Only merge, don't extract"
    echo "  --dry-run           Show what would be done without doing it"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --dataset-dir \"\${CUA_LITE_RAW_DATASETS_ROOT}/OpenGVLab/ScaleCUA-Data\" --keep-parts"
    echo "  $0 --dataset-dir \"\${CUA_LITE_RAW_DATASETS_ROOT}/zyliu/ScaleCUA-Data-Understanding\" --dry-run"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --keep-parts)
            KEEP_PARTS=true
            shift
            ;;
        --no-extract)
            NO_EXTRACT=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --dataset-dir)
            if [ "$#" -lt 2 ] || [ -z "$2" ]; then
                echo -e "${RED}Error: --dataset-dir requires a path${NC}"
                exit 2
            fi
            DATASET_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            echo ""
            show_help
            exit 2
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${DATASET_DIR}" ]; then
    echo -e "${RED}Error: --dataset-dir is required.${NC}"
    echo ""
    show_help
    exit 1
fi
DATA_DIR="${DATASET_DIR}/data"

# Check if data directory exists
if [ ! -d "${DATA_DIR}" ]; then
    echo -e "${RED}Error: Data directory not found: ${DATA_DIR}${NC}"
    exit 1
fi

# Print configuration
echo "============================================================"
echo "Starting raw data processing (merge + extract)..."
echo "Data directory: ${DATA_DIR}"
echo "Keep parts: ${KEEP_PARTS}"
echo "Extract: $([ "$NO_EXTRACT" = true ] && echo "NO" || echo "YES")"
echo "Dry run: ${DRY_RUN}"
echo "============================================================"
echo ""

# Counter for statistics
total_archives=0
total_merged=0
total_extracted=0
total_skipped=0
total_errors=0

# Process each data_* subdirectory
shopt -s nullglob  # Return empty array if no matches

merge_archive() {
    local archive="$1"
    shift
    local tmp
    tmp=$(mktemp "${archive}.tmp.XXXXXX") || return 1
    if cat "$@" > "${tmp}" && gzip -t "${tmp}" 2>/dev/null; then
        mv "${tmp}" "${archive}"
        return 0
    fi
    rm -f "${tmp}"
    return 1
}
for data_subdir in "${DATA_DIR}"/data_*/; do
    if [ ! -d "${data_subdir}" ]; then
        continue
    fi
    
    subdir_name=$(basename "${data_subdir}")
    cd "${data_subdir}" || continue
    
    # Include already-merged archives as well as split parts. Parts are commonly
    # deleted after a successful merge; excluding the merged file makes a later
    # interrupted extraction impossible to resume.
    # Use a temporary file to store unique archive names
    temp_archives=$(mktemp)

    for archive_file in *.tar.gz; do
        if [ -f "${archive_file}" ]; then
            echo "${archive_file}" >> "${temp_archives}"
        fi
    done
    
    for part_file in *.tar.gz.part-*; do
        if [ -f "${part_file}" ]; then
            # Extract base name (e.g., "windows.tar.gz" from "windows.tar.gz.part-000")
            base_name="${part_file%.part-*}"
            echo "${base_name}" >> "${temp_archives}"
        fi
    done
    
    # Get unique archive names and count them
    if [ ! -s "${temp_archives}" ]; then
        rm -f "${temp_archives}"
        echo -e "${YELLOW}Skipping ${subdir_name}/ (no archives or part files found)${NC}"
        echo ""
        continue
    fi
    
    unique_archives=$(sort -u "${temp_archives}")
    archive_count=$(echo "${unique_archives}" | wc -l)
    rm -f "${temp_archives}"
    
    echo -e "${BLUE}Processing ${subdir_name}/ (${archive_count} archives)...${NC}"
    
    # Process each unique archive
    while IFS= read -r archive_name; do
        echo "  ${archive_name}"
        
        # ``nullglob`` makes this an empty array when no parts remain. Using
        # ``ls`` here is unsafe: with no match it degenerates to bare ``ls`` and
        # mistakes every directory entry for an archive part.
        parts=("${archive_name}".part-*)
        
        part_count=${#parts[@]}
        echo "    Found ${part_count} part(s)"
        
        ((total_archives++))

        marker="${archive_name}.extracted"
        if [ -f "${archive_name}" ] && [ ! -f "${marker}" ] \
           && ! gzip -t "${archive_name}" 2>/dev/null; then
            if [ ${#parts[@]} -eq 0 ]; then
                echo -e "    ${RED}✗ Existing merged archive failed gzip validation and no parts remain${NC}"
                ((total_errors++))
                echo ""
                continue
            elif [ "$DRY_RUN" = true ]; then
                echo -e "    ${BLUE}[DRY RUN] Would rebuild invalid ${archive_name} atomically${NC}"
            elif merge_archive "${archive_name}" "${parts[@]}"; then
                echo -e "    ${GREEN}✓ Rebuilt invalid merged archive from parts${NC}"
                ((total_merged++))
            else
                echo -e "    ${RED}✗ Failed to rebuild merged archive from parts${NC}"
                ((total_errors++))
                echo ""
                continue
            fi
        fi
        
        # Check if merged file already exists
        if [ -f "${archive_name}" ]; then
            echo -e "    ${YELLOW}⚠ Merged file already exists${NC}"
            
            # Check if extraction is needed
            if [ "$NO_EXTRACT" = false ] && [ ! -f "${marker}" ]; then
                echo "    Extracting existing merged file..."
                if [ "$DRY_RUN" = true ]; then
                    echo -e "    ${BLUE}[DRY RUN] Would extract ${archive_name}${NC}"
                    ((total_extracted++))
                else
                    if ! gzip -t "${archive_name}" 2>/dev/null; then
                        echo -e "    ${RED}✗ Existing merged archive failed gzip validation${NC}"
                        ((total_errors++))
                    elif ! tar -tzf "${archive_name}" 2>/dev/null | grep -qv '/$'; then
                        echo -e "    ${RED}✗ Archive contains no files${NC}"
                        ((total_errors++))
                    elif tar --no-same-owner --no-same-permissions --touch \
                             --no-overwrite-dir -xzf "${archive_name}"; then
                        touch "${marker}"
                        echo -e "    ${GREEN}✓ Extracted successfully${NC}"
                        ((total_extracted++))
                    else
                        echo -e "    ${RED}✗ Extraction failed${NC}"
                        ((total_errors++))
                    fi
                fi
            else
                echo "    Skipping (already processed)"
                ((total_skipped++))
            fi
            echo ""
            continue
        fi

        if [ ${#parts[@]} -eq 0 ]; then
            echo -e "    ${RED}✗ No merged archive or part files found${NC}"
            ((total_errors++))
            echo ""
            continue
        fi
        
        # Merge parts
        echo "    Merging parts..."
        merge_success=false
        
        if [ "$DRY_RUN" = true ]; then
            echo -e "    ${BLUE}[DRY RUN] Would merge ${part_count} parts into ${archive_name}${NC}"
            merge_success=true
        else
            if merge_archive "${archive_name}" "${parts[@]}"; then
                merge_success=true
            else
                echo -e "    ${RED}✗ Merge failed${NC}"
                ((total_errors++))
            fi
        fi
        
        if [ "$merge_success" = true ]; then
            echo -e "    ${GREEN}✓ Merged successfully${NC}"
            ((total_merged++))
            ready_to_remove_parts=false
            
            # Extract if requested
            if [ "$NO_EXTRACT" = true ]; then
                ready_to_remove_parts=true
            else
                echo "    Extracting..."
                if [ "$DRY_RUN" = true ]; then
                    echo -e "    ${BLUE}[DRY RUN] Would extract ${archive_name}${NC}"
                    ((total_extracted++))
                elif ! tar -tzf "${archive_name}" 2>/dev/null | grep -qv '/$'; then
                    echo -e "    ${RED}✗ Archive contains no files${NC}"
                    ((total_errors++))
                else
                    if tar --no-same-owner --no-same-permissions --touch \
                           --no-overwrite-dir -xzf "${archive_name}"; then
                        touch "${archive_name}.extracted"
                        echo -e "    ${GREEN}✓ Extracted successfully${NC}"
                        ((total_extracted++))
                        ready_to_remove_parts=true
                    else
                        echo -e "    ${RED}✗ Extraction failed${NC}"
                        ((total_errors++))
                        echo ""
                        continue
                    fi
                fi
            fi
            
            # Remove part files if requested
            if [ "$KEEP_PARTS" = false ] && [ "$ready_to_remove_parts" = true ]; then
                echo "    Removing part files..."
                if [ "$DRY_RUN" = true ]; then
                    echo -e "    ${BLUE}[DRY RUN] Would remove ${part_count} part files${NC}"
                else
                    rm -f "${parts[@]}"
                    echo -e "    ${GREEN}✓ Part files removed${NC}"
                fi
            fi
        fi
        
        echo ""
        
    done <<< "${unique_archives}"
    
    echo -e "  ${GREEN}Finished ${subdir_name}/${NC}"
    echo ""
done

# Print summary
echo "============================================================"
echo "Processing complete!"
echo ""
echo "Total archives found:     ${total_archives}"
echo "Total archives merged:    ${total_merged}"
echo "Total archives extracted: ${total_extracted}"
echo "Total skipped:            ${total_skipped}"
echo "Total errors:             ${total_errors}"
echo "============================================================"

# Exit with error code if there were errors
if [ ${total_errors} -gt 0 ]; then
    exit 1
else
    exit 0
fi
