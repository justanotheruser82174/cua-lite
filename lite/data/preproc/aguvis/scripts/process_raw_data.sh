#!/usr/bin/env bash
set -eo pipefail
umask 000

usage() {
    echo "Usage: $0 [--dry-run] [-h|--help]"
    echo ""
    echo "Extract Aguvis image archives. Idempotent — skips already-extracted zips."
    echo "All zips contain <name>/images/<file> structure."
    exit 0
}

DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${CUA_LITE_RAW_DATASETS_ROOT:-}" ]]; then
    echo "ERROR: CUA_LITE_RAW_DATASETS_ROOT is not set." >&2
    exit 1
fi

S1="${CUA_LITE_RAW_DATASETS_ROOT}/xlangai/aguvis-stage1"
S2="${CUA_LITE_RAW_DATASETS_ROOT}/xlangai/aguvis-stage2"

extract_zip() {
    local zip_path="$1"
    local dest_dir="$2"
    local name
    name=$(basename "$zip_path" .zip)

    if [[ ! -f "$zip_path" ]]; then
        echo "  SKIP (not downloaded): $zip_path"
        return
    fi

    local marker="${zip_path}.extracted"
    if [[ -f "$marker" ]]; then
        local count
        count=$(ls -1 "${dest_dir}/${name}/images" 2>/dev/null | wc -l)
        echo "  SKIP (complete, $count files): ${dest_dir}/${name}/images/"
        return
    fi

    echo "  Extracting $(basename "$zip_path") → ${dest_dir}/"
    if $DRY_RUN; then
        echo "    [dry-run] unzip -q -o $zip_path -d $dest_dir"
    else
        unzip -q -o "$zip_path" -d "$dest_dir"
        touch "$marker"
        local count
        count=$(ls -1 "${dest_dir}/${name}/images" 2>/dev/null | wc -l)
        echo "    Done: $count files in ${dest_dir}/${name}/images/"
    fi
}

echo "=== Stage 1 Archives ==="
if [[ -d "$S1" ]]; then
    for zip in "$S1"/*.zip; do
        [[ -f "$zip" ]] || continue
        extract_zip "$zip" "$S1"
    done

    # SeeClick uses split tar.gz parts
    if [[ -f "$S1/seeclick.tar.gz.part_00" ]] && [[ ! -f "$S1/seeclick.tar.gz.extracted" ]]; then
        echo "  Reconstructing seeclick.tar.gz from split parts..."
        if $DRY_RUN; then
            echo "    [dry-run] cat seeclick.tar.gz.part_* | tar --no-same-owner --no-same-permissions --touch --no-overwrite-dir -xzf - -C $S1"
        else
            cat "$S1"/seeclick.tar.gz.part_* | tar \
                --no-same-owner --no-same-permissions --touch --no-overwrite-dir \
                -xzf - -C "$S1"
            touch "$S1/seeclick.tar.gz.extracted"
            echo "    Done: $(ls -1 "$S1/seeclick/images" 2>/dev/null | wc -l) files in $S1/seeclick/images/"
        fi
    elif [[ -f "$S1/seeclick.tar.gz.extracted" ]]; then
        echo "  SKIP (complete): $S1/seeclick/images/"
    fi
else
    echo "  WARNING: Stage 1 directory not found: $S1"
fi

echo ""
echo "=== Stage 2 Archives ==="
if [[ -d "$S2" ]]; then
    for zip in "$S2"/*.zip; do
        [[ -f "$zip" ]] || continue
        name=$(basename "$zip" .zip)
        # miniwob.zip has no wrapper dir — extract into miniwob/
        if [[ "$name" == "miniwob" ]]; then
            if [[ -f "$zip.extracted" ]]; then
                echo "  SKIP (complete, $(ls -1 "$S2/miniwob/images" 2>/dev/null | wc -l) files): $S2/miniwob/images/"
            else
                echo "  Extracting miniwob.zip → $S2/miniwob/"
                if $DRY_RUN; then
                    echo "    [dry-run] unzip -q -o $zip -d $S2/miniwob"
                else
                    unzip -q -o "$zip" -d "$S2/miniwob"
                    touch "$zip.extracted"
                    echo "    Done"
                fi
            fi
        else
            extract_zip "$zip" "$S2"
        fi
    done
else
    echo "  WARNING: Stage 2 directory not found: $S2"
fi

echo ""
echo "=== Verification ==="
echo "Stage 1 image directories:"
for d in "$S1"/*/images; do
    [[ -d "$d" ]] && echo "  $d: $(ls -1 "$d" | wc -l) files"
done
echo "Stage 2 image directories:"
for d in "$S2"/*/images "$S2"/*/*; do
    [[ -d "$d" ]] && echo "  $d: $(ls -1 "$d" | wc -l) files"
done

echo ""
echo "Done."
