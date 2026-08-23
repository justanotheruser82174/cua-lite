#!/bin/bash
#
# Extract base64-encoded images from GUIAct parquet files to disk.
# Usage: ./process_raw_data.sh [--subset web-single|web-multi|smartphone|all] [--dry-run]
#
# GUIAct stores screenshots as base64 strings in *_train_images.parquet.
# web-multi is split-scoped because train/test reuse two IDs for different
# screenshots; web-single and smartphone keep their shared subset directory.
#
# Requires: CUA_LITE_RAW_DATASETS_ROOT to be set and the raw parquets downloaded
# via download_raw_data.sh.
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

SUBSET="all"
DRY_RUN=false
while [ $# -gt 0 ]; do
    case "$1" in
        --subset) SUBSET="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--subset web-single|web-multi|smartphone|all] [--dry-run]"
            echo ""
            echo "Decodes base64 images from *_train_images.parquet into yiye2023/GUIAct/images/."
            echo "Requires CUA_LITE_RAW_DATASETS_ROOT to be set."
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "${CUA_LITE_RAW_DATASETS_ROOT}" ]; then
    echo -e "${RED}Error: CUA_LITE_RAW_DATASETS_ROOT is not set.${NC}"
    exit 1
fi

BASE="${CUA_LITE_RAW_DATASETS_ROOT}/yiye2023/GUIAct"
if [ ! -d "${BASE}" ]; then
    echo -e "${RED}Error: ${BASE} does not exist. Run download_raw_data.sh first.${NC}"
    exit 1
fi

echo "============================================================"
echo "Extracting GUIAct images from base64 parquet..."
echo "Source: ${BASE}"
echo "Subset: ${SUBSET}"
echo "============================================================"

if [ "$DRY_RUN" = true ]; then
    echo -e "${BLUE}[DRY RUN] Would decode base64 images for subset=${SUBSET}${NC}"
    exit 0
fi

# Use umask 000 so extracted images are accessible to all users
umask 000

# Inline Python performs the base64 decoding (memory-safe streaming via pyarrow).
python3 - "${BASE}" "${SUBSET}" <<'PY'
import base64
import sys
from pathlib import Path

import pyarrow.parquet as pq

base = Path(sys.argv[1])
subset_arg = sys.argv[2]

# web-multi train/test are separate namespaces; the other subsets intentionally
# keep one namespace (web-single reuses identical screenshots across splits).
SUBSETS = {
    "web-single": ([
        ("web-single_train_images.parquet", None),
        ("web-single_test_images.parquet", None),
    ], "png"),
    "web-multi":  ([
        ("web-multi_train_images.parquet", "train"),
        ("web-multi_test_images.parquet", "test"),
    ], "jpg"),
    "smartphone": ([
        ("smartphone_train_images.parquet", None),
        ("smartphone_test_images.parquet", None),
    ], "jpg"),
}


def extract_parquet(parquet_path, output_dir, ext):
    if not parquet_path.exists():
        print(f"  Warning: {parquet_path} not found, skipping")
        return
    pf = pq.ParquetFile(parquet_path)
    total = pf.metadata.num_rows
    print(f"\n--- {parquet_path.name} ({total} rows) ---")

    written = 0
    skipped = 0
    for batch in pf.iter_batches(batch_size=200):
        # Locate the index column (image_id)
        index_col = None
        for col_name in ("__index_level_0__", "image_id"):
            if col_name in batch.schema.names:
                index_col = col_name
                break
        if index_col is None:
            # Fall back to pandas for index
            import pandas as pd
            df = pq.read_table(parquet_path).to_pandas()
            for image_id, row in df.iterrows():
                out_path = output_dir / f"{image_id}.{ext}"
                if out_path.exists():
                    skipped += 1
                    continue
                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                tmp_path.write_bytes(base64.b64decode(row["base64"]))
                tmp_path.replace(out_path)
                written += 1
            break

        image_ids = batch[index_col].to_pylist()
        b64_data = batch["base64"].to_pylist()
        for image_id, b64_str in zip(image_ids, b64_data):
            out_path = output_dir / f"{image_id}.{ext}"
            if out_path.exists():
                skipped += 1
                continue
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp_path.write_bytes(base64.b64decode(b64_str))
            tmp_path.replace(out_path)
            written += 1

        if (written + skipped) % 2000 == 0:
            print(f"    Progress: {written} written, {skipped} skipped / {total}")

    print(f"  Done: {written} written, {skipped} already existed")


subsets = list(SUBSETS.keys()) if subset_arg == "all" else [subset_arg]

for subset in subsets:
    parquet_names, ext = SUBSETS[subset]
    for parquet_name, split in parquet_names:
        output_dir = base / "images" / subset
        if split:
            output_dir /= split
        output_dir.mkdir(parents=True, exist_ok=True)
        extract_parquet(base / parquet_name, output_dir, ext)

print("\nAll subsets extracted.")
PY

echo -e "${GREEN}✓ Image extraction complete${NC}"
echo "============================================================"
