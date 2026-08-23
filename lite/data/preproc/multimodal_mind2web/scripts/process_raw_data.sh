#!/bin/bash
#
# Decode inline screenshot JPEGs from Multimodal-Mind2Web parquet shards to
# disk so the Python preprocessor can read them as files.
#
# Usage: ./process_raw_data.sh [--include-test] [--dry-run]
#
# By default only the `train` split is decoded. Test splits
# (test_task / test_website / test_domain) are held-out benchmarks and must
# never be trained on; pass --include-test to decode them manually.
#
# Source:  ${CUA_LITE_RAW_DATASETS_ROOT}/osunlp/Multimodal-Mind2Web/data/<split>-*.parquet
# Output:  ${CUA_LITE_RAW_DATASETS_ROOT}/osunlp/Multimodal-Mind2Web/images/<annotation_id>/<step>.jpg
#
# Requires CUA_LITE_RAW_DATASETS_ROOT and the raw parquets downloaded via
# download_raw_data.sh. Idempotent — existing files are skipped.
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

INCLUDE_TEST=false
DRY_RUN=false
while [ $# -gt 0 ]; do
    case "$1" in
        --include-test) INCLUDE_TEST=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--include-test] [--dry-run]"
            echo ""
            echo "Decodes inline screenshot JPEGs from Multimodal-Mind2Web parquet shards into"
            echo "\${CUA_LITE_RAW_DATASETS_ROOT}/osunlp/Multimodal-Mind2Web/images/<annotation_id>/<step>.jpg"
            echo ""
            echo "Default: decode the 'train' split only. Test splits are benchmark holdouts;"
            echo "pass --include-test to decode them manually (never train on them)."
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "${CUA_LITE_RAW_DATASETS_ROOT}" ]; then
    echo -e "${RED}Error: CUA_LITE_RAW_DATASETS_ROOT is not set.${NC}"
    exit 1
fi

BASE="${CUA_LITE_RAW_DATASETS_ROOT}/osunlp/Multimodal-Mind2Web"
if [ ! -d "${BASE}/data" ]; then
    echo -e "${RED}Error: ${BASE}/data does not exist. Run download_raw_data.sh first.${NC}"
    exit 1
fi

echo "============================================================"
echo "Extracting Multimodal-Mind2Web screenshots..."
echo "Source: ${BASE}/data"
echo "Output: ${BASE}/images"
echo "Splits: $( [ "$INCLUDE_TEST" = true ] && echo "train + test_task + test_website + test_domain" || echo "train" )"
echo "============================================================"

if [ "$DRY_RUN" = true ]; then
    echo -e "${BLUE}[DRY RUN] Would decode screenshots under ${BASE}/images${NC}"
    exit 0
fi

# Ensure output files are readable by all users on shared machines.
umask 000

export CUA_LITE_RAW_DATASETS_ROOT
export MM2W_INCLUDE_TEST="${INCLUDE_TEST}"

python3 - <<'PY'
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

base = Path(os.environ["CUA_LITE_RAW_DATASETS_ROOT"]) / "osunlp" / "Multimodal-Mind2Web"
include_test = os.environ.get("MM2W_INCLUDE_TEST", "false").lower() == "true"

splits = ["train"]
if include_test:
    splits += ["test_task", "test_website", "test_domain"]

for split in splits:
    shards = sorted((base / "data").glob(f"{split}-*.parquet"))
    if not shards:
        print(f"  Warning: no shards found for split={split}", file=sys.stderr)
        continue

    print(f"\n--- split={split}: {len(shards)} shard(s) ---")
    written = 0
    skipped = 0
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(
            batch_size=64,
            columns=["annotation_id", "target_action_index", "screenshot"],
        ):
            aids = batch.column("annotation_id").to_pylist()
            idxs = batch.column("target_action_index").to_pylist()
            shots = batch.column("screenshot").to_pylist()
            for aid, idx_str, shot in zip(aids, idxs, shots):
                if not isinstance(aid, str) or not aid:
                    continue
                try:
                    step_idx = int(idx_str)
                except (TypeError, ValueError):
                    continue
                if not isinstance(shot, dict):
                    continue
                img_bytes = shot.get("bytes")
                if not isinstance(img_bytes, (bytes, bytearray)) or not img_bytes:
                    continue

                out_dir = base / "images" / aid
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{step_idx:02d}.jpg"
                if out_path.exists():
                    skipped += 1
                    continue
                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                tmp_path.write_bytes(bytes(img_bytes))
                tmp_path.replace(out_path)
                written += 1

        print(f"  {shard.name}: running total written={written}, skipped={skipped}")

    print(f"  {split}: done — {written} written, {skipped} already existed")

print("\nAll requested splits extracted.")
PY

echo -e "${GREEN}✓ Screenshot extraction complete${NC}"
echo "============================================================"
