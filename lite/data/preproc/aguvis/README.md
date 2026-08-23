# Aguvis — preprocessing how-to

cua-lite adapter for [`xlangai/aguvis-stage1`](https://huggingface.co/datasets/xlangai/aguvis-stage1)
+ [`xlangai/aguvis-stage2`](https://huggingface.co/datasets/xlangai/aguvis-stage2),
a large composite PyAutoGUI-format GUI dataset. Emits `grounding.action` (Stage 1)
and `use` (Stage 2) into one `cua-lite/Aguvis` dataset. For the format/mapping
spec see [`AGENTS.md`](/lite/data/preproc/aguvis/AGENTS.md).

## Environment

```bash
export CUA_LITE_ROOT="/path/to/cua-lite"                    # project root (used just below)
export CUA_LITE_RAW_DATASETS_ROOT=/path/to/raw-data-mount   # holds xlangai/aguvis-stage1, -stage2
export CUA_LITE_DATASETS_ROOT="${CUA_LITE_ROOT}/.data/huggingface"
uv sync --extra data
```

## 1. Download raw data

```bash
uv run --no-sync bash lite/data/preproc/aguvis/scripts/download_raw_data.sh [--stage 1|2|all] [--dry-run]
```

Downloads `xlangai/aguvis-stage1` and `xlangai/aguvis-stage2` into
`${CUA_LITE_RAW_DATASETS_ROOT}/xlangai/`. `--stage` (default `all`) limits the
download to one stage.

## 2. Extract image archives

```bash
uv run --no-sync bash lite/data/preproc/aguvis/scripts/process_raw_data.sh [--dry-run]
```

Extracts the per-sub-dataset image zips (each contains `<name>/images/<file>`).

## 3. Run preprocessing

```bash
# both stages, all sub-datasets
uv run --no-sync bash lite/data/preproc/aguvis/scripts/process_data.sh --verbose

# or individually / per sub-dataset
uv run python lite/data/preproc/aguvis/grounding-action.py [--subset seeclick] [--head N] [--dry-run]
uv run python lite/data/preproc/aguvis/use.py       [--subset coat]     [--head N] [--dry-run]
```

`--head N` caps records (grounding) / episodes (use) per sub-dataset. Output
lands under `${CUA_LITE_DATASETS_ROOT}/cua-lite/Aguvis/` (layout in
[`AGENTS.md`](/lite/data/preproc/aguvis/AGENTS.md#2-output-directory-structure)).

## 4. Verify

```bash
uv run python -c "
from pathlib import Path
from datasets import load_dataset
from lite.data.staging import iter_partitions
root = Path('${CUA_LITE_DATASETS_ROOT}/cua-lite/Aguvis')
for *_, f in iter_partitions(root):
    d = load_dataset('parquet', data_files=str(f), split='train')
    print(f'{f.relative_to(root)}: {len(d)} rows')
"

# SFT export smoke test for use (--model-id is required: steps are tokenized at export).
# Run a second command against mobile/grounding.action to cover that cohort too.
uv run python -m lite.train.export.export_sft \
    --agent-id qwen3_vl \
    --model-id Qwen/Qwen3-VL-4B-Instruct \
    --data-paths ${CUA_LITE_DATASETS_ROOT}/cua-lite/Aguvis/mobile/use \
    --image-root ${CUA_LITE_DATASETS_ROOT} \
    --head 5 -o /tmp/aguvis-sft-smoke.parquet
```

## 5. Publish to HuggingFace

```bash
uv run python -m lite.data.hf.upload Aguvis --org cua-lite --dry-run   # validate plan
uv run python -m lite.data.hf.upload Aguvis --org cua-lite             # push
```
