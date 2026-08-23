# ScaleCUA Dataset Preprocessing

## Environment Setup

```shell
export CUA_LITE_ROOT="/path/to/cua-lite"                            # project root
export CUA_LITE_RAW_DATASETS_ROOT="/path/to/your/huggingface-cache" # shared base directory for raw datasets
export CUA_LITE_DATASETS_ROOT="${CUA_LITE_ROOT}/.data/huggingface"                      # output directory for processed datasets
uv sync --extra data
```

## Download raw data

**Script:**

```shell
uv run --no-sync bash lite/data/preproc/scalecua/scripts/download_raw_data.sh
```

<details>
<summary>Manual</summary>

```shell
hf download OpenGVLab/ScaleCUA-Data --repo-type dataset --local-dir ${CUA_LITE_RAW_DATASETS_ROOT}/OpenGVLab/ScaleCUA-Data
hf download zyliu/ScaleCUA-Data-Understanding --repo-type dataset --local-dir ${CUA_LITE_RAW_DATASETS_ROOT}/zyliu/ScaleCUA-Data-Understanding
```

</details>

## Process raw data (merge and extract split archives)

**Script:** Run for each dataset that uses split archives:

```shell
uv run --no-sync bash lite/data/preproc/scalecua/scripts/process_raw_data.sh --keep-parts --dataset-dir "${CUA_LITE_RAW_DATASETS_ROOT}/OpenGVLab/ScaleCUA-Data"
uv run --no-sync bash lite/data/preproc/scalecua/scripts/process_raw_data.sh --keep-parts --dataset-dir "${CUA_LITE_RAW_DATASETS_ROOT}/zyliu/ScaleCUA-Data-Understanding"
```

## Process data (run Python preprocessing pipeline)

**Script:**

```shell
uv run --no-sync bash lite/data/preproc/scalecua/scripts/process_data.sh --verbose
```

<details>
<summary>Manual</summary>

```shell
uv run python lite/data/preproc/scalecua/understanding.py --verbose
uv run python lite/data/preproc/scalecua/grounding-action.py --verbose
uv run python lite/data/preproc/scalecua/grounding-point.py --verbose
uv run python lite/data/preproc/scalecua/grounding-bbox.py --verbose
uv run python lite/data/preproc/scalecua/use.py --verbose
```

All five take `--dry-run`, `--verbose`, and `--head N` for bounded smoke runs.
`use.py` also takes `--head-entries N`; `--head` bounds complete trajectories,
while `--head-entries` bounds raw entries per subset.

</details>

## Output

Processed data is written to `${CUA_LITE_DATASETS_ROOT}/cua-lite/ScaleCUA/`. See [AGENTS.md — Output Directory Structure](/lite/data/preproc/scalecua/AGENTS.md#output-directory-structure) for the full layout.

## Verification

```shell
# Count rows per partition (skip the image store)
uv run python -c "
from pathlib import Path
from datasets import load_dataset
from lite.data.staging import iter_partitions
root = Path('${CUA_LITE_DATASETS_ROOT}/cua-lite/ScaleCUA')
for *_, f in iter_partitions(root):
    d = load_dataset('parquet', data_files=str(f), split='train')
    print(f'{f.relative_to(root)}: {len(d)} rows')
"

# Inspect a row
uv run python -c "
from datasets import load_dataset
p = '${CUA_LITE_DATASETS_ROOT}/cua-lite/ScaleCUA/desktop/understanding/train/caption.parquet'
d = load_dataset('parquet', data_files=p, split='train')
print(d[0])
"
```

Smoke-test SFT export (`--model-id` is required — steps are tokenized at
export):

```shell
uv run python -m lite.train.export.export_sft \
    --agent-id qwen3_vl \
    --model-id Qwen/Qwen3-VL-4B-Instruct \
    --data-paths ${CUA_LITE_DATASETS_ROOT}/cua-lite/ScaleCUA/desktop/use \
    --image-root ${CUA_LITE_DATASETS_ROOT} \
    --head 5 -o /tmp/scalecua-sft-smoke.parquet
```

## Publishing to HuggingFace and round-tripping

After the local preproc finishes, push to `cua-lite/ScaleCUA`:

```shell
uv run python -m lite.data.hf.upload ScaleCUA --org cua-lite --dry-run   # validate plan first
uv run python -m lite.data.hf.upload ScaleCUA --org cua-lite             # push
```

To pull a published dataset back into the canonical local layout (e.g. on a fresh machine without the raw upstream):

```shell
uv run python -m lite.data.hf.download ScaleCUA \
    --out ${CUA_LITE_DATASETS_ROOT}/cua-lite/ScaleCUA
```

The downloaded layout uses the same canonical local schema as raw-data preproc output, so `lite.train.export.export_sft` consumes either source identically.
