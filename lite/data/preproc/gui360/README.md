# GUI-360 Dataset Preprocessing

Preprocesses [`vyokky/GUI-360`](https://huggingface.co/datasets/vyokky/GUI-360)
(Windows Office computer-use trajectories) into the canonical cua-lite layout.
For what the data looks like and how it maps, see
[AGENTS.md](/lite/data/preproc/gui360/AGENTS.md).

Produces three desktop cohorts: `use`, `grounding.point`, and
`understanding` (screen parsing).

## Environment Setup

```shell
export CUA_LITE_ROOT="/path/to/cua-lite"                              # project root
export CUA_LITE_RAW_DATASETS_ROOT="/path/to/your/huggingface-cache"   # raw upstream data
export CUA_LITE_DATASETS_ROOT="${CUA_LITE_ROOT}/.data/huggingface"    # canonical output
uv sync --extra data
```

## Download raw data

Downloads only the parts the adapters consume (`grounding_resize`,
`screen_parsing_train_resize`, `train/`) — it skips the ~268 GB `fail/` split and
the unused a11y / action_prediction subsets. Pass `--all` to mirror everything.

```shell
uv run --no-sync bash lite/data/preproc/gui360/scripts/download_raw_data.sh
```

## Process raw data (extract image archives)

Extracts the processed-subset image tars and the `train/image.tar.gz`
(~25 GB → `use` screenshots). Idempotent; skips already-extracted dirs.

```shell
uv run --no-sync bash lite/data/preproc/gui360/scripts/process_raw_data.sh
```

## Process data (run the Python pipeline)

```shell
uv run --no-sync bash lite/data/preproc/gui360/scripts/process_data.sh --verbose
```

<details>
<summary>Manual / per-cohort</summary>

```shell
uv run python lite/data/preproc/gui360/grounding-point.py --verbose
uv run python lite/data/preproc/gui360/understanding.py --verbose
uv run python lite/data/preproc/gui360/use.py --verbose
# Each script also takes --head N for a quick smoke test and --dry-run.
```

</details>

## Output

Processed data lands at `${CUA_LITE_DATASETS_ROOT}/cua-lite/GUI-360/`. See
[AGENTS.md — Output Directory Structure](/lite/data/preproc/gui360/AGENTS.md#2-output-directory-structure).

## Verification

```shell
# Count rows per partition (skip the image store)
uv run python -c "
from pathlib import Path
from datasets import load_dataset
from lite.data.staging import iter_partitions
root = Path('${CUA_LITE_DATASETS_ROOT}/cua-lite/GUI-360')
for *_, f in iter_partitions(root):
    d = load_dataset('parquet', data_files=str(f), split='train')
    print(f'{f.relative_to(root)}: {len(d)} rows')
"

# Inspect a row
uv run python -c "
from datasets import load_dataset
p = '${CUA_LITE_DATASETS_ROOT}/cua-lite/GUI-360/desktop/use/train/use.parquet'
d = load_dataset('parquet', data_files=p, split='train')
print(d[0])
"
```

Smoke-test SFT export (renders messages + serializes images):

```shell
uv run python -m lite.train.export.export_sft \
    --agent-id qwen3_vl \
    --model-id Qwen/Qwen3-VL-4B-Instruct \
    --data-paths ${CUA_LITE_DATASETS_ROOT}/cua-lite/GUI-360/desktop/use \
    --image-root ${CUA_LITE_DATASETS_ROOT} \
    --head 5 -o /tmp/gui360-sft-smoke.parquet
```

## Publishing to HuggingFace and round-tripping

```shell
uv run python -m lite.data.hf.upload GUI-360 --org cua-lite --dry-run   # validate the plan
uv run python -m lite.data.hf.upload GUI-360 --org cua-lite            # push
```

Pull a published dataset back into the canonical local layout (e.g. on a fresh
machine without the raw upstream):

```shell
uv run python -m lite.data.hf.download GUI-360 \
    --out ${CUA_LITE_DATASETS_ROOT}/cua-lite/GUI-360
```

The downloaded layout uses the same canonical local schema as raw-data preproc
output, so `lite.train.export.export_sft` consumes either source identically.
