# OpenCUA (AgentNet) Dataset Preprocessing

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
uv run --no-sync bash lite/data/preproc/opencua/scripts/download_raw_data.sh
```

<details>
<summary>Manual</summary>

```shell
hf download xlangai/AgentNet --repo-type dataset --local-dir ${CUA_LITE_RAW_DATASETS_ROOT}/xlangai/AgentNet
```

</details>

## Process raw data (merge and extract split image archives)

**Script:**

```shell
uv run --no-sync bash lite/data/preproc/opencua/scripts/process_raw_data.sh
```

The script validates any existing merged zip, rebuilds an invalid/interrupted
one through a same-directory temporary archive, and writes its extraction
marker only after `unzip` succeeds.

<details>
<summary>Manual</summary>

```shell
# For Ubuntu images
cd ${CUA_LITE_RAW_DATASETS_ROOT}/xlangai/AgentNet/ubuntu_images
zip -s 0 images.zip --out images-full.zip
unzip images-full.zip -d .
cd -

# For Windows/Mac images
cd ${CUA_LITE_RAW_DATASETS_ROOT}/xlangai/AgentNet/win_mac_images
zip -s 0 images.zip --out images-full.zip
unzip images-full.zip -d .
cd -
```

</details>

## Process data (run Python preprocessing pipeline)

**Script:**

```shell
uv run --no-sync bash lite/data/preproc/opencua/scripts/process_data.sh --verbose
```

<details>
<summary>Manual</summary>

```shell
uv run python lite/data/preproc/opencua/use.py --verbose

# Dry run (preview without processing):
uv run python lite/data/preproc/opencua/use.py --dry-run
```

`use.py` takes `--verbose`, `--dry-run`, and `--head N`; use `--head` for a
bounded smoke run before processing all trajectories (ubuntu 5k + win_mac 18k).

</details>

## Output

Processed data is written to `${CUA_LITE_DATASETS_ROOT}/cua-lite/OpenCUA/`. See [AGENTS.md — Output Directory Structure](/lite/data/preproc/opencua/AGENTS.md#output-directory-structure) for the full layout.

## Verification

```shell
# Count rows in each partition
uv run python -c "
import json, pyarrow.parquet as pq
from pathlib import Path
from lite.data.staging import iter_partitions
root = Path('${CUA_LITE_DATASETS_ROOT}/cua-lite/OpenCUA')
for *_, f in iter_partitions(root):
    t = pq.read_table(f)
    print(f'{f.relative_to(root)}: {t.num_rows} rows')
"
```

Smoke-test SFT export (`--model-id` is required — steps are tokenized at
export):

```shell
uv run python -m lite.train.export.export_sft \
    --agent-id qwen3_vl \
    --model-id Qwen/Qwen3-VL-4B-Instruct \
    --data-paths ${CUA_LITE_DATASETS_ROOT}/cua-lite/OpenCUA/desktop/use \
    --image-root ${CUA_LITE_DATASETS_ROOT} \
    --head 5 -o /tmp/opencua-sft-smoke.parquet
```

## Publish to HuggingFace

```shell
uv run python -m lite.data.hf.upload OpenCUA --org cua-lite --dry-run   # validate plan first
uv run python -m lite.data.hf.upload OpenCUA --org cua-lite -v          # push
```
