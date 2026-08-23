# GUIAct Preprocessing

Preprocess [`yiye2023/GUIAct`](https://huggingface.co/datasets/yiye2023/GUIAct)
into the canonical cua-lite SFT layout. For the data specification (source
format, action mapping, output schema, statistics) see
[`AGENTS.md`](/lite/data/preproc/guiact/AGENTS.md). This file is the how-to.

GUIAct ships three subsets, each mapped to one `(platform, task_type)` cohort:

| Subset | Platform | Task type | Shape |
|--------|----------|-----------|-------|
| web-single | `browser` | `grounding.action` | single screenshot → one action |
| web-multi | `browser` | `use` | multi-step web episode |
| smartphone | `mobile` | `use` | multi-step Android episode |

## 1. Environment

```bash
export CUA_LITE_RAW_DATASETS_ROOT=/path/to/raw         # contains yiye2023/GUIAct/
export CUA_LITE_DATASETS_ROOT=/path/to/processed       # canonical output root
uv sync --extra data
```

## 2. Download raw data

```bash
uv run --no-sync bash lite/data/preproc/guiact/scripts/download_raw_data.sh
```

## 3. Extract images

GUIAct stores screenshots as base64 inside `*_images.parquet`. Decode them first;
web-multi uses `images/web-multi/{train,test}/` while the other subsets use one
directory each.

```bash
uv run --no-sync bash lite/data/preproc/guiact/scripts/process_raw_data.sh --subset all
```

## 4. Run preprocessing

```bash
# web-single grounding.action, then web-multi + smartphone use
uv run python lite/data/preproc/guiact/grounding-action.py
uv run python lite/data/preproc/guiact/use.py

# or run end-to-end
uv run --no-sync bash lite/data/preproc/guiact/scripts/process_data.sh
```

Useful flags (both scripts): `--dry-run`, `--verbose`, `--head N`.
`use.py` additionally takes `--subset web_multi|smartphone|all`.

Output lands under `${CUA_LITE_DATASETS_ROOT}/cua-lite/GUIAct/` — see
[`AGENTS.md` §Output Directory Structure](/lite/data/preproc/guiact/AGENTS.md).

## 5. Verify

```bash
uv run python -c "
from pathlib import Path
import pyarrow.parquet as pq
from lite.data.staging import iter_partitions
root = Path('${CUA_LITE_DATASETS_ROOT}/cua-lite/GUIAct')
for *_, f in iter_partitions(root):
    print(f'{f.relative_to(root)}: {pq.ParquetFile(f).metadata.num_rows} rows')
"
```

Smoke-test SFT export (see [`lite/data/README.md`](/lite/data/README.md)):

```bash
uv run python -m lite.train.export.export_sft \
    --agent-id qwen3_vl \
    --model-id Qwen/Qwen3-VL-4B-Instruct \
    --data-paths ${CUA_LITE_DATASETS_ROOT}/cua-lite/GUIAct/browser/grounding.action \
    --image-root ${CUA_LITE_DATASETS_ROOT} \
    --head 5 -o /tmp/guiact-sft-smoke.parquet
```

## 6. Publish to HF

```bash
uv run python -m lite.data.hf.upload GUIAct --org cua-lite --dry-run   # validate plan
uv run python -m lite.data.hf.upload GUIAct --org cua-lite             # push
```
