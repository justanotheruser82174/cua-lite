# Multimodal-Mind2Web — preprocessing how-to

cua-lite adapter for [`osunlp/Multimodal-Mind2Web`](https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web),
the screenshot-augmented Mind2Web web-agent benchmark. The upstream corpus spans
136 websites across all splits; this adapter emits one `use` cohort built only
from the **1,009 `train` episodes** (73 websites):

    cua-lite/Multimodal-Mind2Web/browser/use/<split>/use.parquet

**Training-safe scope.** Only the upstream `train` split is processed. The
`test_task` / `test_website` / `test_domain` splits are the standard Mind2Web
held-out benchmarks and must NEVER be trained on — this adapter never reads them.
`train` is hash-split into `train` / `validation` (no upstream-split label is
written). For the format/mapping spec see [`AGENTS.md`](/lite/data/preproc/multimodal_mind2web/AGENTS.md).

## Environment

```bash
export CUA_LITE_RAW_DATASETS_ROOT=/path/to/raw         # holds osunlp/Multimodal-Mind2Web/
export CUA_LITE_DATASETS_ROOT=/path/to/processed       # canonical output root
uv sync --extra data
```

## 1. Download raw data

```bash
uv run --no-sync bash lite/data/preproc/multimodal_mind2web/scripts/download_raw_data.sh
```

Downloads `osunlp/Multimodal-Mind2Web` into
`${CUA_LITE_RAW_DATASETS_ROOT}/osunlp/Multimodal-Mind2Web/`.

## 2. Extract screenshots

Mind2Web stores screenshots inline (as bytes) in the parquet shards. Decode the
`train` split to disk first, under
`osunlp/Multimodal-Mind2Web/images/<annotation_id>/<step>.jpg`:

```bash
uv run --no-sync bash lite/data/preproc/multimodal_mind2web/scripts/process_raw_data.sh
```

By default only `train` is decoded. The script is idempotent (existing files are
skipped). `--include-test` decodes the benchmark holdouts manually (never train
on them); `--dry-run` previews.

## 3. Run preprocessing

```bash
# train split only
uv run --no-sync bash lite/data/preproc/multimodal_mind2web/scripts/process_data.sh --verbose

# or directly
uv run python -m lite.data.preproc.multimodal_mind2web.use [--verbose] [--head N] [--dry-run]
```

`--dry-run` reports the episode count without writing. `--head N` caps episodes
for a smoke test. Output lands under
`${CUA_LITE_DATASETS_ROOT}/cua-lite/Multimodal-Mind2Web/` (layout in
[`AGENTS.md`](/lite/data/preproc/multimodal_mind2web/AGENTS.md#2-output-directory-structure)).

## 4. Verify

```bash
uv run python -c "
from pathlib import Path
import pyarrow.parquet as pq
from lite.data.staging import iter_partitions
root = Path('${CUA_LITE_DATASETS_ROOT}/cua-lite/Multimodal-Mind2Web')
for *_, f in iter_partitions(root):
    print(f'{f.relative_to(root)}: {pq.ParquetFile(f).metadata.num_rows} rows')
"
```

Smoke-test SFT export (see [`lite/data/README.md`](/lite/data/README.md)):

```bash
uv run python -m lite.train.export.export_sft \
    --agent-id qwen3_vl \
    --model-id Qwen/Qwen3-VL-4B-Instruct \
    --data-paths ${CUA_LITE_DATASETS_ROOT}/cua-lite/Multimodal-Mind2Web/browser/use \
    --image-root ${CUA_LITE_DATASETS_ROOT} \
    --head 5 -o /tmp/mind2web-sft-smoke.parquet
```

## 5. Publish to HuggingFace

```bash
uv run python -m lite.data.hf.upload Multimodal-Mind2Web --org cua-lite --dry-run   # validate plan
uv run python -m lite.data.hf.upload Multimodal-Mind2Web --org cua-lite             # push
```
