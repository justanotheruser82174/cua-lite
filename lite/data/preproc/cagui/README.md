# CAGUI Dataset Preprocessing

CAGUI (`OpenBMB/CAGUI`) is a Chinese-language Android mobile dataset that
contributes both **understanding** (region-conditioned UI-element functional
captioning + OCR) and **use** (multi-step agent demonstrations) data.
This is cua-lite's first Chinese-only dataset (`metadata.others.language="zh"`).
See [`AGENTS.md`](/lite/data/preproc/cagui/AGENTS.md) for the technical spec
(source format, field mapping, action mapping, error handling, output format).

## Environment Setup

```shell
export CUA_LITE_ROOT="/path/to/cua-lite"                              # project root
export CUA_LITE_RAW_DATASETS_ROOT="/path/to/your/huggingface-cache"   # raw datasets dir
export CUA_LITE_DATASETS_ROOT="${CUA_LITE_ROOT}/.data/huggingface"    # canonical output dir
uv sync --extra data
```

## Download raw data

```shell
uv run --no-sync bash lite/data/preproc/cagui/scripts/download_raw_data.sh
```

<details>
<summary>Manual</summary>

```shell
hf download OpenBMB/CAGUI --repo-type dataset \
    --local-dir ${CUA_LITE_RAW_DATASETS_ROOT}/OpenBMB/CAGUI
```

</details>

## Process raw data (verify on-disk layout)

CAGUI is delivered pre-extracted; this step validates the expected paths and
fails unless the corpus counts match `1500 / 1500 / 1500 / 1500 / 600 / 600`
(grounding records/images, episode directories, and matching episode JSON files).
Idempotent.

```shell
uv run --no-sync bash lite/data/preproc/cagui/scripts/process_raw_data.sh
```

## Process data (run the Python preprocessing pipeline)

```shell
uv run --no-sync bash lite/data/preproc/cagui/scripts/process_data.sh --verbose
```

<details>
<summary>Manual</summary>

Process everything:

```shell
uv run python -m lite.data.preproc.cagui.understanding --verbose
uv run python -m lite.data.preproc.cagui.use --verbose
```

Process a single subset:

```shell
uv run python -m lite.data.preproc.cagui.understanding --subset cap --verbose  # functional captions
uv run python -m lite.data.preproc.cagui.understanding --subset ocr --verbose  # UI text OCR
uv run python -m lite.data.preproc.cagui.use --subset domestic --verbose
```

Smoke test (first N records / episodes, no output written):

```shell
uv run python -m lite.data.preproc.cagui.understanding --head 5 --dry-run --verbose
uv run python -m lite.data.preproc.cagui.use --head 5 --dry-run --verbose
```

Full action-code histogram across all 600 episodes (no output written):

```shell
uv run python -m lite.data.preproc.cagui.use --dry-run
```

</details>

## Output

Processed data lands in the canonical local layout (see
[`AGENTS.md`](/lite/data/preproc/cagui/AGENTS.md#2-output-directory-structure)
for the full tree) at `${CUA_LITE_DATASETS_ROOT}/cua-lite/CAGUI/`. Images are
deduplicated into a content-addressed store; each row's `images` column carries
paths relative to `${CUA_LITE_DATASETS_ROOT}`.

## Metadata fields

Every row carries the canonical `metadata` contract
(`metadata_kind`, `dims`, `extra_tool_schemas`, `valid_actions`, `others`) defined in
the parent [`AGENTS.md`](/lite/data/preproc/AGENTS.md#core-dataset-schema).
For CUA rows, platform and task type are derived from `dims`.
The per-row details live in `metadata.others`:

| key | cohorts | meaning |
|-----|---------|---------|
| `id` | all | per-row unique id (`cagui_cap_<n>` / `cagui_ocr_<n>` / `cagui_domestic_<episode_id>`); also the split-hash key |
| `resolution` | all | screenshot `[width, height]` in pixels. use reads it from the source; understanding **recovers** it from `abs_position / rel_position` |
| `os` | all | always `"android"` |
| `source` | all | always `"OpenBMB/CAGUI"` |
| `source_id` | all | upstream key — `"<variant>/<file>.jpeg"` (understanding) or the raw `episode_id` (use) |
| `language` | all | always `"zh"` — CAGUI is cua-lite's first Chinese-only dataset, so downstream filters can select it by language |
| `ui_positions` | **use only** | the upstream per-step accessibility boxes — every on-screen UI element's bounding box. **One list per action step**, aligned 1:1 with `images` / assistant turns; each box is `[x1, y1, x2, y2]` normalized to `[0, 1000]` (upstream gives `[y, x, h, w]` in `[0, 1]`; we reorder + scale + clamp). It is **auxiliary context, not a training target** — the model is not asked to predict it; it records what elements were on screen at each step. Understanding rows do not carry this key. |

Example `use` `ui_positions` (4-step episode → 4 lists; first step has 24
element boxes):

```python
others["ui_positions"][0][:2]   # → [[30, 50, 265, 167], [265, 50, 500, 167]]
len(others["ui_positions"])     # → 4  == len(row["images"])
```

## Verify before pushing

```shell
# 1. Inspect output layout + row counts
uv run python -c "
from datasets import load_dataset
from pathlib import Path
from lite.data.staging import iter_partitions
root = Path('${CUA_LITE_DATASETS_ROOT}/cua-lite/CAGUI')
for *_, f in iter_partitions(root):
    d = load_dataset('parquet', data_files=str(f), split='train')
    print(f'{f.relative_to(root)}: {len(d)} rows')
"

# 2. Smoke-test use export. Run a second command against mobile/understanding
#    when both task types need coverage; --head is pooled, not per cohort.
uv run python -m lite.train.export.export_sft \
    --agent-id qwen3_vl \
    --model-id Qwen/Qwen3-VL-4B-Instruct \
    --data-paths ${CUA_LITE_DATASETS_ROOT}/cua-lite/CAGUI/mobile/use \
    --image-root ${CUA_LITE_DATASETS_ROOT} \
    --head 5 -o /tmp/cagui-sft-smoke.parquet
```

## Publish to HuggingFace

```shell
uv run python -m lite.data.hf.upload CAGUI --org cua-lite --dry-run   # validate plan first
uv run python -m lite.data.hf.upload CAGUI --org cua-lite             # full repo
```
