# `lite.data` — datasets for cua-lite SFT

Raw preprocessing, rollout staging, and Hub download all produce the **same** local layout; downstream consumers don't care which path was used. This README covers those ingestion paths plus the export step that hands data to the trainer.

```
lite/data/
├── staging.py           # canonical local-format helpers (ImageStore, SplitAssigner, partition_path, …)
├── load.py              # discover_files_under_paths, load_file_as_dataset (used by export_sft)
├── split.py             # CLI: split a parquet into train/eval
├── merge.py             # CLI: concatenate parquet/jsonl files into one
├── utils/               # data-wide row/message helpers
├── hf/
│   ├── upload.py        # canonical local layout → cua-lite/<Name> on HF Hub
│   ├── download.py      # cua-lite/<Name> on HF Hub → canonical local layout
│   ├── stage.py         # rollout log-roots → canonical (publish a rollout run)
│   ├── unstage.py       # canonical → rollout log-root (resume / continue collecting)
│   └── card.py          # render the canonical README + configs YAML at push time
└── preproc/
    └── <dataset>/       # per-dataset adapters (raw → canonical) — see preproc/AGENTS.md
```

For the format spec and the contract that adapters must satisfy, see [`preproc/AGENTS.md`](/lite/data/preproc/AGENTS.md). This README is the **how-to-use** doc.

## Environment

Canonical ingestion paths and the SFT export consume these env vars:

```bash
export CUA_LITE_ROOT="/path/to/cua-lite"                        # project root (used just below)
export CUA_LITE_RAW_DATASETS_ROOT=/path/to/raw-data-mount       # only needed for path A
export CUA_LITE_DATASETS_ROOT="${CUA_LITE_ROOT}/.data/huggingface"
```

`CUA_LITE_DATASETS_ROOT` is where canonical local datasets land (and where SFT export reads from). It mirrors HF's org/repo naming: `${CUA_LITE_DATASETS_ROOT}/cua-lite/<DatasetName>/`.

---

## Path A — process from raw upstream data

Run the per-dataset preproc adapters end to end. Suitable when you have the upstream raw data on disk (NFS / local mount); produces a fresh canonical dataset directly. Each dataset under [`preproc/<name>/`](/lite/data/preproc) ships its own scripts.

ScaleCUA is the reference dataset; the same pattern applies to others:

```bash
# 1. Snapshot the upstream sources from HF
uv run --no-sync bash lite/data/preproc/scalecua/scripts/download_raw_data.sh

# 2. Unpack split tar.gz archives, decode raw images
uv run --no-sync bash lite/data/preproc/scalecua/scripts/process_raw_data.sh \
    --dataset-dir "${CUA_LITE_RAW_DATASETS_ROOT}/OpenGVLab/ScaleCUA-Data"

# 3. Run the Python preproc pipeline (per task type, sequential)
uv run --no-sync bash lite/data/preproc/scalecua/scripts/process_data.sh --verbose
```

Output lands at `${CUA_LITE_DATASETS_ROOT}/cua-lite/ScaleCUA/`:

```
cua-lite/ScaleCUA/
├── images/<hash[:2]>/<hash>.<ext>                # content-addressed image store
├── desktop/
│   ├── grounding.action/<split>/<variant>.parquet
│   ├── grounding.bbox/<split>/<variant>.parquet
│   ├── grounding.point/<split>/<variant>.parquet
│   ├── use/<split>/<variant>.parquet
│   └── understanding/<split>/{caption,screen_transition,user_intention}.parquet
├── browser/ …  (same shape)
└── mobile/  …  (same shape)
```

Each row carries `images: list[str]` paths into the local image store; bytes are not embedded.

---

## Path A2 — stage a rollout run

Use [`lite.data.hf.stage`](/lite/data/hf/stage.py) to absorb rollout log-roots into the canonical local layout before upload:

```bash
uv run python -m lite.data.hf.stage \
    --log-roots .logs/rollout/<run> \
    --name <DatasetName>
```

`stage` has **no hidden default row filter**: without `--filter`, it stages every row. Filter for success or quality before staging, or pass an explicit `--filter "lambda m: ..."` when staging. The published dataset keeps the shareable message/image layout and omits runtime-only attachments.

Image paths are copied into the content-addressed image store in row order. Message image references are stable foreign keys into the row's `images` list, so `{"type": "image", "index": N}` references are never reindexed or renumbered during stage, upload, or download.

---

## Path B — pull from HuggingFace

Mirror an already-published `cua-lite/<Name>` dataset back to the canonical local layout. Suitable when you don't have the upstream raw data, or when you want a known-good copy that someone else already preprocessed.

```bash
uv run python -m lite.data.hf.download ScaleCUA \
    --out ${CUA_LITE_DATASETS_ROOT}/cua-lite/ScaleCUA

# Pull only one cohort (faster — useful for smoke tests)
uv run python -m lite.data.hf.download ScaleCUA \
    --out ${CUA_LITE_DATASETS_ROOT}/cua-lite/ScaleCUA-mobile-grounding-point \
    --allow-patterns "mobile/grounding.point/**"
```

The download tool snapshots the repo, walks parquets (merging shard groups), extracts every embedded image's bytes back into a content-addressed store, and rewrites each row's `images` column to local rel-paths. The output is byte-identical to what Path A would have produced for the same data — same image hashes, same row contents.

---

## Path C — load directly via `datasets`

For ad-hoc inspection / scripting that doesn't need the full local layout. The HF Hub repo is structured so `datasets.load_dataset` works out of the box:

```python
from datasets import load_dataset

# Whole dataset (every cohort merged into one stream)
ds = load_dataset("cua-lite/ScaleCUA")

# Just a single platform
ds = load_dataset("cua-lite/ScaleCUA", "mobile")

# Just one (platform, task_type) cohort — '.' separates platform from task_type
# (the agent registry key in code uses '@' instead — `qwen3_vl@mobile@grounding.point` —
# but HF config_name uses '.' so the dataset-viewer's signed image URLs work)
ds = load_dataset("cua-lite/ScaleCUA", "mobile.grounding.point", split="validation")
```

Rows from Path C carry `images: list[Image]` (PIL-decoded by the `datasets` library) instead of the local form's `images: list[str]`. Use Path B if you need the disk-form for SFT export.

---

## Consuming with `export_sft`

[`lite/train/export/export_sft.py`](/lite/train/export/export_sft.py) reads local dataset layouts (Path A or B). For one-off local distillation it can also read rollout log roots; use staged or downloaded datasets for publish/share flows. It produces a model-ready SFT parquet: PNG-serialized `processed_images` plus pre-tokenized `LiteRLStep` structs in `steps`, ready for the trainer.

```bash
uv run python -m lite.train.export.export_sft \
    --config scripts/configs/qwen3_vl/recipes/sft/default.yaml \
    --model-id Qwen/Qwen3-VL-4B-Instruct \
    --data-paths "${CUA_LITE_DATASETS_ROOT}/cua-lite/ScaleCUA" \
    --image-root "${CUA_LITE_DATASETS_ROOT}" \
    -o /tmp/scalecua-qwen3vl.parquet
```

For data collected by a cua-lite rollout, prefer that env's **rollout** config
(`scripts/configs/<agent>/default/<env>.yaml`). `export_sft` re-renders every
step through the agent adapter, so a mismatched history window or resolution
trains on prompts the model will not see at inference. Recipe configs under
`scripts/configs/<agent>/recipes/sft/` are the fallback for corpora with no
rollout config, and each recipe should state any deliberate difference.

`export_sft` reads only `agent_id` and `agent_kwargs`; `env_kwargs` is ignored.
The env tool surface is replayed from each row's metadata, and the screenshot in
the parquet is already at the env's `display_resolution`. Pass datasets with
`--data-paths` and images with `--image-root=$CUA_LITE_DATASETS_ROOT`; discovery
keeps only the requested `--splits` (default `train`).

Per-row dispatch:

```python
metadata = metadata_from_dict(row["metadata"])
adapter_key = compose_key(agent_id, *metadata.dims)
```

— this string is the agent registry lookup key. For CUA datasets, `dims` is
`[platform, task_type]`, matching the path cohort. The adapter renders the
row's messages into the model's tool-calling format, `export_sft` freezes each
prompt/response boundary with the target model's chat template, and `images`
are loaded from `--image-root` then serialized as PNG bytes into the output
parquet's `processed_images` column.

Common flags:

- `--data-paths` accepts an absolute dataset root or a regex after a literal
  base, such as `.../ScaleCUA/(desktop|browser|mobile)/use`; matched platform
  components remain distinct even when their partition filenames are equal.
- `--splits train validation` — export the whole carve instead of the default `train` only.
- `--head N` / `--sample N` — limit rows after all discovered parquets are
  pooled; use one invocation per cohort when the smoke must cover each cohort.
- `--filter "lambda m: (m.others.get('episode_return') or 0) >= 1.0"` — filter rows by metadata predicate.
- `--row-group-size N` — tune output parquet row-group size.

---

## Quick reference

| I want to … | Use |
|---|---|
| Process a dataset I have raw data for | [Path A](#path-a--process-from-raw-upstream-data) — `uv run --no-sync bash lite/data/preproc/<name>/scripts/process_data.sh` |
| Mirror a published cua-lite dataset locally | [Path B](#path-b--pull-from-huggingface) — `uv run python -m lite.data.hf.download <Name>` |
| Browse / iterate via `datasets` | [Path C](#path-c--load-directly-via-datasets) — `load_dataset("cua-lite/<Name>", "<plat>.<task_type>")` |
| Stage rollout log-roots | [Path A2](#path-a2--stage-a-rollout-run) — `uv run python -m lite.data.hf.stage --log-roots ... --name <Name>` |
| Export local data for SFT training | [`export_sft`](#consuming-with-export_sft) — read local datasets or rollout roots, write adapter-rendered parquet |
| Add a new dataset | See [`preproc/AGENTS.md`](/lite/data/preproc/AGENTS.md) — adapter authoring SOP |
| Publish a local dataset to HF | `uv run python -m lite.data.hf.upload <Name> --org cua-lite` |
