# GUIOdyssey — preprocessing how-to

cua-lite adapter for [`hflqf88888/GUIOdyssey`](https://huggingface.co/datasets/hflqf88888/GUIOdyssey),
a long-horizon cross-app Android dataset (8,334 trajectories, ~128k screenshots).
Emits `use` (multi-step episodes) and `understanding` (per-step screen
captioning). For the format/mapping spec see [`AGENTS.md`](/lite/data/preproc/guiodyssey/AGENTS.md).

## Environment

```bash
export CUA_LITE_ROOT="/path/to/cua-lite"                       # project root (used just below)
export CUA_LITE_RAW_DATASETS_ROOT=/path/to/raw-data-mount      # holds hflqf88888/GUIOdyssey/
export CUA_LITE_DATASETS_ROOT="${CUA_LITE_ROOT}/.data/huggingface"
uv sync --extra data
```

## 1. Download raw data

```bash
uv run --no-sync bash lite/data/preproc/guiodyssey/scripts/download_raw_data.sh [--dry-run]
```

Downloads `hflqf88888/GUIOdyssey` into
`${CUA_LITE_RAW_DATASETS_ROOT}/hflqf88888/GUIOdyssey/`. The screenshots ship as
a **multi-part zip** (`screenshots/screenshots.z01`–`.z08` + `screenshots.zip`,
~87 GiB / 92.6 GB), so budget roughly double that on disk: extracting adds
another ~88 GiB (127,893 PNGs). `annotations/` + `splits/` are loose JSON
(~200 MB).

## 2. Merge + extract the screenshot archive

`scripts/process_raw_data.sh` only **verifies** the on-disk layout and prints
counts — it does **not** extract. Merge and unzip the parts first, using the
same idiom as AMEX in
[`ui_genie_agent/scripts/process_raw_data.sh`](/lite/data/preproc/ui_genie_agent/scripts/process_raw_data.sh)
(archive entries are `screenshots/<uid>_<step>.png`, so `-j` flattens them into
the `screenshots/` dir the adapter reads):

```bash
# UNVERIFIED: the archive layout below was read from `unzip -l`, but this
# merge+extract was not re-run (it rewrites ~88 GiB / 127,893 files, ~1 h).
cd ${CUA_LITE_RAW_DATASETS_ROOT}/hflqf88888/GUIOdyssey/screenshots
zip --fix screenshots.zip --out screenshots_merged.zip
unzip -j -q -o screenshots_merged.zip -d .
cd -
```

Then verify (expects `8334` annotations / `127893` screenshot PNGs / `4`
split files):

```bash
uv run --no-sync bash lite/data/preproc/guiodyssey/scripts/process_raw_data.sh
```

## 3. Run preprocessing

```bash
# both cohorts
uv run --no-sync bash lite/data/preproc/guiodyssey/scripts/process_data.sh --verbose

# or individually
uv run python lite/data/preproc/guiodyssey/use.py [--verbose] [--head N] [--dry-run]
uv run python lite/data/preproc/guiodyssey/understanding.py [--verbose] [--head N] [--dry-run]
```

`use.py --dry-run` prints the action histogram without writing. `--head N`
caps episodes for a smoke test. Output lands under
`${CUA_LITE_DATASETS_ROOT}/cua-lite/GUIOdyssey/` (layout in [`AGENTS.md`](/lite/data/preproc/guiodyssey/AGENTS.md#2-output-directory-structure)).

## 4. Verify

```bash
uv run python -c "
from pathlib import Path
from datasets import load_dataset
from lite.data.staging import iter_partitions
root = Path('${CUA_LITE_DATASETS_ROOT}/cua-lite/GUIOdyssey')
for *_, f in iter_partitions(root):
    d = load_dataset('parquet', data_files=str(f), split='train')
    print(f'{f.relative_to(root)}: {len(d)} rows')
"

# SFT export smoke test. The shared mobile/ root discovers and pools both task
# types; this command tests use. Run it again with mobile/understanding to test
# that cohort too, because --head limits the pooled rows rather than each cohort.
# --model-id is required: steps are tokenized at export
uv run python -m lite.train.export.export_sft \
    --agent-id qwen3_vl \
    --model-id Qwen/Qwen3-VL-4B-Instruct \
    --data-paths ${CUA_LITE_DATASETS_ROOT}/cua-lite/GUIOdyssey/mobile/use \
    --image-root ${CUA_LITE_DATASETS_ROOT} \
    --head 5 -o /tmp/guiodyssey-sft-smoke.parquet
```

## 5. Publish to HuggingFace

```bash
uv run python -m lite.data.hf.upload GUIOdyssey --org cua-lite --dry-run   # validate plan
uv run python -m lite.data.hf.upload GUIOdyssey --org cua-lite             # push
```
