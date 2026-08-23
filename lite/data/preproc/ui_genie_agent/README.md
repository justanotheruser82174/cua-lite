# UI-Genie-Agent — preprocessing how-to

cua-lite adapter for [`HanXiao1999/UI-Genie-Agent-16k`](https://huggingface.co/datasets/HanXiao1999/UI-Genie-Agent-16k),
mobile/Android multi-step agent trajectories. Two subsets are carried as
variants of a single `use` cohort:

    cua-lite/UI-Genie-Agent/mobile/use/<split>/{ui_genie,amex}.parquet

| Variant | Annotations | Screenshots |
|---------|-------------|-------------|
| `ui_genie` | `ui_genie_agent_16k.jsonl` | shipped in the HF repo (`data/screenshots/`) |
| `amex` | `AMEX_Agent_34K.jsonl` | re-annotation of [`Yuxiang007/AMEX`](https://huggingface.co/datasets/Yuxiang007/AMEX) (~94 GB, fetched + unpacked separately) |

For the format/mapping spec see [`AGENTS.md`](/lite/data/preproc/ui_genie_agent/AGENTS.md).

## Environment

```bash
export CUA_LITE_RAW_DATASETS_ROOT=/path/to/raw         # holds HanXiao1999/UI-Genie-Agent-16k/ (+ Yuxiang007/AMEX/ for amex)
export CUA_LITE_DATASETS_ROOT=/path/to/processed       # canonical output root
uv sync --extra data
```

## 1. Download raw data

```bash
uv run --no-sync bash lite/data/preproc/ui_genie_agent/scripts/download_raw_data.sh
```

Downloads `HanXiao1999/UI-Genie-Agent-16k` (~6 GB, includes the `ui_genie`
screenshots) and the `Yuxiang007/AMEX` screenshot zip parts (~94 GB, needed only
for the `amex` subset). Skip the AMEX step if you only want `ui_genie`.

## 2. Prepare screenshots

```bash
uv run --no-sync bash lite/data/preproc/ui_genie_agent/scripts/process_raw_data.sh
```

Verifies the `ui_genie` layout, merges the multi-part AMEX zip
(`screenshot.z01-z08` + `screenshot.zip`), unzips it, and symlinks it under
`HanXiao1999/UI-Genie-Agent-16k/AMEX/screenshot/` where the preprocessor expects
it. Idempotent (skips completed steps); needs `zip` + `unzip` on PATH. If
`Yuxiang007/AMEX` is absent it skips AMEX prep so `ui_genie` still runs.
An interrupted/invalid merged zip is rebuilt through a same-directory temporary
archive and atomically replaced only after `unzip -t` succeeds.

## 3. Run preprocessing

```bash
# both subsets
uv run --no-sync bash lite/data/preproc/ui_genie_agent/scripts/process_data.sh --verbose

# or directly (one subset or all)
uv run python -m lite.data.preproc.ui_genie_agent.use --subset all [--verbose] [--head N] [--dry-run]
```

`--subset ui_genie|amex|all` (default `all`). `--dry-run` reports the plan
without writing; `--head N` caps trajectories per subset for a smoke test.
Output lands under `${CUA_LITE_DATASETS_ROOT}/cua-lite/UI-Genie-Agent/` (layout
in [`AGENTS.md`](/lite/data/preproc/ui_genie_agent/AGENTS.md#2-output-directory-structure)).

## 4. Verify

```bash
uv run python -c "
from pathlib import Path
import pyarrow.parquet as pq
from lite.data.staging import iter_partitions
root = Path('${CUA_LITE_DATASETS_ROOT}/cua-lite/UI-Genie-Agent')
for *_, f in iter_partitions(root):
    print(f'{f.relative_to(root)}: {pq.ParquetFile(f).metadata.num_rows} rows')
"
```

Smoke-test SFT export (see [`lite/data/README.md`](/lite/data/README.md)):

```bash
uv run python -m lite.train.export.export_sft \
    --agent-id qwen3_vl \
    --model-id Qwen/Qwen3-VL-4B-Instruct \
    --data-paths ${CUA_LITE_DATASETS_ROOT}/cua-lite/UI-Genie-Agent/mobile/use \
    --image-root ${CUA_LITE_DATASETS_ROOT} \
    --head 5 -o /tmp/ui-genie-sft-smoke.parquet
```

## 5. Publish to HuggingFace

```bash
uv run python -m lite.data.hf.upload UI-Genie-Agent --org cua-lite --dry-run   # validate plan
uv run python -m lite.data.hf.upload UI-Genie-Agent --org cua-lite             # push
```
