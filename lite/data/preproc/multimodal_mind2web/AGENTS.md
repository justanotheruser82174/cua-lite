---
description:
globs: lite/data/preproc/multimodal_mind2web/**
alwaysApply: false
---

# Multimodal-Mind2Web Dataset Preprocessing Guide

Technical specification for the Multimodal-Mind2Web (`osunlp/Multimodal-Mind2Web`)
adapter. For how to run it, see [`README.md`](/lite/data/preproc/multimodal_mind2web/README.md).
For the cross-dataset contract, see [`lite/data/preproc/AGENTS.md`](/lite/data/preproc/AGENTS.md).

## 1. Overview

Multimodal-Mind2Web is the screenshot-augmented version of Mind2Web: real
human-annotated multi-step web navigation episodes across 100+ websites. Each
step carries a full-page screenshot, the task instruction, and a target DOM
element with a pixel-space bounding box. The adapter emits a single cohort:

- **use** → `browser@use` — one multi-step episode per
  `annotation_id`.

**Training-safe scope.** Only the upstream `train` split is processed. The
`test_task` / `test_website` / `test_domain` splits are the standard Mind2Web
held-out benchmarks and are NEVER read by this adapter — [`use.py`](/lite/data/preproc/multimodal_mind2web/use.py)
hardcodes `SPLIT = "train"`. No upstream-split label is written; the
[`SplitAssigner`](/lite/data/staging.py) carves a `validation` set out of
`train` by hashing `metadata.others.id` (2,000-row val cap).

**Related files:**

| Type | File |
|------|------|
| Parent specification | [`lite/data/preproc/AGENTS.md`](/lite/data/preproc/AGENTS.md) |
| Adapter shared helpers | [`utils.py`](/lite/data/preproc/multimodal_mind2web/utils.py) (staging wiring + OOB check) |
| Preprocessing script | [`use.py`](/lite/data/preproc/multimodal_mind2web/use.py) |
| Shell scripts | [`scripts/download_raw_data.sh`](/lite/data/preproc/multimodal_mind2web/scripts/download_raw_data.sh) (download), [`scripts/process_raw_data.sh`](/lite/data/preproc/multimodal_mind2web/scripts/process_raw_data.sh) (decode inline screenshots), [`scripts/process_data.sh`](/lite/data/preproc/multimodal_mind2web/scripts/process_data.sh) (run Python pipeline) |
| HF repo metadata | [`repo.json`](/lite/data/preproc/multimodal_mind2web/repo.json) |
| Tool definitions | [`lite/core/tools/action_space/base.py`](/lite/core/tools/action_space/base.py) (`LiteDesktopActionSet`) |

**Source data locations** (under `${CUA_LITE_RAW_DATASETS_ROOT}/osunlp/Multimodal-Mind2Web/`):

| Cohort | Source |
|--------|--------|
| use | `data/train-*.parquet` (step rows) + `images/<annotation_id>/<step>.jpg` (decoded screenshots) |

Screenshots are inline bytes in the parquet shards;
[`scripts/process_raw_data.sh`](/lite/data/preproc/multimodal_mind2web/scripts/process_raw_data.sh)
decodes the `train` split to `images/<annotation_id>/<step>.jpg` (step index is
the source `target_action_index`, zero-padded to 2 digits) before preprocessing.

## 2. Output Directory Structure

```
${CUA_LITE_DATASETS_ROOT}/cua-lite/Multimodal-Mind2Web/
├── images/<hash[:2]>/<hash>.<ext>           # content-addressed image store
└── browser/
    └── use/<split>/use.parquet                      # one row per episode
```

`<split>` ∈ {`train`, `validation`}. The cohort's sole variant is `use`,
written with the same `<split>/<variant>.parquet` layout as every other cohort.
Splits are assigned by hashing
`metadata.others.id` (no upstream-split routing — only `train` is processed).

## 3. Data Statistics

Regenerate after a full run:

```bash
uv run python -c "
from pathlib import Path
import pyarrow.parquet as pq
from lite.data.staging import iter_partitions
root = Path('${CUA_LITE_DATASETS_ROOT}/cua-lite/Multimodal-Mind2Web')
for *_, f in iter_partitions(root):
    print(f'{f.relative_to(root)}: {pq.ParquetFile(f).metadata.num_rows} rows')
print('images:', sum(1 for p in (root/'images').rglob('*') if p.is_file()))
"
```

Skip / drop reasons. The **measured** column is the whole `train` split
(1,009 episodes / 7,775 step rows → 597 published rows / 3,933 action steps);
only three reasons ever fire, so a nonzero count anywhere else is news:

| Reason | Behavior | measured |
|--------|----------|---------:|
| Episode contains a `SELECT` step | drop whole episode (`_SelectSkip`) — no native `<select>` tool in the cua-lite action space | 169 |
| Empty `pos_candidates` | skip whole episode (`SkipEpisodeError`) | 236 |
| Target centre outside the captured screenshot | skip whole episode — a well-formed rect the shot does not cover, reported distinctly from an unparseable one | 7 |
| Missing / unreadable screenshot | skip whole episode | 0 |
| Missing / blank `confirmed_task` | skip whole episode | 0 |
| Unparseable `target_action_index` / duplicate index | skip whole episode | 0 |
| Malformed `operation` JSON, malformed `pos_candidates[0]`, malformed `bounding_box_rect` | skip whole episode | 0 |
| `TYPE` with empty `value` | skip whole episode (degenerate step) | 0 |
| Single-step episode | kept — its only action is the EOF SFT label | 0 (shortest episode in the corpus is 2 steps) |
| OOB coordinate after normalization | drop whole episode (`has_oob_coordinate`) | 0 — the bounds test above already refuses them, and the survivors are clamped to [0, 1000] |
| Corrupt / unreadable image at staging time | skip whole episode (`n_corrupt`) | 0 |
| Unknown `original_op` | raise `ValueError` (fail loud — unhandled vocabulary) | 0 |

Final actions are publishable EOF labels, so the terminal step is parsed like
every other step. The remaining drops are mid-episode offences that cannot be
recovered without truncating the source before its real EOF — see
[`lite/data/preproc/AGENTS.md`](/lite/data/preproc/AGENTS.md), §Use notes.

## 4. Source Data Format

The `train-*.parquet` shards are one row per *step*. Columns the adapter reads
([`use.py`](/lite/data/preproc/multimodal_mind2web/use.py)
`_COLUMNS`):

| Field | Type | Notes |
|-------|------|-------|
| `annotation_id` | str | episode id — steps are grouped by it |
| `target_action_index` | str-int | step index within the episode; sorted ascending |
| `target_action_reprs` | str | human-readable action description → `action_description` |
| `confirmed_task` | str | episode goal (identical across an episode's steps) |
| `operation` | JSON str | `{"original_op": ..., "value": ...}` — the action |
| `pos_candidates` | list[JSON str] | `pos_candidates[0].attributes` (nested JSON str) carries `bounding_box_rect` |
| `website`, `domain`, `subdomain` | str | carried into `others` |
| `screenshot` | struct | inline image bytes — decoded to disk by `process_raw_data.sh`, not read by `use.py` |

The target coordinate is the centre of `pos_candidates[0].attributes.bounding_box_rect`
(`"x,y,w,h"` in pixels), normalized by the per-step screenshot size to
[0, 1000] and clamped (`_bbox_center`). Per-step resolution is read from the
decoded JPEG header (`_read_image_size`; PIL's decompression-bomb guard is
lifted since full-page shots can be ~100 Mpx).

## 5. Action Space & Mapping

Browser action surface ([`LiteDesktopActionSet`](/lite/core/tools/action_space/base.py)),
keyed off `operation.original_op`:

| Source `original_op` | Source payload | → tool_call(s) |
|---|---|---|
| `CLICK` | bbox centre | `click(coordinate=C)` |
| `TYPE` | bbox centre + `value` | `click(coordinate=C)`, `type(text=value)` (skip episode if `value` empty) |
| `HOVER` | bbox centre | `mouse_move(coordinate=C)` |
| `ENTER` | — | `key(keys=["enter"])` |
| `SELECT` | — | **drop whole episode** (`_SelectSkip`) — click+type won't reliably replay native `<select>` typeahead |

**Final.** There is no per-step termination signal in the source. The final
executable action stays at EOF with no synthetic `Done.`, `terminate` tool_call,
or fabricated tool result.

## 6. Processing Workflow

1. Glob `data/train-*.parquet` (sorted); raise if none. Group step rows by
   `annotation_id` (`load_train_episodes`).
2. Per episode: sort steps by `target_action_index`; reject empty / duplicate
   indices.
3. Resolve every step screenshot `images/<annotation_id>/<step:02d>.jpg` and
   read its size; skip episode on any missing/unreadable image.
4. Validate `confirmed_task`; build the first `user` turn `[image + task]`. Every
   later screenshot becomes a `role:"tool"` message carrying the preceding
   batch's `id` as `tool_call_id` (rewritten by `finalize_use_messages`), not a `role:"user"`
   turn.
5. Per step: parse `operation`, parse `pos_candidates[0]` → `bounding_box_rect`
   → normalized coordinate, map to tool_calls. Each assistant turn carries
   `tool_calls` plus an `action_description` content part (`target_action_reprs`)
   when present.
6. Keep the final executable action at EOF. The source has no post-action
   screenshot after that action, so there is no synthetic `Done.` and no
   fabricated `role:"tool"` result.
7. `has_oob_coordinate` over the whole episode → drop if any OOB.
8. `stage_entry` hashes images into the content-addressed store, fills metadata
   defaults, assigns a split (hash on `id`, 2,000-row val cap); `flush_buffers`
   writes one parquet per partition.

## 7. Error Handling

§3 owns the reason list; this section owns only the mapping from exception to the
counter `use.py` prints, so the two cannot drift:

| Exception | Counter in the summary line |
|-----------|-----------------------------|
| `_SelectSkip` | `select_dropped` |
| `SkipEpisodeError` (every other per-step and episode-level refusal) | `skipped` |
| `has_oob_coordinate` returning `True` | `oob` |
| `OSError` / `ValueError` / `SyntaxError` from `stage_entry` | `corrupt` |
| `ValueError` for an unknown `original_op` | none — it propagates and fails the run |

The summary line is a single aggregate per counter. The **per-reason** breakdown
in §3 requires `--verbose`, which prints one `[skip] <annotation_id>: <reason>`
line per dropped episode.

## 8. Output Format

`use` row shape (based on `mind2web_03e45ce0-…`; image hashes and final-action
details truncated). Actions are nested in a `computer` batch wrapper for the
CUA platform dim `"browser"`, and each non-final post-action screenshot is a `role:"tool"`
message carrying the `tool_call_id` of the call it answers — never a `role:"user"`
turn.

The source carries no screenshot after the final executable action, so the row
ends on that assistant `tool_calls` turn at EOF: a 3-step episode publishes **3
actions and 3 images**. Only the final assistant call `id` is intentionally unanswered;
intermediate calls must still have `role:"tool"` results. `others.resolution` is
`null` here because this episode's three full-page shots are three different
sizes; it is a pair only when every step agrees (see the rationale beside the
`resolution` assignment in `use.py`).

```jsonc
{
  "images": [
    "cua-lite/Multimodal-Mind2Web/images/f8/f8bf4…jpg",
    "cua-lite/Multimodal-Mind2Web/images/f1/f1ccc…jpg",
    "cua-lite/Multimodal-Mind2Web/images/0f/0fbef…jpg"
  ],
  "messages": [
    {"role": "user", "content": [{"type": "image", "index": 0}, {"type": "text", "text": "Find the latest news article and send an email about it."}]},
    {"role": "assistant",
     "tool_calls": [{"id": "call_0000", "type": "function", "function": {"name": "computer", "arguments": {"actions": [{"action": "click", "coordinate": [827, 72]}]}}}],
     "content": [{"type": "action_description", "text": "[link]  Jets signing former Packers QB Boyle to 1-year dea... -> CLICK"}]},
    {"role": "tool", "tool_call_id": "call_0000", "content": [{"type": "image", "index": 1}]},
    {"role": "assistant",
     "tool_calls": [{"id": "call_0001", "type": "function", "function": {"name": "computer", "arguments": {"actions": [{"action": "click", "coordinate": [755, 113]}]}}}],
     "content": [{"type": "action_description", "text": "[use]   -> CLICK"}]},
    {"role": "tool", "tool_call_id": "call_0001", "content": [{"type": "image", "index": 2}]},
    {"role": "assistant",
     "tool_calls": [{"id": "call_0002", "type": "function", "function": {"name": "computer", "arguments": {"actions": [{"action": "click", "coordinate": […, …]}]}}}],
     "content": [{"type": "action_description", "text": "…"}]}
  ],
  "metadata": {
    "metadata_kind": "cua", "dims": ["browser", "use"],
    "extra_tool_schemas": [], "valid_actions": null,
    "others": {"id": "mind2web_03e45ce0-4375-44aa-b57f-cf439ccbe363", "resolution": null, "os": null,
               "source": "osunlp/Multimodal-Mind2Web", "source_id": "03e45ce0-4375-44aa-b57f-cf439ccbe363",
               "website": "nfl", "domain": "Entertainment", "subdomain": "Sports"}
  }
}
```

A `TYPE` step expands to `click` + `type`; because the source proves one
post-action screenshot for the pair, both land in a **single** `computer`
wrapper under one `id`, answered by one `role:"tool"` result.
