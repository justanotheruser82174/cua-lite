---
description:
globs: lite/data/preproc/guiact/**
alwaysApply: false
---

# GUIAct Dataset Preprocessing Guide

## Overview

Preprocessing specification for the GUIAct dataset ([`yiye2023/GUIAct`](https://huggingface.co/datasets/yiye2023/GUIAct), from *GUICourse*, arXiv:2406.11317). The preprocessed data follows the canonical format in [`lite/data/preproc/AGENTS.md`](/lite/data/preproc/AGENTS.md) and is written to the `cua-lite/GUIAct/` layout via [`lite/data/staging`](/lite/data/staging.py). This file is the data spec; run instructions live in [`README.md`](/lite/data/preproc/guiact/README.md).

GUIAct has three subsets, mapping to two task types:

- **web-single** → `browser@grounding.action` — one-screenshot records (1+ atomic actions each). `actions_history` and `logs` are empty across all records, confirming no multi-step structure; many images carry multiple independent QAs (up to ~12 per image), i.e. independent grounding questions, not steps of one trajectory.
- **web-multi** → `browser@use` — multi-step web episodes (Chinese + English).
- **smartphone** → `mobile@use` — multi-step Android episodes.

The upstream **test** split is honored verbatim as the **validation** split (no extra hash-carving from train); see [`utils.make_splitter`](/lite/data/preproc/guiact/utils.py).

**Related Files:**

| Type | File |
|------|------|
| Parent specification | [`lite/data/preproc/AGENTS.md`](/lite/data/preproc/AGENTS.md) |
| Adapter shared helpers | [`utils.py`](/lite/data/preproc/guiact/utils.py) (staging wiring + coordinate/action converters) |
| HF repo metadata | [`repo.json`](/lite/data/preproc/guiact/repo.json) |
| Tool definitions | [`lite/core/tools/action_space/base.py`](/lite/core/tools/action_space/base.py) (`LiteDesktopActionSet` / `LiteMobileActionSet`) |
| Preprocessing scripts | [`grounding-action.py`](/lite/data/preproc/guiact/grounding-action.py) (web-single), [`use.py`](/lite/data/preproc/guiact/use.py) (web-multi + smartphone) |
| Shell scripts | [`scripts/download_raw_data.sh`](/lite/data/preproc/guiact/scripts/download_raw_data.sh) (download), [`scripts/process_raw_data.sh`](/lite/data/preproc/guiact/scripts/process_raw_data.sh) (decode base64 images), [`scripts/process_data.sh`](/lite/data/preproc/guiact/scripts/process_data.sh) (run Python pipeline) |

**Source Data Locations** (under `${CUA_LITE_RAW_DATASETS_ROOT}/yiye2023/GUIAct/`):

| Subset | JSON annotations (train/test) | Raw image parquet | Extracted images (on disk) | Ext |
|--------|------------------------------|-------------------|----------------------------|-----|
| web-single | `web-single_{train,test}_data.json` | `web-single_{train,test}_images.parquet` | `images/web-single/<image_id>.png` | png |
| web-multi | `web-multi_{train,test}_data.json` | `web-multi_{train,test}_images.parquet` | `images/web-multi/{train,test}/<image_id>.jpg` | jpg |
| smartphone | `smartphone_{train,test}_data.json` | `smartphone_{train,test}_images.parquet` | `images/smartphone/<image_id>.jpg` | jpg |

Images are base64 inside the parquet; [`scripts/process_raw_data.sh`](/lite/data/preproc/guiact/scripts/process_raw_data.sh) decodes them to disk before preprocessing.

---

## Output Directory Structure

```
${CUA_LITE_DATASETS_ROOT}/cua-lite/GUIAct/
├── images/<hash[:2]>/<hash>.<ext>           # content-addressed image store
├── browser/
│   ├── grounding.action/<split>/<variant>.parquet     # web-single
│   └── use/<split>/<variant>.parquet                  # web-multi
└── mobile/
    └── use/<split>/<variant>.parquet                  # smartphone
```

`<split>` ∈ {`train`, `validation`}. Every cohort has one variant, written as `<split>/<variant>.parquet`. `train` rows come from the upstream `*_train_data.json`; `validation` rows come from the upstream `*_test_data.json` (routed by [`SplitAssigner`](/lite/data/staging.py)'s `canonical_fn`).

---

## Data Statistics

<!-- STATS:BEGIN — regenerate with the command below -->
| Partition | train | validation |
|-----------|------:|-----------:|
| browser/grounding.action | 64,612 | 1,372 |
| browser/use | 5,460 | 37 |
| mobile/use | 7,208 | 230 |

Full preprocessing yields **78,919 rows** and **116,257 unique referenced
images**: 65,984 web-single, 5,497 web-multi, and 7,438 smartphone. `train` maps
to the upstream train files and `validation` to test. Drops explicitly include
unexecutable web `select`, true zero-distance scroll, malformed coordinates,
and actionless single-step episodes.
<!-- STATS:END -->

**Drop / skip reasons:**

The current full run drops 1,676/67,660 web-single rows (`select`: 1,659;
empty action list: 10; zero-distance scroll: 7) and 259/13,194 use
trajectories (missing logical history: 156; zero-distance scroll: 63;
actionless single-step episode: 38; source coordinate outside `[0, 1]`: 2).

| Reason | Subset | Behavior |
|--------|--------|----------|
| Empty `actions_label` | web-single | skip record (`SkipTrajectoryError`) |
| Missing image on disk | all | skip record / skip whole episode |
| Corrupt / unreadable image | all | skip affected record / whole episode at staging; count it |
| Unparseable coordinate string | all | skip record / skip whole episode |
| Source coordinate outside `[0, 1]` (beyond rounding tolerance) | all | skip record / whole episode before staging; the current full run finds 2 web-multi trajectories |
| Out-of-bound coordinate after conversion | all | dropped by `has_oob_coordinate` (expected 0 because only in-tolerance rounding is clamped) |

Regenerate the statistics table:

```bash
uv run python -c "
from pathlib import Path
import pyarrow.parquet as pq
from lite.data.staging import iter_partitions
root = Path('${CUA_LITE_DATASETS_ROOT}/cua-lite/GUIAct')
for *_, f in iter_partitions(root):
    print(f'{f.relative_to(root)}: {pq.ParquetFile(f).metadata.num_rows} rows')
print('images:', sum(1 for p in (root/'images').rglob('*') if p.is_file()))
"
```

---

## Source Data Format

Each subset is a JSON list of step-records. Record-level fields:

| Field | Type | Notes |
|-------|------|-------|
| `uid` | str | web-single: `uid_image_<image>_qa_<n>`; web-multi: `uid_record_<episode>_step_<n>`; smartphone: `uid_episode_<episode>_step_<n>` |
| `image_id` | str | resolves to `images/<subdir>/<image_id>.<ext>` |
| `image_size` | `{width, height}` | source resolution |
| `question` | str | task instruction (web-single) / episode goal (multi-step; identical across an episode's steps) |
| `thoughts` | str | short human rationale — **100% populated for web-single only**; empty for web-multi / smartphone |
| `actions_history`, `logs` | list | empty across all records (web-single is single-step; multi-step structure is reconstructed from `uid`) |
| `actions_label` | list (web) / dict (smartphone) | the action(s) for this step — see Action Space below |

The source `related` coordinate fields are normalized `[0, 1]` floats encoded as `<box>x1, y1, x2, y2</box>` or `<point>x, y</point>` strings (there are parallel `absolute` pixel fields, which we ignore in favor of the resolution-independent `related` values).

**Full raw example (web-single):**

```json
{
  "uid": "uid_image_dffe7794-...-c4342caa93de_qa_01",
  "image_id": "dffe7794-...-c4342caa93de",
  "image_size": {"width": 1920, "height": 1200},
  "question": "Sign up for an account",
  "actions_history": [],
  "logs": [],
  "thoughts": "click on the 'Sign Up' button",
  "actions_label": [
    {"name": "click",
     "element": {"id": 3, "absolute": "<box>900, 17, 967, 55</box>",
                 "related": "<box>0.879, 0.022, 0.944, 0.072</box>"}}
  ]
}
```

**Full raw example (smartphone step):** `actions_label` is a single dict, not a list.

```json
{
  "uid": "uid_episode_10270193012375700035_step_03",
  "image_id": "uid_episode_10270193012375700035_step_03",
  "image_size": {"width": 540, "height": 1140},
  "question": "search for reddit and open it",
  "thoughts": "",
  "actions_label": {"name": "tap", "point": {"related": "<point>0.880, 0.640</point>"}}
}
```

---

## Action Space & Mapping

Browser cohorts from raw `web-single` / `web-multi` use [`LiteDesktopActionSet`](/lite/core/tools/action_space/base.py); smartphone uses [`LiteMobileActionSet`](/lite/core/tools/action_space/base.py). Action names are matched case-insensitively (the source has a handful of capitalized anomalies, e.g. `Click`). All coordinates are converted from `[0, 1]` to `[0, 1000]` integers (point → `rel_to_1000`; bbox → center via `bbox_center_to_1000`).

### Web actions → `LiteDesktopActionSet` (web-single + web-multi)

| GUIAct action | Source payload | → tool_call(s) |
|---------------|----------------|----------------|
| `click` | `element.related` bbox (+ `point.related` on web-multi) | `click(coordinate=C)` — prefers the precise `point` when present, else bbox center |
| `select` | option text + bbox | skip row/episode: bbox identifies the dropdown, not the option, and canonical computer actions have no executable selector |
| `hover` | `point`/`element` | `mouse_move(coordinate=C)` |
| `input` | optional `point`/`element` + `text` | `click(coordinate=C)` (if a target is given) then `type(text=...)` |
| `enter` | — | `key(keys=["enter"])` |
| `scroll` | `scroll.related.{down,right}` plus absolute fallback | `scroll(direction, amount)` at five units per source viewport; rounded `±0.00` uses the absolute sign with amount 1, and a true zero-distance no-op is skipped |
| `select_text` | `dual_point.related.{from,to}` | `drag(start_coordinate, coordinate)` |
| `copy` | `text` | `key(keys=["ctrl", "c"])` |
| `answer` | `text` | `response(text)` (web-single / non-terminal) — on a `use` episode's **last** step it is the episode's ANSWER; if executable actions remain, the text stays on that final assistant action turn, otherwise it becomes the content-only final's text |

### Smartphone actions → `LiteMobileActionSet`

| GUIAct action | Source payload | → tool_call(s) |
|---------------|----------------|----------------|
| `tap` | `point.related` | `tap(coordinate=C)` |
| `swipe` | `dual_point.related.{from,to}` | `swipe(start_coordinate, coordinate)` |
| `input` | `text` | `type(text=...)` |
| `enter` | — | `system_button(button="Enter")` |
| `answer` | `text` | `response(text)` — on a `use` episode's **last** step it is the episode's ANSWER; if executable actions remain, the text stays on that final assistant action turn, otherwise it becomes the content-only final's text |

Only the focus click synthesized by `input` is removed when the immediately
preceding source action already clicked the same coordinate. Source-authored
repeated clicks/Enter presses are retained; they include 315 double clicks used
to increment controls and four intentional repeated Enter presses.

---

## Processing Workflow

1. **Read raw JSON** for each subset's `*_train_data.json` and `*_test_data.json`; tag each row with its upstream split label (`train` / `test`).
2. **web-single (grounding.action):** for each record, resolve the image, drop empty `actions_label`, convert actions with `is_terminal=False` (so `answer` → `response`), build a 2-turn message: `user[image + question]` and `assistant[inline_reasoning(thoughts) + tool_calls]`.
3. **web-multi / smartphone (`use`):** group by episode id and logical `actions_history`. Reject the whole trajectory on an interior logical gap: a later action may depend on the missing action history, so publishing its tail as a new row would preserve the screenshot/action pair but delete source-provided context. web-multi train/test image paths and row IDs are split-scoped because episode `03329` reuses two image IDs for different screenshots. Build tool-result boundaries and EOF finals as below; no synthetic `terminate` is appended.
4. **Coordinate normalization:** `related` `[0, 1]` → `[0, 1000]` ints, clamped.
5. **Image dedup:** each resolved image is hashed into the content-addressed store; `images` becomes store-relative paths.
6. **Split assignment:** `SplitAssigner.canonical_fn` reads the transient `metadata.others.split` hint → `train`→`train`, `test`→`validation`. `stage_entry` strips the hint before persistence.
7. **OOB filter:** `has_oob_coordinate` drops any converted row with a coordinate outside `[0, 1000]`; source coordinates outside `[0, 1]` fail conversion earlier.
8. **Flush:** buffers keyed by `(metadata.dims[0], metadata.dims[1], split, variant)` are written to canonical parquet paths via `staging.flush_buffers`.

---

## Error Handling

| Error type | Behavior |
|------------|----------|
| Empty `actions_label` (web-single) | skip record (`SkipTrajectoryError`) |
| Image missing on disk | web-single: skip record; `use`: skip whole episode |
| Image corrupt/unreadable | web-single: skip record; `use`: skip whole episode; continue the run |
| Unparseable `<box>` / `<point>` string | raise `SkipTrajectoryError` → skip record/episode |
| Source coordinate outside `[0, 1]` | raise `SkipTrajectoryError` → skip record/episode (2 web-multi train trajectories) |
| Unknown action name | raise `ValueError` (fail loud — indicates an unhandled vocabulary item) |
| Duplicate source step within one episode | raise `ValueError`; never silently keep one record |
| Missing logical prefix or interior history gap | skip the whole trajectory; web-multi test has 156/193 affected episodes (841/1,065 records, 353 missing decisions): 51 start after step 0 and 105 start at 0 but later jump |
| `select` without an executable selector | skip record/episode rather than falsify it as a click |
| true zero-distance scroll | skip record/episode rather than invent `down(1)` |
| Non-terminal `answer` (`use`) | skip whole episode — it emits a standalone `response` call and a screenshot answers at most one call, so the call would be published unpaired (11 web-multi train episodes) |
| Single-step episode with no executable action and no answer (`use`) | skip whole episode — nothing publishable; a single executable action is kept as the EOF SFT label |
| Out-of-bound coordinate | drop row via `has_oob_coordinate` (counted, printed) |

---

## Output Format

Actions are always nested inside a `computer` batch wrapper
(`platform="browser"` and `"desktop"` both use `computer`; `platform="mobile"` uses
`mobile`) — never emitted as bare top-level calls.

One canonical row (web-single `grounding.action`):

```json
{
  "images": ["cua-lite/GUIAct/images/07/079fee609d07ea7595…png"],
  "messages": [
    {"role": "user", "content": [
      {"type": "image", "index": 0},
      {"type": "text", "text": "Sign up for an account"}
    ]},
    {"role": "assistant",
     "content": [{"type": "inline_reasoning", "text": "click on the 'Sign Up' button"}],
     "tool_calls": [
       {"id": "call_0000",
        "type": "function",
        "function": {
          "name": "computer",
          "arguments": {"actions": [{"action": "click", "coordinate": [767, 44]}]}
        }}
     ]}
  ],
  "metadata": {
    "metadata_kind": "cua",
    "dims": ["browser", "grounding.action"],
    "extra_tool_schemas": [],
    "valid_actions": null,
    "others": {
      "id": "guiact_web_single_uid_image_dffe7794-aa20-48fa-98e9-c4342caa93de_qa_01",
      "resolution": [1920, 1200],
      "os": null,
      "source": "yiye2023/GUIAct",
      "source_id": "uid_image_dffe7794-aa20-48fa-98e9-c4342caa93de_qa_01"
    }
  }
}
```

A `use` row (real row `guiact_web_multi_00159`). Note that each step's collapsed
actions share **one** `computer` wrapper and **one** tool-call `id`, and the next
screenshot arrives as a `role:"tool"` message keyed by that same `id`.

**Image extensions differ by subset**: web-single is `.png`, while **web-multi and
smartphone are `.jpg`** — the `ImageStore` extension follows the raw bytes, so do
not copy `.png` across subsets.

```json
{
  "images": [
    "cua-lite/GUIAct/images/7b/7b6b6f83f27a6a…jpg",
    "cua-lite/GUIAct/images/27/27a6900ee8b0f0…jpg",
    "cua-lite/GUIAct/images/17/1756d1ec010977…jpg"
  ],
  "messages": [
    {"role": "user", "content": [
      {"type": "image", "index": 0},
      {"type": "text", "text": "意外伤害保险的保险时间多久？"}
    ]},
    {"role": "assistant", "tool_calls": [
      {"id": "call_0000", "type": "function", "function": {"name": "computer", "arguments": {"actions": [
        {"action": "click", "coordinate": [620, 199]},
        {"action": "click", "coordinate": [358, 771]}
      ]}}}
    ]},
    {"role": "tool", "tool_call_id": "call_0000", "content": [{"type": "image", "index": 1}]},
    {"role": "assistant", "tool_calls": [
      {"id": "call_0001", "type": "function", "function": {"name": "computer", "arguments": {"actions": [
        {"action": "drag", "coordinate": [791, 918], "start_coordinate": [309, 913]},
        {"action": "key", "keys": ["ctrl", "c"]}
      ]}}}
    ]},
    {"role": "tool", "tool_call_id": "call_0001", "content": [{"type": "image", "index": 2}]},
    {"role": "assistant", "content": [{"type": "text", "text": "意外伤害保险保险期间一般较短，多为一年及一年期以下。定期寿险保险期间一般较长。"}]}
  ],
  "metadata": {
    "metadata_kind": "cua",
    "dims": ["browser", "use"],
    "extra_tool_schemas": [],
    "valid_actions": null,
    "others": {
      "id": "guiact_web_multi_00159",
      "resolution": [1280, 598],
      "os": null,
      "source": "yiye2023/GUIAct",
      "source_id": "guiact_web_multi_00159"
    }
  }
}
```

Both blocks above were regenerated from the raw records and fed back through
`validate_canonical_rows`; only the image hashes are abbreviated at `…`. Neither
elides `metadata` — a `messages`-only block is not a canonical row and cannot be
run through the gate, which is how the `.png`/`.jpg` and answer-text errors that
used to be here survived.

`use` rows have the same tagged CUA `metadata` shape (`dims: ["browser"|"mobile", "use"]`, `os: null`/`"android"`), one image per step in `images`, non-final action assistant turns answered by a `role:"tool"` screenshot result, and either a final assistant action turn at EOF or a content-only assistant final whose text is the terminal `answer`'s text (or the `Done.` marker when the terminal step authored no answer and no executable EOF label). `others.split` is an upstream routing hint that `stage_entry` **pops**, so it never appears on a published row.
