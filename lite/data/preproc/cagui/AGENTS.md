# CAGUI Dataset Preprocessing Guide

CAGUI (`OpenBMB/CAGUI`) is a Chinese-language Android mobile dataset. It
contributes **understanding** (region-conditioned UI-element functional
captioning + OCR) and **use** (multi-step agent demonstrations). All
rows are emitted in the canonical cua-lite format defined in the parent
[`lite/data/preproc/AGENTS.md`](/lite/data/preproc/AGENTS.md); this document is
the dataset-specific technical spec. For how-to-run instructions see
[`README.md`](/lite/data/preproc/cagui/README.md).

CAGUI is the first Chinese-only dataset in cua-lite: every row carries
`metadata.others.language = "zh"` so downstream filters can pick it out.

## 1. Overview

**Related Files**

| Type | File |
|------|------|
| Parent specification | [`lite/data/preproc/AGENTS.md`](/lite/data/preproc/AGENTS.md) |
| Tool definitions | [`lite/core/tools/action_space/base.py`](/lite/core/tools/action_space/base.py) (`LiteMobileActionSet`) |
| Adapter shared helpers | [`utils.py`](/lite/data/preproc/cagui/utils.py) |
| Preprocessing scripts | [`understanding.py`](/lite/data/preproc/cagui/understanding.py), [`use.py`](/lite/data/preproc/cagui/use.py) |
| HF repo metadata | [`repo.json`](/lite/data/preproc/cagui/repo.json) |
| Shell scripts | [`scripts/download_raw_data.sh`](/lite/data/preproc/cagui/scripts/download_raw_data.sh), [`scripts/process_raw_data.sh`](/lite/data/preproc/cagui/scripts/process_raw_data.sh), [`scripts/process_data.sh`](/lite/data/preproc/cagui/scripts/process_data.sh) |

**Source Data Locations**

| Subset | cua-lite cohort | Source path | Raw records |
|--------|-----------------|-------------|-------------|
| `cap` (`bbox2function`) | `mobile/understanding` variant `cap` | `CAGUI_grounding/code/cap.jsonl` + `CAGUI_grounding/images/cap/` | 1500 |
| `ocr` (`bbox2text`) | `mobile/understanding` variant `ocr` | `CAGUI_grounding/code/ocr.jsonl` + `CAGUI_grounding/images/ocr/` | 1500 |
| `domestic` | `mobile/use` | `CAGUI_agent/domestic/<episode_id>/` (one JSON + per-step JPEGs) | 600 |

All paths are relative to `${CUA_LITE_RAW_DATASETS_ROOT}/OpenBMB/CAGUI/`. The
upstream `CAGUI_grounding/` directory contains exactly two task files
(`cap.jsonl`, `ocr.jsonl`); `CAGUI_agent/` contains exactly one region subset
(`domestic`). CAGUI is delivered pre-extracted — there are no archives to merge.

## 2. Output Directory Structure

```
${CUA_LITE_DATASETS_ROOT}/cua-lite/CAGUI/
├── images/<hash[:2]>/<hash>.jpeg                 # content-addressed image store
└── mobile/
    ├── understanding/
    │   ├── <split>/cap.parquet                   # bbox2function (functional captions)
    │   └── <split>/ocr.parquet                   # bbox2text (UI text OCR)
    └── use/
        └── <split>/use.parquet                   # domestic (multi-step agent)
```

- `understanding` has variants `cap` and `ocr`; `use` has variant `use`.
  Every cohort uses the same `<split>/<variant>.parquet` layout, including
  single-variant cohorts.
- `<split>` ∈ {`train`, `validation`}, assigned by hashing `metadata.others.id`
  (CAGUI has no upstream split label) with a 2 000-decision validation cap per cohort.

## 3. Data Statistics

Row counts per partition (regenerate with the command below):

| Cohort | variant | train | validation | total |
|--------|---------|------:|-----------:|------:|
| `mobile/understanding` | `cap` | 1473 | 27 | 1500 |
| `mobile/understanding` | `ocr` | 1465 | 34 | 1499 |
| `mobile/use` | `use` | 585 | 15 | 600 |

Total: 3599 rows; 6447 unique images in the content-addressed store. All
coordinates are within `[0, 1000]` and every `images` path resolves.

**Notable source cases and drops**

| Case | Cohort | Count |
|--------|--------|------:|
| Empty/missing `text` field | understanding (`ocr`) | 1 (`ocr.jsonl` line 417, `id=416`) |
| `NO_ACTION` (code 1) | use | 1, preserved as `wait(duration=1.0)` |
| Step after terminal code 10/11 | use | 0 (whole episode would be skipped) |
| Missing image | both | 0 (full download) |
| Corrupt / unreadable image | both | 0 (full download) |
| Out-of-bound coordinate | use | 0 (source coords are normalized [0,1], clamped) |

`use` action-code distribution (all 600 episodes, ~4000 steps):

| AITW code | Name | Count |
|-----------|------|------:|
| 0 | LONG_POINT | 25 |
| 1 | NO_ACTION / wait | 1 |
| 3 | TYPE | 574 |
| 4 | DUAL_POINT (tap / swipe) | 3316 |
| 10 | STATUS_TASK_COMPLETE | 575 |
| 11 | STATUS_TASK_IMPOSSIBLE | 25 |

Reproduce:

```bash
# Row counts per partition
uv run python -c "
from pathlib import Path
import pyarrow.parquet as pq, os
from lite.data.staging import iter_partitions
root = Path(os.environ['CUA_LITE_DATASETS_ROOT'])/'cua-lite/CAGUI'
for *_, f in iter_partitions(root):
    print(f'{f.relative_to(root)}: {pq.read_table(f).num_rows}')
"
# Use action histogram (no output written)
uv run python -m lite.data.preproc.cagui.use --dry-run
```

## 4. Source Data Format

### CAGUI_grounding (cap.jsonl, ocr.jsonl)

Each line is one record:

```json
{
  "task": "bbox2function",
  "image": "grounding_eval/dataset/images/0.jpeg",
  "id": 0,
  "abs_position": "<1061, 2424, 1159, 2522>",
  "rel_position": "<0.8696721311475409, 0.8938053097345132, 0.95, 0.9299410029498525>",
  "text": "UI元素是一个菜单按钮。其主要功能是弹出一个菜单面板，允许用户选择不同的功能选项……"
}
```

| Field | Type | Notes |
|-------|------|-------|
| `task` | str | `"bbox2function"` (cap) or `"bbox2text"` (ocr); validated per subset |
| `image` | str | path with prefix `grounding_eval/dataset/images/…` that does NOT match on-disk layout — use the **basename** only |
| `id` | int | upstream id; **unreliable** for ocr (e.g. row 1 has `id=629` but basename `1.jpeg`) — do NOT use to locate the image |
| `abs_position` | str | `"<x1, y1, x2, y2>"` pixel bbox; used only to recover resolution, then dropped |
| `rel_position` | str | `"<x1, y1, x2, y2>"` normalized [0, 1] bbox; the load-bearing geometry |
| `text` | str | ground-truth Chinese (functional caption for cap, OCR text for ocr) |

Real images live at `CAGUI_grounding/images/{cap,ocr}/<N>.jpeg`, where `<N>` is
`basename(record["image"])`.

### CAGUI_agent (domestic episodes)

Each episode is a directory with one JSON file (a list of step records) and
per-step JPEGs:

```
domestic/<episode_id>/
├── <episode_id>.json
├── <episode_id>_0.jpeg
├── <episode_id>_1.jpeg
└── …
```

Per-step record:

```json
{
  "episode_id":          "006b3e83f8da89719da76290ed590a3c",
  "episode_length":      10,
  "step_id":             0,
  "instruction":         "饿了么找一下附近可以送外卖的刘大饼香辣土豆片夹馍",
  "image_path":          "domestic/006b…/006b…_0.jpeg",
  "image_width":         1220,
  "image_height":        2712,
  "ui_positions":        "[[0.0497, 0.0295, 0.1172, 0.2352], …]",
  "result_action_type":  4,
  "result_action_text":  "",
  "result_touch_yx":     "[0.2275, 0.1475]",
  "result_lift_yx":      "[0.2275, 0.1475]",
  "duration":            null
}
```

| Field | Type | Notes |
|-------|------|-------|
| `episode_id` | str | per-episode unique hex id |
| `step_id` | int | 0-based step index (implicit in output; dropped) |
| `instruction` | str | Chinese task goal (repeated on every step; taken from step 0) |
| `image_path` | str | relative to `CAGUI_agent/`; the step screenshot |
| `image_width` / `image_height` | int | → `metadata.others.resolution=[w,h]` |
| `ui_positions` | str | JSON list of `[y, x, h, w]` normalized [0,1] boxes of every on-screen element → preserved (see Processing Workflow) |
| `result_action_type` | int | AITW `ActionType` enum (see §5) |
| `result_action_text` | str | populated only for TYPE (code 3) |
| `result_touch_yx` / `result_lift_yx` | str | JSON `[y, x]` normalized [0,1] (**y first**); `[-1,-1]` sentinel for non-touch actions |
| `episode_length`, `duration` | int / null | dropped (derivable / always null) |

## 5. Action Space & Mapping

`use` uses [`LiteMobileActionSet`](/lite/core/tools/action_space/base.py)
(`tap`, `long_press`, `swipe`, `type`, `system_button`, `terminate`).

Source coordinates are normalized `[0, 1]` in **(y, x)** order; targets are int
`[0, 1000]` in **(x, y)** order (swap + scale, clamped).

| AITW code | Name | Mapping |
|-----------|------|---------|
| 0 | LONG_POINT | `long_press(coordinate=[x,y], duration=duration/1000)` from `result_touch_yx` |
| 1 | NO_ACTION | `wait(duration=duration/1000)` |
| 3 | TYPE | `type(text=result_action_text)` |
| 4 | DUAL_POINT | `tap([x,y])` if normalized L2(touch, lift) ≤ `0.04`, else `swipe(start=touch, coord=lift)` |
| 5 | PRESS_BACK | `system_button(button="Back")` |
| 6 | PRESS_HOME | `system_button(button="Home")` |
| 7 | PRESS_ENTER | `system_button(button="Enter")` |
| 10 | STATUS_TASK_COMPLETE | explicit terminator → structural content-only `Done.` final |
| 11 | STATUS_TASK_IMPOSSIBLE | explicit terminator → `metadata.others.terminate_status="failure"` |
| any other | — | `CAGUIActionError` (raise) |

**Terminator handling** (matches the `opencua/use.py` convention): codes
10/11 are NOT emitted as their own assistant turns. The walker stops and records
the status; when at least one real action precedes the terminator, that
terminator screenshot is kept as the post-action result image. A `use` row
**never persists a `terminate` tool_call** — the episode always ends on the
content-only assistant `Done.` final ([`structural_final_message`](/lite/data/utils/messages.py)),
and whatever the terminator asserted moves into `metadata.others` via
[`terminate_outcome_others`](/lite/data/utils/messages.py):

- **Code 10 (`STATUS_TASK_COMPLETE`)** → `status="success"`, which asserts nothing
  beyond "the episode ended" (the `Done.` final already carries that). Nothing is
  recorded; `extra_tool_schemas` stays `[]` and no `terminate_status` key appears.
- **Code 11 (`STATUS_TASK_IMPOSSIBLE`)** → `others.terminate_status: "failure"`
  (plus `others.terminate_reason` when the source authored non-blank text).
  Absence of `terminate_status` therefore means "not a self-reported failure".

Either way the invariant is `len(images) in {#assistant_turns, #assistant_turns + 1}`.

CAGUI ships no thought / action-description text, so `use` assistant turns
carry **only `tool_calls`** (matching the `grounding.action` message schema) —
no `inline_reasoning` / `action_description` is invented.

### Understanding prompts (invented)

The grounding source records have no question prompt. We construct a Chinese
instruction embedding the `rel_position` bbox in `[0, 1000]` integer form (no
coordinate-system note — `[0, 1000]` is the project-wide convention):

| Variant | Prompt template (Python f-string) |
|---------|-----------------------------------|
| `cap` | `请描述截图中位于边界框 [{x1}, {y1}, {x2}, {y2}] 内的 UI 元素的功能。` |
| `ocr` | `请识别截图中位于边界框 [{x1}, {y1}, {x2}, {y2}] 内显示的内容。` |

The assistant reply is the source `text` verbatim (plain `text` content — this
is QA-style understanding, not an action).

**Why the `ocr` prompt asks for "内容" (content) rather than "文字" (text):**
upstream `bbox2text` is mostly literal on-screen text (~93%), but ~7% of rows
(104 / 1500) label non-text elements — `…按钮`×72, `…头像`×24, `…封面`×5,
`…图片`×3 (e.g. `歌曲封面`). A text-only prompt would mislabel those, so the
prompt asks generically for the boxed content. Note also that upstream reuses
166 identical `(bbox, text)` annotations across ~414 rows; most align (same
element, same screen) but a minority are spatially misaligned — this is upstream
annotation noise carried through faithfully.

## 6. Processing Workflow

### Understanding (cap, ocr)

1. Read source JSONL line by line.
2. Validate `task` matches the subset (`bbox2function` / `bbox2text`); raise otherwise.
3. Parse `rel_position` → `[0, 1000]` ints (×1000, round, clamp).
4. Derive image filename via `basename(record["image"])` (NOT `id`); resolve via `resolve_path(rel, "CUA_LITE_RAW_DATASETS_ROOT")` — skip the record on miss.
5. Recover resolution from `abs_position` / `rel_position` (`dim = abs / rel`, using the larger-magnitude edge); drop the raw `abs_position`.
6. Skip the record if `text` is empty/missing.
7. Build the user prompt (template + bbox) and the assistant `text` reply.
8. Stage: hash the image into the content store, fill the 5-key metadata defaults, assign a split (hash of `id`), buffer by `(mobile, understanding, split, variant)`.

### Use (multi-step rollout) (domestic)

1. Glob `CAGUI_agent/domestic/*` (sorted); load each `<id>/<id>.json` (skip corrupt files in place).
2. Walk steps in order:
   - code 10/11 → stop, record status, keep the step image only as the preceding action's post-action result;
   - code 1 → `wait(duration=duration/1000)`;
   - else → convert to tool_calls (`CAGUIActionError` on unmapped codes);
   - resolve the step image (skip the whole episode on miss);
   - convert the step's `ui_positions` `[y,x,h,w]`→`[x1,y1,x2,y2]` in `[0,1000]` and append.
3. Skip the episode if no actionable steps precede the terminator.
4. Record the terminator's outcome into `metadata.others` via `terminate_outcome_others` (code 11 → `terminate_status: "failure"`; code 10 → nothing). No `terminate` tool_call is persisted.
5. Build the messages: first `user` turn `[image + instruction]`, then one assistant turn per action step, then the content-only `Done.` final; `finalize_use_messages` stamps tool-call `id`s and rewrites each intermediate screenshot-only `user` turn into a `role:"tool"` message keyed by the preceding batch's `id`. Assert `len(images) in {#assistant_turns, #assistant_turns + 1}`.
6. Stage as in understanding; drop the whole trajectory if any coordinate is OOB.

## 7. Error Handling

| Error type | Behavior |
|------------|----------|
| Wrong `task` field / unparseable bbox (understanding) | raise `ValueError` |
| JSON parse error (understanding line) | raise `ValueError` |
| Missing image (understanding) | skip the single record |
| Empty/missing `text` (understanding) | skip the single record |
| Missing/unresolvable image (`use`) | skip the whole episode (`SkipEpisodeError`) |
| Corrupt/unreadable image at staging | skip the affected record/episode and continue; count it |
| `NO_ACTION` (code 1) (`use`) | preserve as `wait(duration=1.0)` |
| Missing, non-finite, or non-positive duration for code 0/1 (`use`) | skip the whole episode (`SkipEpisodeError`; 0 in the current raw snapshot) |
| Any step after terminal code 10/11 (`use`) | skip the whole episode instead of silently truncating it |
| Unparseable `[y, x]` coordinate (`use`) | skip the whole episode (`SkipEpisodeError`) |
| Missing or corrupt `<id>.json`, or non-list episode payload (`use`) | fail loud with the annotation path; `process_raw_data.sh` also requires all 600 episode directories to contain their matching JSON |
| Empty `result_action_text` for TYPE (`use`) | raise `CAGUIActionError` (not seen in data) |
| Unmapped action code (`use`) | raise `CAGUIActionError` |
| Out-of-bound coordinate (`use`) | drop the whole trajectory, count it |
| Unparseable `ui_positions` (`use`) | store `[]` for that step (auxiliary, never skips) |

## 8. Output Format

See the parent [`AGENTS.md`](/lite/data/preproc/AGENTS.md) for the full schema.
All CAGUI rows have CUA platform dim `"mobile"`, `os="android"`,
`source="OpenBMB/CAGUI"`, `language="zh"`, and tagged CUA `metadata`.

Understanding row (cap):

```json
{
  "images": ["cua-lite/CAGUI/images/1e/1e03…c2c4.jpeg"],
  "messages": [
    {"role": "user", "content": [
      {"type": "image", "index": 0},
      {"type": "text", "text": "请描述截图中位于边界框 [870, 894, 950, 930] 内的 UI 元素的功能。"}
    ]},
    {"role": "assistant", "content": [
      {"type": "text", "text": "UI元素是一个菜单按钮。其主要功能是弹出一个菜单面板……"}
    ]}
  ],
  "metadata": {
    "metadata_kind": "cua",
    "dims": ["mobile", "understanding"],
    "extra_tool_schemas": [],
    "valid_actions": null,
    "others": {
      "id": "cagui_cap_0",
      "resolution": [1220, 2712],
      "os": "android",
      "source": "OpenBMB/CAGUI",
      "source_id": "cap/0.jpeg",
      "language": "zh"
    }
  }
}
```

`use` row. Actions are nested in a `mobile` batch wrapper
(`platform="mobile"`), and each post-action screenshot is a `role:"tool"` message
carrying the `tool_call_id` of the call it answers — never a `role:"user"` turn. This
episode ends on a code-10 (structural) terminator, so the final turn is a
content-only `Done.` and `extra_tool_schemas` stays `[]`. Note the counts: **4
actions, 5 images** — the terminator step's screenshot answers the last executable
action, so *every* assistant call `id` is paired. Both blocks in this section were
regenerated from the raw records and fed back through `validate_canonical_rows`;
an earlier version of this block showed 4 images and left `call_0003` unanswered,
which the gate **rejects**.

```json
{
  "images": [
    "cua-lite/CAGUI/images/b4/b44cc1fe6e06b98…jpeg",
    "cua-lite/CAGUI/images/19/199a7756f95892c…jpeg",
    "cua-lite/CAGUI/images/88/889c118e0cbe06b…jpeg",
    "cua-lite/CAGUI/images/04/047693fd368b4b7…jpeg",
    "cua-lite/CAGUI/images/a4/a4c9e28bc933f9…jpeg"
  ],
  "messages": [
    {"role": "user", "content": [
      {"type": "image", "index": 0},
      {"type": "text", "text": "饿了么找一下附近可以送外卖的刘大饼香辣土豆片夹馍"}
    ]},
    {"role": "assistant", "tool_calls": [
      {"id": "call_0000", "type": "function", "function": {"name": "mobile", "arguments": {"actions": [{"action": "tap", "coordinate": [148, 228], "clicks": 1}]}}}
    ]},
    {"role": "tool", "tool_call_id": "call_0000", "content": [{"type": "image", "index": 1}]},
    {"role": "assistant", "tool_calls": [
      {"id": "call_0001", "type": "function", "function": {"name": "mobile", "arguments": {"actions": [{"action": "tap", "coordinate": [571, 99], "clicks": 1}]}}}
    ]},
    {"role": "tool", "tool_call_id": "call_0001", "content": [{"type": "image", "index": 2}]},
    {"role": "assistant", "tool_calls": [
      {"id": "call_0002", "type": "function", "function": {"name": "mobile", "arguments": {"actions": [{"action": "type", "text": "刘大饼香辣土豆片夹馍"}]}}}
    ]},
    {"role": "tool", "tool_call_id": "call_0002", "content": [{"type": "image", "index": 3}]},
    {"role": "assistant", "tool_calls": [
      {"id": "call_0003", "type": "function", "function": {"name": "mobile", "arguments": {"actions": [{"action": "tap", "coordinate": [937, 60], "clicks": 1}]}}}
    ]},
    {"role": "tool", "tool_call_id": "call_0003", "content": [{"type": "image", "index": 4}]},
    {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}
  ],
  "metadata": {
    "metadata_kind": "cua",
    "dims": ["mobile", "use"],
    "extra_tool_schemas": [],
    "valid_actions": null,
    "others": {
      "id": "cagui_domestic_006b3e83f8da89719da76290ed590a3c",
      "resolution": [1220, 2712],
      "os": "android",
      "source": "OpenBMB/CAGUI",
      "source_id": "006b3e83f8da89719da76290ed590a3c",
      "language": "zh",
      "ui_positions": [[[30, 50, 265, 167], "…23 more [x1,y1,x2,y2] boxes on this screen…"], "…4 more lists, one per kept action step: 24/29/35/17/38 boxes…"]
    }
  }
}
```
