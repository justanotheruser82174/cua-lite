# GUIOdyssey Dataset Preprocessing Guide

Technical specification for the GUIOdyssey (`hflqf88888/GUIOdyssey`) adapter. For
how to run it, see [`README.md`](/lite/data/preproc/guiodyssey/README.md). For the
cross-dataset contract, see [`lite/data/preproc/AGENTS.md`](/lite/data/preproc/AGENTS.md).

## 1. Overview

GUIOdyssey is a long-horizon cross-app Android dataset — 8,334 task trajectories
over ~128k screenshots, each step carrying rich annotations. From the same raw
data the adapter emits two cohorts:

- **use** — one multi-step episode per trajectory.
- **understanding** — one screen-captioning row per step, from the per-step
  `description` (an independent screen-only OCR/visual description).

**Related files:**

| Type | File |
|------|------|
| Parent specification | [`lite/data/preproc/AGENTS.md`](/lite/data/preproc/AGENTS.md) |
| Adapter shared helpers | [`utils.py`](/lite/data/preproc/guiodyssey/utils.py) |
| Preprocessing scripts | [`use.py`](/lite/data/preproc/guiodyssey/use.py), [`understanding.py`](/lite/data/preproc/guiodyssey/understanding.py) |
| Shell scripts | [`scripts/download_raw_data.sh`](/lite/data/preproc/guiodyssey/scripts/download_raw_data.sh), [`scripts/process_raw_data.sh`](/lite/data/preproc/guiodyssey/scripts/process_raw_data.sh), [`scripts/process_data.sh`](/lite/data/preproc/guiodyssey/scripts/process_data.sh) |
| HF repo metadata | [`repo.json`](/lite/data/preproc/guiodyssey/repo.json) |

**Source data locations** (under `${CUA_LITE_RAW_DATASETS_ROOT}/hflqf88888/GUIOdyssey/`):

| Cohort | Source |
|--------|--------|
| use | `annotations/*.json` (one per episode) + `screenshots/{ep}_{step}.png` |
| understanding | same `annotations/*.json` — one row per step's `description` |

The adapter consumes an extracted tree and walks `annotations/*.json` directly
(not the lightweight `all_annot.json` summary). The upstream screenshot payload
is a multipart zip; extract it first as documented in the README.

## 2. Output Directory Structure

```
cua-lite/GUIOdyssey/
├── images/<hash[:2]>/<hash>.<ext>
└── mobile/
    ├── use/<split>/<variant>.parquet               # one row per episode
    └── understanding/<split>/<variant>.parquet     # one row per step
```

Both cohorts have one variant, so each writes a single `<split>/<variant>.parquet`.

## 3. Data Statistics

Full-run counts on the complete upstream snapshot (8,334 annotations / 127,893
screenshots):

| Cohort | variant | train | validation | total |
|---|---|---:|---:|---:|
| `mobile/use` | `use` | 8,146 | 184 | 8,330 |
| `mobile/understanding` | `understanding` | 125,893 | 2,000 | 127,893 |

The two cohorts total **136,223 rows**, **136,223 unique ids**, and **125,131
unique referenced images**; every image reference resolves. Understanding keeps
all 127,893 source descriptions. Use drops 4/8,334 trajectories (0.048%), each
because a `TEXT` step has an empty source `info` string; no missing/corrupt
images, OOB coordinates, or post-terminal suffixes occur.

Regenerate after a full run:

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
```

Skip / drop reasons:

| Reason | Cohort | Behavior |
|---|---|---|
| Missing image / bad device_info/task_info/instruction / empty low_level_instruction / unparseable action info / steps after a terminator | use | skip whole episode |
| OOB coordinate after normalization | use | drop whole trajectory (rare; coords are already [0,1000]) |
| Missing image / empty description / non-int step | understanding | skip single record |
| Unknown action or system-key string | use | raise (fail loud) |

## 4. Source Data Format

Each `annotations/{episode_id}.json`:

```json
{
  "episode_id": "6565803623678743",
  "device_info": {"product": "...", "w": 1080, "h": 2400, "device_name": "Medium Phone"},
  "task_info": {"category": "Social_Sharing", "app": ["Chrome", "X"],
                "meta_task": "...", "task": "...", "instruction": "Using Chrome, search ..."},
  "step_length": 12,
  "steps": [
    {"step": 0, "screenshot": "6565803623678743_0.png", "action": "CLICK",
     "info": [[374, 621], [374, 621]], "ps": "[(374, 621)]",
     "description": "This is a screenshot of an Android home screen ...",
     "intention": "I am opening the Chrome browser to ...",
     "low_level_instruction": "Open the Chrome browser.",
     "context": "Task just started, nothing has been done."}
  ]
}
```

- **`info` coordinates are already normalized to [0, 1000]** (per the upstream README), NOT device pixels — used directly. `device_info.w/h` is kept only as `others.resolution`.
- `action` ∈ {CLICK, TEXT, SCROLL, LONG_PRESS, COMPLETE, INCOMPLETE}.
- `info` shape: two-point list for CLICK/LONG_PRESS/SCROLL, a system-key string (`"KEY_HOME"`) for system-button CLICK, a text string for TEXT, and **always `""` for COMPLETE/INCOMPLETE** (the upstream README: *"if action is any other value, info is empty (`""`)"*; measured over all 8,334 annotation files: 0/472 INCOMPLETE steps have non-blank `info`).
- `description` — independent screen caption (→ understanding). `intention` — first-person reasoning (→ `inline_reasoning`). `low_level_instruction` — short imperative (→ `action_description`). `context` — unused. `ps` — per the README, *"additional details or context depending on the value of the action field"*: coordinate debug text on action steps (unused), and on **COMPLETE/INCOMPLETE the annotator's why note** — the only place the INCOMPLETE reason exists (472/472 non-blank, median 17 chars, e.g. `"No flights"`, `"Item not found"`), so `use` reads `others.terminate_reason` from `ps`, not `info`.

## 5. Action Space & Mapping

Mobile action space — `LiteMobileActionSet` in [`lite/core/tools/action_space/base.py`](/lite/core/tools/action_space/base.py). The `system_button` enum was extended with `"Recent"` for `KEY_APPSELECT` (Android recent-apps / task switcher).

| Source `action` / `info` | Target tool |
|---|---|
| `CLICK [[x,y],[x,y]]` | `tap(coordinate=[x, y])` |
| `CLICK "KEY_HOME" / "KEY_BACK" / "KEY_APPSELECT"` | `system_button(button="Home" / "Back" / "Recent")` |
| `TEXT "<str>"` | `type(text=...)` |
| `SCROLL [[sx,sy],[ex,ey]]` | `swipe(start_coordinate=[sx,sy], coordinate=[ex,ey])` |
| `LONG_PRESS [[x,y],[x,y]]` | `long_press(coordinate=[x, y])` |
| `COMPLETE` / `INCOMPLETE` | terminator → outcome recorded in `metadata.others` (no persisted tool_call) |

**Terminator handling** (matches cagui/opencua): COMPLETE/INCOMPLETE steps are not emitted as their own turns; when at least one real action precedes the terminator, that terminator screenshot is kept as the post-action result image, so `len(images) in {#assistant_turns, #assistant_turns + 1}` is asserted. A `use` row **never persists a `terminate` tool_call** — every episode ends on the content-only assistant `Done.` final, and the terminator's payload moves to `metadata.others` via [`terminate_outcome_others`](/lite/data/utils/messages.py): `INCOMPLETE` → `others.terminate_status: "failure"` (plus `others.terminate_reason` from the step's **`ps`** — the annotator's why note — which is non-blank on 472/472 INCOMPLETE steps, so the reason is emitted on every failure row); `COMPLETE` records nothing, because `status="success"` asserts no more than the `Done.` final already does. `extra_tool_schemas` therefore stays `[]`.

**Assistant turn content** is built with [`make_assistant_content`](/lite/core/messages/content.py): `inline_reasoning=intention` + `action_description=low_level_instruction`, plus `tool_calls`.

## 6. Processing Workflow

1. Glob `annotations/*.json` (sorted); raise on zero matches.
2. Per episode, validate `device_info.{w,h}` and `task_info.instruction`; skip episode on failure.
3. Walk steps in order; require COMPLETE/INCOMPLETE to be the final source step, then remember status and (for INCOMPLETE) the reason read from `ps` (drop the step).
4. Map each action to tool_calls (coords used directly — already [0,1000]); resolve the step screenshot (skip episode if missing).
5. Build the messages: first `user` turn `[image + instruction]`, one assistant turn per action step (each carrying a single `mobile` batch), then the content-only `Done.` final; `finalize_use_messages` stamps tool-call `id`s and rewrites each intermediate screenshot-only `user` turn into a `role:"tool"` message keyed by the preceding batch's `id`. The terminator's outcome goes to `metadata.others`, not to a tool_call.
6. `has_oob_coordinate` over the whole trajectory → drop trajectory if any OOB.
7. `stage_entry` hashes images, fills metadata defaults, assigns a split (hash on `id`, 2000-row val cap per cohort); `flush_buffers` writes one parquet per partition.
8. Understanding: per step, pick a deterministic caption prompt and emit `(prompt → description)`.

## 7. Error Handling

| Error type | Cohort | Behavior |
|---|---|---|
| Missing image | use | skip whole episode |
| Missing image | understanding | skip single record |
| Corrupt/unreadable image | understanding / use | skip record / whole trajectory and continue |
| Missing device_info/task_info/w/h/instruction | use | skip whole episode |
| Bad device_info/task_info | understanding | skip all the episode's steps |
| Empty low_level_instruction | use | skip whole episode |
| Empty/missing description | understanding | skip single record |
| Unparseable CLICK/SCROLL/LONG_PRESS/TEXT info | use | skip whole episode |
| Step after COMPLETE/INCOMPLETE | use | skip whole episode instead of silently truncating it |
| Unknown action / system-key string | use | raise `GUIOdysseyActionError` |
| Corrupt JSON or non-object annotation file | both | fail loud with source path before run accounting |

**Annotation quality caveat.** A small fraction of `description`/`intention`
fields appear recycled from a different episode (e.g. a George-Washington task
step describing a Steve-Jobs page). Inherent to the source; accepted without
filtering since `low_level_instruction` is clean.

## 8. Output Format

`use` row (real row `guiodyssey_0019778284340925`; reasoning text and image
hashes truncated). Actions are nested in a `mobile` batch wrapper, each
post-action screenshot is a `role:"tool"` message keyed by the batch's `id`,
and no `terminate` tool_call is persisted — this episode's `INCOMPLETE`
terminator surfaces as `others.terminate_status` plus the annotator's
`others.terminate_reason` (from the terminator step's `ps`).

Note the counts: **3 actions, 4 images**. The terminator step's screenshot is
retained as the `role:"tool"` result for the last executable action (§5/§6), so
*every* assistant call `id` is answered before the content-only final. The block below was
regenerated from the raw episode and then fed through `validate_canonical_rows`;
an earlier version showed 3 images and left `call_0002` unanswered, which the gate
**rejects** — do not copy that shape. Only image hashes and the long
`inline_reasoning` / goal strings are abbreviated at `…`.

```json
{
  "images": [
    "cua-lite/GUIOdyssey/images/01/016a64d386962b…png",
    "cua-lite/GUIOdyssey/images/94/948a3a5f690923…png",
    "cua-lite/GUIOdyssey/images/9c/9ce339dda5dd53…png",
    "cua-lite/GUIOdyssey/images/a1/a1a4e8e591b8a7…png"
  ],
  "messages": [
    {"role": "user", "content": [
      {"type": "image", "index": 0},
      {"type": "text", "text": "Locate a nearby sports arena and then use the Google Play Store to download a fitness tr…"}
    ]},
    {"role": "assistant",
     "content": [
       {"type": "inline_reasoning", "text": "I am scrolling through the home screen to locate the necessary apps for my tasks. This a…"},
       {"type": "action_description", "text": "Scroll through the home screen."}
     ],
     "tool_calls": [
       {"id": "call_0000",
        "type": "function",
        "function": {
          "name": "mobile",
          "arguments": {"actions": [{"action": "swipe", "start_coordinate": [274, 516], "coordinate": [649, 538]}]}
        }}
     ]},
    {"role": "tool", "tool_call_id": "call_0000", "content": [{"type": "image", "index": 1}]},
    {"role": "assistant",
     "content": [
       {"type": "inline_reasoning", "text": "I am choosing to click on the Lyft app to explore transportation options to a nearby spo…"},
       {"type": "action_description", "text": "Open the Lyft app."}
     ],
     "tool_calls": [
       {"id": "call_0001",
        "type": "function",
        "function": {
          "name": "mobile",
          "arguments": {"actions": [{"action": "tap", "coordinate": [858, 640], "clicks": 1}]}
        }}
     ]},
    {"role": "tool", "tool_call_id": "call_0001", "content": [{"type": "image", "index": 2}]},
    {"role": "assistant",
     "content": [
       {"type": "inline_reasoning", "text": "I am choosing to tap the 'Get started' button to initiate the process of booking a ride …"},
       {"type": "action_description", "text": "Tap the 'Get started' button."}
     ],
     "tool_calls": [
       {"id": "call_0002",
        "type": "function",
        "function": {
          "name": "mobile",
          "arguments": {"actions": [{"action": "tap", "coordinate": [518, 861], "clicks": 1}]}
        }}
     ]},
    {"role": "tool", "tool_call_id": "call_0002", "content": [{"type": "image", "index": 3}]},
    {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}
  ],
  "metadata": {
    "metadata_kind": "cua", "dims": ["mobile", "use"],
    "extra_tool_schemas": [], "valid_actions": null,
    "others": {"id": "guiodyssey_0019778284340925", "resolution": [1440, 3120], "os": "android",
               "source": "hflqf88888/GUIOdyssey", "source_id": "0019778284340925",
               "category": "Information_Management", "apps": ["Google Play Store", "Lyft"],
               "device_name": "Pixel 7 Pro", "terminate_status": "failure",
               "terminate_reason": "Region limitation"}
  }
}
```

Understanding row:

```json
{
  "images": ["cua-lite/GUIOdyssey/images/fb/fb6db515e4432c…png"],
  "messages": [
    {"role": "user", "content": [{"type": "image", "index": 0}, {"type": "text", "text": "Describe this mobile screenshot in detail."}]},
    {"role": "assistant", "content": [{"type": "text", "text": "This is a screenshot of an Android home screen displaying various app icons such as eBay…"}]}
  ],
  "metadata": {
    "metadata_kind": "cua", "dims": ["mobile", "understanding"],
    "extra_tool_schemas": [], "valid_actions": null,
    "others": {"id": "guiodyssey_0000628297785099_0", "resolution": [1440, 3120], "os": "android",
               "source": "hflqf88888/GUIOdyssey", "source_id": "0000628297785099_0",
               "category": "Web_Shopping", "apps": ["Agoda", "Threads"], "device_name": "Pixel 7 Pro"}
  }
}
```
