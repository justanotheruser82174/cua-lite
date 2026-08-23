---
description: 
globs: lite/data/preproc/scalecua/**
alwaysApply: false
---

# ScaleCUA Dataset Preprocessing Guide

## Overview

This document describes the preprocessing specifications for the ScaleCUA dataset (OpenGVLab/ScaleCUA-Data + zyliu/ScaleCUA-Data-Understanding). The preprocessed data follows the canonical format defined in [`lite/data/preproc/AGENTS.md`](/lite/data/preproc/AGENTS.md) and is written to the `cua-lite/ScaleCUA/` canonical layout via [`lite/data/staging`](/lite/data/staging.py).

**Related Files:**

| Type | File |
|------|------|
| Parent specification | [`lite/data/preproc/AGENTS.md`](/lite/data/preproc/AGENTS.md) |
| Adapter shared helpers | [`utils.py`](/lite/data/preproc/scalecua/utils.py) |
| Tool definitions | [`lite/core/tools/action_space/base.py`](/lite/core/tools/action_space/base.py) |
| Metadata configuration | [`meta.json`](/lite/data/preproc/scalecua/meta.json) |
| Preprocessing scripts | [`understanding.py`](/lite/data/preproc/scalecua/understanding.py), [`grounding-action.py`](/lite/data/preproc/scalecua/grounding-action.py), [`grounding-bbox.py`](/lite/data/preproc/scalecua/grounding-bbox.py), [`grounding-point.py`](/lite/data/preproc/scalecua/grounding-point.py), [`use.py`](/lite/data/preproc/scalecua/use.py) |
| Shell scripts | [`scripts/download_raw_data.sh`](/lite/data/preproc/scalecua/scripts/download_raw_data.sh) (download), [`scripts/process_raw_data.sh`](/lite/data/preproc/scalecua/scripts/process_raw_data.sh) (merge+extract), [`scripts/process_data.sh`](/lite/data/preproc/scalecua/scripts/process_data.sh) (run Python pipeline) |

**Source Data Locations** (annotation paths come from [`meta.json`](/lite/data/preproc/scalecua/meta.json)):

| Task Type | Base Path |
|-----------|-----------|
| Understanding | `${CUA_LITE_RAW_DATASETS_ROOT}/zyliu/ScaleCUA-Data-Understanding/{annotation}` |
| Grounding & Use | `${CUA_LITE_RAW_DATASETS_ROOT}/OpenGVLab/ScaleCUA-Data/{annotation}` |

`scripts/process_raw_data.sh` resumes both source shapes: split
`*.tar.gz.part-*` files and an already-merged `*.tar.gz` whose parts were
deleted. Parts are merged to a same-directory temporary file and atomically
renamed only after gzip validation; an interrupted/invalid merged file is rebuilt
when its parts still exist. The script writes the `.extracted` marker
only after `tar` succeeds; a truncated, concatenated-invalid, or directory-only
archive is an error, never an apparently successful empty extraction. Extraction does not
restore archived ownership, permissions, or timestamps, so a writable shared
raw root can resume through directories created by another user. Split parts are
deleted only after a successful extraction (or an explicit `--no-extract` merge);
an invalid or failed archive keeps them for recovery.

---

## Output Directory Structure

```
${CUA_LITE_DATASETS_ROOT}/cua-lite/ScaleCUA/
├── images/<hash[:2]>/<hash>.<ext>           # content-addressed image store
├── desktop/
│   ├── understanding/<split>/{caption,screen_transition,user_intention}.parquet
│   ├── grounding.action/<split>/<variant>.parquet
│   ├── grounding.bbox/<split>/<variant>.parquet
│   ├── grounding.point/<split>/<variant>.parquet
│   └── use/<split>/<variant>.parquet
├── mobile/
│   └── (same shape as desktop)
└── browser/
    └── (same shape as desktop)
```

`<split>` ∈ {`train`, `validation`}. Every cohort writes `<split>/<variant>.parquet` — one file per variant, whether the cohort has one (`grounding.action`, `grounding.bbox`, `grounding.point`, `use`) or several (`understanding`). Splits are assigned by [`SplitAssigner`](/lite/data/staging.py) keyed off `metadata.others.id`, with a 2,000-decision validation cap per cohort; content-identical row co-location can make the physical row count differ (for example, desktop `grounding.action` has 2,017 validation rows in the 2026-08-16 full run).

Because the split is a hash of `others.id`, **every adapter here keys the id on a
source position, never on a count of rows that survived filtering**: the three
grounding adapters and `understanding.py` use `f"{key}_{line_num}"` (the raw
annotation line index, which advances even for skipped lines) and `use.py` uses
`f"{key}_traj_{trajectory_id}"` (advancing on every merged trajectory, including
one the OOB gate later drops). A kept-row counter would renumber every later row
whenever upstream filtering changed, silently re-drawing the published
train/validation carve. None of these sources supplies an id of its own — the
understanding schema is exactly `{conversations, image, width, height}`.

---

## Data Statistics

| Partition | Desktop | | Mobile | | Browser | |
|-----------|--------:|--------:|--------:|--------:|--------:|--------:|
| | train | val | train | val | train | val |
| grounding.action | 561,301 | 2,017 | 110,455 | 2,000 | 292,752 | 1,996 |
| grounding.bbox | 335,203 | 2,163 | 106,509 | 1,999 | 226,969 | 2,000 |
| grounding.point | 104,695 | 2,000 | 3,732 | 77 | 72,334 | 1,533 |
| use | 75,279 | 1,469 | 21,312 | 398 | 79,089 | 1,711 |
| understanding/caption | 4,977 | 104 | 28,160 | 613 | — | — |
| understanding/screen_transition | 5,657 | 110 | 18,632 | 404 | 15,846 | 379 |
| understanding/user_intention | 5,648 | 113 | 18,649 | 387 | 15,915 | 310 |

Browser caption has no output — all upstream records have empty GPT responses (see below).

The 2026-08-16 full `grounding.action` run read 981,897 records and published
970,521. It skipped 11,242 records whose image root was absent on the audit
host, 123 authored coordinates outside the source bounds, and 11 corrupt host
images. All six partitions passed canonical validation with unique ids. The
desktop count includes 78,594 rows recovered by pointing the five
`windows_aug_action_grounding_20250616_{1..5}` cohorts at their actual
`windows_pure_paste/images` archive directory.

The 2026-08-16 full `grounding.bbox` run read 877,080 records and published
674,843. The remainder was 184,382 sibling point records, 17,438 records under
an image root absent on the audit host, 415 authored boxes outside `[0,1000]`,
and two rows referencing corrupt host images. All 674,843 published rows passed
the canonical validator and had unique ids; they reference 242,870 unique
content-addressed images.

### Upstream data quality: dense_caption empty responses (filtered)

Only `dense_caption` has empty GPT responses in the upstream raw JSONL files. `user_intention` and `screen_transition` are completely clean (0 empty across all 32 batches). All non-understanding data (grounding, navigation) is also clean (0 empty across all 201 entries).

The preprocessing adapter (`understanding.py`) skips records where the assistant response is empty. The skip count is printed at the end of each run.

**Per-batch breakdown** (only batches with empty responses shown; all others are 0%):

| Entry key | Platform | Total | Empty | Empty% |
|-----------|----------|------:|------:|-------:|
| `web_20250407_dense_caption_20250712` | browser | 31,505 | 31,505 | **100%** |
| `ubuntu_20250407_dense_caption_20250712` | desktop | 14,991 | 14,991 | **100%** |
| `mac_20250407_dense_caption_20250712` | desktop | 567 | 567 | **100%** |
| `android_20250407_dense_caption_20250712` | mobile | 7,762 | 2,407 | **31%** |
| `windows_20250612_dense_caption_20250805` | desktop | 1,289 | 259 | **20%** |
| `mac_20250612_dense_caption_20250805` | desktop | 1,370 | 1 | 0.1% |
| `android_20250612_dense_caption_20250703` | mobile | 3,559 | 5 | 0.1% |

**Pattern**: the `20250407` data collection batch (annotation date `20250712`) has a systematic annotation failure for `dense_caption` — web, ubuntu, and mac are 100% empty; android is 31% empty. The same batch's `user_intention` and `screen_transition` are unaffected. One exception: `windows_20250407_dense_caption_20250713` (annotation date `20250713`, not `20250712`) is 0% empty, suggesting the windows captions were annotated in a separate run.

**Aggregate impact on upstream raw data** (before filtering):

| Platform | Total caption rows | Empty rows | Empty% |
|----------|-------------------:|-----------:|-------:|
| browser | 31,505 | 31,505 | **100%** |
| desktop | 25,937 | 15,818 | **61%** |
| mobile | 46,535 | 2,412 | **5.2%** |

The latest full run over the shards available on the audit host read 234,100
understanding records and published 115,904: 83,634 were under image roots absent
from that host, 78 named individual missing images, and 34,484 available records
had empty assistant responses. The published caption totals are desktop 5,081,
mobile 28,773, and browser 0. These host-run counts are deliberately separate from
the source-wide empty-response table above: an unavailable record can also have
an empty response, but the adapter encounters and records the missing image root
first.

### Reproduction

```bash
uv run python -c "
from pathlib import Path
import pyarrow.parquet as pq
from lite.data.staging import iter_partitions
root = Path('${CUA_LITE_DATASETS_ROOT}/cua-lite/ScaleCUA')
for *_, f in iter_partitions(root):
    t = pq.read_table(f)
    print(f'{f.relative_to(root)}: {t.num_rows} rows')
"
```

---

## Source Data Format

All raw records share the same top-level schema:

```json
{
  "image": "<string or list[string]>",
  "conversations": [
    {"from": "human", "value": "<image>\n..."},
    {"from": "gpt", "value": "..."}
  ],
  "width": 2560,
  "height": 1440
}
```

- `image`: relative path resolved against the entry's `root` in `meta.json`. Always a string, except for `screen_transition` which uses a list of 2 strings.
- `conversations`: always single-turn (1 human + 1 gpt). Multi-step navigation trajectories are encoded as single conversation turns where "Previous operations" are text in the human prompt.

### GPT response format by task type

| Task | Response format | Coordinate system |
|------|----------------|-------------------|
| understanding/caption | Free-form text description | N/A |
| understanding/user_intention | Free-form text (intent prediction) | N/A |
| understanding/screen_transition | Free-form text (transition narration) | N/A |
| grounding.action | `<action>\nclick(x=FLOAT, y=FLOAT)\n</action>` | Normalized [0, 1] |
| grounding.bbox | `<ref>DESCRIPTION</ref><box>[[x1, y1, x2, y2]]</box>` | **Already [0, 1000]** (InternVL) |
| grounding.point | `<ref>DESCRIPTION</ref><point>[[x, y]]</point>` | **Already [0, 1000]** (InternVL) |
| use | `<operation>\nDESCRIPTION\n</operation>\n<action>\nACTION(args)\n</action>` | Normalized [0, 1] |
| use (planning) | `<think>\n...\n</think>\n<operation>\n...\n</operation>\n<action>\n...\n</action>` | Normalized [0, 1] |

The available `grounding.action` snapshot contains only executable click-like
labels.  The parser rejects mobile `scroll` and `key` labels rather than
inventing a mobile equivalent; desktop scroll labels, if encountered, use the
canonical default magnitude `amount=3`.

### Full raw record example (navigation)

```json
{
  "image": "settings/free_task_20250605_101622/images/20250605_101625_1.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nPlease generate the next move according to the UI screenshot, task and previous operations.\n\nTask: Turn on get notifications in system settings.\n\nPrevious operations:\nNone"
    },
    {
      "from": "gpt",
      "value": "<operation>\nClick on the Start button in the bottom left corner of the screen to open the Start menu\n</operation>\n<action>\nclick(x=0.0099, y=0.9833)\n</action>"
    }
  ],
  "width": 1920,
  "height": 1080
}
```

### Full raw record example (planning — navigation with CoT)

```json
{
  "image": "settings/free_task_20250605_101622/images/20250605_101625_1.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nPlease generate the next move according to the UI screenshot, task and previous operations.\n\nTask: Turn on get notifications in system settings.\n\nPrevious operations:\nNone"
    },
    {
      "from": "gpt",
      "value": "<think>\nLooking at the desktop screenshot, I can see a Windows desktop with a beautiful hummingbird wallpaper. To access notification settings, I need to first enter the Windows system settings...\n</think>\n<operation>\nClick on the Start button in the bottom left corner of the screen to open the Start menu\n</operation>\n<action>\nclick(x=0.0099, y=0.9833)\n</action>"
    }
  ],
  "width": 1920,
  "height": 1080
}
```

---

## Action Space & Mapping

### Source Action Space (from navigation/planning entries)

Source actions appear inside `<action>...</action>` tags. Coordinates are normalized to `[0, 1]`.

```python
click(x=0.5, y=0.5)                     # Left click
doubleClick(x=0.5, y=0.5)               # Double click
rightClick(x=0.5, y=0.5)                # Right click
scroll(clicks=-4, x=0.5, y=0.5)         # Desktop scroll — negative clicks = up, positive = down
swipe(direction='up', amount=0.7)        # Web scroll (touchscreen notation) — see scroll-direction note below
moveTo(x=0.5, y=0.5)                    # Move cursor (always paired with dragTo)
dragTo(x=0.5, y=0.5, button="left")     # Drag to position
press(keys="enter")                      # Single key press
hotkey("ctrl", "c")                      # Key combination
key("ctrl+s")                            # Raw chord string, desktop/browser only
key(keys=["ctrl", "s"])                  # Pre-tokenized key list, desktop/browser only
keyDown(key="ctrl")                      # Key-down, desktop/browser only
keyUp(key="ctrl")                        # Key-up, desktop/browser only
write(message="hello")                   # Type text
wait(seconds=3)                          # Wait
terminate(status="success")              # End task
```

### Mapping

Source coordinates `[0, 1]` → target coordinates `[0, 1000]`. That rescale applies
to the **pyautogui** action grammar only (`grounding.action`, `use`).

> **⚠️ `grounding.bbox` / `grounding.point` values are NOT pixels and must NOT be
> divided by `width`/`height`.** The InternVL `<box>` / `<point>` grammar is
> already `[0, 1000]`, so the adapters copy the integers through verbatim — see
> `parse_bbox_response` / `parse_point_response`. Measured over 20,466
> coordinate-bearing records sampled from all 69 `conv_style ==
> "internvl_grounding"` annotation files (≤ 300 per file), the max raw `x` is
> **1000** on 1920×1080 shots, **1000** on 5120×1440 shots and **925** on
> 7920×5200 shots — it saturates just under 1000 at every resolution instead of
> tracking image width, which is what absolute pixels would do. 17 records
> (0.083%) exceed 1000; those are upstream annotation noise and
> `has_oob_coordinate` drops them. "Normalizing" these by `width` would divide
> every label by ~2–8 and destroy the whole cohort. Reproduce with:
>
> ```
> uv run python -c "
> import json, os, re, pathlib
> raw = pathlib.Path(os.environ['CUA_LITE_RAW_DATASETS_ROOT'])/'OpenGVLab/ScaleCUA-Data'
> meta = json.load(open('lite/data/preproc/scalecua/meta.json'))
> pat = re.compile(r'<(?:box|point)>\[\[([0-9, ]+)\]\]</(?:box|point)>')
> mx = 0
> for k, v in meta.items():
>     if v.get('conv_style') != 'internvl_grounding': continue
>     p = raw/v['annotation']
>     if not p.exists(): continue
>     with p.open() as fh:
>         for i, line in enumerate(fh):
>             if i >= 300: break
>             c = (json.loads(line).get('conversations') or [None, {}])[1].get('value', '')
>             m = pat.search(c)
>             if m: mx = max(mx, *[int(t) for t in m.group(1).split(',')])
> print('max raw coordinate:', mx)   # 1532 -- the OOB tail, not width-scaled
> "
> ```

#### Desktop/Browser

| Source Action | Target Action |
|--------------|---------------|
| `click(x=0.5, y=0.5)` | `click(coordinate=[500, 500])` |
| `click(x, y, clicks=2)` | `click(coordinate=[...], clicks=2)` |
| `doubleClick(x, y)` | `click(coordinate=[...], clicks=2)` |
| `rightClick(x, y)` | `click(coordinate=[...], button="right")` |
| `scroll(clicks=-4, x, y)` *(desktop)* | `scroll(coordinate=[...], direction="up")` |
| `scroll(clicks=4, x, y)` *(desktop)* | `scroll(coordinate=[...], direction="down")` |
| `swipe(direction='up', amount)` *(web)* | `scroll(direction="down")` — **inverted**, see note |
| `swipe(direction='down', amount)` *(web)* | `scroll(direction="up")` — **inverted**, see note |
| `dragTo(x, y)` | `drag(coordinate=[...])`; the preceding source `moveTo` in the same ordered batch establishes the cursor/start position |
| `key(key)` / `press(key)` | `key(keys=[...])` |
| `hotkey(*keys)` | `key(keys=[...])` |
| `keyDown(key)` | `key_down(keys=[...])` |
| `keyUp(key)` | `key_up(keys=[...])` |
| `write(text)` | `type(text=text)` |
| `wait()` | `wait(duration=...)` |
| `terminate()` | dropped from `messages`; non-success status recorded as `others.terminate_status` |

`key`/`press`/`hotkey`/`keyDown`/`keyUp` route through the shared desktop key
factory, so stored key tokens are lowercase named keys or literal printable
glyphs. Raw chord strings such as `"ctrl+s"` are source-ingress syntax; a list
such as `["ctrl", "s"]` is already tokenized, and `["ctrl+s"]` is rejected as a
malformed token list. Mobile `use` trajectories reject keyboard actions with
`mobile_keyboard` instead of inventing a mobile equivalent — see the
[Lite key vocabulary](/lite/data/preproc/AGENTS.md#keyboard-keys--lowercase-named-keys-plus-printable-glyphs).

> **⚠️ Scroll-direction convention (read before touching scroll code).** ScaleCUA
> encodes scrolling **two ways with two different conventions**, both mapping to
> our single page/viewport convention (`direction="up"` = scroll toward the top /
> see higher content; `"down"` = toward the bottom):
>
> - **Desktop** — `scroll(clicks=N)`, pyautogui-style: **negative clicks = up**,
>   positive = down. (~1k nav actions; `negative→up` is correct — do **not** "fix"
>   it to match other adapters.)
> - **Web** — `swipe(direction=…, amount=…)`, touchscreen-style: swiping **up**
>   pushes content up to reveal what's below — i.e. the page scrolls **down**. So
>   source `swipe('up')` → our `scroll(direction="down")` (**inverted**), and
>   `swipe('down')` → `scroll("up")`. (~27k nav actions, **96%** of all nav scrolls.)
>
> Verified against the source: `<operation>Scroll down to view more…</operation>`
> pairs with `<action>swipe(direction='up')</action>`. The two-convention split is
> easy to get backwards — a naive "negative looks inverted, flip it" change breaks
> the desktop path while leaving the (dominant) web swipe path wrong.

#### Mobile

| Source Action | Target Action |
|--------------|---------------|
| `click(x=0.5, y=0.5)` | `tap(coordinate=[500, 500])` |
| `doubleClick(x, y)` | Two consecutive `tap` actions |
| `rightClick(x, y)` | `long_press(coordinate=[...])` |
| `scroll(clicks, x, y)` | rejected: ScaleCUA mobile use rows must not synthesize desktop scroll into an invented swipe |
| `dragTo(x, y)` | rejected: the source omits the swipe start coordinate |
| `write(text)` | `type(text=text)` |
| `wait()` | `wait(duration=...)` |
| `terminate()` | dropped from `messages`; non-success status recorded as `others.terminate_status` |

---

## Processing Workflow

### Step 1: Read Metadata Configuration

Read [`meta.json`](/lite/data/preproc/scalecua/meta.json), which maps entry keys to annotation paths and configuration. Each adapter filters entries by `conv_style`:

| Adapter | Selection criteria |
|---------|-------------------|
| `understanding.py` | `conv_style` contains `"understanding"` |
| `grounding-action.py` | Key contains `"action_grounding"` |
| `grounding-bbox.py` | `conv_style == "internvl_grounding"` **and the key names a platform**, then keeps rows with `<box>` in response |
| `grounding-point.py` | `conv_style == "internvl_grounding"` **and the key names a platform**, then keeps rows with `<point>` in response |
| `use.py` | Key contains `"navigation"` or `"planning"` |

Note: `grounding-bbox` and `grounding-point` share the same `internvl_grounding` pool and filter by response format at processing time. Each adapter skips the other's format.

#### Entry selection: the one excluded `internvl_grounding` entry

The platform half of the criterion above is a **whole-entry exclusion**, and it is
deliberate. `get_platform_type` derives the platform from keywords in the
`meta.json` key; exactly one entry in the `internvl_grounding` pool has no such
keyword, and both adapters print it by name on every run:

```
internvl_grounding entries: 69  selected: 68  excluded: 1
  EXCLUDED (unknown_platform): icon_internvl_grounding_20250328
```

Re-derive the pool and the exclusion with:

```bash
uv run python -c "
import json
meta = json.load(open('lite/data/preproc/scalecua/meta.json'))
kw = ('iphone', 'android', 'windows', 'mac', 'ubuntu', 'web')
pool = [k for k, v in meta.items() if v.get('conv_style') == 'internvl_grounding']
print(len(pool), 'in pool;', [k for k in pool if not any(w in k.lower() for w in kw)], 'excluded')
"
```

**Why it stays excluded** — it is not GUI grounding data, and it is not the same
task:

- The images are **synthetic blank canvases with clipart/emoji scattered on
  them** (`data/data_20250328/icon_canva/images/*_canvas.png`) — no application,
  no window chrome, no UI. `desktop_`/`mobile_` in the filenames is the canvas
  aspect ratio, which is also why the key carries no platform: there is no device.
- The task is **detect-all**, not locate-one. All 7 distinct user prompts ask for
  every element at once (*"Recognize all interactive components in the image and
  generate a bounding box around each one."*), and the response carries one
  `<ref>/<box>` pair **per element** — mean 9.51 pairs per record. The published
  `grounding.bbox` / `grounding.point` contract is a single-target *"Locate the
  element: {description}"* prompt answered by one `bbox`/`point` call.
- Because the parsers `re.search` only the **first** pair, admitting the entry
  as-is would keep 1 box of N and pair it with a prompt asking for all of them —
  a wrong label, which is worse than dropping the record.

Admitting it properly would mean a **new cohort** (multi-object detection over
synthetic canvases), not a wider allow-list — so the exclusion is the right call
and the run accounting below is what makes it visible.

### Step 2: Resolve Images and Detect OS

- Resolve each image path via `resolve_path(root / image, "CUA_LITE_RAW_DATASETS_ROOT")`. For understanding/grounding (single image), skip the record if missing. For screen_transition (two images), skip if either is missing. For use, skip the **entire trajectory** if any image is missing.
- **OS detection** from keywords in the `meta.json` entry key:

| Keywords in Key | Platform | OS |
|-----------------|----------|----|
| `"iphone"` or `"android"` | mobile | `"ios"` / `"android"` |
| `"windows"`, `"mac"`, `"ubuntu"` | desktop | `"windows"` / `"macos"` / `"ubuntu"` |
| `"web"` | browser | `null` |

### Step 3: Parse and Convert Actions/Responses

Per task type:
- **Understanding**: extract free-form text response. For `user_intention`, convert action definitions in the user prompt from source to target format (e.g. `click(x=0.5, y=0.5)` → `click(coordinate=[500, 500])`).
- **Grounding action**: parse `<action>...</action>` tags, convert pyautogui call to CUA-lite tool call with coordinate normalization `[0,1]` → `[0,1000]`.
- **Grounding bbox/point**: parse `<ref>...</ref><box>...</box>` or `<point>...</point>` tags. **No coordinate arithmetic happens here** — the InternVL integers are already `[0, 1000]` and are copied through verbatim; `width`/`height` are read only to publish `others.resolution`. A `<box>` record is emitted by `grounding-bbox.py` and a `<point>` record by `grounding-point.py`; each script skips the other's format (`is_bbox_format` / `is_point_format`), so there is no bbox→center derivation. Simplify user prompt to `"Locate the element: {description}"`.
- **Navigation/planning**: parse `<think>` (optional), `<operation>`, `<action>` tags. `<think>` → `inline_reasoning`, `<operation>` → `action_description`, `<action>` → tool_calls with coordinate normalization.

### Step 4: Build Messages

- **Understanding**: two-message conversation (user image+question → assistant text).
- **Grounding**: two-message conversation (user image+prompt → assistant tool_calls).
- **Use (multi-step rollout)**: multi-turn conversation reconstructed from the single raw conversation turn. Each step in "Previous operations" becomes an assistant-action turn plus a `role:"tool"` screenshot result whose `tool_call_id` matches that action's `id` (`finalize_use_messages`); only the first screenshot is a `role:"user"` turn. Steps with `<think>` tags get an `inline_reasoning` content part; all steps get an `action_description` content part.
- Remove `<image>\n` prefix from all user prompts.

### Step 5: Stage Entry

For each well-formed record:
1. Check for out-of-bound coordinates (`has_oob_coordinate`) — drop if any
2. Hash images into `ImageStore` via `stage_entry()`
3. Assign train/validation split via `SplitAssigner`
4. Buffer by `(metadata.dims[0], metadata.dims[1], split, variant)` key

### Step 6: Flush Buffers

Write all partition buffers to canonical parquet paths via `staging.flush_buffers()`.

---

## Error Handling

Following the guidelines in [`lite/data/preproc/AGENTS.md`](/lite/data/preproc/AGENTS.md).

| Error Type | Behavior |
|------------|----------|
| Missing image file (any image in sample/trajectory) | Skip whole sample/trajectory |
| JSON parse error | Raise `ValueError` |
| Invalid action/code format | Skip record or trajectory (task-dependent) |
| Ambiguous/malformed data | Skip with warning in verbose mode |
| Out-of-bound coordinates | Drop record/trajectory (counted, printed at end) |
| Corrupt/unreadable image | Skip the affected record/trajectory; continue the entry |

### Run accounting — the ledger must close

`understanding.py`, `grounding-bbox.py` and `grounding-point.py` print, **without
`--verbose`**:

```
Records read: R  kept: W  skipped: S        with   R == W + S
Skip reasons: {<reason>: n, ...}
```

`process_*_entry` returns `(rows, n_records, skips)` so the accounting travels
with the rows and a caller cannot report one without the other. Any remainder is
folded into `skips["unaccounted"]`, so a `continue` added later without its own
counter surfaces by name instead of vanishing. Two further unconditional lines:

- one `EXCLUDED (<reason>): <key>` per entry dropped at **selection** time (see
  [Entry selection](#entry-selection-the-one-excluded-internvl_grounding-entry));
- one `DROPPED ALL: <partition>/<key>: N records -> 0 rows ({reasons})` per entry
  that read records and wrote no row.

**`missing_image_root_absent` is kept distinct from `missing_image`** on purpose:
the former means the entry's whole image root is not on this host (an unextracted
`.tar.gz` shard), the latter that an individual file named by a present root is
gone. Only the second is an adapter loss; conflating them makes a host-data gap
read as a data bug. Reason vocabulary:

| Reason | Meaning |
|--------|---------|
| `missing_image` | image root present, this record's file absent |
| `missing_image_root_absent` | the entry's image root does not exist on this host |
| `image_corrupt_on_host` | image exists but Pillow cannot decode or verify it on this host |
| `empty_assistant_response` | upstream GPT response blank (understanding only) |
| `invalid_click_count` | source `clicks` is outside canonical single/double/triple click; values such as 40 are source value-entry leakage, not expanded into invented repeated clicks |
| `unsupported_key` | source key payload does not normalize to canonical `list[str]` tokens: literal printable glyphs or lowercase named keys; prose accidentally placed in `press(keys=...)` is not guessed from its wording |
| `mobile_keyboard` | mobile `use` source authored a desktop keyboard action (`key`, `press`, `hotkey`, `keyDown`, or `keyUp`); the adapter drops the trajectory instead of projecting it to a mobile-only action |
| `point_format` / `bbox_format` | the sibling adapter's format, from the shared `internvl_grounding` pool |
| `duplicate_step` | same task, source trajectory, history position, image, and action were already seen; redundant variant is dropped |
| `history_gap` | a trajectory jumps over one or more authored decision positions; drop the whole trajectory instead of pairing an earlier action with a later observation |
| `post_terminal_step` | source continues the same task/trajectory after an already completed `terminate`; the suffix cannot be part of that canonical episode |
| `unmatched_step` | continuation has no earlier open lane from the same task and source trajectory |
| `oob_coordinate` | dropped by `has_oob_coordinate` |
| `unaccounted` | a skip branch with no counter — should always be 0 |

Malformed JSON and missing/invalid required image fields are not skip reasons:
the adapter raises with the annotation path and line number, because silently
omitting an unreadable authored record would make the ledger look like a valid
quality filter.

---

## Output Format

See [`lite/data/preproc/AGENTS.md`](/lite/data/preproc/AGENTS.md) for the complete output format specification.

GUI primitives are **never** bare top-level calls: they are nested inside a
`computer` batch wrapper on CUA platform dims `"desktop"` and `"browser"`, and a
`mobile` wrapper on CUA platform dim `"mobile"`. In `use` rows each post-action screenshot is
a `role:"tool"` message carrying the `tool_call_id` of the call it answers — never a
`role:"user"` turn. See the canonical schema in
[§ Use of the parent contract](/lite/data/preproc/AGENTS.md#3-use-multi-step-rollout-tasks).

### Use (multi-step rollout) example

Real row `internal_ubuntu_navigation_boost_instruction_20250624_traj_0`
(text and image hashes truncated):

```json
{
  "images": [
    "cua-lite/ScaleCUA/images/a7/a705951c71c70e3071…png",
    "cua-lite/ScaleCUA/images/9c/9cf105292a2b736ebc…png"
  ],
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "index": 0},
        {"type": "text", "text": "Insert a blank line between the text \"Agent\" and \"123456\" on the first page in…"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "action_description", "text": "Press Enter at the end of the \"Agent\" text where a dot appears next to the wor…"}
      ],
      "tool_calls": [
        {"id": "call_0000",
         "type": "function",
         "function": {
           "name": "computer",
           "arguments": {"actions": [{"action": "key", "keys": ["enter"]}]}
         }}
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_0000",
      "content": [{"type": "image", "index": 1}]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "Done."}]
    }
  ],
  "metadata": {
    "metadata_kind": "cua",
    "dims": ["desktop", "use"],
    "extra_tool_schemas": [],
    "valid_actions": null,
    "others": {
      "id": "internal_ubuntu_navigation_boost_instruction_20250624_traj_0",
      "resolution": [1920, 1080],
      "os": "ubuntu",
      "source": "OpenGVLab/ScaleCUA-Data",
      "source_id": "internal_ubuntu_navigation_boost_instruction_20250624"
    }
  }
}
```

Planning entries additionally carry an `inline_reasoning` content part ahead of
`action_description`. A `use` row never persists a `terminate` tool_call:
non-success source terminate payload is preserved as `others.terminate_status`
(plus `others.terminate_reason` when the source authored non-blank text) via
[`terminate_outcome_others`](/lite/data/utils/messages.py). If terminal cleanup
leaves no executable EOF label, the row ends on content-only `Done.`; otherwise
the final executable action remains the EOF SFT label.

The 2026-08-16 full run publishes 179,258 `use` trajectories. The strict
history-continuity gate removes 1,549 of the 180,807 otherwise constructable
rows (0.857%) and accounts for all 12,563 source records in those available
gap trajectories under `history_gap`. A trajectory whose history is contiguous
but simply ends without `terminate` is not a gap: it is retained with its final
executable action as the EOF label.

No `response` tool_call is persisted either. A final `response(answer=…)` becomes
the content-only final. For the 314 trajectories that continue acting after a
`response` (1,190 response-only steps), the no-op source record and its screenshot
are removed, its text is prepended to the next executable turn's inline reasoning,
and that next action's own screenshot is retained. If several `response` records
follow the last executable action, only the last (the source's combined answer)
is retained; earlier partial answers are superseded rather than causing the whole
trajectory to be dropped. A non-empty successful
`terminate.info` is later source output and therefore takes precedence over a
terminal response; otherwise the final falls back to `Done.`.

### Grounding action example

```json
{
  "images": [
    "cua-lite/ScaleCUA/images/9c/9cad8eb6bc8fee0a58…png"
  ],
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "index": 0},
        {"type": "text", "text": "Click on the Outlook app icon in the taskbar to open your email client"}
      ]
    },
    {
      "role": "assistant",
      "tool_calls": [
        {"id": "call_0000",
         "type": "function",
         "function": {
           "name": "computer",
           "arguments": {"actions": [{"action": "click", "coordinate": [218, 986]}]}
         }}
      ]
    }
  ],
  "metadata": {
    "metadata_kind": "cua",
    "dims": ["desktop", "grounding.action"],
    "extra_tool_schemas": [],
    "valid_actions": null,
    "others": {
      "id": "windows_action_grounding_20250328_0",
      "resolution": [2560, 1440],
      "os": "windows",
      "source": "OpenGVLab/ScaleCUA-Data",
      "source_id": "windows_action_grounding_20250328"
    }
  }
}
```
