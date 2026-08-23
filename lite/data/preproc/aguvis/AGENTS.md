# Aguvis Dataset Preprocessing Guide

Technical specification for the Aguvis (`xlangai/aguvis-stage1` + `stage2`) adapter.
For how to run it, see [`README.md`](/lite/data/preproc/aguvis/README.md). For the
cross-dataset contract, see [`lite/data/preproc/AGENTS.md`](/lite/data/preproc/AGENTS.md).

## 1. Overview

Aguvis is a large-scale composite GUI dataset using a unified **PyAutoGUI action
format** across mobile / browser / desktop. Both stages land in **one** cua-lite
dataset `Aguvis`, with the upstream sub-dataset carried as the cohort *variant*:

- **Stage 1 → `grounding.action`** — single-step locate-and-click (8 sub-datasets).
- **Stage 2 → `use`** — multi-step trajectories (5 sub-datasets).

**Related files:**

| Type | File |
|------|------|
| Parent specification | [`lite/data/preproc/AGENTS.md`](/lite/data/preproc/AGENTS.md) |
| Adapter shared helpers + PyAutoGUI parser | [`utils.py`](/lite/data/preproc/aguvis/utils.py) |
| Preprocessing scripts | [`grounding-action.py`](/lite/data/preproc/aguvis/grounding-action.py), [`use.py`](/lite/data/preproc/aguvis/use.py) |
| Shell scripts | [`scripts/download_raw_data.sh`](/lite/data/preproc/aguvis/scripts/download_raw_data.sh), [`scripts/process_raw_data.sh`](/lite/data/preproc/aguvis/scripts/process_raw_data.sh), [`scripts/process_data.sh`](/lite/data/preproc/aguvis/scripts/process_data.sh) |
| HF repo metadata | [`repo.json`](/lite/data/preproc/aguvis/repo.json) |

**Source data locations** (under `${CUA_LITE_RAW_DATASETS_ROOT}/`):

| Stage | Sub-datasets → variant (platform) | JSON | Image dir |
|---|---|---|---|
| 1 (`xlangai/aguvis-stage1`) | seeclick, seeclick_mi (browser) | `seeclick.json`, `seeclick_mi.json` | `seeclick/seeclick_web_imgs` |
| 1 | guienv (browser) | `guienv.json` | `guienvs/images` |
| 1 | webui (browser) | `webui350k.json` | `webui350k/images` |
| 1 | ricosca, rico_icon, widget_captioning, ui_refexp (mobile) | `ricosca.json`, `ricoig16k.json`, `widget_captioning.json`, `ui_refexp.json` | `<name>/images` |
| 1 | omniact (desktop) | `omniact_fix.json` | `omniact/images` |
| 2 (`xlangai/aguvis-stage2`) | android_control, aitw, coat (mobile) | `android_control.json`, `aitw-l1.json`, `coat.json` | `<name>/images`, `aitw-v1/images` |
| 2 | guide, miniwob (browser) | `guide.json`, `miniwob-l1.json` | `guide-v2/images`, `miniwob/images` |

**Excluded** (already processed from their original sources by other adapters):
GUIAct → `guiact/`, AMEX → `ui_genie/`, Multimodal-Mind2Web → `multimodal_mind2web/`,
GUI-Odyssey → `guiodyssey/`. `aitz` is listed in the upstream docs but has **no
file on disk**, so it is not produced. Stage-2 uses the **l1** annotation level.

## 2. Output Directory Structure

```
cua-lite/Aguvis/
├── images/<hash[:2]>/<hash>.<ext>
├── browser/
│   ├── grounding.action/<split>/{seeclick,seeclick_mi,guienv,webui}.parquet
│   └── use/<split>/{guide,miniwob}.parquet
├── mobile/
│   ├── grounding.action/<split>/{ricosca,rico_icon,widget_captioning,ui_refexp}.parquet
│   └── use/<split>/{android_control,aitw,coat}.parquet
└── desktop/
    └── grounding.action/<split>/<variant>.parquet                 # variant: omniact
```

## 3. Data Statistics

Regenerate after a full run:

```bash
uv run python -c "
from pathlib import Path
from datasets import load_dataset
from lite.data.staging import iter_partitions
root = Path('${CUA_LITE_DATASETS_ROOT}/cua-lite/Aguvis')
for *_, f in iter_partitions(root):
    d = load_dataset('parquet', data_files=str(f), split='train')
    print(f'{f.relative_to(root)}: {len(d)} rows')
"
```

`use.py` prints its run accounting **unconditionally**, denominated in SOURCE
RECORDS — one row is a whole episode and consumes many records, so records are
the only unit in which a run reconciles:

```
Records read: R  kept: K  skipped: S      with   R == K + S
Skip reasons: {<reason>: n, ...}
```

An unbucketed drop path added later surfaces as `unaccounted` rather than
vanishing. In the 2026-08-16 full run, `guide` read **13,544 records / 973
episodes** and published 967 rows; six episodes (100 source records) were
skipped because their argument-less `scroll()` had no authored direction. The
full five-cohort use run read 128,997 records and published 21,549 rows: 13,586
`android_control`, 2,330 `aitw`, 1,904 `coat`, 967 `guide`, and 2,762 `miniwob`.
It accounted for all 741 skipped source records as `unsupported_action` (145),
`too_few_images` (16), `no_image_name` (88), or `partial_episode` (492).

Skip / drop reasons — the `use` names are the literal `skips` keys:

| Reason | Cohort | Behavior |
|---|---|---|
| Missing image | grounding | skip record |
| `missing_image` / `bad_turns` / `empty_goal` / `partial_episode` | use | skip whole episode; `partial_episode` rejects a COAT fragment whose first record already names prior actions |
| `missing_image_dir_absent` | use | as above, but the subset's whole image directory is not on this host — the **host lacking data**, not the adapter dropping a row |
| Unparseable / unsupported PyAutoGUI call (`unsupported_action`) | both | skip record (grounding) / skip episode (use) |
| `empty_action_code` / `no_tool_calls` | use | skip episode |
| `post_terminal_steps` | use | skip episode instead of silently truncating a suffix after `terminate` (0 in the full source) |
| OOB coordinate (`oob_coordinate`) | both | drop row (grounding) / drop trajectory (use) |
| Corrupt/unreadable image (`corrupt_image`) | grounding / use | drop row / trajectory and continue |
| No images after terminal cleanup (`too_few_images`) | use | skip episode |
| `no_image_name` / `beyond_head` | use | never grouped into an episode / outside the `--head` prefix |
| No terminator step | use | **kept** — the final executable action remains the EOF SFT label. This is the normal ending for `android_control` / `coat` / `guide` / `miniwob`, which carry no terminator at all |

## 4. Source Data Format

**Stage 1** — one record per sample:
```json
{"image": "screenshot.jpg",
 "conversations": [{"from":"human","value":"<image>\nClick on the search bar"},
                   {"from":"gpt","value":"pyautogui.click(x=0.8018, y=0.6183)"}]}
```
**Multi-turn records.** `seeclick` (94.7% multi-turn, ~16 pairs/record), `webui`
(~49%), and `rico_icon` (~43%) pack several `(instruction → action)` grounding
pairs on **one** screenshot. Each pair is emitted as its own `grounding.action`
sample — `human[i]` paired with the following `gpt[i]` (NOT `human[0]` with
`gpt[-1]`, which mismatches the instruction with the wrong action). The other
five Stage-1 subsets are single-turn. `seeclick_mi` packs `"image": [a, b, c]`
with one image per pair (alternating human/gpt). Row id is
`aguvis_<subset>_<record_idx>_<pair_idx>`.

**Stage 2** — one record per step; action split across two gpt turns:
```json
{"image": "<ep_id>_<step>.jpg",
 "conversations": [{"from":"system","value":"..."},
                   {"from":"human","value":"<image>\n...Instruction: <goal>\n\nPrevious actions:\n..."},
                   {"from":"gpt","value":"Action: Go back to the previous screen\n"},
                   {"from":"gpt","value":"mobile.back()"}]}
```
COAT's first gpt turn is a verbose `Observation: ... Thought: ... Action: ...` block.

## 5. Action Space & Mapping

PyAutoGUI is parsed via `ast.parse` (`utils.pyautogui_to_tool_calls`), platform-aware
(mobile → `LiteMobileActionSet`, desktop/browser → `LiteDesktopActionSet`). Coordinates
are normalized [0,1] → `int(round(v*1000))`, clamped to [0,1000].

| Source call | Mobile | Desktop / Browser |
|---|---|---|
| `click(x,y)` | `tap` | `click` |
| `rightClick`/`middleClick`/`doubleClick`/`tripleClick` | — | `click(button=…/clicks=…)` |
| `moveTo` | — | `mouse_move` |
| `dragTo(x,y[,button])` | reject: source omits the swipe start, so inventing `[500,500]` is not faithful | `drag(button)` |
| `mobile.swipe(from_coord,to_coord)` | `swipe(start,end)` | — |
| `scroll(page=±0.1)` | synthesized vertical `swipe` | `scroll(direction, amount=1)` |
| `scroll(<wheel clicks>)` | **raise** (no wheel on a touchscreen) | `scroll(direction, amount=|clicks|)` |
| `scroll()` — **no argument** | direction read from the step's own `Action:` prose → the same swipe | same, `amount=1` |
| `hscroll` | — | `scroll(left/right)` |
| `write(msg)` / `typewrite` | `type` | `type` |
| `press(key)` / `press(keys=…)` | `system_button` if mappable else `type` | `key(keys=[normalized_token])` |
| `hotkey(a,b)` / `hotkey(keys=[a,b])` | mapped system buttons when every key has a mobile equivalent; otherwise reject the episode | `key(keys=[...])` |
| `mobile.home/back/recent/menu/enter` | `system_button("Home"/…/"Recent"/…)` | — |
| `open_app(name)` | `open_app` | — |
| `wait/sleep` | `wait(duration)` | `wait(duration)` |
| `long_press(x,y[,duration])` | `long_press` | — |
| `terminate(status[,reason])` | terminal outcome | terminal outcome (`"fail"`→`"failure"`) |
| `answer/response(text)` | `response` | `response` |
| `select_option` / anything else | **raise** `AguvisParseError` | **raise** |

**Note:** `hotkey(keys=…)` and `press(keys=…)` keyword forms are handled in
addition to their positional forms. `"Recent"` was added to the mobile
`system_button` enum (shared with the GUIOdyssey adapter).
Desktop/browser keys use the shared canonical vocabulary: side-specific modifier
spellings such as `winleft` collapse to `meta`, while an unknown
multi-character token rejects the source row/episode instead of surviving until
backend execution.

### `scroll` has THREE source encodings, not two

Re-derive with
`python -c "…"`-style AST census over the raw files, never from this table alone:

| encoding | where | count | what the code carries |
|---|---|---|---|
| `scroll(page=±0.1)` / `hscroll(page=±0.1)` | `android_control.json` only | 9,833 | direction only — `page` never takes a third value |
| `scroll(<int>)` | `omniact_fix.json` only | 168 | direction + wheel clicks |
| **`scroll()`** | **`guide.json` only** | **392 calls in 254 of its 973 episodes** | **nothing — not even the direction** |

The bare form's direction is stated in the same step's first gpt turn, the text
the row already publishes as its `action_description`. Measured over all 392:
**386 resolve to `down`, 6 to nothing, and 0 to `up` or to both**; `download`
(33 occurrences) cannot be misread as `down` because the match is
word-bounded. Cross-checked against the pixels — the vertical content shift
between each step's screenshot and the next — the text agrees at **23/23**
where the pixel evidence is confident (`ZNCC ≥ 0.7`) and **35/37** at
`≥ 0.6`, against a detector ceiling of **92.8% / 84.9%** measured the same way
on 400 *labelled* `android_control` scrolls; one of the two residual
disagreements was inspected by eye and is a detector artifact (a repeating
Canva template grid), not a wrong label. Both magnitudes are
`DIRECTION_ONLY_SWIPE_TRAVEL` — one direction-only gesture, not two.

The 6 that state no direction are refused, not inherited. Two of them say only
*"continue scrolling"*, and their previous step does carry an explicit `down`
— but the other four describe **typing**, not scrolling at all, so
"inherit from the previous step" is a rule whose preconditions this corpus does
not uniformly meet. Refusing costs 6 records / 6 episodes.

**`guide.json` has a second argument-less encoding and it is the one that
actually binds:** `doubleClick()` with no coordinate, **730 calls in 365
episodes**, and those 365 are a strict superset of the 254 above. All 730 have a
prior cursor-setting click in the same episode, so the parser tracks PyAutoGUI's
cursor and emits the no-argument `doubleClick()` at that coordinate. With both
scroll direction and cursor recovery, the measured GUIDE corpus publishes 966
of 973 episodes; the six directionless scroll cases remain refused.

### Stage-2 inner monologue
- first gpt turn `"Action: <desc>"` → `action_description` content part.
- COAT `"...Thought: <t> ... Action: <a>"` → `inline_reasoning=<t>` + `action_description=<a>`.
- last gpt turn (`pyautogui.*`/`mobile.*`) → `tool_calls`.
- a source `terminate` step is never persisted as a tool_call. If terminal
  cleanup leaves no executable EOF action, the episode ends on a content-only
  `Done.` final; a non-success status is preserved as
  `metadata.others.terminate_status` (plus `terminate_reason` when the source
  authored non-blank text). If an executable action remains at EOF, keep it as
  the supervised final label instead of fabricating `Done.` behind it.

## 6. Processing Workflow

1. Load each sub-dataset JSON.
2. **Grounding**: per record, clean the instruction (strip `<image>`), resolve the image (skip if missing), parse the gpt PyAutoGUI call → tool_calls, build `user(image+instr) → assistant(tool_calls)`.
3. **Use (multi-step rollout)**: group and sort source records, reject continuation fragments, and batch consecutive actions that share the same source screenshot before creating a single tool-result boundary. Track the desktop/browser cursor for coordinate-free PyAutoGUI calls, then apply the EOF policy and `finalize_use_messages`.
4. `has_oob_coordinate` → drop row/trajectory if any coord OOB.
5. `stage_entry` hashes images, normalizes tagged CUA metadata, assigns split (hash on `id`, 2000-row val cap per cohort); `flush_buffers` writes one parquet per `(metadata.dims[0], metadata.dims[1], split, variant)`.

MiniWoB's upstream JSON stores one next-move target per record, but its screenshot
step is not always its action index. Full-source verification finds 56/2,762
episodes where 161 later records advance `Previous actions` strictly by one while
retaining the same `(step, image)` observation; all 161 actions differ. Fifty are
`click-shades`: the next screenshot visibly contains the cumulative result of all
same-observation clicks (for `seed12`, five selected squares appear together).
The remaining six are `unicode-test` (4), `click-button` (1), and
`use-autocomplete` (1), whose annotations are noisier but still contain distinct,
ordered actions rather than duplicate records. The adapter batches on the explicit
source observation key only; an identical path at a different source step remains a
real tool-result boundary.

## 7. Error Handling

The `use` cohort's reasons are owned by §3's table (they are the `skips` keys the
run prints); this table adds only what is not a counted skip:

| Error type | Behavior |
|---|---|
| Missing image (grounding) | skip record, count |
| Final action has no post-action screenshot (`use`) | keep that action as the EOF SFT label — not a skip, so it has no bucket |
| Subset JSON absent on this host | one `SOURCE ABSENT ON THIS HOST` line; 0 records read, so it denominates nothing |

## 8. Output Format

Actions are **never** bare top-level calls: they are nested inside a
`mobile` batch wrapper on mobile rows and a `computer` wrapper on desktop/browser
rows. In `use` rows each non-final post-action screenshot is a `role:"tool"`
message carrying the `tool_call_id` of the call it answers. A final EOF action may
have no `role:"tool"` result, and no `terminate` tool_call is persisted (the
outcome moves to `metadata.others` — see
[`terminate_outcome_others`](/lite/data/utils/messages.py)).

Both blocks below follow adapter output that has been fed back through
`validate_canonical_rows`. The grounding row is a real row; the `use` row keeps
the same message shape but abbreviates final-action details because the point is
the EOF contract.

Grounding row (real row `aguvis_ricosca_0_0`):
```json
{"images": ["cua-lite/Aguvis/images/65/65c508f3210c0b…jpg"],
 "messages": [
   {"role":"user","content":[{"type":"image","index":0},{"type":"text","text":"select icon below zip code:"}]},
   {"role":"assistant","tool_calls":[
     {"id":"call_0000","type":"function","function":{"name":"mobile","arguments":{"actions":[{"action":"tap","coordinate":[510,467],"clicks":1}]}}}]}],
 "metadata": {"metadata_kind":"cua","dims":["mobile","grounding.action"],"extra_tool_schemas":[],"valid_actions":null,
   "others":{"id":"aguvis_ricosca_0_0","resolution":[540,960],"os":"android","source":"xlangai/aguvis-stage1","source_id":"34571"}}}
```

`use` row shape (goal text and final-action details truncated):
```jsonc
{"images": ["cua-lite/Aguvis/images/b3/b31150fe668b5e…png","cua-lite/Aguvis/images/33/33a9b3a9e13ce7…png"],
 "messages": [
   {"role":"user","content":[{"type":"image","index":0},{"type":"text","text":"If I lose connection to the internet, I want to make the agents.txt fi…"}]},
   {"role":"assistant",
    "tool_calls":[{"id":"call_0000","type":"function","function":{"name":"mobile","arguments":{"actions":[{"action":"tap","coordinate":[941,438],"clicks":1}]}}}],
    "content":[{"type":"action_description","text":"Click on the three dots next to the agents.txt"}]},
   {"role":"tool","tool_call_id":"call_0000","content":[{"type":"image","index":1}]},
   {"role":"assistant",
    "tool_calls":[{"id":"call_0001","type":"function","function":{"name":"mobile","arguments":{"actions":[{"action":"tap","coordinate":[…],"clicks":1}]}}}],
    "content":[{"type":"action_description","text":"…"}]}],
 "metadata": {"metadata_kind":"cua","dims":["mobile","use"],
   "extra_tool_schemas":[], "valid_actions":null,
   "others":{"id":"aguvis_android_control_100","resolution":[1080,2400],"os":"android","source":"xlangai/aguvis-stage2","source_id":"100"}}}
```

**Two images, two actions.** `android_control` carries no terminator step, so the
last step has no post-action screenshot. That final action is still the
supervised EOF label; `validate_canonical_rows` allows this terminal unpaired
call, while intermediate unpaired calls are still rejected.

Persisted standalone extras (`open_app`, `response`) are never nested inside a
action batch — they stay top-level calls alongside it, and `open_app` carries its
canonical nested schema in `extra_tool_schemas`. Terminal `terminate` is consumed by the
final policy for `use` rows, not persisted as a tool call.
