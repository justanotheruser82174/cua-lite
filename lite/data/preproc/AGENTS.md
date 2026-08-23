# Dataset Preprocessing Specification

This document is the **contract** for adapters under [`lite/data/preproc/<dataset>/`](/lite/data/preproc): what they must read from raw upstream data, and what they must emit. For end-user docs (how to run preproc, how to consume in SFT export, how to round-trip via HuggingFace), see [`lite/data/README.md`](/lite/data/README.md).

The cua-lite dataset format supports four task families: visual understanding, UI-element grounding (action / point / bbox), and multi-step goal-oriented use, across desktop / browser / mobile platforms.

## Pipeline overview

Two ingestion paths converge on the same canonical local format. Adapters in this directory own the **left** path; the right path is `lite.data.hf.download`. Anything that consumes a cua-lite dataset (SFT export, training, dedup tooling) reads from the canonical layout regardless of which path produced it.

```
   raw upstream data                  cua-lite/<Name> on HuggingFace
          │                                       │
   lite/data/preproc/<X>/<task>.py        lite.data.hf.download
   (this directory; calls staging API)   (snapshot + ImageStore.put)
          │                                       │
          └────────────► canonical local ◄────────┘
                         layout (below)
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
       lite.data.hf       lite.train.export      any tooling
       .upload (push)     .export_sft            (filtering, mixing)
```

Canonical format helpers live in [`lite/data/staging.py`](/lite/data/staging.py): `ImageStore`, `SplitAssigner`, `partition_path`, `iter_partitions`, `flush_buffers`, `DatasetStats`. Adapters only ever import from `staging`; they don't know about the HF tooling and don't know how SFT export consumes the rows.

## Canonical local layout

```
${CUA_LITE_DATASETS_ROOT}/cua-lite/<DatasetName>/
  images/<hash[:2]>/<hash>.<ext>                  # content-addressed image store
  <platform>/<task_type>/<split>/<variant>.parquet # one file per variant
  README.md   (rendered at HF push time)
  stats.json  (rendered at HF push time)
```

- `<platform>` ∈ {`desktop`, `browser`, `mobile`}.
- `<task_type>` is the CUA task dimension literal — one of `understanding`, `grounding.action`, `grounding.point`, `grounding.bbox`, `use`. It's filename-safe and doubles as the directory component.
- `<split>` ∈ {`train`, `validation`}. Assigned by [`SplitAssigner`](/lite/data/staging.py) — preferring an upstream-canonical label from `metadata.others.split` if present, otherwise a deterministic SHA-256 hash of `metadata.others.id` with a per-cohort 2 000-decision validation cap; then content-identical rows are co-located into whichever split their group's first row got. See [What the validation split guarantees](#what-the-validation-split-guarantees).
- A CUA cohort = `metadata.dims == [platform, task_type]`. Every cohort writes `<split>/<variant>.parquet` — one file per variant, whether it has one or several (e.g. ScaleCUA's `understanding/{caption,user_intention,screen_transition}`).
- Local parquets are **never sharded**. HF push reshards by image-count for the viewer; HF download merges shards back. The local form is always one parquet per partition.

## Identifier vocabulary

A cohort has **one** human-readable identifier that parses the same way at every layer:

| layer | spelling | example |
|---|---|---|
| CUA `metadata.dims[1]` task literal | period sub-format | `"grounding.action"` |
| on-disk dir component | period sub-format | `mobile/grounding.action/train/ricosca.parquet` |
| agent registry lookup key | `<agent>@<plat>@<task_type>` | `"qwen3_vl@mobile@grounding.action"` |

`.` is the separator for `task_type` sub-format and on-disk paths — it is filename-safe, HF `config_name`-safe, and URL-safe (RFC 3986 unreserved, never percent-encoded). `@` is used only in agent registry lookup keys (not in filenames or metadata values). `:` is not used in any identifier.

`task_type` sub-format historically used `:` (e.g. `grounding:action`); old datasets bearing the colon form must be migrated by rewriting the CUA task dim and renaming dir components.

## Adapter authoring rules

- **Raise on format drift; count known bad source records.** A source-local defect
  that the adapter can identify precisely (for example an empty hotkey, an
  unsupported Office API action, or a corrupt image) drops the whole row — or
  the whole trajectory for `use` — under a named skip reason. Never delete only
  one step from a trajectory. An unexpected schema/action shape, JSON parse
  failure, or missing required field is format drift: raise with the dataset,
  source file, and line/key instead of hiding it in a broad exception handler.
  Missing external resources are counted separately because a partial host
  snapshot is not upstream annotation corruption.
- **Defaults apply only to absent optional fields.** If the source protocol defines an optional field's default, an absent field may use it. A field that is present but malformed must be rejected and counted; never replace a bad authored value with the default, because that silently invents supervision.
- **Preserve action order and multiplicity.** Never apply generic adjacent-action
  deduplication: two identical clicks, key presses, or scrolls can be the intended
  operation. Put several atomic actions in one `computer`/`mobile` batch only
  when the source proves they share one pre-action observation and one result
  boundary. Conversely, equal screenshot paths do not prove a shared boundary:
  if the source logical history advances, keep the separate action/result turns.
- **Image handling.** Resolve each raw image path via `resolve_path(rel, "CUA_LITE_RAW_DATASETS_ROOT")` from [`lite.utils.path`](/lite/utils/path.py); it returns the absolute path. Hand that to `ImageStore.put(abs_path)` — the store hashes the bytes, deduplicates, and returns a rel-path of the form `cua-lite/<DatasetName>/images/<hash[:2]>/<hash>.<ext>` that goes into the row's `images` list verbatim. If any image referenced by a record is missing, **skip the whole record**: for single-image tasks (understanding / grounding), skip the line; for `screen_transition` (two images), skip if either is missing; for use, skip the **entire trajectory** — assert `len(images) == len(steps)` after assembly. Once a row is assembled, `{"type": "image", "index": N}` parts are stable foreign keys into that row's `images` list. Staging, upload, and download may rewrite image paths/bytes, but they must never reindex or renumber message image references.
- **Environment variables.** Writing runs expect both `CUA_LITE_RAW_DATASETS_ROOT` (raw input root) and `CUA_LITE_DATASETS_ROOT` (canonical output root). `--dry-run` requires only the raw root and must neither create nor inspect the output root. Adapters read the former implicitly via `resolve_path`; the latter via `staging.dataset_root(name)`.
- **No upload-side leakage in row metadata.** Don't write `metadata.split` (the path encodes it). Don't add an `image_ids` column (the filename is the SHA-256). Don't pre-populate `metadata.extra_tool_schemas` / `valid_actions` with rollout-only env tools — those are populated by the gym env at rollout time. Raw-data preprocessors should leave `extra_tool_schemas=[]` by default. If a row persists a non-default standalone extra-tool call, that row must carry its matching nested Lite schema. Source terminal `terminate` for `use` is consumed into `metadata.others` and, when no executable EOF label remains, the `Done.` final; it is not persisted as a tool call or schema. Private rollout sidecars such as `raw_response` belong only to raw logs and must be absent from canonical rows.
- **Coordinate range.** All coordinates in tool-call arguments are normalized to `[0, 1000]`. After building each row, check for out-of-bound values using `has_oob_coordinate(entry)` — import it from the single canonical owner, [`lite/data/preproc/common.py`](/lite/data/preproc/common.py) (`from lite.data.preproc.common import has_oob_coordinate`). Do **not** hand-copy it into your adapter: the per-source copies scanned only `computer` (never `mobile`), only `coordinate` (never `start_coordinate`), and only `list` (never `tuple`), and each of those gaps let bad rows publish. **Drop the entire row** if any coordinate is outside the range. For `use` trajectories, drop the **entire trajectory** (all steps). Count and print the number of drops at the end — this is expected (upstream annotation noise, typically < 0.1% of rows).
- **Image integrity.** `ImageStore` defaults to `verify=True`, which runs `PIL.Image.open(src).verify()` on every image before hashing. Decode failures raise `CorruptImageError` at `put()` time — **before** the row is constructed, so no dangling image references can enter the parquet. Adapters may count and skip that specific exception; bulk adapters use `prestage_images()` so one corrupt image is isolated instead of aborting the whole entry. They must not catch a broad `ValueError`/`OSError` around `stage_entry`, because that hides canonical-validation and output-write failures as corrupt source data. If a batch of images has a high corruption rate, investigate upstream rather than disabling verification.
- **Raw archive extraction.** Directory existence is not proof that a large archive finished extracting. Write a completion marker only after archive validation and extraction both succeed, and trust that marker on rerun. Never delete split parts after a failed validation or extraction. Shared raw roots must not restore archived owner, mode, or timestamps (`tar --no-same-owner --no-same-permissions --touch --no-overwrite-dir`), because existing writable directories may belong to another user even though their contents are resumable.
- **Parquet serialization.** `messages` and `metadata` are stored as JSON strings in the parquet (via `staging.serialize_opaque_json_fields`), not as nested Arrow structs. This avoids pyarrow schema-union issues when different rows have different tool-call shapes. Consumers deserialize with `json.loads`.

---

## Core Dataset Schema

Each processed dataset subset is stored as a Parquet file (one file per CUA `(metadata.dims[0], metadata.dims[1], split, variant)` partition). Each row is a complete training sample with the schema below.

```json
{
  "images": <list[string]>,
  "messages": <list[dict]>,
  "metadata": <dict>
}
```

### Field Descriptions

Only two top-level keys are reserved (`images`, `messages`). Everything else lives in `metadata`.

- **`images`** (`list[string]`): paths into the dataset's content-addressed image store
  - Paths are relative to `${CUA_LITE_DATASETS_ROOT}` and are prefixed with `cua-lite/<DatasetName>/images/`. Example entry: `"cua-lite/ScaleCUA/images/ab/abcdef…789.png"`.
  - To get the absolute path on disk: `${CUA_LITE_DATASETS_ROOT}/<image-path>`.
  - For multi-dataset SFT mixes, point `--image-root` at `${CUA_LITE_DATASETS_ROOT}` to resolve every dataset's images uniformly.
  - The path filename `<hash>.<ext>` is the SHA-256 of the image bytes — the path is itself the content id; no separate `image_ids` column is carried.

- **`messages`** (`list[dict]`): Conversation turns in OpenAI-style message format
  - Each message is a dictionary with `role` and `content` fields
  - Each `content` item is a dict with a `type` discriminator. See [Content Part Types](#content-part-types) for the allowed kinds per role.
  - This is different for different task types. See [Dataset Task Types](#dataset-task-types) for detailed examples

- **`metadata`** (`dict`): Metadata object — must be `LiteCUAMetadata.to_dict()` for current preproc rows.
  - **`metadata_kind`** (`string`): `"cua"` for current CUA preproc rows.
  - **`dims`** (`list[string]`): CUA routing dimensions, `[platform, task_type]`. Platform is one of `"desktop"`, `"browser"`, `"mobile"`. Task category is one of `"understanding"`, `"grounding.action"`, `"grounding.point"`, `"grounding.bbox"`, `"use"`. The task literal is itself filename-safe and registry-key-safe; `compose_key(agent_id, *metadata.dims)` forms the agent registry lookup key.
  - **`extra_tool_schemas`** (`list[dict]`): Additional tool definitions beyond the default action space (e.g. browsergym's `back` / `goto`, or a dataset-specific non-terminal extra). Each entry uses the canonical nested Lite schema format: `{"type": "function", "function": {"name": "...", "parameters": {...}}}`. Provider/model-visible schemas such as `{"type": "function", "name": "...", "parameters": {...}}` are provider-flat wire and may appear only at provider/model seams or as legacy migration input, not in persisted dataset rows. Preprocessed dataset rows emit `[]` by default; if a row persists a standalone extra-tool call, that row must include the matching nested schema. `use` rows do not persist source terminal `terminate` by default.
  - **`valid_actions`** (`list[string] | None`): Subset of **action** names (the CARRY layer) to keep; `None` keeps them all. NOT tool names — on every action-batch space the two differ: `valid_actions=["click"]` names an action, while the emitted tool is `computer`. Gating narrows which actions the action-batch tool carries, not which tool is emitted. Respected across the agent layer, not by one family: the shared pipeline `assemble_tool_schemas` applies it for every adapter ([`lite/agents/core/adapter/base.py`](/lite/agents/core/adapter/base.py) `_assemble_tool_schemas`), and Fara, Qwen2.5-VL, MAI-UI and the API agents (Claude / GPT) additionally branch on it directly via `filter_tool_schemas_for_valid_actions`. **Preprocessed dataset rows always emit `None`** — adapters that want non-default subsets set this at rollout time.
  - **`others`** (`dict`): Catch-all for the remaining per-row fields. Suggested keys:
    - **`id`** (`string`): Unique identifier for each data sample (per-row). Used as the SplitAssigner's hash key.
    - **`resolution`** (`list[int]`): Display resolution as `[width, height]`. Example: `[1920, 1080]`
    - **`os`** (`string | null`): Operating system context. `null` when not relevant. Examples: `"ubuntu"`, `"windows"`, `"macos"`, `"android"`, `"ios"`
    - **`source`** (`string`): Source dataset identifier. Examples: `"OpenGVLab/ScaleCUA-Data"`, `"zyliu/ScaleCUA-Data-Understanding"`
    - **`source_id`** (`string`): Original identifier key from the source dataset. Example: `"windows_action_grounding_20250328"`
    - **`split`** (`string`, optional): Upstream-canonical split label (e.g. `"train"`, `"test_task"`). A **transient routing hint**, not a persisted field: when present, the SplitAssigner routes the row by this label instead of hashing, and [`SourceStaging.stage_entry`](/lite/data/preproc/common.py) pops it once consumed, so it never reaches the parquet. Don't add this for datasets without an upstream split.

**Split is a path artifact, not a metadata field.** Don't write `metadata.split` — partition file paths (`.../<split>/<variant>.parquet`) carry that information; HF's `datasets` library picks it up via the configs YAML.

### What the validation split guarantees

Stated positively and negatively, because the negative half is structural and is **not** being fixed — and because the dataset card makes this claim to every consumer.

**Guaranteed: no validation *sample* also appears in train.** `SplitAssigner` keys a `content_fingerprint` (the row's `images` + `messages`, ids excluded) and gives every member of a group the split its first member got. This exists because the split key is per-row while a *sample* is not: upstream corpora routinely re-publish one sample under two `source_id`s, and hashing each id independently put the same row on both sides. An upstream split label does not save you either — the co-location therefore wraps `canonical_fn` too.

**Not guaranteed: images are shared across the split, and heavily.** A screenshot legitimately backs many *distinct* samples — a grounding cohort annotates dozens of elements on one screen, a `use` episode revisits a screen — so only whole samples are co-located, never images. With validation fraction `f`, a validation row whose image backs `r` published rows in its cohort keeps that image out of train with probability only `f^(r−1)`: at `f = 0.02` that is 2% for `r = 2`. So in practice **every reused image is on both sides**. Keying the split on the image hash instead is the wrong fix, not an unimplemented one: the most-shared images back thousands of unrelated rows, so it would co-locate whole cohorts rather than duplicate groups.

Read the validation number accordingly: it measures generalisation to unseen **samples**, not to unseen **screens**. If you need a screen-level or episode-level holdout, carve it yourself; the published `validation` partition is not it.

**And where `val_cap` binds, the carve is order-dependent.** The cap accepts the first `val_cap` hash-selected row decisions per bucket and sends later ones to train, so that bucket's validation slice is a function of source iteration order, not of the hash — any upstream reorder or filter change silently re-draws it. Content co-location is applied afterwards and can move physical twins, so the final validation row count may be above or below the decision cap and cannot by itself prove whether the cap bound. Each run records its parameters in `<staged>/split.json` (`SplitAssigner.describe()`), which `lite.data.hf.upload` folds into the card's notes.

## Content Part Types

Every element of a message's `content` list is a dict with a `type` discriminator. The allowed kinds depend on the message role:

| Kind | Shape | Allowed on |
|---|---|---|
| `text` | `{"type": "text", "text": <str>}` | system, user, assistant (plain text — QA answers / captions / generic output) |
| `image` | `{"type": "image", "index": <int>}` | user (references `images[index]`) |
| `metadata` | `{"type": "metadata", "data": <dict>}` | user, assistant (env/agent-internal structured side-channel, e.g. `page_title`; stripped before the model by `keep_model_visible_content`) |
| `action_description` | `{"type": "action_description", "text": <str>}` | assistant (natural-language description of this turn's action in a use / grounding task) |
| `inline_reasoning` | `{"type": "inline_reasoning", "text": <str>}` | assistant (prompted CoT — non-native; e.g. `<think>` content from a non-reasoning model that was system-prompted into emitting it) |
| `history_summary` | `{"type": "history_summary", "text": <str>}` | assistant (cumulative trajectory summary the model emits, e.g. StepGUI's `summary:` field) |

**Per-role rules:**

- **`system`** — `content` is `str` OR `list[TextContent]`.
- **`user`**   — `content` is `str` OR `list[TextContent | ImageContent | MetadataContent]`.
- **`assistant`** — `content` is `list[TextContent | ActionDescriptionContent | InlineReasoningContent | HistorySummaryContent | MetadataContent]`. Use `text` for plain responses (QA / understanding / captioning); use the specific kinds when semantically applicable (`action_description` for use / grounding action turns). `MetadataContent` is the typed structured side-channel / forward-compat slot (replaced the old `dict[str,Any]`).

**Top-level field on assistant messages:**

- **`reasoning_content`** (`str`, optional) — the model's **native** reasoning trace from a dedicated channel (e.g. Qwen3-VL `<think>...</think>` thinking mode, OpenAI Responses API reasoning items, Claude `thinking` blocks). Emitted by Qwen3-VL and Qwen3.5 (`<think>...</think>` → this slot), by OpenCUA when the upstream response carries it, and by the Claude API agent (extended-thinking blocks); families without a native thinking channel drop it explicitly (`result.pop("reasoning_content", None)`). Dataset preprocessing should emit prompted CoT (e.g. ScaleCUA `<think>` tags, OpenCUA `## Thought`) as `InlineReasoningContent` inside `content` instead — those models are not native-thinking.

## Action Spaces

The canonical action definitions — the `@tool` methods a preproc adapter calls to build a
tool_call — live in [`lite/core/tools/action_space/base.py`](/lite/core/tools/action_space/base.py),
on the provider-free core action sets `LiteDesktopActionSet` / `LiteMobileActionSet` /
`LitePointActionSet` / `LiteBBoxActionSet`. The agent-layer dialect classes in
[`lite/agents/core/action_space/base.py`](/lite/agents/core/action_space/base.py) multiply-inherit
those core sets and are what the registry keys resolve to; each adapter knows its own action space
and inlines tool schemas during `render_step()` — the per-turn message renderer (replaces the old
`convert_sample_to_agent()` per-sample renderer).

| Registry Key | Registry Class (agents layer) | Core action set (`lite.core.tools.action_space`) | Description |
|---|---|---|---|
| `"lite@point"` | `LitePointActionSpace` | `LitePointActionSet` | Standalone point tool |
| `"lite@bbox"` | `LiteBBoxActionSpace` | `LiteBBoxActionSet` | Standalone bounding-box tool |
| `"lite@desktop"` | `LiteDesktopActionSpace` | `LiteDesktopActionSet` | Ordered desktop actions carried by `computer` |
| `"lite@mobile"` | `LiteMobileActionSpace` | `LiteMobileActionSet` | Ordered mobile actions carried by `mobile` |

Usage — a preproc adapter builds calls off the **core** set (this is what every adapter under
`lite/data/preproc/` imports); the registry is only needed when you want the dialect object:

```python
from lite.core.tools.action_space import LiteDesktopActionSet
from lite.agents.core.action_space import ActionSpaceRegistry

# Build a canonical call directly from the core action set
action = LiteDesktopActionSet.click(coordinate=[500, 300])
# -> {
#      'type': 'function',
#      'function': {
#          'name': 'computer',
#          'arguments': {'actions': [{'action': 'click', 'coordinate': [500, 300]}]},
#      },
#    }

# Or resolve the agent-layer dialect by registry key
desktop = ActionSpaceRegistry.get("lite@desktop")   # LiteDesktopActionSpace instance
```

### Keyboard keys — lowercase named keys plus printable glyphs

`key` / `key_down` / `key_up` / `hold_key` keys are **normalized to Lite's
stored key-token vocabulary at the core action-set chokepoint** —
`LiteDesktopActionSet.key()`
& friends, via [`lite/core/tools/action_space/keys.py`](/lite/core/tools/action_space/keys.py)
(`normalize_keys`). Backend spellings are projected later by
[`lite/gym/utils/backend/keys.py`](/lite/gym/utils/backend/keys.py). Every
adapter's `convert_*_from_agent` funnels through this factory, so the **stored
Lite action uses this vocabulary** regardless of how the model spelled it.
Provider `convert_*_to_agent` paths may then re-render the stored token form into
that provider's wire syntax:

- **casing / camelCase** fold: `Ctrl`/`CTRL` → `ctrl`, `ArrowDown` → `down`, `Return` → `enter`.
- **aliases** fold to named keys or glyphs: `control`→`ctrl`,
  `cmd`/`super`/`win`→`meta`, `option`→`alt`, `escape`→`esc`,
  `pgdn`/`page_down`→`pagedown`, and
  `plus`/`minus`/`equal`→`+`/`-`/`=`.
- Raw string chords split on `+`: `"ctrl+a"` → `["ctrl","a"]`. A list
  preserves token boundaries: `["ctrl","a"]` is already tokenized, while
  `["ctrl+a"]` is invalid source data.
- literal plus is stored as the glyph token `+`: `"+"`, `"ctrl++"`,
  `["ctrl","+"]`, and `"ctrl+plus"` all produce `+` as the stored token.
- stored named keys: `ctrl alt shift meta · enter tab space backspace delete esc ·
  up down left right home end pageup pagedown insert · capslock menu clear
  printscreen kp_enter · volumeup volumedown volumemute playpause nexttrack
  prevtrack · f1..f24`.
- stored glyph tokens: single visible ASCII characters. Letter glyphs are
  stored lowercase; digits and punctuation are stored literally, including `+`,
  `-`, `=`, `,`, and `\`.

At **rollout** time each gym env projects this stored form to its backend's
spelling — Playwright `Control`/`ArrowDown`, xdotool `ctrl`/`Down`, pyautogui
`ctrl`/`down`, or the equivalent pynput / Selenium names — via the projectors in
`lite.gym.utils.backend.keys`, which **raise loudly** on an unmappable key (never
a silent no-op).

**For preproc authors:** always build key actions through the corresponding
`LiteDesktopActionSet` key-action factory (`key`, `key_down`, `key_up`, or
`hold_key`; never hand-assemble the `tool_call` dict) so they get normalized,
then reject source tokens that do not become one literal printable character or
one member of the named-key set. The canonical publish gate enforces this
invariant; a string-only JSON schema is not sufficient because an unknown
multi-character token would fail only when a backend executes it. The
**stored / round-tripped keys are lowercase names plus printable glyphs** —
source casing is intentionally not preserved. On-policy RL is unaffected (it
tokenizes the verbatim sampled output, not this structural view;
`preserve_raw_response=True`). `raw_response` is a private runtime sidecar and
is stripped before publish; raw preprocessors should not emit it.

---

## Dataset Task Types

The four task families and their `messages` schemas. The on-disk layout for each is described in the [Canonical local layout](#canonical-local-layout) section above.

1. **[Understanding](#1-understanding-tasks)** — image comprehension without tool calls (captioning, QA, screen transition).
2. **[Grounding](#2-grounding-tasks)** — locate/interact with a UI element using a tool call.
   - [`grounding.action`](#a-grounding-action-tasks) — full atomic action(s) on the element.
   - [`grounding.point`](#b-grounding-point-tasks) — center coordinate.
   - [`grounding.bbox`](#c-grounding-bbox-tasks) — bounding box.
3. **[Use (multi-step rollout)](#3-use-multi-step-rollout-tasks)** — multi-step goal-oriented interactions with state feedback after each action.

---

### 1. Understanding Tasks

Understanding tasks involve image comprehension without tool usage.

**Configuration:**
- **`messages`**:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "index": 0},
        {"type": "text", "text": "Please explain the first screenshot to me."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "This is a screenshot of a desktop."}
      ]
    }
  ]
}
```

**Notes:**
- The `"index"` field in image content references the zero-based position in the `images` array
- Multiple conversation turns are supported for follow-up questions
- The assistant's reply uses `text` — understanding is plain QA, not action-specific
- Understanding rows must not include assistant `tool_calls`, even when a matching tool schema is available; they also do not include `role:"tool"` result rows.

---

### 2. Grounding Tasks

Grounding tasks involve locating or interacting with UI elements using tool calls.

#### a. Grounding Action Tasks

Action grounding tasks require the agent to perform specific actions on UI elements.

**Desktop Configuration:**
- **`messages`**:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "index": 0},
        {"type": "text", "text": "Search for 'machine learning' in the browser search bar."}
      ]
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "call_0000",
          "type": "function",
          "function": {
            "name": "computer",
            "arguments": {
              "actions": [
                {"action": "click", "coordinate": [500, 100]},
                {"action": "type", "text": "machine learning"},
                {"action": "key", "keys": ["enter"]}
              ]
            }
          }
        }
      ]
    }
  ]
}
```

**Mobile Configuration:**
- **`messages`**:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "index": 0},
        {"type": "text", "text": "Type 'coffee shops near me' in the search field and submit."}
      ]
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "call_0000",
          "type": "function",
          "function": {
            "name": "mobile",
            "arguments": {
              "actions": [
                {"action": "tap", "coordinate": [500, 300]},
                {"action": "type", "text": "coffee shops near me"},
                {"action": "tap", "coordinate": [850, 300]}
              ]
            }
          }
        }
      ]
    }
  ]
}
```

**Notes:**
- The `tool_calls` field contains one or more nested Lite calls: `{"id": "...", "type": "function", "function": {"name": "...", "arguments": {...}}}`
- `grounding.*` rows are supervised labels, not environment rollouts. Their assistant `tool_calls` must carry canonical `id` fields, but they do **not** get `role:"tool"` result rows. Do not append empty or synthetic feedback after a grounding label call.
- For `grounding.action`, desktop/browser rows use `function.name="computer"` action-batch calls and mobile rows use `function.name="mobile"` action-batch calls; atomic actions live under `function.arguments.actions`. Key tokens use the stored Lite vocabulary (`"enter"`, not `"Return"`; `"+"`, not `"plus"`).
- For `use` rows, source preprocessors emit actions inside length-1 `computer` / `mobile` action-batch calls with `function.arguments.actions` unless a raw source step proves one post-action screenshot/result boundary for a multi-action action-batch call. Persisted standalone extras such as nav, `open_app`, and `response` are never nested in action batches; terminal `terminate` is consumed by the final policy by default and does not imply a `terminate` schema.
- Never remove adjacent equal actions as a generic cleanup. Repeated clicks, key presses, and scrolls are executable labels (for example, two clicks may increment a counter twice). Only suppress an action when the adapter can trace both copies to one source action plus its own deterministic expansion, and leave a focused regression for that exact conversion.
- Multiple conversation turns are supported for follow-up questions
- Desktop actions use atomic functions like `key`, `click`, `type`, `scroll`, etc.
- Mobile actions use atomic functions like `tap`, `swipe`, `system_button`, etc.

---

#### b. Grounding Point Tasks

Point grounding tasks require the agent to identify the coordinate of a UI element.

**Configuration:**
- **`messages`**:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "index": 0},
        {"type": "text", "text": "Where is the save button located?"}
      ]
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "call_0000",
          "type": "function",
          "function": {
            "name": "point",
            "arguments": {"coordinate": [850, 120]}
          }
        }
      ]
    }
  ]
}
```

**Notes:**
- The `coordinate` represents the center point `[x, y]` of the target element
- Multiple conversation turns are supported for follow-up questions

---

#### c. Grounding Bbox Tasks

Bounding box grounding tasks require the agent to identify the bounding box of a UI element.

**Configuration:**
- **`messages`**:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "index": 0},
        {"type": "text", "text": "Locate the login button on the screen."}
      ]
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "call_0000",
          "type": "function",
          "function": {
            "name": "bbox",
            "arguments": {"coordinate": [380, 450, 620, 520]}
          }
        }
      ]
    }
  ]
}
```

**Notes:**
- The `coordinate` format is `[x_min, y_min, x_max, y_max]` representing the bounding box
- Multiple conversation turns are supported for follow-up questions

---

### 3. Use (multi-step rollout) Tasks

Use (multi-step rollout) tasks involve multiple interaction turns to complete a goal, with the agent receiving new screenshots after each action.

**Desktop Configuration:**
- **`messages`**:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "index": 0},
        {"type": "text", "text": "Open the Pikachu picture on the desktop using GIMP, and then select the 'Venetian Blinds' filter in the animation."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "inline_reasoning", "text": "I need to start working towards the goal of opening the Pikachu picture on the desktop using GIMP and then applying the 'Venetian Blinds' filter..."},
        {"type": "action_description", "text": "Click on the GIMP application icon in the left taskbar to launch the image editing software."}
      ],
      "tool_calls": [
        {
          "id": "call_0000",
          "type": "function",
          "function": {
            "name": "computer",
            "arguments": {
              "actions": [
                {"action": "click", "coordinate": [18, 508]}
              ]
            }
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_0000",
      "content": [{"type": "image", "index": 1}]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "inline_reasoning", "text": "GIMP has successfully launched and is now ready for use..."},
        {"type": "action_description", "text": "Press Ctrl+O to open the file dialog in GIMP, which will allow me to browse and select the pikachu.jpeg file from the desktop."}
      ],
      "tool_calls": [
        {
          "id": "call_0001",
          "type": "function",
          "function": {
            "name": "computer",
            "arguments": {
              "actions": [
                {"action": "key", "keys": ["ctrl", "o"]}
              ]
            }
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_0001",
      "content": [{"type": "image", "index": 2}]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "action_description", "text": "Navigate to the desktop folder and click on pikachu.jpeg to select it."}
      ],
      "tool_calls": [
        {
          "id": "call_0002",
          "type": "function",
          "function": {
            "name": "computer",
            "arguments": {
              "actions": [
                {"action": "click", "coordinate": [500, 350]}
              ]
            }
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_0002",
      "content": [{"type": "image", "index": 3}]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "Done."}
      ]
    }
  ]
}
```

**Mobile Configuration:**
- **`messages`**: Similar structure to desktop, using mobile-specific actions like `tap`, `swipe`, `system_button`, etc.

**Notes:**
- The agent receives an initial screenshot and goal from the user
- After each non-terminal action, a `role:"tool"` message provides the paired result via `tool_call_id` matching the assistant call's `id`
- Each assistant turn includes:
  - **`content`**: a list of structured parts. For `use` turns this is typically:
    - an `inline_reasoning` part (optional) — the agent's reasoning about the current state and strategy (prompted CoT, NOT a model-native channel; native reasoning would go in the top-level `reasoning_content` field instead — see [Content Part Types](#content-part-types))
    - an `action_description` part — human-readable description of the action being taken
  - **`tool_calls`**: the actual nested Lite call(s) to execute; actions use `function.name="computer"` / `"mobile"` with atomic steps under `function.arguments.actions`
- `use` tasks typically involve >=2 interaction turns
- A `use` row has two legal endings:
  - a content-only assistant final (`"Done."` or a real source answer), when the
    source supplied terminal text/status but no executable EOF label;
  - a final assistant `tool_calls` turn with no trailing `role:"tool"` result,
    when the source ends after an executable action and did not capture the
    post-action observation. That final action is still a supervised label.
- `"Done."` is the trajectory-END SIGNAL, not a universal literal: use it only
  when the source produced **no information**, or when the only information is a
  terminate *justification* (which goes to `metadata.others`, not the turn).
  `structural_final_message()` in [`lite/data/utils/messages.py`](/lite/data/utils/messages.py)
  spells both — no argument gives the `[{"type": "text", "text": "Done."}]`
  marker, an argument gives that text.
- For `use` rows a source terminal `terminate(status=...)` is **not** persisted
  as a tool call: `status="success"` is structural and records nothing beyond
  the marker; non-success status/reason payloads move to `metadata.others` via
  `lite/data/utils/messages.py::terminate_outcome_others`. If popping the
  terminal call leaves executable calls in the same final assistant turn, keep
  those calls as the EOF label and do not append a separate final.
- A terminal source row's screenshot is not a result for the dropped `terminate`
  itself. Keep it only when the raw format uses that row as the post-action
  screenshot for the previous non-terminal action; otherwise discard it.
- **Never truncate mid-episode.** A final EOF action is legal because the source
  row itself ended there. Stopping at step `k` of `n` is different: the source's
  own remaining steps falsify that endpoint, so a step that cannot be parsed in
  the middle of an episode drops the **episode**.
- Coordinates are normalized to [0, 1000] range

---

## Adding a new dataset

A new dataset lives at `lite/data/preproc/<name>/` and contributes one Python script per task type it produces. Follow this SOP — the existing `scalecua/` adapter is the reference implementation.

### 1. Lay out the directory

```
lite/data/preproc/<name>/
├── __init__.py                      # one-line module docstring is fine
├── AGENTS.md                        # dataset-specific guide (selection criteria, mapping tables)
├── README.md                        # user-facing how-to: download raw, run preproc
├── repo.json                        # static HF repo metadata (read by hub_card)
├── utils.py                         # adapter shared helpers (image-store + splitter wiring)
├── <task>.py                        # one per emitted (platform, task_type) family
└── scripts/
    ├── download_raw_data.sh         # `hf` CLI snapshot of upstream
    ├── process_raw_data.sh          # untar / unpack / format conversion
    └── process_data.sh              # wrap the python adapters end-to-end
```

`repo.json` is required for HF publishing — the keys are read by [`lite.data.hf.card.load_repo_json`](/lite/data/hf/card.py):

```json
{
  "description": "cua-lite preprocessed version of <upstream>. <one-paragraph summary>.",
  "original_urls": ["https://huggingface.co/datasets/<upstream-org>/<upstream-name>"],
  "license": "See original dataset (<upstream>).",
  "citation": "See https://huggingface.co/datasets/<upstream-org>/<upstream-name>",
  "extra_notes": ""
}
```

`AGENTS.md` is the technical specification for the dataset — what the source data looks like, how it's transformed, and what the output looks like. It must NOT duplicate the how-to-run content in `README.md`. Required sections, in order:

| # | Section | What to write |
|---|---------|---------------|
| 1 | **Overview** | Related Files table (preprocessing scripts, shell scripts, metadata files) + Source Data Locations table (where raw data lives, keyed by task type or variant). |
| 2 | **Output Directory Structure** | Tree diagram of the canonical output layout under `cua-lite/<Name>/`. |
| 3 | **Data Statistics** | Row counts per partition, skip/drop reasons with counts, dataset-specific field distributions (e.g. quality fields, OS breakdown). Include a reproducible command to regenerate these numbers. |
| 4 | **Source Data Format** | Field tables for the raw upstream data (record-level and, for multi-step data, step-level). Include one full unabridged raw example. |
| 5 | **Action Space & Mapping** | Source action definitions (function signatures or grammar) + source→target mapping table. Omit for understanding-only datasets. |
| 6 | **Processing Workflow** | High-level numbered steps covering the full transform: reading raw data, image resolution + skip logic, OS detection, coordinate normalization, message construction, split assignment, flushing. |
| 7 | **Error Handling** | Table of error type → behavior (skip record, skip trajectory, raise). |
| 8 | **Output Format** | One full canonical output record as JSON — the exact shape that goes into the parquet. |

`README.md` is the user-facing how-to: environment setup, download commands, run commands, output location (one-line reference to AGENTS.md for the layout), verification commands, HF push command. No overlap with AGENTS.md content.

### 2. Adapter skeleton

Every task script follows the same structure: iterate raw records, build canonical rows, hand each through `stage_entry`, flush at the end. Keep the per-record parsing (action mapping, message construction) inline in the file — that's where the dataset-specific knowledge lives and it stays auditable in one place.

A `<name>/utils.py` parallel to `scalecua/utils.py` makes the boilerplate one import.
**Prefer `SourceStaging`**: unless your dataset needs a different key, bucket or cap,
`_STAGING = SourceStaging(DATASET_NAME)` plus four alias assignments is the whole file —
see [`lite/data/preproc/scalecua/utils.py`](/lite/data/preproc/scalecua/utils.py). Do not
copy the staging mechanics into new adapters; `SourceStaging`
([`common.py`](/lite/data/preproc/common.py)) owns image hashing, metadata defaults,
split policy recording, transient split-hint stripping, validation, and partition keys.

```python
# lite/data/preproc/<name>/utils.py
from lite.data.preproc.common import SourceStaging

DATASET_NAME = "<HFRepoName>"  # used as cua-lite/<HFRepoName>
_STAGING = SourceStaging(DATASET_NAME)


out_dir_for = _STAGING.out_dir_for
make_image_store = _STAGING.make_image_store
make_splitter = _STAGING.make_splitter
stage_entry = _STAGING.stage_entry
```

A task script:

```python
# lite/data/preproc/<name>/<task>.py
from collections import defaultdict
from lite.core.metadata import LiteCUAMetadata
from lite.data import staging
from lite.data.preproc.common import has_oob_coordinate
from lite.data.preproc.<name>.utils import (
    make_image_store, make_splitter, out_dir_for, stage_entry,
)
from lite.utils.path import resolve_path

VARIANT = "<variant_name>"  # category-stem; e.g. "action" / "bbox" / "point" / sub-task name

def main():
    out_dir = out_dir_for()
    store = make_image_store(out_dir)
    splitter = make_splitter()
    buffers: dict[tuple, list[dict]] = defaultdict(list)

    n_oob = n_corrupt = 0
    for record in iter_raw_records(...):                       # dataset-specific
        try:
            abs_imgs = [resolve_path(p, "CUA_LITE_RAW_DATASETS_ROOT") for p in image_paths(record)]
        except FileNotFoundError:
            continue                                           # skip whole sample if any image missing
        row = {
            "images":   abs_imgs,                              # abs paths; stage_entry rewrites to store rel-paths
            "messages": build_messages(record),                # OpenAI-style turns; see Content Part Types
            "metadata": LiteCUAMetadata(
                dims=(derive_platform(record), "<grounding.point|grounding.action|...>"),
                extra_tool_schemas=[],
                valid_actions=None,
                others={
                    "id":         per_row_unique_id(record),
                    "resolution": [width, height],
                    "os":         derive_os(record),
                    "source":     "<upstream-org>/<upstream-name>",
                    "source_id":  upstream_batch_key(record),
                },
            ).to_dict(),
        }
        if has_oob_coordinate(row):                            # drop rows with coordinates outside [0, 1000]
            n_oob += 1
            continue
        try:
            bk, e = stage_entry(row, store=store, splitter=splitter, variant=VARIANT)
        except staging.CorruptImageError:
            n_corrupt += 1
            continue
        buffers[bk].append(e)

    staging.flush_buffers(out_dir, buffers)
    n_rows = sum(len(rs) for rs in buffers.values())
    if n_oob:
        print(f"Dropped {n_oob} rows with out-of-bound coordinates")
    if n_corrupt:
        print(f"Dropped {n_corrupt} rows with corrupt images")
    print(f"Wrote {n_rows} rows across {len(buffers)} partitions to {out_dir}")
```

`stage_entry` mutates the row in place: rehashes images in list order, normalizes tagged CUA metadata, records the split policy, decides the split, strips transient `metadata.others.split`, validates the canonical row, and returns the partition key `(metadata.dims[0], metadata.dims[1], split, variant)`. It does not synthesize rollout-only env-owned tools and does not rewrite message image indices. `flush_buffers` writes one parquet per partition, always as `<split>/<variant>.parquet` — the variant is in the filename so a later run can tell which producer wrote a file.

### 3. Variant choice

A "variant" is a sub-cohort within a CUA `metadata.dims == [platform, task_type]` cell. Use it when one task type spans multiple semantically distinct sub-sources that you want to keep separately countable + addressable:

- ScaleCUA `understanding` has three variants: `caption`, `user_intention`, `screen_transition` — distinct prompts, kept apart in the stats table and as separate parquet files.
- ScaleCUA `grounding.action` has only one variant (default `action`), so it writes `train/action.parquet` / `validation/action.parquet`.
- OS-Atlas's single per-platform parquet mixes sub-sources by `source_id` prefix (`windows_*`, `linux_*`, etc.); a `variant_fn` derives the variant from the row.

The `SplitAssigner` validation decision cap is per-bucket including variant, so each variant gets its own ~2 000-decision budget; content-identical row co-location can make the final physical count differ.

### 4. Wiring scripts

`scripts/process_data.sh` invokes the adapters end-to-end. The pattern (mirroring `scalecua/scripts/process_data.sh`):

Preprocessing dependencies live in the project `data` extra. Run
`uv sync --extra data` once; wrappers also select that extra explicitly so a
clean checkout has the preprocessing and Hugging Face data dependencies.

```bash
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"   # worktree's lite/ shadows editable install

uv run python "${PREPROC_DIR}/<task1>.py" "${ARGS[@]}"
uv run python "${PREPROC_DIR}/<task2>.py" "${ARGS[@]}"
# ...
echo "Done. Output under \${CUA_LITE_DATASETS_ROOT}/cua-lite/<Name>/."
```

### 5. Verify before pushing

```bash
# 1. Inspect output layout + a sample row
uv run python -c "
from datasets import load_dataset
from pathlib import Path
from lite.data.staging import iter_partitions
root = Path('${CUA_LITE_DATASETS_ROOT}/cua-lite/<Name>')
for *_, f in iter_partitions(root):
    d = load_dataset('parquet', data_files=str(f), split='train')
    print(f'{f.relative_to(root)}: {len(d)} rows')
"

# 2. Smoke-test SFT export — see lite/data/README.md
uv run python -m lite.train.export.export_sft \
    --agent-id qwen3_vl \
    --model-id Qwen/Qwen3-VL-4B-Instruct \
    --data-paths ${CUA_LITE_DATASETS_ROOT}/cua-lite/<Name>/<plat>/<task_type> \
    --image-root ${CUA_LITE_DATASETS_ROOT} \
    --head 5 -o /tmp/<name>-sft-smoke.parquet
```

`--head` and `--sample` cap the pooled input after every discovered parquet is
loaded; neither guarantees one row from each partition. When a change spans
multiple platforms or task types, run one bounded export per affected cohort
and inspect every output rather than treating one pooled smoke as coverage.

#### Dataset adapter smoke tests

This table records repo-local strict tests that can replace ad-hoc bounded
smoke notes for adapter changes. These tests use
synthetic raw rows, canonical staging, validation, and parquet roundtrips where
possible; they do not replace full real-raw row-count, skip-count, visual
spot-check, or SFT-export smoke.

| Dataset | Strict reproducible test command | Coverage |
|---|---|---|
| CAGUI | `uv run --extra data --extra dev pytest -q tests/data/preproc/cagui/test_preproc_cagui_understanding_entrypoint.py tests/data/preproc/cagui/test_preproc_cagui_tool_io_contract.py::test_cagui_post_action_screenshot_is_tool tests/data/preproc/cagui/test_preproc_cagui_tool_io_contract.py::test_cagui_status_task_complete_becomes_done tests/data/preproc/cagui/test_preproc_cagui_tool_io_contract.py::test_cagui_status_task_impossible_moves_to_metadata_others` | Canonical row covered; keep full real-raw/export smoke separate. |
| GUIOdyssey | `uv run --extra data --extra dev pytest -q tests/data/preproc/guiodyssey/test_preproc_guiodyssey_understanding_entrypoint.py tests/data/preproc/guiodyssey/test_preproc_guiodyssey_tool_io_contract.py::test_guiodyssey_post_action_screenshot_is_tool tests/data/preproc/guiodyssey/test_preproc_guiodyssey_tool_io_contract.py::test_guiodyssey_coordinates_are_already_normalized_not_device_scaled tests/data/preproc/guiodyssey/test_preproc_guiodyssey_tool_io_contract.py::test_guiodyssey_complete_becomes_done tests/data/preproc/guiodyssey/test_preproc_guiodyssey_tool_io_contract.py::test_guiodyssey_incomplete_reason_comes_from_ps_not_info tests/data/preproc/guiodyssey/test_preproc_guiodyssey_tool_io_contract.py::test_guiodyssey_blank_ps_emits_no_reason` | Canonical row and already-normalized source coordinates covered. |
| UI-Genie Agent | `uv run --extra data --extra dev pytest -q tests/data/preproc/ui_genie_agent/test_preproc_ui_genie_entrypoint.py tests/data/preproc/ui_genie_agent/test_preproc_ui_genie_tool_io_contract.py::test_ui_genie_post_action_screenshot_is_tool tests/data/preproc/ui_genie_agent/test_preproc_ui_genie_tool_io_contract.py::test_ui_genie_terminal_answer_becomes_the_final_turn_text tests/data/preproc/ui_genie_agent/test_preproc_ui_genie_tool_io_contract.py::test_ui_genie_missing_terminal_terminate_keeps_final_action tests/data/preproc/ui_genie_agent/test_preproc_ui_genie_tool_io_contract.py::test_ui_genie_failure_terminate_moves_to_metadata_others` | Canonical row covered; no-terminal rows are covered as final EOF actions. |
| Multimodal-Mind2Web | `uv run --extra data --extra dev pytest -q tests/data/preproc/multimodal_mind2web/test_preproc_mind2web_tool_io_contract.py::test_mind2web_emits_rows_with_final_action_at_eof tests/data/preproc/multimodal_mind2web/test_preproc_mind2web_tool_io_contract.py::test_mind2web_single_step_episode_publishes_final_action tests/data/preproc/common/test_common_publish_gate.py::test_script_accepts_a_head_smoke_bound` | Canonical row covered; final EOF actions cover 1/2/3/5-step shapes. |
| Aguvis | `uv run --extra data --extra dev pytest -q tests/data/preproc/aguvis/test_preproc_aguvis_use_entrypoint.py tests/data/preproc/aguvis/test_aguvis_publish_gate.py::test_aguvis_grounding_row_passes_publish_gate tests/data/preproc/aguvis/test_preproc_aguvis_tool_io_contract.py::test_aguvis_grounding_action_derives_extra_tool_schemas tests/data/preproc/aguvis/test_preproc_aguvis_tool_io_contract.py::test_aguvis_rejects_steps_after_structural_terminator tests/data/preproc/aguvis/test_preproc_aguvis_tool_io_contract.py::test_aguvis_failure_terminator_moves_to_metadata_others tests/data/preproc/aguvis/test_preproc_aguvis_tool_io_contract.py::test_aguvis_without_terminator_keeps_final_action_at_eof tests/data/preproc/aguvis/test_preproc_aguvis_tool_io_contract.py::test_aguvis_single_step_episode_publishes_final_action` | Canonical row covered; no-terminator and single-step rows are final EOF actions. Keep full real-raw skip-count/export smoke separate. |
| GUIAct | `uv run --extra data --extra dev pytest -q tests/data/preproc/guiact/test_preproc_guiact_use_entrypoint.py tests/data/preproc/guiact/test_preproc_guiact_tool_io_contract.py::test_guiact_grounding_action_derives_response_schema tests/data/preproc/guiact/test_preproc_guiact_tool_io_contract.py::test_guiact_post_action_screenshot_is_tool tests/data/preproc/guiact/test_preproc_guiact_tool_io_contract.py::test_guiact_terminal_step_without_answer_still_publishes tests/data/preproc/guiact/test_preproc_guiact_tool_io_contract.py::test_guiact_single_step_episode_publishes_final_action` | Canonical row covered; browser and smartphone roundtrips keep final executable EOF labels. |
| GUI-360 | `uv run --extra data --extra dev pytest -q tests/data/preproc/gui360/test_preproc_gui360_use_entrypoint.py tests/data/preproc/gui360/test_preproc_gui360_understanding_entrypoint.py tests/data/preproc/gui360/test_gui360_publish_gate.py::test_gui360_grounding_point_row_passes_publish_gate tests/data/preproc/gui360/test_preproc_gui360_tool_io_contract.py::test_gui360_post_action_screenshot_is_tool_and_type_batches tests/data/preproc/gui360/test_preproc_gui360_tool_io_contract.py::test_gui360_keeps_terminal_executable_action_at_eof tests/data/preproc/gui360/test_gui360_use_publish.py::test_gui360_single_step_trajectory_publishes_final_action` | Canonical row and terminal/single-step executable EOF labels covered. |
| ScaleCUA | `uv run --extra data --extra dev pytest -q tests/data/preproc/scalecua/test_preproc_scalecua_cohorts.py tests/data/preproc/scalecua/test_preproc_scalecua_tool_io_contract.py::test_scalecua_post_action_screenshot_is_tool tests/data/preproc/scalecua/test_preproc_scalecua_tool_io_contract.py::test_scalecua_lone_final_terminate_becomes_done_without_schema tests/data/preproc/scalecua/test_preproc_scalecua_tool_io_contract.py::test_scalecua_failure_reason_survives_into_metadata_others tests/data/preproc/scalecua/test_preproc_scalecua_tool_io_contract.py::test_scalecua_answerless_trajectory_can_end_on_final_action tests/data/preproc/scalecua/test_preproc_scalecua_tool_io_contract.py::test_scalecua_terminal_answer_can_share_final_action_turn` | Canonical row and final executable action with/without answer text covered. Keep full real-raw/export smoke separate. |
| OpenCUA | `uv run --extra data --extra dev pytest -q tests/data/preproc/opencua/test_preproc_opencua_win_mac_entrypoint.py tests/data/preproc/opencua/test_preproc_opencua_tool_io_contract.py::test_opencua_terminal_only_terminate_becomes_done_and_tool_result tests/data/preproc/opencua/test_preproc_opencua_tool_io_contract.py::test_opencua_non_terminating_final_keeps_final_action tests/data/preproc/opencua/test_preproc_opencua_tool_io_contract.py::test_opencua_failure_terminate_moves_to_metadata_others tests/data/preproc/opencua/test_preproc_opencua_tool_io_contract.py::test_opencua_iter_examples_head_stops_at_trajectory_boundary` | Canonical `ubuntu`/`win_mac` roundtrips, AgentNet metadata sidecars, and head-bounded trajectory iteration covered. |

### 6. Publish

Freeze the producing revision first, then publish and pull the same revision back — the
same four-step shape the rollout datasets use (see [`devs/data/lite.osworld/AGENTS.md`](/devs/data/lite.osworld/AGENTS.md)).

```bash
: "${HF_ORG:?set HF_ORG to your Hub user/org for the private smoke repo}"
COMMIT="$(git rev-parse --short HEAD)"                                                 # freeze the code revision

uv run python -m lite.data.hf.upload <Name> --org "$HF_ORG" --private --tag "$COMMIT"   # smoke repo
uv run python -m lite.data.hf.upload <Name> --org <org> --dry-run                       # validate final publish plan
uv run python -m lite.data.hf.upload <Name> --org <org> --tag "$COMMIT"                 # publish

# Consumer / verification: pull the published revision back into canonical local layout.
uv run python -m lite.data.hf.download <Name> \
  --org <org> \
  --revision "$COMMIT" \
  --out "${CUA_LITE_DATASETS_ROOT}/cua-lite/<Name>"
```

`--tag` is written explicitly here but it is also the **default**: `lite.data.hf.upload` resolves
an empty `--tag` to the clean current cua-lite short `HEAD`. Rollout provenance can include a
`-dirty` suffix, and raw preproc rows do not automatically carry `metadata.others.commit`, so freeze
the producing tree before publishing instead of relying on row provenance to police it. Pass
`--tag <name>` to override, `--tag NONE` to skip. `--dry-run` never tags. Tag the smoke repo too —
an untagged smoke push cannot be traced back to the code that produced it.

Push is resume-safe (`--skip-existing`), batches 50 files per HF commit (HF caps at 128/hr), reshards to ≤ 200 images per parquet for the dataset viewer, and orphan-cleans paths from prior pushes that are no longer in the new layout. The README.md and stats.json are rendered at push time from `repo.json` + on-disk parquet stats. Private runtime sidecars such as `raw_response` must already be absent from canonical rows before publish.

### 7. Common gotchas

- **Splitter key.** `source_id` in many upstream datasets is a per-*batch* category label (e.g. `"windows_action_grounding_20250328"`), not per-row unique. Using it as the split key would dump whole batches into a single split. Use a per-row unique field (`id`, or compose `f"{source_id}_{line_num}"`).
- **Don't mutate the tagged metadata shape.** Always emit `LiteCUAMetadata.to_dict()` for CUA preproc rows (`metadata_kind`, `dims`, `extra_tool_schemas`, `valid_actions`, `others`); fill defaults for the runtime fields. Adapters that omit them break `metadata_from_dict` for downstream consumers.
- **Image deduplication is intra-dataset only.** `ImageStore` only sees images one dataset emits; identical images across datasets are deduped on HF's xet layer at push time but not on local disk. Don't try to share image stores between datasets — `image_rel_prefix(name)` ties paths to a specific dataset.
- **Don't carry HF-specific state in row metadata.** No `metadata.split`, no `image_ids` column, no embedded image bytes locally. Those forms exist only on HF (sharded parquets with `images: list[Image]`); `lite.data.hf.download` translates back to canonical on pull.
- **Action regex substring matching.** When parsing action names from free-text (e.g. `click(...)` from ScaleCUA action grounding), use a negative lookbehind `(?<![a-zA-Z])` to prevent `click` from matching inside `rightClick` or `doubleClick`. Test your regex against the full action vocabulary before wiring up the adapter.
- **OOB coordinate filtering.** After building each entry and before buffering, call `has_oob_coordinate(entry)` and skip entries that return `True`. For `use`, this means dropping the entire trajectory. Print a summary count of dropped entries at the end of the run.

### 8. Verification checklist

After a full preproc run, verify along these directions:

- **Coordinate range**: no tool-call coordinate outside `[0, 1000]` (the OOB filter should have caught them; double-check with a parquet scan).
- **Image references**: every path in `images` resolves to an existing file under the image store.
- **Message format**: `messages` deserialize cleanly; non-final `use` assistant action/result-producing calls are paired with `role:"tool"` results by `tool_call_id` matching the assistant call's `id`; a final assistant tool-call turn may end at EOF without a synthetic result; `grounding.*` assistant calls are label targets and have no synthetic tool result; tool-call schemas match the action space class.
- **Visual spot-check**: sample a handful of rows per task type, render the image + overlay the coordinates, and eyeball whether actions land on the described UI element.
