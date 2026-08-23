# Lite.ScaleCUA Teacher-Data Pipeline

This directory owns the Lite.ScaleCUA GPT teacher-data workflow.

Lite.ScaleCUA materializes ScaleCUA's OSWorld-shaped `rl` and `train` task
catalogs, then runs them on the Lite.OSWorld desktop runtime. It uses the same
screenshot, coordinate-action, terminal, and 30-turn interaction substrate as
Lite.OSWorld. It therefore reuses the shared quality filter at
`devs/data/lite.osworld/filter.py`; its task pool, prompt, rollout logs, and
published dataset remain separate.

## Collection Targets

Lite.ScaleCUA exposes two registered splits, `rl` and `train`. Both include rows
with `metadata.others.exclude_reason`; teacher-data collection must filter those
rows before rollout.

| Split | Registered rows | Runnable rows | HF config |
|---|---:|---:|---|
| `rl` | 2,049 | 1,828 | `desktop.use.rl` |
| `train` | 20,289 | 16,179 | `desktop.use.train` |

Every collect command must include:

```bash
--filter "lambda m: not m.others.get('exclude_reason')"
```

`scripts/rollout.py` applies the filter before task execution.

## Prompt Design

Canonical recipe:
`scripts/configs/gpt/recipes/collect/lite.scalecua.yaml`.

The policy requires:

1. A concise `Thought` grounded in the current screenshot, prior action result,
   exact task values, and remaining global requirements.
2. At most three mechanically coupled tool calls per turn. Any action that
   changes the next UI surface ends the turn; do not predict unseen controls.
3. GUI-first execution. Terminal use is limited to explicitly command-line,
   inherently OS-level, or simple filesystem tasks with no suitable GUI.
   Interpreters, dependency installs, multiline scripts, complex shell logic,
   hidden config edits, and guessed internal paths are forbidden.
4. Exact custom-color hex values, clean temporary state, application-level
   saves, and committed settings.
5. At most one harmless reversible mistake, only when the task is simple,
   stable, and safely within budget. Never mention training or an intentional
   mistake in the inline reasoning.
6. Final completion only after the requested state is visibly verified and
   saved. Do not combine an ordinary state-changing action with termination.

The effective agent policy should remain aligned with the Lite.OSWorld recipe,
but environment-specific changes belong in this file.

## Shared Filter

Lite.ScaleCUA uses `devs/data/lite.osworld/filter.py`. It is an **annotation**
pass for ordinary quality gates: it keeps those trajectories and tags them in
`metadata.others.exclude_reason` (comma-joined; the key is omitted when clean).
Two publish-invalid classes are hard-dropped before staging: typed `/opt/env/`
tool leaks and out-of-range GUI coordinates. Downstream consumers filter with
`not m.others.get('exclude_reason')` (the same idiom as the task-level
`exclude_reason` above, though the meaning differs: task-level marks unrunnable
tasks, trajectory-level marks quality gates).

```bash
uv run python devs/data/lite.osworld/filter.py \
  --log-root <raw-log-root> \
  --out <annotated-log-root> \
  --drop-loops --drop-undo-storm
```

It tags, in `exclude_reason`:

- `incomplete` — `terminated != true`;
- `dependency_install` — apt/pip/conda/snap/flatpak installs;
- `complex_shell` — a non-teachable terminal *operation* (**operation-driven, not
  structure-driven**: for/while loops, multi-line blocks, `;` / `&&` / single `|`
  of simple commands are KEPT; tagged are `$()` / `<()` / backtick / heredoc,
  `python -c` / `bash -c`, running or authoring code scripts, `sed -i` in-place
  edits, dotfile / `.desktop` authoring, and awk state machines);
- `footgun:loop` / `footgun:undo_storm` — repeated-action loops / undo storms.

It hard-drops:

- typed `/opt/env/` paths — env-only tool leaks that are not reproducible on the faithful guest;
- OOB coordinates — coordinates outside normalized `[0, 1000]`, which fail publish validation.

Reward is deliberately **not** a tag: `episode_return` is already in
`metadata.others.episode_return`, so a consumer thresholds it directly (`episode_return > 0.5`).
`--drop-loops` / `--drop-undo-storm` are **still required** — in annotate mode they no longer drop,
they gate whether `footgun:loop` / `footgun:undo_storm` get **tagged**, so the
canonical annotate command must pass them.

On every kept trajectory it also strips `screenshot` and `wait` (keeps bare
Ctrl+S), flattens inline reasoning, and normalizes the content-only final turn to one plain `text` part (`{"type": "text", "text": "Done."}`) — unconditionally, whatever it held before (`inline_reasoning`, `action_description`, both, or anything else). A final turn that DOES carry `tool_calls` is untouched. Synthetic `terminate(status="success")` is opt-in only, and when enabled
the staged row must include the matching canonical nested `terminate` schema in
`metadata.extra_tool_schemas`.

There is no blanket terminal-command-count threshold because legitimate OS
tasks may need several simple commands.

Tests:

```bash
PYTHONPATH="$PWD" uv run pytest -n 0 \
  devs/data/lite.osworld/tests/test_lite_osworld_filter.py -q
```

## Complete Workflow

Run from the repository root. Freeze the code revision, task catalog lock,
prompt, task filter, and log root for each batch. A resume must use the
identical command.

### 1. Install And Configure

```bash
uv sync --locked --extra quick-start --extra gym
uv run --no-sync bash lite/gym/envs/lite/scalecua/scripts/install.sh
uv run python scripts/serve_env.py --port 30250 --env-ids lite.scalecua

HOST_IP=$(hostname -I | awk '{print $1}')
export CUA_LITE_ENV_SERVER_URL=http://${HOST_IP}:30250
: "${OPENAI_API_KEY:?set OPENAI_API_KEY before collection}"
# Optional; set only for a custom endpoint.
# export OPENAI_BASE_URL="..."

COMMIT="$(git rev-parse --short HEAD)"
TASK_FILTER="lambda m: not m.others.get('exclude_reason')"
```

Use the install script rather than manually importing catalogs. It ensures the
Lite.OSWorld base image is available and validates the ScaleCUA catalog lock.

### 2. Collect

Collect each source into its own subfolder (`rl` / `train`), keeping HF config
names separable at stage time.

```bash
uv run python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id lite.scalecua \
  --splits rl \
  --filter "$TASK_FILTER" \
  --concurrency 32 \
  --max-attempts 2 \
  --save-data true \
  --save-video true \
  --save-gif false \
  --config-path scripts/configs/gpt/recipes/collect/lite.scalecua.yaml \
  --log-root ".data/rollout/lite.scalecua/gpt/$COMMIT"

uv run python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id lite.scalecua \
  --splits train \
  --filter "$TASK_FILTER" \
  --sample 16179 \
  --concurrency 32 \
  --max-attempts 2 \
  --save-data true \
  --save-video true \
  --save-gif false \
  --config-path scripts/configs/gpt/recipes/collect/lite.scalecua.yaml \
  --log-root ".data/rollout/lite.scalecua/gpt/$COMMIT"
```

`--sample 16179` shuffles the whole runnable `train` set (16,179 = the post-filter
count; sampling N==len returns every task in random order, deterministic under the
default `--seed 42`). Without it, the catalog is grouped by domain, so tasks run
`chrome → multi_apps → vlc → gimp → …` in blocks; the shuffle interleaves domains so
any partial/resumed run stays domain-balanced. The final dataset is identical either
way — only execution order differs. (`rl` is small enough to leave in catalog order.)

Re-run the identical command to resume.

### 3. Annotate And Review

`filter.py` keeps ordinary quality-failed trajectories and writes
`metadata.others.exclude_reason` (see [Shared Filter](#shared-filter)); typed
`/opt/env/` leaks and OOB coordinates are hard-dropped.

```bash
uv run python devs/data/lite.osworld/filter.py \
  --log-root ".data/rollout/lite.scalecua/gpt/$COMMIT/rl" \
  --out ".data/rollout/lite.scalecua/gpt/$COMMIT/rl_annotated" \
  --drop-loops --drop-undo-storm

uv run python devs/data/lite.osworld/filter.py \
  --log-root ".data/rollout/lite.scalecua/gpt/$COMMIT/train" \
  --out ".data/rollout/lite.scalecua/gpt/$COMMIT/train_annotated" \
  --drop-loops --drop-undo-storm
```

Review the `exclude_reason` tag counts and sample every tag class, plus a sample
of clean (untagged) and terminal trajectories, before publishing.

### 4. Stage, Upload Transport, And Download

```bash
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
READBACK_ROOT="$PWD/.data/huggingface-readback"

uv run python -m lite.data.hf.stage \
  --log-roots ".data/rollout/lite.scalecua/gpt/$COMMIT/rl_annotated" \
              ".data/rollout/lite.scalecua/gpt/$COMMIT/train_annotated" \
  --config-names desktop.use.rl desktop.use.train \
  --name Lite.ScaleCUA \
  --description "Lite.ScaleCUA GPT-5.5 grounded teacher trajectories; ordinary quality gates tagged in metadata.others.exclude_reason, publish-invalid tool leaks/OOB coordinates hard-dropped (filter with not exclude_reason and episode_return>0.5)"

: "${HF_ORG:?set HF_ORG to your Hub user/org for the private smoke repo}"
uv run python -m lite.data.hf.upload Lite.ScaleCUA --org "$HF_ORG" --private --tag "$COMMIT"

uv run python -m lite.data.hf.download Lite.ScaleCUA \
  --org "$HF_ORG" \
  --revision "$COMMIT" \
  --out "${READBACK_ROOT}/cua-lite/Lite.ScaleCUA"
```

Record `stage`'s final `seen=... kept=... dropped_by_filter=...` line and the
per-config row lines as the publish gate. Upload/download are transport/layout
smokes only; row content was already gated by `stage`.

### 5. Export SFT Parquet

```bash
uv run python -m lite.train.export.export_sft \
  --config scripts/configs/qwen3_5/default/lite.osworld.yaml \
  --model-id Qwen/Qwen3.5-9B \
  --data-paths "${READBACK_ROOT}/cua-lite/Lite.ScaleCUA" \
  --image-root "${READBACK_ROOT}" \
  --filter "lambda m: not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) > 0.5" \
  --num-proc 16 \
  -o .data/sft/qwen3_5/lite-scalecua/train.parquet
```

`--config` is the **rollout** config, not an SFT-only recipe under
`scripts/configs/*/recipes/sft/`: `export_sft` re-renders every step through the
agent adapter, so exporting under a different history window or resolution than
the rollout used trains the model on prompts it will never see at inference.
There is no `qwen3_5/default/lite.scalecua.yaml` — `lite.scalecua` is an
OSWorld-task adapter riding the `lite.osworld` desktop substrate, and the only
fields `export_sft` reads (`agent_id`, `agent_kwargs`)
are identical across the family's `default/*.yaml`.

For a joint Lite.OSWorld + Lite.ScaleCUA run, pass both dataset paths to one
export command. The processor/model ID must match training because tokenization
and the chat template are frozen during export.

Keep fail-fast enabled; use `--no-strict` only for an identified and recorded
corrupt source row.

### 6. Continue Collecting Across Machines (per-split resume)

Split the full collection across machines (or resume after an interruption) without
re-running what's already been *attempted*. Coordinate through a **throwaway temp HF
dataset** — NOT the canonical `Lite.ScaleCUA`, and **NOT filtered**.

Why raw + temp: this round-trip's only job is to tell every machine "which
`(task, sample)` is already done" so it isn't redone. `scripts/rollout.py` resume keys
off `sample_*/summary.json` presence (`get_pending`); `hf.unstage` recreates those
summaries from any staged dataset. So stage the **RAW** log-root — **skip `filter.py`**:
failures carry a summary too, so resume then **skips every attempted sample (success OR
failure)** rather than re-running failures. `filter.py` (the annotation pass — it keeps
ordinary quality-failed trajectories, tags `exclude_reason`, and hard-drops
publish-invalid tool leaks/OOB rows; see [Shared Filter](#shared-filter)) runs
**ONCE at the very end** on the final merged log-root to produce the canonical dataset
(steps 3 → 4). Delete the temp dataset afterward.

**One temp dataset, both configs.** Stage `rl` and `train` into a SINGLE repo as the two
configs `desktop.use.rl` / `desktop.use.train` (the same layout the canonical dataset
uses), so the temp dataset IS the accumulating superset and whichever machine finishes
last can produce the complete canonical upload from its own disk. Screenshots ride
embedded in the parquet, so a downloaded config carries its own images. Requirements:
every machine shares the same catalog lock / `$COMMIT` (so `task_id`s match); each
`--config-names ↔ --splits` pair below uses the registry split name.

Machine A — publish raw progress, both configs, no `filter.py`. This WIP upload
is transport/resume coordination only; it is not canonical publish validation:
```bash
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
RESUME_ROOT=".data/rollout/lite.scalecua/gpt/$COMMIT"

uv run python -m lite.data.hf.stage \
  --log-roots "$RESUME_ROOT/rl" "$RESUME_ROOT/train" \
  --config-names desktop.use.rl desktop.use.train \
  --name Lite.ScaleCUA.wip --description "TEMP resume coordination (raw; delete after merge)"
: "${HF_ORG:?set HF_ORG to your Hub user/org for the private temp repo}"
uv run python -m lite.data.hf.upload Lite.ScaleCUA.wip --org "$HF_ORG" --private --tag "$COMMIT"
```

Machine B — ONE download → unstage EACH config into its own registry-split dir → resume.
`hf.unstage --config-names` reads ONLY that config's parquet, so `rl` rows land under
`rl/` and `train` rows under `train/` with no cross-contamination (a plain `unstage`
globs *all* parquet into the single `--splits` dir, misfiling the other config and
breaking its resume + the final config split). `--filter "$TASK_FILTER"` here is the
*task* exclude-reason gate, unrelated to `filter.py`:
```bash
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
RESUME_ROOT=".data/rollout/lite.scalecua/gpt/$COMMIT"

# one full download (both configs; images are embedded in the parquet)
uv run python -m lite.data.hf.download Lite.ScaleCUA.wip \
  --org "$HF_ORG" \
  --out "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA.wip"

# route each config to its registry split (no --allow-patterns needed)
uv run python -m lite.data.hf.unstage \
  --dataset "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA.wip" \
  --log-root "$RESUME_ROOT" --splits rl    --config-names desktop.use.rl
uv run python -m lite.data.hf.unstage \
  --dataset "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA.wip" \
  --log-root "$RESUME_ROOT" --splits train --config-names desktop.use.train

# resume each split — identical to the §2 collect commands (resume skips attempted)
uv run python scripts/rollout.py --model-id gpt-5.5 --env-id lite.scalecua --splits rl \
  --filter "$TASK_FILTER" --concurrency 32 --max-attempts 2 \
  --save-data true --save-video true --save-gif false \
  --config-path scripts/configs/gpt/recipes/collect/lite.scalecua.yaml --log-root "$RESUME_ROOT"
uv run python scripts/rollout.py --model-id gpt-5.5 --env-id lite.scalecua --splits train \
  --filter "$TASK_FILTER" --sample 16179 --concurrency 32 --max-attempts 2 \
  --save-data true --save-video true --save-gif false \
  --config-path scripts/configs/gpt/recipes/collect/lite.scalecua.yaml --log-root "$RESUME_ROOT"
```

Each machine republishes its grown log-root exactly as Machine A did (stage both configs
→ upload `Lite.ScaleCUA.wip`), so the temp dataset always holds the union. The machine
that finishes last holds the complete superset on disk; it runs `filter.py` (step 3, the
annotation pass) → stages the CANONICAL `Lite.ScaleCUA` with both configs → uploads
(step 4) → exports SFT (step 5), then deletes `Lite.ScaleCUA.wip`.

Verified: the unit round-trip
(`tests/data/hf/test_unstage.py::test_multi_config_unstage_routes_each_config_to_its_split`),
plus an offline unstage of the real staged dataset — `--config-names desktop.use.rl
--splits rl` / `desktop.use.train --splits train` routed **1839 `rl` + 3844 `train`**
trajectories into their own split dirs with **zero cross-contamination**, so per-split
resume is correct. Because RAW stages every attempted sample (success **or** failure),
resume treats *attempted* as done and never re-runs it; `filter.py` annotates
ordinary quality gates and hard-drops publish-invalid tool leaks/OOB rows once at
the end.
