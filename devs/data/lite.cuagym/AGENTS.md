# Lite.CUAGym Teacher-Data Pipeline

This directory owns the Lite.CUAGym GPT teacher-data workflow.

Lite.CUAGym imports CUA-Gym web, cross-app, and desktop task bundles. Its
Docker image extends `cua-lite/lite.osworld:latest`, and its desktop tasks use
the same screenshot, coordinate-action, terminal, and 30-turn interaction
substrate as Lite.OSWorld. It therefore reuses the shared quality filter at
`devs/data/lite.osworld/filter.py`; its task pool, prompt, rollout logs, and
published dataset remain separate.

## Collection Targets

Lite.CUAGym exposes one registered `train` split of 10,910 tasks (1,505 web +
9,405 desktop — every pinned upstream row is registered; none is dropped). For
teacher-data collection, use either a frozen audited `--prompt-data` parquet or
an explicitly frozen sample/seed/task-id list. The full registered train split
includes known upstream setup/reward failures, so a seed alone does not define a
publishable collection.

**494 of those 10,910 rows are unusable as default training signals** and carry
a task-level `metadata.others.exclude_reason` from the closed vocabulary in
[/lite/gym/envs/lite/cuagym/src/utils/dataset.py](/lite/gym/envs/lite/cuagym/src/utils/dataset.py)
(broken/empty/no-sentinel/mismatched `reward.py`, unbuildable GitHub/Trello
mocks, Google Drive blank-render rows, and deterministic pinned setup defects).
Nothing is dropped — the rows are annotated and you filter them out, leaving
10,416 default-collectable tasks:

```bash
--filter "lambda m: not m.others.get('exclude_reason')"
```

`rollout.py --filter` applies only when tasks are selected from the registry.
If you collect from `--prompt-data`, apply the task-level filter before freezing
that parquet; `--filter` is rejected with `--prompt-data` by design. Freezing a
`--prompt-data` parquet without that filter costs container boots, not data:
`guard_excluded` in
[/lite/gym/envs/lite/cuagym/main.py](/lite/gym/envs/lite/cuagym/main.py) refuses
each such row at setup with `CuaGymTaskError(kind="excluded_task")` — a terminal
(non-retryable) error, so it is not re-run by `--max-attempts`, produces no
trajectory, and is excluded from the reported mean rather than averaged in as a
zero (see [/devs/envs/lite.cuagym/UPSTREAM_ISSUES.md](/devs/envs/lite.cuagym/UPSTREAM_ISSUES.md)).

Note the two independent namespaces that share the `exclude_reason` key: the
**task-level** one above (catalog rows, applied at import, filtered before
rollout) and the **trajectory-level** one written by the shared filter after
rollout ([Shared Filter](#shared-filter)). They never collide — the first lives
on registry task metadata, the second on collected trajectory metadata — but do
not read a count of one as a count of the other.

## Prompt Design

Canonical recipe:
`scripts/configs/gpt/recipes/collect/lite.cuagym.yaml`.

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

Lite.CUAGym uses [/devs/data/lite.osworld/filter.py](/devs/data/lite.osworld/filter.py).
It is mostly an **annotation** pass: it keeps trajectories and tags quality gates
in the *trajectory-level* `metadata.others.exclude_reason` (comma-joined; the key
is omitted when clean) — a separate namespace from the *task-level*
`exclude_reason` on catalog rows (see [Collection Targets](#collection-targets)).
Two publish-invalid classes are physically dropped before staging: trajectories
whose agent typed a `/opt/env/` path, and trajectories with GUI coordinates
outside normalized `[0, 1000]`. Downstream consumers filter with
`not m.others.get('exclude_reason')`.

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
- `footgun:loop` / `footgun:undo_storm` — repeated-action loops / undo storms
  when `--drop-loops` / `--drop-undo-storm` are passed;
- `footgun:no_submit` — no explicit final submit tool (`terminate`/`response`),
  only when `--drop-no-submit` is passed; this is separate from the default
  content-only final turn policy (normalized to one plain `text` part, not preserved as emitted);
- `oob_coordinate` — a coordinate outside normalized `[0, 1000]`;
- `reward_vision_disagree` — scalar reward and multi-frame visual judgement disagree.

Reward is deliberately **not** a tag: `episode_return` is already in
`metadata.others.episode_return`, so a consumer thresholds it directly.

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

Run from the repository root. Freeze the code revision, audited task parquet,
prompt, and log root for each batch. A resume must use the identical command.

### 1. Install And Configure

```bash
uv sync --locked --extra quick-start --extra gym
uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh

# Default agent + reward-judge route. Set these before starting the env-server;
# the CUA-Gym reward judge defaults to the same endpoint settings.
: "${OPENAI_API_KEY:?set OPENAI_API_KEY before collection}"
# Optional; set only for a custom endpoint.
# export OPENAI_BASE_URL="..."
# Set LITE_CUAGYM_JUDGE_* for judge-specific model/base URL/API key/retry/timeout;
# VLM_* are compatibility aliases with lower precedence.

# Start your own env-server on a free port:
uv run python scripts/serve_env.py --port <PORT> --env-ids lite.cuagym
HOST_IP=$(hostname -I | awk '{print $1}')
export CUA_LITE_ENV_SERVER_URL=http://${HOST_IP}:<PORT>

COMMIT="$(git rev-parse --short HEAD)"
CUAGYM_INPUT="<frozen-audited-prompt-data.parquet>"
```

Use the install script rather than a plain Docker build. It provisions pinned
task catalogs/assets, ensures the Lite.OSWorld-derived base when an image build
is needed, and stamps the env-server freshness label.

The image BAKES the web mocks that the imported catalog references. Normal
source/lock/importer changes are covered by image freshness and the mock build
stamp, but manually corrupted local caches are not a source change. After any
forced re-import or catalog repair, run
`uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh provision`;
after HF mirror/cache repair, run maintainer helper
`uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh assets`;
then run
`uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh rebuild`
if local mock dists or the image may be stale.
Sanity-check the import line before building: it must report ~1505 web tasks
across **31 apps** — `across 0 apps` means the catalog is broken.

### 2. Collect

```bash
uv run python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id lite.cuagym \
  --prompt-data "$CUAGYM_INPUT" \
  --concurrency 15 \
  --max-attempts 3 \
  --save-data true \
  --save-video false \
  --save-gif false \
  --config-path scripts/configs/gpt/recipes/collect/lite.cuagym.yaml \
  --log-root ".data/rollout/lite.cuagym/gpt/$COMMIT"
```

For a fresh sampled pool, freeze `--sample`, `--seed`, and the resolved task
IDs. Re-run the identical command to resume.

### 3. Annotate And Review

`filter.py` keeps every trajectory except the `/opt/env/` and OOB-coordinate
hard-drop cases and writes `metadata.others.exclude_reason` (see
[Shared Filter](#shared-filter)).

```bash
uv run python devs/data/lite.osworld/filter.py \
  --log-root ".data/rollout/lite.cuagym/gpt/$COMMIT/train" \
  --out ".data/rollout/lite.cuagym/gpt/$COMMIT/train_annotated" \
  --drop-loops --drop-undo-storm
```

The `$COMMIT/train` path exists only when every row of the frozen
`--prompt-data` parquet carries `split: "train"`; rows without a `split` field
land under `$COMMIT/parquet/` instead (see
[/lite/infer/rollout.py](/lite/infer/rollout.py)), so freeze the parquet with an
explicit `train` split on every row.

Review the `exclude_reason` tag counts and sample every tag class, plus a sample
of clean (untagged) and terminal trajectories, before publishing.

### 4. Stage, Upload Transport, And Download

```bash
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
READBACK_ROOT="$PWD/.data/huggingface-readback"

uv run python -m lite.data.hf.stage \
  --log-roots ".data/rollout/lite.cuagym/gpt/$COMMIT/train_annotated" \
  --name Lite.CUAGym \
  --repo-dir devs/data/lite.cuagym

: "${HF_ORG:?set HF_ORG to your Hub user/org for the private smoke repo}"
uv run python -m lite.data.hf.upload Lite.CUAGym --org "$HF_ORG" --private --tag "$COMMIT"

uv run python -m lite.data.hf.download Lite.CUAGym \
  --org "$HF_ORG" \
  --revision "$COMMIT" \
  --out "${READBACK_ROOT}/cua-lite/Lite.CUAGym"
```

Record `stage`'s final `seen=... kept=... dropped_by_filter=...` line and the
per-config row lines as the publish gate. Upload/download are transport/layout
smokes only; row content was already gated by `stage`.

### 5. Export SFT Parquet

```bash
uv run python -m lite.train.export.export_sft \
  --config scripts/configs/qwen3_5/default/lite.cuagym.yaml \
  --model-id Qwen/Qwen3.5-9B \
  --data-paths "${READBACK_ROOT}/cua-lite/Lite.CUAGym" \
  --image-root "${READBACK_ROOT}" \
  --filter "lambda m: not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) > 0.5" \
  --num-proc 16 \
  -o .data/sft/qwen3_5/lite-cuagym/train.parquet
```

`--config` is the **rollout** config, not an SFT-only recipe under
`scripts/configs/*/recipes/sft/`: `export_sft` re-renders every step through the
agent adapter, so exporting under a different history window or resolution than
the rollout used trains the model on prompts it will never see at inference.

For a joint Lite.OSWorld + Lite.CUAGym run, pass both dataset paths to one
export command. The processor/model ID must match training because tokenization
and the chat template are frozen during export.

Keep fail-fast enabled; use `--no-strict` only for an identified and recorded
corrupt source row.
