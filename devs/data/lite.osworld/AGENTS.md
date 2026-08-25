# Lite.OSWorld Teacher-Data Pipeline

This directory owns Lite.OSWorld GPT teacher-data collection and the shared
desktop trajectory filter used by Lite.OSWorld-family teacher-data workflows.

Lite.CUAGym and Lite.ScaleCUA have their own workflow documentation in
`devs/data/lite.cuagym/AGENTS.md` and `devs/data/lite.scalecua/AGENTS.md`.
They reuse `filter.py`; the datasets, prompts, rollout inputs, and published
artifacts remain separate.

## Collection Targets

Collect only the training sub-splits. Do not collect the eval split for SFT.

The synth sub-split currently includes rows with
`metadata.others.exclude_reason` (quarantined/unrunnable tasks such as OCR
verification gaps or upstream live-site drift); teacher-data collection must
filter those rows before rollout. Perturb currently has no task-level
exclusions, but every collect command still carries the same filter.

| Split | Registered rows | Runnable rows | HF config |
|---|---:|---:|---|
| `train.synth` | 1,722 | 1,704 | `desktop.use.synth` |
| `train.perturb` | 707 | 707 | `desktop.use.perturb` |

Every collect command must include:

```bash
--filter "lambda m: not m.others.get('exclude_reason')"
```

`scripts/rollout.py` applies the filter before task execution. (This is the
same task-level idiom as Lite.ScaleCUA — see
`devs/data/lite.scalecua/AGENTS.md`.)

Collect, filter, and stage the two sources separately. `train.perturb` is
derived from eval setups, so it must stay identifiable for train/eval-leakage
review before mixing into any SFT set.

## Prompt Design

Canonical recipe:
`scripts/configs/gpt/recipes/collect/lite.osworld.yaml`.

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

Prompt changes alter the training distribution. Run matched-task smoke tests
before a full collection.

## Shared Filter

`filter.py` is the single quality-**annotation** pass before staging for desktop
datasets. It **keeps every ordinary quality-failed trajectory** and tags gates in
`metadata.others.exclude_reason` (comma-joined; the key is omitted when clean) —
downstream consumers filter with `not m.others.get('exclude_reason')`, exactly
like the task-level idiom. The only physical drops are env-tool leaks and OOB
coordinates: trajectories whose agent typed a `/opt/env/` path are not
reproducible in the faithful guest, and out-of-range GUI coordinates fail the
staging row-format check.

```bash
uv run python devs/data/lite.osworld/filter.py \
  --log-root <raw-log-root> \
  --out <annotated-log-root> \
  --drop-loops --drop-undo-storm
```

It tags, in `exclude_reason`:

- `incomplete` — `terminated != true`;
- `dependency_install` — apt/pip/conda/snap/flatpak installs;
- `complex_shell` — a non-teachable terminal *operation* (see below);
- `footgun:loop` / `footgun:undo_storm` / `footgun:no_submit` — ≥3 identical consecutive actions / ≥4 Ctrl+Z / no submit action;

Reward is deliberately **not** a tag: `episode_return` is already in
`metadata.others.episode_return`, so a consumer thresholds it directly (`episode_return > 0.5`).
Ordinary quality gates are not dropped. `--drop-loops` / `--drop-undo-storm` are
**still required** —
in annotate mode they no longer drop, they gate whether `footgun:loop` /
`footgun:undo_storm` get **tagged**, so the canonical annotate command passes
them.

`complex_shell` is **operation-driven, not structure-driven**: for/while loops,
multi-line blocks, `;` / `&&` / single `|` of SIMPLE commands are kept (efficient
repetition). Tagged are operations a small GUI model can't ground from a
screenshot: nested/substituted execution (`$()` / `<()` / backtick / heredoc),
inline interpreters (`python -c` / `bash -c`) and running or authoring code
scripts, `sed -i` in-place file surgery, dotfile / `.desktop` authoring, and awk
state machines (`next` / `exit`).

On every (kept) trajectory it also: strips no-op `screenshot` and `wait` calls
or action-batch child actions, preserving the canonical action-batch call and any remaining
child actions; keeps bare Ctrl+S; flattens inline reasoning to one line; keeps
the content-only final channel, normalized to one plain `text` part — `"Done."`
is now MANDATORY, not merely common: whatever the turn held before
(`inline_reasoning`, `action_description`, both) is replaced. A final turn
carrying `tool_calls` (a QA answer submitted through `response`, a `terminate`)
is untouched.
Synthetic `terminate(status="success")` is opt-in only, and when enabled the
staged row must include the matching canonical nested `terminate` schema in
`metadata.extra_tool_schemas`.

There is intentionally no blanket terminal-command-count threshold. Some OS
tasks need several simple commands; tag unsafe operations, not an arbitrary
count.

Tests:

```bash
PYTHONPATH="$PWD" uv run pytest -n 0 \
  devs/data/lite.osworld/tests/test_lite_osworld_filter.py -q
```

## Complete Workflow

Run from the repository root. Pipeline: collect → filter/annotate → stage →
upload/download → `export_sft`. Freeze the code revision, input task set,
prompt, and log root for each batch. A resume must use the identical command.

### 1. Install And Configure

```bash
uv sync --locked --extra quick-start --extra gym
uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/install.sh
uv run python scripts/serve_env.py --port 30200 --env-ids lite.osworld

HOST_IP=$(hostname -I | awk '{print $1}')
export CUA_LITE_ENV_SERVER_URL=http://${HOST_IP}:30200
: "${OPENAI_API_KEY:?set OPENAI_API_KEY before collection}"
# Optional; set only for a custom endpoint.
# export OPENAI_BASE_URL="..."

COMMIT="$(git rev-parse --short HEAD)"
```

Use the install script rather than a plain Docker build; it stamps the source
freshness label required by env-server. Collection should use env-server mode;
the commands below assume a 32-ish rollout batch against that server.

### 2. Collect

Collect each source into its OWN subfolder (`train.synth` / `train.perturb`
are registered sub-splits): stage maps log-roots 1:1 to config names, so the
canonical `desktop.use.synth` / `desktop.use.perturb` separation depends on
keeping them apart here.

```bash
for SUB in synth perturb; do
  uv run python scripts/rollout.py \
    --model-id gpt-5.5 \
    --env-id lite.osworld \
    --splits "train.$SUB" \
    --concurrency 32 \
    --max-attempts 3 \
    --save-data true \
    --save-video false \
    --save-gif false \
    --filter "lambda m: not m.others.get('exclude_reason')" \
    --config-path scripts/configs/gpt/recipes/collect/lite.osworld.yaml \
    --log-root ".data/rollout/lite.osworld/gpt/$COMMIT"
done
```

This covers all 2,429 registered train tasks (1,722 synth + 707 perturb); the
`--filter` runs the 2,411 runnable (1,704 synth + 707 perturb), skipping the
18 quarantined synth rows. Re-run the same command to resume.

### 3. Annotate And Review

`filter.py` writes `metadata.others.exclude_reason` for ordinary quality gates
(see [Shared Filter](#shared-filter)); only `/opt/env` leaks and OOB coordinates
are physically dropped.

```bash
for SUB in synth perturb; do
  uv run python devs/data/lite.osworld/filter.py \
    --log-root ".data/rollout/lite.osworld/gpt/$COMMIT/train.$SUB" \
    --out ".data/rollout/lite.osworld/gpt/$COMMIT/train.${SUB}_annotated" \
    --drop-loops --drop-undo-storm
done
```

Review the hard-drop counts, the `exclude_reason` tag counts, and sample every
tag class, plus a sample of clean (untagged) and terminal trajectories, before
publishing. Re-run into a fresh annotated root, or pass `--overwrite` only when
intentionally replacing the entire previous output tree.

### 4. Stage, Upload Transport, And Download

```bash
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
READBACK_ROOT="$PWD/.data/huggingface-readback"

uv run python -m lite.data.hf.stage \
  --log-roots ".data/rollout/lite.osworld/gpt/$COMMIT/train.synth_annotated" \
              ".data/rollout/lite.osworld/gpt/$COMMIT/train.perturb_annotated" \
  --config-names desktop.use.synth desktop.use.perturb \
  --name Lite.OSWorld \
  --repo-dir devs/data/lite.osworld

: "${HF_ORG:?set HF_ORG to your Hub user/org for the private smoke repo}"
uv run python -m lite.data.hf.upload Lite.OSWorld --org "$HF_ORG" --private --tag "$COMMIT"

# Consumer / verification: pull the uploaded revision into canonical local layout.
# NOTE: download verifies LAYOUT only; row content was already gated by stage above.
# upload/download are transport/layout steps, and export_sft below is a conversion smoke.
# See /devs/migration/AGENTS.md#1-download-the-source.
uv run python -m lite.data.hf.download Lite.OSWorld \
  --org "$HF_ORG" \
  --revision "$COMMIT" \
  --out "${READBACK_ROOT}/cua-lite/Lite.OSWorld"
```

Record `stage`'s final `seen=... kept=... dropped_by_filter=...` line and the
per-config row lines as the publish gate. Upload is transport only; use the
release org only after the private upload/readback/export smoke is approved.

### 5. Export SFT Parquet

```bash
uv run python -m lite.train.export.export_sft \
  --config scripts/configs/qwen3_5/default/lite.osworld.yaml \
  --model-id Qwen/Qwen3.5-9B \
  --data-paths "${READBACK_ROOT}/cua-lite/Lite.OSWorld" \
  --image-root "${READBACK_ROOT}" \
  --filter "lambda m: not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) > 0.5" \
  --num-proc 16 \
  -o .data/sft/qwen3_5/lite-osworld/train.parquet
```

`--config` is the **rollout** config, not an SFT-only recipe under
`scripts/configs/*/recipes/sft/`: `export_sft` re-renders every step through the
agent adapter, so exporting under a different history window or resolution than
the rollout used trains the model on prompts it will never see at inference.

The processor/model ID must match training because tokenization and the chat
template are frozen during export. Keep fail-fast enabled; use `--no-strict`
only for an identified and recorded corrupt source row.
