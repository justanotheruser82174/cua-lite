# Lite.CUAWorld Teacher-Data Pipeline

This directory owns the Lite.CUAWorld GPT teacher-data workflow.

Lite.CUAWorld re-hosts the **gym-anything (CMU CUA-World)** desktop software task suite — **40
softwares, one Docker image each** (`lite.cuaworld.<software>`). Its desktop tasks use the same
screenshot, coordinate-action, and terminal substrate as Lite.OSWorld, so it
**reuses the shared quality filter at [`devs/data/lite.osworld/filter.py`](/devs/data/lite.osworld/filter.py)**
(like Lite.CUAGym and Lite.ScaleCUA). Its task pool, prompt, rollout logs, and published dataset
remain separate. Env content lives in the HF materials dataset `cua-lite/lite.cuaworld-assets`
(pinned by `data/assets.lock.yaml`); this repo is engine + pipeline only.

## Collection Targets

Each software registers the splits present in the current materials: `train`
(rollout target for SFT), `eval` = **CUAWorld-Test**, and sometimes
`long_horizon` = **CUAWorld-Long** (1/software). For SFT, roll out `train` on the
38 rolloutable softwares; collect `eval`/`long_horizon` only as held-out reference
sets with an explicit software list, and keep them **identifiable and out of any
SFT mix** (mixing them into train is eval leakage). Current locked-materials snapshot
over the 40 softwares:

| split | registered | live after task excludes | HF config |
|---|---:|---:|---|
| `train` | 2,419 | 1,745 | `desktop.use.<software>` (one config per software) |
| `eval` (CUAWorld-Test) | 626 | 440 | held-out reference, **not** SFT |
| `long_horizon` (CUAWorld-Long) | 38 | 18 | held-out reference, **not** SFT |

**Excluded softwares** (single-digit / incomplete onboarding): `knime` (0 train) and `freecad`
(5 train, all validation-excluded as `gameable_full`) → **38 softwares** carry a real train pool
(18–135 each after task excludes).

**Excluded tasks (task-level, pre-rollout).** A no-LLM validation sweep flagged 880 tasks
(across all splits) whose setup/verify pipeline is broken or gameable regardless of the agent.
They are recorded in
[`/lite/gym/envs/lite/cuaworld/data/validation_excludes.json`](/lite/gym/envs/lite/cuaworld/data/validation_excludes.json)
and the engine bakes each into `metadata.others['exclude_reason']` at registration
(`src/software.py::_exclude_reasons`). This is the **same `exclude_reason` idiom** the shared filter
uses at the trajectory level (below), so both compose under one downstream selector. Reason-code
breakdown: [`/devs/envs/lite.cuaworld/UPSTREAM_ISSUES.md`](/devs/envs/lite.cuaworld/UPSTREAM_ISSUES.md).

## Prompt Design

Canonical recipe: `scripts/configs/gpt/recipes/collect/lite.cuaworld.yaml` — **functionally
identical to `lite.osworld`'s collect recipe** except `env_id` (a nominal default; `rollout.py`
overrides it per software via `--env-id`). The policy requires:

1. A concise `Thought` grounded in the current screenshot, prior action result, exact task values,
   and remaining global requirements.
2. At most three mechanically coupled tool calls per turn. Any action that changes the next UI
   surface ends the turn; do not predict unseen controls.
3. GUI-first execution. Terminal use is limited to explicitly command-line, inherently OS-level, or
   simple filesystem tasks with no suitable GUI. Interpreters, dependency installs, multiline
   scripts, complex shell logic, hidden config edits, and guessed internal paths are forbidden.
4. Exact custom-color hex values, clean temporary state, application-level saves, committed settings.
5. At most one harmless reversible mistake, only when the task is simple, stable, and safely within
   budget. Never mention training or an intentional mistake in the inline reasoning.
6. Final completion only after the requested state is visibly verified and saved. Do not combine an
   ordinary state-changing action with termination.

The agent policy stays aligned with the Lite.OSWorld recipe; environment-specific changes belong in
this file. Prompt changes alter the training distribution — smoke-test a few softwares first.

## Shared Filter

Lite.CUAWorld uses [`devs/data/lite.osworld/filter.py`](/devs/data/lite.osworld/filter.py). It is mostly an
**annotation** pass: it keeps trajectories and tags quality gates in
`metadata.others.exclude_reason` (comma-joined; the key is omitted when clean). Two
publish-invalid classes are physically dropped before staging: a trajectory whose
agent typed a `/opt/env/` path leaked an env-only tool tree and is
non-reproducible, and any trajectory with GUI coordinates outside normalized
`[0, 1000]` would fail publish validation. Downstream selects the training set with
`not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) > 0.5`.

```bash
uv run python devs/data/lite.osworld/filter.py \
  --log-root <raw-log-root> \
  --out <annotated-log-root> \
  --drop-loops --drop-undo-storm
```

It tags, in `exclude_reason`:

- `incomplete` — `terminated != true`;
- `dependency_install` — apt/pip/conda/snap/flatpak installs;
- `complex_shell` — a non-teachable terminal *operation* (operation-driven, not structure-driven:
  loops / `;` / `&&` / single `|` of simple commands are KEPT; tagged are `$()` / `<()` / backtick /
  heredoc, `python -c` / `bash -c`, running or authoring code scripts, `sed -i`, dotfile authoring,
  awk state machines);
- `footgun:loop` / `footgun:undo_storm` — ≥3 identical consecutive actions or
  ≥4 Ctrl+Z when `--drop-loops` / `--drop-undo-storm` are passed;
- `footgun:no_submit` — no explicit final submit tool (`terminate`/`response`),
  only when `--drop-no-submit` is passed; this is separate from the default
  content-only final turn policy (normalized to one plain `text` part, not preserved as emitted);
- `oob_coordinate` — a coordinate outside normalized `[0, 1000]`;
- `reward_vision_disagree` — scalar reward and multi-frame visual judgement disagree.

Reward is **not** a tag (`episode_return` is in `metadata.others.episode_return` for the consumer to threshold).
On every kept trajectory it strips `screenshot` and `wait` (keeps a bare
Ctrl+S), flattens inline reasoning, and normalizes the content-only final turn to one plain `text` part (`{"type": "text", "text": "Done."}`) — unconditionally, whatever it held before (`inline_reasoning`, `action_description`, both, or anything else). A final turn that DOES carry `tool_calls` is untouched. Synthetic
`terminate(status="success")` is opt-in only, and when enabled the staged row must include the matching
canonical nested `terminate` schema in `metadata.extra_tool_schemas`.

Tests: `uv run pytest devs/data/lite.osworld/tests/test_lite_osworld_filter.py`.

## Complete Workflow

Run from the repository root. Freeze the code revision, task set, prompt, and log root per batch; a
resume must use the identical command.

### 1. Install And Configure

```bash
uv sync --locked --extra quick-start --extra gym

# Default agent + VLM judge route. Set these before starting the env-server;
# the host-side CUAWorld judge reads the same process environment.
: "${OPENAI_API_KEY:?set OPENAI_API_KEY before collection}"
# Optional; set only for a custom endpoint.
# export OPENAI_BASE_URL="..."

# Usually no VLM_* vars are needed: the default judge model is in
# /lite/gym/envs/lite/cuaworld/configs/default.yaml and litellm reads OPENAI_*.
# VLM_MODEL overrides LITE_CUAWORLD_VLM_MODEL, which overrides that default.
# Only set VLM_BASE_URL / VLM_API_KEY when overriding the judge route/key.

# Optional: point at a local materials checkout to avoid HF reads while validating
# local materials edits. Leave unset for the locked public HF materials.
# export LITE_CUAWORLD_MATERIALS_REPO=/path/to/lite.cuaworld-assets

COMMIT="$(git rev-parse --short HEAD)"

# Build the explicit software list you will collect/eval ONCE (stamps the
# lite.src_hash freshness label env-server needs).
# NOTE: images MUST be built by the current engine (stdio protocol v4 — the default-user revert
# bumped it from v3 so stale `:shim` images hard-fail instead of silently running the old
# contract); images from an older protocol are rejected at runtime. Full-suite
# validation can cover all 40, while collection usually uses the rolloutable subset.
uv run --no-sync bash lite/gym/envs/lite/cuaworld/scripts/install.sh build <software>   # per software
# uv run --no-sync bash lite/gym/envs/lite/cuaworld/scripts/install.sh rebuild <software>
# uv run --no-sync bash lite/gym/envs/lite/cuaworld/scripts/install.sh provision <software>  # no Docker build

# Start your own env-server on a free port:
CUAWORLD_ENVS="<space-separated lite.cuaworld.<software> ids you built>"
uv run python scripts/serve_env.py --port <PORT> --env-ids $CUAWORLD_ENVS &
HOST_IP=$(hostname -I | awk '{print $1}')
export CUA_LITE_ENV_SERVER_URL=http://${HOST_IP}:<PORT>
```

Before scaling collection, run the same one-task smoke command shown in the
env README and default GPT config, then inspect the saved trajectory shape:

```bash
uv run python scripts/rollout.py --model-id gpt-5.5 \
  --env-id lite.cuaworld.pymol --splits eval --head 1 \
  --concurrency 1 --max-attempts 1 --save-data true \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --config-path scripts/configs/gpt/default/lite.cuaworld.yaml
```

### 2. Collect

Loop the 38 rolloutable softwares on `train`, each into its own subfolder (stage maps log-roots 1:1 to config
names). The rollout `--filter` excludes every task the engine flagged with `exclude_reason`.
Keep total live rollout concurrency across CUAGym/CUAWorld at 48 or lower; when both are active,
budget them explicitly (for example 24 + 24) instead of letting one campaign monopolize the host.
Start from 24 for CUAWorld and raise only after the env-server, Docker, and provider error rates are stable.

```bash
CUAWORLD_CONCURRENCY="${CUAWORLD_CONCURRENCY:-24}"
for SW in <the 38 rolloutable softwares>; do
  uv run python scripts/rollout.py \
    --model-id gpt-5.5 \
    --env-id "lite.cuaworld.$SW" \
    --splits train \
    --concurrency "$CUAWORLD_CONCURRENCY" \
    --max-attempts 3 \
    --save-data true \
    --save-video false \
    --save-gif false \
    --config-path scripts/configs/gpt/recipes/collect/lite.cuaworld.yaml \
    --filter "lambda m: not m.others.get('exclude_reason')" \
    --log-root ".data/rollout/lite.cuaworld/gpt/$COMMIT/$SW"
done
```

Re-run the identical command to resume. If you collect `eval`/`long_horizon` as held-out reference
sets, use a separate explicit software list and a separate log root, and keep those splits out of the
SFT mix.

### 3. Annotate And Review

`filter.py` keeps every trajectory except the `/opt/env/` and OOB-coordinate
hard-drops and writes `metadata.others.exclude_reason`.

```bash
for SW in <softwares>; do
  uv run python devs/data/lite.osworld/filter.py \
    --log-root ".data/rollout/lite.cuaworld/gpt/$COMMIT/$SW/train" \
    --out     ".data/rollout/lite.cuaworld/gpt/$COMMIT/$SW/train_annotated" \
    --drop-loops --drop-undo-storm
done
```

Review the `exclude_reason` tag counts and sample each tag class + clean/terminal trajectories
before publishing. `devs/data/lite.cuaworld/analyze.py --log-root <root>` reports per-software yield,
failure breakdown, WAIT-patterns, and programmatic-vs-VLM yield (watch for an implausibly-high VLM
yield = garbage passing).

### 4. Stage, Upload Transport, And Download

```bash
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
READBACK_ROOT="$PWD/.data/huggingface-readback"

uv run python -m lite.data.hf.stage \
  --log-roots .data/rollout/lite.cuaworld/gpt/$COMMIT/*/train_annotated \
  --config-names <one desktop.use.<software> per --log-roots, same order> \
  --name Lite.CUAWorld \
  --repo-dir devs/data/lite.cuaworld

# Private smoke upload; use the release org only for the approved final publish.
: "${HF_ORG:?set HF_ORG to your Hub user/org for the private smoke repo}"
uv run python -m lite.data.hf.upload Lite.CUAWorld --org "$HF_ORG" --private --tag "$COMMIT"

uv run python -m lite.data.hf.download Lite.CUAWorld \
  --org "$HF_ORG" \
  --revision "$COMMIT" \
  --out "${READBACK_ROOT}/cua-lite/Lite.CUAWorld"
```

Record `stage`'s final `seen=... kept=... dropped_by_filter=...` line and the
per-config row lines as the publish gate. Upload/download are transport/layout
smokes only; row content was already gated by `stage`.

### 5. Export SFT Parquet

```bash
uv run python -m lite.train.export.export_sft \
  --config scripts/configs/qwen3_5/default/lite.cuaworld.yaml \
  --model-id Qwen/Qwen3.5-9B \
  --data-paths "${READBACK_ROOT}/cua-lite/Lite.CUAWorld" \
  --image-root "${READBACK_ROOT}" \
  --filter "lambda m: not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) > 0.5" \
  --num-proc 16 \
  -o .data/sft/qwen3_5/lite-cuaworld/train.parquet
```

`--config` is the **rollout** config, not an SFT-only recipe under
`scripts/configs/*/recipes/sft/`: `export_sft` re-renders every step through the
agent adapter, so exporting under a different history window or resolution than
the rollout used trains the model on prompts it will never see at inference.
(`lite.cuaworld.yaml` pins one software via `env_id`, but `export_sft` reads
only `agent_id` and `agent_kwargs` from the config — the
agent-side render settings are identical across every `lite.cuaworld.*` software.)

The processor/model ID must match training (tokenization + chat template are frozen at export).
Keep fail-fast on; use `--no-strict` only for an identified, recorded corrupt source row.

## Cost / time, resources & disk hygiene

Desktop is slow (~2–3 min/task). On a shared host, a 40-wide run spins up to 40 desktop containers
at once — **run a resource watchdog** and throttle `--concurrency` down if free RAM drops (each
container is memory-heavy; the host is co-tenant with other jobs). `blender3d` needs `gpus=1` (the
only GPU env); `gmat`'s upstream download is dead (rescue the retagged image). After **stage**
captures a batch, delete its raw `$SW/<split>` and `$SW/<split>_annotated` roots. Clean up only your
own env-server containers (named by your port) afterward. Task/verifier defects:
[/devs/envs/lite.cuaworld/UPSTREAM_ISSUES.md](/devs/envs/lite.cuaworld/UPSTREAM_ISSUES.md).
