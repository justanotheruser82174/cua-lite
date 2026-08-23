# lite.scalecua Rollout Plan

This plan intentionally mirrors the visual audit discipline in
`devs/envs/lite.osworld/validate/rollout/plan.md`. A rollout process exiting 0
is only a transport signal; it is not proof that the task is correct.

1. Run static validation and `install.sh status`.
2. Run one direct `rl` smoke rollout with explicit `--filter`.
3. Run one direct `train` smoke rollout with explicit `--filter`.
4. Run one direct `eval` smoke rollout with explicit `--filter`.
5. Start a new env-server and run one `rl` smoke rollout through it.
6. Inspect screenshots, actions, annotated screenshots, and results.
7. Run the 1000-task `gpt-5.5` batch gate through env-server.

The coordinator owns persistent repo edits, batch manifests, audit sidecars,
rerun roots, and final gate decisions. Subagents may inspect completed task
logs, run probes, and draft audit rows in parallel, but they return findings to
the coordinator instead of mutating shared files directly.

Env-server smoke must use a new server port and should stop the server after the
run. A rollout only counts as smoke success when reset screenshot and first step
artifact show the task actually entered the desktop environment.

## Required Visual Inspection

For every smoke rollout and every batch trajectory, open these artifacts in the
task log directory:

- `sample_00/images/*.png`: inspect canonical trajectory images referenced from `trajectory.parquet`.
- `turn_0000/prompt_images/0000_reset.png`: when debug artifacts are present, verify setup state, active app/window, and desktop health.
- `prompt_images/*.png`: inspect every image-bearing debug turn's prompt-image cache when present.
- `01_prompt.txt`: confirm the rendered model prompt matches the task and visible state.
- `02_response.txt`: inspect the raw model output for parse failures or unsupported tool use.
- `03_actions.json`: verify model action -> lite action -> executed action fields.
- `prompt_images_annotated/*.png`: verify click targets, scroll direction/amount,
  key chords, typed text, and multi-action parsing against the matching
  `prompt_images/<name>.png`.
- `04_results.json`: verify reward/termination/error matches the visual outcome.
- `05_timing.json`: when present, check unusually long predict/action phases before
  attributing a visual mismatch to the environment.

Mark the rollout as failed when any of these are wrong, even if
`summary.json` exists or the command exited 0. Record task id, split, log root,
visual notes, action-chain notes, reward, and issue classification in
`logs.md`.

## 1000-task Batch Gate

Generate deterministic stratified prompt-data:

```bash
uv run python devs/envs/lite.scalecua/validate/rollout/make_batch_prompt_data.py \
  --per-domain 50 \
  --seed 20260714 \
  --output .exps/validate/lite.scalecua/batch/gpt-5.5-1000-instructionfilter.prompt.parquet \
  --manifest .exps/validate/lite.scalecua/batch/gpt-5.5-1000-instructionfilter.manifest.json
```

This yields 1000 runnable tasks: 500 `train` and 500 `rl`, with 50 tasks per
`metadata.others.domain` per split. This is the same domain field used by
`lite.osworld`; ScaleCUA `related_apps` can be recorded as audit context but
must not drive sampling. Do not pass `--filter` to rollout when using this
`--prompt-data`; the generator has already filtered `exclude_reason`. The
generator clears `CUA_LITE_ENV_SERVER_URL` / `CUA_LITE_ENV_SERVER_TOKEN`
internally before importing `lite.gym`, so it samples from the local current
catalog even if the shell has a remote env-server configured.

Before launching a clean batch, validate the prompt against the current catalog:

```bash
uv run python devs/envs/lite.scalecua/validate/rollout/check_prompt_data.py \
  --prompt-data .exps/validate/lite.scalecua/batch/gpt-5.5-1000-instructionfilter.prompt.parquet
```

This check must be rerun after every importer/filter change. If it reports any
current `exclude_reason` hits, regenerate prompt-data and use a fresh env-server
for clean accounting; the older rollout root is diagnostic only.

Run the batch through a fresh dedicated env-server. Do not run the 1000-task
batch in direct mode, and do not accept the batch until the visual audit sidecar
and `vision_gate.py validate` checks below pass.
ScaleCUA uses the normal `cua-lite/lite.osworld:latest` runtime. Start a fresh
dedicated env-server for batch validation and do not pass fixed-tag
`--env-kwargs`.

The current clean acceptance root is
`.exps/validate/lite.scalecua/batch/gpt-5.5-1000-latest-20260716`. Do not reuse
the stopped diagnostic roots
`.exps/validate/lite.scalecua/batch/gpt-5.5-1000-fixedbase-20260715`,
`.exps/validate/lite.scalecua/batch/gpt-5.5-1000-profileflush-recreationgridfix`
or
`.exps/validate/lite.scalecua/batch/gpt-5.5-1000-profileflush-recreation-atfallback`
or
`.exps/validate/lite.scalecua/batch/gpt-5.5-1000-compatfix`
or
`.exps/validate/lite.scalecua/batch/gpt-5.5-1000-profileblur-compatfix`
or
`.exps/validate/lite.scalecua/batch/gpt-5.5-1000-gimpwindow-keyplus-compatfix`.

```bash
HOST_IP=$(hostname -I | awk '{print $1}')

env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
  uv run python scripts/serve_env.py \
  --host 0.0.0.0 \
  --port 30321 \
  --env-ids lite.scalecua \
  --token lite-scalecua-1000-latest-20260716 \
  --idle-ttl-sec 900 \
```

Leave `--max-live-envs` unset for normal batches so the env-server derives its
admission cap from host capacity. Use `--max-live-envs <N>` only as an explicit
advanced override for a constrained-cap repro, and record the reason with the
artifact root.

In a separate shell:

```bash
HOST_IP=$(hostname -I | awk '{print $1}')

CUA_LITE_ENV_SERVER_URL=http://${HOST_IP}:30321 \
CUA_LITE_ENV_SERVER_TOKEN=lite-scalecua-1000-latest-20260716 \
  uv run python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id lite.scalecua \
  --prompt-data .exps/validate/lite.scalecua/batch/gpt-5.5-1000-instructionfilter.prompt.parquet \
  --concurrency 8 \
  --max-attempts 1 \
  --config-path scripts/configs/gpt/default/lite.scalecua.yaml \
  --save-data true --save-video false --save-gif false \
  --log-root .exps/validate/lite.scalecua/batch/gpt-5.5-1000-latest-20260716
```

The acceptance root uses `--max-attempts 1`. If a trajectory is rerun for a
typed transient reason, save that rerun under a separate targeted root and keep
the original row in the 1000-task visual audit.

After rollout, stop the dedicated env-server and check for leftovers:

```bash
docker ps -a --format '{{.ID}} {{.Names}} {{.Image}} {{.Status}}' \
  | rg 'lite-env-30321-' || true
```

Remove only leftover `lite-env-30321-*` containers from this batch. Do not touch
unrelated env-server or direct-mode containers.

Every trajectory must receive a visual audit row in:

```text
.exps/validate/lite.scalecua/batch/gpt-5.5-1000-latest-20260716.visual_audit.jsonl
```

Use the fail-closed queue/report helper to make that review auditable:

```bash
uv run python devs/envs/lite.scalecua/validate/rollout/vision_gate.py build-queue \
  --log-root .exps/validate/lite.scalecua/batch/gpt-5.5-1000-latest-20260716 \
  --queue .exps/validate/lite.scalecua/batch/gpt-5.5-1000-latest-20260716.vision_queue.jsonl

uv run python devs/envs/lite.scalecua/validate/rollout/vision_gate.py validate \
  --queue .exps/validate/lite.scalecua/batch/gpt-5.5-1000-latest-20260716.vision_queue.jsonl \
  --sidecar .exps/validate/lite.scalecua/batch/gpt-5.5-1000-latest-20260716.visual_audit.jsonl \
  --report .exps/validate/lite.scalecua/batch/gpt-5.5-1000-latest-20260716.vision_gate.report.json \
  --expected-total 1000 \
  --min-success-rate 0.70
```

Start visual review while rollout is still running. Whenever a task log
directory has terminal results, add or update its audit row. Maintain three live
queues:

- reported `reward=1`: check first for `false_success`.
- reported `reward=0`: continuously inspect across domains for
  `false_failure`, setup bugs, action bugs, and transients.
- reported `0 < reward < 1`: inspect as partial-credit cases; classify whether
  the partial score reflects genuinely partial completion or an evaluator
  mismatch.

Do not wait for all 1000 trajectories to finish before opening screenshots.

Required fields per row: `task_id`, `split`, `domain`, `log_dir`,
`reported_reward`, `reported_terminated`, `reported_truncated`, `setup_ok`,
`action_ok`, `eval_ok`, `visual_label`, `visual_success`, `notes`, `reviewer`,
`checked_at`.

`visual_success` is true when the final state visually satisfies the task,
including `true_success` and `false_failure`; it is false for partial, blocked,
transient, setup/action failures, and hidden states that cannot be visually
decided.

`visual_label` values:

- `true_success`: reward 1 and screenshot/action/end state satisfy the task.
- `true_failure`: reward 0 and the visual end state does not satisfy the task.
- `partial_success`: partial reward and visual/action evidence shows some but
  not all task criteria were satisfied.
- `false_success`: reward 1 but the visual end state is wrong.
- `false_failure`: reward 0 or partial reward but the visual end state is fully
  correct.
- `setup_failure`: reset/setup did not create the intended task state.
- `action_parse_failure`: model action was parsed/executed incorrectly by
  CUA-Lite.
- `transient_failure`: API/Docker/desktop/network transient; rerun before
  counting.
- `blocked_upstream`: task should be filtered by `exclude_reason`.
- `not_visually_decidable`: screenshot cannot prove hidden state such as Chrome
  Preferences, bookmarks, extension settings, VLC `vlcrc`, or file content.
- `ambiguous_needs_evaluator_probe`: visual evidence appears to disagree with
  reward, but official getter/metric semantics depend on persisted state; run a
  targeted file/getter probe before classifying it as a false success/failure.

The gate uses visual labels, not raw `summary.json`:

- 1000/1000 trajectories audited.
- 0 unresolved `transient_failure`.
- 0 `false_success` and 0 `false_failure`.
- overall visual success rate >= 70%.
- `train` visual success rate >= 70%.
- `rl` visual success rate >= 70%.
- every domain has 100 audited trajectories and a recorded per-domain success
  rate.

## Failure Classes

- `setup_miss`: screenshot shows wrong app/state, missing file, auth page, dead
  desktop, or unnormalized setup action.
- `action_mismatch`: annotated action does not match intended coordinate,
  scroll, key, text, or multi-action semantics.
- `eval_mismatch`: visual end state and `04_results.json` disagree.
- `blocked_upstream`: task requires unsupported auth/proxy/asset not captured by
  `exclude_reason`.
- `transient`: API quota, Docker capacity, or desktop boot failure that reruns
  cleanly on the same task.
