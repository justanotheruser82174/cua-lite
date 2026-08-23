# lite.scalecua Developer Guide

This env integrates ScaleCUA OSWorld tasks into CUA-Lite. Keep the scope on
adapter work: task catalog import, asset locking, strict setup/eval transport,
judge overlay resolution, env-server wiring, and validation evidence.

## Scope

- CUA-Lite owns `lite.gym.envs.lite.scalecua`, generated JSONL catalogs,
  ScaleCUA judge overlay loading, strict setup/postconfig dispatch, base image
  checks, registry export, env-server integration, and rollout validation.
- `lite.osworld` owns the desktop substrate and the official OSWorld eval
  exclusion map. Shared VM/container parity gaps that should benefit every
  `lite.osworld`-based env may be fixed in `lite.gym.envs.lite.osworld/**`,
  but must carry `lite.osworld` regression evidence. ScaleCUA-specific
  importer/setup/eval/judge/material compatibility stays in `lite.scalecua`.
- ScaleCUA upstream owns task JSON, judge functions, eval examples, and source
  provenance. Do not silently patch task semantics in the adapter.

## Hard Rules

- Runtime splits are exactly `train` and `rl`. Evaluation-set tasks should use
  `lite.osworld`'s canonical `eval` split.
- `metadata.others.domain` is canonical and must follow `lite.osworld`
  semantics. Use it for sampling, filters, registry queries, and reports.
  ScaleCUA `related_apps` / `snapshot` are audit context only.
- `metadata.others.exclude_reason` is the only default rollout/export filter
  key. Runnable rows omit the key; excluded rows use a non-empty string.
- HF generated/RL rows may use `exclude_reason="proxy_required"` when the
  task depends on a proxy unavailable in the local runtime.
- Do not add `lite.scalecua` to image freshness specs or create a Dockerfile in
  Phase 1/2. This env consumes the configured `lite.osworld` runtime image and
  uses the normal `cua-lite/lite.osworld:latest` lifecycle.

## Generated State

Generated runtime catalogs live under `lite/gym/envs/lite/scalecua/data/` and
are ignored by git. Bulky upstream snapshots and judge overlays live under
`.cache/lite.scalecua_tasks/**`. They must be recoverable from:

- `lite/gym/envs/lite/scalecua/data/assets.lock.yaml`
- pinned HF `extreme1228/ScaleCUA`
- `lite.osworld/data/eval.jsonl` for optional OSWorld-id domain alignment

Do not commit generated JSONL, HF snapshots, rollout logs, or judge staging
directories.

## Validation Standard

Before treating an implementation as ready, run:

- static catalog validation for train/RL counts, URL pinning, action allowlists,
  task-source identity, and judge coverage;
- unit tests for catalog loading, image contract, strict setup/eval, judge
  resolver, install/status, and registry export;
- visual smoke rollouts for `train`, `rl`, and env-server `rl`;
- a `gpt-5.5` batch gate with one clean 1000-task root plus one independent
  confirmation root: 500 `train` and 500 `rl`, 50 per
  `metadata.others.domain` per split. Every trajectory must be visually audited;
  final success rates are computed from visual labels, not raw `summary.json`.
  Overall, `train`, and `rl` visual success rates must each be >=70%, with no
  unresolved false-success or false-failure rows. Diagnose every mismatch using
  both rollout screenshots/actions and the corresponding SCALE-CUA task JSON,
  generated getter/metric, and OSWorld worker/evaluator code.

Record evidence in `validate/rollout/logs.md`. Record upstream data defects in
`UPSTREAM_ISSUES.md`; do not convert them into silent adapter patches. Record
container substrate gaps, migration adapter gaps, GUI persistence/flush risks,
asset materialization gaps, and unresolved probe queues in `gaps.md`.
