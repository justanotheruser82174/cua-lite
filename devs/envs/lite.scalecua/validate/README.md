# lite.scalecua Validation Plan

## Static Gates

- split counts and per-domain counts match `../scalecua.md`;
- only `train` and `rl` catalogs are present for `lite.scalecua`;
- runnable rows omit `exclude_reason`;
- no runnable row contains `ubuntu_osworld_file_cache/resolve/main`;
- all setup/postconfig actions are dicts with allowed types;
- representative judge functions/getters resolve through
  `lite.gym.envs.lite.scalecua.src.osworld.judges.resolve_metric` and
  `resolve_getter`;
- filtered parquet has only `problem` and `metadata` and contains no excluded
  rows.

Default export smoke must include the filter before sampling:

```bash
uv run python -m lite.train.export.export_tasks \
  --env-id lite.scalecua \
  --split rl \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  -o /tmp/lite.scalecua.rl.parquet
```

## Oracle Gates

Oracle validation is scoped to `rl` first, then `train` regression. It runs in
parallel with large rollout validation and is documented in `oracle/plan.md`.

Do not wait for rollout completion or perfect eval stability before writing
oracle fixtures. Oracle replay failures are triage inputs: classify each as a
fixture/action bug, setup transport bug, eval adapter bug, upstream task
mismatch, or unsupported task family, then fix, exclude, or document the
targeted follow-up.

Required tiers:

| Tier | Total fixtures | Purpose |
| --- | ---: | --- |
| rl_full | 1,839 current runnable RL tasks | north-star gate: every supported RL task has an oracle |
| smoke | 248 | thin drift smoke across all domains |
| recommended | 648 | secondary replay gate for common and high-risk train+rl setup/eval families |
| full sentinel | 2,008 | long-tail replay gate for major evaluator changes |

The smoke/recommended/full tier counts are replay budgets, not full matrix
coverage and not substitutes for `rl_full`. Full matrix candidate lists are
generated separately:

- `oracle/select_fixtures.py --coverage-target all-eval-setup` covers every
  current train/RL `split/domain/setup/eval` combination;
- `oracle/select_fixtures.py --coverage-target all-full` additionally
  distinguishes evaluator postconfig sets;
- `oracle/select_fixtures.py --coverage-target all-tasks --splits rl` emits
  the current RL full candidate backlog;
- `oracle/coverage_inventory.py --splits rl` audits committed fixtures against
  the current runnable RL catalog and reports remaining north-star gaps.

Current fixture files:

- `lite/gym/envs/lite/scalecua/data/oracle/rl.jsonl`: 1,839 RL rows.
- `lite/gym/envs/lite/scalecua/data/oracle/train.jsonl`: 217 train rows.

These aggregate files replace the earlier per-batch/per-domain JSONL shards.
Source-backed provenance now lives in `src/gen/oracle/domains/<domain>.py`.
Legacy batch names remain only in row `fixture_id` / `source` fields so replay
evidence stays traceable without keeping dozens of batch source files.

Current RL inventory status:

- `coverage_inventory.py --splits rl --require-clean`: 1,839 / 1,839 runnable
  RL fixture rows, 0 fixture problems, 0 duplicate fixture tasks.
- `verified_inventory.py --splits rl --require-artifacts --require-complete`:
  1,839 / 1,839 fixture rows verified, 1,839 / 1,839 runnable RL catalog rows
  verified, 0 artifact-missing strict pass rows.

Workflow:

- audit current committed coverage with `oracle/coverage_inventory.py` and
  `oracle/coverage_inventory.py --splits rl`;
- generate candidate coverage with `oracle/select_fixtures.py`;
- use `--coverage-target all-eval-setup` as the complete oracle-action backlog
  and `--coverage-target all-full` for postconfig-sensitive changes;
- start converting candidates into executable `oracle_actions` or
  `oracle_trajectory` immediately; this is a parallel workstream, not a
  post-rollout cleanup task;
- commit curated fixtures under `lite/gym/envs/lite/scalecua/data/oracle/`;
- run `oracle/validate.py` for both negative no-op checks and positive replay
  checks. The current validator is a
  direct-mode tool with session-scoped cleanup, so use it for small smoke/debug
  runs, isolated shard replay, or one bounded-concurrency process at a time. For
  recommended/full sweeps, either keep a single bounded validator with orphan
  checks before/after the run, or first add an env-server-backed oracle runner;
- inspect oracle screenshots for reward/screen mismatch and append findings to
  `oracle/logs.md`.

## Rollout Gates

Smoke:

```bash
uv run --no-sync bash lite/gym/envs/lite/scalecua/scripts/install.sh

uv run python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id lite.scalecua \
  --splits rl \
  --head 1 \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --config-path scripts/configs/gpt/default/lite.scalecua.yaml
```

Batch gate:

- generate prompt-data with
  `devs/envs/lite.scalecua/validate/rollout/make_batch_prompt_data.py`;
  the generator clears `CUA_LITE_ENV_SERVER_URL` / `CUA_LITE_ENV_SERVER_TOKEN`
  internally so sampling always reads the local current catalog;
- validate prompt-data against the current catalog with
  `devs/envs/lite.scalecua/validate/rollout/check_prompt_data.py`; rerun this
  after every importer/filter change before treating an older prompt parquet as
  clean accounting input;
- sample exactly 1000 non-excluded trajectories: 500 `train` and 500 `rl`,
  with 50 tasks per OSWorld domain per split;
- start a fresh dedicated `scripts/serve_env.py` for `lite.scalecua` and run
  `scripts/rollout.py --model-id gpt-5.5 --env-id lite.scalecua` through
  `CUA_LITE_ENV_SERVER_URL`/`CUA_LITE_ENV_SERVER_TOKEN` with
  `--prompt-data .exps/validate/lite.scalecua/batch/gpt-5.5-1000.prompt.parquet`
  and rollout `--concurrency 8`;
- stop that env-server after rollout and verify there are no
  `lite-env-<batch-port>-*` containers left behind;
- visually audit every trajectory into
  `.exps/validate/lite.scalecua/batch/gpt-5.5-1000.visual_audit.jsonl`;
- start visual audit while rollout is running, prioritizing reported reward-1
  rows for false-success checks, continuously sampling reported reward-0 rows
  for false-failure/setup/action/transient checks, and separately auditing
  partial rewards as partial-credit/evaluator-mismatch cases;
- compute success from visual labels, not raw `summary.json`; every mismatch
  must be judged against screenshots plus the relevant SCALE-CUA task,
  generated getter/metric, and OSWorld worker/evaluator code;
- require >=70% visual success overall, >=70% `train`, >=70% `rl`, 0
  unresolved `false_success`, and 0 unresolved `false_failure`.

GPT-5.5 is expected to land around the 70-80% range on supported
OSWorld-style desktop tasks. If the visual success rate drops below 70%, treat
that as a migration/eval/task-quality alarm first: audit representative
reward-0 rows visually, compare the official SCALE-CUA getter/metric and task
JSON, run targeted persisted-state probes, and add oracle/no-op coverage before
classifying the drop as model failure.

Write commands, commit, asset identity, sample IDs, success table, failure
taxonomy, and visual notes to `rollout/logs.md`.
