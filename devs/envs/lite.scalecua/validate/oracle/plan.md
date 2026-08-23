# lite.scalecua Oracle Validation Plan

This directory is intentionally small. It mirrors the `lite.osworld` oracle
validation workflow, with ScaleCUA-specific fixture rows pointing at imported
`lite.scalecua` catalog tasks instead of embedding full task specs.

## Directory Contract

Keep only the core validation files here:

| File | Role |
| --- | --- |
| `validate.py` | Replays oracle fixtures in real `lite.scalecua` containers. |
| `verified_inventory.py` | Audits replay evidence from `validate.py` reports and screenshots. |
| `coverage_inventory.py` | Audits fixture coverage against the current runnable catalog. |
| `select_fixtures.py` | Generates candidate backlogs for coverage planning; candidates are not evidence. |
| `plan.md` | This operational contract. |
| `logs.md` | Append-only findings and replay evidence log. |

Do not put per-worker notes, one-off probes, candidate JSONL, screenshots, or
ad hoc inventories in this directory. Put generated artifacts under
`.exps/validate/lite.scalecua/oracle/`.

## North Star

The release target is the `rl` split. Every non-excluded RL task must end in
exactly one of these states:

1. It has a release-countable oracle fixture with strict replay evidence.
2. It has an exact `metadata.others.exclude_reason` explaining why it is not
   supported.

Fixture coverage is not replay evidence. A task only counts as oracle-covered
after `validate.py` proves both gates:

1. **No-op negative gate**: reset, do nothing, run production eval, reward must
   match `expected_pre_reward`, normally `0.0`.
2. **Oracle positive gate**: reset, replay `oracle_actions` or
   `oracle_trajectory`, run production eval, reward must match
   `expected_reward`, normally `1.0`.

The validator writes required screenshots and `result.json` next to each
fixture. `verified_inventory.py --require-artifacts` is the release evidence
gate.

## Fixture Source

Committed oracle fixtures live in split-level aggregate files:

```text
lite/gym/envs/lite/scalecua/data/oracle/rl.jsonl
lite/gym/envs/lite/scalecua/data/oracle/train.jsonl
```

Fixture rows must point to current non-excluded catalog rows:

```json
{
  "fixture_id": "oracle_rl_chrome_seed_01_0001",
  "split": "rl",
  "task_id": "scalecua_osworld_rl_chrome_...",
  "domain": "chrome",
  "expected_pre_reward": 0.0,
  "expected_reward": 1.0,
  "oracle_actions": [
    {"type": "execute", "parameters": {"command": "true"}}
  ]
}
```

Prefer source-backed rows under
`lite/gym/envs/lite/scalecua/src/gen/oracle/domains/`; the generator writes
those domain sources into the aggregate `rl.jsonl` / `train.jsonl` files.
Regenerate and run the generator check before counting a row:

```bash
uv run python -m lite.gym.envs.lite.scalecua.src.gen.oracle --check
```

## Authoring Rule

Start from the closest `lite.osworld` oracle recipe for the same domain and
setup/eval family. Reuse the operational pattern: process cleanup, profile or
document mutation, gold-file generation, app state normalization, postconfig
ordering, and evidence artifacts. Recompute all target values from the current
ScaleCUA evaluator; do not copy constants unless the evaluator goal signature
and target state match exactly.

If an oracle fails, classify before changing code:

| Failure class | Owner action |
| --- | --- |
| oracle recipe bug | Fix fixture/action and replay. |
| ScaleCUA setup/eval migration bug | Fix in `lite.gym.envs.lite.scalecua`. |
| OSWorld VM/container runtime parity gap | Fix in the shared `lite.osworld` image/runtime only when it is general OSWorld behavior. |
| upstream task/eval mismatch | Add exact `exclude_reason` and document in `UPSTREAM_ISSUES.md`. |
| unsupported task family | Add exact `exclude_reason`. |
| transient infra failure | Rerun with the same fixture after isolating timeout/network/container state. |

Do not loosen eval, pre-apply the target in setup/postconfig, or move a
failing fixture into release-countable paths just to improve coverage numbers.

## Replay Commands

Full RL replay:

```bash
uv run python devs/envs/lite.scalecua/validate/oracle/validate.py \
  --fixtures lite/gym/envs/lite/scalecua/data/oracle/rl.jsonl \
  --artifacts .exps/validate/lite.scalecua/oracle/<run-id> \
  --report .exps/validate/lite.scalecua/oracle/<run-id>.report.jsonl \
  --require-rl-flush-fired \
  --concurrency 16 \
  --reset-timeout 600 \
  --oracle-timeout 180
```

Target one fixture or task:

```bash
uv run python devs/envs/lite.scalecua/validate/oracle/validate.py \
  --fixtures lite/gym/envs/lite/scalecua/data/oracle/rl.jsonl \
  --filter <fixture_id_or_task_id_substring> \
  --artifacts .exps/validate/lite.scalecua/oracle/debug-<id> \
  --report .exps/validate/lite.scalecua/oracle/debug-<id>.report.jsonl \
  --concurrency 1
```

`validate.py` uses direct mode, derives a session id from `--artifacts` by
default, and sweeps only matching `lite.scalecua` containers at startup, signal
handling, and final cleanup. Do not run two validator processes with the same
`--session-id`.

## Inventory Commands

Coverage inventory for RL closure:

```bash
uv run python devs/envs/lite.scalecua/validate/oracle/coverage_inventory.py \
  --fixtures lite/gym/envs/lite/scalecua/data/oracle \
  --catalog-dir lite/gym/envs/lite/scalecua/data \
  --splits rl \
  --require-clean \
  --output .exps/validate/lite.scalecua/oracle/coverage/rl.coverage.json \
  --markdown .exps/validate/lite.scalecua/oracle/coverage/rl.coverage.md
```

Verified evidence inventory for RL closure:

```bash
uv run python devs/envs/lite.scalecua/validate/oracle/verified_inventory.py \
  --fixtures lite/gym/envs/lite/scalecua/data/oracle \
  --reports .exps/validate/lite.scalecua/oracle \
  --catalog-dir lite/gym/envs/lite/scalecua/data \
  --splits rl \
  --require-artifacts \
  --require-complete \
  --require-catalog-complete \
  --output .exps/validate/lite.scalecua/oracle/coverage/rl.verified.json \
  --markdown .exps/validate/lite.scalecua/oracle/coverage/rl.verified.md
```

Candidate backlog refresh, which is planning input only:

```bash
uv run python devs/envs/lite.scalecua/validate/oracle/select_fixtures.py \
  --coverage-target all-tasks \
  --splits rl \
  --output .exps/validate/lite.scalecua/oracle/candidates/rl_full.candidates.current.jsonl \
  --report .exps/validate/lite.scalecua/oracle/candidates/rl_full.coverage.current.json
```

## Promotion Gate

A fixture can count toward RL oracle closure only when all are true:

- the task is current, non-excluded, and in the intended split;
- `coverage_inventory.py --splits rl --require-clean` has zero fixture
  problems and zero duplicate fixture-task rows;
- `validate.py` has a passing report row with no-op reward matching
  `expected_pre_reward` and replay reward matching `expected_reward`;
- required artifacts exist: `result.json`, `00_noop_reset.png`,
  `01_noop_final.png`, `10_oracle_reset.png`, and
  `11_oracle_after_actions.png`;
- `verified_inventory.py --splits rl --require-artifacts` counts the fixture as
  verified;
- screenshots and debug payloads have been visually inspected when the task is
  newly introduced, failing, flaky, or suspicious.

Append every fix, exclusion, flaky rerun, and verified replay milestone to
`logs.md`.
