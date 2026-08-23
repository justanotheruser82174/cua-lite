# lite.scalecua Data Directory

This directory contains the lite.scalecua runtime catalog contract:
source locks, generated task catalogs, and committed validation fixtures.

Generated task catalogs live here at runtime and are guarded by
`catalog.lock.json`.

Bulky upstream snapshots, judge overlays, and reports remain in the repo-root
cache:

```text
.cache/lite.scalecua_tasks/
```

Expected generated files:

| File | Meaning |
| --- | --- |
| `train.jsonl` | `train` split imported from HF `osworld/generated_tasks` |
| `rl.jsonl` | `rl` split imported from HF `osworld/rl_tasks` |
| `catalog.lock.json` | row/hash/source identity lock for generated catalogs |
| `.cache/lite.scalecua_tasks/judge_functions/` | ScaleCUA generated/RL judge overlays |
| `.cache/lite.scalecua_tasks/.asset_identity` | installed task-source identity |
| `.cache/lite.scalecua_tasks/import_report.json` | counts, flags, exclusions, judge coverage, and validation evidence |

Exported parquet task lists are downstream artifacts. Recreate them from the
registry instead of committing them:

```bash
uv run python -m lite.train.export.export_tasks \
  --env-id lite.scalecua \
  --split train \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  -o /tmp/lite.scalecua.train.parquet
```

`lite.scalecua` has no env-local Dockerfile or image layer. It reuses the
normal `cua-lite/lite.osworld:latest` runtime unless a diagnostic run
explicitly overrides the image. Evaluation-set tasks should be run from
`lite.osworld`'s canonical `eval` split, not from this data directory.

## Oracle Fixtures

`oracle/` is reserved for curated drift sentinels over ScaleCUA `train` and
`rl` tasks. These files are not runtime catalogs.

Expected fixture files:

| File | Purpose |
| --- | --- |
| `oracle/rl.jsonl` | Canonical checked-in oracle fixtures for runnable `rl` tasks |
| `oracle/train.jsonl` | Canonical checked-in oracle fixtures for curated `train` tasks |

Fixture selection and replay are documented in
`devs/envs/lite.scalecua/validate/oracle/plan.md`.
