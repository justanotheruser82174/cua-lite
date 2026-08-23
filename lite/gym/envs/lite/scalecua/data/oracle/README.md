# lite.scalecua Oracle Fixtures

This directory holds curated oracle validation fixtures for ScaleCUA `rl` and
`train` tasks. It does not contain runtime task catalogs.

The published fixture files are split-level aggregates, matching the OSWorld
pattern of keeping generated data as a small set of canonical JSONL files:

| File | Meaning |
| --- | --- |
| `rl.jsonl` | Release-critical oracle fixtures for non-excluded `rl_tasks`. |
| `train.jsonl` | Secondary regression fixtures for generated/train tasks. |

Do not add per-worker, per-batch, or per-domain JSONL files here. Keep promoted
fixture recipes as registered Python source under `src/gen/oracle/domains/`;
keep diagnostic artifacts under `.exps/validate/lite.scalecua/oracle/`.

Source-backed rows are generated into these aggregate files:

```bash
uv run python -m lite.gym.envs.lite.scalecua.src.gen.oracle --check
uv run python -m lite.gym.envs.lite.scalecua.src.gen.oracle
```

Each JSONL row points to an imported task:

```json
{
  "fixture_id": "oracle_rl_chrome_seed_01_0001",
  "split": "rl",
  "task_id": "scalecua_osworld_rl_chrome_...",
  "domain": "chrome",
  "coverage": {
    "setup": ["launch", "chrome_open_tabs"],
    "postconfig": ["launch", "sleep"],
    "result": ["active_url_from_accessTree"],
    "expected": ["rule"],
    "func": ["is_expected_url_pattern_match"],
    "combo": [["active_url_from_accessTree", "rule", "is_expected_url_pattern_match"]]
  },
  "expected_pre_reward": 0.0,
  "expected_reward": 1.0,
  "oracle_actions": [],
  "oracle_trajectory": null,
  "source": "rl_auto_chrome_b2"
}
```

Exactly one of `oracle_actions` or `oracle_trajectory` must be populated.
Source-backed rows are grouped by domain module. Legacy batch names remain in
`fixture_id` / `source` fields for replay provenance; they are not source file
names.
Selection, replay, inventory, and promotion gates live in
`devs/envs/lite.scalecua/validate/oracle/plan.md`.
