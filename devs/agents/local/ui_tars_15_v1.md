See [AGENTS.md](/devs/agents/local/AGENTS.md) and [shared guidelines](/devs/agents/AGENTS.md).

# UI-TARS-1.5-7B

**Goal:** Develop the ByteDance-Seed/UI-TARS-1.5-7B (v1 adapter) workflow.

**Supported environments:** `lite.demo`, `lite.osworld`, `osworld`, `webgym`, `android_env`, `androidworld`

**Reference implementation:**
- `${CUA_LITE_REFERENCES_ROOT}/OSWorld/scripts/python/run_multienv_uitars15_v1.py`
- `${CUA_LITE_REFERENCES_ROOT}/OSWorld/mm_agents/ui_tars_15_v1.py`

**Branch / Worktree:** `ui_tars_15_v1` · `../cua-lite-ui_tars_15_v1/`

**Agent slug:** `ui_tars_15_v1`

**Launch:** `uv run python scripts/rollout.py --model-id ByteDance-Seed/UI-TARS-1.5-7B --head 1`

