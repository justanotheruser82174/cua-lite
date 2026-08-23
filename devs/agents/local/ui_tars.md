See [AGENTS.md](/devs/agents/local/AGENTS.md) and [shared guidelines](/devs/agents/AGENTS.md).

# UI-TARS-7B-DPO

**Goal:** Develop the ByteDance-Seed/UI-TARS-7B-DPO workflow.

**Supported environments:** `lite.demo`, `lite.osworld`, `osworld`, `webgym`, `androidworld`

**Reference implementation:**
- `${CUA_LITE_REFERENCES_ROOT}/OSWorld/scripts/python/run_multienv_uitars.py`
- `${CUA_LITE_REFERENCES_ROOT}/OSWorld/mm_agents/uitars_agent.py`

**Branch / Worktree:** `ui_tars` · `../cua-lite-ui_tars/`

**Agent slug:** `ui_tars`

**Launch:** `uv run python scripts/rollout.py --model-id ByteDance-Seed/UI-TARS-7B-DPO --head 1`

