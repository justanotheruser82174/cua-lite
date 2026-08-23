See [AGENTS.md](/devs/agents/local/AGENTS.md) and [shared guidelines](/devs/agents/AGENTS.md).

# Qwen3-VL

**Goal:** Develop the Qwen3-VL workflow (covers both `Qwen/Qwen3-VL-8B-Thinking` and `Qwen/Qwen3-VL-8B-Instruct`). The Thinking variant produces a `<think>...</think>` reasoning block before the action; the Instruct variant does not.

**Supported environments:** `lite.demo`, `lite.osworld`, `osworld`, `webgym`, `android_env`, `androidworld`

**Reference implementation:**
- `${CUA_LITE_REFERENCES_ROOT}/OSWorld/scripts/python/run_multienv_qwen3vl.py`
- `${CUA_LITE_REFERENCES_ROOT}/OSWorld/mm_agents/qwen3vl_agent.py`

**Branch / Worktree:** `qwen3_vl` · `../cua-lite-qwen3_vl/`

**Agent slug:** `qwen3_vl`

**Launch:**
```bash
uv run python scripts/rollout.py --model-id Qwen/Qwen3-VL-8B-Thinking --head 1
uv run python scripts/rollout.py --model-id Qwen/Qwen3-VL-8B-Instruct --head 1
```

**Intentional deviations:** See [`/lite/agents/models/README.md`](/lite/agents/models/README.md) for the full list.
