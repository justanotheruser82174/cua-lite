See [AGENTS.md](/devs/agents/local/AGENTS.md) and [shared guidelines](/devs/agents/AGENTS.md).

# Qwen3.5-VL

**Goal:** Develop the Qwen3.5-VL workflow (covers `Qwen/Qwen3.5-VL-*-Instruct`). The VL variant ships the vision adapter plus the `<tool_call>` / XML action wire format used by the OSWorld reference agent.

**Supported environments:** `lite.demo`, `lite.osworld`, `osworld`, `webgym`, `android_env`, `androidworld`

**Reference implementation:**
- `${CUA_LITE_REFERENCES_ROOT}/OSWorld/scripts/python/run_multienv_qwen35vl.py`
- `${CUA_LITE_REFERENCES_ROOT}/OSWorld/mm_agents/qwen35vl_agent.py`
- Upstream PR: <https://github.com/xlang-ai/OSWorld/pull/448>

**Branch / Worktree:** `qwen3_5` · `../cua-lite-qwen3_5/`

**Agent slug:** `qwen3_5`

**Launch:**
```bash
uv run python scripts/rollout.py --model-id Qwen/Qwen3.5-VL-8B-Instruct --head 1
```

**Intentional deviations:** See [`/lite/agents/models/README.md`](/lite/agents/models/README.md) for the full list.
