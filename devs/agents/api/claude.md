See [AGENTS.md](/devs/agents/api/AGENTS.md) and [shared guidelines](/devs/agents/AGENTS.md).

# Claude Agent

**Goal:** Computer-use agent using Anthropic's Claude models via liteLLM.

**Supported environments:** desktop (`lite.demo`, `lite.osworld`, `osworld`) + browser (`webgym`) + mobile (`androidworld`)

**References** (priority order — follow the first two; the third is secondary, **if it conflicts with the first two, the first two win**):
- **official implementation:** `${CUA_LITE_REFERENCES_ROOT}/claude-quickstarts/computer-use-demo` (official quickstart)
- **official doc:** https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool
- **official doc (vision/limits):** https://platform.claude.com/docs/en/build-with-claude/vision#evaluate-image-size
- **secondary / supplementary:** `${CUA_LITE_REFERENCES_ROOT}/cua/libs/python/agent/agent/loops/anthropic.py` (third-party; read for alternative framings only)

**Branch / Worktree:** `api` · `../cua-lite-api/`

**Agent slug:** `claude`

**Files:**
- `lite/agents/models/claude/action_space.py` — `ClaudeDesktopActionSpace`, `ClaudeMobileActionSpace`
- `lite/agents/models/claude/agent.py` — `ClaudeDesktopUseAgent(key=r"claude@(desktop|browser)@use")`, `ClaudeMobileUseAgent(key="claude@mobile@use")`

**Launch:**
```bash
uv run python scripts/rollout.py --model-id claude-opus-4-6 --head 1
uv run python scripts/rollout.py --model-id claude-sonnet-4-6 --head 1
```

## Key Details

- **API:** `litellm.acompletion` (Chat Completions)
- **Resolution:** controlled by env config to stay within API auto-downsampling limits (no agent-side downscale):
  - Desktop: 1024×768 (safe for all Claude; long=1024 < 1568)
  - Mobile: 720×1280 (safe for all Claude; long=1280 < 1568)
  - API limits: Opus 4.7 max 2576px long edge; older models max 1568px long edge, ~1.15 Mpx
- **Coordinate system:** Claude uses pixel `[x, y]` arrays; CUA-lite uses normalized `[0, 1000]`. Tool schema declares `display_width_px` / `display_height_px` = env resolution (1:1 with screenshot pixels)
- **Model version mapping:** `_get_tool_config_for_model()` selects tool version + beta flag:
  - Claude 4.6/4.5 → `computer_20251124`
  - Claude 4 → `computer_20250124`
  - Claude 3.7 → `computer_20250124`
  - Claude 3.5 → `computer_20241022`
- **extra_tools:** Passed as Anthropic function tools alongside the computer tool
- **Mobile `observation_text: "a11y:pixel"` / `"a11y:norm"`:** injects a11y tree (UI labels + centers, pixel or `[0, 1000]`) into the reset `LiteEnvObservation.text` and, per turn, into each post-action `LiteToolResult.text` (`role:"tool"` message)
