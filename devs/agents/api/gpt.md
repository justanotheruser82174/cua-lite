See [AGENTS.md](/devs/agents/api/AGENTS.md) and [shared guidelines](/devs/agents/AGENTS.md).

# GPT Agent

**Goal:** Computer-use agent using OpenAI's GPT models via liteLLM.

**Supported environments:** desktop (`lite.demo`, `lite.osworld`, `osworld`) + browser (`webgym`) + mobile (`androidworld`)

**References** (priority order — follow the first two; the third is secondary, **if it conflicts with the first two, the first two win**):
- **official implementation:** `${CUA_LITE_REFERENCES_ROOT}/openai-cua-sample-app` (official quickstart)
- **official doc:** https://developers.openai.com/api/docs/guides/tools-computer-use
- **official doc (vision/limits):** https://developers.openai.com/api/docs/guides/images-vision#model-sizing-behavior
- **secondary / supplementary:** `${CUA_LITE_REFERENCES_ROOT}/cua/libs/python/agent/agent/loops/openai.py` (third-party; read for alternative framings only)

**Branch / Worktree:** `api` · `../cua-lite-api/`

**Agent slug:** `gpt`

**Files:**
- `lite/agents/models/gpt/action_space.py` — `GPTDesktopActionSpace`, `GPTMobileActionSpace`
- `lite/agents/models/gpt/agent.py` — `GPTDesktopUseAgent(key=r"gpt@(desktop|browser)@use")`, `GPTMobileUseAgent(key="gpt@mobile@use")`

**Launch:**
```bash
uv run python scripts/rollout.py --model-id gpt-5.5 --head 1
```

## Key Details

- **API:** `litellm.aresponses` (Responses API — NOT Chat Completions)
- **Resolution (adaptive coord-frame):** the model-frame size used for click coord normalization is **derived from the API itself**, not predicted client-side. Three steps per turn:
  1. **Optional client-side stretch** — `_resize_for_api(b64, self.resolution)`: if user set `agent_kwargs.resolution=(W, H)`, exact-stretch (non-AR-preserving) to W×H; if `None` (default), pass native screenshot through. No smart-fit, no patch-budget math.
  2. **API call** — `litellm.aresponses(...)`. The API may auto-resize the input per its documented limits (gpt-5.x: 2048 px/2500 patches for `high`; 6000 px/10000 patches for `original`) or endpoint-specific edge cases (some hosted routes occasionally cap `detail=original` at short-side 768; non-deterministic).
  3. **Adaptive coord-frame** — `_call_api_with_actual_dim(...)` GETs `/responses/{id}/input_items?include[]=message.input_image.image_url` and reads the actual processed image dims from the echoed PNG. Those dims drive `pixel_to_norm` for click coords. If sent ≠ processed, `logger.warning` fires but the trajectory continues — no `raise`.
  - Trajectory + logger keep the native PNG; only this step's b64 is resized client-side.
  - Falls back to sent dims if the GET fails (transient network).
- **Default rollout config dims:** Desktop 1920×1080 (osworld / lite.osworld); Mobile 1080×2400 (androidworld / androidlab). Grounding configs (osworld_g / screenspot_pro) omit `resolution` to send native screenshots (often 2560×1440+) for max pixel fidelity.
- **`detail`:** always `"original"` per OpenAI computer-use doc ("prefer `original`; avoid `high` or `low`"). Override via `api_kwargs={"detail": "..."}`.
- **Coordinate system:** GPT uses separate `x`, `y` fields; CUA-lite uses `[x, y]` arrays normalized to `[0, 1000]`
- **Native tool:** GA `{"type": "computer"}` (no display_* fields)
- **extra_tools:** Passed as Responses API function tools
- **Mobile `observation_text: "a11y:pixel"` / `"a11y:norm"`:** injects a11y tree (UI labels + centers, pixel or `[0, 1000]`) into the reset `LiteEnvObservation.text` and, per turn, into each post-action `LiteToolResult.text` (`role:"tool"` message)
