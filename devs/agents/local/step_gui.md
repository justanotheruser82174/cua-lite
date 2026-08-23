See [AGENTS.md](/devs/agents/local/AGENTS.md) and [shared guidelines](/devs/agents/AGENTS.md).

# Step-GUI (GELab-Zero)

**Goal:** Develop the Step-GUI / **GELab-Zero-4B-preview** mobile navigation workflow (`stepfun-ai/GELab-Zero-4B-preview`). GELab-Zero is StepFun AI's 4B-parameter vision-language GUI agent model (see [arXiv:2512.15431](https://arxiv.org/abs/2512.15431)). It uses a **Chinese** task prompt with an inline action space description, emits chain-of-thought inside literal `<THINK>...</THINK>` tags (uppercase, plain BPE text), followed by a **tab-separated key:value** action string (NOT JSON function-calling). The model outputs a running `summary` field at each step, which serves as the compressed history for the next turn. **Mobile-only.**

**Supported environments:** `android_env`, `androidworld`

**Reference implementation:**
- `${CUA_LITE_REFERENCES_ROOT}/gelab-zero/copilot_tools/parser_0920_summary.py` 
- `${CUA_LITE_REFERENCES_ROOT}/gelab-zero/copilot_agent_server/local_server.py` 
- `${CUA_LITE_REFERENCES_ROOT}/gelab-zero/tools/ask_llm_v2.py` 
- `${CUA_LITE_REFERENCES_ROOT}/gelab-zero/mcp_server_config.yaml` 
- `${CUA_LITE_REFERENCES_ROOT}/gelab-zero/copilot_front_end/pu_frontend_executor.py` 

The reference repo ships the full inference stack (server + client + ADB frontend + parsers) but has **no AndroidWorld eval driver**. Env-loop concerns (post-action delay, screenshot timing) cross-check against `${CUA_LITE_REFERENCES_ROOT}/android_world/android_world/agents/m3a.py` instead.

**Branch / Worktree:** `step_gui` · `../cua-lite-step_gui/`

**Agent slug:** `step_gui`

**Launch:**
```bash
# Single trajectory (sample)
uv run python scripts/rollout.py --model-id stepfun-ai/GELab-Zero-4B-preview --env-id androidworld --head 1

# Multi-task rollout (matches upstream near-greedy decoding via the yaml config)
CUDA_VISIBLE_DEVICES=6,7 uv run python scripts/rollout.py \
  --model-id stepfun-ai/GELab-Zero-4B-preview --env-id androidworld \
  --config-path scripts/configs/step_gui/default/androidworld.yaml \
  --sample 64 --concurrency 8
```

**Intentional deviations:**

- **Coordinate space — pass-through, no rescale**: Step-GUI's prompt declares `[0, 1000]`, cua-lite normalized space is `[0, 1000]`. The coordinate systems are identical — no conversion is needed in either direction. Round-trip is byte-exact.
- **`<THINK>`, NOT `<think>` or `<thinking>`**: the literal 3-token string `<THINK>...</THINK>` (uppercase) is part of the SFT distribution. The parser (`str2action`, line 264) normalizes case and typos (`<TINK>`, `<think>`, spacing) before extraction — we replicate these tolerances. `<THINK>` text is stored as an `InlineReasoningContent` part inside `LiteMessage.content` (non-native CoT — per the LiteAssistantMessage type contract, only Qwen3-VL's native `<think>` uses the top-level `reasoning_content` slot), same as MAI-UI.
- **Tab-separated key:value format, NOT `<tool_call>` JSON**: Step-GUI's SFT distribution uses `explain:...\taction:CLICK\tpoint:x,y\tsummary:...` — tab-separated key:value pairs, not JSON function-calling. The adapter's `parse_raw_assistant_response` splits on `\t`, parses `key:value` pairs, and maps them to the `mobile_use` tool call dict. The reverse path (`format_agent_tool_call_as_wire_text`) serializes back to the same tab-separated format for history rendering. This is the most significant structural difference from MAI-UI (which uses `<tool_call>{json}</tool_call>`).
- **Content parts + `tool_calls` folded at render time**: parsed `<THINK>` becomes an `InlineReasoningContent` part, `explain:` becomes `ActionDescriptionContent`, and `summary:` becomes `HistorySummaryContent`, which is the canonical home for the rolling trajectory summary in current rows. `STEPGUIMobileBaseAdapter.convert_message_to_agent` folds reasoning + content parts + `tool_calls` back into a single `<THINK>...</THINK>\nexplain:...\taction:...\tsummary:...\t<fields>` text block just before `apply_chat_template` runs and drops the structured `tool_calls` field, so the agent class itself needs no override (`STEPGUIMobileAgent` body is `pass`).
- **No `tools=` to `apply_chat_template`**: Step-GUI was SFT'd with the action space inline in the Chinese system prompt (`task_define_prompt`), not via a `<tools>` schema block. We must NOT pass `tools=` to avoid triggering any chat template `{% if tools %}` branch that would inject off-distribution schema text.
- **Model-generated `summary` field as history**: the reference uses a fundamentally different history strategy from MAI-UI. Instead of keeping a window of recent user/assistant turns, the model emits a `summary` field at each step that compresses all prior actions into a single text. Only the latest `summary` is included in the next prompt (via `make_status_prompt`, not multi-turn history). In cua-lite, we implement this via a custom `StepGUIHistoryProtocol` that: (1) always sends a **single user message** (system prompt + task + summary history + current screenshot + format instructions), (2) extracts the `summary` from the previous assistant's `HistorySummaryContent` content part, and (3) includes only the **current** screenshot (no prior images). It is a single-current-prompt protocol like `Qwen3VLHistoryProtocol(full_history_size=1)`, but generates the summary from the model's own `summary` field rather than formatting cua-lite action descriptions externally.
- **Single user message, not multi-turn**: the reference sends exactly **one user message** per inference call (system prompt + status prompt + image), not a multi-turn conversation. All history context is inlined via the `summary` text in that single message. This matches `env2messages4ask` which always returns `[{"role": "user", "content": [...]}]`.
- **Image resize on (configurable)**: the reference optionally resizes screenshots to `728×728` JPEG (`mcp_server_config.yaml:17-20`). We replicate this via `image_preprocess` config in the adapter / sampling kwargs. Unlike MAI-UI (which sends raw images), Step-GUI was likely trained with resized images given the config default.
- **Near-greedy sampling by default**: the upstream defaults are `temperature=0.1`, `top_p=0.95`, `frequency_penalty=0.0`, `max_tokens=4096` (from `mcp_server_config.yaml:10-14`). These live in [`scripts/configs/step_gui/default/androidworld.yaml`](/scripts/configs/step_gui/default/androidworld.yaml) under `agent_kwargs.sampling_kwargs`. Unlike MAI-UI's strict greedy (`temperature=0.0`), Step-GUI uses near-greedy with nucleus sampling — expect slight non-determinism between rollouts.
- **Chinese system prompt (reference-shaped, with local INFO semantics)**: the `task_define_prompt` and `make_status_prompt` are in Chinese. We keep the prompt aligned with the reference distribution, except `INFO` is documented as a final answer because cua-lite maps it to terminal `response(text=...)`. Translating the rest of the prompt would shift the SFT distribution.
- **9-action enum, no `system_button` / `open` / `drag`**: the model's SFT'd action space is `CLICK`, `LONGPRESS`, `TYPE`, `SLIDE`, `AWAKE`, `WAIT`, `INFO`, `COMPLETE`, `ABORT`. Notably absent vs cua-lite: `system_button` (Back/Home/Enter/Menu — no HOT_KEY action), `drag` (SLIDE covers both swipe and drag semantics), `screenshot` (implicit). Step-GUI runtime configs must not expose `system_button`; cua-lite `open_app` maps to `AWAKE`; `terminate` splits into `COMPLETE` (success) / `ABORT` (failure). The reverse path maps all 9 actions back to cua-lite equivalents.
- **`explain` field passthrough**: the model emits an `explain:` field describing the action purpose. This is captured in `LiteMessage.content` (as text) for logging and human review but does not affect action execution. It is also included when folding back for prompt rendering.
- **INFO ↔ response mapping**: Step-GUI's `INFO` action is treated as a final answer in cua-lite. On the forward path, canonical `response(text=...)` renders as `INFO` with `value`; on the reverse path, `INFO` becomes terminal `response(text=...)`.

**Verification:** action-space round-trip tests are in `tests/agents/models/step_gui/test_step_gui_action_space.py`. Adapter parser + history protocol tests are in `tests/agents/models/step_gui/test_step_gui_adapter.py`. End-to-end live verification against a running emulator is via the sample/rollout commands above.
