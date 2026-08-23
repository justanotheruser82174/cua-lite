See [AGENTS.md](/devs/agents/local/AGENTS.md) and [shared guidelines](/devs/agents/AGENTS.md).

# MAI-UI

**Goal:** Develop the Tongyi-MAI MAI-UI navigation workflow (covers both `Tongyi-MAI/MAI-UI-2B` and `Tongyi-MAI/MAI-UI-8B`). MAI-UI is a Qwen3-VL fine-tune from Alibaba's Tongyi MAI team that emits prompted-CoT in literal `<thinking>...</thinking>` tags (NOT the Qwen3 reasoning channel `<think>` token) followed by a single `<tool_call>{"name":"mobile_use", ...}</tool_call>` block. Mobile-only.

**Supported environments:** `android_env`, `androidworld`

**Reference implementation:**
- `${CUA_LITE_REFERENCES_ROOT}/MAI-UI/src/mai_naivigation_agent.py`
- `${CUA_LITE_REFERENCES_ROOT}/MAI-UI/src/prompt.py`
- `${CUA_LITE_REFERENCES_ROOT}/MAI-UI/cookbook/run_agent.ipynb`

The reference repo only ships the prediction wrapper (`mai_naivigation_agent.py`) — there is no AndroidWorld eval driver. Env-loop concerns (post-action wait, `hide_automation_ui`, screenshot capture timing) cross-check against `${CUA_LITE_REFERENCES_ROOT}/android_world/android_world/agents/m3a.py` instead.

**Branch / Worktree:** `mai_ui` · `../cua-lite-mai_ui/`

**Agent slug:** `mai_ui`

**Launch:**
```bash
# Single trajectory (sample)
uv run python scripts/rollout.py --model-id Tongyi-MAI/MAI-UI-2B --env-id androidworld --head 1
uv run python scripts/rollout.py --model-id Tongyi-MAI/MAI-UI-8B --env-id androidworld --head 1

# Multi-task rollout (matches upstream greedy decoding via the yaml config)
CUDA_VISIBLE_DEVICES=6,7 uv run python scripts/rollout.py \
  --model-id Tongyi-MAI/MAI-UI-8B --env-id androidworld \
  --config-path scripts/configs/mai_ui/default/androidworld.yaml \
  --sample 64 --concurrency 8
```

**Intentional deviations:**
- **Coordinate rescale 999 ↔ 1000**: MAI-UI's training space is `[0, 999]` (`SCALE_FACTOR=999`); cua-lite normalized space is `[0, 1000]`. The action space rescales both directions; round-trip drift ≤ 1 unit per axis.
- **Swipe = direction-only forward, anchor synthesis on reverse**: MAI-UI's `swipe` schema is `{direction, coordinate(optional anchor)}` — no precise start/end. cua-lite `swipe(start, end)` translates to MAI-UI `swipe(direction=<dominant>, coordinate=start)` (lossy). Reverse path synthesizes a fixed-offset endpoint from the anchor + direction. Use `drag` for precise start/end gestures.
- **`<thinking>`, NOT `<think>`**: the literal 3-BPE-token string `<thinking>...</thinking>` is part of the SFT distribution. The Qwen3 special tokens `<think>` (id 151667) / `</think>` (id 151668) are deliberately NOT used. The `ing` suffix matters.
- **Inline reasoning + `tool_calls` folded at render time**: parsed `<thinking>` content lives in an `InlineReasoningContent` part inside `LiteMessage.content` (non-native CoT — per the LiteAssistantMessage type contract, only Qwen3-VL's native `<think>` uses the top-level `reasoning_content` slot). `MAIUIMobileUseAdapter.convert_message_to_agent` folds inline reasoning (read via `get_inline_reasoning` — strict inline-only, since MAI-UI's SFT distribution has no native thinking slot) and `tool_calls` into a single `<thinking>...</thinking>\n<tool_call>{compact_json}</tool_call>` content text block right before `apply_chat_template` runs, then drops the structured `tool_calls` field from the rendered copy. The agent class itself needs no override (`MAIUIMobileAgent` body is `pass`). This fold is required because the Qwen2-VL chat_template's tool-call branch uses `tojson` (with spaces) while upstream `mem2response` uses `json.dumps(separators=(",", ":"))` (no spaces) — the byte-level mismatch would shift the SFT distribution.
- **No `tools=` to `apply_chat_template`**: the chat_template's `{% if tools %}` branch would inject a Qwen-style `<tools>{schema}</tools>` block that MAI-UI never saw during SFT. The action space lives inline in `MAI_MOBILE_SYS_PROMPT` as JSON examples instead.
- **No image resize**: the navigation pipeline sends raw 1080×2400 (or whatever the env returns) screenshots, matching `mai_naivigation_agent.py:_prepare_images` (which only does `Image.open + .convert("RGB")`). The grounding eval (`evaluation/grounding/eval_local.py`, `eval_server.py`) DOES call `qwen_vl_utils.smart_resize` with `max_pixels=2.1M-6.5M` — but that's a separate code path with a different system prompt and a different coordinate convention (resized-pixel space, not absolute `[0, 999]`). We only port the navigation path here; grounding is deferred to v2.
- **Greedy decoding by default**: the upstream `runtime_conf` defaults are `temperature=0.0, top_p=1.0, top_k=-1, max_tokens=2048, repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0, seed=42`. These live in [`scripts/configs/mai_ui/default/androidworld.yaml`](/scripts/configs/mai_ui/default/androidworld.yaml) under `agent_kwargs.sampling_kwargs` (mirroring the `api_kwargs` convention used by API agents). The yaml-driven path avoids hard-coding sampling defaults in `LOCAL_AGENTS`.
- **MCP / `ask_user` / `double_click` deferred to v2**: MAI-UI's reference exposes an MCP-augmented mode via a separate Jinja `Template` (`MAI_MOBILE_SYS_PROMPT_ASK_USER_MCP`) that adds `ask_user` + `double_click` actions and a `## MCP Tools` block. v1 ports the base navigation prompt only. The adapter already has a resolved `metadata.extra_tool_schemas` render path (with `env_kwargs.extra_tools` remaining only the env-side name selector), so future MCP tools can be added without restructuring the adapter.
- **History protocol**: a thin `MAIUIHistoryProtocol(UITarsHistoryProtocol, full_history_size=3)` that forces the windowing-style `_apply_windowing` even when `num_turns ≤ full_history_size`, to match MAI-UI's reference layout (text-only first user, then assistant-only old turns, then user(image-only) + assistant pairs in the window) byte-for-byte.

**Verification:** end-to-end byte-exact match against the reference's `mem2response` output and full chat_template render for `n_history ∈ {0, 1, 2, 5}` is in `tests/agents/models/mai_ui/test_mai_ui_adapter.py` and a one-off script at `/tmp/mai_ui_verify.py`.
