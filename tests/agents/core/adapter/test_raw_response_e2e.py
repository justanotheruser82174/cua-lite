"""End-to-end byte-fidelity tests for the ``raw_response`` short-circuit.

These tests load the actual HuggingFace processor + chat_template for
Qwen3-VL (Instruct and Thinking) and MAI-UI, feed a fabricated raw response
through the full pipeline

    parse_raw_assistant_response
      → convert_message_from_agent (produces lite_message)
      → attach raw_response sidecar (with matching adapter_key)
      → convert_message_to_agent  (base-class short-circuit fires)
      → processor.apply_chat_template (renders past assistant turn)

and assert that the rendered past-turn bytes equal what the model actually
produced (modulo chat_template role tokens / native thinking stripping for
Qwen3-VL-Thinking).

Tests skip cleanly when the model isn't cached in
``~/.cache/huggingface/hub`` (CI-friendly; full set requires local
checkpoints).

Run: uv run pytest tests/agents/core/adapter/test_raw_response_e2e.py -v
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest

from lite.agents.bootstrap import register_all

register_all()


_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def _model_cached(model_id: str) -> bool:
    """True iff the HF repo is cached locally (snapshots dir has at least one commit)."""
    repo_dir = _HF_CACHE / f"models--{model_id.replace('/', '--')}"
    if not repo_dir.is_dir():
        return False
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(d.is_dir() for d in snapshots.iterdir())


@functools.lru_cache
def _load_processor(model_id: str):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


def _build_trajectory_messages(
    adapter,
    *,
    user_text: str,
    raw_response: str,
    followup_user_text: str = "next observation",
):
    """Produce a 3-message trajectory [user, assistant_raw, user_followup].

    The assistant turn carries a matching ``raw_response`` sidecar so
    ``convert_message_to_agent`` short-circuits and returns the raw text
    verbatim (ready for the chat_template).
    """
    agent_message = adapter.parse_raw_assistant_response(raw_response)
    lite_assistant = adapter.convert_message_from_agent(agent_message)
    lite_assistant["raw_response"] = {
        "text": raw_response,
        "adapter_key": adapter._registry_key,
    }

    messages = [
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
        lite_assistant,
        {"role": "user", "content": [{"type": "text", "text": followup_user_text}]},
    ]
    # Run through the adapter's public convert_message_to_agent to exercise
    # the short-circuit. We want final chat_template input.
    return [adapter.convert_message_to_agent(m) for m in messages]


# =============================================================================
# Qwen3-VL-Thinking: past-turn <think> must be stripped by template itself
# =============================================================================

@pytest.mark.skipif(
    not _model_cached("Qwen/Qwen3-VL-8B-Thinking"),
    reason="Qwen3-VL-8B-Thinking weights not cached locally",
)
def test_qwen3vl_thinking_raw_renders_verbatim_past_turn():
    """Qwen3-VL-Thinking chat_template strips `<think>` on past turns.
    With our short-circuit emitting ``{role, content=[text: raw]}`` and no
    ``reasoning_content`` field, the template's auto-split logic extracts
    the ``<think>...</think>`` block and drops it for past turns. The raw
    ``<tool_call>{...}</tool_call>`` block stays in content verbatim."""
    from lite.agents.models.qwen3_vl.adapter import Qwen3VLDesktopUseAdapter

    adapter = Qwen3VLDesktopUseAdapter()
    processor = _load_processor("Qwen/Qwen3-VL-8B-Thinking")

    # Non-canonical JSON whitespace — deliberate drift from `json.dumps` default
    raw = (
        "<think>\nplanning the click\n</think>\n"
        "Action: click the icon\n"
        "<tool_call>\n{\"name\" :\"computer_use\",   "
        "\"arguments\":{\"action\":\"left_click\",\"coordinate\":[500,300]}}\n</tool_call>"
    )

    messages = _build_trajectory_messages(
        adapter,
        user_text="Please click the icon at 500,300.",
        raw_response=raw,
        followup_user_text="Now done?",
    )

    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # The past-turn assistant block (between the first `<|im_start|>assistant\n`
    # and the next `<|im_end|>`) should contain the JSON verbatim — same
    # non-canonical whitespace.
    lines = rendered.split("<|im_start|>")
    assistant_block = next(
        (ln for ln in lines if ln.startswith("assistant\n")), None
    )
    assert assistant_block is not None, f"No assistant block in render: {rendered!r}"
    # tool_call JSON with our non-canonical spacing survives:
    assert '"name" :"computer_use"' in assistant_block
    assert '"coordinate":[500,300]' in assistant_block
    # <think> is stripped on past turns (template enforces this):
    assert "<think>\nplanning the click\n</think>" not in assistant_block, (
        f"past-turn <think> block leaked into past context: {assistant_block!r}"
    )


# =============================================================================
# Qwen3-VL-Instruct: no <think> should be introduced spuriously
# =============================================================================

@pytest.mark.skipif(
    not _model_cached("Qwen/Qwen3-VL-4B-Instruct"),
    reason="Qwen3-VL-4B-Instruct weights not cached locally",
)
def test_qwen3vl_instruct_no_spurious_think_injected():
    """Instruct variant has no reasoning channel. Short-circuit should not
    cause any `<think>` tag to appear in the rendered prompt."""
    from lite.agents.models.qwen3_vl.adapter import Qwen3VLDesktopUseAdapter

    adapter = Qwen3VLDesktopUseAdapter()
    processor = _load_processor("Qwen/Qwen3-VL-4B-Instruct")

    raw = (
        "Action: click here\n"
        "<tool_call>\n{\"name\": \"computer_use\", "
        "\"arguments\":{\"action\":\"left_click\",\"coordinate\":[100,200]}}\n</tool_call>"
    )
    messages = _build_trajectory_messages(
        adapter,
        user_text="Click at 100,200.",
        raw_response=raw,
    )

    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # No <think> tag anywhere in the past-turn assistant block.
    lines = rendered.split("<|im_start|>")
    assistant_block = next(
        (ln for ln in lines if ln.startswith("assistant\n")), None
    )
    assert assistant_block is not None
    assert "<think>" not in assistant_block
    assert "</think>" not in assistant_block
    # tool_call JSON verbatim:
    assert '"coordinate":[100,200]' in assistant_block


# =============================================================================
# MAI-UI: tool_call JSON whitespace must be preserved byte-exact
# =============================================================================

@pytest.mark.skipif(
    not _model_cached("Tongyi-MAI/MAI-UI-2B"),
    reason="MAI-UI-2B weights not cached locally",
)
def test_mai_ui_tool_call_json_whitespace_preserved():
    """MAI-UI's canonical path uses ``json.dumps(..., separators=(",", ":"))``
    which compacts whitespace. Short-circuit must bypass that and keep
    the model's original spacing byte-exact."""
    from lite.agents.models.mai_ui.adapter import MAIUIMobileUseAdapter

    adapter = MAIUIMobileUseAdapter()
    processor = _load_processor("Tongyi-MAI/MAI-UI-2B")

    # JSON with spaces — canonical re-serialization would compact to
    # `{"name":"mobile_use","arguments":{"action":"tap","coordinate":[100,200]}}`.
    raw = (
        "<tool_call>\n"
        '{"name": "mobile_use", "arguments": {"action": "tap", "coordinate": [100, 200]}}\n'
        "</tool_call>"
    )
    messages = _build_trajectory_messages(
        adapter,
        user_text="Tap at 100,200.",
        raw_response=raw,
    )

    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Verbatim: spaces after `:` and `,` preserved (NOT `"name":"mobile_use"`).
    assert '"name": "mobile_use"' in rendered, (
        f"JSON whitespace was compacted — short-circuit didn't fire:\n{rendered!r}"
    )
    assert '"coordinate": [100, 200]' in rendered
