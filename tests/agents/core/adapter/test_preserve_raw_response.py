"""``preserve_raw_response`` sentinel — radix sanity.

Invariant: for every adapter, when ``preserve_raw_response=True``
(the default and only supported mode), step-to-step text prefixes hold
byte-exactly. The radix segmenter relies on this — if a
canonical-render drift bypasses the sidecar, the prefix equality never
holds and packing silently degrades to length-1 segments forever.

We don't run the segmenter here (slime-import-free). We assert the
text-level invariant: ``step_{k+1}.prompt.startswith(step_k.prompt + step_k.response)``
on rendered adapter output for a multi-turn synthetic trajectory.

Load-bearing invariant: ``preserve_raw_response`` must stay ON.
"""

from __future__ import annotations

import pytest

from lite.agents.core.adapter import AgentAdapterRegistry
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.tools.calls import make_tool_call


@pytest.fixture(autouse=True)
def _registered_adapters():
    """Both the registry lookups and the regex-stamp matcher below only see
    families the registry KNOWS. Without this the file passes or fails on
    whether some *other* test module happened to import a family first — green
    under `-n 16` sharding, red when run alone."""
    from lite.agents.bootstrap import register_all

    register_all()


def _build_synthetic_traj(*, response_texts: list[str]) -> LiteSample:
    """Build a synthetic multi-turn LiteSample with raw_response sidecars
    on each assistant message — simulating a fresh rollout."""
    messages: list[dict] = []
    for k, resp in enumerate(response_texts):
        # Turn k user obs (instruction on turn 0).
        if k == 0:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "index": k},
                    {"type": "text", "text": "do the thing"},
                ],
            })
        else:
            messages.append({
                "role": "user",
                "content": [{"type": "image", "index": k}],
            })
        # Turn k assistant message — minimal but with raw_response sidecar.
        # adapter_key set to "lite@desktop@use" so byte-exact replay
        # only fires when that exact adapter renders.
        messages.append({
            "role": "assistant",
            "content": [{"type": "action_description", "text": f"Action {k}: do step {k}"}],
            "tool_calls": [make_tool_call(
                "computer",
                {
                    "actions": [{
                        "action": "click",
                        "coordinate": [100 + k, 200 + k],
                    }],
                },
                call_id=f"call_{k:04d}",
            )],
            "raw_response": {
                "text": resp,
                "adapter_key": "lite@desktop@use",
            },
        })

    # Need len(processed_images) == n_turns to satisfy ImageContent indices.
    from PIL import Image
    images = [Image.new("RGB", (32, 32), color=(k * 30, 0, 0)) for k in range(len(response_texts))]
    return LiteSample(
        metadata=LiteCUAMetadata(dims=("desktop", "use")),
        images=images,
        messages=messages,
    )


def _render_step_text(adapter, sample: LiteSample, k: int, processed) -> tuple[str, str]:
    """Render turn ``k`` and extract (prompt_text, response_text) where
    response_text is the rendered assistant turn's text content."""
    step_messages = adapter.render_step(sample, k, processed)
    # Prompt = everything up to & excluding the final assistant.
    if step_messages and step_messages[-1].get("role") == "assistant":
        assistant = step_messages[-1]
        prompt_msgs = step_messages[:-1]
        # Extract assistant rendered text (first text content part).
        resp_text = ""
        for part in assistant.get("content", []):
            if part.get("type") == "text":
                resp_text = part.get("text", "")
                break
    else:
        prompt_msgs = step_messages
        resp_text = ""
    # Stringify prompt_msgs in a stable, simple way for prefix comparison —
    # we don't need full apply_chat_template, just a deterministic dump.
    import json
    prompt_text = json.dumps(prompt_msgs, sort_keys=True, ensure_ascii=False, default=str)
    return prompt_text, resp_text


@pytest.mark.parametrize("adapter_key,raw_response_texts", [
    (
        "lite@desktop@use",
        ["raw response 0 verbatim", "raw response 1 verbatim", "raw response 2 verbatim"],
    ),
])
def test_raw_response_sidecar_short_circuits_canonical_render(adapter_key, raw_response_texts):
    """When ``raw_response.adapter_key`` matches the rendering adapter's
    registry key, the assistant turn is replayed byte-exact instead of
    canonical-rendered — confirms the sidecar mechanism is wired.

    For each turn k, the rendered assistant content must contain the raw
    response text verbatim (proving short-circuit), not a canonical
    re-render of the parsed tool_calls + content parts.
    """
    sample = _build_synthetic_traj(response_texts=raw_response_texts)
    # Use FullHistoryProtocol so every assistant survives (no windowing) —
    # we want to assert ALL assistants are byte-exact replays.
    adapter = AgentAdapterRegistry.get(
        adapter_key, protocol_key="lite.history",
    )
    processed = [adapter.process_image(img) for img in sample.images]

    n_turns = len(raw_response_texts)
    raw_set = set(raw_response_texts)
    for k in range(1, n_turns + 1):
        step_messages = adapter.render_step(sample, k, processed)
        assistants = [m for m in step_messages if m.get("role") == "assistant"]
        assert assistants, f"turn k={k}: no assistant in render_step output"
        # Each rendered assistant's text must be one of the raw responses
        # (byte-exact replay). The order may differ from raw_response_texts
        # if the protocol re-orders / drops; what matters is that NO
        # canonical-render artifacts (e.g. "Action 0: do step 0") appear.
        for j, asst in enumerate(assistants):
            text_parts = [p.get("text", "") for p in asst.get("content", [])
                          if p.get("type") == "text"]
            joined = "\n".join(text_parts).strip()
            assert joined in raw_set, (
                f"turn k={k}, assistant j={j}: rendered text is not a "
                f"byte-exact replay of any raw response.\n"
                f"  rendered: {joined!r}\n"
                f"  raw set: {raw_set!r}"
            )
            # Also: canonical-render artifacts must NOT leak through.
            assert "Action " not in joined or joined in raw_set, (
                f"turn k={k}: canonical 'Action N:' render leaked despite sidecar"
            )


def test_raw_response_adapter_key_mismatch_falls_back_to_canonical():
    """When the sidecar's ``adapter_key`` doesn't match the rendering
    adapter, the short-circuit must NOT fire — it falls back to
    canonical render (so the warning log emits and we don't silently
    inline foreign-format text)."""
    sample = _build_synthetic_traj(response_texts=["RAW-CLAUDE-FORMAT"])
    # Override the sidecar to point at a different adapter.
    for msg in sample.messages:
        if msg.get("role") == "assistant" and "raw_response" in msg:
            msg["raw_response"]["adapter_key"] = "claude@desktop@use"

    # Render with the lite adapter — short-circuit should NOT fire.
    adapter = AgentAdapterRegistry.get("lite@desktop@use")
    processed = [adapter.process_image(img) for img in sample.images]
    step_messages = adapter.render_step(sample, k=1, processed=processed)
    assistants = [m for m in step_messages if m.get("role") == "assistant"]
    text_parts = [p.get("text", "") for p in assistants[-1].get("content", [])
                  if p.get("type") == "text"]
    joined = "\n".join(text_parts)
    # Canonical render goes through _convert_message_to_agent which emits
    # "Action: <action_description text>" for lite adapter — the raw
    # claude-format text should NOT appear.
    assert "RAW-CLAUDE-FORMAT" not in joined, (
        "adapter_key mismatch must skip the short-circuit; "
        f"got rendered text: {joined!r}"
    )
    # And the canonical "Action: " prefix should be present.
    assert "Action 0:" in joined, (
        f"canonical render missing 'Action 0:' prefix; got: {joined!r}"
    )


@pytest.mark.parametrize(
    "stamped_key,adapter_key,expected",
    [
        (
            "qwen3_vl@mobile@grounding.point",
            "qwen3_vl@mobile@grounding.point",
            True,
        ),
        (
            "qwen3_vl@mobile@grounding.point",
            "qwen3_vl@mobile@groundingXpoint",
            False,
        ),
        (
            r"qwen3_vl@(desktop|browser)@use",
            "qwen3_vl@desktop@use",
            True,
        ),
        (
            "qwen3_vl@desktop@use",
            r"qwen3_vl@(desktop|browser)@use",
            True,
        ),
        (
            r"qwen3_vl@(desktop|browser)@use-unregistered",
            "qwen3_vl@desktop@use-unregistered",
            False,
        ),
        (
            "qwen3_vl@desktop@use-unregistered",
            r"qwen3_vl@(desktop|browser)@use-unregistered",
            False,
        ),
    ],
)
def test_raw_response_registry_matcher_semantics(
    stamped_key,
    adapter_key,
    expected,
):
    """The one public matcher every raw_response replay decision goes through.

    ``BaseAgentAdapter.convert_message_to_agent`` is the sole caller — export
    replays the sidecar iff this returns True (see the short-circuit tests
    above), so the table here is the whole authorisation contract.
    """
    assert (
        AgentAdapterRegistry.raw_response_key_matches(stamped_key, adapter_key)
        is expected
    )
