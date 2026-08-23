"""Regression tests for the model-boundary content sanitizer.

Locks the contract behind the fix for the metadata ``TemplateError`` (issue #42):
cua-lite-internal side-channel content parts (``MetadataContent`` and friends)
must be stripped before a message reaches the model, while model-facing
``text``/``image`` parts and the *persistent* message history survive untouched.

Run:
    uv run pytest tests/core/messages/test_keep_model_visible_content.py -q
"""
from __future__ import annotations

import pytest

from lite.core.messages import keep_model_visible_content


def _types(msg):
    return [p.get("type") for p in msg["content"]]


def test_strips_metadata_keeps_text_and_image():
    msgs = [{"role": "user", "content": [
        {"type": "image", "index": 0},
        {"type": "text", "text": "hi"},
        {"type": "metadata", "data": {"page_title": "X", "url": "y"}},
    ]}]
    out = keep_model_visible_content(msgs)
    assert _types(out[0]) == ["image", "text"]  # metadata dropped, order preserved


def test_strips_every_internal_content_kind():
    """All cua-lite-internal kinds go — metadata (user) AND the assistant ones —
    so a future internal type (or a stray one on the wrong message) can't leak."""
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "keep"},
        {"type": "metadata", "data": {}},
        {"type": "action_description", "text": "Action: click"},
        {"type": "inline_reasoning", "text": "Thought: ..."},
        {"type": "history_summary", "text": "so far ..."},
        {"type": "image", "index": 1},
    ]}]
    out = keep_model_visible_content(msgs)
    assert _types(out[0]) == ["text", "image"]


def test_does_not_mutate_input_so_history_consumers_keep_working():
    """The stored history must keep internal metadata for downstream protocols.

    The sanitizer must operate on a copy because webgym history reads
    ``page_title`` metadata after model-visible filtering.
    """
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "x"},
        {"type": "metadata", "data": {"page_title": "Search"}},
    ]}]
    keep_model_visible_content(msgs)
    # input untouched → next turn's consumers still find the metadata
    assert any(p.get("type") == "metadata" for p in msgs[0]["content"])
    assert msgs[0]["content"][1]["data"]["page_title"] == "Search"


def test_drops_provider_native_untyped_blocks():
    """Provider-native no-type blocks are not canonical Lite content.

    A provider/template projection that needs them must preserve them at that
    owner boundary instead of teaching core message filtering provider keys.
    """
    msgs = [{"role": "user", "content": [
        {"image": "<bytes>"},
        {"video": "<bytes>"},
        {"type": "image", "index": 0},
        {"type": "metadata", "data": {}},
    ]}]
    out = keep_model_visible_content(msgs)
    assert out[0]["content"] == [{"type": "image", "index": 0}]


def test_string_content_and_missing_content_untouched():
    msgs = [
        {"role": "system", "content": "system prompt"},   # str content
        {"role": "assistant", "tool_calls": [{"x": 1}]},   # no content key
    ]
    out = keep_model_visible_content(msgs)
    assert out[0]["content"] == "system prompt"
    assert "content" not in out[1]


def test_empty_after_strip_yields_empty_list_not_error():
    msgs = [{"role": "user", "content": [{"type": "metadata", "data": {}}]}]
    out = keep_model_visible_content(msgs)
    assert out[0]["content"] == []


def test_real_qwen35_template_raises_on_metadata_but_not_after_sanitize():
    """End-to-end contract against the REAL chat template: a metadata part makes
    qwen3.5's template raise ``Unexpected item type``; the sanitized copy renders
    cleanly. Guards the whole reason the sanitizer exists. Skips if the tokenizer
    isn't locally available (the chat_template lives in tokenizer_config.json)."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    except Exception as e:  # not cached / offline
        pytest.skip(f"Qwen3.5 tokenizer unavailable: {e}")

    bad = [{"role": "user", "content": [
        {"type": "text", "text": "page"},
        {"type": "metadata", "data": {"page_title": "Home"}},
    ]}]
    with pytest.raises(Exception, match="Unexpected item type"):
        tok.apply_chat_template(bad, tokenize=False, add_generation_prompt=True)

    good = keep_model_visible_content(bad)
    # must not raise
    tok.apply_chat_template(good, tokenize=False, add_generation_prompt=True)


@pytest.mark.asyncio
async def test_base_predict_boundary_strips_metadata_before_model(monkeypatch):
    """Wiring guard: ``AdapterBasedAgent._predict_with_details`` MUST call
    ``keep_model_visible_content`` — a metadata-bearing rendered AgentStep (as the
    webgym/browsergym protocols produce) must reach ``build_generation_prompt``
    with NO metadata part. Without this test a refactor could silently drop the
    base.py boundary call and reintroduce issue #42 while all the unit tests above
    still pass."""
    from lite.agents.bootstrap import register_all
    from lite.agents.models import AgentRegistry
    from lite.core import LiteCUAMetadata, LiteSample

    register_all()

    async def _fake_gen(**_):
        return {"response": "Action: click\n<tool_call>{\"name\": \"x\"}</tool_call>"}

    agent = AgentRegistry.get(
        "qwen3_vl@desktop@use", generate_fn=_fake_gen, processor=None,
    )
    # Force the rendered step to carry a metadata part (protocol windowing keeps it).
    monkeypatch.setattr(
        agent.adapter, "render_step",
        lambda *a, **k: [
            {"role": "user", "content": [
                {"type": "text", "text": "obs"},
                {"type": "metadata", "data": {"page_title": "X"}},
            ]},
        ],
    )
    captured: dict = {}
    agent.build_generation_prompt = lambda messages: (
        captured.__setitem__("messages", messages) or "stub"
    )

    sample = LiteSample(
        metadata=LiteCUAMetadata(
            dims=(LiteCUAMetadata.Platform.DESKTOP, LiteCUAMetadata.TaskType.USE)
        ),
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    )
    await agent._predict_with_details(sample, processed_images=[])

    types = [
        p.get("type")
        for m in captured["messages"]
        for p in (m.get("content") or [])
        if isinstance(p, dict)
    ]
    assert "metadata" not in types, "base.py boundary did NOT strip metadata"
    assert "text" in types  # model-facing content preserved
