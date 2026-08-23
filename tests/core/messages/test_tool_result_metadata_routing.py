"""Metadata content-item routing for observation and tool-result messages.

Guardrails for structured ``metadata`` on observation and tool-result messages.
Two invariants must hold for both the turn-0 ``role:"user"`` observation and
per-call ``role:"tool"`` results:

  1. The observation's structured ``metadata`` (page_title, effect tags, …)
     must keep flowing as a first-class
     ``{"type":"metadata","data":…}`` content ITEM — the sole channel the
     rendering / protocol / goal-image layers read (never a bare kwarg).
     ``build_initial_user_message`` emits exactly that item
     (``lite/agents/core/agent/utils/messages.py``); the per-call
     ``role:"tool"`` builder (``build_tool_result_message``) must emit the SAME
     item shape.

Hermetic: pure content-dict assertions on message builders — no model /
processor / network.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/core/messages/test_tool_result_metadata_routing.py \
        -p no:cacheprovider -q
"""

from __future__ import annotations

from lite.agents.core.agent.utils.messages import (
    build_initial_user_message,
    build_tool_result_message,
)


def _metadata_items(msg: dict) -> list[dict]:
    return [c for c in msg["content"] if c.get("type") == "metadata"]


# -----------------------------------------------------------------------------
# turn-0 observation builder emits metadata as a content item
# -----------------------------------------------------------------------------
def test_build_initial_user_message_emits_metadata_item() -> None:
    """Current contract: ``build_initial_user_message`` routes
    ``metadata`` into a ``{"type":"metadata","data":…}`` content item (verbatim
    ``data``), NOT a top-level message field. This is the single channel every
    downstream renderer/protocol reads."""
    msg = build_initial_user_message("t", metadata={"page_title": "X"})

    items = _metadata_items(msg)
    assert items == [{"type": "metadata", "data": {"page_title": "X"}}]
    # The screenshot + text parts still ride the same message (nothing displaced).
    assert {"type": "image", "index": 0} in msg["content"]
    assert {"type": "text", "text": "t"} in msg["content"]


# -----------------------------------------------------------------------------
# per-call role:"tool" builder emits the same item
# -----------------------------------------------------------------------------
def test_tool_result_metadata_becomes_content_item() -> None:
    """Current contract: the per-call ``role:"tool"`` result builder
    (``build_tool_result_message``) emits a tool result's ``metadata`` as a
    ``{"type":"metadata","data":…}`` content item — symmetric with the turn-N
    observation channel above — so page_title / effect tags / goal-image
    indices keep flowing through the metadata channel once the observation rides a
    ``role:"tool"`` message instead of a ``role:"user"`` bubble.

    Target shape (documented so the diff on flip is reviewable):
        build_tool_result_message(
            call_id="call_0", image_indices=(0,), text="t",
            metadata={"page_title": "X"},
        ) -> {
            "role": "tool", "tool_call_id": "call_0",
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "t"},
                {"type": "metadata", "data": {"page_title": "X"}},
            ],
        }

    """
    msg = build_tool_result_message(
        call_id="call_0", image_indices=(0,), text="t", metadata={"page_title": "X"}
    )
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_0"
    assert _metadata_items(msg) == [{"type": "metadata", "data": {"page_title": "X"}}]


def test_tool_result_error_projects_to_labelled_text_without_new_content_type() -> None:
    """Native results keep ``text`` and ``error`` separate; prompt projection is labelled text."""
    msg = build_tool_result_message(
        call_id="call_0",
        image_indices=(0,),
        text="## AXTree:\nbody",
        metadata={"page_title": "X"},
        error="unsupported action: bogus",
    )

    text_items = [c for c in msg["content"] if c.get("type") == "text"]
    assert text_items == [
        {
            "type": "text",
            "text": "## AXTree:\nbody\n\n## Error from previous action:\n"
            "unsupported action: bogus",
        }
    ]
    assert [c for c in msg["content"] if c.get("type") == "error"] == []
    assert _metadata_items(msg) == [{"type": "metadata", "data": {"page_title": "X"}}]


def test_tool_result_error_only_projects_to_labelled_text() -> None:
    msg = build_tool_result_message(
        call_id="call_0",
        image_indices=(),
        text=None,
        error="unsupported action: goto",
    )

    assert msg == {
        "role": "tool",
        "tool_call_id": "call_0",
        "content": [
            {
                "type": "text",
                "text": "## Error from previous action:\nunsupported action: goto",
            }
        ],
    }
