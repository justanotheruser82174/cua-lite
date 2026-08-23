"""WebVoyager role:tool grouping protocol coverage."""

from __future__ import annotations

from lite.agents.extensions.webharbor.webvoyager.protocol import (
    WebVoyagerQwen3_5HistoryProtocol,
    WebVoyagerQwen3VLHistoryProtocol,
)
from lite.core.tools import make_tool_call


def _user(index: int, text: str = "Do the task.") -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "image", "index": index},
            {"type": "text", "text": text},
        ],
    }


def _tool(
    call_id: str,
    *,
    index: int | None = None,
    text: str = "obs",
    metadata: dict | None = None,
) -> dict:
    content: list[dict] = []
    if index is not None:
        content.append({"type": "image", "index": index})
    if text:
        content.append({"type": "text", "text": text})
    if metadata:
        content.append({"type": "metadata", "data": metadata})
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _assistant(step: int, call_ids: list[str], *, name: str = "click") -> dict:
    return {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": f"Memory: step {step}\nProgress: step {step}\nAction: click {step}",
            }
        ],
        "tool_calls": [
            make_tool_call(name, {"coordinate": [step, step]}, call_id=call_id)
            for call_id in call_ids
        ],
    }


def test_webvoyager_qwen3_vl_does_not_repair_unmatched_tool_inside_window() -> None:
    messages = [
        _user(0, "Find the product."),
        _assistant(0, ["call_0"]),
        _tool("call_0", index=1, text="boundary result"),
        _assistant(1, ["call_1"]),
        _tool("other_call", index=2, text="orphan tool result"),
        _assistant(2, []),
    ]

    out = WebVoyagerQwen3VLHistoryProtocol(full_history_size=2).process_messages(messages)

    unmatched = next(message for message in out if message.get("tool_call_id") == "other_call")
    assert unmatched["role"] == "tool"
    assert unmatched["content"] == messages[4]["content"]


def test_webvoyager_qwen3_5_does_not_repair_unmatched_tool_inside_window() -> None:
    messages = [
        _user(0, "Find the product."),
        _assistant(0, ["call_0"]),
        _tool("call_0", index=1, text="boundary result"),
        _assistant(1, ["call_1"]),
        _tool("other_call", index=2, text="orphan tool result"),
        _assistant(2, []),
    ]

    out = WebVoyagerQwen3_5HistoryProtocol(history_n=2, image_max=3).process_messages(messages)

    unmatched = next(message for message in out if message.get("tool_call_id") == "other_call")
    assert unmatched["role"] == "tool"
    assert unmatched["content"] == messages[4]["content"]


def test_webvoyager_qwen3_vl_summary_injection_keeps_boundary_tool_text() -> None:
    tool_text = "Current page plus error text."
    messages = [
        _user(0, "Find the cheapest laptop."),
        _assistant(0, ["call_0"]),
        _tool("call_0", index=1, text=tool_text, metadata={"is_error": True}),
    ]

    out = WebVoyagerQwen3VLHistoryProtocol(full_history_size=1).process_messages(messages)

    first_message = out[0]
    assert first_message["role"] == "user"
    texts = [
        item["text"] for item in first_message["content"]
        if item.get("type") == "text"
    ]
    assert any("Find the cheapest laptop." in text for text in texts)
    assert tool_text in texts


def test_webvoyager_qwen3_5_summary_injection_keeps_boundary_tool_text() -> None:
    tool_text = "Current page plus error text."
    messages = [
        _user(0, "Find the cheapest laptop."),
        _assistant(0, ["call_0"]),
        _tool("call_0", index=1, text=tool_text, metadata={"is_error": True}),
    ]

    out = WebVoyagerQwen3_5HistoryProtocol(history_n=1, image_max=3).process_messages(messages)

    first_message = out[0]
    assert first_message["role"] == "user"
    texts = [
        item["text"] for item in first_message["content"]
        if item.get("type") == "text"
    ]
    assert any("Find the cheapest laptop." in text for text in texts)
    assert tool_text in texts
