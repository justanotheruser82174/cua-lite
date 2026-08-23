"""UI-TARS role:tool grouping protocol coverage."""

from __future__ import annotations

from lite.agents.models.ui_tars.protocol import UITarsHistoryProtocol
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


def _content_types(message: dict) -> list[str]:
    return [item["type"] for item in message["content"]]


def test_windowed_role_tool_image_error_keeps_text_and_metadata() -> None:
    error_text = "## Error from previous action:\ninvalid action: mouse_move"
    messages = [
        _user(0, "Find the product."),
        _assistant(0, ["old_call"]),
        _tool("old_call", index=1, text="old screen"),
        _assistant(1, ["call_0"]),
        _tool("call_0", index=2, text=error_text, metadata={"is_error": True}),
        _assistant(2, []),
    ]

    out = UITarsHistoryProtocol(full_history_size=2).process_messages(messages)

    tool_message = next(
        message for message in out
        if message.get("role") == "tool" and message.get("tool_call_id") == "call_0"
    )
    assert _content_types(tool_message) == ["image", "text", "metadata"]
    assert any(
        item.get("type") == "text" and item.get("text") == error_text
        for item in tool_message["content"]
    )
    assert any(
        item == {"type": "metadata", "data": {"is_error": True}}
        for item in tool_message["content"]
    )


def test_windowed_role_tool_projected_error_keeps_fresh_carrier_once() -> None:
    projected_text = (
        "## AXTree:\nbutton Search\n\n"
        "## Error from previous action:\ninvalid action: screenshot"
    )
    messages = [
        _user(0, "Find the product."),
        _assistant(0, ["call_0"]),
        _tool("call_0", index=1, text=projected_text, metadata={"is_error": True}),
        _assistant(1, []),
    ]

    out = UITarsHistoryProtocol(full_history_size=2).process_messages(messages)

    tool_message = next(
        message for message in out
        if message.get("role") == "tool" and message.get("tool_call_id") == "call_0"
    )
    assert [
        item for item in tool_message["content"]
        if item.get("type") == "image"
    ] == [{"type": "image", "index": 1}]
    texts = [
        item["text"] for item in tool_message["content"]
        if item.get("type") == "text"
    ]
    joined = "\n".join(texts)
    assert joined.count("## Error from previous action:") == 1
    assert joined.count("invalid action: screenshot") == 1
    assert projected_text in joined
