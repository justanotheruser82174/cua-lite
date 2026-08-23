"""MAI-UI history preserves real MCP text-tool output while budgeting images."""

from __future__ import annotations

from lite.agents.models.mai_ui.protocol import MAIUIHistoryProtocol
from lite.core.tools import make_tool_call


def _user(index: int, text: str | None = None) -> dict:
    content: list[dict] = [{"type": "image", "index": index}]
    if text is not None:
        content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def _assistant(text: str, call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "action_description", "text": text}],
        "tool_calls": [
            make_tool_call("bash", {"command": "echo"}, call_id=call_id)
        ],
    }


def _tool_text(call_id: str, text: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": [{"type": "text", "text": text}]}


def _image_window_messages() -> list[dict]:
    return [
        _user(0, "task"),
        _assistant("first visible action", "c0"),
        _user(1),
        _assistant("second visible action", "c1"),
        _user(2),
        _assistant("run text tool", "c2"),
        _tool_text("c2", "BASH-OUT-IN-WINDOW"),
        _assistant("consume text output", "c3"),
        _tool_text("c3", "BASH-OUT-IN-WINDOW-2"),
        _assistant("consume more text output", "c4"),
        _user(3),
        _assistant("last visible action", "c5"),
    ]


def _old_text_messages() -> list[dict]:
    messages = _image_window_messages()
    messages.insert(2, _tool_text("c0", "BASH-OUT-OLD"))
    messages.insert(3, _assistant("consume old text output", "old_consumer"))
    return messages


def _text(messages: list[dict]) -> str:
    return "\n".join(
        part["text"]
        for message in messages
        for part in message.get("content") or []
        if isinstance(part, dict) and part.get("type") in {"text", "action_description"}
    )


def _image_indices(messages: list[dict]) -> list[int]:
    return [
        part["index"]
        for message in messages
        for part in message.get("content") or []
        if isinstance(part, dict) and part.get("type") == "image"
    ]


def test_text_only_turns_do_not_consume_mai_ui_image_window() -> None:
    protocol = MAIUIHistoryProtocol(full_history_size=2)

    out = protocol.process_messages(_image_window_messages())

    assert _image_indices(out) == [2, 3]
    assert "BASH-OUT-IN-WINDOW" in _text(out)
    assert "BASH-OUT-IN-WINDOW-2" in _text(out)


def test_aged_out_text_tool_output_survives_mai_ui_window() -> None:
    protocol = MAIUIHistoryProtocol(full_history_size=1)

    out = protocol.process_messages(_old_text_messages())

    assert _image_indices(out) == [3]
    assert "BASH-OUT-OLD" in _text(out)
