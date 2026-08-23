"""EvoCUA role:tool window coverage not covered by generic protocol rows."""

from __future__ import annotations

from lite.agents.models.evocua.adapter import EvoCUADesktopUseAdapter
from lite.agents.models.qwen3_vl.protocol import Qwen3VLHistoryProtocol
from lite.core.tools import make_tool_call

ERROR_TEXT = "## Error from previous action:\nSENTINEL_LOCAL_WINDOW_invalid_click"


def _user(index: int, text: str = "SENTINEL_LOCAL_WINDOW_goal") -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "image", "index": index},
            {"type": "text", "text": text},
        ],
    }


def _assistant(step: int, call_id: str | None = None) -> dict:
    tool_calls = []
    if call_id is not None:
        tool_calls.append(
            make_tool_call(
                "computer",
                {"actions": [{"action": "click", "coordinate": [step, step]}]},
                call_id=call_id,
            )
        )
    return {
        "role": "assistant",
        "content": [{"type": "action_description", "text": f"click step {step}"}],
        "tool_calls": tool_calls,
    }


def _tool(call_id: str, index: int, text: str, *, error: bool = False) -> dict:
    content = [
        {"type": "image", "index": index},
        {"type": "text", "text": text},
    ]
    if error:
        content.append({"type": "metadata", "data": {"is_error": True}})
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _all_text(messages: list[dict]) -> str:
    return "\n".join(
        part["text"]
        for message in messages
        for part in (message.get("content") or [])
        if isinstance(part, dict) and part.get("type") == "text"
    )


def test_evocua_window_keeps_current_role_tool_error_as_tool_feedback() -> None:
    protocol = EvoCUADesktopUseAdapter().protocol
    assert isinstance(protocol, Qwen3VLHistoryProtocol)
    protocol.full_history_size = 2
    messages = [
        _user(0),
        _assistant(0, "call_0"),
        _tool("call_0", 1, "SENTINEL_EVOCUA_evicted_ordinary_result"),
        _assistant(1, "call_1"),
        _tool("call_1", 2, "SENTINEL_EVOCUA_boundary_screen"),
        _assistant(2, "call_2"),
        _tool("call_2", 3, ERROR_TEXT, error=True),
        _assistant(3),
    ]

    out = protocol.process_messages(messages)

    current_tool = next(message for message in out if message.get("tool_call_id") == "call_2")
    assert current_tool["role"] == "tool"
    assert current_tool["content"] == [
        {"type": "image", "index": 3},
        {"type": "text", "text": ERROR_TEXT},
        {"type": "metadata", "data": {"is_error": True}},
    ]
    text = _all_text(out)
    assert "SENTINEL_EVOCUA_evicted_ordinary_result" not in text
    assert "SENTINEL_EVOCUA_boundary_screen" in text
    assert ERROR_TEXT in text
