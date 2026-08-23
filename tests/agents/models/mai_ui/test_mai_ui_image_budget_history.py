"""MAI-UI inherited history windows keep evicted image-error text."""

from __future__ import annotations

from lite.agents.models.mai_ui.protocol import MAIUIHistoryProtocol
from lite.core.tools import make_tool_call
from lite.core.tools.results import TOOL_RESULT_ERROR_SECTION_HEADER


def _action(desc: str, i: int) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "action_description", "text": desc}],
        "tool_calls": [
            make_tool_call(
                "computer",
                {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                call_id=f"g{i}",
            )
        ],
    }


def _img_result(idx: int) -> dict:
    return {
        "role": "tool",
        "tool_call_id": "x",
        "content": [{"type": "image", "index": idx}],
    }


def _img_error_result(idx: int, text: str, *, metadata: bool = True) -> dict:
    content = [
        {"type": "image", "index": idx},
        {"type": "text", "text": text},
    ]
    if metadata:
        content.append({"type": "metadata", "data": {"is_error": True}})
    return {"role": "tool", "tool_call_id": "x", "content": content}


def _turn0() -> dict:
    return {
        "role": "user",
        "content": [{"type": "image", "index": 0}, {"type": "text", "text": "goal"}],
    }


def _rendered_text(messages: list[dict]) -> str:
    out = []
    for message in messages:
        for part in message.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "text":
                out.append(part["text"])
    return "\n".join(out)


def test_mai_ui_evicted_image_tool_result_with_own_error_header_survives() -> None:
    error_text = f"{TOOL_RESULT_ERROR_SECTION_HEADER}\nSENTINEL_click_failed_without_metadata"
    msgs = [
        _turn0(),
        _action("bad click", 0),
        _img_error_result(1, error_text, metadata=False),
        _action("recover", 1),
        _img_result(2),
        _action("finish", 2),
        _img_result(3),
    ]

    text = _rendered_text(MAIUIHistoryProtocol(full_history_size=1).process_messages(msgs))

    assert error_text in text
