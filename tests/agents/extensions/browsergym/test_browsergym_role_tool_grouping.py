"""BrowserGym role:tool grouping extension coverage."""

from __future__ import annotations

from lite.agents.extensions.browsergym.goal_image import splice_goal_images
from lite.agents.models.qwen3_vl.protocol import Qwen3VLHistoryProtocol
from lite.core.tools import make_tool_call


def _tool(call_id: str, *, index: int | None = None, text: str = "obs") -> dict:
    content: list[dict] = []
    if index is not None:
        content.append({"type": "image", "index": index})
    if text:
        content.append({"type": "text", "text": text})
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


def test_goal_image_splice_survives_role_tool_window_boundary() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 3},
                {"type": "image", "index": 4},
                {"type": "image", "index": 0},
                {"type": "text", "text": "Buy the product matching the reference image."},
                {"type": "metadata", "data": {"goal_image_indices": [3, 4]}},
            ],
        },
        _assistant(0, ["call_0"]),
        _tool("call_0", index=1, text="Current page."),
    ]

    result = Qwen3VLHistoryProtocol(full_history_size=1).process_messages(messages)
    splice_goal_images(messages, result)
    first_user = next(message for message in result if message["role"] == "user")

    assert first_user.get("tool_call_id") is None
    image_indices = [
        item["index"] for item in first_user["content"] if item.get("type") == "image"
    ]
    assert image_indices[:2] == [3, 4]
    assert 1 in image_indices
