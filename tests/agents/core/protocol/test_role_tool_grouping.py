"""Protocol-window boundary coverage for grouped role:tool observations."""

from __future__ import annotations

from lite.agents.core.protocol.window import append_with_boundary_tool_projection
from lite.core.tools import make_tool_call


def _user(index: int, text: str = "Do the task.") -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "image", "index": index},
            {"type": "text", "text": text},
        ],
    }


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


def test_boundary_projection_keeps_tool_results_after_assistant() -> None:
    assistant = _assistant(0, ["call_0"])
    result = [_user(0), assistant]
    observations = [
        _tool("other_call", index=1, text="orphan tool result"),
        _tool("call_0", index=2, text="late result for old assistant"),
    ]

    append_with_boundary_tool_projection(result, observations)

    assert [message["role"] for message in result] == [
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert result[2]["tool_call_id"] == "other_call"
    assert result[3]["tool_call_id"] == "call_0"


def test_boundary_projection_projects_window_leading_tool_result() -> None:
    result = []
    observations = [
        _tool("call_0", index=2, text="window-leading result"),
    ]

    append_with_boundary_tool_projection(result, observations)

    assert result == [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 2},
                {"type": "text", "text": "window-leading result"},
            ],
        }
    ]
