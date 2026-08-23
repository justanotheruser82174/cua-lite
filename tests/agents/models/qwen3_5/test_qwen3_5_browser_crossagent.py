"""Qwen3.5 browser cross-agent student adapter coverage.

Run:
    uv run pytest tests/agents/models/qwen3_5/test_qwen3_5_browser_crossagent.py -v
"""

from __future__ import annotations

import json

from lite.agents.bootstrap import register_all
from lite.agents.core.action_space.base import LiteDesktopActionSpace
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.tools import make_tool_call
from lite.core.tools.extra_tools import LiteBrowserNavToolSet, LiteFinishToolSet

register_all()


def _browser_sample() -> LiteSample:
    """Build a browser trajectory with goto, click, and response actions."""

    nav = [
        LiteBrowserNavToolSet.goto(url="https://example.com"),
        LiteDesktopActionSpace.click(coordinate=[500, 300]),
        make_tool_call("response", {"text": "the answer is 42"}),
    ]
    messages: list[dict] = []
    for i, tool_call in enumerate(nav):
        content = [{"type": "image", "index": i}]
        if i == 0:
            content.append({"type": "text", "text": "Find the answer."})
        messages.append({"role": "user", "content": content})
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": f"step {i}"}],
                "tool_calls": [tool_call],
            }
        )

    return LiteSample(
        metadata=LiteCUAMetadata(
            dims=(LiteCUAMetadata.Platform.BROWSER, LiteCUAMetadata.TaskType.USE),
            extra_tool_schemas=[
                *LiteBrowserNavToolSet.get_tool_schemas(include=["goto", "back"]),
                LiteFinishToolSet.get_tool_schema("response"),
            ],
        ),
        messages=messages,
        images=[f"img{i}.png" for i in range(len(nav))],
    )


def _serialize_assistants(step: list[dict]) -> str:
    parts = []
    for message in step:
        if message.get("role") == "assistant":
            parts.append(json.dumps(message.get("content")))
            parts.append(json.dumps(message.get("tool_calls")))
    return "\n".join(parts)


def test_qwen3_5_student_native_answer_and_nav() -> None:
    sample = _browser_sample()
    steps = (
        AgentAdapterRegistry.get(
            "qwen3_5@browser@use",
            metadata=sample.metadata,
        )
        .unroll(sample)
        .steps
    )

    blob = _serialize_assistants(steps[-1])
    assert "<function=goto>" in blob
    assert "answer" in blob
    assert "<function=response>" not in blob

