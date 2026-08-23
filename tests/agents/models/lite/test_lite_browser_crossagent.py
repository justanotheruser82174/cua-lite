"""Lite browser cross-agent student adapter coverage.

Run:
    uv run pytest tests/agents/models/lite/test_lite_browser_crossagent.py -v
"""

from __future__ import annotations

from lite.agents.bootstrap import register_all
from lite.agents.core.action_space.base import LiteDesktopActionSpace
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_name
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


def test_lite_student_identity() -> None:
    """Lite is canonical: nav and answer pass through unchanged."""

    sample = _browser_sample()
    steps = (
        AgentAdapterRegistry.get(
            "lite@browser@use",
            metadata=sample.metadata,
        )
        .unroll(sample)
        .steps
    )

    targets = [
        tool_call_name([m for m in step if m["role"] == "assistant"][-1]["tool_calls"][0])
        for step in steps
    ]
    assert targets == ["goto", "computer", "response"]

