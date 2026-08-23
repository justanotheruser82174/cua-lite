"""Claude browser cross-agent teacher coverage.

Run:
    uv run pytest tests/agents/models/claude/test_claude_browser_crossagent.py -v
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents.models._support.provider_fakes import FakeEnv

from lite.agents.bootstrap import register_all
from lite.agents.core.action_space import ActionSpaceRegistry
from lite.agents.core.agent import AgentRegistry
from lite.core import LiteCUAMetadata
from lite.core.tools import make_tool_call
from lite.core.tools.extra_tools import LiteBrowserNavToolSet, LiteFinishToolSet

register_all()


@pytest.mark.parametrize("space_key,gpt_style", [("claude@browser", False)])
def test_teacher_native_action_from_agent_to_canonical(
    space_key: str,
    gpt_style: bool,
) -> None:
    space = ActionSpaceRegistry.get(space_key)
    key = "type" if gpt_style else "action"
    click = (
        {"type": "click", "x": 640, "y": 384}
        if gpt_style
        else {"action": "left_click", "coordinate": [640, 384]}
    )

    action = space.convert_tool_calls_from_agent([click], resolution=(1280, 768))[0]

    assert action == make_tool_call(
        "computer",
        {"actions": [{"action": "click", "coordinate": [500, 500]}]},
    )
    with pytest.raises(ValueError, match="unknown"):
        space.convert_tool_calls_from_agent(
            [{key: "goto", "url": "https://example.com"}],
            resolution=(1280, 768),
        )


async def test_teacher_browser_tools_from_extra_tools_and_valid_actions(monkeypatch) -> None:
    """Claude agents surface browser nav and finish schemas to providers."""

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="done", tool_calls=[], role="assistant"),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        model_dump=lambda: {"choices": []},
    )
    agent = AgentRegistry.get("claude@browser@use")
    agent.metadata = LiteCUAMetadata(
        valid_actions=["click", "type", "key", "scroll", "wait"],
        extra_tool_schemas=[
            *LiteBrowserNavToolSet.get_tool_schemas(include=["goto", "back"]),
            LiteFinishToolSet.get_tool_schema("response"),
            LiteFinishToolSet.get_tool_schema("terminate"),
        ],
    )
    mock = AsyncMock(return_value=response)
    monkeypatch.setattr("litellm.acompletion", mock)

    await agent.sample(FakeEnv(terminate_after=1), max_steps=2)

    names = {tool.get("name") for tool in mock.call_args.kwargs["tools"]}
    assert {"goto", "back", "response", "terminate"} <= names
