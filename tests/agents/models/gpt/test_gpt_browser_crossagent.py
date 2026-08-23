"""GPT browser cross-agent teacher coverage.

Run:
    uv run pytest tests/agents/models/gpt/test_gpt_browser_crossagent.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from agents.models._support.provider_fakes import FakeEnv

from lite.agents.bootstrap import register_all
from lite.agents.core.action_space import ActionSpaceRegistry
from lite.agents.core.agent import AgentRegistry
from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
from lite.agents.models.gpt.utils.parse import parse_output_items_with_provenance
from lite.core import LiteCUAMetadata
from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.extra_tools import LiteBrowserNavToolSet, LiteFinishToolSet

register_all()


def test_teacher_native_action_from_agent_to_canonical() -> None:
    space = ActionSpaceRegistry.get("gpt@browser")
    click = {"type": "click", "x": 640, "y": 384}

    action = space.convert_tool_calls_from_agent([click], resolution=(1280, 768))[0]

    assert action == make_tool_call(
        "computer",
        {"actions": [{"action": "click", "coordinate": [500, 500]}]},
    )


def test_extra_tool_gpt_parse_to_canonical() -> None:
    """GPT-emitted extra tools survive the provider parse boundary."""

    parsed = parse_output_items_with_provenance(
        [
            {
                "type": "function_call",
                "call_id": "call_0000",
                "name": "back",
                "arguments": "{}",
            },
            {
                "type": "function_call",
                "call_id": "call_0001",
                "name": "goto",
                "arguments": '{"url": "https://example.com"}',
            },
            {
                "type": "function_call",
                "call_id": "call_0002",
                "name": "summarize",
                "arguments": '{"text": "page summary"}',
            },
        ],
        GPTDesktopActionSpace(),
        (1280, 768),
        extra_tool_names=frozenset({"back", "goto", "summarize"}),
    ).message
    canonical = parsed["tool_calls"]

    assert [tool_call_name(call) for call in canonical] == ["back", "goto", "summarize"]
    assert tool_call_arguments(canonical[2]) == {"text": "page summary"}


async def test_teacher_browser_tools_from_extra_tools_and_valid_actions(monkeypatch) -> None:
    """GPT agents surface browser nav and finish schemas to providers."""

    monkeypatch.setattr(
        "lite.agents.models.gpt.utils.image_io._fetch_processed_image_dims",
        AsyncMock(return_value=[(800, 600)]),
    )
    gpt_response = {
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}],
        "id": "resp_test",
        "usage": {},
    }
    agent = AgentRegistry.get("gpt@browser@use")
    agent.metadata = LiteCUAMetadata(
        valid_actions=["click", "type", "key", "scroll", "wait"],
        extra_tool_schemas=[
            *LiteBrowserNavToolSet.get_tool_schemas(include=["goto", "back"]),
            LiteFinishToolSet.get_tool_schema("response"),
            LiteFinishToolSet.get_tool_schema("terminate"),
        ],
    )
    mock = AsyncMock(return_value=gpt_response)
    monkeypatch.setattr("litellm.aresponses", mock)

    await agent.sample(FakeEnv(terminate_after=1), max_steps=2)

    names = {tool.get("name") for tool in mock.call_args.kwargs["tools"]}
    assert {"goto", "back", "response", "terminate"} <= names
