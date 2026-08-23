"""Qwen3-VL browser cross-agent student adapter coverage.

Run:
    uv run pytest tests/agents/models/qwen3_vl/test_qwen3_vl_browser_crossagent.py -v
"""

from __future__ import annotations

from lite.agents.bootstrap import register_all
from lite.agents.core.action_space import ActionSpaceRegistry
from lite.agents.core.action_space.base import LiteDesktopActionSpace
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
from lite.agents.models.gpt.utils.parse import parse_output_items_with_provenance
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_arguments, tool_call_name
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


def test_qwen3_vl_student_native_answer_and_nav() -> None:
    sample = _browser_sample()
    adapter = AgentAdapterRegistry.get("qwen3_vl@browser@use", metadata=sample.metadata)
    steps = adapter.unroll(sample).steps
    rendered = [
        [m for m in step if m["role"] == "assistant"][-1]["tool_calls"][0] for step in steps
    ]

    assert rendered[0]["name"] == "goto"
    assert (
        rendered[1]["name"] == "computer_use"
        and rendered[1]["arguments"]["action"] == "left_click"
    )
    assert rendered[2]["name"] == "computer_use"
    assert rendered[2]["arguments"]["action"] == "answer"
    assert all(action["name"] != "response" for action in rendered)


def test_qwen3_vl_student_reparse_identity() -> None:
    """Qwen3-VL browser wire calls re-parse back to canonical Lite actions."""

    space = ActionSpaceRegistry.get("qwen3_vl@browser")
    sample = _browser_sample()
    adapter = AgentAdapterRegistry.get("qwen3_vl@browser@use", metadata=sample.metadata)
    steps = adapter.unroll(sample).steps
    recovered = []
    for step in steps:
        tool_calls = [m for m in step if m["role"] == "assistant"][-1]["tool_calls"]
        recovered.extend(
            tool_call_name(tool_call)
            for tool_call in space.convert_tool_calls_from_agent(tool_calls)
        )

    assert recovered == ["goto", "computer", "response"]


def test_extra_tool_roundtrip_preserves_gpt_canonical_calls() -> None:
    """Qwen3-VL browser wire preserves extra tools parsed from GPT."""

    space = ActionSpaceRegistry.get("qwen3_vl@browser")
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
    )
    canonical = parsed.message["tool_calls"]

    qwen_wire = space.convert_tool_calls_to_agent(canonical)
    recovered = space.convert_tool_calls_from_agent(qwen_wire)

    assert [tool_call_name(call) for call in canonical] == ["back", "goto", "summarize"]
    assert tool_call_arguments(canonical[2]) == {"text": "page summary"}
    assert [(tool_call_name(call), tool_call_arguments(call)) for call in recovered] == [
        (tool_call_name(call), tool_call_arguments(call)) for call in canonical
    ]
