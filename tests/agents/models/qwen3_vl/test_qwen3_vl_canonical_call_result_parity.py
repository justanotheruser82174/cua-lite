"""Qwen3-VL canonical call/result parity tests.

Qwen3-VL owns native ``computer_use`` canonicalization and the structured
tool-call side of the Qwen render path. Core result alignment is covered in
``tests/agents/core/agent/test_canonical_call_result_parity.py``.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/agents/models/qwen3_vl/test_qwen3_vl_canonical_call_result_parity.py \
        -p no:cacheprovider -q
"""

from __future__ import annotations

from typing import Any

import pytest
from PIL import Image

from lite.agents.bootstrap import register_all
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.models.qwen3_vl.action_space import Qwen3VLDesktopActionSpace
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.tools import make_tool_schema
from lite.core.tools.calls import (
    stamp_tool_call_list_ids,
    tool_call_arguments,
    tool_call_id,
    tool_call_name,
)

register_all()


_ADAPTER_KEY = "qwen3_vl@desktop@use"
_GOTO_SCHEMA: dict[str, Any] = make_tool_schema(
    "goto",
    description="Navigate to a URL.",
    parameters={
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    },
)

_NATIVE_CLICK_A = {
    "name": "computer_use",
    "arguments": {"action": "left_click", "coordinate": [10, 20]},
}
_NATIVE_CLICK_B = {
    "name": "computer_use",
    "arguments": {"action": "left_click", "coordinate": [30, 40]},
}
_NATIVE_GOTO = {"name": "goto", "arguments": {"url": "https://example.com"}}


def _canonical(native_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Model-emitted native calls -> canonical Lite calls, with ids."""
    canonical = Qwen3VLDesktopActionSpace().convert_tool_calls_from_agent(
        native_calls,
        active_extra_tool_names={"goto"},
        active_extra_tool_schemas=[_GOTO_SCHEMA],
    )
    return stamp_tool_call_list_ids(canonical, preserve=False)


def _role_tool_count(messages: list[dict[str, Any]]) -> int:
    return sum(1 for message in messages if message.get("role") == "tool")


def _native_call_count(messages: list[dict[str, Any]]) -> int:
    """Count native ``<tool_call>`` blocks across Qwen prompt dialects."""
    total = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        total += len(message.get("tool_calls") or [])
        for part in message.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "text":
                total += part.get("text", "").count("<tool_call>")
    return total


def _multi_call_sample(canonical: list[dict[str, Any]]) -> LiteSample:
    """user / assistant(N calls) / tool x N - the shape the corpus never had."""
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "Open the page and click twice."},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "Step 0: act."}],
            "tool_calls": canonical,
        },
    ]
    for offset, call in enumerate(canonical):
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id(call),
                "content": [
                    {"type": "image", "index": offset + 1},
                    {"type": "text", "text": f"obs for {tool_call_name(call)}"},
                ],
            }
        )
    return LiteSample(
        metadata=LiteCUAMetadata(dims=("desktop", "use"), extra_tool_schemas=[_GOTO_SCHEMA]),
        images=[
            Image.new("RGB", (32, 32), color=(k * 30 % 256, 0, 0))
            for k in range(len(canonical) + 1)
        ],
        messages=messages,
    )


def test_adjacent_native_gui_calls_merge_into_one_canonical_call() -> None:
    """Two native clicks -> one canonical action-batch call."""
    canonical = _canonical([_NATIVE_CLICK_A, _NATIVE_CLICK_B])

    assert [tool_call_name(call) for call in canonical] == ["computer"]
    assert tool_call_arguments(canonical[0])["actions"] == [
        {"action": "click", "coordinate": [10, 20]},
        {"action": "click", "coordinate": [30, 40]},
    ]


def test_gui_text_tool_gui_stays_three_canonical_calls() -> None:
    """A non-GUI tool between two GUI calls breaks the adjacent merge run."""
    canonical = _canonical([_NATIVE_CLICK_A, _NATIVE_GOTO, _NATIVE_CLICK_B])

    assert [tool_call_name(call) for call in canonical] == ["computer", "goto", "computer"]
    assert tool_call_arguments(canonical[0])["actions"] == [
        {"action": "click", "coordinate": [10, 20]}
    ]
    assert tool_call_arguments(canonical[2])["actions"] == [
        {"action": "click", "coordinate": [30, 40]}
    ]


@pytest.mark.parametrize("adapter_key", [_ADAPTER_KEY])
@pytest.mark.parametrize(
    "native_calls,expected_canonical",
    [
        pytest.param([_NATIVE_CLICK_A, _NATIVE_CLICK_B], 1, id="merged-2-native-1-canonical"),
        pytest.param(
            [_NATIVE_CLICK_A, _NATIVE_GOTO, _NATIVE_CLICK_B],
            3,
            id="unmerged-3-native-3-canonical",
        ),
    ],
)
def test_unroll_keeps_one_role_tool_message_per_canonical_call(
    adapter_key: str, native_calls: list[dict[str, Any]], expected_canonical: int
) -> None:
    """N canonical ``role:"tool"`` messages reach the Qwen3-VL prompt."""
    canonical = _canonical(native_calls)
    assert len(canonical) == expected_canonical

    step = AgentAdapterRegistry.get(adapter_key).unroll(_multi_call_sample(canonical)).steps[-1]

    assert _role_tool_count(step) == expected_canonical


@pytest.mark.parametrize("adapter_key", [_ADAPTER_KEY])
def test_unmerged_turn_renders_three_native_calls_answered_by_three_role_tools(
    adapter_key: str,
) -> None:
    """The non-merged control: native and result counts agree."""
    canonical = _canonical([_NATIVE_CLICK_A, _NATIVE_GOTO, _NATIVE_CLICK_B])
    assert len(canonical) == 3

    step = AgentAdapterRegistry.get(adapter_key).unroll(_multi_call_sample(canonical)).steps[-1]

    assert _native_call_count(step) == 3
    assert _role_tool_count(step) == 3


@pytest.mark.parametrize("adapter_key", [_ADAPTER_KEY])
def test_merged_turn_renders_two_native_calls_answered_by_one_role_tool(
    adapter_key: str,
) -> None:
    """A merged action batch renders two native calls and one tool response."""
    canonical = _canonical([_NATIVE_CLICK_A, _NATIVE_CLICK_B])
    assert len(canonical) == 1

    step = AgentAdapterRegistry.get(adapter_key).unroll(_multi_call_sample(canonical)).steps[-1]

    assert _native_call_count(step) == 2
    assert _role_tool_count(step) == 1
