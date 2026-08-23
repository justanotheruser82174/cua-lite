"""GPT desktop parser/action-space batching roundtrip tests.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/agents/models/gpt/test_gpt_adapter_batching_roundtrip.py \
        -p no:cacheprovider -q
"""

from __future__ import annotations

from typing import Any

from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
from lite.agents.models.gpt.utils.parse import parse_output_items_with_provenance
from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_arguments, tool_call_name


def _names(tool_calls: list[dict[str, Any]]) -> list[str]:
    return [tool_call_name(tc) for tc in tool_calls]


def _args(tool_call: dict[str, Any]) -> dict[str, Any]:
    return tool_call_arguments(tool_call)


def _action_name(action: Any) -> str | None:
    if not isinstance(action, dict):
        return None
    return action.get("action")


def _computer_actions(tool_calls: list[dict[str, Any]]) -> list[Any]:
    comp = next(tc for tc in tool_calls if tool_call_name(tc) == "computer")
    return tool_call_arguments(comp)["actions"]


def test_gpt_single_action_roundtrips_as_length1_wrapper() -> None:
    """GPT native actions round-trip through the canonical length-1 wrapper."""
    space = GPTDesktopActionSpace()
    lite_calls = space.convert_tool_calls_from_agent(
        [{"type": "type", "text": "hello"}],
        resolution=(1280, 768),
    )

    assert lite_calls == [
        make_tool_call(
            "computer",
            {"actions": [{"action": "type", "text": "hello"}]},
        )
    ]

    wire = space.convert_tool_calls_to_agent(lite_calls, resolution=(1280, 768))
    back = space.convert_tool_calls_from_agent(wire, resolution=(1280, 768))
    assert back == lite_calls


def test_gpt_native_batched_computer_call_parse() -> None:
    """A GPT native ``computer_call`` action list parses to one canonical batch."""
    space = GPTDesktopActionSpace()
    items = [
        {
            "type": "computer_call",
            "call_id": "call_0000",
            "actions": [
                {"type": "click", "x": 640, "y": 384},
                {"type": "type", "text": "hello"},
            ],
        },
    ]
    msg = parse_output_items_with_provenance(items, space, (1280, 768)).message
    tcs = msg["tool_calls"]

    assert len(tcs) == 1
    assert tool_call_name(tcs[0]) == "computer"
    assert [_action_name(a) for a in _computer_actions(tcs)] == ["click", "type"]


def test_gpt_native_screenshot_is_folded_into_action_batch() -> None:
    msg = parse_output_items_with_provenance(
        [
            {
                "type": "computer_call",
                "call_id": "call_0001",
                "actions": [
                    {"type": "screenshot"},
                    {"type": "click", "x": 100, "y": 100},
                ],
            },
        ],
        GPTDesktopActionSpace(),
        (1000, 1000),
    ).message

    assert msg["tool_calls"] == [
        make_tool_call(
            "computer",
            {
                "actions": [
                    {"action": "screenshot"},
                    {"action": "click", "coordinate": [100, 100]},
                ]
            },
            call_id="call_0000",
        ),
    ]


def test_gpt_native_batched_computer_call_preserves_extra_order() -> None:
    items = [
        {
            "type": "computer_call",
            "call_id": "call_0002",
            "actions": [{"type": "click", "x": 100, "y": 100}],
        },
        {
            "type": "function_call",
            "call_id": "call_0000",
            "name": "bash",
            "arguments": '{"command": "pwd"}',
        },
        {
            "type": "computer_call",
            "call_id": "call_0003",
            "actions": [{"type": "type", "text": "hello"}],
        },
    ]

    msg = parse_output_items_with_provenance(
        items,
        GPTDesktopActionSpace(),
        (1000, 1000),
        extra_tool_names=frozenset({"bash"}),
    ).message

    assert _names(msg["tool_calls"]) == ["computer", "bash", "computer"]
    assert _args(msg["tool_calls"][0]) == {
        "actions": [{"action": "click", "coordinate": [100, 100]}]
    }
    assert _args(msg["tool_calls"][1]) == {"command": "pwd"}
    assert _args(msg["tool_calls"][2]) == {"actions": [{"action": "type", "text": "hello"}]}


def test_gpt_native_single_computer_call_uses_length1_wrapper() -> None:
    msg = parse_output_items_with_provenance(
        [
            {
                "type": "computer_call",
                "call_id": "call_0004",
                "actions": [{"type": "click", "x": 100, "y": 100}],
            }
        ],
        GPTDesktopActionSpace(),
        (1000, 1000),
    ).message

    assert msg["tool_calls"] == [
        make_tool_call(
            "computer",
            {"actions": [{"action": "click", "coordinate": [100, 100]}]},
            call_id="call_0000",
        )
    ]


def test_gpt_canonical_computer_batch_unwraps_to_native_actions() -> None:
    actions = GPTDesktopActionSpace().convert_tool_calls_to_agent(
        [
            make_tool_call(
                "computer",
                {
                    "actions": [
                        {"action": "click", "coordinate": [500, 500]},
                        {"action": "type", "text": "hello"},
                    ]
                },
            )
        ],
        resolution=(1280, 768),
    )

    assert actions == [
        {"type": "click", "x": 640, "y": 384},
        {"type": "type", "text": "hello"},
    ]


def test_gpt_desktop_rejects_legacy_computer_function_call() -> None:
    msg = parse_output_items_with_provenance(
        [{"type": "function_call", "name": "computer", "arguments": '{"type": "click"}'}],
        GPTDesktopActionSpace(),
        (1000, 1000),
    ).message

    assert msg["tool_calls"] == []
