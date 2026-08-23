"""Coordinate parsing belongs at Qwen3-VL action-space parser boundaries."""

from __future__ import annotations

import pytest

from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.models.qwen3_vl.action_space import (
    Qwen3VLDesktopActionSpace,
    Qwen3VLMobileActionSpace,
)
from lite.core.tools.calls import tool_call_arguments


@pytest.mark.parametrize(
    ("space", "agent_call", "match"),
    [
        (
            Qwen3VLDesktopActionSpace(),
            {"name": "computer_use", "arguments": {"action": "left_click"}},
            "coordinate is required",
        ),
        (
            Qwen3VLDesktopActionSpace(),
            {
                "name": "computer_use",
                "arguments": {"action": "left_click", "coordinate": ["nan", 0]},
            },
            "finite numeric",
        ),
        (
            Qwen3VLDesktopActionSpace(),
            {
                "name": "computer_use",
                "arguments": {"action": "left_click", "coordinate": [1, 2, 3]},
            },
            "exactly 2",
        ),
        (
            Qwen3VLMobileActionSpace(),
            {
                "name": "mobile_use",
                "arguments": {"action": "swipe", "coordinate": [100, 200]},
            },
            "coordinate2 is required",
        ),
        (
            Qwen3VLMobileActionSpace(),
            {
                "name": "mobile_use",
                "arguments": {
                    "action": "swipe",
                    "coordinate": [100, 200],
                    "coordinate2": [float("inf"), 300],
                },
            },
            "finite numeric",
        ),
    ],
)
def test_required_coordinate_parser_boundaries_raise(space, agent_call, match) -> None:
    with pytest.raises(ModelToolCallParseError, match=match):
        space.convert_tool_calls_from_agent([agent_call])


def test_optional_scroll_coordinate_is_omitted_when_malformed() -> None:
    parsed = Qwen3VLDesktopActionSpace().convert_tool_calls_from_agent(
        [
            {
                "name": "computer_use",
                "arguments": {
                    "action": "scroll",
                    "pixels": -300,
                    "coordinate": ["nan", 0],
                },
            }
        ]
    )

    assert tool_call_arguments(parsed[0])["actions"] == [
        {"action": "scroll", "direction": "down", "amount": 3}
    ]

