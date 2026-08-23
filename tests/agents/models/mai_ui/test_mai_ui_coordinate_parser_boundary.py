"""Coordinate parsing belongs at MAI UI action-space parser boundaries."""

from __future__ import annotations

import pytest

from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.models.mai_ui.action_space import MAIUIMobileActionSpace
from lite.core.tools.calls import tool_call_arguments


@pytest.mark.parametrize(
    ("agent_call", "match"),
    [
        (
            {"name": "mobile_use", "arguments": {"action": "click"}},
            "coordinate is required",
        ),
        (
            {
                "name": "mobile_use",
                "arguments": {"action": "click", "coordinate": ["nan", 0]},
            },
            "finite numeric",
        ),
        (
            {
                "name": "mobile_use",
                "arguments": {"action": "click", "coordinate": [1, 2, 3]},
            },
            "exactly 2 or 4",
        ),
        (
            {
                "name": "mobile_use",
                "arguments": {
                    "action": "drag",
                    "start_coordinate": [100, 200],
                },
            },
            "end_coordinate is required",
        ),
        (
            {
                "name": "mobile_use",
                "arguments": {
                    "action": "drag",
                    "start_coordinate": [100, 200],
                    "end_coordinate": [float("inf"), 300],
                },
            },
            "finite numeric",
        ),
    ],
)
def test_required_coordinate_parser_boundaries_raise(agent_call, match) -> None:
    with pytest.raises(ModelToolCallParseError, match=match):
        MAIUIMobileActionSpace().convert_tool_calls_from_agent([agent_call])


def test_optional_swipe_anchor_is_omitted_when_malformed() -> None:
    parsed = MAIUIMobileActionSpace().convert_tool_calls_from_agent(
        [
            {
                "name": "mobile_use",
                "arguments": {
                    "action": "swipe",
                    "direction": "down",
                    "coordinate": ["nan", 0],
                },
            }
        ]
    )

    assert tool_call_arguments(parsed[0])["actions"] == [
        {"action": "swipe", "start_coordinate": [500, 500], "coordinate": [500, 800]}
    ]

