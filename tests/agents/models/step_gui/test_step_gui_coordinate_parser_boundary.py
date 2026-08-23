"""Coordinate parsing belongs at Step GUI action-space parser boundaries."""

from __future__ import annotations

import pytest

from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.models.step_gui.action_space import STEPGUIMobileActionSpace


@pytest.mark.parametrize(
    ("agent_call", "match"),
    [
        (
            {
                "name": "mobile_use",
                "arguments": {"action": "SLIDE", "point1": [100, 200]},
            },
            "point2 is required",
        ),
        (
            {
                "name": "mobile_use",
                "arguments": {"action": "CLICK", "point": ["bad", 200]},
            },
            "finite numeric",
        ),
    ],
)
def test_required_coordinate_parser_boundaries_raise(agent_call, match) -> None:
    with pytest.raises(ModelToolCallParseError, match=match):
        STEPGUIMobileActionSpace().convert_tool_calls_from_agent([agent_call])

