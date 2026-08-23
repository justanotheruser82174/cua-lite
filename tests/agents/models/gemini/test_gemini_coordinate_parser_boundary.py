"""Coordinate parsing belongs at Gemini action-space parser boundaries."""

from __future__ import annotations

import pytest

from lite.agents.core.action_space.base import LiteDesktopActionSpace, LiteMobileActionSpace
from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.models.gemini.action_space import (
    GeminiDesktopActionSpace,
    GeminiMobileActionSpace,
)

RESOLUTION = (1000, 1000)
NON_FINITE = [float("nan"), float("inf"), float("-inf")]


@pytest.mark.parametrize(
    ("space", "agent_action", "match"),
    [
        (
            GeminiDesktopActionSpace(),
            {"name": "click", "arguments": {"x": "bad", "y": 200}},
            "click requires numeric x",
        ),
        (
            GeminiDesktopActionSpace(),
            {"name": "scroll", "arguments": {"x": 100, "direction": "down"}},
            "scroll requires numeric y",
        ),
        (
            GeminiMobileActionSpace(),
            {"name": "click", "arguments": {"x": "bad", "y": 200}},
            "click requires numeric x",
        ),
        (
            GeminiMobileActionSpace(),
            {
                "name": "drag_and_drop",
                "arguments": {
                    "start_x": 1,
                    "start_y": 2,
                    "end_x": "bad",
                    "end_y": 4,
                },
            },
            "drag_and_drop requires numeric end_x",
        ),
    ],
)
def test_provider_parse_coordinate_boundaries_raise(space, agent_action, match) -> None:
    with pytest.raises(ModelToolCallParseError, match=match):
        space.convert_tool_calls_from_agent([agent_action], resolution=RESOLUTION)


@pytest.mark.parametrize("bad", NON_FINITE)
@pytest.mark.parametrize(
    ("space", "agent_action_for", "match"),
    [
        (
            GeminiDesktopActionSpace(),
            lambda bad: {"name": "click", "arguments": {"x": bad, "y": 20}},
            "click requires numeric x",
        ),
        (
            GeminiMobileActionSpace(),
            lambda bad: {"name": "click", "arguments": {"x": 10, "y": bad}},
            "click requires numeric y",
        ),
    ],
)
def test_non_finite_provider_number_is_a_parse_error(
    space, agent_action_for, match, bad
) -> None:
    with pytest.raises(ModelToolCallParseError, match=match):
        space.convert_tool_calls_from_agent([agent_action_for(bad)], resolution=RESOLUTION)


def test_finite_float_coordinates_still_parse() -> None:
    """The finite check must reject only non-finite values, not floats."""
    assert GeminiDesktopActionSpace().convert_tool_calls_from_agent(
        [{"name": "click", "arguments": {"x": 500.4, "y": 300.6}}],
        resolution=RESOLUTION,
    ) == [LiteDesktopActionSpace.click(coordinate=[500, 300])]
    assert GeminiMobileActionSpace().convert_tool_calls_from_agent(
        [{"name": "click", "arguments": {"x": 500.4, "y": 300.6}}],
        resolution=RESOLUTION,
    ) == [LiteMobileActionSpace.tap(coordinate=[500, 300])]

