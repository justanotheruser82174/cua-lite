"""Coordinate parsing belongs at Claude action-space parser boundaries."""

from __future__ import annotations

import pytest

from lite.agents.core.action_space.base import (
    LiteDesktopActionSpace,
    LitePointActionSpace,
)
from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.models.claude.action_space import (
    ClaudeDesktopActionSpace,
    ClaudeDesktopGroundingPointActionSpace,
    ClaudeMobileActionSpace,
)
from lite.core.errors import LiteContractError
from lite.core.tools.calls import make_tool_call, tool_call_arguments

RESOLUTION = (1000, 1000)
NON_FINITE = [float("nan"), float("inf"), float("-inf")]


@pytest.mark.parametrize(
    ("space", "tool_call", "match"),
    [
        (
            ClaudeDesktopActionSpace(),
            LiteDesktopActionSpace.click(),
            "click requires coordinate",
        ),
        (
            ClaudeDesktopActionSpace(),
            LiteDesktopActionSpace.scroll(direction="down", amount=3),
            "scroll requires coordinate",
        ),
        (
            ClaudeDesktopActionSpace(),
            LiteDesktopActionSpace.drag(coordinate=[300, 400]),
            "drag requires start_coordinate",
        ),
        (
            ClaudeDesktopActionSpace(),
            make_tool_call("computer", {"actions": [{"action": "mouse_move"}]}),
            "mouse_move requires coordinate",
        ),
        (
            ClaudeDesktopGroundingPointActionSpace(),
            make_tool_call("point", {}),
            "point requires coordinate",
        ),
    ],
)
def test_required_coordinate_replay_boundaries_raise(space, tool_call, match) -> None:
    with pytest.raises(ValueError, match=match):
        space.convert_tool_calls_to_agent([tool_call], resolution=RESOLUTION)


@pytest.mark.parametrize(
    ("coordinate", "error", "match"),
    [
        ([10], ValueError, "click requires numeric coordinate"),
        ([10, 20, 30, 40], LiteContractError, "malformed normalized coordinate"),
        (["nan", 20], ValueError, "click requires numeric coordinate"),
        ([float("inf"), 20], ValueError, "click requires numeric coordinate"),
    ],
)
def test_replay_rejects_a_malformed_canonical_coordinate(
    coordinate, error, match
) -> None:
    """Replay's own arity/finiteness half, not just missing coordinates."""
    click = {"actions": [{"action": "click", "coordinate": coordinate}]}
    with pytest.raises(error, match=match):
        ClaudeDesktopActionSpace().convert_tool_calls_to_agent(
            [make_tool_call("computer", click)],
            resolution=RESOLUTION,
        )


@pytest.mark.parametrize(
    "arguments",
    [
        {"actions": [{"action": "click"}]},
        {"actions": [{"action": "click", "coordinate": [10, 20, 30, 40]}]},
        {"actions": [{"action": "click", "coordinate": ["nan", 20]}]},
    ],
)
def test_replay_failure_is_not_reported_as_malformed_model_output(arguments) -> None:
    """Replay failures must not be ``ModelToolCallParseError``."""
    with pytest.raises(Exception) as excinfo:  # noqa: B017 - the type is the assertion
        ClaudeDesktopActionSpace().convert_tool_calls_to_agent(
            [make_tool_call("computer", arguments)],
            resolution=RESOLUTION,
        )

    assert not isinstance(excinfo.value, ModelToolCallParseError)


@pytest.mark.parametrize(
    ("space", "agent_action", "canonical"),
    [
        (
            ClaudeDesktopActionSpace(),
            {"action": "left_click", "coordinate": [0, 0]},
            LiteDesktopActionSpace.click(coordinate=[0, 0]),
        ),
        (
            ClaudeDesktopGroundingPointActionSpace(),
            {"action": "left_click", "coordinate": [0, 0]},
            LitePointActionSpace.point(coordinate=[0, 0]),
        ),
    ],
)
def test_the_origin_stays_a_legal_coordinate_in_both_directions(
    space, agent_action, canonical
) -> None:
    """``[0, 0]`` is real model output, not a missing-coordinate sentinel."""
    parsed = space.convert_tool_calls_from_agent([agent_action], resolution=RESOLUTION)
    assert parsed == [canonical]

    assert space.convert_tool_calls_to_agent(parsed, resolution=RESOLUTION) == [
        agent_action
    ]


def test_an_omitted_optional_coordinate_leaves_no_coordinate_key() -> None:
    """Absent means absent, not ``None`` and not the origin."""
    action = tool_call_arguments(
        ClaudeDesktopActionSpace().convert_tool_calls_from_agent(
            [
                {
                    "action": "scroll",
                    "scroll_direction": "down",
                    "scroll_amount": 3,
                }
            ],
            resolution=RESOLUTION,
        )[0]
    )["actions"][0]

    assert "coordinate" not in action
    assert action.get("coordinate") is None
    assert action.get("coordinate") != [0, 0]


@pytest.mark.parametrize(
    ("space", "agent_action", "match"),
    [
        (
            ClaudeDesktopActionSpace(),
            {"action": "left_click"},
            "left_click requires coordinate",
        ),
        (
            ClaudeDesktopActionSpace(),
            {"action": "left_click", "coordinate": ["bad", 200]},
            "left_click requires valid coordinate",
        ),
        (
            ClaudeDesktopActionSpace(),
            {"action": "left_click", "coordinate": [100]},
            "left_click requires valid coordinate",
        ),
        (
            ClaudeDesktopActionSpace(),
            {"action": "mouse_move"},
            "mouse_move requires coordinate",
        ),
        (
            ClaudeDesktopGroundingPointActionSpace(),
            {"action": "left_click", "coordinate": ["bad", 200]},
            "left_click requires valid coordinate",
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
            ClaudeDesktopActionSpace(),
            lambda bad: {"action": "left_click", "coordinate": [bad, 20]},
            "left_click requires valid coordinate",
        ),
        (
            ClaudeDesktopActionSpace(),
            lambda bad: {"action": "left_click", "coordinate": [10, bad]},
            "left_click requires valid coordinate",
        ),
        (
            ClaudeDesktopActionSpace(),
            lambda bad: {
                "action": "scroll",
                "coordinate": [10, 20],
                "scroll_amount": bad,
            },
            "scroll requires numeric scroll_amount",
        ),
        (
            ClaudeDesktopActionSpace(),
            lambda bad: {"action": "wait", "duration": bad},
            "wait requires numeric duration",
        ),
        (
            ClaudeDesktopGroundingPointActionSpace(),
            lambda bad: {"action": "left_click", "coordinate": [bad, 20]},
            "left_click requires valid coordinate",
        ),
        (
            ClaudeMobileActionSpace(),
            lambda bad: {"name": "tap", "arguments": {"coordinate": [bad, 20]}},
            "malformed Claude mobile arguments for tap",
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
    assert ClaudeDesktopActionSpace().convert_tool_calls_from_agent(
        [{"action": "left_click", "coordinate": [500.4, 300.6]}],
        resolution=RESOLUTION,
    ) == [LiteDesktopActionSpace.click(coordinate=[500, 301])]


@pytest.mark.parametrize("coordinate", [[10], [10, 20, 30], [10, 20, 30, 40]])
@pytest.mark.parametrize(
    ("space", "agent_action_for", "match"),
    [
        (
            ClaudeDesktopActionSpace(),
            lambda coordinate: {"action": "left_click", "coordinate": coordinate},
            "left_click requires valid coordinate",
        ),
        (
            ClaudeDesktopActionSpace(),
            lambda coordinate: {"action": "mouse_move", "coordinate": coordinate},
            "mouse_move requires valid coordinate",
        ),
        (
            ClaudeDesktopActionSpace(),
            lambda coordinate: {
                "action": "left_click_drag",
                "start_coordinate": coordinate,
                "end_coordinate": [40, 50],
            },
            "left_click_drag requires valid start_coordinate",
        ),
        (
            ClaudeDesktopGroundingPointActionSpace(),
            lambda coordinate: {"action": "left_click", "coordinate": coordinate},
            "left_click requires valid coordinate",
        ),
        (
            ClaudeMobileActionSpace(),
            lambda coordinate: {"name": "tap", "arguments": {"coordinate": coordinate}},
            "malformed Claude mobile arguments for tap",
        ),
    ],
)
def test_claude_wrong_arity_coordinate_is_not_narrowed(
    space, agent_action_for, match, coordinate
) -> None:
    with pytest.raises(ModelToolCallParseError, match=match):
        space.convert_tool_calls_from_agent(
            [agent_action_for(coordinate)],
            resolution=RESOLUTION,
        )


def test_claude_mouse_half_press_omits_coordinate_where_native_allows_it() -> None:
    space = ClaudeDesktopActionSpace()

    assert space.convert_tool_calls_to_agent(
        [LiteDesktopActionSpace.mouse_down()],
        resolution=RESOLUTION,
    ) == [{"action": "left_mouse_down"}]
    assert space.convert_tool_calls_to_agent(
        [LiteDesktopActionSpace.mouse_up()],
        resolution=RESOLUTION,
    ) == [{"action": "left_mouse_up"}]
