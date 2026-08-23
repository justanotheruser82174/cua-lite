"""Coordinate parsing belongs at GPT action-space parser boundaries."""

from __future__ import annotations

import pytest

from lite.agents.core.action_space.base import LiteDesktopActionSpace, LitePointActionSpace
from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.models.gpt.action_space import (
    GPTDesktopActionSpace,
    GPTDesktopGroundingPointActionSpace,
)
from lite.core.errors import LiteContractError
from lite.core.tools.calls import make_tool_call, tool_call_arguments

RESOLUTION = (1000, 1000)
NON_FINITE = [float("nan"), float("inf"), float("-inf")]


@pytest.mark.parametrize(
    ("space", "tool_call", "match"),
    [
        (
            GPTDesktopActionSpace(),
            LiteDesktopActionSpace.click(),
            "click requires coordinate",
        ),
        (
            GPTDesktopActionSpace(),
            LiteDesktopActionSpace.scroll(direction="down", amount=3),
            "scroll requires coordinate",
        ),
        (
            GPTDesktopActionSpace(),
            LiteDesktopActionSpace.drag(coordinate=[300, 400]),
            "drag requires start_coordinate",
        ),
        (
            GPTDesktopActionSpace(),
            make_tool_call("computer", {"actions": [{"action": "mouse_move"}]}),
            "mouse_move requires coordinate",
        ),
        (
            GPTDesktopGroundingPointActionSpace(),
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
        GPTDesktopActionSpace().convert_tool_calls_to_agent(
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
        GPTDesktopActionSpace().convert_tool_calls_to_agent(
            [make_tool_call("computer", arguments)],
            resolution=RESOLUTION,
        )

    assert not isinstance(excinfo.value, ModelToolCallParseError)


@pytest.mark.parametrize(
    ("space", "agent_action", "canonical"),
    [
        (
            GPTDesktopActionSpace(),
            {"type": "click", "x": 0, "y": 0},
            LiteDesktopActionSpace.click(coordinate=[0, 0]),
        ),
        (
            GPTDesktopGroundingPointActionSpace(),
            {"type": "click", "x": 0, "y": 0},
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
        GPTDesktopActionSpace().convert_tool_calls_from_agent(
            [{"type": "scroll", "scroll_y": 300}],
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
            GPTDesktopActionSpace(),
            {"type": "click"},
            "click requires x/y coordinates",
        ),
        (
            GPTDesktopActionSpace(),
            {"type": "click", "x": "bad", "y": 200},
            "click requires numeric x/y coordinates",
        ),
        (
            GPTDesktopActionSpace(),
            {"type": "scroll", "x": 100, "scroll_y": -100},
            "scroll requires numeric x/y coordinates",
        ),
        (
            GPTDesktopActionSpace(),
            {"type": "drag"},
            "drag requires start and end coordinates",
        ),
        (
            GPTDesktopGroundingPointActionSpace(),
            {"type": "click", "x": "bad", "y": 200},
            "click requires numeric x/y coordinates",
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
            GPTDesktopActionSpace(),
            lambda bad: {"type": "click", "x": bad, "y": 20},
            "click requires numeric x/y coordinates",
        ),
        (
            GPTDesktopActionSpace(),
            lambda bad: {"type": "click", "x": 10, "y": bad},
            "click requires numeric x/y coordinates",
        ),
        (
            GPTDesktopActionSpace(),
            lambda bad: {"type": "scroll", "x": 10, "y": 20, "scroll_y": bad},
            "scroll requires numeric scroll_y",
        ),
        (
            GPTDesktopActionSpace(),
            lambda bad: {
                "type": "drag",
                "start_x": bad,
                "start_y": 1,
                "end_x": 2,
                "end_y": 3,
            },
            "drag requires numeric start_x",
        ),
        (
            GPTDesktopGroundingPointActionSpace(),
            lambda bad: {"type": "click", "x": bad, "y": 20},
            "click requires numeric x/y coordinates",
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
    assert GPTDesktopActionSpace().convert_tool_calls_from_agent(
        [{"type": "click", "x": 500.4, "y": 300.6}],
        resolution=RESOLUTION,
    ) == [LiteDesktopActionSpace.click(coordinate=[500, 300])]


def test_gpt_scalar_coordinate_wire_cannot_carry_extra_values() -> None:
    """GPT's x/y scalars cannot silently carry extra coordinate values."""
    with pytest.raises(ModelToolCallParseError, match="click requires numeric x/y coordinates"):
        GPTDesktopActionSpace().convert_tool_calls_from_agent(
            [{"type": "click", "x": [10, 20, 30, 40], "y": 5}],
            resolution=RESOLUTION,
        )

