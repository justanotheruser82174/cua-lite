"""Coordinate parsing belongs at UI-TARS action-space parser boundaries."""

from __future__ import annotations

import pytest

from lite.agents.core.action_space.base import LiteBBoxActionSpace
from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.models.ui_tars.action_space import (
    UITarsBBoxActionSpace,
    UITarsDesktopActionSpace,
    UITarsDesktopGroundingPointActionSpace,
)
from lite.agents.models.ui_tars.adapter import UITarsGroundingBBoxAdapter
from lite.core.tools.calls import make_tool_call


@pytest.mark.parametrize(
    ("space", "agent_call", "match"),
    [
        (
            UITarsDesktopActionSpace(),
            {"name": "click", "arguments": {}},
            "start_box is required",
        ),
        (
            UITarsDesktopActionSpace(),
            {"name": "click", "arguments": {"start_box": ["bad", 200]}},
            "finite numeric",
        ),
        (
            UITarsDesktopActionSpace(),
            {"name": "click", "arguments": {"start_box": [10, 20, 30, 40]}},
            "exactly 2",
        ),
        (
            UITarsDesktopGroundingPointActionSpace(),
            {"name": "click", "arguments": {"start_box": [10, 20, 30, 40]}},
            "exactly 2",
        ),
        (
            UITarsDesktopGroundingPointActionSpace(),
            {"name": "click", "arguments": {"start_box": ["nan", 20]}},
            "finite numeric",
        ),
        (
            UITarsDesktopGroundingPointActionSpace(),
            {"name": "click", "arguments": {}},
            "start_box is required",
        ),
    ],
)
def test_required_coordinate_parser_boundaries_raise(space, agent_call, match) -> None:
    with pytest.raises(ModelToolCallParseError, match=match):
        space.convert_tool_calls_from_agent([agent_call])


def test_text_render_refuses_to_narrow_an_over_long_box() -> None:
    assert (
        UITarsDesktopActionSpace().format_tool_call_as_text(
            {"name": "click", "arguments": {"start_box": [10, 20, 30, 40]}}
        )
        == "click(start_box='(10,20,30,40)')"
    )


def test_text_render_bytes_are_unchanged_for_a_two_value_box() -> None:
    """These are model-facing bytes: the arity fix must not move them."""
    assert (
        UITarsDesktopActionSpace().format_tool_call_as_text(
            {"name": "click", "arguments": {"start_box": [500, 300]}}
        )
        == "click(start_box='(500,300)')"
    )


def test_bbox_renders_the_wire_click() -> None:
    """Model-facing bytes: a box leaves in the spelling the parser accepts."""
    space = UITarsBBoxActionSpace()

    assert space.format_tool_calls_as_text(
        space.convert_tool_calls_to_agent(
            [LiteBBoxActionSpace.bbox(coordinate=[10, 20, 30, 40])]
        )
    ) == "click(start_box='(10,20,30,40)')"


def test_bbox_render_parse_round_trip_is_exact() -> None:
    """Canonical -> wire text -> canonical, byte-exact and argument-preserving."""
    space = UITarsBBoxActionSpace()
    adapter = UITarsGroundingBBoxAdapter()
    canonical = LiteBBoxActionSpace.bbox(coordinate=[10, 20, 30, 40])

    text = space.format_tool_calls_as_text(
        space.convert_tool_calls_to_agent([canonical])
    )
    reparsed = adapter.convert_message_from_agent(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": f"Action: {text}"}],
        }
    )

    assert reparsed["tool_calls"] == [canonical]


def test_bbox_render_refuses_to_narrow_an_over_long_box() -> None:
    """Every value is printed, so a rejected box cannot look well-formed."""
    space = UITarsBBoxActionSpace()

    assert space.format_tool_calls_as_text(
        space.convert_tool_calls_to_agent(
            [LiteBBoxActionSpace.bbox(coordinate=[10, 20, 30, 40, 50])]
        )
    ) == "click(start_box='(10,20,30,40,50)')"


def test_bbox_render_passes_env_extras_through() -> None:
    """Mirror of the parse-side pass-through: names this space does not own."""
    assert UITarsBBoxActionSpace().convert_tool_calls_to_agent(
        [make_tool_call("report_infeasible", {"reason": "not on screen"})]
    ) == [{"name": "report_infeasible", "arguments": {"reason": "not on screen"}}]


def test_bbox_parse_truncates_fractional_box_values() -> None:
    """``required_coord`` returns ints, so a fractional box loses its fraction."""
    assert UITarsBBoxActionSpace().convert_tool_calls_from_agent(
        [{"name": "click", "arguments": {"start_box": [10.6, 20.6, 30.6, 40.6]}}]
    ) == [LiteBBoxActionSpace.bbox(coordinate=[10, 20, 30, 40])]


def test_bbox_parse_rejects_a_non_finite_box() -> None:
    with pytest.raises(ModelToolCallParseError, match="finite numeric"):
        UITarsBBoxActionSpace().convert_tool_calls_from_agent(
            [{"name": "bbox", "arguments": {"coordinate": ["nan", 0, 10, "x"]}}]
        )


def test_bbox_parse_rejects_a_missing_box() -> None:
    with pytest.raises(ModelToolCallParseError, match="coordinate is required"):
        UITarsBBoxActionSpace().convert_tool_calls_from_agent(
            [{"name": "bbox", "arguments": {"bbox": ["nan", 0, 10, "x"]}}]
        )


def test_bbox_parse_rejects_a_two_value_box() -> None:
    with pytest.raises(ModelToolCallParseError, match="exactly 4"):
        UITarsBBoxActionSpace().convert_tool_calls_from_agent(
            [{"name": "click", "arguments": {"start_box": [10, 20]}}]
        )


def test_bbox_parse_rejects_an_over_long_box() -> None:
    """Too many values must fail instead of narrowing to a well-formed box."""
    with pytest.raises(ModelToolCallParseError, match="exactly 4 numeric values; got 5"):
        UITarsBBoxActionSpace().convert_tool_calls_from_agent(
            [{"name": "click", "arguments": {"start_box": [10, 20, 30, 40, 50]}}]
        )


def test_bbox_parse_maps_the_wire_click_onto_the_advertised_verb() -> None:
    assert UITarsBBoxActionSpace().convert_tool_calls_from_agent(
        [{"name": "click", "arguments": {"start_box": [10, 20, 30, 40]}}]
    ) == [LiteBBoxActionSpace.bbox(coordinate=[10, 20, 30, 40])]


def test_bbox_parse_passes_env_extras_through() -> None:
    assert UITarsBBoxActionSpace().convert_tool_calls_from_agent(
        [{"name": "report_infeasible", "arguments": {"reason": "not on screen"}}]
    ) == [make_tool_call("report_infeasible", {"reason": "not on screen"})]

