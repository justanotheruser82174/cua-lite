"""Coordinate parsing belongs at UI-TARS 1.5 action-space parser boundaries."""

from __future__ import annotations

import pytest
from PIL import Image

from lite.agents.core.action_space.base import LiteBBoxActionSpace, LiteDesktopActionSpace
from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.models.ui_tars_15_v1.action_space import (
    UITars15V1BBoxActionSpace,
    UITars15V1DesktopActionSpace,
    UITars15V1DesktopGroundingPointActionSpace,
)
from lite.agents.models.ui_tars_15_v1.adapter import (
    UITars15V1DesktopUseAdapter,
    UITars15V1GroundingBBoxAdapter,
)
from lite.core.tools.calls import make_tool_call


@pytest.mark.parametrize(
    ("space", "agent_call", "match"),
    [
        (
            UITars15V1DesktopActionSpace(),
            {"name": "click", "arguments": {"start_box": [10, 20, 30, 40]}},
            "exactly 2",
        ),
        (
            UITars15V1DesktopActionSpace(),
            {"name": "drag", "arguments": {"start_box": [10, 20], "end_box": None}},
            "end_box is required",
        ),
        (
            UITars15V1DesktopGroundingPointActionSpace(),
            {"name": "click", "arguments": {"start_box": [10, 20, 30, 40]}},
            "exactly 2",
        ),
        (
            UITars15V1DesktopGroundingPointActionSpace(),
            {"name": "click", "arguments": {"start_box": ["nan", 20]}},
            "finite numeric",
        ),
        (
            UITars15V1DesktopGroundingPointActionSpace(),
            {"name": "click", "arguments": {}},
            "start_box is required",
        ),
    ],
)
def test_required_coordinate_parser_boundaries_raise(space, agent_call, match) -> None:
    with pytest.raises(ModelToolCallParseError, match=match):
        space.convert_tool_calls_from_agent([agent_call])


def test_render_keeps_extra_long_box_intact() -> None:
    """The v1 render projection must not shorten an over-long box."""
    projected = UITars15V1DesktopActionSpace().convert_tool_calls_to_agent(
        [LiteDesktopActionSpace.click(coordinate=[10, 20, 30, 40])]
    )

    assert projected == [
        {"name": "click", "arguments": {"start_box": [10, 20, 30, 40]}}
    ]


def test_parse_hands_extra_long_box_to_the_parser() -> None:
    """End-to-end v1 parse: the action-space parser owns the failure."""
    adapter = UITars15V1DesktopUseAdapter()
    message = {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Thought: click it\nAction: click(start_box='(10,20,30,40)')",
            }
        ],
    }

    with pytest.raises(ModelToolCallParseError, match="exactly 2"):
        adapter.convert_message_from_agent(message)


def test_adapter_render_leaves_an_over_long_box_undenormalized() -> None:
    """The v1 render path leaves malformed boxes visibly malformed."""
    adapter = UITars15V1DesktopUseAdapter()
    adapter._process_image_after_target(Image.new("RGB", (1920, 1080)))
    sr_w, sr_h = adapter._last_smart_resize_wh

    def render(coordinate: list[int]) -> str:
        rendered = adapter._convert_message_to_agent(
            {
                "role": "assistant",
                "tool_calls": [LiteDesktopActionSpace.click(coordinate=coordinate)],
            }
        )
        return rendered["content"][0]["text"]

    denormalized = f"({int(500 / 1000 * sr_w)},{int(300 / 1000 * sr_h)})"
    assert render([500, 300]) == f"Action: click(start_box='{denormalized}')"
    assert denormalized != "(500,300)"

    assert render([10, 20, 30, 40]) == "Action: click(start_box='(10,20,30,40)')"


def test_text_render_refuses_to_narrow_an_over_long_box() -> None:
    assert (
        UITars15V1DesktopActionSpace().format_tool_call_as_text(
            {"name": "click", "arguments": {"start_box": [10, 20, 30, 40]}}
        )
        == "click(start_box='(10,20,30,40)')"
    )


def test_text_render_bytes_are_unchanged_for_a_two_value_box() -> None:
    """These are model-facing bytes: the arity fix must not move them."""
    assert (
        UITars15V1DesktopActionSpace().format_tool_call_as_text(
            {"name": "click", "arguments": {"start_box": [500, 300]}}
        )
        == "click(start_box='(500,300)')"
    )


def test_bbox_renders_the_wire_click() -> None:
    """Model-facing bytes: a box leaves in the spelling the parser accepts."""
    space = UITars15V1BBoxActionSpace()

    assert space.format_tool_calls_as_text(
        space.convert_tool_calls_to_agent(
            [LiteBBoxActionSpace.bbox(coordinate=[10, 20, 30, 40])]
        )
    ) == "click(start_box='(10,20,30,40)')"


def test_bbox_render_parse_round_trip_is_exact() -> None:
    """Canonical -> wire text -> canonical, byte-exact and argument-preserving."""
    space = UITars15V1BBoxActionSpace()
    adapter = UITars15V1GroundingBBoxAdapter()
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
    space = UITars15V1BBoxActionSpace()

    assert space.format_tool_calls_as_text(
        space.convert_tool_calls_to_agent(
            [LiteBBoxActionSpace.bbox(coordinate=[10, 20, 30, 40, 50])]
        )
    ) == "click(start_box='(10,20,30,40,50)')"


def test_bbox_render_passes_env_extras_through() -> None:
    """Mirror of the parse-side pass-through: names this space does not own."""
    assert UITars15V1BBoxActionSpace().convert_tool_calls_to_agent(
        [make_tool_call("report_infeasible", {"reason": "not on screen"})]
    ) == [{"name": "report_infeasible", "arguments": {"reason": "not on screen"}}]


def test_bbox_parse_truncates_fractional_box_values() -> None:
    """``required_coord`` returns ints, so a fractional box loses its fraction."""
    assert UITars15V1BBoxActionSpace().convert_tool_calls_from_agent(
        [{"name": "click", "arguments": {"start_box": [10.6, 20.6, 30.6, 40.6]}}]
    ) == [LiteBBoxActionSpace.bbox(coordinate=[10, 20, 30, 40])]


def test_bbox_parse_rejects_a_non_finite_box() -> None:
    with pytest.raises(ModelToolCallParseError, match="finite numeric"):
        UITars15V1BBoxActionSpace().convert_tool_calls_from_agent(
            [{"name": "bbox", "arguments": {"coordinate": ["nan", 0, 10, "x"]}}]
        )


def test_bbox_parse_rejects_a_missing_box() -> None:
    with pytest.raises(ModelToolCallParseError, match="coordinate is required"):
        UITars15V1BBoxActionSpace().convert_tool_calls_from_agent(
            [{"name": "bbox", "arguments": {"bbox": ["nan", 0, 10, "x"]}}]
        )


def test_bbox_parse_rejects_a_two_value_box() -> None:
    with pytest.raises(ModelToolCallParseError, match="exactly 4"):
        UITars15V1BBoxActionSpace().convert_tool_calls_from_agent(
            [{"name": "click", "arguments": {"start_box": [10, 20]}}]
        )


def test_bbox_parse_rejects_an_over_long_box() -> None:
    """Too many values must fail instead of narrowing to a well-formed box."""
    with pytest.raises(ModelToolCallParseError, match="exactly 4 numeric values; got 5"):
        UITars15V1BBoxActionSpace().convert_tool_calls_from_agent(
            [{"name": "click", "arguments": {"start_box": [10, 20, 30, 40, 50]}}]
        )


def test_bbox_parse_maps_the_wire_click_onto_the_advertised_verb() -> None:
    assert UITars15V1BBoxActionSpace().convert_tool_calls_from_agent(
        [{"name": "click", "arguments": {"start_box": [10, 20, 30, 40]}}]
    ) == [LiteBBoxActionSpace.bbox(coordinate=[10, 20, 30, 40])]


def test_bbox_parse_passes_env_extras_through() -> None:
    assert UITars15V1BBoxActionSpace().convert_tool_calls_from_agent(
        [{"name": "report_infeasible", "arguments": {"reason": "not on screen"}}]
    ) == [make_tool_call("report_infeasible", {"reason": "not on screen"})]

