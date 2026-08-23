"""Tests for agent action-space coordinate parsing helpers."""

from __future__ import annotations

import pytest

import lite.agents.core.action_space.errors as action_space_errors
import lite.agents.core.action_space.utils.geometry as agent_geometry
from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.core.action_space.utils.geometry import (
    optional_coord,
    required_coord,
)


def test_agent_coordinate_parser_owner_has_no_ambiguous_coord_alias() -> None:
    assert "coord" not in agent_geometry.__all__
    assert not hasattr(agent_geometry, "coord")


def test_model_output_parse_error_is_owned_by_the_action_space_layer() -> None:
    """The parse error belongs to the action-space parse boundary that raises it."""

    assert action_space_errors.__all__ == ["ModelToolCallParseError"]
    assert agent_geometry.ModelToolCallParseError is ModelToolCallParseError
    assert "ModelToolCallParseError" not in agent_geometry.__all__


@pytest.mark.parametrize(
    "raw,expected",
    [
        ([1.9, "2"], [1, 2]),
        ("[3, 4]", [3, 4]),
        ("(5,6)", [5, 6]),
    ],
)
def test_coordinate_parsers_preserve_existing_casting(raw, expected) -> None:
    assert optional_coord(raw, dimensions=2) == expected
    assert required_coord(raw, dimensions=2) == expected


@pytest.mark.parametrize("raw", [None, ["nan", 0], [float("inf"), 0], ["bad", 0]])
def test_optional_coord_returns_none_for_missing_or_malformed(raw) -> None:
    assert optional_coord(raw, dimensions=2) is None


@pytest.mark.parametrize(
    "raw,match",
    [
        (None, "target is required"),
        (["nan", 0], "finite numeric"),
        ([1, 2, 3], "exactly 2"),
    ],
)
def test_required_coord_rejects_missing_malformed_or_wrong_shape(raw, match) -> None:
    with pytest.raises(ModelToolCallParseError, match=match):
        required_coord(raw, dimensions=2, name="target")


def test_required_coord_rejects_an_integer_too_large_to_be_a_pixel() -> None:
    with pytest.raises(ModelToolCallParseError, match="finite numeric"):
        required_coord([10**400, 0], dimensions=2)


def test_required_coord_shape_checking_is_opt_in_via_dimensions() -> None:
    assert required_coord([1, 2, 3, 4]) == [1, 2, 3, 4]


@pytest.mark.parametrize("raw", [None, ["nan", 0], [1, 2, 3], [10**400, 0]])
def test_optional_coord_never_substitutes_an_origin_default(raw) -> None:
    with pytest.raises(ModelToolCallParseError):
        required_coord(raw, dimensions=2)

    absorbed = optional_coord(raw, dimensions=2)
    assert absorbed is None
    assert absorbed != [0, 0]
