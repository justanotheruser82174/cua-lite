"""Golden-lock tests for core action-space coordinate/numeric helpers.

Every expected value here was computed by running the live code and is
hardcoded so any silent behavior change during consolidation is caught. A
deliberate future change must update these goldens explicitly.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/core/tools/action_space/test_core_geometry.py -p no:cacheprovider -q
"""

from __future__ import annotations

import pytest

import lite.core.tools.action_space as action_space_facade
import lite.core.tools.action_space.geometry as geometry_module
from lite.core.errors import LiteContractError
from lite.core.tools.action_space.geometry import (
    pixel_to_norm,
    strict_norm_to_pixel,
)


def test_pixel_to_norm_center() -> None:
    assert pixel_to_norm(960, 540, 1920, 1080) == [500, 500]


def test_pixel_to_norm_origin() -> None:
    assert pixel_to_norm(0, 0, 1920, 1080) == [0, 0]


def test_pixel_to_norm_max() -> None:
    assert pixel_to_norm(1920, 1080, 1920, 1080) == [1000, 1000]


def test_pixel_to_norm_rounding_thirds() -> None:
    assert pixel_to_norm(1, 1, 3, 3) == [333, 333]


def test_pixel_to_norm_bankers_rounding_ties() -> None:
    assert pixel_to_norm(1, 1, 2000, 2000) == [0, 0]
    assert pixel_to_norm(3, 3, 2000, 2000) == [2, 2]
    assert pixel_to_norm(5, 5, 2000, 2000) == [2, 2]


def test_pixel_to_norm_no_clamp_over() -> None:
    assert pixel_to_norm(2000, 0, 1920, 1080) == [1042, 0]


def test_pixel_to_norm_no_clamp_negative() -> None:
    assert pixel_to_norm(-100, 0, 1920, 1080) == [-52, 0]


def test_strict_norm_to_pixel_center() -> None:
    assert strict_norm_to_pixel([500, 500], 1920, 1080, clamp=False) == (960, 540)


def test_strict_norm_to_pixel_clamp_policy_is_explicit() -> None:
    assert strict_norm_to_pixel([500, 500], 1920, 1080, clamp=True) == (960, 540)
    assert strict_norm_to_pixel([1000, 1000], 1920, 1080, clamp=True) == (1919, 1079)
    assert strict_norm_to_pixel([1000, 1000], 1920, 1080, clamp=False) == (1920, 1080)
    with pytest.raises(LiteContractError):
        strict_norm_to_pixel(None, 1920, 1080, clamp=True)


def test_strict_norm_to_pixel_requires_clamp_argument() -> None:
    with pytest.raises(TypeError):
        strict_norm_to_pixel([500, 500], 1920, 1080)  # type: ignore[call-arg]


@pytest.mark.parametrize("bad", [["nan", 0], [float("inf"), 0], ["bad", 0]])
def test_strict_norm_to_pixel_rejects_non_finite_values(bad) -> None:
    for rounding in (True, False):
        with pytest.raises(
            LiteContractError, match="coordinate values must be finite numbers"
        ):
            strict_norm_to_pixel(bad, 1920, 1080, clamp=True, round=rounding)


def test_projection_helpers_are_geometry_owner_contracts() -> None:
    assert "strict_norm_to_pixel" in geometry_module.__all__
    assert "norm_coord_to_pixel" not in geometry_module.__all__
    assert "norm_xy_to_pixel" not in geometry_module.__all__
    assert not hasattr(geometry_module, "norm_coord_to_pixel")
    assert not hasattr(geometry_module, "norm_xy_to_pixel")

    assert "strict_norm_to_pixel" not in action_space_facade.__all__
    assert "norm_coord_to_pixel" not in action_space_facade.__all__
    assert "norm_xy_to_pixel" not in action_space_facade.__all__
    assert not hasattr(action_space_facade, "strict_norm_to_pixel")
    assert not hasattr(action_space_facade, "norm_coord_to_pixel")
    assert not hasattr(action_space_facade, "norm_xy_to_pixel")
