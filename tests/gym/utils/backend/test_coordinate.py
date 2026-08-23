"""Unit tests for the one norm→pixel conversion.

Run: uv run pytest tests/gym/utils/backend/test_coordinate.py
"""
from __future__ import annotations

import pytest

from lite.gym.utils.backend.coordinate import norm_to_pixel, parse_norm_pair_values

W, H = 1920, 1080


def test_boundaries_round_and_clamp():
    assert norm_to_pixel([0, 0], W, H) == (0, 0)
    assert norm_to_pixel([500, 500], W, H) == (960, 540)
    # 1000 → w would be off-canvas; the canonical contract clamps to w-1.
    assert norm_to_pixel([1000, 1000], W, H) == (W - 1, H - 1)
    # negative / overshoot clamp too
    assert norm_to_pixel([-5, 2000], W, H) == (0, H - 1)


def test_no_clamp_reproduces_legacy_off_canvas():
    assert norm_to_pixel([1000, 1000], W, H, clamp=False) == (W, H)


def test_rounding_is_round_not_truncate():
    # 999/1000*1920 = 1918.08 → 1918; 501/1000*1080 = 541.08 → 541
    assert norm_to_pixel([999, 501], W, H, clamp=False) == (1918, 541)
    # .5 boundary uses python round (banker's) — pin it so it can't drift
    assert norm_to_pixel([625, 625], 1000, 1000, clamp=False) == (625, 625)


def test_as_float_subpixel():
    fx, fy = norm_to_pixel([333, 667], W, H, as_float=True, clamp=False)
    assert abs(fx - 639.36) < 1e-9 and abs(fy - 720.36) < 1e-9
    assert isinstance(fx, float)


def test_parse_norm_pair_values_accepts_shaped_numeric_pairs():
    assert parse_norm_pair_values(["1", "2.5"]) == (1.0, 2.5)
    assert parse_norm_pair_values((0, 1000)) == (0.0, 1000.0)


@pytest.mark.parametrize(
    "bad",
    [
        ["x", 0],
        [0, object()],
        [10 ** 5000, 0],
        [float("nan"), 0],
        [0, float("inf")],
    ],
)
def test_parse_norm_pair_values_rejects_invalid_values(bad):
    with pytest.raises(ValueError, match="coordinate values must be finite numbers"):
        parse_norm_pair_values(bad)


def test_malformed_policies():
    for bad in (None, [], [5], "xy", 7):
        assert norm_to_pixel(bad, W, H, on_malformed="center") == (W // 2, H // 2)
        assert norm_to_pixel(bad, W, H, on_malformed="none") is None
        with pytest.raises(ValueError):
            norm_to_pixel(bad, W, H, on_malformed="raise")
    # float-mode center
    assert norm_to_pixel(None, W, H, as_float=True, on_malformed="center") == (W / 2.0, H / 2.0)


def test_non_numeric_elements_raise_never_center():
    with pytest.raises((TypeError, ValueError)):
        norm_to_pixel(["a", "b"], W, H)
