"""Backend coordinate resize/projection geometry tests."""

from __future__ import annotations

import pytest

from lite.core.tools.action_space.geometry import strict_norm_to_pixel
from lite.gym.utils.backend.coordinate import norm_to_pixel as backend_norm_to_pixel

W = 1920
H = 1080


def test_strict_projection_vs_backend_differ_only_at_the_1000_edge() -> None:
    differ = [
        n
        for n in range(1001)
        if strict_norm_to_pixel([n, n], W, H, clamp=False, round=True)
        != backend_norm_to_pixel([n, n], W, H, clamp=True)
    ]
    assert differ == [1000]
    assert strict_norm_to_pixel([1000, 1000], W, H, clamp=False) == (1920, 1080)
    assert backend_norm_to_pixel([1000, 1000], W, H, clamp=True) == (1919, 1079)


def test_strict_norm_to_pixel_matches_backend_raise_mode_everywhere() -> None:
    xs = list(range(0, 1001, 25)) + [1, 3, 7, 999, 1000]
    pairs = 0
    for clamp in (True, False):
        for x in xs:
            for y in (0, 1, 499, 500, 501, 999, 1000):
                assert strict_norm_to_pixel([x, y], W, H, clamp=clamp) == (
                    backend_norm_to_pixel([x, y], W, H, clamp=clamp, on_malformed="raise")
                )
                pairs += 1
    assert pairs == 2 * len(xs) * 7 == 644


def test_backend_norm_to_pixel_clamp_semantics() -> None:
    assert backend_norm_to_pixel([1000, 1000], W, H, clamp=False) == (1920, 1080)
    assert backend_norm_to_pixel([-5, 2000], W, H) == (0, 1079)


def test_backend_malformed_policies() -> None:
    assert backend_norm_to_pixel(None, W, H) == (W // 2, H // 2)
    assert backend_norm_to_pixel(None, W, H, on_malformed="none") is None
    with pytest.raises(ValueError, match="malformed normalized coordinate"):
        backend_norm_to_pixel(None, W, H, on_malformed="raise")
    assert backend_norm_to_pixel(None, W, H, as_float=True) == (W / 2.0, H / 2.0)


def test_backend_as_float_subpixel_goldens() -> None:
    assert backend_norm_to_pixel([333, 667], W, H, as_float=True, clamp=False) == (
        639.36,
        720.36,
    )
    assert backend_norm_to_pixel([1000, 1000], W, H, as_float=True, clamp=True) == (
        1919.0,
        1079.0,
    )
