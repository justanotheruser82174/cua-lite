"""Core normalized-coordinate geometry tests."""

from __future__ import annotations

import pytest

from lite.core.tools.action_space.geometry import pixel_to_norm, strict_norm_to_pixel

W = 1920
H = 1080
_NORM_TO_PIXEL_DIAGONAL = {
    0: ((0, 0), (0, 0)),
    1: ((2, 1), (1, 1)),
    250: ((480, 270), (480, 270)),
    333: ((639, 360), (639, 359)),
    500: ((960, 540), (960, 540)),
    501: ((962, 541), (961, 541)),
    625: ((1200, 675), (1200, 675)),
    750: ((1440, 810), (1440, 810)),
    999: ((1918, 1079), (1918, 1078)),
    1000: ((1920, 1080), (1920, 1080)),
}


@pytest.mark.parametrize(("n", "want"), sorted(_NORM_TO_PIXEL_DIAGONAL.items()))
def test_strict_norm_to_pixel_both_rounding_modes(
    n: int, want: tuple[tuple[int, int], tuple[int, int]]
) -> None:
    rounded, truncated = want
    assert strict_norm_to_pixel([n, n], W, H, clamp=False, round=True) == rounded
    assert strict_norm_to_pixel([n, n], W, H, clamp=False, round=False) == truncated


def test_truncating_mode_differs_from_rounding_on_960_of_1001() -> None:
    differ = [
        n
        for n in range(1001)
        if strict_norm_to_pixel([n, n], W, H, clamp=False, round=False)
        != strict_norm_to_pixel([n, n], W, H, clamp=False, round=True)
    ]
    assert len(differ) == 960
    assert 0 not in differ and 500 not in differ and 1000 not in differ
    assert differ[0] == 1 and differ[-1] == 999


def test_strict_norm_to_pixel_clamp_semantics() -> None:
    assert strict_norm_to_pixel([1000, 1000], W, H, clamp=True) == (1919, 1079)
    assert strict_norm_to_pixel([1000, 1000], W, H, clamp=False) == (1920, 1080)
    assert strict_norm_to_pixel([-5, 2000], W, H, clamp=True) == (0, 1079)
    assert strict_norm_to_pixel([-5, 2000], W, H, clamp=False) == (-10, 2160)


def test_round_trip_norm_to_pixel_to_norm_is_lossless_when_rounding() -> None:
    bad = [
        n
        for n in range(1001)
        if pixel_to_norm(*strict_norm_to_pixel([n, n], W, H, clamp=False, round=True), W, H)
        != [n, n]
    ]
    assert bad == []


def test_round_trip_is_lossy_when_truncating() -> None:
    bad = {
        n: pixel_to_norm(*strict_norm_to_pixel([n, n], W, H, clamp=False, round=False), W, H)
        for n in range(1001)
        if pixel_to_norm(*strict_norm_to_pixel([n, n], W, H, clamp=False, round=False), W, H)
        != [n, n]
    }
    assert len(bad) == 460
    assert min(bad) == 7 and max(bad) == 999
    assert bad[7] == [7, 6]
    assert bad[999] == [999, 998]
    assert all(0 <= n - back[1] <= 1 and 0 <= n - back[0] <= 1 for n, back in bad.items())


def test_pixel_to_norm_goldens() -> None:
    assert pixel_to_norm(0, 0, W, H) == [0, 0]
    assert pixel_to_norm(960, 540, W, H) == [500, 500]
    assert pixel_to_norm(1919, 1079, W, H) == [999, 999]
    assert pixel_to_norm(1920, 1080, W, H) == [1000, 1000]
    assert pixel_to_norm(1, 1, W, H) == [1, 1]


def test_strict_norm_to_pixel_rejects_malformed() -> None:
    from lite.core.errors import LiteContractError

    for bad in (None, "nope", [], [1], [1, 2, 3], 5):
        with pytest.raises(LiteContractError, match="malformed normalized coordinate"):
            strict_norm_to_pixel(bad, W, H, clamp=True)
