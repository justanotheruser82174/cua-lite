"""Canonical image resize helper geometry."""

from __future__ import annotations

import inspect

import pytest

from lite.utils.image import ceil_by_factor, floor_by_factor, round_by_factor, smart_resize

_SCREEN_SIZES = [
    (1080, 1920),
    (2160, 3840),
    (720, 1280),
    (1280, 800),
    (1680, 1050),
    (2560, 1440),
    (1080, 2560),
    (1440, 3440),
    (768, 1366),
    (1200, 1920),
    (1800, 2880),
    (768, 1024),
    (1080, 3840),
    (2400, 3840),
    (1440, 2560),
]
_CANONICAL_RESIZE_GOLDENS = {
    (1080, 1920): (728, 1316),
    (2160, 3840): (728, 1316),
    (720, 1280): (728, 1288),
    (1280, 800): (1260, 784),
    (1680, 1050): (1260, 784),
    (2560, 1440): (1316, 728),
    (1080, 2560): (644, 1540),
    (1440, 3440): (644, 1540),
    (768, 1366): (728, 1316),
    (1200, 1920): (784, 1260),
    (1800, 2880): (784, 1260),
    (768, 1024): (756, 1036),
    (1080, 3840): (504, 1876),
    (2400, 3840): (784, 1260),
    (1440, 2560): (728, 1316),
}


def test_canonical_defaults_reproduce_the_canonical_golden_grid() -> None:
    actual = {(h, w): smart_resize(h, w) for h, w in _SCREEN_SIZES}
    assert actual == _CANONICAL_RESIZE_GOLDENS


def test_by_factor_helpers() -> None:
    assert round_by_factor(1080, 28) == 1092
    assert round_by_factor(1078, 28) == 1064
    assert ceil_by_factor(1080, 28) == 1092
    assert floor_by_factor(1080, 28) == 1064
    assert round_by_factor(1080, 32) == 1088
    assert ceil_by_factor(1080, 32) == 1088
    assert floor_by_factor(1080, 32) == 1056


def test_canonical_signature_defaults() -> None:
    defaults = {
        p.name: p.default
        for p in inspect.signature(smart_resize).parameters.values()
        if p.default is not inspect.Parameter.empty
    }
    assert defaults == {
        "factor": 28,
        "min_pixels": 3136,
        "max_pixels": 1_003_520,
        "max_long_side": 8192,
    }


@pytest.mark.parametrize(
    ("h", "w", "message"),
    [
        (1, 100, "height:1 or width:100 must be >= 2"),
        (100, 1, "height:100 or width:1 must be >= 2"),
        (1, 1, "height:1 or width:1 must be >= 2"),
        (0, 0, "height:0 or width:0 must be >= 2"),
        (-5, 500, "height:-5 or width:500 must be >= 2"),
    ],
)
def test_canonical_min_dimension_guard(h: int, w: int, message: str) -> None:
    with pytest.raises(ValueError, match=r"must be >= 2") as exc:
        smart_resize(h, w)
    assert str(exc.value) == message


@pytest.mark.parametrize(
    ("h", "w", "message"),
    [
        (1000, 2, "absolute aspect ratio must be < 200, got 1000/2"),
        (20, 4001, "absolute aspect ratio must be < 200, got 20/4001"),
        (3, 700, "absolute aspect ratio must be < 200, got 3/700"),
    ],
)
def test_canonical_aspect_ratio_guard(h: int, w: int, message: str) -> None:
    with pytest.raises(ValueError, match=r"absolute aspect ratio must be < 200") as exc:
        smart_resize(h, w)
    assert str(exc.value) == message


def test_canonical_aspect_ratio_guard_is_strictly_greater_than_200() -> None:
    assert smart_resize(10, 2000, min_pixels=3136, max_pixels=12_845_056) == (28, 812)
    with pytest.raises(ValueError, match=r"absolute aspect ratio"):
        smart_resize(10, 2010)


def test_canonical_long_side_guard_rescales_it_does_not_raise() -> None:
    assert smart_resize(9000, 1000, min_pixels=3136, max_pixels=12_845_056) == (8204, 896)
    assert 8204 > 8192
    assert smart_resize(10000, 10000, min_pixels=3136, max_pixels=12_845_056) == (3584, 3584)
    assert smart_resize(8193, 8193, min_pixels=3136, max_pixels=12_845_056) == (3584, 3584)
    assert smart_resize(8192, 8192, min_pixels=3136, max_pixels=12_845_056) == (3584, 3584)


def test_canonical_long_side_knob_is_honoured() -> None:
    assert smart_resize(4000, 4000, min_pixels=3136, max_pixels=12_845_056) == (3584, 3584)
    assert smart_resize(4000, 4000, min_pixels=3136, max_pixels=12_845_056, max_long_side=1024) == (
        1036,
        1036,
    )
    assert smart_resize(4000, 4000, min_pixels=3136, max_pixels=12_845_056, max_long_side=2048) == (
        2044,
        2044,
    )


def test_canonical_min_pixels_branch_scales_up() -> None:
    assert smart_resize(2, 2) == (56, 56)
    assert smart_resize(5, 5) == (56, 56)
    assert smart_resize(10, 20) == (56, 84)
