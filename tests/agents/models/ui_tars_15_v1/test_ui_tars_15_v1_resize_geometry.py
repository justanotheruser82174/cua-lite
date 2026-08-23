"""UI-TARS 1.5 resize geometry tests."""

from __future__ import annotations

import inspect

import pytest

from lite.utils.image import smart_resize

_BUDGET = (28, 3136, 12845056)
_UT_CONSTANTS = {"factor": 28, "min_pixels": 3136, "max_pixels": 12_845_056}
_AGREE = {
    (1080, 1920): (1092, 1932),
    (2160, 3840): (2156, 3836),
    (720, 1280): (728, 1288),
    (1280, 800): (1288, 812),
    (1680, 1050): (1680, 1064),
    (2560, 1440): (2548, 1428),
    (1080, 2560): (1092, 2548),
    (1440, 3440): (1428, 3444),
    (768, 1366): (756, 1372),
    (768, 1024): (756, 1036),
}
_CONSOLIDATED = [
    ((1, 100), (ValueError, "height:1 or width:100 must be >= 2"), (28, 112)),
    ((100, 1), (ValueError, "height:100 or width:1 must be >= 2"), (112, 28)),
    ((1, 1), (ValueError, "height:1 or width:1 must be >= 2"), (56, 56)),
    ((1000, 2), (ValueError, "absolute aspect ratio must be < 200, got 1000/2"), (1008, 28)),
    ((20, 4001), (ValueError, "absolute aspect ratio must be < 200, got 20/4001"), (28, 4004)),
    ((3, 700), (ValueError, "absolute aspect ratio must be < 200, got 3/700"), (28, 700)),
    ((9000, 1000), (8204, 896), (8988, 1008)),
    ((10000, 10000), (3584, 3584), (3556, 3556)),
    ((8193, 8193), (3584, 3584), (3556, 3556)),
    (
        (0, 0),
        (ValueError, "height:0 or width:0 must be >= 2"),
        (ZeroDivisionError, "division by zero"),
    ),
]


def _call(fn, h: int, w: int) -> object:
    try:
        return fn(h, w, **_UT_CONSTANTS)
    except Exception as e:  # noqa: BLE001 - characterizes the raise itself.
        return (type(e), str(e))


def _ut_smart_resize():
    from lite.agents.models.ui_tars_15_v1.adapter import _smart_resize

    return _smart_resize


def test_ui_tars_15_v1_constants() -> None:
    from lite.agents.models.ui_tars_15_v1 import adapter as m

    assert (m._IMAGE_FACTOR, m._MIN_PIXELS, m._MAX_PIXELS) == _BUDGET
    assert m._MIN_PIXELS == 4 * 28 * 28
    assert m._MAX_PIXELS == 16384 * 28 * 28


@pytest.mark.parametrize(("hw", "want"), sorted(_AGREE.items()))
def test_canonical_and_ui_tars_agree_on_real_screenshots(
    hw: tuple[int, int], want: tuple[int, int]
) -> None:
    h, w = hw
    assert smart_resize(h, w, **_UT_CONSTANTS) == want
    assert _ut_smart_resize()(h, w, **_UT_CONSTANTS) == want


def test_agreement_is_total_on_the_realistic_grid() -> None:
    ut = _ut_smart_resize()
    differ = [hw for hw in _AGREE if smart_resize(*hw, **_UT_CONSTANTS) != ut(*hw, **_UT_CONSTANTS)]
    assert differ == []


@pytest.mark.parametrize(
    ("hw", "now", "was"),
    _CONSOLIDATED,
    ids=[f"{h}x{w}" for (h, w), _n, _w in _CONSOLIDATED],
)
def test_ui_tars_consolidation_changed_only_degenerate_inputs(
    hw: tuple[int, int], now: object, was: object
) -> None:
    h, w = hw
    assert _call(smart_resize, h, w) == now
    assert _call(_ut_smart_resize(), h, w) == now
    assert now != was


def test_ui_tars_wrapper_only_rebinds_constants() -> None:
    params = set(inspect.signature(_ut_smart_resize()).parameters)
    assert params == {"height", "width", "factor", "min_pixels", "max_pixels"}
    assert "max_long_side" not in params
