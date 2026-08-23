"""FARA resize geometry tests."""

from __future__ import annotations

_BUDGET = (28, 3136, 12845056)


def test_fara_constants() -> None:
    from lite.agents.models.fara import adapter as m

    assert m._FACTOR == 28
    assert m._MAX_PIXELS == 16384 * 28 * 28 == 12_845_056


def test_fara_resize_knobs() -> None:
    from lite.agents.models.fara.adapter import FaraBaseAdapter, FaraDesktopGroundingPointAdapter

    factor, _min_pixels, max_pixels = _BUDGET
    for cls in (FaraBaseAdapter, FaraDesktopGroundingPointAdapter):
        assert cls.smart_resize_factor == factor
        assert cls.smart_resize_max_pixels == max_pixels
        assert cls.smart_resize_enabled is True
