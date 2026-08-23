"""EvoCUA resize geometry tests."""

from __future__ import annotations

_BUDGET = (32, 3136, 13107200)


def test_evocua_resize_knobs() -> None:
    from lite.agents.models.evocua.adapter import (
        EvoCUADesktopGroundingPointAdapter,
        EvoCUADesktopUseAdapter,
    )

    factor, _min_pixels, max_pixels = _BUDGET
    for cls in (EvoCUADesktopUseAdapter, EvoCUADesktopGroundingPointAdapter):
        assert cls.smart_resize_factor == factor
        assert cls.smart_resize_max_pixels == max_pixels
        assert cls.smart_resize_enabled is True
