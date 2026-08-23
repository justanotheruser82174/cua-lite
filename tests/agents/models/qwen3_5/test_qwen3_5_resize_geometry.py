"""Qwen3.5 resize geometry tests."""

from __future__ import annotations

_BUDGET = (32, 3136, 13107200)


def test_qwen3_5_resize_knobs() -> None:
    from lite.agents.models.qwen3_5.adapter import Qwen3_5BaseAdapter, Qwen3_5MobileUseAdapter

    factor, _min_pixels, max_pixels = _BUDGET
    assert Qwen3_5BaseAdapter.smart_resize_factor == factor
    assert Qwen3_5BaseAdapter.smart_resize_max_pixels == max_pixels
    assert Qwen3_5BaseAdapter.smart_resize_enabled is True
    assert Qwen3_5MobileUseAdapter.smart_resize_factor == factor
    assert Qwen3_5MobileUseAdapter.smart_resize_max_pixels == max_pixels
    assert Qwen3_5MobileUseAdapter.smart_resize_enabled is False
