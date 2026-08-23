"""Qwen2.5-VL resize geometry tests."""

from __future__ import annotations

from PIL import Image

_BUDGET = (28, 3136, 2007040)
_IMAGE_LEVEL_GOLDENS = [
    ((1920, 1080), (1876, 1036)),
    ((3840, 2160), (1876, 1036)),
    ((1280, 720), (1288, 728)),
]


def test_qwen2_5_vl_constants() -> None:
    from lite.agents.models.qwen2_5_vl import adapter as m

    assert m._FACTOR == 28
    assert m._MAX_PIXELS == 14 * 14 * 4 * 2560 == 2_007_040


def test_qwen2_5_vl_resize_knobs() -> None:
    from lite.agents.models.qwen2_5_vl.adapter import (
        Qwen2_5VLBaseAdapter,
        Qwen2_5VLMobileUseAdapter,
    )

    factor, _min_pixels, max_pixels = _BUDGET
    for cls in (Qwen2_5VLBaseAdapter, Qwen2_5VLMobileUseAdapter):
        assert cls.smart_resize_factor == factor
        assert cls.smart_resize_max_pixels == max_pixels
        assert cls.smart_resize_enabled is True


def test_qwen2_5_vl_image_level_resize() -> None:
    from lite.agents.models.qwen2_5_vl.adapter import Qwen2_5VLBaseAdapter

    adapter = Qwen2_5VLBaseAdapter()
    for src, want in _IMAGE_LEVEL_GOLDENS:
        assert adapter._smart_resize_image(Image.new("RGB", src)).size == want
