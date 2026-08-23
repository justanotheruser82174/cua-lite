"""Qwen3-VL resize geometry tests."""

from __future__ import annotations

from PIL import Image

_BUDGET = (32, 3136, 13107200)
_IMAGE_LEVEL_GOLDENS = [
    ((1920, 1080), (1920, 1088)),
    ((3840, 2160), (3840, 2176)),
    ((1280, 720), (1280, 704)),
]


def test_qwen3_vl_constants() -> None:
    from lite.agents.models.qwen3_vl import adapter as m

    assert m._QWEN3VL_FACTOR == 32
    assert m._QWEN3VL_MAX_PIXELS == 16 * 16 * 4 * 12800 == 13_107_200


def test_qwen3_vl_resize_knobs() -> None:
    from lite.agents.models.qwen3_vl.adapter import Qwen3VLBaseAdapter, Qwen3VLMobileUseAdapter

    factor, _min_pixels, max_pixels = _BUDGET
    assert Qwen3VLBaseAdapter.smart_resize_factor == factor
    assert Qwen3VLBaseAdapter.smart_resize_max_pixels == max_pixels
    assert Qwen3VLBaseAdapter.smart_resize_enabled is True
    assert Qwen3VLMobileUseAdapter.smart_resize_factor == factor
    assert Qwen3VLMobileUseAdapter.smart_resize_max_pixels == max_pixels
    assert Qwen3VLMobileUseAdapter.smart_resize_enabled is False


def test_qwen3_vl_image_level_resize() -> None:
    from lite.agents.models.qwen3_vl.adapter import Qwen3VLBaseAdapter

    adapter = Qwen3VLBaseAdapter()
    for src, want in _IMAGE_LEVEL_GOLDENS:
        assert adapter._smart_resize_image(Image.new("RGB", src)).size == want


def test_qwen3_vl_mobile_adapters_do_not_resize() -> None:
    from lite.agents.models.qwen3_vl.adapter import Qwen3VLBaseAdapter, Qwen3VLMobileUseAdapter

    adapter = Qwen3VLMobileUseAdapter()
    img = Image.new("RGB", (720, 1520))
    assert adapter._process_image_after_target(img) is img
    assert Qwen3VLBaseAdapter()._process_image_after_target(img).size == (704, 1536)
