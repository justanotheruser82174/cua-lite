"""MAI-UI resize geometry tests."""

from __future__ import annotations

from PIL import Image

_IMAGE_LEVEL_GOLDENS = [
    ((1920, 1080), (1920, 1088)),
    ((3840, 2160), (3392, 1920)),
    ((1280, 720), (1280, 704)),
]


def test_mai_ui_resize_knobs() -> None:
    from lite.agents.models.mai_ui.adapter import MAIUIGroundingPointAdapter as C

    assert C.smart_resize_factor == 16 * 2 == 32
    assert C.smart_resize_min_pixels == 16 * 16 * 4 == 1024
    assert C.smart_resize_max_pixels == 6_553_600


def test_mai_ui_image_level_resize() -> None:
    from lite.agents.models.mai_ui.adapter import MAIUIGroundingPointAdapter

    adapter = MAIUIGroundingPointAdapter()
    for src, want in _IMAGE_LEVEL_GOLDENS:
        assert adapter._process_image_after_target(Image.new("RGB", src)).size == want
