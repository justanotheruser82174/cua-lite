"""UI-TARS resize schema-adjacent goldens."""

from __future__ import annotations

from PIL import Image

from lite.agents.models.ui_tars.adapter import UITarsDesktopGroundingPointAdapter
from lite.agents.models.ui_tars_15_v1.adapter import UITars15V1DesktopGroundingPointAdapter


# The two UI-TARS families carry different default min_pixels:
# ui_tars = 100*28*28 -> 40x40 upscales to (280, 280), while
# ui_tars_15_v1 = 4*28*28 -> 40x40 upscales to (56, 56).
def test_ui_tars_linear_resize_min_pixels():
    img = Image.new("RGB", (40, 40))
    assert UITarsDesktopGroundingPointAdapter()._linear_resize_image(img.copy()).size == (
        280,
        280,
    )
    assert UITars15V1DesktopGroundingPointAdapter()._linear_resize_image(img.copy()).size == (
        56,
        56,
    )
