"""Site-level pins for the round+clamp coordinate conversion.

tests/gym/utils/backend/test_coordinate.py pins ``norm_to_pixel`` itself; these tests pin
that each FLIPPED call site actually routes through the canonical clamping
conversion — a site quietly reverting to a hand-rolled truncate would put a
model-emitted ``[1000, 1000]`` at ``(w, h)`` (off-canvas) instead of
``(w-1, h-1)``. One test per flipped site: captcha, waa, sandbox base
(exec-stdio dispatch), and the shared cursor image helper.

Run: uv run pytest tests/gym/utils/backend/test_coordinate_sites.py
"""
from __future__ import annotations

import asyncio
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from lite.gym.errors import EnvDepsMissingError

W, H = 1920, 1080


def test_captcha_to_pixel_clamps_1000_to_edge():
    try:
        from lite.gym.envs.captcha.main import LocalCaptchaEnv
    except EnvDepsMissingError as e:  # captcha deps absent until its install.sh runs
        pytest.skip(f"captcha env deps not installed: {e}")
    # _to_pixel never reads self — call it unbound on a bare instance.
    env = LocalCaptchaEnv.__new__(LocalCaptchaEnv)
    assert LocalCaptchaEnv._to_pixel(env, [1000, 1000], W, H) == (W - 1, H - 1)
    assert LocalCaptchaEnv._to_pixel(env, [0, 0], W, H) == (0, 0)


def test_waa_px_clamps_1000_to_edge():
    pytest.importorskip("httpx", reason="httpx not installed")
    from lite.gym.envs.waa.main import _px

    assert _px([1000, 1000], W, H) == (W - 1, H - 1)
    assert _px([0, 0], W, H) == (0, 0)


def test_sandbox_dispatch_click_clamps_1000_to_edge():
    from lite.gym.sandbox.base import _dispatch_desktop_action

    class _Iface:
        async def left_click(self, x, y):
            pass

    computer = SimpleNamespace(interface=_Iface())
    calls = asyncio.run(
        _dispatch_desktop_action(computer, "click", {"coordinate": [1000, 1000]}, W, H)
    )
    assert calls == [
        {"call": "computer.interface.left_click", "args": {"x": W - 1, "y": H - 1}}
    ]


def test_cursor_pastes_at_clamped_edge(monkeypatch):
    from PIL import Image

    import lite.gym.wrappers as wrappers

    w, h = 64, 48
    monkeypatch.setattr(
        wrappers,
        "load_cursor_sprite",
        lambda height=24: Image.new("RGBA", (3, 3), (255, 0, 0, 255)),
    )

    buf = io.BytesIO()
    Image.new("RGB", (w, h), (0, 0, 0)).save(buf, format="PNG")
    out_png = wrappers.overlay_cursor_px(buf.getvalue(), w, h)

    out = Image.open(io.BytesIO(out_png)).convert("RGB")
    # Clamped paste at (w-1, h-1): exactly the corner pixel turns red. An
    # unclamped (w, h) paste would draw nothing; an off-by-one would color
    # (w-2, h-2) instead.
    assert out.getpixel((w - 1, h - 1)) == (255, 0, 0)
    assert out.getpixel((w - 2, h - 2)) == (0, 0, 0)


def test_cursor_real_asset_size_and_straight_alpha():
    """Golden for the shared cursor helper with the REAL asset.

    This pins (a) the resized sprite is exactly 16x24 and (b) straight
    (non-premultiplied) alpha compositing: an opaque sprite pixel over a black
    background reproduces the sprite's own RGB, a fully-transparent pixel leaves
    the background untouched, and nothing is drawn outside the 16x24 box.
    """
    from PIL import Image

    from lite.gym.wrappers import load_cursor_sprite, overlay_cursor_px

    sprite = load_cursor_sprite()
    assert sprite.size == (16, 24), f"cursor sprite {sprite.size} != (16, 24)"
    assert sprite.mode == "RGBA"

    # A fully-opaque and a fully-transparent source pixel pin straight-alpha.
    opaque = next(
        ((x, y, sprite.getpixel((x, y))[:3])
         for y in range(24) for x in range(16)
         if sprite.getpixel((x, y))[3] == 255),
        None,
    )
    transparent = next(
        ((x, y) for y in range(24) for x in range(16)
         if sprite.getpixel((x, y))[3] == 0),
        None,
    )
    assert opaque is not None and transparent is not None

    # Paste tip at (0, 0): the sprite box is [0:16] x [0:24] and its top-left is
    # the pointer.
    w, h = 200, 200
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (0, 0, 0)).save(buf, format="PNG")
    out_png = overlay_cursor_px(buf.getvalue(), 0, 0)
    out = Image.open(io.BytesIO(out_png)).convert("RGB")

    ox, oy, rgb = opaque
    # Opaque sprite pixel over black == its own RGB (straight alpha, no brighten).
    assert out.getpixel((ox, oy)) == rgb
    # Fully-transparent sprite pixel → background shows through unchanged.
    assert out.getpixel(transparent) == (0, 0, 0)
    # Nothing drawn outside the 16x24 box.
    assert out.getpixel((16, 0)) == (0, 0, 0)
    assert out.getpixel((0, 24)) == (0, 0, 0)
    assert out.getpixel((100, 100)) == (0, 0, 0)


def test_sandbox_cursor_asset_matches_shared_helper():
    """The sandbox /opt/lite/cursor.png source matches load_cursor_sprite(24)."""
    from PIL import Image, ImageChops

    from lite.gym.wrappers import load_cursor_sprite

    root = Path(__file__).resolve().parents[4]
    dockerfile = root / "lite/gym/sandbox/docker/Dockerfile.linux"
    assert (
        "COPY docker/assets/cursor.png /opt/lite/cursor.png"
        in dockerfile.read_text(encoding="utf-8")
    )

    sandbox_sprite = Image.open(
        root / "lite/gym/sandbox/docker/assets/cursor.png"
    ).convert("RGBA")
    shared_sprite = load_cursor_sprite(height=24)

    assert sandbox_sprite.size == shared_sprite.size == (16, 24)
    assert ImageChops.difference(sandbox_sprite, shared_sprite).getbbox() is None
