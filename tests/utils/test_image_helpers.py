"""Unit tests for screenshot image helpers.

These helpers had ZERO coverage and their producers aren't required-live, so
this file pins the bytes-migration contract directly:

  * :func:`lite.utils.image.encode_png` — PIL → PNG bytes, bytes passthrough
    (no double-encode), ``bytearray`` → ``bytes``;
  * :func:`lite.utils.image.png_from_b64` — b64 str → bytes, ``None`` → ``None``
    (distinct from ``''`` → ``b''``), round-trip;
  * :func:`lite.utils.image.decode_image` — the load-bearing
    ``.convert("RGB")`` guard on a palette / L / RGBA PNG.

Run:  uv run pytest tests/utils/test_image_helpers.py -x -q
"""
from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from lite.utils.image import decode_image, encode_png, png_from_b64


def _png_bytes(mode: str = "RGB", size: tuple[int, int] = (8, 6)) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# encode_png
# ---------------------------------------------------------------------------


def test_encode_png_from_pil_returns_png_bytes():
    img = Image.new("RGB", (10, 12))
    out = encode_png(img)
    assert isinstance(out, bytes)
    # Bytes are a real PNG that reopens to the same-size image.
    reopened = Image.open(io.BytesIO(out))
    assert reopened.format == "PNG"
    assert reopened.size == (10, 12)


def test_encode_png_bytes_passthrough_no_double_encode():
    raw = _png_bytes()
    out = encode_png(raw)
    assert isinstance(out, bytes)
    assert out == raw  # verbatim — NOT re-encoded


def test_encode_png_bytearray_normalized_to_bytes():
    raw = _png_bytes()
    out = encode_png(bytearray(raw))
    assert type(out) is bytes  # bytearray → bytes so the codec's ``+=`` is happy
    assert out == raw


# ---------------------------------------------------------------------------
# png_from_b64
# ---------------------------------------------------------------------------


def test_png_from_b64_decodes_str_to_bytes():
    raw = _png_bytes()
    s = base64.b64encode(raw).decode("ascii")
    out = png_from_b64(s)
    assert isinstance(out, bytes)
    assert out == raw


def test_png_from_b64_none_returns_none():
    assert png_from_b64(None) is None


def test_png_from_b64_empty_string_returns_empty_bytes():
    out = png_from_b64("")
    assert out == b""
    assert out is not None  # distinct from the None case


def test_png_from_b64_roundtrip_with_encode_png():
    raw = encode_png(Image.new("RGB", (16, 9)))
    s = base64.b64encode(raw).decode("ascii")
    assert png_from_b64(s) == raw


# ---------------------------------------------------------------------------
# decode_image — the ``.convert("RGB")`` guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["P", "L", "RGBA"])
def test_decode_image_normalizes_to_rgb(mode: str):
    """A palette / grayscale / alpha PNG must decode to mode ``RGB`` — dropping
    the ``.convert("RGB")`` is a silent channel/tensor-shape regression."""
    png = _png_bytes(mode=mode)
    img = decode_image(png)
    assert img.mode == "RGB"


def test_decode_image_rgb_passthrough():
    img = decode_image(_png_bytes(mode="RGB"))
    assert img.mode == "RGB"
    assert img.size == (8, 6)
