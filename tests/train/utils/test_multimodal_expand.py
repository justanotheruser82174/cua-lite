"""Trainer-side ``multimodal_expand.expand`` unit tests.

Verifies that the expand hook correctly materializes cua-lite's lazy
``{images, indices}`` payload into the standard
``multimodal_train_inputs`` shape that slime consumes downstream
(``materialize_lazy_payloads`` → ``get_batch`` → forward).

After the bytes-only simplification, each per-image dict carries
``{"image_data": uint8_tensor}`` with PNG bytes — ``expand`` decodes,
runs the vision ``image_processor``, and concatenates along the
K-window dimension.

A stub processor is installed via monkeypatch so the tests don't need
to download a real HF checkpoint.
"""
from __future__ import annotations

import pytest
import torch

from lite.train.utils.multimodal_expand import expand

# ---------------------------------------------------------------------------
# Helpers + stub processor
# ---------------------------------------------------------------------------

class _StubImageProcessor:
    """Returns ``pixel_values=(W, H)`` per image — a deterministic
    decoder-fidelity probe: if PNG decode worked the recovered PIL has
    the original size, and the stub's output reflects that."""
    def __call__(self, images, return_tensors="pt"):
        pvs = torch.stack([
            torch.tensor([float(im.size[0]), float(im.size[1])], dtype=torch.float32)
            for im in images
        ])
        grids = torch.tensor([[1, 2, 2]] * len(images), dtype=torch.long)
        return {"pixel_values": pvs, "image_grid_thw": grids}


class _StubProcessor:
    image_processor = _StubImageProcessor()


class _StubArgs:
    """Minimal stand-in for slime's ``args`` namespace — only the field
    ``expand`` reads (``hf_checkpoint``) is populated."""
    hf_checkpoint = "/dev/null"  # sentinel — stub processor is pre-installed


def _patch_stub_processor(monkeypatch):
    """Install the stub processor into the module-level cache so ``expand``
    skips HF model load. Uses monkeypatch so the cache is reset cleanly
    between tests — without this, a stub from one test leaks into the next."""
    from lite.train.utils import multimodal_expand
    monkeypatch.setattr(multimodal_expand, "_processor", _StubProcessor())


def _encoded_image(w: int, h: int) -> torch.Tensor:
    """A valid PNG-encoded image of the requested dimensions, wrapped in the
    uint8 tensor that the producer side ships through ray.put."""
    from io import BytesIO

    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (w, h), color=(100, 150, 200)).save(buf, format="PNG")
    return torch.frombuffer(bytearray(buf.getvalue()), dtype=torch.uint8)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_expand_basic(monkeypatch):
    """Two distinct images shipped as PNG bytes → expand decodes + runs
    image_processor + cats. The stub returns (W, H) per image so the row
    contents reflect successful round-trip decoding."""
    _patch_stub_processor(monkeypatch)
    payload = {
        "images":  {
            0: {"image_data": _encoded_image(100, 200)},
            1: {"image_data": _encoded_image(300, 400)},
        },
        "indices": (0, 1),
    }
    out = expand(payload, args=_StubArgs())
    assert set(out.keys()) == {"pixel_values", "image_grid_thw"}
    assert out["pixel_values"].shape == (2, 2)
    assert out["pixel_values"][0].tolist() == [100.0, 200.0]
    assert out["pixel_values"][1].tolist() == [300.0, 400.0]


def test_expand_with_duplicates_in_indices(monkeypatch):
    """K-fold window: indices=(0,0,1) → row 0 and row 1 identical (same
    image decoded twice), row 2 distinct. The lazy mechanism MUST
    materialize the duplication faithfully (downstream cross-Sample concat
    relies on per-Sample shape, not on dedup)."""
    _patch_stub_processor(monkeypatch)
    payload = {
        "images":  {
            0: {"image_data": _encoded_image(50, 60)},
            1: {"image_data": _encoded_image(70, 80)},
        },
        "indices": (0, 0, 1),
    }
    out = expand(payload, args=_StubArgs())
    assert out["pixel_values"].shape == (3, 2)
    assert out["pixel_values"][0].tolist() == [50.0, 60.0]
    assert out["pixel_values"][1].tolist() == [50.0, 60.0]
    assert out["pixel_values"][2].tolist() == [70.0, 80.0]


def test_expand_none_payload():
    """Defensive: None in → None out. Slime won't normally call expand on
    None payloads, but we accept it so the contract is total."""
    assert expand(None) is None


def test_expand_empty_indices(monkeypatch):
    """A Sample with no images visible (text-only step in a protocol that
    drops the window) → empty dict output."""
    _patch_stub_processor(monkeypatch)
    payload = {
        "images":  {0: {"image_data": _encoded_image(10, 10)}},
        "indices": (),
    }
    assert expand(payload, args=_StubArgs()) == {}


def test_expand_bf16_dtype(monkeypatch):
    """Float outputs cast to bf16 by default. Integer outputs untouched."""
    _patch_stub_processor(monkeypatch)
    monkeypatch.delenv("CUA_LITE_MULTIMODAL_FP32", raising=False)
    payload = {
        "images":  {0: {"image_data": _encoded_image(100, 100)}},
        "indices": (0,),
    }
    out = expand(payload, args=_StubArgs())
    assert out["pixel_values"].dtype == torch.bfloat16
    assert out["image_grid_thw"].dtype == torch.long


def test_expand_fp32_escape(monkeypatch):
    """``CUA_LITE_MULTIMODAL_FP32=1`` skips the bf16 cast (debug escape)."""
    _patch_stub_processor(monkeypatch)
    monkeypatch.setenv("CUA_LITE_MULTIMODAL_FP32", "1")
    payload = {
        "images":  {0: {"image_data": _encoded_image(100, 100)}},
        "indices": (0,),
    }
    out = expand(payload, args=_StubArgs())
    assert out["pixel_values"].dtype == torch.float32


def test_expand_missing_hf_checkpoint_in_args(monkeypatch):
    """No ``args.hf_checkpoint`` (CLI flag not passed, or slime didn't
    thread it through) + first payload → actionable AssertionError naming
    the slime arg, not a bare ``KeyError`` deep inside load_processor."""
    from lite.train.utils import multimodal_expand
    monkeypatch.setattr(multimodal_expand, "_processor", None)

    class _NoCkpt:
        hf_checkpoint = None

    payload = {
        "images":  {0: {"image_data": _encoded_image(10, 10)}},
        "indices": (0,),
    }
    with pytest.raises(AssertionError, match="args.hf_checkpoint"):
        expand(payload, args=_NoCkpt())


def test_expand_friendly_error_on_missing_index(monkeypatch):
    """Producer-side bug: indices reference an image not present in the
    dict. The error must name BOTH the bad index AND the available keys."""
    _patch_stub_processor(monkeypatch)
    payload = {
        "images":  {0: {"image_data": _encoded_image(10, 10)}},
        "indices": (0, 99),
    }
    with pytest.raises(KeyError, match="reference image index 99"):
        expand(payload, args=_StubArgs())


def test_expand_swallows_unexpected_kwargs(monkeypatch):
    """slime may add new kwargs to the expand-fn protocol in the future.
    The ``**_`` catch-all in our signature must absorb them without
    raising, so a slime upgrade can't suddenly break our hook."""
    _patch_stub_processor(monkeypatch)
    payload = {"images": {0: {"image_data": _encoded_image(10, 10)}}, "indices": (0,)}
    out = expand(payload, args=_StubArgs(), some_future_kwarg=42, another="x")
    assert "pixel_values" in out


def test_expand_grpo_shape_with_shared_trajectory_dict(monkeypatch):
    """Real GRPO producer shape: N trajectories × K turns, each Sample's
    payload references the SAME trajectory_images dict per trajectory.
    Verifies expand() processes per-Sample correctly regardless of dict
    sharing (matters because materialize_lazy_payloads dispatches expand
    across payloads concurrently — sharing is purely for ray.put dedup,
    not a per-call concern)."""
    _patch_stub_processor(monkeypatch)
    N_TRAJ, K_TURNS = 3, 5

    samples = []
    for t in range(N_TRAJ):
        traj_images = {
            i: {"image_data": _encoded_image(50 + 10*t, 60 + 10*t + i)}
            for i in range(K_TURNS)
        }
        for k in range(K_TURNS):
            samples.append({"images": traj_images, "indices": (k,)})

    assert samples[0]["images"] is samples[1]["images"]
    assert samples[K_TURNS]["images"] is not samples[0]["images"]

    outputs = [expand(s, args=_StubArgs()) for s in samples]

    assert len(outputs) == N_TRAJ * K_TURNS
    for t in range(N_TRAJ):
        for k in range(K_TURNS):
            out = outputs[t * K_TURNS + k]
            assert out["pixel_values"][0].tolist() == [50.0 + 10*t, 60.0 + 10*t + k]


def test_expand_grpo_k_fold_window(monkeypatch):
    """Multi-turn protocol with full_history_size=K>1: each Sample's
    indices is a K-tuple referencing K different images in the
    trajectory dict. ``expand`` must concat all K images per Sample —
    THE motivating case for the lazy design."""
    _patch_stub_processor(monkeypatch)
    K_WINDOW = 4
    traj_images = {
        i: {"image_data": _encoded_image(100 + i, 200 + i)}
        for i in range(K_WINDOW + 2)
    }
    payload = {"images": traj_images, "indices": (1, 2, 3, 4)}
    out = expand(payload, args=_StubArgs())
    assert out["pixel_values"].shape == (K_WINDOW, 2)
    assert out["pixel_values"][0].tolist() == [101.0, 201.0]
    assert out["pixel_values"][3].tolist() == [104.0, 204.0]
