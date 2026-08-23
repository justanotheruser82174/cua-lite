"""``slime.utils.data.materialize_lazy_payloads`` tests.

This is the trainer-side hook that hydrates ``multimodal_lazy_payloads``
(PNG bytes staged by the rollout) into the standard
``multimodal_train_inputs`` shape. It runs once per RL iter from
``actor._get_rollout_data`` — moved here from a per-micro-batch JIT block
in ``slime/backends/megatron_utils/data.py::get_batch`` so the same
payload isn't expanded N× per phase (log_probs + every grpo_iter inner
train step).

We exercise the real function (not a re-implementation) so a future
change to the hydrate semantics is caught directly. The expand path
itself uses cua-lite's ``multimodal_expand`` hook; we stub the HF
processor so the test doesn't need a checkpoint.
"""
from __future__ import annotations

import importlib
from io import BytesIO

import pytest
import torch
from PIL import Image


def materialize_lazy_payloads(*args, **kwargs):
    try:
        slime_data = importlib.import_module("slime.utils.data")
    except ModuleNotFoundError as exc:
        missing = exc.name or "slime.utils.data"
        pytest.skip(f"{missing} not installed; cannot import slime.utils.data")

    hydrate = getattr(slime_data, "materialize_lazy_payloads", None)
    if hydrate is None:
        pytest.skip("slime.utils.data.materialize_lazy_payloads not installed")
    return hydrate(*args, **kwargs)

# ---------------------------------------------------------------------------
# Stub processor + helpers
# ---------------------------------------------------------------------------

class _StubImageProcessor:
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
    """Mimics the slime CLI args namespace with the two fields the hydrate
    path reads: ``multimodal_lazy_expand_fn_path`` and ``hf_checkpoint``."""
    multimodal_lazy_expand_fn_path = "lite.train.utils.multimodal_expand.expand"
    hf_checkpoint = "/dev/null"  # stub processor is pre-installed


@pytest.fixture(autouse=True)
def _patch_stub_processor(monkeypatch):
    """Install the stub processor so every expand() call in this module
    skips the HF model load."""
    from lite.train.utils import multimodal_expand
    monkeypatch.setattr(multimodal_expand, "_processor", _StubProcessor())


def _png_tensor(w: int, h: int) -> torch.Tensor:
    buf = BytesIO()
    Image.new("RGB", (w, h), color=(100, 150, 200)).save(buf, format="PNG")
    return torch.frombuffer(bytearray(buf.getvalue()), dtype=torch.uint8)


def _img_dict(w: int, h: int) -> dict:
    return {"image_data": _png_tensor(w, h)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_hydrate_materializes_lazy():
    """rollout_data carrying lazy payloads → after hydrate, the lazy key
    is gone, ``multimodal_train_inputs`` holds one materialized dict per
    Sample, shapes match what the downstream concat layer expects."""
    shared = {0: _img_dict(30, 40), 1: _img_dict(70, 80)}
    rollout_data = {
        "multimodal_lazy_payloads": [
            {"images": shared, "indices": (0,)},
            {"images": shared, "indices": (0, 1)},
            {"images": shared, "indices": (1,)},
        ],
    }

    materialize_lazy_payloads(_StubArgs(), rollout_data)

    assert "multimodal_lazy_payloads" not in rollout_data
    mm = rollout_data["multimodal_train_inputs"]
    assert len(mm) == 3
    # Stub returns (W, H) per image → row count = len(indices)
    assert mm[0]["pixel_values"].shape == (1, 2)
    assert mm[1]["pixel_values"].shape == (2, 2)
    assert mm[2]["pixel_values"].shape == (1, 2)


def test_hydrate_preserves_nonmonotonic_duplicate_image_order():
    """``indices`` is a positional binding to prompt image pads, not a set."""
    shared = {0: _img_dict(10, 20), 2: _img_dict(30, 40)}
    rollout_data = {
        "multimodal_lazy_payloads": [
            {"images": shared, "indices": (2, 0, 2)},
        ],
    }

    materialize_lazy_payloads(_StubArgs(), rollout_data)

    pixel_values = rollout_data["multimodal_train_inputs"][0]["pixel_values"]
    assert torch.equal(
        pixel_values,
        torch.tensor(
            [[30.0, 40.0], [10.0, 20.0], [30.0, 40.0]],
            dtype=torch.float32,
        ),
    )


def test_assert_lazy_and_legacy_mutex():
    """A Sample carrying BOTH lazy_payloads AND a non-None
    multimodal_train_inputs entry violates the producer-side mutex —
    AssertionError fires (defense-in-depth against rollout bugs)."""
    shared = {0: _img_dict(10, 10)}
    rollout_data = {
        "multimodal_lazy_payloads": [
            {"images": shared, "indices": (0,)},
        ],
        "multimodal_train_inputs": [
            {"pixel_values": torch.zeros(1, 4)},  # not None → violation
        ],
    }
    with pytest.raises(AssertionError, match="mutually exclusive"):
        materialize_lazy_payloads(_StubArgs(), rollout_data)


def test_no_lazy_is_noop():
    """No lazy key in rollout_data → hydrate is a no-op; the legacy slot
    survives untouched and feeds the downstream concat path verbatim."""
    legacy_mm = [{"pixel_values": torch.zeros(1, 4)}, {"pixel_values": torch.ones(1, 4)}]
    rollout_data = {"multimodal_train_inputs": legacy_mm}

    materialize_lazy_payloads(_StubArgs(), rollout_data)

    assert rollout_data["multimodal_train_inputs"] is legacy_mm
    assert "multimodal_lazy_payloads" not in rollout_data


def test_legacy_with_all_none_passes():
    """rollout_data has lazy + a legacy list whose entries are all None.
    The mutex is satisfied (no Sample carries both); hydrate proceeds and
    overwrites the train_inputs slot."""
    shared = {0: _img_dict(15, 25)}
    rollout_data = {
        "multimodal_lazy_payloads": [
            {"images": shared, "indices": (0,)},
            {"images": shared, "indices": (0,)},
        ],
        "multimodal_train_inputs": [None, None],
    }

    materialize_lazy_payloads(_StubArgs(), rollout_data)

    mm = rollout_data["multimodal_train_inputs"]
    assert len(mm) == 2
    assert mm[0]["pixel_values"].shape == (1, 2)
    assert mm[1]["pixel_values"].shape == (1, 2)


def test_none_entries_passthrough():
    """Text-only Samples in a multimodal batch carry ``None`` as their
    lazy payload (no images at that step). Hydrate must preserve the
    None slot so downstream cat-loop's ``if mm_input_dict is not None``
    guard works."""
    shared = {0: _img_dict(20, 30)}
    rollout_data = {
        "multimodal_lazy_payloads": [
            None,
            {"images": shared, "indices": (0,)},
            None,
        ],
    }
    materialize_lazy_payloads(_StubArgs(), rollout_data)
    mm = rollout_data["multimodal_train_inputs"]
    assert mm[0] is None
    assert mm[2] is None
    assert mm[1]["pixel_values"].shape == (1, 2)


def _get_batch_like_multimodal_concat(mm_inputs: list[dict | None]) -> dict:
    """Small local mirror of get_batch's multimodal microbatch concat branch."""
    present = [entry for entry in mm_inputs if entry is not None]
    assert present
    return {
        key: torch.cat([entry[key] for entry in present], dim=0)
        for key in present[0]
    }


def test_materialize_then_get_batch_like_mixed_none_image_microbatch(monkeypatch):
    """Actor-ish integration: materialize lazy payloads once, then feed a
    mixed text/image microbatch through get_batch-style multimodal concat."""
    monkeypatch.delenv("CUA_LITE_MULTIMODAL_FP32", raising=False)
    shared = {0: _img_dict(20, 30), 1: _img_dict(40, 50)}
    rollout_data = {
        "multimodal_lazy_payloads": [
            None,
            {"images": shared, "indices": (0,)},
            None,
            {"images": shared, "indices": (1,)},
        ],
    }

    materialize_lazy_payloads(_StubArgs(), rollout_data)
    mm = rollout_data["multimodal_train_inputs"]

    assert [entry is None for entry in mm] == [True, False, True, False]
    microbatch = _get_batch_like_multimodal_concat(mm)
    assert torch.equal(
        microbatch["pixel_values"],
        torch.tensor([[20.0, 30.0], [40.0, 50.0]], dtype=torch.bfloat16),
    )
    assert microbatch["image_grid_thw"].shape == (2, 3)


def test_missing_expand_fn_path():
    """rollout_data carries lazy payloads but ``args`` doesn't set the
    CLI flag → AssertionError tells the user to pass the flag."""
    class _NoFlag:
        hf_checkpoint = "/dev/null"
        # no multimodal_lazy_expand_fn_path
    shared = {0: _img_dict(10, 10)}
    rollout_data = {
        "multimodal_lazy_payloads": [{"images": shared, "indices": (0,)}],
    }
    with pytest.raises(AssertionError, match="multimodal-lazy-expand-fn-path"):
        materialize_lazy_payloads(_NoFlag(), rollout_data)


def test_parallel_path_matches_serial():
    """Force the serial path (workers=1) and the parallel path and
    confirm they produce identical results. Guards against subtle
    ordering / thread-local issues in the ThreadPoolExecutor branch."""
    shared = {i: _img_dict(10 + i, 20 + i) for i in range(20)}
    payloads_a = [{"images": shared, "indices": (i,)} for i in range(20)]
    payloads_b = [{"images": shared, "indices": (i,)} for i in range(20)]

    rd_serial = {"multimodal_lazy_payloads": payloads_a}
    rd_parallel = {"multimodal_lazy_payloads": payloads_b}

    materialize_lazy_payloads(_StubArgs(), rd_serial, max_workers=1)
    materialize_lazy_payloads(_StubArgs(), rd_parallel, max_workers=8)

    for a, b in zip(rd_serial["multimodal_train_inputs"], rd_parallel["multimodal_train_inputs"]):
        assert torch.equal(a["pixel_values"], b["pixel_values"])
        assert torch.equal(a["image_grid_thw"], b["image_grid_thw"])
