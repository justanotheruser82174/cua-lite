"""Unit tests for the model-agnostic image dedup.

Covers ``lite/train/utils/multimodal_dedup.py``: ``CachingImageProcessor``'s
per-image ``id()`` memoization + ``BatchFeature`` merge, and ``apply()``'s
shallow-copy isolation. A stub image_processor stands in for the real HF one
(no model download), so these run on the host too.

Run: ``python -m pytest tests/train/utils/test_multimodal_dedup.py``
"""
from __future__ import annotations

import copy
import pickle

import torch

from lite.train.utils.multimodal_dedup import CachingImageProcessor, apply


class _Img:
    """Sentinel 'image': distinct instances may share ``val`` (content) but
    each has its own ``id()`` — the cache key. Mirrors decoded PIL objects."""

    def __init__(self, val: int) -> None:
        self.val = val


class _StubImageProcessor:
    """Per-image features: ``pixel_values`` one row ``[val, val]`` +
    ``image_grid_thw`` row ``[1, 2, 2]``. A batched call is exactly the
    row-wise stack, so per-image-then-``_merge_features`` must equal it."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, images, return_tensors="pt"):
        self.calls += len(images)
        pvs = torch.stack([torch.tensor([float(im.val), float(im.val)]) for im in images])
        grids = torch.tensor([[1, 2, 2]] * len(images), dtype=torch.long)
        return {"pixel_values": pvs, "image_grid_thw": grids}


class _StubProcessor:
    def __init__(self) -> None:
        self.image_processor = _StubImageProcessor()


def _equal(a, b) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    return all(torch.equal(a[k], b[k]) for k in a.keys())


def test_merge_byte_identical_to_batched():
    a, b = _Img(1), _Img(2)
    cip = CachingImageProcessor(_StubImageProcessor())
    got = cip(images=[a, b, a], return_tensors="pt")
    want = _StubImageProcessor()(images=[a, b, a], return_tensors="pt")
    assert _equal(got, want)


def test_dedup_fires_by_id():
    a, b = _Img(1), _Img(2)
    cip = CachingImageProcessor(_StubImageProcessor())
    cip(images=[a, b, a], return_tensors="pt")
    assert cip.real_computations == 2  # a, b each computed once
    assert cip.total_requested == 3
    assert cip._ip.calls == 2  # underlying processor saw only the 2 unique


def test_prime_makes_fan_out_read_only():
    a, b = _Img(1), _Img(2)
    cip = CachingImageProcessor(_StubImageProcessor())
    cip.prime([a, b], return_tensors="pt")
    assert cip.real_computations == 2
    cip(images=[a, b, a], return_tensors="pt")  # the parallel-phase call
    assert cip.real_computations == 2  # zero new compute → cache is read-only


def test_single_image_no_spurious_cat():
    a = _Img(7)
    cip = CachingImageProcessor(_StubImageProcessor())
    assert _equal(cip(images=[a]), _StubImageProcessor()(images=[a]))


def test_id_based_not_value_based():
    # equal content, distinct objects → NOT deduped (key is id(), not value)
    a1, a2 = _Img(1), _Img(1)
    cip = CachingImageProcessor(_StubImageProcessor())
    cip(images=[a1, a2], return_tensors="pt")
    assert cip.real_computations == 2


def test_apply_does_not_mutate_original():
    proc = _StubProcessor()
    orig_ip = proc.image_processor
    traj = apply(proc, unique_images=[_Img(1)], images_kwargs={"return_tensors": "pt"})
    assert traj is not proc  # a copy
    assert proc.image_processor is orig_ip  # original never mutated
    assert isinstance(traj.image_processor, CachingImageProcessor)


def test_getattr_survives_copy_and_pickle():
    # The __getattr__ guard must not infinite-recurse under copy/deepcopy/pickle
    # dunder probing (a real boundary — see the guard's comment).
    cip = CachingImageProcessor(_StubImageProcessor())
    copy.copy(cip)
    copy.deepcopy(cip)
    pickle.loads(pickle.dumps(cip))
