"""CUA-Lite train-boundary checks for Slime ``Sample`` transport fields.

These tests intentionally live in the CUA-Lite repo rather than
``slime/tests`` because they pin the fields CUA-Lite sends through Slime's
``Sample.to_dict()`` / ``Sample.from_dict()`` boundary.
"""
from __future__ import annotations

import pytest


def _make_sample(sample_cls, **overrides):
    base = dict(
        index=42,
        prompt="hello",
        tokens=[1, 2, 3],
        response="world",
        response_length=5,
        loss_mask=[1, 1, 1],
        status=sample_cls.Status.COMPLETED,
    )
    base.update(overrides)
    return sample_cls(**base)


def test_slime_sample_round_trip_preserves_cua_lite_multimodal_transport_fields():
    torch = pytest.importorskip("torch", reason="torch not installed")
    slime_types = pytest.importorskip("slime.utils.types", reason="slime not installed")
    sample_cls = slime_types.Sample

    train_inputs = {
        "pixel_values": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "image_grid_thw": torch.tensor([[1, 1, 2]], dtype=torch.long),
    }
    lazy_payload = {
        "images": {2: {"image_data": torch.tensor([137, 80, 78, 71], dtype=torch.uint8)}},
        "indices": (2,),
    }

    eager = sample_cls.from_dict(
        _make_sample(sample_cls, multimodal_train_inputs=train_inputs).to_dict()
    )
    lazy = sample_cls.from_dict(
        _make_sample(sample_cls, multimodal_lazy_payloads=lazy_payload).to_dict()
    )

    assert torch.equal(
        eager.multimodal_train_inputs["pixel_values"],
        train_inputs["pixel_values"],
    )
    assert torch.equal(
        eager.multimodal_train_inputs["image_grid_thw"],
        train_inputs["image_grid_thw"],
    )
    assert lazy.multimodal_lazy_payloads["indices"] == (2,)
    assert set(lazy.multimodal_lazy_payloads["images"]) == {2}
    assert torch.equal(
        lazy.multimodal_lazy_payloads["images"][2]["image_data"],
        lazy_payload["images"][2]["image_data"],
    )
