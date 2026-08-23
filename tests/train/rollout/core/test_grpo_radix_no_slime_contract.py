"""No-Slime GRPO/radix packing contract tests.

These tests exercise the real ``lite/train/rollout/core/segmenter.py`` source
with a deliberately tiny ``slime`` stub. They are not Slime integration tests;
they pin the producer-side contract that remains meaningful on a host where
``slime.utils`` is unavailable.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from lite.core import LiteCUAMetadata, LiteSample
from lite.core.samples import (
    STATUS_COMPLETED,
    STATUS_TRUNCATED,
    LiteRLSample,
    LiteRLStep,
)


class _SampleStub:
    class Status:
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        FAILED = "failed"

    def __init__(self, **kwargs):
        self.group_index = kwargs.pop("group_index", None)
        self.index = kwargs.pop("index", None)
        self.group_id = kwargs.pop("group_id", None)
        self.prompt = kwargs.pop("prompt", "")
        self.tokens = kwargs.pop("tokens", [])
        self.loss_mask = kwargs.pop("loss_mask", [])
        self.rollout_log_probs = kwargs.pop("rollout_log_probs", None)
        self.response_length = kwargs.pop("response_length", 0)
        self.response = kwargs.pop("response", "")
        self.label = kwargs.pop("label", None)
        self.reward = kwargs.pop("reward", 0.0)
        self.status = kwargs.pop("status", self.Status.PENDING)
        self.metadata = kwargs.pop("metadata", {})
        self.multimodal_train_inputs = kwargs.pop("multimodal_train_inputs", None)
        self.multimodal_lazy_payloads = kwargs.pop("multimodal_lazy_payloads", None)
        for key, value in kwargs.items():
            setattr(self, key, value)


def _build_processor_kwargs(payload: dict) -> dict:
    return {
        "images": payload.get("images"),
        "images_kwargs": {"return_tensors": "pt"},
    }


def _segmenter_under_test(monkeypatch):
    slime = types.ModuleType("slime")
    slime.__path__ = []
    utils = types.ModuleType("slime.utils")
    utils.__path__ = []
    processing_utils = types.ModuleType("slime.utils.processing_utils")
    processing_utils.build_processor_kwargs = _build_processor_kwargs
    types_mod = types.ModuleType("slime.utils.types")
    types_mod.Sample = _SampleStub

    monkeypatch.setitem(sys.modules, "slime", slime)
    monkeypatch.setitem(sys.modules, "slime.utils", utils)
    monkeypatch.setitem(sys.modules, "slime.utils.processing_utils", processing_utils)
    monkeypatch.setitem(sys.modules, "slime.utils.types", types_mod)

    module_name = "_cua_lite_no_slime_segmenter"
    module_path = Path(__file__).parents[4] / "lite/train/rollout/core/segmenter.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _ids(text: str) -> list[int]:
    return [ord(ch) for ch in text]


def _img(idx: int) -> Image.Image:
    img = Image.new("RGB", (4, 4), color=(idx * 40, idx * 40, idx * 40))
    img._cua_test_id = idx
    return img


class _RecordingImageProcessor:
    def __call__(self, images=None, return_tensors=None):
        n = len(images or [])
        return {
            "pixel_values": torch.ones((n, 1), dtype=torch.float32),
            "image_grid_thw": torch.ones((n, 3), dtype=torch.long),
        }


class _RecordingProcessor:
    image_token = "<|image_pad|>"

    def __init__(self):
        self.image_processor = _RecordingImageProcessor()
        self.image_orders: list[tuple[int, ...]] = []

    def __call__(
        self, text=None, images=None, text_kwargs=None, images_kwargs=None, **kwargs
    ):
        imgs = list(images or [])
        self.image_orders.append(tuple(getattr(img, "_cua_test_id") for img in imgs))
        image_inputs = self.image_processor(images=imgs, return_tensors="pt")
        return {
            "input_ids": _ids(text or ""),
            **image_inputs,
        }


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return _ids(text)


def _trajectory() -> LiteRLSample:
    prompt0 = "user <|image_pad|><|image_pad|>\n"
    response0 = "Action: click\n"
    tool_error_delta = "role:tool text/error feedback\n<|image_pad|>\n"
    prompt1 = prompt0 + response0 + tool_error_delta
    response1 = "Action: wait\n"

    return LiteRLSample(
        processed_images=[_img(0), _img(1), _img(2)],
        steps=[
            LiteRLStep(
                prompt=prompt0,
                image_indices=(1, 0),
                response=response0,
                response_tokens=_ids(response0),
                response_log_probs=[-0.1] * len(response0),
                reward=0.0,
                status=STATUS_COMPLETED,
            ),
            LiteRLStep(
                prompt=prompt1,
                image_indices=(1, 0, 2),
                response=response1,
                response_tokens=_ids(response1),
                response_log_probs=[-0.2] * len(response1),
                reward=1.0,
                status=STATUS_TRUNCATED,
            ),
        ],
        lite_sample=LiteSample(metadata=LiteCUAMetadata(dims=("desktop", "use"))),
        episode_return=1.0,
    )


def _original_sample() -> _SampleStub:
    return _SampleStub(
        group_index=3,
        index=77,
        prompt="original",
        label="label",
        metadata={"env_key": "lite.demo@train"},
    )


def _state(hook_path: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        args=SimpleNamespace(multimodal_lazy_expand_fn_path=hook_path),
        aborted=False,
    )


def test_no_slime_radix_packs_tool_feedback_and_preserves_group_contract(monkeypatch):
    segmenter = _segmenter_under_test(monkeypatch)
    monkeypatch.setenv("CUA_LITE_ROLLOUT_PROC_WORKERS", "1")
    monkeypatch.setenv("CUA_LITE_MULTIMODAL_LAZY_EXPAND", "0")
    monkeypatch.delenv("CUA_LITE_DISABLE_RADIX", raising=False)

    rl_sample = _trajectory()
    processor = _RecordingProcessor()
    samples = segmenter.build_segment_samples(
        rl_sample=rl_sample,
        original_sample=_original_sample(),
        processor=processor,
        tokenizer=_Tokenizer(),
        state=_state(hook_path=None),
    )

    assert processor.image_orders == [(1, 0), (1, 0, 2)]
    assert len(samples) == 1
    sample = samples[0]
    assert sample.group_id == 77
    assert sample.metadata["turn_range"] == (0, 1)
    assert sample.metadata["n_turns"] == 2
    assert sample.metadata["step_statuses"] == (STATUS_COMPLETED, STATUS_TRUNCATED)
    assert sample.status == _SampleStub.Status.TRUNCATED

    response0 = rl_sample.steps[0].response
    response1 = rl_sample.steps[1].response
    tool_delta = rl_sample.steps[1].prompt[
        len(rl_sample.steps[0].prompt + response0):
    ]
    assert sample.response == response0 + response1
    assert sample.loss_mask == (
        [1] * len(response0)
        + [0] * len(tool_delta)
        + [1] * len(response1)
    )
    assert sum(sample.loss_mask) == len(response0) + len(response1)
    assert sample.multimodal_train_inputs["pixel_values"].shape[0] == 3
    assert sample.multimodal_lazy_payloads is None


def test_no_slime_lazy_payload_uses_segment_tail_image_binding(monkeypatch):
    segmenter = _segmenter_under_test(monkeypatch)
    monkeypatch.setenv("CUA_LITE_ROLLOUT_PROC_WORKERS", "1")
    monkeypatch.setenv("CUA_LITE_MULTIMODAL_LAZY_EXPAND", "1")
    monkeypatch.delenv("CUA_LITE_DISABLE_RADIX", raising=False)

    rl_sample = _trajectory()
    processor = _RecordingProcessor()
    samples = segmenter.build_segment_samples(
        rl_sample=rl_sample,
        original_sample=_original_sample(),
        processor=processor,
        tokenizer=_Tokenizer(),
        state=_state(
            hook_path="lite.train.utils.multimodal_expand.expand",
        ),
    )

    assert processor.image_orders == [(1, 0), (1, 0, 2)]
    assert len(samples) == 1
    sample = samples[0]
    assert sample.metadata["turn_range"] == (0, 1)
    assert sample.multimodal_train_inputs is None
    payload = sample.multimodal_lazy_payloads
    assert payload["indices"] == (1, 0, 2)
    assert set(payload["images"]) == {0, 1, 2}
    assert all(
        entry["image_data"].dtype == torch.uint8
        for entry in payload["images"].values()
    )


def test_no_slime_segmenter_breaks_when_prompt_image_binding_changes(monkeypatch):
    segmenter = _segmenter_under_test(monkeypatch)

    class _Step:
        def __init__(self, image_indices):
            self.image_indices = image_indices

    per_step = [
        {
            "prompt_tokens": [1, 2, 3],
            "response_tokens": [10],
            "step": _Step((1, 0)),
        },
        {
            "prompt_tokens": [1, 2, 3, 10, 99],
            "response_tokens": [11],
            "step": _Step((0, 1, 2)),
        },
    ]

    assert segmenter._segment_steps(per_step) == [[0], [1]]
