"""Hermetic radix coverage for terminal EOF finish turns."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from lite.core import LiteCUAMetadata, LiteSample
from lite.core.samples import STATUS_COMPLETED, LiteRLSample, LiteRLStep


class _SegmenterSampleStub:
    class Status:
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        FAILED = "failed"

    def __init__(self, **kwargs):
        self.group_index = kwargs.pop("group_index")
        self.index = kwargs.pop("index")
        self.prompt = kwargs.pop("prompt")
        self.label = kwargs.pop("label", None)
        self.metadata = kwargs.pop("metadata", None)
        for key, value in kwargs.items():
            setattr(self, key, value)


def _stub_build_processor_kwargs(payload: dict) -> dict:
    return {
        "images": payload.get("images"),
        "images_kwargs": {"return_tensors": "pt"},
    }


def _segmenter_under_test(monkeypatch):
    slime = types.ModuleType("slime")
    slime.__path__ = []
    utils = types.ModuleType("slime.utils")
    utils.__path__ = []
    proc = types.ModuleType("slime.utils.processing_utils")
    proc.build_processor_kwargs = _stub_build_processor_kwargs
    types_mod = types.ModuleType("slime.utils.types")
    types_mod.Sample = _SegmenterSampleStub

    monkeypatch.setitem(sys.modules, "slime", slime)
    monkeypatch.setitem(sys.modules, "slime.utils", utils)
    monkeypatch.setitem(sys.modules, "slime.utils.processing_utils", proc)
    monkeypatch.setitem(sys.modules, "slime.utils.types", types_mod)

    module_name = "_cua_lite_test_terminal_eof_segmenter"
    module_path = Path(__file__).parents[4] / "lite/train/rollout/core/segmenter.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module.build_segment_samples, _SegmenterSampleStub


class _TinyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(ch) for ch in text]


def _make_original(Sample):
    return Sample(
        group_index=0,
        index=0,
        prompt="orig prompt",
        tokens=[],
        loss_mask=[],
        response_length=0,
        response="",
        label=None,
        reward=0.0,
        status=Sample.Status.PENDING,
        metadata={"env_key": "lite.demo@test"},
    )


@pytest.mark.parametrize(
    ("finish_response", "finish_token"),
    [
        ("<tool_call>response(text='Done.')</tool_call>", 30),
        ("<tool_call>terminate(status='success')</tool_call>", 31),
    ],
)
def test_radix_packed_eof_finish_has_no_fabricated_terminal_delta(
    monkeypatch,
    finish_response,
    finish_token,
):
    monkeypatch.delenv("CUA_LITE_DISABLE_RADIX", raising=False)
    build_segment_samples, Sample = _segmenter_under_test(monkeypatch)
    steps = [
        LiteRLStep(
            prompt="first prompt",
            image_indices=(),
            response="<tool_call>click()</tool_call>",
            response_tokens=[10, 11],
            response_log_probs=[-0.1, -0.2],
            reward=0.0,
            status=STATUS_COMPLETED,
            prompt_tokens=[1, 2],
        ),
        LiteRLStep(
            prompt="second prompt with real tool feedback",
            image_indices=(),
            response=finish_response,
            response_tokens=[finish_token],
            response_log_probs=[-0.3],
            reward=1.0,
            status=STATUS_COMPLETED,
            prompt_tokens=[1, 2, 10, 11, 20],
        ),
    ]
    rl_sample = LiteRLSample(
        processed_images=[],
        steps=steps,
        lite_sample=LiteSample(metadata=LiteCUAMetadata(dims=("desktop", "use"))),
        episode_return=1.0,
        terminated=True,
        truncated=False,
    )

    samples = build_segment_samples(
        rl_sample=rl_sample,
        original_sample=_make_original(Sample),
        processor=None,
        tokenizer=_TinyTokenizer(),
        state=SimpleNamespace(aborted=False),
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.metadata["turn_range"] == (0, 1)
    assert sample.tokens == [1, 2, 10, 11, 20, finish_token]
    assert sample.response_length == 4
    assert sample.loss_mask == [1, 1, 0, 1]
    assert sample.rollout_log_probs == [-0.1, -0.2, 0.0, -0.3]
    assert sample.tokens[-1] == finish_token
    assert sample.loss_mask[-1] == 1
    assert "Task terminated:" not in sample.response
    assert "Final answer submitted:" not in sample.response
