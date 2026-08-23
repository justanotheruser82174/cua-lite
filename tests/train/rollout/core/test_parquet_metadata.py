"""Hermetic rollout metadata lane tests for GRPO segment emission.

These tests load the real rollout segmenter, and one generate() path, behind a
tiny Slime surface. They pin the train-side metadata contract without requiring
Slime, GPUs, Docker, or live env services.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from lite.core import LiteCUAMetadata, LiteGenericMetadata, LiteSample
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


class _Tokenizer:
    eos_token_id = 999_999
    eos_token = "<eos>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(ch) for ch in text]


class _GenerateStateStub:
    def __init__(self, args=None):
        self.args = args
        self.tokenizer = _Tokenizer()
        self.processor = None
        self.aborted = False
        self.pendings = set()
        self.remaining_batch_size = 0
        self.semaphore = asyncio.Semaphore(1)


def _build_processor_kwargs(payload: dict) -> dict:
    return {
        "images": payload.get("images"),
        "images_kwargs": {"return_tensors": "pt"},
    }


def _install_slime_stubs(monkeypatch) -> None:
    slime = types.ModuleType("slime")
    slime.__path__ = []
    rollout = types.ModuleType("slime.rollout")
    rollout.__path__ = []
    base_types = types.ModuleType("slime.rollout.base_types")
    base_types.RolloutFnEvalOutput = object
    base_types.RolloutFnTrainOutput = object
    filter_hub = types.ModuleType("slime.rollout.filter_hub")
    filter_hub.__path__ = []
    filter_base_types = types.ModuleType("slime.rollout.filter_hub.base_types")
    filter_base_types.MetricGatherer = object
    filter_base_types.call_dynamic_filter = lambda *args, **kwargs: None
    sglang_rollout = types.ModuleType("slime.rollout.sglang_rollout")
    sglang_rollout.GenerateState = _GenerateStateStub
    sglang_rollout.generate_and_rm = lambda *args, **kwargs: None

    utils = types.ModuleType("slime.utils")
    utils.__path__ = []
    async_utils = types.ModuleType("slime.utils.async_utils")
    async_utils.run = lambda coro: asyncio.run(coro)
    http_utils = types.ModuleType("slime.utils.http_utils")

    async def _unused_post(url, payload):
        raise AssertionError("HTTP post should not run in metadata tests")

    http_utils.get = lambda *args, **kwargs: None
    http_utils.post = _unused_post
    misc = types.ModuleType("slime.utils.misc")
    misc.load_function = lambda path: None
    processing_utils = types.ModuleType("slime.utils.processing_utils")
    processing_utils.build_processor_kwargs = _build_processor_kwargs
    processing_utils.encode_image_for_rollout_engine = lambda image: "encoded"
    types_mod = types.ModuleType("slime.utils.types")
    types_mod.Sample = _SampleStub

    for name, module in {
        "slime": slime,
        "slime.rollout": rollout,
        "slime.rollout.base_types": base_types,
        "slime.rollout.filter_hub": filter_hub,
        "slime.rollout.filter_hub.base_types": filter_base_types,
        "slime.rollout.sglang_rollout": sglang_rollout,
        "slime.utils": utils,
        "slime.utils.async_utils": async_utils,
        "slime.utils.http_utils": http_utils,
        "slime.utils.misc": misc,
        "slime.utils.processing_utils": processing_utils,
        "slime.utils.types": types_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def _load_module(monkeypatch, module_name: str, relative_path: str):
    module_path = Path(__file__).parents[4] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _segmenter_under_test(monkeypatch):
    _install_slime_stubs(monkeypatch)
    return _load_module(
        monkeypatch,
        "_cua_lite_rollout_metadata_segmenter",
        "lite/train/rollout/core/segmenter.py",
    )


def _engine_under_test(monkeypatch):
    segmenter = _segmenter_under_test(monkeypatch)
    segmenter_owner = types.ModuleType("lite.train.rollout.core.segmenter")
    segmenter_owner.build_segment_samples = segmenter.build_segment_samples
    monkeypatch.setitem(sys.modules, "lite.train.rollout.core.segmenter", segmenter_owner)
    return _load_module(
        monkeypatch,
        "_cua_lite_rollout_metadata_engine",
        "lite/train/rollout/core/engine.py",
    )


def _ids(text: str) -> list[int]:
    return [ord(ch) for ch in text]


def _steps() -> list[LiteRLStep]:
    prompt0 = "p0:"
    response0 = "a0"
    delta1 = "|obs1|"
    prompt1 = prompt0 + response0 + delta1
    response1 = "a1!"
    return [
        LiteRLStep(
            prompt=prompt0,
            image_indices=(),
            response=response0,
            response_tokens=_ids(response0),
            response_log_probs=[-0.1, -0.2],
            reward=0.0,
            status=STATUS_COMPLETED,
        ),
        LiteRLStep(
            prompt=prompt1,
            image_indices=(),
            response=response1,
            response_tokens=_ids(response1),
            response_log_probs=[-0.3, -0.4, -0.5],
            reward=2.5,
            status=STATUS_TRUNCATED,
        ),
    ]


def _rl_sample(*, metadata=None, episode_return: float = 2.5) -> LiteRLSample:
    return LiteRLSample(
        processed_images=[],
        steps=_steps(),
        lite_sample=LiteSample(
            metadata=metadata
            or LiteCUAMetadata(dims=("desktop", "use"), others={"task_id": "task-a"})
        ),
        episode_return=episode_return,
        terminated=True,
        truncated=False,
    )


def _original_sample() -> _SampleStub:
    return _SampleStub(
        group_index=3,
        index=17,
        group_id=17,
        prompt="source prompt",
        label="label",
        metadata={
            "env_key": "lite.demo@task-a",
            "split": "train",
            "agent_key": "qwen3_vl@desktop@use",
            "others": {"source": "prompt_data"},
        },
    )


def _assert_segment_metadata(metadata: dict) -> None:
    assert metadata["env_key"] == "lite.demo@task-a"
    assert metadata["split"] == "train"
    assert metadata["agent_key"] == "qwen3_vl@desktop@use"
    assert metadata["n_turns"] == 2
    assert metadata["turn_range"] == (0, 1)
    assert metadata["step_statuses"] == (STATUS_COMPLETED, STATUS_TRUNCATED)
    assert metadata["others"] == {
        "source": "prompt_data",
        "episode_return": 2.5,
    }
    assert "episode_return" not in metadata
    assert "platform" not in metadata
    assert "task_type" not in metadata


def test_emit_rl_sample_preserves_rollout_metadata_without_flat_task_fields(monkeypatch):
    segmenter = _segmenter_under_test(monkeypatch)
    rl_sample = _rl_sample()
    steps = rl_sample.steps

    sample = segmenter._emit_rl_sample(
        segment=steps,
        original_sample=_original_sample(),
        rl_sample=rl_sample,
        per_step_prompt_ids=[
            [1, 2, 3],
            [1, 2, 3, *steps[0].response_tokens, 4, 5],
        ],
        multimodal_train_inputs=None,
    )

    _assert_segment_metadata(sample.metadata)
    assert sample.group_index == 3
    assert sample.index == 17
    assert sample.group_id == 17
    assert sample.reward == 2.5
    assert sample.status == _SampleStub.Status.TRUNCATED


def test_build_segment_samples_emits_metadata_for_packed_trajectory(monkeypatch):
    segmenter = _segmenter_under_test(monkeypatch)
    monkeypatch.delenv("CUA_LITE_DISABLE_RADIX", raising=False)
    monkeypatch.setenv("CUA_LITE_MULTIMODAL_LAZY_EXPAND", "0")

    samples = segmenter.build_segment_samples(
        rl_sample=_rl_sample(),
        original_sample=_original_sample(),
        processor=None,
        tokenizer=_Tokenizer(),
        state=SimpleNamespace(
            args=SimpleNamespace(multimodal_lazy_expand_fn_path=None),
            aborted=False,
        ),
    )

    assert len(samples) == 1
    sample = samples[0]
    _assert_segment_metadata(sample.metadata)
    assert sample.response == "a0a1!"
    assert sample.loss_mask == [1, 1, *([0] * len("|obs1|")), 1, 1, 1]
    assert sample.rollout_log_probs == [
        -0.1,
        -0.2,
        *([0.0] * len("|obs1|")),
        -0.3,
        -0.4,
        -0.5,
    ]


def test_lite_sample_rows_persist_canonical_metadata_and_deserialize(tmp_path) -> None:
    import pyarrow.parquet as pq

    from lite.utils.parquet import write_records_to_parquet

    cua_metadata = LiteCUAMetadata(
        dims=("browser", "use"),
        valid_actions=["click", "type"],
        others={"task_id": "browser-task"},
    )
    generic_metadata = LiteGenericMetadata(
        dims=("math",),
        others={"dataset": "geo3k"},
    )
    records = [
        LiteSample(metadata=metadata, images=[], messages=[]).to_dict()
        for metadata in (cua_metadata, generic_metadata)
    ]

    path = tmp_path / "trajectory.parquet"
    write_records_to_parquet(records, path, json_fields=("messages", "metadata"))
    persisted = pq.read_table(path).to_pylist()

    for row, metadata in zip(persisted, (cua_metadata, generic_metadata)):
        row_metadata = json.loads(row["metadata"])
        assert row_metadata == metadata.to_dict()
        assert "platform" not in row_metadata
        assert "task_type" not in row_metadata

        restored = LiteSample.from_dict(
            {
                "images": [],
                "messages": json.loads(row["messages"]),
                "metadata": row_metadata,
            }
        )
        assert type(restored.metadata) is type(metadata)
        assert restored.metadata.to_dict() == metadata.to_dict()


def test_generate_preserves_sample_metadata_through_segment_build(monkeypatch):
    engine = _engine_under_test(monkeypatch)
    monkeypatch.delenv("CUA_LITE_DISABLE_RADIX", raising=False)
    monkeypatch.setenv("CUA_LITE_MULTIMODAL_LAZY_EXPAND", "0")

    class _Env:
        metadata = LiteCUAMetadata(dims=("desktop", "use"))

        async def close(self):
            self.closed = True

    class _Agent:
        async def sample(self, env, max_steps=100, hooks=None):
            return _rl_sample()

    captured = {}

    def _get_agent(key, processor=None, generate_fn=None, metadata=None, **kwargs):
        captured["agent_key"] = key
        captured["metadata"] = metadata
        return _Agent()

    env = _Env()
    engine.gym = SimpleNamespace(make=lambda env_key, **kwargs: env)
    engine.register_all = lambda: None
    engine.AgentRegistry = SimpleNamespace(get=_get_agent)
    engine.GenerateState = _GenerateStateStub
    engine._current_rollout_id = 0

    args = SimpleNamespace(
        partial_rollout=False,
        group_shared_seed=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        agent_id="qwen3_vl",
        agent_kwargs={},
        env_kwargs={},
    )
    sample = _original_sample()
    result = asyncio.run(
        engine.generate(args, sample, {"temperature": 1.0, "sampling_seed": 7})
    )

    assert len(result) == 1
    _assert_segment_metadata(result[0].metadata)
    assert captured["agent_key"] == "qwen3_vl@desktop@use"
    assert captured["metadata"] is env.metadata
    assert env.closed is True
    assert env not in engine._active_envs
