"""Logger parquet contracts that cross data/export boundaries.

These tests start from a real ``TrajectoryLogger`` output directory. They do
not hand-write a trajectory parquet, because the logger's persisted paths,
metadata, and turn artifacts are the contract under refactor.

Pure ``lite.data.hf.stage`` policy tests live in ``test_stage.py``. Tests here
stay only when the raw logger artifact is the input contract being consumed by
stage, upload/download, or SFT export.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import pandas as pd
from PIL import Image

from lite.agents.core.agent.hooks import SampleStepData
from lite.agents.core.agent.logger import TrajectoryLogger
from lite.agents.types import PredictResult
from lite.core import (
    LiteCUAMetadata,
    LiteRLSample,
    LiteRLStep,
    LiteSample,
)
from lite.core.tools.calls import make_tool_call
from lite.core.tools.results import LiteToolResult
from lite.data.hf import upload as hf_upload
from lite.data.hf.download import download_dataset
from lite.data.hf.stage import stage
from lite.data.staging import (
    ImageStore,
    coerce_messages,
    coerce_meta,
    image_rel_prefix,
    iter_parquet_rows,
    iter_partitions,
    partition_path,
)
from lite.gym.types import LiteEnvStepResult
from lite.infer.debug.log_contract import check_log_contract
from lite.train.export import export_sft
from lite.utils.parquet import write_records_to_parquet

_NATIVE_OBSERVATION_TEXT = "## AXTree:\nbutton selected"
_NATIVE_RESULT_ERROR = "click was outside active element"
_PROJECTED_ERROR_TEXT = (
    f"{_NATIVE_OBSERVATION_TEXT}\n\n"
    f"## Error from previous action:\n{_NATIVE_RESULT_ERROR}"
)


class _FakeTokenizer:
    eos_token_id = 99999

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        toks: list[int] = []
        i = 0
        marker = "<|im_end|>"
        while i < len(text):
            if text.startswith(marker, i):
                toks.append(self.eos_token_id)
                i += len(marker)
            else:
                toks.append(ord(text[i]))
                i += 1
        return toks


class _FakeQwenProcessor:
    """Small Qwen-like renderer for export_sft boundary tests."""

    tokenizer = _FakeTokenizer()
    image_token = "<|image_pad|>"

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
    ) -> str:
        rendered = ""
        for message in messages:
            rendered += f"<|im_start|>{message['role']}\n"
            for part in message.get("content") or []:
                kind = part.get("type")
                if kind == "text":
                    rendered += part.get("text", "")
                elif kind == "image":
                    rendered += "<|vision_start|><|image_pad|><|vision_end|>"
            for call in message.get("tool_calls") or []:
                # Adapter-unrolled Qwen messages use model agent-wire calls,
                # not persisted Lite tool-call envelopes.
                rendered += (
                    "\n<tool_call>\n"
                    + json.dumps({"name": call["name"], "arguments": call["arguments"]})
                    + "\n</tool_call>"
                )
            rendered += "<|im_end|>\n"
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered


class _SFTSampleStub:
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


class _SFTConsumerImageProcessor:
    def __call__(self, images=None, return_tensors=None):
        import torch

        n = len(images or [])
        return {
            "pixel_values": torch.ones((n, 1), dtype=torch.float32),
            "image_grid_thw": torch.ones((n, 3), dtype=torch.long),
        }


class _SFTConsumerProcessor:
    image_token = "<|image_pad|>"

    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.image_processor = _SFTConsumerImageProcessor()
        self.image_calls: list[tuple[tuple[int, int], ...]] = []

    def __call__(self, text=None, images=None, text_kwargs=None, **kwargs):
        import torch

        imgs = list(images or [])
        self.image_calls.append(tuple(img.size for img in imgs))
        return {
            "input_ids": self.tokenizer.encode(text or "", add_special_tokens=False),
            "pixel_values": torch.ones((len(imgs), 1), dtype=torch.float32),
            "image_grid_thw": torch.ones((len(imgs), 3), dtype=torch.long),
        }


def _install_sft_rollout_under_test(monkeypatch, processor: _SFTConsumerProcessor):
    """Load the real rollout_sft module behind minimal Slime stubs."""

    def build_processor_kwargs(payload: dict) -> dict:
        return {
            "images": payload.get("images"),
            "images_kwargs": {"return_tensors": "pt"},
        }

    slime = types.ModuleType("slime")
    slime.__path__ = []
    utils = types.ModuleType("slime.utils")
    utils.__path__ = []
    proc = types.ModuleType("slime.utils.processing_utils")
    proc.build_processor_kwargs = build_processor_kwargs
    proc.load_processor = lambda *args, **kwargs: processor
    types_mod = types.ModuleType("slime.utils.types")
    types_mod.Sample = _SFTSampleStub
    monkeypatch.setitem(sys.modules, "slime", slime)
    monkeypatch.setitem(sys.modules, "slime.utils", utils)
    monkeypatch.setitem(sys.modules, "slime.utils.processing_utils", proc)
    monkeypatch.setitem(sys.modules, "slime.utils.types", types_mod)

    root = Path(__file__).parents[3]
    segmenter_spec = importlib.util.spec_from_file_location(
        "_cua_lite_t6_segmenter",
        root / "lite/train/rollout/core/segmenter.py",
    )
    assert segmenter_spec is not None and segmenter_spec.loader is not None
    segmenter = importlib.util.module_from_spec(segmenter_spec)
    monkeypatch.setitem(sys.modules, "_cua_lite_t6_segmenter", segmenter)
    segmenter_spec.loader.exec_module(segmenter)

    core = types.ModuleType("lite.train.rollout.core")
    core.build_segment_samples = segmenter.build_segment_samples
    core.dummy_sample = lambda sample: _SFTSampleStub(
        group_index=sample.group_index,
        index=sample.index,
        prompt=sample.prompt,
        tokens=[],
        loss_mask=[],
        response_length=0,
        response="",
        label=getattr(sample, "label", None),
        reward=0.0,
        status=_SFTSampleStub.Status.COMPLETED,
        metadata={},
    )
    monkeypatch.setitem(sys.modules, "lite.train.rollout.core", core)

    sft_spec = importlib.util.spec_from_file_location(
        "_cua_lite_t6_rollout_sft",
        root / "lite/train/rollout/sft.py",
    )
    assert sft_spec is not None and sft_spec.loader is not None
    rollout_sft = importlib.util.module_from_spec(sft_spec)
    monkeypatch.setitem(sys.modules, "_cua_lite_t6_rollout_sft", rollout_sft)
    sft_spec.loader.exec_module(rollout_sft)
    rollout_sft.PROCESSOR = processor
    return rollout_sft


def _consume_sft_export(
    out: dict,
    monkeypatch,
) -> tuple[list[_SFTSampleStub], _SFTConsumerProcessor]:
    processor = _SFTConsumerProcessor()
    rollout_sft = _install_sft_rollout_under_test(monkeypatch, processor)
    monkeypatch.setenv("CUA_LITE_ROLLOUT_PROC_WORKERS", "1")
    monkeypatch.setenv("CUA_LITE_MULTIMODAL_LAZY_EXPAND", "0")
    monkeypatch.delenv("CUA_LITE_DISABLE_RADIX", raising=False)

    sample = _SFTSampleStub(
        group_index=0,
        index=0,
        prompt=out["steps"],
        tokens=[],
        loss_mask=[],
        response_length=0,
        response="",
        label=None,
        reward=0.0,
        status=_SFTSampleStub.Status.PENDING,
        metadata=out["processed_images"],
    )

    class _Buffer:
        def get_samples(self, batch_size):
            return [(sample,)]

    args = types.SimpleNamespace(
        rollout_batch_size=1,
        rollout_global_dataset=True,
        hf_checkpoint="fake-qwen",
        multimodal_lazy_expand_fn_path=None,
    )
    return (
        rollout_sft.generate_rollout(args, rollout_id=0, data_buffer=_Buffer(), evaluation=False),
        processor,
    )


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _messages() -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "Click the center button, then finish."},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "click the center button"}],
            "tool_calls": [make_tool_call(
                "computer",
                {
                    "actions": [
                        {
                            "action": "click",
                            "coordinate": [500, 500],
                            "button": "left",
                        }
                    ]
                },
                call_id="call_0000",
            )],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0000",
            "content": [
                {"type": "image", "index": 1},
                {"type": "text", "text": _PROJECTED_ERROR_TEXT},
                {"type": "metadata", "data": {"url": "about:blank"}},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
    ]


def _write_logger_run(log_dir: Path, *, debug_artifacts: bool = False) -> Path:
    images = [
        Image.new("RGB", (96, 64), "white"),
        Image.new("RGB", (96, 64), "lightblue"),
    ]
    metadata = LiteCUAMetadata(
        dims=("desktop", "use"),
        extra_tool_schemas=[],
        valid_actions=["click", "wait"],
        others={"env_id": "logger.test", "task_id": "task_logger_path"},
    )
    messages = _messages()
    click_call = messages[1]["tool_calls"][0]
    final_message = messages[-1]

    logger = TrajectoryLogger(
        log_dir,
        env_id="logger.test",
        task_id="task_logger_path",
        debug_artifacts=debug_artifacts,
        provenance={"model_id": "gpt-5.5", "agent_id": "gpt.desktop"},
    )
    logger.on_step(
        SampleStepData(
            step_idx=0,
            image=images[0],
            predict_result=PredictResult(
                lite_message=messages[1],
                agent_message={"role": "assistant", "content": "Action: click"},
                step=LiteRLStep(prompt="prompt 0", image_indices=(0,), response="raw 0"),
            ),
            step_result=LiteEnvStepResult(
                results=[
                    LiteToolResult(
                        tool_call_id="call_0000",
                        images=[_png_bytes(images[1])],
                        text=_NATIVE_OBSERVATION_TEXT,
                        metadata={"url": "about:blank"},
                        error=_NATIVE_RESULT_ERROR,
                    )
                ],
                reward=0.0,
                terminated=False,
                truncated=False,
                info={
                    "executed_actions": [
                        {"name": "click", "arguments": {"coordinate": [48, 32]}}
                    ],
                },
            ),
            actions=[click_call],
        )
    )
    logger.on_step(
        SampleStepData(
            step_idx=1,
            image=images[1],
            predict_result=PredictResult(
                lite_message=final_message,
                agent_message={"role": "assistant", "content": "Done."},
                step=LiteRLStep(prompt="prompt 1", image_indices=(0, 1), response="Done."),
            ),
            step_result=LiteEnvStepResult(
                results=[],
                reward=1.0,
                terminated=True,
                truncated=False,
                info={"stop_reason": "content_only_final"},
            ),
            actions=[],
        )
    )
    logger.on_complete(
        LiteRLSample(
            lite_sample=LiteSample(metadata=metadata, images=images, messages=messages),
            processed_images=list(images),
            steps=[
                LiteRLStep(prompt="prompt 0", image_indices=(0,), response="raw 0"),
                LiteRLStep(prompt="prompt 1", image_indices=(0, 1), response="Done."),
            ],
            episode_return=1.0,
            terminated=True,
            truncated=False,
        )
    )
    return log_dir / "trajectory.parquet"


def _export_row(
    row: dict,
    *,
    image_root: Path | None,
    agent_id: str = "qwen3_vl",
    agent_kwargs: dict | None = None,
) -> dict:
    return export_sft._convert_sample(
        row,
        agent_id=agent_id,
        agent_kwargs=agent_kwargs or {},
        model_id="fake-qwen",
        image_root=str(image_root) if image_root is not None else None,
        strict=True,
    )


def _assert_export_shape(out: dict) -> tuple[tuple[tuple[int, ...], str, str], ...]:
    assert out["_error"] == ""
    assert len(out["processed_images"]) == 2
    assert len(out["steps"]) == 2
    signature = []
    for step in out["steps"]:
        assert step["prompt"].count("<|image_pad|>") == len(step["image_indices"])
        signature.append((tuple(step["image_indices"]), step["prompt"], step["response"]))
    assert signature[0][0] == (0,)
    assert signature[1][0] == (0, 1)
    assert "computer_use" in signature[0][2]
    assert "left_click" in signature[0][2]
    assert "Done." in signature[1][2]
    assert "<tool_call>" not in signature[1][2]
    return tuple(signature)


def _assert_export_metadata_matches_row(row: dict, out: dict) -> dict:
    row_metadata = coerce_meta(row["metadata"])
    exported_metadata = json.loads(out["metadata"])
    assert exported_metadata == row_metadata
    assert exported_metadata["metadata_kind"] == "cua"
    assert exported_metadata["dims"] == ["desktop", "use"]
    assert isinstance(exported_metadata["others"], dict)
    assert exported_metadata["others"]["task_id"] == "task_logger_path"
    return exported_metadata


def _message_texts(row: dict) -> list[str]:
    texts: list[str] = []
    for msg in coerce_messages(row["messages"]):
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, str):
            texts.append(content)
            continue
        if not isinstance(content, list):
            continue
        texts.extend(
            part["text"]
            for part in content
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            )
        )
    return texts


def _assert_projected_error_once_in_messages(row: dict) -> None:
    assert sum(text.count(_PROJECTED_ERROR_TEXT) for text in _message_texts(row)) == 1


def _assert_projected_error_once_in_export(out: dict) -> None:
    joined = "\n".join(
        f"{step['prompt']}\n{step['response']}" for step in out["steps"]
    )
    assert joined.count(_PROJECTED_ERROR_TEXT) == 1


def _assert_sft_consumer_shape(out: dict, monkeypatch) -> None:
    samples, processor = _consume_sft_export(out, monkeypatch)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.metadata["turn_range"] == (0, 1)
    assert sample.metadata["n_turns"] == 2
    assert sample.multimodal_lazy_payloads is None
    assert sample.multimodal_train_inputs["pixel_values"].shape[0] == 2
    assert processor.image_calls == [((96, 64),), ((96, 64), (96, 64))]
    assert sample.response.count(_PROJECTED_ERROR_TEXT) == 0
    assert "Done." in sample.response


def _message_image_indices(row: dict) -> tuple[int, ...]:
    indices: list[int] = []
    for msg in coerce_messages(row["messages"]):
        for part in msg.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "image":
                indices.append(part["index"])
    return tuple(indices)


def _assert_row_image_binding(
    row: dict,
    *,
    expected_indices: tuple[int, ...],
    image_root: Path | None = None,
) -> None:
    images = list(row["images"])
    assert len(images) > max(expected_indices)
    assert _message_image_indices(row) == expected_indices
    if image_root is None:
        return
    for rel in images:
        path = Path(rel)
        assert not path.is_absolute()
        assert (image_root / path).is_file()


def _read_one_partition_row(dataset_dir: Path) -> dict:
    partitions = list(iter_partitions(dataset_dir))
    assert len(partitions) == 1
    rows = list(iter_parquet_rows(partitions[0][4]))
    assert len(rows) == 1
    return rows[0]


def _materialize_upload_snapshot(staging_dir: Path, *, name: str, snapshot_dir: Path) -> None:
    store = ImageStore(staging_dir / "images", rel_prefix=image_rel_prefix(name))
    for platform, task_type, split, variant, parquet_path in iter_partitions(staging_dir):
        rows = list(iter_parquet_rows(parquet_path))
        dataset = hf_upload._rows_to_dataset(rows, store)
        shard_path = partition_path(
            Path(""),
            platform=platform,
            task_type=task_type,
            split=split,
            variant=variant or task_type.split(".")[-1],
            shard_idx=0,
            shard_total=1,
        )
        out = snapshot_dir / shard_path
        out.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(str(out), batch_size=50, write_page_index=True)


def _goal_image_messages() -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 1},
                {"type": "image", "index": 2},
                {"type": "image", "index": 0},
                {"type": "text", "text": "Find this product, then finish."},
                {"type": "metadata", "data": {"goal_image_indices": [1, 2]}},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "click the product"}],
            "tool_calls": [make_tool_call(
                "computer",
                {
                    "actions": [
                        {
                            "action": "click",
                            "coordinate": [500, 500],
                            "button": "left",
                        }
                    ]
                },
                call_id="call_0000",
            )],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0000",
            "content": [
                {"type": "image", "index": 3},
                {"type": "text", "text": "## AXTree:\nproduct opened"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
    ]


def _write_goal_image_logger_run(log_dir: Path) -> Path:
    images = [
        Image.new("RGB", (96, 64), (10, 20, 30)),
        Image.new("RGB", (96, 64), (200, 10, 10)),
        Image.new("RGB", (96, 64), (10, 200, 10)),
        Image.new("RGB", (96, 64), (10, 10, 200)),
    ]
    metadata = LiteCUAMetadata(
        dims=("browser", "use"),
        extra_tool_schemas=[],
        valid_actions=["click"],
        others={
            "env_id": "browsergym.visualwebarena",
            "task_id": "task_goal_image_path",
        },
    )
    messages = _goal_image_messages()
    logger = TrajectoryLogger(
        log_dir,
        env_id="browsergym.visualwebarena",
        task_id="task_goal_image_path",
        provenance={"model_id": "gpt-5.5", "agent_id": "qwen3_vl"},
    )
    logger.on_complete(
        LiteRLSample(
            lite_sample=LiteSample(metadata=metadata, images=images, messages=messages),
            processed_images=list(images),
            steps=[
                LiteRLStep(prompt="prompt 0", image_indices=(1, 2, 0), response="raw 0"),
                LiteRLStep(prompt="prompt 1", image_indices=(1, 2, 3), response="Done."),
            ],
            episode_return=1.0,
            terminated=True,
            truncated=False,
        )
    )
    return log_dir / "trajectory.parquet"


def _processed_rgb(processed_blob: bytes) -> tuple[int, int, int]:
    return Image.open(io.BytesIO(processed_blob)).convert("RGB").getpixel((0, 0))


def _assert_goal_image_export_shape(
    out: dict,
) -> tuple[tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]], ...]:
    assert out["_error"] == ""
    assert len(out["processed_images"]) == 4
    assert len(out["steps"]) == 2
    signature = []
    for step in out["steps"]:
        indices = tuple(step["image_indices"])
        assert step["prompt"].count("<|image_pad|>") == len(indices)
        signature.append((
            indices,
            tuple(_processed_rgb(out["processed_images"][idx]) for idx in indices),
        ))
    assert signature[0][0] == (1, 2, 0)
    assert signature[0][1] == ((200, 10, 10), (10, 200, 10), (10, 20, 30))
    assert signature[1][0] == (1, 2, 3)
    assert signature[1][1] == ((200, 10, 10), (10, 200, 10), (10, 10, 200))
    first_prompt = out["steps"][0]["prompt"]
    before_goal, between_goals, between_goal_and_page, _after_page = first_prompt.split(
        "<|image_pad|>",
        3,
    )
    assert "Task reference image" in before_goal
    assert "Current screenshot:" not in between_goals
    assert "Current screenshot:" in between_goal_and_page
    return tuple(signature)


def test_stage_resolves_logger_relative_images_against_project_root(tmp_path, monkeypatch):
    producer_root = tmp_path / "producer"
    producer_root.mkdir()
    monkeypatch.setattr("lite.agents.core.agent.logger.project_root", lambda: producer_root)
    monkeypatch.setattr("lite.data.staging.project_root", lambda: producer_root)

    raw_parquet = _write_logger_run(
        producer_root / ".data" / "rollout" / "train" / "task_logger_path" / "sample_00"
    )
    raw_row = pd.read_parquet(raw_parquet).to_dict("records")[0]
    raw_images = list(raw_row["images"])

    assert raw_images
    assert not Path(raw_images[0]).is_absolute()
    assert raw_images[0].startswith(".data/rollout/")

    name = "LoggerCwdBase"
    staged_root = tmp_path / "staged"
    staged_dataset = staged_root / "cua-lite" / name
    other_cwd = tmp_path / "other-cwd"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    stage([producer_root / ".data"], name=name, out_dir=staged_dataset, filter_expr=None)

    staged_row = _read_one_partition_row(staged_dataset)
    assert len(staged_row["images"]) == len(raw_images)
    for rel in staged_row["images"]:
        assert (staged_root / rel).is_file()


def test_logger_parquet_stage_upload_download_export_sft_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(export_sft, "_get_processor", lambda _model_id: _FakeQwenProcessor())

    raw_parquet = _write_logger_run(tmp_path / "logs" / "train" / "task_logger_path" / "sample_00")
    sample_dir = raw_parquet.parent

    summary = json.loads((sample_dir / "summary.json").read_text())
    assert "stop_reason" not in summary
    turn_results = json.loads((sample_dir / "turn_0000" / "04_results.json").read_text())
    assert turn_results["results"][0]["text"] == _NATIVE_OBSERVATION_TEXT
    assert turn_results["results"][0]["error"] == _NATIVE_RESULT_ERROR
    final_turn_results = json.loads((sample_dir / "turn_0001" / "04_results.json").read_text())
    assert final_turn_results["info"]["stop_reason"] == "content_only_final"
    assert check_log_contract(sample_dir) == []

    raw_row = pd.read_parquet(raw_parquet).to_dict("records")[0]
    _assert_row_image_binding(raw_row, expected_indices=(0, 1))
    _assert_projected_error_once_in_messages(raw_row)
    raw_export = _export_row(raw_row, image_root=None)
    raw_metadata = _assert_export_metadata_matches_row(raw_row, raw_export)
    assert raw_metadata["valid_actions"] == ["click", "wait"]
    raw_signature = _assert_export_shape(raw_export)
    _assert_projected_error_once_in_export(raw_export)
    _assert_sft_consumer_shape(raw_export, monkeypatch)

    name = "LoggerPathE2E"
    staged_root = tmp_path / "staged"
    staged_dataset = staged_root / "cua-lite" / name
    stage([tmp_path / "logs"], name=name, out_dir=staged_dataset, filter_expr=None)
    staged_row = _read_one_partition_row(staged_dataset)
    assert not (staged_dataset / "trajectory_images").exists()
    assert all(
        str(path).startswith(f"cua-lite/{name}/images/")
        for path in staged_row["images"]
    )
    assert "stop_reason" not in coerce_meta(staged_row["metadata"])["others"]
    _assert_row_image_binding(staged_row, expected_indices=(0, 1), image_root=staged_root)
    _assert_projected_error_once_in_messages(staged_row)
    staged_export = _export_row(staged_row, image_root=staged_root)
    staged_metadata = _assert_export_metadata_matches_row(staged_row, staged_export)
    assert "stop_reason" not in staged_metadata["others"]
    staged_signature = _assert_export_shape(
        staged_export
    )
    _assert_projected_error_once_in_export(staged_export)
    _assert_sft_consumer_shape(staged_export, monkeypatch)
    assert staged_signature == raw_signature

    snapshot_dir = tmp_path / "snapshot"
    _materialize_upload_snapshot(staged_dataset, name=name, snapshot_dir=snapshot_dir)
    downloaded_root = tmp_path / "downloaded"
    downloaded_dataset = downloaded_root / "cua-lite" / name
    download_dataset(name, out_dir=downloaded_dataset, snapshot_dir=snapshot_dir)
    downloaded_row = _read_one_partition_row(downloaded_dataset)
    _assert_row_image_binding(
        downloaded_row,
        expected_indices=(0, 1),
        image_root=downloaded_root,
    )
    _assert_projected_error_once_in_messages(downloaded_row)
    downloaded_export = _export_row(downloaded_row, image_root=downloaded_root)
    downloaded_metadata = _assert_export_metadata_matches_row(
        downloaded_row,
        downloaded_export,
    )
    assert "stop_reason" not in downloaded_metadata["others"]
    downloaded_signature = _assert_export_shape(
        downloaded_export
    )
    _assert_projected_error_once_in_export(downloaded_export)
    _assert_sft_consumer_shape(downloaded_export, monkeypatch)
    assert downloaded_signature == raw_signature


def test_raw_response_replay_is_raw_only_not_canonical_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(export_sft, "_get_processor", lambda _model_id: _FakeQwenProcessor())

    raw_parquet = _write_logger_run(tmp_path / "logs" / "train" / "task_raw_response" / "sample_00")
    row = pd.read_parquet(raw_parquet).to_dict("records")[0]
    messages = coerce_messages(row["messages"])
    messages[1]["raw_response"] = {
        "adapter_key": "qwen3_vl@desktop@use",
        "text": "same-family raw native click",
    }
    row["messages"] = messages
    row["metadata"] = coerce_meta(row["metadata"])

    raw_export = _export_row(row, image_root=None)
    assert raw_export["_error"] == ""
    assert "same-family raw native click" in raw_export["steps"][0]["response"]

    write_records_to_parquet(
        [row],
        raw_parquet,
        json_fields=("messages", "metadata"),
    )

    name = "RawResponseCanonicalBoundary"
    staged_root = tmp_path / "staged"
    staged_dataset = staged_root / "cua-lite" / name
    stage([tmp_path / "logs"], name=name, out_dir=staged_dataset, filter_expr=None)
    staged_row = _read_one_partition_row(staged_dataset)
    assert "raw_response" not in coerce_messages(staged_row["messages"])[1]

    canonical_export = _export_row(staged_row, image_root=staged_root)
    assert canonical_export["_error"] == ""
    assert "same-family raw native click" not in canonical_export["steps"][0]["response"]
    assert "computer_use" in canonical_export["steps"][0]["response"]


def test_goal_image_order_survives_raw_stage_download_export_sft_contract(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(export_sft, "_get_processor", lambda _model_id: _FakeQwenProcessor())

    raw_parquet = _write_goal_image_logger_run(
        tmp_path / "logs" / "train" / "task_goal_image_path" / "sample_00"
    )
    raw_row = pd.read_parquet(raw_parquet).to_dict("records")[0]
    _assert_row_image_binding(raw_row, expected_indices=(1, 2, 0, 3))
    agent_kwargs = {
        "protocol_key": "browsergym.goal_image.qwen3_vl.history",
        "protocol_kwargs": {"full_history_size": 1},
    }
    raw_signature = _assert_goal_image_export_shape(
        _export_row(raw_row, image_root=None, agent_kwargs=agent_kwargs)
    )

    name = "GoalImagePathE2E"
    staged_root = tmp_path / "staged"
    staged_dataset = staged_root / "cua-lite" / name
    stage([tmp_path / "logs"], name=name, out_dir=staged_dataset, filter_expr=None)
    staged_row = _read_one_partition_row(staged_dataset)
    _assert_row_image_binding(
        staged_row,
        expected_indices=(1, 2, 0, 3),
        image_root=staged_root,
    )
    staged_signature = _assert_goal_image_export_shape(
        _export_row(
            staged_row,
            image_root=staged_root,
            agent_kwargs=agent_kwargs,
        )
    )
    assert staged_signature == raw_signature

    snapshot_dir = tmp_path / "snapshot"
    _materialize_upload_snapshot(staged_dataset, name=name, snapshot_dir=snapshot_dir)
    downloaded_root = tmp_path / "downloaded"
    downloaded_dataset = downloaded_root / "cua-lite" / name
    download_dataset(name, out_dir=downloaded_dataset, snapshot_dir=snapshot_dir)
    downloaded_row = _read_one_partition_row(downloaded_dataset)
    _assert_row_image_binding(
        downloaded_row,
        expected_indices=(1, 2, 0, 3),
        image_root=downloaded_root,
    )
    downloaded_signature = _assert_goal_image_export_shape(
        _export_row(
            downloaded_row,
            image_root=downloaded_root,
            agent_kwargs=agent_kwargs,
        )
    )
    assert downloaded_signature == raw_signature

