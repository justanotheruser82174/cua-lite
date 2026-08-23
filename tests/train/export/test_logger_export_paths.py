"""SFT export checks for raw parquet rows produced by TrajectoryLogger."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from lite.agents.core.agent.hooks import SampleStepData
from lite.agents.core.agent.logger import TrajectoryLogger
from lite.agents.types import PredictResult
from lite.core import LiteCUAMetadata, LiteRLSample, LiteRLStep, LiteSample
from lite.core.tools.calls import make_tool_call
from lite.core.tools.results import LiteToolResult
from lite.data.staging import coerce_messages
from lite.gym.types import LiteEnvStepResult
from lite.train.export import export_sft

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


def _write_terminal_tool_image_logger_run(log_dir: Path) -> Path:
    images = [
        Image.new("RGB", (96, 64), (30, 40, 50)),
        Image.new("RGB", (96, 64), (210, 40, 50)),
    ]
    metadata = LiteCUAMetadata(
        dims=("desktop", "use"),
        extra_tool_schemas=[],
        valid_actions=["click", "wait"],
        others={"env_id": "logger.test", "task_id": "task_terminal_tool_image"},
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "Click once and stop on terminal feedback."},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "click once"}],
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
                {"type": "text", "text": "terminal image-bearing feedback"},
            ],
        },
    ]
    logger = TrajectoryLogger(
        log_dir,
        env_id="logger.test",
        task_id="task_terminal_tool_image",
        provenance={"model_id": "gpt-5.5", "agent_id": "gpt.desktop"},
    )
    logger.on_complete(
        LiteRLSample(
            lite_sample=LiteSample(metadata=metadata, images=images, messages=messages),
            processed_images=list(images),
            steps=[
                LiteRLStep(prompt="prompt 0", image_indices=(0,), response="raw 0"),
                LiteRLStep(prompt="prompt 1", image_indices=(0, 1), response=""),
            ],
            episode_return=1.0,
            terminated=True,
            truncated=False,
        )
    )
    return log_dir / "trajectory.parquet"


def _write_byte_identical_logger_run(log_dir: Path) -> Path:
    image = Image.new("RGB", (96, 64), (77, 88, 99))
    images = [image.copy(), image.copy()]
    metadata = LiteCUAMetadata(
        dims=("desktop", "use"),
        extra_tool_schemas=[],
        valid_actions=["click", "wait"],
        others={"env_id": "logger.test", "task_id": "task_byte_identical_images"},
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "image", "index": 1},
                {"type": "text", "text": "Compare the two screenshots."},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
    ]
    logger = TrajectoryLogger(
        log_dir,
        env_id="logger.test",
        task_id="task_byte_identical_images",
        provenance={"model_id": "gpt-5.5", "agent_id": "gpt.desktop"},
    )
    logger.on_complete(
        LiteRLSample(
            lite_sample=LiteSample(metadata=metadata, images=images, messages=messages),
            processed_images=list(images),
            steps=[LiteRLStep(prompt="prompt 0", image_indices=(0, 1), response="Done.")],
            episode_return=1.0,
            terminated=True,
            truncated=False,
        )
    )
    return log_dir / "trajectory.parquet"


def _parallel_goal_image_messages() -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 1},
                {"type": "image", "index": 2},
                {"type": "image", "index": 0},
                {"type": "text", "text": "Find these references, then finish."},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "inspect both targets"}],
            "tool_calls": [
                make_tool_call(
                    "computer",
                    {
                        "actions": [
                            {
                                "action": "click",
                                "coordinate": [100, 200],
                                "button": "left",
                            }
                        ]
                    },
                    call_id="call_0000",
                ),
                make_tool_call(
                    "computer",
                    {
                        "actions": [
                            {
                                "action": "click",
                                "coordinate": [300, 400],
                                "button": "left",
                            }
                        ]
                    },
                    call_id="call_0001",
                ),
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0000",
            "content": [
                {"type": "image", "index": 3},
                {"type": "text", "text": "first parallel result"},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0001",
            "content": [
                {"type": "image", "index": 4},
                {"type": "text", "text": "second parallel result"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
    ]


def _write_parallel_goal_image_logger_run(log_dir: Path) -> Path:
    images = [
        Image.new("RGB", (96, 64), (10, 20, 30)),
        Image.new("RGB", (96, 64), (200, 10, 10)),
        Image.new("RGB", (96, 64), (10, 200, 10)),
        Image.new("RGB", (96, 64), (10, 10, 200)),
        Image.new("RGB", (96, 64), (240, 240, 20)),
    ]
    metadata = LiteCUAMetadata(
        dims=("browser", "use"),
        extra_tool_schemas=[],
        valid_actions=["click"],
        others={
            "env_id": "browsergym.visualwebarena",
            "task_id": "task_goal_parallel_images",
        },
    )
    messages = _parallel_goal_image_messages()
    logger = TrajectoryLogger(
        log_dir,
        env_id="browsergym.visualwebarena",
        task_id="task_goal_parallel_images",
        provenance={"model_id": "gpt-5.5", "agent_id": "qwen3_5"},
    )
    logger.on_complete(
        LiteRLSample(
            lite_sample=LiteSample(metadata=metadata, images=images, messages=messages),
            processed_images=list(images),
            steps=[
                LiteRLStep(prompt="prompt 0", image_indices=(1, 2, 0), response="raw 0"),
                LiteRLStep(prompt="prompt 1", image_indices=(1, 2, 0, 3, 4), response="Done."),
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


def _assert_export_shape(out: dict) -> None:
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


def _processed_rgb(processed_blob: bytes) -> tuple[int, int, int]:
    return Image.open(io.BytesIO(processed_blob)).convert("RGB").getpixel((0, 0))


def _assert_qwen35_parallel_goal_image_export_shape(out: dict) -> None:
    assert out["_error"] == ""
    assert len(out["processed_images"]) == 5
    assert len(out["steps"]) == 2

    signature = []
    for step in out["steps"]:
        indices = tuple(step["image_indices"])
        assert step["prompt"].count("<|image_pad|>") == len(indices)
        signature.append((
            indices,
            tuple(_processed_rgb(out["processed_images"][idx]) for idx in indices),
        ))

    assert signature[0] == (
        (1, 2, 0),
        ((200, 10, 10), (10, 200, 10), (10, 20, 30)),
    )
    assert signature[1] == (
        (1, 2, 0, 3, 4),
        (
            (200, 10, 10),
            (10, 200, 10),
            (10, 20, 30),
            (10, 10, 200),
            (240, 240, 20),
        ),
    )
    assert out["steps"][0]["response"].count("<function=computer_use>") == 2


def test_terminal_image_bearing_role_tool_survives_raw_parquet_and_export(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(export_sft, "_get_processor", lambda _model_id: _FakeQwenProcessor())

    raw_parquet = _write_terminal_tool_image_logger_run(
        tmp_path / "logs" / "train" / "task_terminal_tool_image" / "sample_00"
    )
    raw_row = pd.read_parquet(raw_parquet).to_dict("records")[0]

    _assert_row_image_binding(raw_row, expected_indices=(0, 1))
    raw_paths = [Path(p) for p in raw_row["images"]]
    assert all(path.is_file() for path in raw_paths)
    messages = coerce_messages(raw_row["messages"])
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["content"][0] == {"type": "image", "index": 1}

    out = _export_row(raw_row, image_root=None)

    assert out["_error"] == ""
    assert len(out["processed_images"]) == 2
    assert [_processed_rgb(blob) for blob in out["processed_images"]] == [
        (30, 40, 50),
        (210, 40, 50),
    ]
    # The terminal tool observation is retained as a trajectory image, but it is
    # not converted into a synthetic SFT target because no later assistant turn exists.
    assert len(out["steps"]) == 1
    assert out["steps"][0]["image_indices"] == [0]
    rendered = "\n".join(
        f"{step['prompt']}\n{step['response']}" for step in out["steps"]
    )
    assert "terminal image-bearing feedback" not in rendered


def test_byte_identical_images_remain_positionally_distinct_through_export(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(export_sft, "_get_processor", lambda _model_id: _FakeQwenProcessor())

    raw_parquet = _write_byte_identical_logger_run(
        tmp_path / "logs" / "train" / "task_byte_identical_images" / "sample_00"
    )
    raw_row = pd.read_parquet(raw_parquet).to_dict("records")[0]

    _assert_row_image_binding(raw_row, expected_indices=(0, 1))
    out = _export_row(raw_row, image_root=None)

    assert out["_error"] == ""
    assert len(out["processed_images"]) == 2
    assert out["processed_images"][0] == out["processed_images"][1]
    assert out["steps"][0]["image_indices"] == [0, 1]
    assert out["steps"][0]["prompt"].count("<|image_pad|>") == 2


@pytest.mark.parametrize("image_dir_name", ["trajectory_images", "images"])
def test_raw_export_resolves_sample_local_image_dirs(
    tmp_path,
    monkeypatch,
    image_dir_name,
):
    monkeypatch.setattr(export_sft, "_get_processor", lambda _model_id: _FakeQwenProcessor())

    raw_parquet = _write_logger_run(
        tmp_path / "logs" / "train" / f"task_{image_dir_name}_paths" / "sample_00"
    )
    sample_dir = raw_parquet.parent
    row = pd.read_parquet(raw_parquet).to_dict("records")[0]

    rel_images = []
    for image_idx, source in enumerate(row["images"]):
        rel = Path(image_dir_name) / f"{image_idx:06d}.png"
        target = sample_dir / rel
        if target.resolve() != Path(source).resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(source).read_bytes())
        rel_images.append(str(rel))
    row["images"] = rel_images

    _assert_row_image_binding(row, expected_indices=(0, 1), image_root=sample_dir)
    out = _export_row(row, image_root=sample_dir)
    _assert_export_shape(out)


def test_debug_prompt_image_caches_are_ignored_by_export(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(export_sft, "_get_processor", lambda _model_id: _FakeQwenProcessor())

    raw_parquet = _write_logger_run(
        tmp_path / "logs" / "train" / "task_debug_prompt_cache" / "sample_00",
        debug_artifacts=True,
    )
    raw_row = pd.read_parquet(raw_parquet).to_dict("records")[0]

    assert all("/prompt_images/" not in str(path) for path in raw_row["images"])
    assert all("/prompt_images_annotated/" not in str(path) for path in raw_row["images"])
    _assert_export_shape(_export_row(raw_row, image_root=None))


def test_qwen35_goal_image_parallel_tool_images_keep_global_indices(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(export_sft, "_get_processor", lambda _model_id: _FakeQwenProcessor())

    raw_parquet = _write_parallel_goal_image_logger_run(
        tmp_path / "logs" / "train" / "task_goal_parallel_images" / "sample_00"
    )
    raw_row = pd.read_parquet(raw_parquet).to_dict("records")[0]

    _assert_row_image_binding(raw_row, expected_indices=(1, 2, 0, 3, 4))
    _assert_qwen35_parallel_goal_image_export_shape(
        _export_row(
            raw_row,
            image_root=None,
            agent_id="qwen3_5",
            agent_kwargs={
                "protocol_key": "browsergym.goal_image.qwen3_5.history",
                "protocol_kwargs": {"history_n": 2, "image_max": 10},
            },
        )
    )
