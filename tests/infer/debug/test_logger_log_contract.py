"""Debug log-contract checks for directories written by TrajectoryLogger."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from lite.agents.core.agent.hooks import SampleStepData
from lite.agents.core.agent.logger import TrajectoryLogger
from lite.agents.types import PredictResult
from lite.core import LiteCUAMetadata, LiteRLSample, LiteRLStep, LiteSample
from lite.core.tools.calls import make_tool_call
from lite.core.tools.results import LiteToolResult
from lite.data.staging import coerce_messages
from lite.gym.types import LiteEnvStepResult
from lite.infer.debug.log_contract import check_log_contract

_NATIVE_OBSERVATION_TEXT = "## AXTree:\nbutton selected"
_NATIVE_RESULT_ERROR = "click was outside active element"
_PROJECTED_ERROR_TEXT = (
    f"{_NATIVE_OBSERVATION_TEXT}\n\n"
    f"## Error from previous action:\n{_NATIVE_RESULT_ERROR}"
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


def test_logger_log_contract_checks_env_result_images_are_debug_only(tmp_path):
    raw_parquet = _write_logger_run(
        tmp_path / "logs" / "train" / "task_result_image_binding" / "sample_00"
    )
    sample_dir = raw_parquet.parent
    assert check_log_contract(sample_dir) == []

    results_path = sample_dir / "turn_0000" / "04_results.json"
    payload = json.loads(results_path.read_text())
    image_info = payload["results"][0]["images"][0]
    assert image_info["path"] == "env_result_images/0000_0000_from_call_0000.png"
    assert image_info["source"] == "env_result_images"
    assert (sample_dir / "turn_0000" / image_info["path"]).is_file()
    assert not (sample_dir / "turn_0000" / "result_images").exists()

    payload["results"][0]["images"][0] = {
        **image_info,
        "path": "images/000001.png",
    }
    results_path.write_text(json.dumps(payload))

    reasons = check_log_contract(sample_dir)
    assert any("image.path must be relative under env_result_images/" in r for r in reasons)


def test_logger_debug_prompt_image_caches_are_renamed(tmp_path):
    raw_parquet = _write_logger_run(
        tmp_path / "logs" / "train" / "task_debug_prompt_cache" / "sample_00",
        debug_artifacts=True,
    )
    sample_dir = raw_parquet.parent
    turn0 = sample_dir / "turn_0000"
    turn1 = sample_dir / "turn_0001"

    assert (turn0 / "prompt_images" / "0000_reset.png").is_file()
    assert (turn0 / "prompt_images_annotated" / "0000_reset.png").is_file()
    assert (turn1 / "prompt_images" / "0001_from_call_0000.png").is_file()
    assert not (turn0 / "images").exists()
    assert not (turn0 / "annotated").exists()
    assert not (turn1 / "images").exists()
    assert not (turn1 / "annotated").exists()
    assert check_log_contract(sample_dir) == []


def test_logger_log_contract_check_is_read_only_for_parquet(tmp_path):
    raw_parquet = _write_logger_run(
        tmp_path / "logs" / "train" / "task_debug_invariant" / "sample_00"
    )
    sample_dir = raw_parquet.parent
    before_row = pd.read_parquet(raw_parquet).to_dict("records")[0]

    assert check_log_contract(sample_dir) == []

    after_row = pd.read_parquet(raw_parquet).to_dict("records")[0]

    assert list(after_row["images"]) == list(before_row["images"])
    assert coerce_messages(after_row["messages"]) == coerce_messages(before_row["messages"])
