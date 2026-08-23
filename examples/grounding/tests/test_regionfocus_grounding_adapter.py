from __future__ import annotations

import io
import json

import pandas as pd
import pytest
from PIL import Image

from examples.grounding.adapter import (
    Qwen3_5RegionFocusGroundingAgent,
    Qwen3VLRegionFocusGroundingAgent,
)
from examples.grounding.utils import rf_metrics
from lite.agents.core.agent.logger import TrajectoryLogger
from lite.core import (
    STATUS_FAILED,
    LiteCUAMetadata,
)
from lite.core.messages.final import pop_model_output_error
from lite.core.messages.turns import count_sample_turns
from lite.core.tools.calls import tool_call_arguments, tool_call_id, tool_call_name
from lite.data.staging import coerce_messages, coerce_meta
from lite.data.utils.rows import validate_canonical_rows
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult


def _png_bytes(color: str = "white") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, format="PNG")
    return buf.getvalue()


def _qwen_point(x: int, y: int) -> str:
    payload = {
        "name": "computer_use",
        "arguments": {"action": "left_click", "coordinate": [x, y]},
    }
    return f"<tool_call>\n{json.dumps(payload)}\n</tool_call>"


def _qwen35_point(x: int, y: int) -> str:
    return (
        "<tool_call>\n"
        "<function=computer_use>\n"
        "<parameter=action>\nleft_click\n</parameter>\n"
        "<parameter=coordinate>\n"
        f"[{x}, {y}]\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )


class _FakeProcessor:
    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
        enable_thinking: bool = False,
    ) -> str:
        rendered: list[str] = []
        for message in messages:
            rendered.append(f"<{message.get('role')}>")
            for part in message.get("content") or []:
                if part.get("type") == "image":
                    rendered.append("<|image_pad|>")
                elif part.get("type") == "text":
                    rendered.append(part.get("text", ""))
            for call in message.get("tool_calls") or []:
                rendered.append(f"<tool:{tool_call_name(call)}>")
        if add_generation_prompt:
            rendered.append("<assistant>")
        return "\n".join(rendered)


class _FakeGroundingEnv:
    def __init__(self) -> None:
        self.metadata = LiteCUAMetadata(
            dims=(
                LiteCUAMetadata.Platform.DESKTOP,
                LiteCUAMetadata.TaskType.GROUNDING_POINT,
            ),
            others={"env_id": "regionfocus.test", "task_id": "task"},
        )
        self.actions: list[dict] | None = None
        self.closed = False

    async def reset(self) -> LiteEnvObservation:
        return LiteEnvObservation(image=_png_bytes(), text="Click the target")

    async def step(self, actions: list[dict]) -> LiteEnvStepResult:
        self.actions = actions
        return LiteEnvStepResult(
            reward=1.0,
            terminated=True,
            truncated=False,
            info={
                "annotation": {
                    "box_type": "bbox",
                    "box_coordinates": [0, 0, 64, 64],
                    "image_size": [64, 64],
                }
            },
        )

    async def close(self) -> None:
        self.closed = True


def _agent_with_responses(
    responses: list[str],
    *,
    crop_count: int = 2,
) -> Qwen3VLRegionFocusGroundingAgent:
    response_iter = iter(responses)

    async def generate_fn(**kwargs):
        return {
            "response": next(response_iter),
            "response_tokens": [1],
            "response_log_probs": [0.0],
        }

    agent = Qwen3VLRegionFocusGroundingAgent(
        processor=_FakeProcessor(),
        generate_fn=generate_fn,
    )
    agent._crop_and_upsample = lambda image, seed_point: [
        {"image": image.copy(), "norm_box": (0, 0, 1000, 1000)}
        for _ in range(crop_count)
    ]
    return agent


@pytest.mark.asyncio
async def test_regionfocus_sample_persists_trainable_path_without_index_drift(tmp_path):
    agent = _agent_with_responses([
        _qwen_point(10, 10),   # initial
        "NO - refine it",     # judge
        _qwen_point(20, 20),   # seed
        _qwen_point(30, 30),   # refine candidate 1
        _qwen_point(50, 50),   # refine candidate 2
        "Selected point: 2",   # aggregate
    ])

    env = _FakeGroundingEnv()
    logger = TrajectoryLogger(tmp_path / "run")

    sample = await agent.sample(env, hooks=[logger])

    assert env.closed is True
    assert env.actions is not None
    assert tool_call_id(env.actions[0]) == "call_0004"
    assert len(sample.steps) == 6
    assert len(sample.lite_sample.images) == len(sample.processed_images) == 5
    assert [tuple(step.image_indices) for step in sample.steps] == [
        (0,),  # initial screenshot
        (1,),  # judge marker image
        (0,),  # seed on original screenshot
        (2,),  # crop 1
        (3,),  # crop 2
        (4,),  # aggregate marker image
    ]
    for step in sample.steps:
        assert step.prompt.count("<|image_pad|>") == len(step.image_indices)

    row = pd.read_parquet(tmp_path / "run" / "trajectory.parquet").iloc[0]
    messages = coerce_messages(row["messages"])
    metadata = coerce_meta(row["metadata"])
    image_paths = list(row["images"])
    assert len(image_paths) == 5
    assert metadata["metadata_kind"] == "cua"
    assert metadata["dims"] == ["desktop", "grounding.point"]
    assert "platform" not in metadata
    assert "task_type" not in metadata

    referenced_images = {
        part["index"]
        for message in messages
        for part in message.get("content", [])
        if part.get("type") == "image"
    }
    assert referenced_images == set(range(len(image_paths)))

    calls = [call for message in messages for call in (message.get("tool_calls") or [])]
    assert [tool_call_id(call) for call in calls] == [
        "call_0000",
        "call_0001",
        "call_0002",
        "call_0003",
    ]
    tool_results = [
        message["tool_call_id"]
        for message in messages
        if message.get("role") == "tool"
    ]
    assert tool_results == []
    assert count_sample_turns(sample.lite_sample) == len(sample.steps)
    assert messages[-1]["regionfocus_trace"]["path"] == "regionfocus"
    validate_canonical_rows(
        [{"images": image_paths, "messages": messages, "metadata": metadata}],
        "regionfocus/trajectory",
    )


@pytest.mark.asyncio
async def test_regionfocus_sample_final_action_uses_aggregate_selected_candidate():
    agent = _agent_with_responses([
        _qwen_point(10, 10),   # initial
        "NO - refine it",     # judge
        _qwen_point(20, 20),   # seed
        _qwen_point(30, 30),   # refine candidate 1
        _qwen_point(50, 50),   # refine candidate 2
        "Selected point: 1",   # aggregate chooses the earlier candidate
    ])
    env = _FakeGroundingEnv()

    sample = await agent.sample(env)

    assert env.actions is not None
    assert tool_call_name(env.actions[0]) == "point"
    assert tool_call_arguments(env.actions[0])["coordinate"] == [30, 30]
    assert tool_call_id(env.actions[0]) == "call_0004"
    final_message = sample.lite_sample.messages[-1]
    trace = final_message["regionfocus_trace"]
    assert trace["path"] == "regionfocus"
    assert trace["selected"]["point"] == [30, 30]
    assert trace["selected"]["aggregation_selected_index"] == 0


@pytest.mark.asyncio
async def test_regionfocus_sample_refine_empty_seed_final_action_uses_seed():
    agent = _agent_with_responses([
        _qwen_point(10, 10),  # initial
        "NO - refine it",    # judge
        _qwen_point(20, 20),  # seed
        "not a point",       # invalid refine candidate 1
        "still no point",    # invalid refine candidate 2
    ])
    env = _FakeGroundingEnv()

    sample = await agent.sample(env)

    assert env.actions is not None
    assert tool_call_name(env.actions[0]) == "point"
    assert tool_call_arguments(env.actions[0])["coordinate"] == [20, 20]
    assert tool_call_id(env.actions[0]) == "call_0002"
    assert len(sample.steps) == 3
    final_message = sample.lite_sample.messages[-1]
    assert tool_call_arguments(final_message["tool_calls"][0])["coordinate"] == [20, 20]
    trace = final_message["regionfocus_trace"]
    assert trace["path"] == "refine_empty_seed"
    assert trace["selected"]["point"] == [20, 20]


@pytest.mark.asyncio
async def test_regionfocus_replay_uses_online_prompts_and_crop_local_targets():
    agent = _agent_with_responses([
        _qwen_point(10, 10),     # initial
        "NO - refine it",       # judge
        _qwen_point(500, 500),   # seed
        _qwen_point(500, 500),   # refine candidate 1 -> original [750, 500]
        _qwen_point(500, 500),   # refine candidate 2 -> original [250, 500]
        "Selected point: 2",     # aggregate
    ])
    agent._crop_and_upsample = lambda image, seed_point: [
        {"image": image.copy(), "norm_box": (500, 0, 500, 1000)},
        {"image": image.copy(), "norm_box": (0, 0, 500, 1000)},
    ]
    env = _FakeGroundingEnv()

    sample = await agent.sample(env)

    assert env.actions is not None
    assert tool_call_arguments(env.actions[0])["coordinate"] == [250, 500]
    assert count_sample_turns(sample.lite_sample) == len(sample.steps) == 6

    calls = [
        call
        for message in sample.lite_sample.messages
        for call in (message.get("tool_calls") or [])
    ]
    assert [tool_call_arguments(call)["coordinate"] for call in calls] == [
        [10, 10],
        [500, 500],
        [500, 500],
        [500, 500],
    ]

    agent_sample = agent.adapter.unroll(sample.lite_sample)
    assert len(agent_sample.steps) == len(sample.steps)
    judge_system = agent_sample.steps[1][0]["content"][0]["text"]
    aggregate_system = agent_sample.steps[-1][0]["content"][0]["text"]
    assert "Judge whether the highlighted point is correct" in judge_system
    assert "precise UI grounding selector" in aggregate_system
    assert "# Tools" not in judge_system
    assert "# Tools" not in aggregate_system

    trace = sample.lite_sample.messages[-1]["regionfocus_trace"]
    assert trace["selected"]["point"] == [250, 500]
    assert trace["selected"]["local_point"] == [500, 500]


@pytest.mark.asyncio
async def test_regionfocus_no_tool_response_is_terminal_model_output_failure(tmp_path):
    agent = _agent_with_responses(["not a tool call"])
    env = _FakeGroundingEnv()
    logger = TrajectoryLogger(tmp_path / "run")

    sample = await agent.sample(env, hooks=[logger])

    assert env.closed is True
    assert env.actions is None
    assert sample.terminated is True
    assert sample.steps[0].status == STATUS_FAILED
    final_message = sample.lite_sample.messages[-1]
    assert final_message["regionfocus_trace"]["path"] == "initial_invalid"
    assert pop_model_output_error(final_message) == (
        "RegionFocus final response produced no tool_calls"
    )

    actions = json.loads((tmp_path / "run" / "turn_0000" / "03_actions.json").read_text())
    results = json.loads((tmp_path / "run" / "turn_0000" / "04_results.json").read_text())
    assert actions["executed_actions"] == []
    assert actions["lite_message"]["regionfocus_trace"]["path"] == "initial_invalid"
    assert (
        actions["lite_message"]["_lite_model_output_error"]
        == "RegionFocus final response produced no tool_calls"
    )
    assert results["reward"] == 0.0
    assert results["terminated"] is True
    assert results["truncated"] is False
    assert results["info"]["executed_actions"] == []
    assert (
        results["info"]["model_output_error"]
        == "RegionFocus final response produced no tool_calls"
    )


@pytest.mark.asyncio
async def test_regionfocus_crop_refine_oob_projection_is_recorded_and_skipped():
    agent = _agent_with_responses([
        _qwen_point(10, 10),       # initial
        "NO - refine it",         # judge
        _qwen_point(500, 500),     # seed
        _qwen_point(1200, 1200),   # crop local point outside [0, 1000]
    ], crop_count=1)
    agent._crop_and_upsample = lambda image, seed_point: [
        {"image": image.copy(), "norm_box": (900, 900, 100, 100)}
    ]
    env = _FakeGroundingEnv()

    sample = await agent.sample(env)

    assert env.actions is not None
    assert tool_call_arguments(env.actions[0])["coordinate"] == [500, 500]
    final_message = sample.lite_sample.messages[-1]
    trace = final_message["regionfocus_trace"]
    assert trace["path"] == "refine_empty_seed"
    result = trace["refine_candidates"][0]["result"]
    assert result["point"] is None
    assert result["local_point"] == [1200, 1200]
    assert "outside normalized [0,1000]" in result["projection_error"]


@pytest.mark.asyncio
async def test_qwen35_regionfocus_point_prompts_keep_tools_before_response_format():
    seen: dict[str, str] = {}

    async def generate_fn(**kwargs):
        seen["prompt"] = kwargs["prompt"]
        return {
            "response": _qwen35_point(10, 10),
            "response_tokens": [1],
            "response_log_probs": [0.0],
        }

    agent = Qwen3_5RegionFocusGroundingAgent(
        processor=_FakeProcessor(),
        generate_fn=generate_fn,
    )

    parsed = await agent._generate_point_on_image(
        image=Image.new("RGB", (64, 64), "white"),
        prompt_text="Click the target.",
    )

    assert parsed is not None
    assert seen["prompt"].index("# Tools") < seen["prompt"].index("# Response format")


def test_regionfocus_annotation_evaluator_supports_shared_grounding_shapes():
    assert rf_metrics.evaluate_point_against_annotation(
        {"bbox": [10, 20, 30, 40], "img_size": [100, 100]},
        [200, 300],
    ) is True
    assert rf_metrics.evaluate_point_against_annotation(
        {"bbox": [10, 20, 30, 40], "img_size": [100, 100]},
        [900, 300],
    ) is False
    assert rf_metrics.evaluate_point_against_annotation(
        {
            "box_type": "bbox",
            "box_coordinates": [10, 20, 30, 40],
            "image_size": [100, 100],
        },
        [200, 300],
    ) is True
    assert rf_metrics.evaluate_point_against_annotation(
        {
            "box_type": "polygon",
            "box_coordinates": [10, 10, 50, 10, 50, 50, 10, 50],
            "image_size": [100, 100],
        },
        [200, 200],
    ) is True
