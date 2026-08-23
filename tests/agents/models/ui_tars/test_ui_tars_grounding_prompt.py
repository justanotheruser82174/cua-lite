"""UI-TARS grounding.point prompt surface."""

from __future__ import annotations

import dataclasses

import pytest
from lite_samples import sample_grounding_point

# Import for registration side effects.
import lite.agents.models.ui_tars.adapter  # noqa: F401
from lite.agents.core.action_space.base import LitePointActionSpace
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.models.ui_tars.action_space import (
    UITarsDesktopGroundingPointActionSpace,
)

GROUNDING_ENVS = ("screenspot_pro", "osworld_g")
EXPECTED_POINT_BLOCK = "click(start_box='<|box_start|>(x1,y1)<|box_end|>')"


def _sample_for_env(env_id: str):
    sample = sample_grounding_point()
    return dataclasses.replace(
        sample,
        metadata=dataclasses.replace(
            sample.metadata,
            dims=("desktop", "grounding.point"),
            valid_actions=["point"],
            others={
                "resolution": [1920, 1080],
                "source": "test",
                "env_id": env_id,
                "task_id": f"{env_id}.fixture",
            },
        ),
    )


def _prompt_text(adapter, sample) -> str:
    for message in adapter.unroll(sample).steps[-1]:
        for part in message.get("content", []):
            text = part.get("text")
            if isinstance(text, str) and "## Action Space" in text:
                return text
    raise AssertionError("rendered prompt did not contain an action-space block")


def _action_space_block(prompt: str) -> str:
    return prompt.split("## Action Space", 1)[1].split("## User Instruction", 1)[0].strip()


@pytest.mark.parametrize("env_id", GROUNDING_ENVS)
def test_ui_tars_grounding_valid_actions_point_renders_only_point_surface(
    env_id: str,
) -> None:
    sample = _sample_for_env(env_id)
    adapter = AgentAdapterRegistry.get(
        "ui_tars@desktop@grounding.point",
        metadata=sample.metadata,
    )

    action_space = _action_space_block(_prompt_text(adapter, sample))

    assert action_space == EXPECTED_POINT_BLOCK
    for leaked in (
        "left_double(",
        "right_single(",
        "drag(",
        "hotkey(",
        "type(",
        "scroll(",
        "wait(",
        "finished(",
        "call_user(",
    ):
        assert leaked not in action_space, (
            f"ui_tars@desktop@grounding.point for {env_id} leaked "
            f"{leaked!r} into {action_space!r}"
        )


def test_ui_tars_grounding_to_agent_projects_to_bare_model_function_projection() -> None:
    calls = UITarsDesktopGroundingPointActionSpace().convert_tool_calls_to_agent(
        [
            LitePointActionSpace.point(coordinate=[850, 120]),
        ]
    )

    assert calls == [{"name": "click", "arguments": {"start_box": [850, 120]}}]
    assert set(calls[0]) == {"name", "arguments"}
