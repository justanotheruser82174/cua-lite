"""GPT provider-history utility behavior."""

from __future__ import annotations

from typing import Any

from agents.models._support.provider_fakes import png_bytes

from lite.agents.core.agent.utils.loop import execute_lite_turn, record_lite_env_result
from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
from lite.agents.models.gpt.utils.history import (
    _append_desktop_provider_feedback,
    _format_computer_action,
)
from lite.agents.models.gpt.utils.loop import _append_gpt_terminal_tool_feedback
from lite.agents.models.gpt.utils.parse import parse_output_items_with_provenance
from lite.core import LiteCUAMetadata, LiteRLStep, LiteSample
from lite.core.messages.final import CONTENT_ONLY_FINAL_REASON, STOP_REASON_INFO_KEY
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.results import LiteToolResult
from lite.gym.types import LiteEnvStepResult


def _sample() -> LiteSample:
    return LiteSample(metadata=LiteCUAMetadata())


def _step_result(*, text: str | None, images: list[bytes] | None = None) -> LiteEnvStepResult:
    return LiteEnvStepResult(
        results=[
            LiteToolResult(
                tool_call_id=None,
                images=list(images or []),
                text=text,
            )
        ],
        reward=0.0,
        terminated=False,
    )


async def _append_gpt_unpaired_feedback(
    step_result: LiteEnvStepResult,
) -> tuple[list[dict[str, Any]], LiteSample, tuple[int, ...]]:
    input_items: list[dict[str, Any]] = []
    trajectory = _sample()
    _, image_indices = await _append_desktop_provider_feedback(
        input_items=input_items,
        output_items=[],
        parsed_provider_calls=(),
        trajectory=trajectory,
        step_result=step_result,
        lite_tool_calls=[],
        resolution=None,
        sent_image_b64=None,
        sent_image_index=None,
        detail="original",
        use_chaining=True,
        model_output_error=None,
    )
    return input_items, trajectory, image_indices


async def test_gpt_unpaired_feedback_preserves_text_and_latest_image():
    input_items, trajectory, image_indices = await _append_gpt_unpaired_feedback(
        _step_result(text="try again", images=[png_bytes(8, 6)])
    )

    assert image_indices == (0,)
    assert input_items == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "try again"},
                {
                    "type": "input_image",
                    "image_url": input_items[0]["content"][1]["image_url"],
                    "detail": "original",
                    "_cua_lite_image_index": 0,
                },
            ],
        }
    ]
    assert input_items[0]["content"][1]["image_url"].startswith("data:image/png;base64,")
    assert trajectory.messages == [
        {
            "role": "user",
            "content": [{"type": "image", "index": 0}, {"type": "text", "text": "try again"}],
        }
    ]


async def test_gpt_unpaired_feedback_preserves_empty_string_text():
    input_items, trajectory, image_indices = await _append_gpt_unpaired_feedback(
        _step_result(text="")
    )

    assert image_indices == ()
    assert trajectory.messages == [{"role": "user", "content": [{"type": "text", "text": ""}]}]
    assert input_items == [{"role": "user", "content": [{"type": "input_text", "text": ""}]}]


def _parsed_gpt_tool_calls() -> list[dict[str, Any]]:
    parsed = parse_output_items_with_provenance(
        [
            {
                "type": "computer_call",
                "call_id": "provider_call_1",
                "actions": [{"type": "screenshot"}],
            }
        ],
        action_space=GPTDesktopActionSpace(),
        resolution=(800, 600),
        active_provider_tool_names=frozenset({"computer"}),
    )
    return parsed.message["tool_calls"]


def test_gpt_compacted_keypress_history_uses_unambiguous_key_list():
    action = {"actions": [{"type": "keypress", "keys": ["ctrl", "+"]}]}

    assert _format_computer_action(action) == "keypress(['ctrl', '+'])"


async def test_gpt_max_steps_one_tool_call_persists_terminal_feedback():
    tool_calls = _parsed_gpt_tool_calls()
    trajectory = _sample()
    rl_step = LiteRLStep(prompt="{}", image_indices=(0,), response="{}")
    step_result = LiteEnvStepResult(
        results=[
            LiteToolResult(
                tool_call_id=tool_calls[0]["id"],
                images=[png_bytes(8, 6)],
                text="screen after action",
            )
        ],
        reward=0.0,
        terminated=False,
    )

    _, terminated, truncated = await record_lite_env_result(
        trajectory=trajectory,
        persisted_actions=tool_calls,
        step_result=step_result,
        rl_step=rl_step,
        step=0,
        max_steps=1,
        episode_return=0.0,
        append_terminal_tool_feedback=_append_gpt_terminal_tool_feedback,
    )

    assert terminated is False
    assert truncated is True
    assert step_result.truncated is True
    assert rl_step.status == "truncated"
    assert trajectory.messages == [
        {
            "role": "tool",
            "tool_call_id": tool_calls[0]["id"],
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "screen after action"},
            ],
        }
    ]


class _FirstResponseTerminatesEnv:
    async def step(self, actions):
        self.actions = list(actions)
        return LiteEnvStepResult(reward=1.0, terminated=True, results=[])


async def test_content_only_response_that_env_terminates_on_first_turn_stays_terminal():
    env = _FirstResponseTerminatesEnv()
    trajectory = _sample()
    trajectory.messages.append(
        {"role": "assistant", "content": [{"type": "text", "text": "done"}], "tool_calls": []}
    )
    rl_step = LiteRLStep(prompt="{}", image_indices=(), response="done")

    turn = await execute_lite_turn(
        env=env,
        trajectory=trajectory,
        lite_message=trajectory.messages[-1],
        model_output_error=None,
        rl_step=rl_step,
        tool_execution_timeout_s=None,
    )
    _, terminated, truncated = await record_lite_env_result(
        trajectory=trajectory,
        persisted_actions=turn.persisted_actions,
        step_result=turn.step_result,
        rl_step=rl_step,
        step=0,
        max_steps=1,
        episode_return=0.0,
        append_terminal_tool_feedback=_append_gpt_terminal_tool_feedback,
    )

    assert terminated is True
    assert truncated is False
    assert turn.persisted_actions == []
    assert tool_call_name(env.actions[0]) == "response"
    assert tool_call_arguments(env.actions[0]) == {"text": "done"}
    assert turn.step_result.info[STOP_REASON_INFO_KEY] == CONTENT_ONLY_FINAL_REASON
    assert trajectory.messages == [
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]}
    ]
