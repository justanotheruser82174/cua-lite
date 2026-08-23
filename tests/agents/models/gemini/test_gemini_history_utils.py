"""Gemini provider-history utility behavior."""

from __future__ import annotations

from typing import Any

from agents.models._support.provider_fakes import png_bytes

from lite.agents.core.agent.utils.loop import record_lite_env_result
from lite.agents.models.gemini.action_space import GeminiDesktopActionSpace
from lite.agents.models.gemini.utils.history import append_gemini_provider_feedback
from lite.agents.models.gemini.utils.loop import _append_gemini_terminal_tool_feedback
from lite.agents.models.gemini.utils.parse import parse_response_with_provenance
from lite.core import LiteCUAMetadata, LiteRLStep, LiteSample
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


async def _append_gemini_unpaired_feedback(
    step_result: LiteEnvStepResult,
) -> tuple[list[dict[str, Any]], LiteSample, tuple[int, ...]]:
    contents: list[dict[str, Any]] = []
    trajectory = _sample()
    _, image_index = await append_gemini_provider_feedback(
        contents=contents,
        trajectory=trajectory,
        step_result=step_result,
        lite_tool_calls=[],
        provider_calls=(),
        resize_target=None,
    )
    return contents, trajectory, () if image_index is None else (image_index,)


async def test_gemini_unpaired_feedback_preserves_text_and_latest_image():
    contents, trajectory, image_indices = await _append_gemini_unpaired_feedback(
        _step_result(text="try again", images=[png_bytes(8, 6)])
    )

    assert image_indices == (0,)
    assert contents == [
        {
            "role": "user",
            "parts": [
                {"text": "try again"},
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": contents[0]["parts"][1]["inlineData"]["data"],
                    },
                    "_cua_lite_image_index": 0,
                },
            ],
        }
    ]
    assert contents[0]["parts"][1]["inlineData"]["data"]
    assert trajectory.messages == [
        {
            "role": "user",
            "content": [{"type": "image", "index": 0}, {"type": "text", "text": "try again"}],
        }
    ]


async def test_gemini_unpaired_feedback_preserves_empty_string_text():
    contents, trajectory, image_indices = await _append_gemini_unpaired_feedback(
        _step_result(text="")
    )

    assert image_indices == ()
    assert trajectory.messages == [{"role": "user", "content": [{"type": "text", "text": ""}]}]
    assert contents == [{"role": "user", "parts": [{"text": ""}]}]


def _parsed_gemini_tool_calls() -> list[dict[str, Any]]:
    parsed = parse_response_with_provenance(
        {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "id": "provider_call_1",
                                    "name": "take_screenshot",
                                    "args": {},
                                }
                            }
                        ],
                    },
                }
            ]
        },
        action_space=GeminiDesktopActionSpace(),
        admissible_verbs=GeminiDesktopActionSpace().visible_predefined_verbs(
            platform="desktop",
            valid_actions=None,
        ),
    )
    return parsed.message["tool_calls"]


async def test_gemini_max_steps_one_tool_call_persists_terminal_feedback():
    tool_calls = _parsed_gemini_tool_calls()
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
        append_terminal_tool_feedback=_append_gemini_terminal_tool_feedback,
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
