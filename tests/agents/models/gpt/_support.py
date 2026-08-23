"""Shared support for GPT agent characterization tests."""

from __future__ import annotations

import asyncio
import base64
import io
from typing import Any
from unittest.mock import AsyncMock

from agents.models._support.provider_fakes import png_bytes
from PIL import Image

from lite.core import (
    LiteCUAMetadata,
    LiteSample,
)
from lite.core.tools.calls import tool_call_arguments, tool_call_id, tool_call_name
from lite.core.tools.extra_tools import BASH_TOOL_NAME, LiteShellToolSet
from lite.core.tools.results import LiteToolResult
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult

# The production bash schema, so env-side admission in these tests is the real one:
# it requires ``command``.
_BASH_SCHEMA = LiteShellToolSet.get_tool_schema(BASH_TOOL_NAME)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _make_sample(w: int = 800, h: int = 600) -> LiteSample:
    meta = LiteCUAMetadata(
        dims=(LiteCUAMetadata.Platform.DESKTOP, LiteCUAMetadata.TaskType.USE),
        others={"resolution": [w, h]},
    )
    img = Image.new("RGB", (w, h), color="white")
    sample = LiteSample(metadata=meta, images=[img])
    sample.messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "do the thing"},
                {"type": "image", "index": 0},
            ],
        }
    ]
    return sample


def _fake_response(
    output: list[dict[str, Any]] | None = None,
    incomplete_reason: str | None = None,
) -> dict[str, Any]:
    if output is None:
        output = [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "done"}],
            }
        ]
    resp: dict[str, Any] = {"output": output, "id": "resp_test", "usage": {}}
    if incomplete_reason is not None:
        # The Responses API's spelling of a chat-completions ``finish_reason``:
        # ``status="incomplete"`` plus ``incomplete_details.reason``.
        resp["status"] = "incomplete"
        resp["incomplete_details"] = {"reason": incomplete_reason}
    return resp


def _stub_gpt_echoed_dim_fetch(monkeypatch):
    """Keep GPT sample tests hermetic after the mocked provider response."""
    monkeypatch.setattr(
        "lite.agents.models.gpt.utils.image_io._fetch_processed_image_dims",
        AsyncMock(return_value=[(800, 600)]),
    )


async def _tools_sent(agent, monkeypatch, env: Any = None) -> list[dict[str, Any]]:
    """The provider tool list ``agent.sample()`` actually puts on the wire.

    The sample loop hands its assembled tool list to ``litellm.aresponses``
    unmodified, so the mocked request payload is the public read of the agent's
    advertised tool surface — no private tool-assembly helper needed. An empty
    tool list is sent as ``tools=None``, normalized back to ``[]`` here.
    """
    mock = AsyncMock(return_value=_fake_response())
    monkeypatch.setattr("litellm.aresponses", mock)
    await agent.sample(env if env is not None else _FakeEnv(terminate_after=1), max_steps=2)
    return mock.call_args.kwargs["tools"] or []


def _collect_input_image_details(input_items: list[dict[str, Any]]) -> list[str | None]:
    details: list[str | None] = []
    for item in input_items:
        content = item.get("content")
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "input_image":
                    details.append(c.get("detail"))
    return details


def _first_sent_input_image_png(input_items: list[dict[str, Any]]) -> bytes:
    """Decode the first ``input_image`` block's PNG bytes from a Responses
    ``input`` payload — the provider-visible image actually sent for the turn."""
    for item in input_items:
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "input_image":
                return base64.b64decode(block["image_url"].split("base64,", 1)[1])
    raise AssertionError("no input_image block found in sent input")


def _png_bytes(w: int = 800, h: int = 600) -> bytes:
    return png_bytes(w, h)


def _colored_png_bytes(
    w: int = 800,
    h: int = 600,
    color: tuple[int, int, int] = (32, 64, 96),
) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _image_rgb(image: Image.Image) -> tuple[int, int, int]:
    return image.convert("RGB").getpixel((0, 0))


class _FakeEnv:
    def __init__(self, *, terminate_after: int = 1, step_sleep: float = 0.0):
        self.metadata = LiteCUAMetadata(
            dims=(LiteCUAMetadata.Platform.DESKTOP, LiteCUAMetadata.TaskType.USE),
            others={"resolution": [800, 600]},
        )
        self._shot = png_bytes(800, 600)
        self._step_count = 0
        self._terminate_after = terminate_after
        self._step_sleep = step_sleep
        self.closed = False

    async def reset(self):
        return LiteEnvObservation(image=self._shot, text="instr")

    async def step(self, actions):
        if self._step_sleep:
            await asyncio.sleep(self._step_sleep)
        self._step_count += 1
        finish = any(tool_call_name(action) in {"response", "terminate"} for action in actions)
        done = finish or self._step_count >= self._terminate_after
        return LiteEnvStepResult(
            reward=1.0 if done else 0.0,
            terminated=done,
            results=[
                LiteToolResult(tool_call_id=tool_call_id(action), images=[self._shot], text="instr")
                for action in actions
            ],
        )

    async def close(self):
        self.closed = True


class _CloseRaisesEnv(_FakeEnv):
    async def close(self):
        self.closed = True
        raise RuntimeError("gpt close exploded")


class _CloseCancelledEnv(_FakeEnv):
    async def close(self):
        self.closed = True
        raise asyncio.CancelledError


class _RecordingFakeEnv(_FakeEnv):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.actions_seen: list[list[dict[str, Any]]] = []

    async def step(self, actions):
        self.actions_seen.append(actions)
        return await super().step(actions)


class _RejectEmptyActionsEnv(_RecordingFakeEnv):
    async def step(self, actions):
        if actions == []:
            raise AssertionError("content-only final must not call env.step([])")
        result = await super().step(actions)
        return result


class _ContentOnlyContinuationEnv(_RecordingFakeEnv):
    async def step(self, actions):
        if actions == []:
            raise AssertionError("content-only response must not call env.step([])")
        self.actions_seen.append(actions)
        self._step_count += 1
        submitted = tool_call_arguments(actions[0]).get("text", "") if actions else ""
        done = self._step_count >= self._terminate_after
        return LiteEnvStepResult(
            reward=1.0 if done else 0.0,
            terminated=done,
            results=[
                LiteToolResult(
                    tool_call_id=None,
                    images=[self._shot],
                    text=f"attempt {self._step_count}: {submitted}",
                )
            ],
        )


class _MobileContentOnlyContinuationEnv:
    def __init__(self, *, terminate_after: int = 2, feedback_image: bool = True):
        self.metadata = LiteCUAMetadata(
            dims=(LiteCUAMetadata.Platform.MOBILE, LiteCUAMetadata.TaskType.USE),
            others={"resolution": [800, 600]},
        )
        self._shot = png_bytes(800, 600)
        self._step_count = 0
        self._terminate_after = terminate_after
        self._feedback_image = feedback_image
        self.actions_seen: list[list[dict[str, Any]]] = []
        self.closed = False

    async def reset(self):
        return LiteEnvObservation(image=self._shot, text="open settings")

    async def step(self, actions):
        if actions == []:
            raise AssertionError("content-only response must not call env.step([])")
        self.actions_seen.append(actions)
        self._step_count += 1
        submitted = tool_call_arguments(actions[0]).get("text", "") if actions else ""
        done = self._step_count >= self._terminate_after
        return LiteEnvStepResult(
            reward=1.0 if done else 0.0,
            terminated=done,
            results=[
                LiteToolResult(
                    tool_call_id=None,
                    images=[self._shot] if self._feedback_image else [],
                    text=f"attempt {self._step_count}: {submitted}",
                )
            ],
        )

    async def close(self):
        self.closed = True


class _ToolResultsEnv(_FakeEnv):
    async def step(self, actions):
        result = await super().step(actions)
        if actions:
            call = actions[0]
            result.results = [
                LiteToolResult(tool_call_id=tool_call_id(call), text="per-call stdout"),
            ]
        return result


class _ImageToolResultsEnv(_FakeEnv):
    async def step(self, actions):
        result = await super().step(actions)
        if actions:
            call = actions[0]
            result.results = [
                LiteToolResult(
                    tool_call_id=tool_call_id(call),
                    images=[self._shot],
                    text="visual obs",
                ),
            ]
        return result


class _MultiImageToolResultsEnv(_FakeEnv):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._result_shots = [
            _colored_png_bytes(color=(180, 20, 20)),
            _colored_png_bytes(color=(20, 180, 20)),
            _colored_png_bytes(color=(20, 20, 180)),
        ]

    async def step(self, actions):
        result = await super().step(actions)
        if actions and not any(
            tool_call_name(action) in {"response", "terminate"} for action in actions
        ):
            call = actions[0]
            result.results = [
                LiteToolResult(
                    tool_call_id=tool_call_id(call),
                    images=list(self._result_shots),
                    text="visual obs",
                )
            ]
        return result


class _MixedComputerAndBashResultsEnv(_FakeEnv):
    async def step(self, actions):
        result = await super().step(actions)
        result.results = []
        for call in actions:
            if tool_call_name(call) == "computer":
                result.results.append(
                    LiteToolResult(
                        tool_call_id=tool_call_id(call),
                        images=[self._shot],
                        text="computer screen text",
                    )
                )
            elif tool_call_name(call) == "bash":
                result.results.append(
                    LiteToolResult(
                        tool_call_id=tool_call_id(call),
                        text="bash stdout",
                    )
                )
        return result


class _MixedComputerAndVisualExtraResultsEnv(_FakeEnv):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._computer_shot = _colored_png_bytes(color=(110, 30, 30))
        self._extra_shot = _colored_png_bytes(color=(30, 110, 30))

    async def step(self, actions):
        result = await super().step(actions)
        result.results = []
        for call in actions:
            if tool_call_name(call) == "computer":
                result.results.append(
                    LiteToolResult(
                        tool_call_id=tool_call_id(call),
                        images=[self._computer_shot],
                        text="computer screen text",
                    )
                )
            elif tool_call_name(call) == "visual_extra":
                result.results.append(
                    LiteToolResult(
                        tool_call_id=tool_call_id(call),
                        images=[self._extra_shot],
                        text="extra visual text",
                    )
                )
        return result


class _TerminalNoResultsEnv(_RecordingFakeEnv):
    async def step(self, actions):
        self.actions_seen.append(actions)
        self._step_count += 1
        return LiteEnvStepResult(reward=1.0, terminated=True, results=[])


class _ReversedResultOrderEnv(_FakeEnv):
    """Answers a two-call turn in the OPPOSITE order the calls were emitted.

    Also exercises all three per-call text channels: plain text, text plus an
    ``error``, and ``metadata``.
    """

    async def step(self, actions):
        result = await super().step(actions)
        per_call: list[LiteToolResult] = []
        for call in actions:
            if tool_call_name(call) == "computer":
                per_call.append(
                    LiteToolResult(
                        tool_call_id=tool_call_id(call),
                        images=[self._shot],
                        text="computer screen text",
                        metadata={"source": "screen"},
                    )
                )
            elif tool_call_name(call) == "bash":
                per_call.append(
                    LiteToolResult(
                        tool_call_id=tool_call_id(call),
                        text="bash stdout",
                        error="exit status 1",
                    )
                )
        result.results = list(reversed(per_call))
        return result


class _DropsOneResultEnv(_FakeEnv):
    """Non-terminal env that answers only the FIRST call of a multi-call turn."""

    async def step(self, actions):
        result = await super().step(actions)
        result.results = [
            LiteToolResult(tool_call_id=tool_call_id(actions[0]), text="only one result")
        ]
        return result


__all__ = [name for name in globals() if not name.startswith("__")]
