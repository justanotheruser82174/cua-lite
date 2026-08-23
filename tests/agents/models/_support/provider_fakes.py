"""Shared fakes for the agent-model characterization tests.

These are plain helper classes/functions, not pytest fixtures:

  * :func:`png_b64` returns a solid-white PNG as base64.
  * :func:`png_bytes` returns the same solid-white PNG as raw bytes.
  * :class:`FakeMobileEnv` and :class:`FakeEnv` are minimal async envs that
    return a screenshot on reset and each step, then auto-terminate after
    ``terminate_after`` steps.

Import them directly:

    from agents.models._support.provider_fakes import FakeEnv, png_b64
"""

from __future__ import annotations

import base64
import io

from PIL import Image

from lite.core import LiteCUAMetadata
from lite.core.tools.calls import tool_call_id
from lite.core.tools.results import LiteToolResult
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult


def png_bytes(w: int, h: int) -> bytes:
    """Solid-white PNG of the given dims, as raw bytes."""
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color="white").save(buf, format="PNG")
    return buf.getvalue()


def png_b64(w: int, h: int) -> str:
    """Solid-white PNG of the given dims, base64-encoded."""
    return base64.b64encode(png_bytes(w, h)).decode("utf-8")


class FakeMobileEnv:
    """Minimal mobile fake env with 1080x2400 screenshots."""

    def __init__(self, terminate_after: int = 1):
        self.metadata = LiteCUAMetadata(
            dims=(LiteCUAMetadata.Platform.MOBILE, LiteCUAMetadata.TaskType.USE),
            others={"resolution": [1080, 2400]},
        )
        self._shot = png_bytes(1080, 2400)
        self._step_count = 0
        self._terminate_after = terminate_after
        self.actions_seen = []

    async def reset(self):
        return LiteEnvObservation(image=self._shot, text="open settings")

    async def step(self, actions):
        self.actions_seen.append(list(actions))
        self._step_count += 1
        done = self._step_count >= self._terminate_after
        return LiteEnvStepResult(
            reward=1.0 if done else 0.0,
            terminated=done,
            results=[
                LiteToolResult(
                    tool_call_id=tool_call_id(action),
                    images=[self._shot],
                    text="open settings",
                )
                for action in actions
            ],
        )

    async def close(self):
        pass


class FakeEnv:
    """Minimal desktop fake env with 800x600 screenshots.

    Set ``step_sleep`` to simulate a slow env for timeout tests.
    Set ``terminate_after`` to auto-stop after N steps.
    """

    def __init__(self, *, terminate_after: int = 1, step_sleep: float = 0.0):
        self.metadata = LiteCUAMetadata(
            dims=(LiteCUAMetadata.Platform.DESKTOP, LiteCUAMetadata.TaskType.USE),
            others={"resolution": [800, 600]},
        )
        self._shot = png_bytes(800, 600)
        self._step_count = 0
        self._terminate_after = terminate_after
        self._step_sleep = step_sleep
        self.actions_seen = []

    async def reset(self):
        return LiteEnvObservation(image=self._shot, text="instr")

    async def step(self, actions):
        self.actions_seen.append(list(actions))
        if self._step_sleep:
            import asyncio

            await asyncio.sleep(self._step_sleep)
        self._step_count += 1
        done = self._step_count >= self._terminate_after
        return LiteEnvStepResult(
            reward=1.0 if done else 0.0,
            terminated=done,
            results=[
                LiteToolResult(
                    tool_call_id=tool_call_id(action),
                    images=[self._shot],
                    text="instr",
                )
                for action in actions
            ],
        )

    async def close(self):
        pass
