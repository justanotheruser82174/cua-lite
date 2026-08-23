"""Source-contract test for bounding sample-side rollout hangs."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_ENGINE_SRC = (_ROOT / "lite" / "train" / "rollout" / "core" / "engine.py").read_text()
_DEGRADED = object()


class _HangingAgent:
    async def sample(self, env, *args, **kwargs):
        await asyncio.Event().wait()


async def _bounded_sample(agent, env, *, timeout: float):
    try:
        return await asyncio.wait_for(agent.sample(env), timeout=timeout)
    except TimeoutError:
        return _DEGRADED


@pytest.mark.xfail(
    strict=True,
    reason=(
        "engine.generate() awaits `agent.sample(env)` with no asyncio.wait_for "
        "ceiling; a sample-side hang is not an exception, so it evades the "
        "_empty_sample net and wedges the slot"
    ),
)
async def test_sample_side_hang_degrades_not_wedges() -> None:
    result = await _bounded_sample(_HangingAgent(), env=object(), timeout=0.2)
    assert result is _DEGRADED

    compact = "".join(_ENGINE_SRC.split())
    assert "agent.sample(env)" in compact, (
        "expected the agent.sample(env) call site in engine.py; "
        "test is anchored to the wrong source file"
    )
    assert "wait_for(agent.sample(" in compact, (
        "engine.py's `await agent.sample(env)` is not wrapped in asyncio.wait_for"
    )
