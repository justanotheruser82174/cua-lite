"""Live end-to-end tests for the CUA sandbox env.

Marked ``pytest.mark.live`` so the default run skips it. It also skips unless
``cua-sandbox`` and ``CUA_API_KEY`` are present.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

pytestmark = pytest.mark.live

_HAVE_CUA = importlib.util.find_spec("cua_sandbox") is not None
_HAVE_KEY = bool(os.environ.get("CUA_API_KEY"))


def _act(name, **args):
    return {"name": name, "arguments": args}


@pytest.mark.skipif(not (_HAVE_CUA and _HAVE_KEY), reason="needs cua + CUA_API_KEY")
async def test_cua_sandbox_cloud_smoke():
    from lite.gym.envs.cua.sandbox import CuaSandboxEnv

    env = CuaSandboxEnv(
        instruction="focus the desktop",
        max_steps=3,
        extra_tools=["terminate"],
        local=False,
    )
    try:
        obs = await env.reset()
        assert obs.text == "focus the desktop"
        assert obs.image
        r2 = await env.step([_act("wait", duration=0.2)])
        assert not r2.terminated and r2.reward is None
        r3 = await env.step([_act("terminate", status="success")])
        assert r3.terminated
    finally:
        await env.close()
