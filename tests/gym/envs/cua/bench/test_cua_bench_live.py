"""Live end-to-end tests for the CUA-Bench env.

Marked ``pytest.mark.live`` so the default run skips it. It also skips unless
``cua-bench`` and ``CUA_BENCH_DATASET_ROOT`` are present.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

pytestmark = pytest.mark.live

_HAVE_CB = importlib.util.find_spec("cua_bench") is not None
_HAVE_DATASET = bool(os.environ.get("CUA_BENCH_DATASET_ROOT"))


def _act(name, **args):
    return {"name": name, "arguments": args}


@pytest.mark.skipif(
    not (_HAVE_CB and _HAVE_DATASET),
    reason="needs cua-bench + CUA_BENCH_DATASET_ROOT",
)
async def test_cua_bench_webtop_smoke():
    import lite.gym as gym

    env_ids = [e for e in gym.registry.registered_env_ids() if e.startswith("cua.bench.")]
    assert env_ids, "no cua.bench.* envs registered - check CUA_BENCH_DATASET_ROOT"
    eid = sorted(env_ids)[0]
    variants = gym.registry.task_ids(eid)
    tid = next(iter(variants.values()))[0] if isinstance(variants, dict) else list(variants)[0]
    env = gym.make(f"{eid}@{tid}", max_steps=3, extra_tools=["terminate"])
    try:
        obs = await env.reset()
        assert obs.text and obs.image
        r2 = await env.step([_act("terminate", status="success")])
        assert r2.terminated
        assert r2.reward is None or isinstance(r2.reward, float)
    finally:
        await env.close()
