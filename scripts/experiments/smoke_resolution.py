"""Smoke test browser env screenshot sizes for the 1280x720 native viewport.

For each selected browser-navigation env, make one task with ``gym.make``
(direct or env-server-backed, depending on the caller's env vars), reset it,
decode the observation screenshot, and compare it to this file's EXPECTED
table. The table captures env-specific screenshot behavior.

Run (from repo root):
    uv run python scripts/experiments/smoke_resolution.py
"""

from __future__ import annotations

import asyncio
import io

from PIL import Image

import lite.gym as gym

# Per-env expected screenshot size for a 1280×720 native viewport.
# webgym/online_mind2web screenshot the page viewport directly → exactly 1280×720.
# webharbor.webvoyager drives Selenium Chrome with --window-size=1280,720, which sizes the
# OUTER window; the screenshot captures the inner viewport, so width is exactly
# 1280 but height is 720 minus a constant ~87px of Chrome chrome (toolbar) → 633.
EXPECTED = {
    "webgym": (1280, 720),
    "online_mind2web": (1280, 720),
    "webharbor.webvoyager": (1280, 633),
}
ENVS = ["webgym", "webharbor.webvoyager", "online_mind2web"]


def _first_task(env_id: str) -> str:
    ids = gym.registry.task_ids(env_id)
    if isinstance(ids, dict):
        for split in ids.values():
            if split:
                return split[0]
        raise RuntimeError(f"{env_id}: no tasks")
    return ids[0]


async def _check(env_id: str) -> tuple[str, bool, str]:
    """Reset one task for env_id and report whether its screenshot matches EXPECTED."""
    tid = _first_task(env_id)
    # skip_eval=True: the resolution check only needs reset()'s screenshot, not
    # the VLM judge (which would demand OPENAI_API_KEY at reset).
    env = gym.make(f"{env_id}@{tid}", max_steps=2, skip_eval=True)
    try:
        obs = await env.reset()
        png = obs.image
        if not png:
            return env_id, False, f"{tid}: reset returned no screenshot"
        size = Image.open(io.BytesIO(png)).size
        exp = EXPECTED[env_id]
        ok = size == exp
        return env_id, ok, f"{tid}: screenshot {size[0]}x{size[1]} (expect {exp[0]}x{exp[1]})"
    finally:
        await env.close()


async def main() -> None:
    expected = ", ".join(f"{env_id}={size[0]}x{size[1]}" for env_id, size in EXPECTED.items())
    print(f"=== native-resolution smoke ({expected}) ===", flush=True)
    all_ok = True
    for env_id in ENVS:
        try:
            _id, ok, msg = await _check(env_id)
        except Exception as e:  # surface, keep going
            ok, msg = False, f"ERROR {type(e).__name__}: {e}"
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {env_id:12s} {msg}", flush=True)
    print("=== RESULT:", "ALL PASS" if all_ok else "FAILURES PRESENT", "===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
