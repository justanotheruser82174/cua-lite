"""Golden-lock the API-retry backoff *formula* for the Claude and GPT agents.

Claude's retry (``ClaudeDesktopUseAgent`` / ``ClaudeMobileUseAgent``
``_acompletion_with_retry``) and GPT's (``_aresponses_with_retry``) share one
agent-side loop, ``lite.agents.core.agent.utils.retry.acompletion_with_retry`` —
capped ``min(max_delay, base*2**attempt)`` scaled by ``U(0.5, 1.5)`` jitter.
With ``random.random()`` pinned to 0.5 the jitter factor is exactly 1.0 and the
cap (default 60s) never engages at the small Claude defaults, so the recorded
sequence equals the bare geometric ``base*2**attempt``. These tests freeze the
exact sleep sequences each impl produces today.

Strategy: the backoff math (and ``asyncio.sleep`` / ``random.random``) lives in
``lite.agents.core.agent.utils.retry``; monkeypatch that module's
``asyncio.sleep`` to RECORD delays (never sleep), pin its ``random.random`` to
0.5, and patch the underlying ``litellm`` call (on the agent module that injects
it) to always raise, then assert the recorded list. No network, no GPU, no model
— fake clients only.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/agents/core/agent/test_retry_delay.py -p no:cacheprovider -q
"""

from __future__ import annotations

import asyncio
from typing import Any

import litellm
import pytest

import lite.agents.core.agent.utils.retry as agent_retry
from lite.agents.models.claude.agent import ClaudeDesktopUseAgent, ClaudeMobileUseAgent
from lite.agents.models.gpt.agent import GPTDesktopUseAgent


def _install_sleep_recorder(monkeypatch) -> list[float]:
    """Patch the shared retry loop's ``asyncio.sleep`` to append the delay and
    return at once, and pin its jitter (``random.random`` → 0.5 so the
    ``0.5 + random.random()`` factor is exactly 1.0)."""
    recorded: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(agent_retry.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(agent_retry.random, "random", lambda: 0.5)
    return recorded


def _install_always_fail(monkeypatch, attr: str) -> None:
    """Patch ``litellm.<attr>`` to always raise (drives the full retry loop).

    The agent imports ``litellm`` lazily inside its retry method and calls
    ``litellm.<attr>``, so patching the real litellm module attribute is what the
    loop actually invokes."""

    async def boom(**_kwargs: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(litellm, attr, boom)


# ===========================================================================
# Claude: capped exponential * U(0.5, 1.5) jitter (jitter pinned to 1.0 here).
# The cap (default 60s) does not engage at Claude's small defaults, so the
# recorded sequence is the bare geometric one.
# ===========================================================================
class TestClaudeRetryFormula:
    def test_claude_desktop_capped_jitter_one(self, monkeypatch):
        # base=0.5, max=4 -> sleeps before attempts 1..4 -> [0.5, 1, 2, 4].
        # Default api_retry_max_delay=60 never caps these; jitter pinned to 1.0.
        agent = ClaudeDesktopUseAgent(api_retry_max=4, api_retry_base_delay=0.5)
        recorded = _install_sleep_recorder(monkeypatch)
        _install_always_fail(monkeypatch, "acompletion")

        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(agent._acompletion_with_retry(model="x", messages=[]))

        assert recorded == [0.5, 1.0, 2.0, 4.0]

    def test_claude_desktop_default_max_three(self, monkeypatch):
        # The shipped default api_retry_max=3 -> three sleeps [0.5, 1, 2].
        agent = ClaudeDesktopUseAgent()  # api_retry_max=3, base_delay=0.5
        assert agent.api_retry_max == 3
        assert agent.api_retry_base_delay == 0.5
        recorded = _install_sleep_recorder(monkeypatch)
        _install_always_fail(monkeypatch, "acompletion")

        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(agent._acompletion_with_retry(model="x", messages=[]))

        assert recorded == [0.5, 1.0, 2.0]

    def test_claude_succeeds_after_failures_no_extra_sleep(self, monkeypatch):
        """Fail twice then succeed: exactly two sleeps [0.5, 1.0], no raise."""
        agent = ClaudeDesktopUseAgent(api_retry_max=4, api_retry_base_delay=0.5)
        recorded = _install_sleep_recorder(monkeypatch)

        calls = {"n": 0}
        sentinel = object()

        async def flaky(**_kwargs: Any) -> Any:
            calls["n"] += 1
            if calls["n"] <= 2:
                raise RuntimeError("transient")
            return sentinel

        monkeypatch.setattr(litellm, "acompletion", flaky)

        out = asyncio.run(agent._acompletion_with_retry(model="x", messages=[]))
        assert out is sentinel
        assert recorded == [0.5, 1.0]

    def test_claude_mobile_retry_matches_desktop(self, monkeypatch):
        """Desktop and mobile Claude share one retry formula."""
        desktop = ClaudeDesktopUseAgent(api_retry_max=4, api_retry_base_delay=0.5)
        mobile = ClaudeMobileUseAgent(api_retry_max=4, api_retry_base_delay=0.5)

        rec_d = _install_sleep_recorder(monkeypatch)
        _install_always_fail(monkeypatch, "acompletion")
        with pytest.raises(RuntimeError):
            asyncio.run(desktop._acompletion_with_retry(model="x", messages=[]))

        rec_m: list[float] = []

        async def fake_sleep_m(delay: float) -> None:
            rec_m.append(delay)

        monkeypatch.setattr(agent_retry.asyncio, "sleep", fake_sleep_m)
        with pytest.raises(RuntimeError):
            asyncio.run(mobile._acompletion_with_retry(model="x", messages=[]))

        assert rec_d == [0.5, 1.0, 2.0, 4.0]
        assert rec_m == rec_d


# ===========================================================================
# GPT: capped exponential * U(0.5, 1.5) jitter
# ===========================================================================
class TestGPTRetryFormula:
    def test_gpt_capped_geometric_jitter_one(self, monkeypatch):
        # _install_sleep_recorder pins random.random -> 0.5 so jitter factor
        # (0.5 + 0.5) == 1.0, isolating the capped geometric sequence.
        # Defaults: base=2, max_delay=60. With max_retries=6:
        # [2, 4, 8, 16, 32, 60] (cap engages at attempt 5).
        agent = GPTDesktopUseAgent(
            api_retry_max=6, api_retry_base_delay=2.0, api_retry_max_delay=60.0,
        )
        recorded = _install_sleep_recorder(monkeypatch)
        _install_always_fail(monkeypatch, "aresponses")

        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(agent._aresponses_with_retry(model="x"))

        assert recorded == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0]

    def test_gpt_full_default_max_retries(self, monkeypatch):
        # The shipped desktop default api_retry_max=24 -> 24 sleeps, cap holds
        # at 60 from attempt 5 onward.
        agent = GPTDesktopUseAgent()  # api_retry_max=24, base=2, max=60
        assert agent.api_retry_max == 24
        recorded = _install_sleep_recorder(monkeypatch)
        _install_always_fail(monkeypatch, "aresponses")

        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(agent._aresponses_with_retry(model="x"))

        expected = [2.0, 4.0, 8.0, 16.0, 32.0] + [60.0] * 19
        assert recorded == expected

    def test_gpt_jitter_bounds(self, monkeypatch):
        # Over several jitter seeds, each delay must land in
        # [0.5, 1.5] * min(max_delay, base*2**attempt).
        import random as _random

        base, max_delay, max_retries = 2.0, 60.0, 8
        uncapped = [min(max_delay, base * (2 ** a)) for a in range(max_retries)]
        agent = GPTDesktopUseAgent(
            api_retry_max=max_retries, api_retry_base_delay=base,
            api_retry_max_delay=max_delay,
        )

        for seed in range(5):
            rng = _random.Random(seed)
            recorded: list[float] = []

            async def fake_sleep(delay: float, _rec=recorded) -> None:
                _rec.append(delay)

            monkeypatch.setattr(agent_retry.random, "random", rng.random)
            monkeypatch.setattr(agent_retry.asyncio, "sleep", fake_sleep)
            _install_always_fail(monkeypatch, "aresponses")

            with pytest.raises(RuntimeError):
                asyncio.run(agent._aresponses_with_retry(model="x"))

            assert len(recorded) == max_retries
            for delay, capped in zip(recorded, uncapped):
                assert 0.5 * capped <= delay <= 1.5 * capped
