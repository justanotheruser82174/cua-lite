"""Split GPT agent characterization tests.

Run:
    uv run pytest tests/agents/models/gpt/test_gpt_*.py -v
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock

import pytest
from agents.models.gpt._support import (
    _CloseCancelledEnv,
    _CloseRaisesEnv,
    _fake_response,
    _FakeEnv,
)

from lite.agents.core.agent import AgentRegistry
from lite.agents.models.gpt.agent import GPTDesktopUseAgent
from lite.agents.models.gpt.utils.responses import ResponseAPIError
from lite.core import STATUS_COMPLETED, STATUS_TRUNCATED

# -----------------------------------------------------------------------------
# Config rejection
# -----------------------------------------------------------------------------


class TestGPTConfigRejection:
    """Reject stale or unknown GPT config instead of accepting silent no-ops."""

    async def test_default_raises_on_failed_status(self, monkeypatch):
        failed = {"output": [], "status": "failed", "error": {"message": "oops"}}
        mock = AsyncMock(return_value=failed)
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(model_id="gpt-5.5")
        with pytest.raises(ResponseAPIError):
            await agent.sample(_FakeEnv(), max_steps=5)

    def test_fail_fast_on_api_error_is_rejected_at_construction(self):
        with pytest.raises(TypeError, match="fail_fast_on_api_error"):
            GPTDesktopUseAgent(
                model_id="gpt-5.5",
                api_kwargs={"max_output_tokens": 4096},
                fail_fast_on_api_error=False,
            )

    def test_registry_rejects_stale_fail_fast_on_api_error(self):
        with pytest.raises(TypeError, match="fail_fast_on_api_error"):
            AgentRegistry.get(
                "gpt@desktop@use",
                model_id="gpt-5.5",
                api_kwargs={"max_output_tokens": 4096},
                fail_fast_on_api_error=False,
            )

    def test_preserve_raw_response_is_not_a_gpt_runtime_knob(self):
        with pytest.raises(TypeError, match="preserve_raw_response"):
            GPTDesktopUseAgent(
                model_id="gpt-5.5",
                preserve_raw_response=True,
            )

    def test_registry_rejects_stale_preserve_raw_response(self):
        with pytest.raises(TypeError, match="preserve_raw_response"):
            AgentRegistry.get(
                "gpt@desktop@use",
                model_id="gpt-5.5",
                preserve_raw_response=True,
            )

    def test_registry_rejects_unknown_gpt_config(self):
        with pytest.raises(TypeError, match="unknown_gpt_config"):
            AgentRegistry.get("gpt@desktop@use", unknown_gpt_config=True)


# -----------------------------------------------------------------------------
# tool_execution_timeout_s
# -----------------------------------------------------------------------------


class TestToolExecutionTimeout:
    """asyncio.wait_for wrapping env.step when kwarg is set."""

    async def test_timeout_fires_on_slow_env(self, monkeypatch):
        resp = _fake_response(
            [
                {
                    "type": "computer_call",
                    "call_id": "call_1",
                    "actions": [{"type": "screenshot"}],
                }
            ]
        )
        # Second response never reached because timeout fires on first env.step
        mock = AsyncMock(return_value=resp)
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            api_kwargs={"max_output_tokens": 4096},
            tool_execution_timeout_s=0.05,
        )
        # Env sleeps 1s on step; timeout 0.05s should fire.
        with pytest.raises(asyncio.TimeoutError):
            await agent.sample(_FakeEnv(step_sleep=1.0), max_steps=3)

    async def test_default_none_no_timeout(self, monkeypatch):
        resp = _fake_response(
            [
                {
                    "type": "computer_call",
                    "call_id": "call_1",
                    "actions": [{"type": "screenshot"}],
                }
            ]
        )
        mock = AsyncMock(return_value=resp)
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(model_id="gpt-5.5")
        # Env sleeps 0.1s (normal step) — should complete without firing
        result = await agent.sample(_FakeEnv(step_sleep=0.1, terminate_after=1), max_steps=3)
        assert result is not None


class TestEnvCloseTeardown:
    async def test_close_failure_logged_and_swallowed_after_success(self, monkeypatch, caplog):
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)
        caplog.set_level(logging.WARNING, logger="lite.agents.models.gpt.agent")

        agent = GPTDesktopUseAgent(model_id="gpt-5.5")
        result = await agent.sample(_CloseRaisesEnv(terminate_after=1), max_steps=2)

        assert result.terminated is True
        assert mock.call_count == 1
        assert "env.close() failed: gpt close exploded" in caplog.text

    async def test_close_runs_when_on_complete_raises(self, monkeypatch):
        class _CompleteRaises:
            def on_complete(self, result):
                raise RuntimeError("gpt hook complete exploded")

        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _FakeEnv(terminate_after=1)
        agent = GPTDesktopUseAgent(model_id="gpt-5.5")

        with pytest.raises(RuntimeError, match="gpt hook complete exploded"):
            await agent.sample(env, max_steps=2, hooks=[_CompleteRaises()])

        assert env.closed is True

    async def test_close_cancellation_propagates(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _CloseCancelledEnv(terminate_after=1)
        agent = GPTDesktopUseAgent(model_id="gpt-5.5")

        with pytest.raises(asyncio.CancelledError):
            await agent.sample(env, max_steps=2)

        assert env.closed is True

    async def test_provider_error_is_not_masked_by_close_failure(self, monkeypatch, caplog):
        async def provider_boom(**kwargs):
            raise RuntimeError("gpt provider exploded")

        monkeypatch.setattr("litellm.aresponses", provider_boom)
        caplog.set_level(logging.WARNING, logger="lite.agents.models.gpt.agent")

        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            api_retry_max=0,
            api_retry_base_delay=0.0,
        )

        with pytest.raises(RuntimeError, match="gpt provider exploded"):
            await agent.sample(_CloseRaisesEnv(terminate_after=1), max_steps=2)

        assert "env.close() failed: gpt close exploded" in caplog.text


class TestStepStatusFromIncompleteDetails:
    """The Responses API's token-budget stop signal must reach the step status.

    The loop used to hardcode ``STATUS_COMPLETED``, so ``incomplete_details.
    reason == "max_output_tokens"`` was recorded as a clean finish. Distinct from
    the env STEP-budget relabel, which the class below covers.
    """

    @pytest.mark.parametrize(
        ("incomplete_reason", "status"),
        [
            ("max_output_tokens", STATUS_TRUNCATED),
            ("max_tokens", STATUS_TRUNCATED),
            ("length", STATUS_TRUNCATED),
            ("content_filter", STATUS_COMPLETED),
            (None, STATUS_COMPLETED),
        ],
        ids=[
            "responses-api budget",
            "chat-completions spelling",
            "length",
            "not a budget stop",
            "complete response",
        ],
    )
    async def test_incomplete_reason_sets_the_step_status(
        self, monkeypatch, incomplete_reason, status
    ):
        resp = _fake_response(
            [
                {
                    "type": "computer_call",
                    "call_id": "call_1",
                    "actions": [{"type": "screenshot"}],
                }
            ],
            incomplete_reason=incomplete_reason,
        )
        monkeypatch.setattr("litellm.aresponses", AsyncMock(return_value=resp))

        agent = GPTDesktopUseAgent(model_id="gpt-5.5")
        result = await agent.sample(_FakeEnv(terminate_after=1), max_steps=3)

        assert result.terminated is True
        assert [s.status for s in result.steps] == [status]


# -----------------------------------------------------------------------------
# api_retry_max / api_retry_base_delay
# -----------------------------------------------------------------------------


class TestAPIRetry:
    """exponential-backoff retry wrapping litellm.aresponses."""

    async def test_default_retries_on_transient_error(self, monkeypatch):
        calls = {"n": 0}

        async def flaky(**kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("transient boom")
            return _fake_response()

        monkeypatch.setattr("litellm.aresponses", flaky)
        # Speed up the test by zeroing the base delay
        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            api_kwargs={"max_output_tokens": 4096},
            api_retry_base_delay=0.0,
        )
        result = await agent._aresponses_with_retry(model="x")
        assert calls["n"] == 3  # 2 failures + 1 success
        assert result is not None

    async def test_retries_exhausted_raises(self, monkeypatch):
        async def always_failing(**kwargs):
            raise RuntimeError("forever boom")

        monkeypatch.setattr("litellm.aresponses", always_failing)
        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            api_kwargs={"max_output_tokens": 4096},
            api_retry_max=2,
            api_retry_base_delay=0.0,
        )
        with pytest.raises(RuntimeError, match="forever boom"):
            await agent._aresponses_with_retry(model="x")

    async def test_max_zero_disables_retry(self, monkeypatch):
        calls = {"n": 0}

        async def once_failing(**kwargs):
            calls["n"] += 1
            raise RuntimeError("first call boom")

        monkeypatch.setattr("litellm.aresponses", once_failing)
        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            api_kwargs={"max_output_tokens": 4096},
            api_retry_max=0,
            api_retry_base_delay=0.0,
        )
        with pytest.raises(RuntimeError):
            await agent._aresponses_with_retry(model="x")
        assert calls["n"] == 1  # no retry
