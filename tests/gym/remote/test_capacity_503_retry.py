"""Acceptance tests for the capacity-503 + Retry-After + client-retry chain.

This is the gating test for the "atomic ship" requirement
rev 5 P1: the server emits 503 + Retry-After on
:class:`~lite.gym.errors.CapacityExhausted`. The client retries
idempotent RPCs before re-raising as :class:`CapacityExhausted` (so the
training loop's group-redraw filter can match by type).

Tests run against a tiny in-process FastAPI fixture — no real env, no
real env-server. Covers:

* ``client.reset()`` retries 503 then succeeds
* ``client.step()`` does not retry 503, because replaying actions needs
  request-id dedupe before it is safe
* Retry-After header is honored (lengthy header => skipped or
  bounded for test speed)
* After the ``_CLIENT_503_DEADLINE_S`` wall-clock deadline elapses,
  the client re-raises ``CapacityExhausted`` (NOT generic
  ``HTTPStatusError``)
* The server-side ``@app.exception_handler(CapacityExhausted)`` maps
  the typed exception to 503 + Retry-After uniformly

Run::

    uv run pytest -xvs tests/gym/remote/test_capacity_503_retry.py
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from gym.remote.conftest import running_mock_server

from lite.core.tools import make_tool_call
from lite.core.tools.results import LiteToolResult
from lite.gym.errors import CapacityExhausted
from lite.gym.remote import client as client_module
from lite.gym.remote.client import LiteEnvClient
from lite.gym.remote.frame import encode_reset_observation, encode_step_result
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult

# NOT marked ``live``: everything here runs against the in-process ASGI app in
# ``running_mock_server`` (see the module docstring), so it has to be collected
# by the default ``-m "not live and not stress"`` filter — it is the gate on the
# client's 503 retry chain.


# ---------------------------------------------------------------------------
# Mock env-server fixture (no real envs, just minimal /instances surface)
# ---------------------------------------------------------------------------

class _MockState:
    """Per-test counters: how many 503s the mock has emitted, etc.

    ``reset_fail_count`` and ``step_fail_count`` are independent so tests
    can fail one endpoint while leaving the other healthy (e.g. test
    /step exhaustion in isolation without /reset also failing).
    """

    def __init__(
        self,
        reset_fail_count: int = 0,
        step_fail_count: int = 0,
        retry_after_s: int = 1,
    ) -> None:
        self.reset_fail_count = reset_fail_count
        self.step_fail_count = step_fail_count
        self.retry_after_s = retry_after_s
        self.reset_calls = 0
        self.step_calls = 0
        self.create_calls = 0


def _build_app(state: _MockState) -> FastAPI:
    """Mock env-server that 503s for ``fail_count`` requests then 200s.

    Mirrors the real server's exception_handler shape so the client's
    parser exercises the actual production path.
    """
    app = FastAPI()

    @app.exception_handler(CapacityExhausted)
    async def _capacity_exhausted_handler(
        _request: Any, exc: CapacityExhausted,
    ) -> JSONResponse:
        # Same body expression as the production handler — the class owns the
        # payload, so this mock cannot drift from the real env transport shape.
        return JSONResponse(
            status_code=exc.http_status,
            headers={"Retry-After": str(int(round(exc.retry_after_s)))},
            content={**exc.to_payload(), "detail": str(exc)},
        )

    @app.post("/instances", status_code=201)
    async def create(_body: dict[str, Any]) -> dict[str, Any]:
        state.create_calls += 1
        return {"id": "inst-1", "metadata": None}

    @app.post("/instances/{id_}/reset")
    async def reset(id_: str) -> Response:
        state.reset_calls += 1
        if state.reset_calls <= state.reset_fail_count:
            raise CapacityExhausted(
                what=f"mock pool full ({state.reset_calls}/{state.reset_fail_count})",
                retry_after_s=state.retry_after_s,
            )
        return Response(
            content=encode_reset_observation(
                LiteEnvObservation(image=None, text="ok")
            ),
            media_type="application/octet-stream",
        )

    @app.post("/instances/{id_}/step")
    async def step(id_: str, _body: dict[str, Any]) -> Response:
        state.step_calls += 1
        if state.step_calls <= state.step_fail_count:
            raise CapacityExhausted(
                what=f"mock lease expired ({state.step_calls}/{state.step_fail_count})",
                retry_after_s=state.retry_after_s,
            )
        return Response(
            content=encode_step_result(LiteEnvStepResult(
                reward=1.0,
                terminated=False,
                truncated=False,
                info={},
                results=[LiteToolResult(tool_call_id="noop_0", text="stepped")],
            )),
            media_type="application/octet-stream",
        )

    @app.delete("/instances/{id_}")
    async def close(id_: str) -> dict[str, Any]:
        return {"id": id_, "status": "closed"}

    @app.get("/envs/{env_id}/tasks")
    async def list_tasks(env_id: str) -> dict[str, Any]:
        return {"splits": {"train": ["task-1"]}, "metadata": {}}

    return app


def _running_server(state: _MockState):
    """Shared in-process uvicorn harness over this file's mock app."""
    return running_mock_server(_build_app(state))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _run_client_scenario(url: str, scenario) -> Any:
    """Helper: run one async scenario in a single event loop so the client's
    long-lived ``httpx.AsyncClient`` doesn't span loops (which closes its
    transport mid-call). Each test uses one scenario per loop.
    """
    async def _go():
        client = LiteEnvClient(url, "demo@task-1")
        try:
            return await scenario(client)
        finally:
            await client.close()
    return asyncio.run(_go())


def test_reset_retries_503_then_succeeds() -> None:
    """One 503 from /reset; the client retries on Retry-After and succeeds."""
    state = _MockState(reset_fail_count=1, retry_after_s=1)
    with _running_server(state) as url:
        async def scenario(client):
            result = await client.reset()
            assert result.text == "ok"
        _run_client_scenario(url, scenario)
        assert state.reset_calls == 2  # one 503 + one success


def test_step_503_raises_capacity_exhausted_without_replay() -> None:
    """/step 503s are typed, but are not replayed."""
    state = _MockState(step_fail_count=1, retry_after_s=1)
    with _running_server(state) as url:
        async def scenario(client):
            await client.reset()  # reset is healthy in this scenario
            with pytest.raises(CapacityExhausted) as exc_info:
                await client.step([make_tool_call("noop")])
            assert "mock lease expired" in exc_info.value.what
        _run_client_scenario(url, scenario)
        assert state.reset_calls == 1
        assert state.step_calls == 1


def test_reset_503_deadline_raises_capacity_exhausted(monkeypatch) -> None:
    """503s past the wall-clock deadline → client re-raises CapacityExhausted (typed).

    Uses a short deadline + small Retry-After to keep the test under a
    second. The exact retry count depends on scheduling; we only assert
    the typed raise and that at least one retry occurred (the very first
    call always burns one).
    """
    monkeypatch.setattr(client_module, "_CLIENT_503_DEADLINE_S", 0.5)
    state = _MockState(
        reset_fail_count=999,    # never lets the client succeed
        retry_after_s=1,         # >= remaining deadline triggers the give-up branch
    )
    with _running_server(state) as url:
        async def scenario(client):
            with pytest.raises(CapacityExhausted) as exc_info:
                await client.reset()
            assert "mock pool full" in exc_info.value.what
        _run_client_scenario(url, scenario)
        assert state.reset_calls >= 1


def test_step_503_does_not_sleep_for_retry_after() -> None:
    """/step 503 should surface immediately, ignoring Retry-After."""
    state = _MockState(
        step_fail_count=999,
        retry_after_s=2,
    )
    with _running_server(state) as url:
        async def scenario(client):
            await client.reset()
            t0 = time.monotonic()
            with pytest.raises(CapacityExhausted) as exc_info:
                await client.step([make_tool_call("noop")])
            elapsed = time.monotonic() - t0
            assert "mock lease expired" in exc_info.value.what
            assert elapsed < 1.0, (
                f"/step 503 should ignore Retry-After immediately, got {elapsed:.1f} s"
            )
        _run_client_scenario(url, scenario)
        assert state.step_calls == 1


def test_retry_after_header_is_honored() -> None:
    """Client sleeps the server's Retry-After value (not exp-backoff)."""
    state = _MockState(reset_fail_count=1, retry_after_s=2)
    with _running_server(state) as url:
        async def scenario(client):
            t0 = time.monotonic()
            await client.reset()
            elapsed = time.monotonic() - t0
            # Lower bound 1.5 s asserts the client *honored* Retry-After:
            # 2 (anything less means it ignored the header and used its own
            # fast backoff). Upper bound is generous (10 s) so heavily-loaded
            # CI runners don't flake on event-loop contention from parallel
            # test workers.
            assert 1.5 < elapsed < 10.0, (
                f"Retry-After: 2 should produce ~2 s wait, got {elapsed:.1f} s"
            )
        _run_client_scenario(url, scenario)
