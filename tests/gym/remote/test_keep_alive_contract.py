"""The env-server's idle keep-alive must outlive the client pool's.

The client pool keeps idle connections for ``HTTPX_KEEPALIVE_EXPIRY_SEC``. The
server timeout must be longer so the client, not the server, retires pooled
connections. The socket test below compares uvicorn's default against the Lite
timeout over the same idle gap.

Run:
    uv run pytest tests/gym/remote/test_keep_alive_contract.py -q
"""
from __future__ import annotations

import asyncio

import httpx
import pytest
from gym.remote.conftest import running_mock_server
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from lite.gym.remote.alive import SERVER_KEEP_ALIVE_TIMEOUT_SEC
from lite.gym.remote.client import HTTPX_KEEPALIVE_EXPIRY_SEC

#: uvicorn's own default, and therefore the pre-fix behaviour of every
#: ``scripts/serve_env.py`` deployment.
UVICORN_DEFAULT_KEEP_ALIVE = 5.0

#: One idle gap, shared by both halves of the A/B. Must exceed
#: :data:`UVICORN_DEFAULT_KEEP_ALIVE` for the control to be meaningful.
IDLE_GAP_SEC = 6.0


async def _ping(request):
    return JSONResponse({"ok": True})


def _app() -> Starlette:
    return Starlette(routes=[Route("/ping", _ping, methods=["GET"])])


def _pooled_client() -> httpx.AsyncClient:
    """A client configured exactly like ``LiteEnvClient``'s long-lived pool."""
    return httpx.AsyncClient(
        timeout=30.0,
        limits=httpx.Limits(
            max_connections=200,
            max_keepalive_connections=100,
            keepalive_expiry=HTTPX_KEEPALIVE_EXPIRY_SEC,
        ),
    )


def _server_closed_the_connection(client: httpx.AsyncClient) -> bool:
    """True when the peer has sent FIN on the pooled connection.

    This is the precondition of the production failure: a connection the client
    still considers pooled, which the server has unilaterally closed. Whether
    the resulting request then survives depends on where the close lands
    relative to httpcore's readability check — a race. Eliminating the close
    eliminates the race, so the close is the right thing to assert on.
    """
    connections = list(client._transport._pool.connections)
    assert len(connections) == 1, f"expected one pooled connection, got {connections}"
    return bool(
        connections[0]._connection._network_stream.get_extra_info("is_readable")
    )


# =============================================================================
# The invariant (cheap — no sockets, no sleeping)
# =============================================================================

def test_server_keep_alive_strictly_outlives_the_client_pool() -> None:
    """The server side must never retire a connection the pool may reuse."""
    assert SERVER_KEEP_ALIVE_TIMEOUT_SEC > HTTPX_KEEPALIVE_EXPIRY_SEC
    assert UVICORN_DEFAULT_KEEP_ALIVE < HTTPX_KEEPALIVE_EXPIRY_SEC, (
        "control assumption: uvicorn's default is the broken side"
    )


def test_the_bar_is_a_sixty_second_think_time_not_a_six_second_one() -> None:
    """The requirement is that an agent may think for 60 s between two RPCs and
    still find its pooled connection alive. The margin is far wider than that —
    the server outlives the client pool itself, so the client is always the side
    that retires a connection, at any think-time the pool will hand one out
    for."""
    assert SERVER_KEEP_ALIVE_TIMEOUT_SEC >= 60.0
    assert SERVER_KEEP_ALIVE_TIMEOUT_SEC - HTTPX_KEEPALIVE_EXPIRY_SEC >= 60.0


# =============================================================================
# The idle-gap repro (one shared 6 s sleep drives both halves)
# =============================================================================

@pytest.mark.asyncio
async def test_idle_gap_closes_the_pooled_connection_only_at_uvicorns_default() -> None:
    """One sleep drives the default-control and Lite-timeout halves."""
    with running_mock_server(
        _app(), timeout_keep_alive=UVICORN_DEFAULT_KEEP_ALIVE,
    ) as control_url, running_mock_server(
        _app(), timeout_keep_alive=SERVER_KEEP_ALIVE_TIMEOUT_SEC,
    ) as fixed_url:
        control, fixed = _pooled_client(), _pooled_client()
        try:
            assert (await control.get(f"{control_url}/ping")).status_code == 200
            assert (await fixed.get(f"{fixed_url}/ping")).status_code == 200

            await asyncio.sleep(IDLE_GAP_SEC)

            assert _server_closed_the_connection(control) is True, (
                "control is vacuous: uvicorn's 5s default did not close a "
                f"connection idle for {IDLE_GAP_SEC}s"
            )
            assert _server_closed_the_connection(fixed) is False, (
                "the server closed a connection the client pool still holds — "
                "the exact precondition of the /step ReadError"
            )

            before = list(fixed._transport._pool.connections)[0]
            assert (await fixed.get(f"{fixed_url}/ping")).status_code == 200
            after = list(fixed._transport._pool.connections)[0]
            assert after is before, "the pooled connection was silently replaced"
        finally:
            await control.aclose()
            await fixed.aclose()
