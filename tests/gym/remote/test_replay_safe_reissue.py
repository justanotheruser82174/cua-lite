"""The one retry rule, on the server's env↔container hop.

``may_reissue`` asks the two facts that decide whether a failed call may be sent
again, both read off something their owner set: ``replay_safe`` (the caller's
idempotence knowledge) and
:func:`~lite.gym.utils.backend.rpc.reached_worker` (the type the raiser
stamped). These tests pin the corners that used to be wrong:

* the server hop — a mid-response cut is sent once on ``/step``, while a
  connect-refused is re-issued because nothing executed;
* the producer half, against real sockets, including the one member of
  ``requests.ConnectionError`` that spans BOTH phases — a hangup after the
  request was written. ``json_rpc`` used to retry that member ``attempts`` times
  and then type it never-reached.

Run::

    uv run pytest tests/gym/remote/test_replay_safe_reissue.py -v
"""
from __future__ import annotations

import http.server
import socket
import threading

import pytest
from fastapi.testclient import TestClient
from gym.remote.conftest import bearer_header as _h
from gym.remote.conftest import make_test_admission as _make_test_admission

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools import make_tool_call
from lite.gym.base import LiteBaseEnv
from lite.gym.remote import server as srv
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult
from lite.gym.utils.backend.rpc import (
    is_transport_error,
    json_rpc,
    may_reissue,
    reached_worker,
)

TOKEN = "zzh-dev"


class _FlakyEnv(LiteBaseEnv):
    """Fails ``n_failures`` times with ``exc`` on the named op, then succeeds."""

    def __init__(self, *, op: str, exc: BaseException, n_failures: int) -> None:
        self._op = op
        self._exc = exc
        self._left = n_failures
        self.reset_calls = 0
        self.step_calls = 0

    def _runtime_metadata(self) -> LiteCUAMetadata:
        return LiteCUAMetadata(dims=("browser", "use"))

    def _maybe_fail(self, op: str) -> None:
        if op == self._op and self._left > 0:
            self._left -= 1
            raise self._exc

    async def reset(self) -> LiteEnvObservation:
        self.reset_calls += 1
        self._maybe_fail("reset")
        return LiteEnvObservation(image=None, text="ok")

    async def step(self, actions: list) -> LiteEnvStepResult:
        self.step_calls += 1
        self._maybe_fail("step")
        return LiteEnvStepResult()

    async def close(self) -> None:
        pass


def _client(env: _FlakyEnv, monkeypatch) -> tuple[TestClient, srv.State]:
    monkeypatch.setattr(srv, "_OUTER_RETRY_BASE_DELAY_S", 0.0)
    state = srv.State(
        admission=_make_test_admission(max_live_envs=10), idle_ttl_sec=3600.0,
    )
    app = srv.make_app(state, token=TOKEN)
    monkeypatch.setattr(srv.gym, "make", lambda env_key, **kw: env)
    # ``raise_server_exceptions=False``: we assert the 500 the client would
    # see, not the exception re-raised into the test process.
    return TestClient(app, raise_server_exceptions=False), state


def _create(c: TestClient) -> str:
    r = c.post("/instances", json={"env_key": "x", "env_kwargs": {}}, headers=_h(TOKEN))
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_mid_response_cut_on_step_is_not_reissued(monkeypatch):
    """The cut on /step: sent once, surfaced. ``replay_safe`` is False AND the
    raiser typed the failure as reached-the-worker, so nothing licenses a
    replay — the op may already have applied the action."""
    env = _FlakyEnv(
        op="step",
        exc=ConnectionResetError("osworld server /step cut mid-response: boom"),
        n_failures=1,
    )
    c, state = _client(env, monkeypatch)
    with c:
        id_ = _create(c)
        assert c.post(f"/instances/{id_}/reset", headers=_h(TOKEN)).status_code == 200
        r = c.post(
            f"/instances/{id_}/step",
            json={"actions": [make_tool_call("wait")]},
            headers=_h(TOKEN),
        )
    assert r.status_code == 500, r.text
    assert env.step_calls == 1
    assert state.outer_retry_fired["step"] == 0


def test_connect_refused_on_step_is_reissued(monkeypatch):
    """A connect phase that never completed PROVES nothing executed, so a
    ``/step`` may be re-issued even though it is not replay-safe.

    This is the fact ``replay_safe`` alone cannot express. Losing it regresses
    mid-episode recovery from 4 attempts to 0: a momentarily unreachable
    container fails the whole trajectory instead of recovering."""
    env = _FlakyEnv(
        op="step",
        exc=ConnectionRefusedError(
            "osworld server /step unreachable after 3 tries: [Errno 111]"
        ),
        n_failures=2,
    )
    c, state = _client(env, monkeypatch)
    with c:
        id_ = _create(c)
        assert c.post(f"/instances/{id_}/reset", headers=_h(TOKEN)).status_code == 200
        r = c.post(
            f"/instances/{id_}/step",
            json={"actions": [make_tool_call("wait")]},
            headers=_h(TOKEN),
        )
    assert r.status_code == 200, r.text
    assert env.step_calls == 3
    assert state.outer_retry_fired["step"] == 2
    assert state.outer_retry_recovered["step"] == 1


# ---------------------------------------------------------------------------
# The producer side, against a real socket
# ---------------------------------------------------------------------------

class _HangupAfterRequestHandler(http.server.BaseHTTPRequestHandler):
    """Read the ENTIRE request, then hang up with no response at all.

    This is ``http.client.RemoteDisconnected`` — the ambiguous member: the
    request was fully delivered, so the op may have run, yet ``requests`` reports
    it under the same ``ConnectionError`` class as a refused connect.
    """

    protocol_version = "HTTP/1.1"
    requests_seen = 0

    def do_POST(self):  # noqa: N802 — stdlib naming
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        type(self).requests_seen += 1
        self.close_connection = True  # no send_response: the peer just vanishes

    def log_message(self, *args):
        pass


def _no_sleep(monkeypatch) -> list[float]:
    """Replace ``json_rpc``'s backoff with a recorder.

    The module attribute is swapped, not ``time.sleep`` itself, so the sleep
    schedule becomes an assertion instead of wall-clock: an empty list means the
    loop never took its retry arm.
    """
    import types

    from lite.gym.utils.backend import rpc as rpc_mod

    slept: list[float] = []
    monkeypatch.setattr(rpc_mod, "time", types.SimpleNamespace(sleep=slept.append))
    return slept


def test_json_rpc_sends_a_hangup_after_the_request_exactly_once(monkeypatch):
    """The live hole this pins, against a real socket.

    A ``/step`` whose connection dies *after* the request was written used to be
    retried ``attempts`` times inside ``json_rpc`` and then flattened to a
    never-reached claim, which licensed the server's outer loop to send it up to
    4 more times. Up to 7 executions of one action, on a hop that must send it at
    most once.

    Two facts are asserted: the server observed exactly ONE request even though
    ``attempts=3``, and the type handed upstream answers ``reached_worker`` True.
    """
    pytest.importorskip("requests")
    slept = _no_sleep(monkeypatch)
    _HangupAfterRequestHandler.requests_seen = 0
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _HangupAfterRequestHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        with pytest.raises(ConnectionError) as ei:
            json_rpc(
                f"http://127.0.0.1:{httpd.server_port}", "/step",
                {"actions": [{"name": "click"}]},
                timeout=5.0, label="osworld", attempts=3,
            )
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert _HangupAfterRequestHandler.requests_seen == 1
    assert slept == []
    assert type(ei.value) is ConnectionResetError
    assert "lost the connection after the request was sent" in str(ei.value)
    assert is_transport_error(ei.value) is True
    assert reached_worker(ei.value) is True
    assert may_reissue(ei.value, replay_safe=False) is False
    assert may_reissue(ei.value, replay_safe=True) is True


def test_json_rpc_still_retries_the_connect_phase(monkeypatch):
    """The other half, which the split must NOT cost: refused-connect blips at
    boot are why the loop exists, and they are genuinely never-reached.

    The backoff schedule is the evidence — three connect attempts, not one — so a
    future "simplification" that drops the loop fails here instead of silently
    turning a warm-boot DNAT re-assert window into a failed episode."""
    pytest.importorskip("requests")
    slept = _no_sleep(monkeypatch)
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens here now
    with pytest.raises(ConnectionError) as ei:
        json_rpc(
            f"http://127.0.0.1:{port}", "/step", {"actions": []},
            timeout=1.0, label="osworld", attempts=3,
        )
    assert slept == [1.0, 2.0, 3.0]
    assert "unreachable after 3 tries" in str(ei.value)
    assert type(ei.value) is ConnectionRefusedError
    assert reached_worker(ei.value) is False
    assert may_reissue(ei.value, replay_safe=False) is True
