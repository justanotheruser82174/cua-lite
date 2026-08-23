"""Shared helpers for the env-server (``lite.gym.remote``) test suite.

De-duplicates the admission/auth and socket-harness boilerplate that
several remote tests need but each used to hand-roll:

  * :func:`permissive_admission_config` — an :class:`AdmissionConfig` whose
    L1 emergency thresholds (RAM percent / absolute free / load-per-cpu) are
    set so high they never fire on whatever CI host runs the tests, so L2
    (the in-flight cap) is the only thing exercised.
  * :func:`make_test_admission` — the gate wrapping that config.
  * :func:`bearer_header` — the ``Authorization: Bearer <token>`` header dict.
  * :func:`running_mock_server` / :func:`free_port` — an in-process uvicorn
    serving a mock app on a free localhost port, for REAL-LiteEnvClient-vs-
    mock-server tests (actual httpx transport, unlike fastapi.testclient).

Import them directly (they take params, so they are plain functions, not
fixtures)::

    from gym.remote.conftest import make_test_admission, bearer_header

Run::

    env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
        uv run pytest tests/gym/remote -q
"""
from __future__ import annotations

import contextlib
import os
import re
import time
from uuid import uuid4

import pytest

from lite.gym.remote.admission import AdmissionConfig, AdmissionGate

_LIVE_SCOPE_SAFE_RE = re.compile(r"[^A-Za-z0-9_]+")
_LIVE_RUN_ID = f"{os.getpid()}_{time.time_ns()}_{uuid4().hex[:12]}"


def unique_live_scope(label: str) -> str:
    """Return a run-unique, Docker-name-safe live-test scope segment."""
    safe_label = _LIVE_SCOPE_SAFE_RE.sub("_", label).strip("_") or "scope"
    return f"{safe_label}_{_LIVE_RUN_ID}_{uuid4().hex[:8]}"


def unique_live_token(label: str) -> str:
    """Return a run-unique bearer token for passthrough live-server tests."""
    return f"token_{unique_live_scope(label)}"


@pytest.fixture(autouse=True)
def _isolate_serve_locally():
    """Snapshot + restore the process-global serve-locally flag around each test.

    ``lite.gym.remote.server.make_app`` calls ``serve_locally()``, which flips
    the module-global ``lite.gym.utils.server.routing._serve_locally`` True for the
    whole worker process and never resets it (correct in production — a server
    process is server-mode for life). But this suite builds many apps in one
    worker, so without isolation a server-mode test leaves the flag True and a
    later test needing client/remote routing (e.g.
    ``test_ensure_skipped_in_remote_mode``) inherits the stale True — passing in
    isolation but failing in-suite. Restore the pre-test value after each test so
    one test's app build can't leak server-mode into the next.
    """
    import lite.gym.utils.server.routing as _routing
    saved = _routing._serve_locally
    try:
        yield
    finally:
        _routing._serve_locally = saved


def permissive_admission_config(max_live_envs: int = 100) -> AdmissionConfig:
    """L1 thresholds set so high they never fire — for L2-only tests."""
    return AdmissionConfig(
        max_live_envs=max_live_envs,
        emergency_ram_pct=99.9,
        emergency_ram_free_min_bytes=1,  # 1 byte = never fires
        emergency_load_per_cpu=9999.0,
        docker_create_concurrency=8,
    )


def make_test_admission(max_live_envs: int = 100) -> AdmissionGate:
    """Permissive admission gate so L1 (host sensors) never fires on
    whatever host these tests happen to run on."""
    return AdmissionGate(permissive_admission_config(max_live_envs=max_live_envs))


def bearer_header(token: str) -> dict[str, str]:
    """``Authorization: Bearer <token>`` header dict."""
    return {"Authorization": f"Bearer {token}"}


def free_port() -> int:
    """Pick an ephemeral port the OS confirms is free at call time."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def running_mock_server(app, *, timeout_keep_alive: float | None = None):
    """In-process uvicorn serving ``app`` on a free localhost port; yields the
    base URL. For REAL-``LiteEnvClient``-vs-mock-server tests (the actual
    httpx transport, unlike fastapi.testclient) — shared by
    test_capacity_503_retry.py and test_client_identity_inertness.py.

    ``timeout_keep_alive`` overrides uvicorn's idle keep-alive (default 5 s).
    ``test_keep_alive_contract.py`` uses it to stand up both sides of the I16
    A/B in one process."""
    import socket
    import threading
    import time

    import uvicorn

    port = free_port()
    keep_alive_kwargs = (
        {} if timeout_keep_alive is None
        else {"timeout_keep_alive": timeout_keep_alive}
    )
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                            **keep_alive_kwargs)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # Wait for the socket to accept connections (uvicorn's "Started server"
    # comes ~50ms after thread start; poll the actual TCP listen).
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.05)
    else:
        raise RuntimeError(f"mock server failed to listen on {port}")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
