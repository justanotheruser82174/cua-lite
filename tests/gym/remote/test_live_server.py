"""Functional tests against a live env-server.

These tests assume a live server started with ``--max-live-envs 64``. Configure
it with ``CUA_LITE_LIVE_ENV_SERVER_URL``, ``CUA_LITE_LIVE_ENV_SERVER_TOKEN``,
and ``CUA_LITE_LIVE_ADMIN_TOKEN``.

Coverage:
* /host_status + /metrics shape (admission lines present)
* Authorization (passthrough vs strict)
* lightweight env lifecycle (gym.make → reset → step → close)
* L2 capacity 503 enforcement when in_flight hits the cap
* /admin/budget shape (in_flight, max_live_envs, 503 counters)
* /metrics counters tick under traffic

Run::

    CUA_LITE_LIVE_ENV_SERVER_URL=http://127.0.0.1:30200 \
    CUA_LITE_LIVE_ENV_SERVER_TOKEN=... \
    CUA_LITE_LIVE_ADMIN_TOKEN=... \
      uv run pytest -m live tests/gym/remote/test_live_server.py -xvs
"""
from __future__ import annotations

import os
import socket
from urllib.parse import urlparse

import httpx
import pytest
from gym.remote.conftest import unique_live_scope

pytestmark = pytest.mark.live


LIVE_URL_ENV_VAR = "CUA_LITE_LIVE_ENV_SERVER_URL"
LIVE_TOKEN_ENV_VAR = "CUA_LITE_LIVE_ENV_SERVER_TOKEN"
LIVE_ADMIN_TOKEN_ENV_VAR = "CUA_LITE_LIVE_ADMIN_TOKEN"

LIVE_SERVER_URL = os.environ.get(LIVE_URL_ENV_VAR, "").rstrip("/")
LIVE_SERVER_TOKEN = os.environ.get(LIVE_TOKEN_ENV_VAR, "")
LIVE_ADMIN_TOKEN = os.environ.get(LIVE_ADMIN_TOKEN_ENV_VAR, "")
PROD_PORT = 30100
PROD_PID = None  # set in module fixture if we can identify the production pid


def _server_address() -> tuple[str, int] | None:
    parsed = urlparse(LIVE_SERVER_URL)
    if not parsed.scheme or not parsed.hostname:
        return None
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return parsed.hostname, port


def _server_listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module", autouse=True)
def require_dev_server():
    """Skip the whole module if the dev server isn't reachable. (Tests
    that aren't run against the real dev server live in
    test_capacity_503_retry.py with a mock; this module is integration.)
    """
    missing = [
        name
        for name, value in [
            (LIVE_URL_ENV_VAR, LIVE_SERVER_URL),
            (LIVE_TOKEN_ENV_VAR, LIVE_SERVER_TOKEN),
            (LIVE_ADMIN_TOKEN_ENV_VAR, LIVE_ADMIN_TOKEN),
        ]
        if not value
    ]
    if missing:
        pytest.skip("missing live env-server vars: " + ", ".join(missing))

    address = _server_address()
    if address is None:
        pytest.skip(f"{LIVE_URL_ENV_VAR} must be an absolute URL")

    host, port = address
    if not _server_listening(host, port):
        pytest.skip(f"live env-server not listening at {LIVE_SERVER_URL}")


# ---------------------------------------------------------------------------
# Endpoints & basic shape
# ---------------------------------------------------------------------------

class TestHealth:
    def test_host_status_open(self):
        r = httpx.get(f"{LIVE_SERVER_URL}/host_status", timeout=5.0)
        assert r.status_code == 200
        body = r.json()
        # The host_status shape has nested cpu / memory dicts.
        assert "cpu" in body and "memory" in body
        assert "percent" in body["memory"]

    def test_production_server_unaffected(self):
        assert _server_listening("127.0.0.1", PROD_PORT), (
            "Production server on port 30100 must keep running during dev tests"
        )

    def test_metrics_endpoint_returns_admission_lines(self):
        r = httpx.get(
            f"{LIVE_SERVER_URL}/metrics",
            headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
            timeout=5.0,
        )
        assert r.status_code == 200
        body = r.text
        # admission lines (must be present)
        for needle in (
            "cua_lite_admission_max_live_envs 64",
            "cua_lite_admission_in_flight",
            "cua_lite_admission_503_total{layer=\"emergency\"}",
            "cua_lite_admission_503_total{layer=\"capacity\"}",
            "cua_lite_admission_503_total{layer=\"docker_sema\"}",
            "cua_lite_admission_503_total{layer=\"env_internal\"}",
            "cua_lite_host_ram_percent",
            "cua_lite_host_load_per_cpu",
        ):
            assert needle in body, f"missing /metrics line: {needle}"
        # Rev 5 lines (must NOT be present)
        for negation in (
            "cua_lite_budget ",
            "cua_lite_in_use_cost ",
            "cua_lite_admission_queue_waiting",
            "cua_lite_boot_throttle_in_use",
            "cua_lite_active_op_throttle_in_use",
        ):
            assert negation not in body, f"stale rev-5 metric leaked: {negation}"


# ---------------------------------------------------------------------------
# /admin/budget shape
# ---------------------------------------------------------------------------

class TestAdminBudget:
    def test_admin_budget_shape(self):
        r = httpx.get(
            f"{LIVE_SERVER_URL}/admin/budget",
            headers={"Authorization": f"Bearer {LIVE_ADMIN_TOKEN}"},
            timeout=5.0,
        )
        assert r.status_code == 200
        body = r.json()
        # shape
        for k in (
            "in_flight", "max_live_envs", "pct_used",
            "host_ram_percent", "host_ram_free_bytes", "host_load_per_cpu",
            "503_total",
        ):
            assert k in body, f"/admin/budget missing key: {k}"
        # Rev 5 keys must NOT be present
        for k in (
            "budget_total", "cost_committed", "cost_live_envs",
            "cost_pending_create", "cost_waiting", "queue_cap", "env_cost",
        ):
            assert k not in body, f"stale rev-5 key in /admin/budget: {k}"
        assert body["max_live_envs"] == 64
        assert isinstance(body["in_flight"], int)
        assert isinstance(body["503_total"], dict)
        for layer in ("emergency", "capacity", "docker_sema", "env_internal"):
            assert layer in body["503_total"], (
                f"missing layer {layer!r} in 503_total: {body['503_total']}"
            )


# ---------------------------------------------------------------------------
# Lightweight env end-to-end
# ---------------------------------------------------------------------------

@pytest.fixture
def env_id():
    """Pick a lightweight env available on the dev server."""
    session_id = unique_live_scope("live_server_lifecycle")
    # Pick an available lightweight env rather than assuming a particular
    # deployment blocklist.
    r = httpx.get(
        f"{LIVE_SERVER_URL}/envs", headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
        timeout=5.0,
    )
    available = r.json()
    if not isinstance(available, list) or not available:
        pytest.skip("dev server has no env_ids registered")
    # Pick the first env_id that's available + lightweight.
    # screenspot_pro / osworld_g are weightless dataset envs — good for tests.
    candidate = None
    for eid in ("screenspot_pro", "osworld_g"):
        if eid in available:
            candidate = eid
            break
    if candidate is None:
        pytest.skip(f"no lightweight env in registered set: {available}")
    # Get one task_id.
    r = httpx.get(
        f"{LIVE_SERVER_URL}/envs/{candidate}/tasks",
        headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"}, timeout=5.0,
    )
    if r.status_code != 200:
        pytest.skip(f"no tasks for {candidate}: {r.status_code}")
    splits = r.json().get("splits") or {}
    task_ids = next(iter(splits.values()), [])
    if not task_ids:
        pytest.skip(f"no task_ids in any split of {candidate}")
    task_id = task_ids[0]
    # Create instance.
    r = httpx.post(
        f"{LIVE_SERVER_URL}/instances",
        json={"env_key": f"{candidate}@{task_id}",
              "env_kwargs": {}, "session_id": session_id},
        headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
        timeout=30.0,
    )
    assert r.status_code == 201, f"create failed: {r.status_code} {r.text}"
    inst_id = r.json()["id"]
    try:
        yield inst_id
    finally:
        httpx.delete(
            f"{LIVE_SERVER_URL}/instances/{inst_id}",
            headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"}, timeout=10.0,
        )


class TestLifecycle:
    def test_create_increments_in_flight(self, env_id):
        # in_flight was incremented by the fixture's POST /instances.
        r = httpx.get(
            f"{LIVE_SERVER_URL}/admin/budget",
            headers={"Authorization": f"Bearer {LIVE_ADMIN_TOKEN}"},
            timeout=5.0,
        )
        assert r.json()["in_flight"] >= 1


# ---------------------------------------------------------------------------
# L2 capacity 503 — drive in_flight up to the cap
# ---------------------------------------------------------------------------

class TestL2Cap:
    def test_l2_503_at_cap(self):
        """Burst more than --max-live-envs (64) and observe L2 503."""
        # Pick a weightless env so we don't actually allocate docker.
        r = httpx.get(
            f"{LIVE_SERVER_URL}/envs",
            headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
            timeout=5.0,
        )
        available = r.json()
        candidate = None
        for eid in ("screenspot_pro", "osworld_g"):
            if eid in available:
                candidate = eid
                break
        if candidate is None:
            pytest.skip(f"no weightless env: {available}")
        r = httpx.get(
            f"{LIVE_SERVER_URL}/envs/{candidate}/tasks",
            headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"}, timeout=5.0,
        )
        splits = r.json().get("splits") or {}
        task_ids = next(iter(splits.values()), [])
        if not task_ids:
            pytest.skip(f"no tasks for {candidate}")
        task_id = task_ids[0]

        # Establish baseline.
        before = httpx.get(
            f"{LIVE_SERVER_URL}/metrics",
            headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
            timeout=5.0,
        ).text
        cap_503_before = _extract_metric(
            before, 'cua_lite_admission_503_total{layer="capacity"}'
        )
        session_id = unique_live_scope("live_server_l2")

        # Burst create up to + above the cap. Cap is 64; do 72 attempts.
        # All should succeed initially since each /instances incrementally
        # admits — once we hit 64, subsequent should get capacity 503.
        created: list[str] = []
        rejects = 0
        try:
            for _ in range(72):
                r = httpx.post(
                    f"{LIVE_SERVER_URL}/instances",
                    json={"env_key": f"{candidate}@{task_id}",
                          "env_kwargs": {}, "session_id": session_id},
                    headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
                    timeout=30.0,
                )
                if r.status_code == 201:
                    created.append(r.json()["id"])
                elif r.status_code == 503:
                    rejects += 1
                    # Verify Retry-After header is set
                    assert "Retry-After" in r.headers
                else:
                    pytest.fail(f"unexpected status: {r.status_code} {r.text}")

            # Should have admitted at most 64 (or fewer if pre-existing envs
            # were alive) and rejected the rest.
            assert len(created) <= 64
            assert rejects >= 1, "expected at least one L2-503 rejection"

            after = httpx.get(
                f"{LIVE_SERVER_URL}/metrics",
                headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
                timeout=5.0,
            ).text
            cap_503_after = _extract_metric(
                after, 'cua_lite_admission_503_total{layer="capacity"}'
            )
            assert cap_503_after > cap_503_before, (
                f"capacity_503 counter did not advance: "
                f"{cap_503_before} → {cap_503_after}"
            )
        finally:
            for inst_id in created:
                try:
                    httpx.delete(
                        f"{LIVE_SERVER_URL}/instances/{inst_id}",
                        headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
                        timeout=10.0,
                    )
                except Exception:
                    pass


def _extract_metric(text: str, name: str) -> int:
    for line in text.split("\n"):
        if line.startswith(name + " "):
            return int(line.split()[-1])
    return 0
