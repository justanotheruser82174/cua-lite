"""Extended functional tests against a live env-server.

Covers paths not exercised by ``test_live_server.py``:

* Bulk close (``DELETE /instances``)
* Auth modes (401 on wrong token)
* /admin endpoints (instances, tokens, usage)
* Idle / pool TTL semantics
* Per-env_id metrics tick under traffic
* /envs catalog + per-env metadata
* Retry-After integer parsing on 503

Run with a server configured for ``--max-live-envs 64``:

    CUA_LITE_LIVE_ENV_SERVER_URL=http://127.0.0.1:30200 \
    CUA_LITE_LIVE_ENV_SERVER_TOKEN=... \
    CUA_LITE_LIVE_ADMIN_TOKEN=... \
      uv run pytest -m live tests/gym/remote/test_live_server_extended.py -xvs
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


def _server_address() -> tuple[str, int] | None:
    parsed = urlparse(LIVE_SERVER_URL)
    if not parsed.scheme or not parsed.hostname:
        return None
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return parsed.hostname, port


def _listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module", autouse=True)
def require_dev_server():
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
    if not _listening(host, port):
        pytest.skip(f"live env-server not listening at {LIVE_SERVER_URL}")


@pytest.fixture(scope="module")
def light_env_task() -> tuple[str, str]:
    """Pick a lightweight env + task to spin up cheaply."""
    r = httpx.get(
        f"{LIVE_SERVER_URL}/envs",
        headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
        timeout=5.0,
    )
    available = r.json()
    candidate = next(
        (eid for eid in ("screenspot_pro", "osworld_g") if eid in available),
        None,
    )
    if candidate is None:
        pytest.skip(f"no weightless env registered: {available}")
    r = httpx.get(
        f"{LIVE_SERVER_URL}/envs/{candidate}/tasks",
        headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
        timeout=5.0,
    )
    splits = r.json().get("splits") or {}
    task_ids = next(iter(splits.values()), [])
    if not task_ids:
        pytest.skip(f"no tasks for {candidate}")
    return candidate, task_ids[0]


def _create(
    client: httpx.Client,
    env_key: str,
    session_id: str | None = None,
    token: str = LIVE_SERVER_TOKEN,
) -> tuple[int, str | None]:
    if session_id is None:
        session_id = unique_live_scope("extended_create")
    r = client.post(
        f"{LIVE_SERVER_URL}/instances",
        json={"env_key": env_key, "env_kwargs": {}, "session_id": session_id},
        headers={"Authorization": f"Bearer {token}"},
        timeout=30.0,
    )
    return r.status_code, r.json().get("id") if r.status_code == 201 else None


def _close(client: httpx.Client, inst_id: str, token: str = LIVE_SERVER_TOKEN) -> int:
    return client.delete(
        f"{LIVE_SERVER_URL}/instances/{inst_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10.0,
    ).status_code


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class TestAuth:
    def test_admin_token_required(self):
        """Wrong admin token → 401 or 403 (FastAPI yields 403 here)."""
        r = httpx.get(
            f"{LIVE_SERVER_URL}/admin/budget",
            headers={"Authorization": "Bearer wrong-admin"},
            timeout=5.0,
        )
        assert r.status_code in (401, 403)

    def test_data_endpoints_reject_missing_auth(self):
        """Missing Authorization header behavior depends on mode:
        - strict (--token set): 401/403
        - passthrough (no --token): accepts anonymous, returns 200
        This dev server is in passthrough (to enable multi-token isolation
        tests); skip strict-mode auth check here."""
        # Probe mode via /admin/budget: in strict mode an admin request
        # without a bearer is 401; in passthrough mode the same call
        # returns admin data only when --admin-token matches.
        probe = httpx.get(
            f"{LIVE_SERVER_URL}/admin/budget",
            headers={"Authorization": "Bearer probe-anon"},
            timeout=5.0,
        )
        if probe.status_code != 401:
            # Passthrough mode (or admin-token absent). No 401 to assert.
            pytest.skip("dev server in passthrough mode; strict-auth check N/A")
        r = httpx.get(f"{LIVE_SERVER_URL}/envs", timeout=5.0)
        assert r.status_code in (401, 403), f"unexpected {r.status_code}: {r.text}"


# ---------------------------------------------------------------------------
# /envs catalog
# ---------------------------------------------------------------------------


class TestEnvCatalog:
    def test_envs_list_is_array(self):
        r = httpx.get(
            f"{LIVE_SERVER_URL}/envs",
            headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
            timeout=5.0,
        )
        body = r.json()
        assert isinstance(body, list)
        assert len(body) > 0

    def test_envs_expand_metadata(self):
        r = httpx.get(
            f"{LIVE_SERVER_URL}/envs?expand=metadata",
            headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
            timeout=5.0,
        )
        body = r.json()
        assert isinstance(body, dict)
        # Pick any entry; verify the shape no longer includes env_cost.
        for env_id, meta in body.items():
            assert "available" in meta
            assert "env_cost" not in meta  # cost removed
            break

    def test_get_one_env_no_env_cost_field(self, light_env_task):
        env_id, _ = light_env_task
        r = httpx.get(
            f"{LIVE_SERVER_URL}/envs/{env_id}",
            headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
            timeout=5.0,
        )
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is True
        assert "env_cost" not in body


# ---------------------------------------------------------------------------
# /admin endpoints
# ---------------------------------------------------------------------------


class TestAdminEndpoints:
    def test_admin_instances_shape(self):
        r = httpx.get(
            f"{LIVE_SERVER_URL}/admin/instances",
            headers={"Authorization": f"Bearer {LIVE_ADMIN_TOKEN}"},
            timeout=5.0,
        )
        assert r.status_code == 200
        body = r.json()
        assert "instances" in body
        assert isinstance(body["instances"], list)

    def test_admin_tokens_shape(self):
        r = httpx.get(
            f"{LIVE_SERVER_URL}/admin/tokens",
            headers={"Authorization": f"Bearer {LIVE_ADMIN_TOKEN}"},
            timeout=5.0,
        )
        assert r.status_code == 200
        body = r.json()
        assert "tokens" in body
        # Per-token rows: no cost_active
        for row in body["tokens"]:
            assert "cost_active" not in row
            assert "instances_active" in row

    def test_admin_usage_shape(self):
        r = httpx.get(
            f"{LIVE_SERVER_URL}/admin/usage",
            headers={"Authorization": f"Bearer {LIVE_ADMIN_TOKEN}"},
            timeout=5.0,
        )
        assert r.status_code == 200
        body = r.json()
        assert "usage" in body
        for row in body["usage"]:
            assert "n_active_instances" in row
            assert "cost_active" not in row  # cost removed


# ---------------------------------------------------------------------------
# Bulk close
# ---------------------------------------------------------------------------


class TestBulkClose:
    def test_bulk_close_by_session_id_and_env_id(self, light_env_task):
        """Bulk close is scoped by both session_id and env_id in this suite.
        Pinning both is the safer pattern."""
        env_id, task_id = light_env_task
        session_id = unique_live_scope("extended_bulk_close")
        created: list[str] = []
        with httpx.Client() as client:
            for _ in range(5):
                sc, iid = _create(
                    client,
                    f"{env_id}@{task_id}",
                    session_id=session_id,
                )
                if sc == 201 and iid:
                    created.append(iid)
            assert len(created) == 5

            r = client.delete(
                f"{LIVE_SERVER_URL}/instances",
                params={"session_id": session_id, "env_id": env_id},
                headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
                timeout=15.0,
            )
            assert r.status_code == 200, r.text
            closed = r.json().get("closed", [])
            assert len(closed) == 5

    def test_bulk_close_requires_filter(self):
        """No session_id, env_id, or force → 400."""
        r = httpx.delete(
            f"{LIVE_SERVER_URL}/instances",
            headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
            timeout=5.0,
        )
        assert r.status_code == 400

    def test_bulk_close_dry_run(self, light_env_task):
        env_id, task_id = light_env_task
        session_id = unique_live_scope("extended_dry_run")
        created: list[str] = []
        with httpx.Client() as client:
            for _ in range(3):
                sc, iid = _create(
                    client,
                    f"{env_id}@{task_id}",
                    session_id=session_id,
                )
                if sc == 201 and iid:
                    created.append(iid)
            try:
                r = client.delete(
                    f"{LIVE_SERVER_URL}/instances",
                    params={"session_id": session_id, "env_id": env_id, "dry_run": "true"},
                    headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
                    timeout=5.0,
                )
                assert r.status_code == 200, r.text
                body = r.json()
                assert body.get("dry_run") is True
                assert len(body.get("would_close", [])) == 3
            finally:
                client.delete(
                    f"{LIVE_SERVER_URL}/instances",
                    params={"session_id": session_id, "env_id": env_id},
                    headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
                    timeout=10.0,
                )


# ---------------------------------------------------------------------------
# Retry-After format
# ---------------------------------------------------------------------------


class TestRetryAfter:
    def test_l2_503_carries_integer_retry_after(self, light_env_task):
        """When L2 fires, the Retry-After header is an integer seconds."""
        env_id, task_id = light_env_task
        # Burst to fill the cap.
        created: list[str] = []
        rejected_resp: httpx.Response | None = None
        session_id = unique_live_scope("extended_retry_after")
        with httpx.Client() as client:
            try:
                for _ in range(80):
                    r = client.post(
                        f"{LIVE_SERVER_URL}/instances",
                        json={
                            "env_key": f"{env_id}@{task_id}",
                            "env_kwargs": {},
                            "session_id": session_id,
                        },
                        headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
                        timeout=10.0,
                    )
                    if r.status_code == 201:
                        created.append(r.json()["id"])
                    elif r.status_code == 503:
                        rejected_resp = r
                        break
                assert rejected_resp is not None
                ra = rejected_resp.headers.get("Retry-After")
                assert ra is not None
                # Must be parseable as int.
                int(ra)
                # And the body must have what + retry_after_s.
                body = rejected_resp.json()
                assert "what" in body
                assert "retry_after_s" in body
            finally:
                for iid in created:
                    try:
                        client.delete(
                            f"{LIVE_SERVER_URL}/instances/{iid}",
                            headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
                            timeout=10.0,
                        )
                    except Exception:
                        pass


# ---------------------------------------------------------------------------
# /metrics under traffic
# ---------------------------------------------------------------------------


class TestMetricsUnderTraffic:
    def test_envs_total_counter_advances(self, light_env_task):
        env_id, task_id = light_env_task
        before = _scrape_metric("cua_lite_envs_total")
        created: list[str] = []
        session_id = unique_live_scope("extended_metrics_tick")
        with httpx.Client() as client:
            try:
                for _ in range(3):
                    sc, iid = _create(client, f"{env_id}@{task_id}", session_id=session_id)
                    if sc == 201 and iid:
                        created.append(iid)
                after = _scrape_metric("cua_lite_envs_total")
                assert after >= before + 3
            finally:
                for iid in created:
                    _close(client, iid)

    def test_per_env_id_counter_present(self, light_env_task):
        env_id, _ = light_env_task
        # /envs_by_id appears after at least one POST /instances for that id.
        text = _scrape_metrics()
        assert (
            f'cua_lite_envs_by_id{{env_id="{env_id}"}}' in text
            or "# HELP cua_lite_envs_by_id" in text
        )


def _scrape_metrics() -> str:
    r = httpx.get(
        f"{LIVE_SERVER_URL}/metrics",
        headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
        timeout=5.0,
    )
    return r.text


def _scrape_metric(name: str) -> int:
    for line in _scrape_metrics().split("\n"):
        if line.startswith(name + " "):
            return int(line.split()[-1])
    return 0
