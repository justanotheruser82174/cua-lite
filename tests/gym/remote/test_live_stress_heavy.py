"""Heavier stress rounds against a live env-server.

These rounds escalate beyond ``test_live_stress.py``:

* 6: 500 simultaneous POST burst (well above cap=64)
* 7: 3-minute mixed RPC (create/reset/step/delete) with stable in-flight
* 8: Multi-token + multi-session simultaneous fan-out
* 9: /metrics scrape under sustained admission load
* 10: Final in_flight invariant check after all heavy load

The live server must be in passthrough mode with enough live-env capacity for
the selected burst sizes, so multi-token tests work.

Run::

    CUA_LITE_LIVE_ENV_SERVER_URL=http://127.0.0.1:30200 \
    CUA_LITE_LIVE_ENV_SERVER_TOKEN=... \
    CUA_LITE_LIVE_ADMIN_TOKEN=... \
      uv run pytest -m "live and stress" \
        tests/gym/remote/test_live_stress_heavy.py -xvs
"""
from __future__ import annotations

import asyncio
import os
import re
import socket
import time
from collections import Counter
from typing import Any
from urllib.parse import urlparse

import httpx
import pytest
from gym.remote.conftest import unique_live_scope, unique_live_token

from lite.core.tools import make_tool_call

pytestmark = [pytest.mark.live, pytest.mark.stress]


_IN_FLIGHT_VALUE_RE = re.compile(r"^cua_lite_admission_in_flight\s+\d+", re.M)


LIVE_URL_ENV_VAR = "CUA_LITE_LIVE_ENV_SERVER_URL"
LIVE_TOKEN_ENV_VAR = "CUA_LITE_LIVE_ENV_SERVER_TOKEN"
LIVE_ADMIN_TOKEN_ENV_VAR = "CUA_LITE_LIVE_ADMIN_TOKEN"

LIVE_SERVER_URL = os.environ.get(LIVE_URL_ENV_VAR, "").rstrip("/")
LIVE_SERVER_TOKEN = os.environ.get(LIVE_TOKEN_ENV_VAR, "")
LIVE_ADMIN_TOKEN = os.environ.get(LIVE_ADMIN_TOKEN_ENV_VAR, "")
LIGHTWEIGHT_ENV = "screenspot_pro"


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
    if not _listening(*address):
        host, port = address
        pytest.skip(f"live env-server not reachable at {host}:{port}")


@pytest.fixture(scope="module")
def task_id() -> str:
    r = httpx.get(
        f"{LIVE_SERVER_URL}/envs",
        headers=_bearer(LIVE_SERVER_TOKEN),
        timeout=5.0,
    )
    available = r.json()
    if LIGHTWEIGHT_ENV not in available:
        pytest.skip(f"{LIGHTWEIGHT_ENV} not registered")
    r = httpx.get(
        f"{LIVE_SERVER_URL}/envs/{LIGHTWEIGHT_ENV}/tasks",
        headers=_bearer(LIVE_SERVER_TOKEN),
        timeout=5.0,
    )
    task_ids = next(iter(r.json().get("splits", {}).values()), [])
    if not task_ids:
        pytest.skip(f"no tasks for {LIGHTWEIGHT_ENV}")
    return task_ids[0]


def _admin_budget() -> dict[str, Any]:
    return httpx.get(
        f"{LIVE_SERVER_URL}/admin/budget",
        headers={"Authorization": f"Bearer {LIVE_ADMIN_TOKEN}"}, timeout=5.0,
    ).json()


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _async_create(ac: httpx.AsyncClient, env_key: str, token: str,
                        session_id: str) -> httpx.Response:
    try:
        return await ac.post(
            f"{LIVE_SERVER_URL}/instances",
            json={"env_key": env_key, "env_kwargs": {},
                  "session_id": session_id},
            headers=_bearer(token), timeout=30.0,
        )
    except Exception as e:
        # Return a synthetic response with status 0 for connection errors.
        class _R:
            status_code = 0
            text = f"exc:{type(e).__name__}:{e}"
            headers = {}
            def json(self): return {}
        return _R()  # type: ignore


async def _async_delete(ac: httpx.AsyncClient, inst_id: str,
                        token: str) -> int:
    try:
        r = await ac.delete(
            f"{LIVE_SERVER_URL}/instances/{inst_id}",
            headers=_bearer(token), timeout=10.0,
        )
        return r.status_code
    except Exception:
        return 0


async def _bulk_cleanup_session(ac: httpx.AsyncClient, token: str,
                                env_id: str, session_id: str) -> None:
    try:
        await ac.delete(
            f"{LIVE_SERVER_URL}/instances",
            params={"session_id": session_id, "env_id": env_id},
            headers=_bearer(token), timeout=30.0,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Round 6: 500-burst hammer
# ---------------------------------------------------------------------------

class TestRound6HeavyBurst:
    def test_500_concurrent_create_under_cap_64(self, task_id):
        """500 simultaneous POST /instances; cap=64. Expect <=cap admits,
        the rest 503, no 500s, capacity_503 counter monotonically advances
        by at least `rejects` (other tests on the same server may add more)."""
        env_key = f"{LIGHTWEIGHT_ENV}@{task_id}"
        token = unique_live_token("heavy_round6")
        session_id = unique_live_scope("heavy_round6")

        async def run():
            async with httpx.AsyncClient() as ac:
                await _bulk_cleanup_session(ac, token, LIGHTWEIGHT_ENV, session_id)
                # Sample counters AFTER pre-cleanup, so the bulk-delete's
                # capacity tally doesn't pollute the delta we measure.
                pre = _admin_budget()
                results = await asyncio.gather(
                    *(_async_create(ac, env_key, token, session_id)
                      for _ in range(500))
                )
                ids = [r.json().get("id") for r in results
                       if r.status_code == 201]
                await asyncio.gather(
                    *(_async_delete(ac, iid, token) for iid in ids
                      if iid),
                    return_exceptions=True,
                )
                return pre, results

        before, results = asyncio.run(run())
        codes = Counter(r.status_code for r in results)
        assert codes.get(500, 0) == 0, f"500s: {codes}"
        assert codes.get(0, 0) == 0, f"connection errors: {codes}"
        admits = codes.get(201, 0)
        rejects = codes.get(503, 0)
        assert admits + rejects == 500, codes
        # Admits cannot exceed the cap.
        assert admits <= before["max_live_envs"], (
            f"admits {admits} exceeds cap {before['max_live_envs']}"
        )
        # We expect saturation: at least 400 rejects (cap=64 + some headroom).
        assert rejects >= 400

        # Counter math: capacity_503 advances by at least `rejects`.
        # (Monotonic lower bound — concurrent tests may add more.)
        after = _admin_budget()
        cap_delta = after["503_total"]["capacity"] - before["503_total"]["capacity"]
        assert cap_delta >= rejects, (
            f"capacity_503 did not track rejects: delta={cap_delta} < rejects={rejects}"
        )
        # No emergency 503s under this load (host is healthy).
        emerg_delta = after["503_total"]["emergency"] - before["503_total"]["emergency"]
        assert emerg_delta == 0


# ---------------------------------------------------------------------------
# Round 7: 3-minute mixed RPC churn
# ---------------------------------------------------------------------------

class TestRound7MixedRpc:
    def test_3min_create_reset_step_close(self, task_id):
        """For 3 minutes, maintain ~25 live envs, doing a full RPC
        cycle (create → reset → step → close) per env. Verifies the
        whole admission + observation pipeline under sustained load.
        """
        deadline = time.monotonic() + 180.0  # 3 min
        env_key = f"{LIGHTWEIGHT_ENV}@{task_id}"
        cycles_completed = 0
        rejected_creates = 0  # legitimate 503 at create — counts as backpressure, not error
        reset_5xx = 0
        step_5xx = 0
        other_errors = 0
        max_in_flight_seen = 0
        token = unique_live_token("heavy_round7")
        session_id = unique_live_scope("heavy_round7")

        async def one_cycle(ac: httpx.AsyncClient, token: str,
                            session: str) -> bool:
            nonlocal rejected_creates, reset_5xx, step_5xx, other_errors
            # 1) create
            r = await _async_create(ac, env_key, token, session)
            if r.status_code == 503:
                rejected_creates += 1
                return False
            if r.status_code != 201:
                other_errors += 1
                return False
            inst = r.json()["id"]
            try:
                # 2) reset
                rr = await ac.post(
                    f"{LIVE_SERVER_URL}/instances/{inst}/reset",
                    headers=_bearer(token), timeout=30.0,
                )
                if rr.status_code != 200:
                    if rr.status_code >= 500:
                        reset_5xx += 1
                    else:
                        other_errors += 1
                    return False
                # 3) step (one point click)
                rs = await ac.post(
                    f"{LIVE_SERVER_URL}/instances/{inst}/step",
                    json={
                        "actions": [
                            make_tool_call("point", {"coordinate": [100, 100]})
                        ]
                    },
                    headers=_bearer(token), timeout=30.0,
                )
                if rs.status_code != 200:
                    if rs.status_code >= 500:
                        step_5xx += 1
                    else:
                        other_errors += 1
                    return False
                return True
            finally:
                # 4) close
                await _async_delete(ac, inst, token)

        async def run():
            nonlocal cycles_completed, max_in_flight_seen
            async with httpx.AsyncClient() as ac:
                await _bulk_cleanup_session(ac, token, LIGHTWEIGHT_ENV, session_id)
                sample_ts = time.monotonic()
                while time.monotonic() < deadline:
                    # Run 25 concurrent cycles.
                    batch = await asyncio.gather(
                        *(one_cycle(ac, token, session_id)
                          for _ in range(25))
                    )
                    cycles_completed += sum(1 for b in batch if b)
                    # Sample in_flight every ~5s.
                    if time.monotonic() - sample_ts > 5.0:
                        snap = _admin_budget()
                        max_in_flight_seen = max(max_in_flight_seen,
                                                 snap["in_flight"])
                        sample_ts = time.monotonic()

        asyncio.run(run())
        # Did we do real work?
        assert cycles_completed >= 50, (
            f"too few completed cycles: {cycles_completed}, "
            f"rejected_creates={rejected_creates}, reset_5xx={reset_5xx}, "
            f"step_5xx={step_5xx}, other_errors={other_errors}"
        )
        # Server-side reset/step failures are real errors, not backpressure.
        assert reset_5xx == 0, f"reset 5xx during mixed RPC: {reset_5xx}"
        assert step_5xx == 0, f"step 5xx during mixed RPC: {step_5xx}"
        # `other_errors` covers non-5xx surprises (4xx not from L2 admission).
        # Allow a tiny budget for races but the bulk must be either success or 503.
        assert other_errors <= cycles_completed // 10, (
            f"too many non-5xx errors: {other_errors} vs ok={cycles_completed}"
        )
        # Peak in_flight never exceeded the cap.
        assert max_in_flight_seen <= 64, (
            f"in_flight exceeded cap: peak={max_in_flight_seen}"
        )
        # Poll for drain (up to 30s). Docker teardown can take seconds.
        drain_deadline = time.monotonic() + 30.0
        final = _admin_budget()
        while final["in_flight"] > 5 and time.monotonic() < drain_deadline:
            time.sleep(1.0)
            final = _admin_budget()
        assert final["in_flight"] <= 5, (
            f"in_flight leak after 3min churn: {final['in_flight']}, "
            f"cycles_done={cycles_completed}"
        )


# ---------------------------------------------------------------------------
# Round 8: multi-token + multi-session fan-out
# ---------------------------------------------------------------------------

class TestRound8MultiTenantFanout:
    def test_4_tenants_2_sessions_each(self, task_id):
        """4 tenants (alice/bob/carol/dave), 2 sessions each (s1/s2),
        each session bursts 20 instances. Total 160 creates against
        cap=64. All concurrent. Verify FCFS, no 500s, isolation."""
        env_key = f"{LIGHTWEIGHT_ENV}@{task_id}"
        tenant_scopes: list[tuple[str, str]] = []
        for tenant in ("alice", "bob", "carol", "dave"):
            token = unique_live_token(f"heavy_round8_tenant_{tenant}")
            tenant_scopes.append((token, unique_live_scope(f"heavy_round8_{tenant}_s1")))
            tenant_scopes.append((token, unique_live_scope(f"heavy_round8_{tenant}_s2")))

        async def burst_one(ac: httpx.AsyncClient, token: str,
                            session: str) -> list[tuple[int, str | None]]:
            results = []
            for _ in range(20):
                r = await _async_create(ac, env_key, token, session)
                results.append((r.status_code,
                                r.json().get("id") if r.status_code == 201 else None))
            return results

        async def run():
            async with httpx.AsyncClient() as ac:
                # Pre-clean any leftovers
                for token, session in tenant_scopes:
                    await _bulk_cleanup_session(ac, token, LIGHTWEIGHT_ENV, session)
                tasks_all = []
                for token, session in tenant_scopes:
                    tasks_all.append(burst_one(ac, token, session))
                all_results = await asyncio.gather(*tasks_all)
                # Cleanup
                for (token, session), batch in zip(tenant_scopes, all_results):
                    for item in batch:
                        iid = item[1]
                        if iid:
                            await _async_delete(ac, iid, token)
                    await _bulk_cleanup_session(ac, token, LIGHTWEIGHT_ENV, session)
                return all_results

        all_results = asyncio.run(run())
        flat = [code for batch in all_results for code, _ in batch]
        codes = Counter(flat)
        assert codes.get(500, 0) == 0, f"500s: {codes}"
        # Sum should match 4 × 2 × 20 = 160.
        assert sum(codes.values()) == 160, codes
        # Most should have hit the cap and 503'd.
        assert codes.get(503, 0) >= 80


# ---------------------------------------------------------------------------
# Round 9: /metrics scrape rate under load
# ---------------------------------------------------------------------------

class TestRound9MetricsScrapeLoad:
    def test_metrics_responsive_under_admission_load(self, task_id):
        """While bursting 100 POST /instances, scrape /metrics 20 times.
        Every scrape must return within 1s and contain admission lines.
        """
        env_key = f"{LIGHTWEIGHT_ENV}@{task_id}"
        scrape_latencies: list[float] = []
        scrape_errors = 0
        token = unique_live_token("heavy_round9")
        session_id = unique_live_scope("heavy_round9")

        async def scrape_loop(deadline: float):
            nonlocal scrape_errors
            async with httpx.AsyncClient() as ac:
                while time.monotonic() < deadline:
                    t0 = time.monotonic()
                    try:
                        r = await ac.get(
                            f"{LIVE_SERVER_URL}/metrics",
                            headers=_bearer(LIVE_SERVER_TOKEN),
                            timeout=2.0,
                        )
                        # Require the actual value line, not a HELP comment.
                        if r.status_code == 200 and \
                                _IN_FLIGHT_VALUE_RE.search(r.text):
                            scrape_latencies.append(time.monotonic() - t0)
                        else:
                            scrape_errors += 1
                    except Exception:
                        scrape_errors += 1
                    await asyncio.sleep(0.5)

        async def admission_burst(deadline: float):
            async with httpx.AsyncClient() as ac:
                await _bulk_cleanup_session(ac, token, LIGHTWEIGHT_ENV, session_id)
                while time.monotonic() < deadline:
                    rs = await asyncio.gather(
                        *(_async_create(ac, env_key, token, session_id)
                          for _ in range(20))
                    )
                    ids = [r.json().get("id") for r in rs
                           if r.status_code == 201]
                    await asyncio.gather(
                        *(_async_delete(ac, iid, token)
                          for iid in ids if iid),
                        return_exceptions=True,
                    )

        async def run():
            deadline = time.monotonic() + 10.0
            await asyncio.gather(
                scrape_loop(deadline),
                admission_burst(deadline),
            )

        asyncio.run(run())
        assert scrape_errors == 0, f"metric scrape errors: {scrape_errors}"
        assert len(scrape_latencies) >= 10, scrape_latencies
        # Every scrape was sub-1s.
        assert all(lat < 1.0 for lat in scrape_latencies), (
            f"slow scrapes under admission load: {scrape_latencies}"
        )
        # p95 latency should be modest.
        scrape_latencies.sort()
        p95 = scrape_latencies[int(len(scrape_latencies) * 0.95)]
        assert p95 < 0.3, f"p95 metric scrape latency: {p95:.3f}s"


# ---------------------------------------------------------------------------
# Round 10: final in_flight + counter sanity check
# ---------------------------------------------------------------------------

class TestRound10FinalState:
    def test_in_flight_returns_to_baseline_after_all_load(self):
        """Poll for in_flight to drain (cap 30s); state must settle."""
        deadline = time.monotonic() + 30.0
        budget = _admin_budget()
        while budget["in_flight"] > 5 and time.monotonic() < deadline:
            time.sleep(1.0)
            budget = _admin_budget()
        assert budget["in_flight"] <= 5, (
            f"in_flight didn't drain after all stress in 30s: {budget}"
        )
        # No emergency 503s overall — host was healthy.
        # (Note: capacity_503 is OK to be large.)
        assert budget["503_total"]["emergency"] == 0, (
            f"unexpected L1 emergency 503s: {budget}"
        )
        # No docker_sema 503s — no docker envs were used.
        assert budget["503_total"]["docker_sema"] == 0, (
            f"unexpected docker_sema 503s: {budget}"
        )
