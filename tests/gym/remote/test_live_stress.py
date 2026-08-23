"""Stress tests against a live env-server.

These tests deliberately push the admission gate hard. Each test runs
for tens of seconds to minutes and asserts:

* Server stays responsive (no hung requests).
* Counters in /metrics stay self-consistent (no leaks, no torn reads).
* Production server on port 30100 is unaffected throughout.

Run::

    CUA_LITE_LIVE_ENV_SERVER_URL=http://127.0.0.1:30200 \
    CUA_LITE_LIVE_ENV_SERVER_TOKEN=... \
    CUA_LITE_LIVE_ADMIN_TOKEN=... \
      uv run pytest -m "live and stress" \
        tests/gym/remote/test_live_stress.py -xvs

These are intentionally NOT in the default test set — they're slow and
require a live env-server to be running.
"""
from __future__ import annotations

import asyncio
import os
import socket
import time
from collections import Counter
from typing import Any
from urllib.parse import urlparse

import httpx
import pytest
from gym.remote.conftest import unique_live_scope

pytestmark = [pytest.mark.live, pytest.mark.stress]


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
        name for name, value in [
            (LIVE_URL_ENV_VAR, LIVE_SERVER_URL),
            (LIVE_TOKEN_ENV_VAR, LIVE_SERVER_TOKEN),
            (LIVE_ADMIN_TOKEN_ENV_VAR, LIVE_ADMIN_TOKEN),
        ] if not value
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
def task_id() -> str:
    """Pick a usable task_id for the lightweight env."""
    r = httpx.get(
        f"{LIVE_SERVER_URL}/envs", headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
        timeout=5.0,
    )
    available = r.json()
    if LIGHTWEIGHT_ENV not in available:
        pytest.skip(f"{LIGHTWEIGHT_ENV} not registered")
    r = httpx.get(
        f"{LIVE_SERVER_URL}/envs/{LIGHTWEIGHT_ENV}/tasks",
        headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"}, timeout=5.0,
    )
    splits = r.json().get("splits") or {}
    task_ids = next(iter(splits.values()), [])
    if not task_ids:
        pytest.skip(f"no tasks for {LIGHTWEIGHT_ENV}")
    return task_ids[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metric(text: str, name: str) -> int:
    for line in text.split("\n"):
        if line.startswith(name + " "):
            return int(line.split()[-1])
    return 0


def _admin_budget() -> dict[str, Any]:
    r = httpx.get(
        f"{LIVE_SERVER_URL}/admin/budget",
        headers={"Authorization": f"Bearer {LIVE_ADMIN_TOKEN}"},
        timeout=5.0,
    )
    return r.json()


def _session_id(label: str) -> str:
    return unique_live_scope(f"stress_{label}")


async def _create_one(
    client: httpx.AsyncClient,
    task_id: str,
    session_id: str,
) -> tuple[int, str | None]:
    try:
        r = await client.post(
            f"{LIVE_SERVER_URL}/instances",
            json={"env_key": f"{LIGHTWEIGHT_ENV}@{task_id}",
                  "env_kwargs": {}, "session_id": session_id},
            headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
            timeout=30.0,
        )
        return (r.status_code, r.json().get("id") if r.status_code == 201 else None)
    except Exception as e:
        return (0, f"exc:{type(e).__name__}:{e}")


async def _delete_one(client: httpx.AsyncClient, inst_id: str) -> int:
    try:
        r = await client.delete(
            f"{LIVE_SERVER_URL}/instances/{inst_id}",
            headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
            timeout=10.0,
        )
        return r.status_code
    except Exception:
        return 0


async def _cleanup_session(client: httpx.AsyncClient, session_id: str) -> int:
    """Bulk-close instances from this test's run-unique session."""
    r = await client.delete(
        f"{LIVE_SERVER_URL}/instances",
        params={"session_id": session_id, "env_id": LIGHTWEIGHT_ENV},
        headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
        timeout=30.0,
    )
    if r.status_code == 200:
        return len(r.json().get("closed", []))
    return 0


# ---------------------------------------------------------------------------
# Round 1: high-concurrency burst — admission cap holds, counters consistent
# ---------------------------------------------------------------------------

class TestRound1Burst:
    def test_burst_200_with_cap_64(self, task_id):
        """200 concurrent POST /instances against a max_live_envs=64 server.
        Expect 64 admissions + 136 capacity 503s.
        """
        before = _admin_budget()
        session_id = _session_id("burst")

        async def run():
            async with httpx.AsyncClient() as client:
                results = await asyncio.gather(
                    *(_create_one(client, task_id, session_id) for _ in range(200))
                )
                # Cleanup what we created.
                created_ids = [iid for sc, iid in results if sc == 201 and iid]
                await asyncio.gather(
                    *(_delete_one(client, iid) for iid in created_ids),
                    return_exceptions=True,
                )
                await _cleanup_session(client, session_id)
                return results

        results = asyncio.run(run())
        codes = Counter(sc for sc, _ in results)
        # Server cap is 64; expect at most 64 admissions.
        assert codes.get(201, 0) <= 64, f"over-admitted: {codes}"
        # At least 100 should have been rejected by L2.
        assert codes.get(503, 0) >= 100, f"too few 503s: {codes}"

        after = _admin_budget()
        # Counter advanced by ~ number of rejects.
        rejects = codes.get(503, 0)
        cap_503_delta = (
            after["503_total"]["capacity"] - before["503_total"]["capacity"]
        )
        assert cap_503_delta >= rejects - 5, (  # 5-slack for concurrent tests
            f"capacity_503 didn't track rejects: delta={cap_503_delta} "
            f"vs rejects={rejects}"
        )
        # Eventually in_flight returns to 0 (or what it was before).
        time.sleep(2.0)
        final = _admin_budget()
        assert final["in_flight"] <= before["in_flight"] + 1


# ---------------------------------------------------------------------------
# Round 2: sustained churn — create/close in a loop, watch for leaks
# ---------------------------------------------------------------------------

class TestRound2Churn:
    def test_sustained_create_close_60s(self, task_id):
        """For 60 seconds, create + immediately close envs as fast as we
        can. Verify in_flight stays bounded and no monotonic creep.
        """
        deadline = time.monotonic() + 60.0
        creates = 0
        closes = 0
        peaks: list[int] = []
        session_id = _session_id("churn")

        async def churn():
            nonlocal creates, closes
            async with httpx.AsyncClient() as client:
                try:
                    while time.monotonic() < deadline:
                        sc, iid = await _create_one(client, task_id, session_id)
                        if sc == 201 and iid:
                            creates += 1
                            d = await _delete_one(client, iid)
                            if d == 200:
                                closes += 1
                        # Sample in_flight occasionally
                        if creates % 50 == 0:
                            snap = _admin_budget()
                            peaks.append(snap["in_flight"])
                finally:
                    await _cleanup_session(client, session_id)

        asyncio.run(churn())
        assert creates >= 20, f"churn too slow: {creates} creates in 60s"
        # Peak in_flight never exceeded the cap.
        assert all(p <= 64 for p in peaks), f"in_flight exceeded cap: {peaks}"
        # After the churn, in_flight should drain near 0 within idle TTL,
        # but since we deleted everything synchronously, check immediately.
        time.sleep(2.0)
        final = _admin_budget()
        assert final["in_flight"] <= 5, (
            f"in_flight leak: {final['in_flight']} after churn "
            f"(creates={creates}, closes={closes})"
        )


# ---------------------------------------------------------------------------
# Round 3: parallel-tenant — multiple sessions don't poison each other
# ---------------------------------------------------------------------------

class TestRound3Tenants:
    def test_two_tenants_share_cap(self, task_id):
        """Tenant A bursts 40, tenant B bursts 40. Cap is 64.
        FCFS: whoever's POSTs land first wins; remainder gets 503.
        Both eventually make some progress.
        """
        async def burst(sid: str, n: int) -> Counter:
            async with httpx.AsyncClient() as client:
                results = await asyncio.gather(
                    *(_create_one(client, task_id, sid) for _ in range(n))
                )
                codes = Counter(sc for sc, _ in results)
                ids = [iid for sc, iid in results if sc == 201 and iid]
                await asyncio.gather(
                    *(_delete_one(client, iid) for iid in ids),
                    return_exceptions=True,
                )
                return codes

        async def run():
            base = _session_id("tenants")
            return await asyncio.gather(burst(f"{base}-alice", 40), burst(f"{base}-bob", 40))

        a_codes, b_codes = asyncio.run(run())
        # Combined admissions ≤ cap.
        total_admitted = a_codes.get(201, 0) + b_codes.get(201, 0)
        assert total_admitted <= 64, f"combined over-admit: {a_codes} {b_codes}"
        # Both tenants made some progress.
        assert a_codes.get(201, 0) >= 1
        assert b_codes.get(201, 0) >= 1


# ---------------------------------------------------------------------------
# Round 4: production server stays unaffected throughout
# ---------------------------------------------------------------------------

class TestRound4ProdIsolation:
    def test_production_responsive_during_stress(self, task_id):
        """Hit the production server's /host_status while stressing dev.
        Production must respond promptly."""
        results: list[float] = []
        session_id = _session_id("prod-isolation")

        async def measure_prod():
            async with httpx.AsyncClient() as client:
                for _ in range(10):
                    t0 = time.monotonic()
                    try:
                        r = await client.get(
                            "http://127.0.0.1:30100/host_status",
                            timeout=5.0,
                        )
                        if r.status_code == 200:
                            results.append(time.monotonic() - t0)
                    except Exception:
                        pass
                    await asyncio.sleep(0.1)

        async def stress_dev():
            async with httpx.AsyncClient() as client:
                await asyncio.gather(
                    *(_create_one(client, task_id, session_id) for _ in range(50)),
                    return_exceptions=True,
                )

        async def run():
            await asyncio.gather(measure_prod(), stress_dev())

        asyncio.run(run())
        assert len(results) >= 8, f"prod missed responses: {results}"
        # Prod responses should all be < 1s.
        assert all(latency < 1.0 for latency in results), (
            f"prod latency degraded under dev stress: {results}"
        )

        # Clean up.
        async def cleanup():
            async with httpx.AsyncClient() as client:
                await _cleanup_session(client, session_id)
        asyncio.run(cleanup())


# ---------------------------------------------------------------------------
# Round 5: soak — long-running steady traffic, no memory creep
# ---------------------------------------------------------------------------

class TestRound5Soak:
    def test_5min_soak_under_load(self, task_id):
        """5-minute soak: maintain ~30 live envs with churn. Verify final
        state is clean and counters didn't grow unboundedly.
        """
        deadline = time.monotonic() + 300.0  # 5 minutes
        creates = 0
        closes = 0
        rejects = 0
        session_id = _session_id("soak")

        async def churn_at_steady_load():
            nonlocal creates, closes, rejects
            async with httpx.AsyncClient(timeout=30.0) as client:
                live_ids: list[str] = []
                try:
                    while time.monotonic() < deadline:
                        # If we have fewer than 30 live envs, create some.
                        while len(live_ids) < 30 and time.monotonic() < deadline:
                            sc, iid = await _create_one(client, task_id, session_id)
                            if sc == 201 and iid:
                                live_ids.append(iid)
                                creates += 1
                            elif sc == 503:
                                rejects += 1
                                break
                        # Drop one to keep churn going.
                        if live_ids:
                            victim = live_ids.pop()
                            if (await _delete_one(client, victim)) == 200:
                                closes += 1
                    # Final cleanup.
                    await asyncio.gather(
                        *(_delete_one(client, iid) for iid in live_ids),
                        return_exceptions=True,
                    )
                finally:
                    await _cleanup_session(client, session_id)

        asyncio.run(churn_at_steady_load())
        # Sanity: did we actually do work?
        assert creates >= 50, f"soak too slow: {creates} creates"
        # in_flight should drain to near 0 (or back to baseline).
        time.sleep(3.0)
        final = _admin_budget()
        assert final["in_flight"] <= 5, (
            f"soak left {final['in_flight']} in_flight envs "
            f"(creates={creates}, closes={closes}, rejects={rejects})"
        )
