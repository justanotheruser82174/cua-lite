"""Edge cases, multi-token/multi-session isolation, observation
validation, and weird concurrency patterns against the dev
env-server.

These tests do not require docker; they use the lightweight
``screenspot_pro`` grounding env which returns deterministic
observations (text instruction + base64 PNG screenshot).

Run::

    CUA_LITE_LIVE_ENV_SERVER_URL=http://127.0.0.1:30200 \
    CUA_LITE_LIVE_ENV_SERVER_TOKEN=... \
    CUA_LITE_LIVE_ADMIN_TOKEN=... \
      uv run pytest -m live tests/gym/remote/test_live_edge_cases.py -xvs
"""
from __future__ import annotations

import asyncio
import os
import socket
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import httpx
import pytest
from gym.remote.conftest import unique_live_scope, unique_live_token

from lite.core.tools import make_tool_call
from lite.gym.remote.frame import decode_reset_observation, decode_step_result

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
def env_task() -> tuple[str, str]:
    r = httpx.get(
        f"{LIVE_SERVER_URL}/envs", headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
        timeout=5.0,
    )
    available = r.json()
    if "screenspot_pro" not in available:
        pytest.skip("screenspot_pro not registered")
    r = httpx.get(
        f"{LIVE_SERVER_URL}/envs/screenspot_pro/tasks",
        headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"}, timeout=5.0,
    )
    task_ids = next(iter(r.json().get("splits", {}).values()), [])
    if not task_ids:
        pytest.skip("no tasks")
    return "screenspot_pro", task_ids[0]


# Helpers
def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _create(client: httpx.Client, env_key: str, *,
            token: str = LIVE_SERVER_TOKEN, session_id: str | None = None,
            env_kwargs: dict | None = None) -> httpx.Response:
    if session_id is None:
        session_id = unique_live_scope("edge_create")
    return client.post(
        f"{LIVE_SERVER_URL}/instances",
        json={"env_key": env_key, "env_kwargs": env_kwargs or {},
              "session_id": session_id},
        headers=_bearer(token), timeout=30.0,
    )


def _reset(client: httpx.Client, inst_id: str,
           token: str = LIVE_SERVER_TOKEN) -> httpx.Response:
    return client.post(
        f"{LIVE_SERVER_URL}/instances/{inst_id}/reset",
        headers=_bearer(token), timeout=30.0,
    )


def _step(client: httpx.Client, inst_id: str, actions: list[dict],
          token: str = LIVE_SERVER_TOKEN) -> httpx.Response:
    return client.post(
        f"{LIVE_SERVER_URL}/instances/{inst_id}/step",
        json={"actions": actions},
        headers=_bearer(token), timeout=30.0,
    )


def _delete(client: httpx.Client, inst_id: str,
            token: str = LIVE_SERVER_TOKEN) -> httpx.Response:
    return client.delete(
        f"{LIVE_SERVER_URL}/instances/{inst_id}",
        headers=_bearer(token), timeout=10.0,
    )


def _decode_reset(r: httpx.Response):
    """Decode a /reset binary frame into a LiteEnvObservation."""
    return decode_reset_observation(r.content)


def _decode_step(r: httpx.Response):
    """Decode a /step binary frame into a LiteEnvStepResult."""
    return decode_step_result(r.content)


# ---------------------------------------------------------------------------
# Observation validation — reset returns a real screenshot
# ---------------------------------------------------------------------------

class TestObservationShape:
    """The grounding env returns a deterministic observation. Validate
    the reset response env transport shape AND that the screenshot bytes are
    a valid PNG."""

    def test_reset_returns_text_and_screenshot(self, env_task):
        env_id, task_id = env_task
        with httpx.Client() as client:
            r = _create(
                client,
                f"{env_id}@{task_id}",
                session_id=unique_live_scope("edge_observation"),
            )
            assert r.status_code == 201
            inst_id = r.json()["id"]
            try:
                r = _reset(client, inst_id)
                assert r.status_code == 200, r.text
                obs = _decode_reset(r)
                # Observation shape
                assert isinstance(obs.text, str)
                assert len(obs.text) > 0, "empty instruction"
                assert obs.image
                # Sanity check the PNG header (0x89PNG).
                raw = obs.image
                assert len(raw) > 1000, "screenshot suspiciously small"
                assert raw[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
            finally:
                _delete(client, inst_id)

    def test_reset_is_deterministic_for_same_task(self, env_task):
        """Grounding env: same task_id should produce same instruction +
        same screenshot byte-for-byte."""
        env_id, task_id = env_task
        with httpx.Client() as client:
            session_id = unique_live_scope("edge_deterministic")
            inst_a = _create(client, f"{env_id}@{task_id}",
                             session_id=session_id).json()["id"]
            inst_b = _create(client, f"{env_id}@{task_id}",
                             session_id=session_id).json()["id"]
            try:
                ra = _decode_reset(_reset(client, inst_a))
                rb = _decode_reset(_reset(client, inst_b))
                assert ra.text == rb.text
                # Screenshots should be byte-identical for grounding env.
                assert (ra.images[-1] if ra.images else None) == (
                    rb.images[-1] if rb.images else None
                )
            finally:
                _delete(client, inst_a)
                _delete(client, inst_b)

    def test_consecutive_resets_idempotent(self, env_task):
        """Two resets in a row should yield the same observation."""
        env_id, task_id = env_task
        with httpx.Client() as client:
            inst = _create(client, f"{env_id}@{task_id}",
                           session_id=unique_live_scope("edge_consecutive")).json()["id"]
            try:
                r1 = _decode_reset(_reset(client, inst))
                r2 = _decode_reset(_reset(client, inst))
                assert r1.text == r2.text
            finally:
                _delete(client, inst)


# ---------------------------------------------------------------------------
# step on grounding env
# ---------------------------------------------------------------------------

class TestStepActions:
    """Grounding env accepts a single 'point' action with x,y coords.
    Verify it scores correctly + rejects malformed input."""

    def test_step_with_valid_point_action(self, env_task):
        env_id, task_id = env_task
        with httpx.Client() as client:
            inst = _create(client, f"{env_id}@{task_id}",
                           session_id=unique_live_scope("edge_step_valid")).json()["id"]
            try:
                _reset(client, inst)
                # Submit a point click action — grounding scores against
                # the ground-truth bbox in metadata.
                r = _step(client, inst, [
                    make_tool_call("point", {"coordinate": [500, 500]})
                ])
                assert r.status_code == 200, r.text
                result = _decode_step(r)
                # Grounding env terminates after one step.
                assert result.terminated is True
                # Reward should be 0 or 1 (point in/out of bbox).
                assert result.reward in (0, 1, 0.0, 1.0)
            finally:
                _delete(client, inst)

    def test_step_with_empty_actions(self, env_task):
        env_id, task_id = env_task
        with httpx.Client() as client:
            inst = _create(client, f"{env_id}@{task_id}",
                           session_id=unique_live_scope("edge_step_empty")).json()["id"]
            try:
                _reset(client, inst)
                r = _step(client, inst, [])
                # Empty actions list — env may return 200 with no-op or
                # 400 / 500. Either is acceptable; we just want no crash
                # at the wire layer.
                assert r.status_code in (200, 400, 500), r.text
            finally:
                _delete(client, inst)

    def test_step_before_reset_handled(self, env_task):
        env_id, task_id = env_task
        with httpx.Client() as client:
            inst = _create(client, f"{env_id}@{task_id}",
                           session_id=unique_live_scope("edge_step_before_reset")).json()["id"]
            try:
                # NO reset, straight to step.
                r = _step(client, inst, [
                    make_tool_call("point", {"coordinate": [0, 0]})
                ])
                # Should either work or fail cleanly with a status code.
                # 409 = the explicit step-before-reset guard.
                assert r.status_code in (200, 400, 409, 500), r.text
            finally:
                _delete(client, inst)


# ---------------------------------------------------------------------------
# Lifecycle edge cases
# ---------------------------------------------------------------------------

class TestLifecycleEdges:
    def test_get_unknown_instance_404(self):
        r = httpx.get(
            f"{LIVE_SERVER_URL}/instances/0123456789abcdef0123456789abcdef",
            headers=_bearer(LIVE_SERVER_TOKEN), timeout=5.0,
        )
        assert r.status_code == 404

    def test_reset_unknown_instance_404(self):
        r = httpx.post(
            f"{LIVE_SERVER_URL}/instances/deadbeef0011223344556677889900aa/reset",
            headers=_bearer(LIVE_SERVER_TOKEN), timeout=5.0,
        )
        assert r.status_code == 404

    def test_step_unknown_instance_404(self):
        r = httpx.post(
            f"{LIVE_SERVER_URL}/instances/cafebabe11223344556677889900aabb/step",
            json={"actions": []},
            headers=_bearer(LIVE_SERVER_TOKEN), timeout=5.0,
        )
        assert r.status_code == 404

    def test_delete_unknown_instance(self):
        """Server returns 200 (idempotent) or 404 — both acceptable."""
        r = httpx.delete(
            f"{LIVE_SERVER_URL}/instances/feed1234aaaabbbbccccdddd00001111",
            headers=_bearer(LIVE_SERVER_TOKEN), timeout=5.0,
        )
        assert r.status_code in (200, 404), r.text

    def test_double_delete_handled(self, env_task):
        """Second DELETE returns either 200 (idempotent) or 404 (gone)
        depending on server/version idempotency behavior. No 5xx."""
        env_id, task_id = env_task
        with httpx.Client() as client:
            inst = _create(client, f"{env_id}@{task_id}",
                           session_id=unique_live_scope("edge_double_delete")).json()["id"]
            r1 = _delete(client, inst)
            r2 = _delete(client, inst)
            assert r1.status_code == 200
            assert r2.status_code in (200, 404), r2.text

    def test_step_on_deleted_instance(self, env_task):
        env_id, task_id = env_task
        with httpx.Client() as client:
            inst = _create(client, f"{env_id}@{task_id}",
                           session_id=unique_live_scope("edge_post_delete")).json()["id"]
            _delete(client, inst)
            r = _step(client, inst, [])
            assert r.status_code == 404

    def test_create_with_bad_env_key_400(self):
        r = httpx.post(
            f"{LIVE_SERVER_URL}/instances",
            json={"env_key": "no_at_sign", "env_kwargs": {},
                  "session_id": unique_live_scope("edge_bad_env_key")},
            headers=_bearer(LIVE_SERVER_TOKEN), timeout=10.0,
        )
        # split_key requires a non-empty name → likely raises in handler.
        assert r.status_code in (400, 422), r.text

    def test_create_with_unknown_env_id_4xx(self):
        r = httpx.post(
            f"{LIVE_SERVER_URL}/instances",
            json={"env_key": "no_such_env@whatever", "env_kwargs": {},
                  "session_id": unique_live_scope("edge_unknown_env")},
            headers=_bearer(LIVE_SERVER_TOKEN), timeout=10.0,
        )
        assert r.status_code in (400, 404, 501), r.text


# ---------------------------------------------------------------------------
# Multi-token isolation
# ---------------------------------------------------------------------------

class TestMultiTokenIsolation:
    """Two distinct tokens must not see each other's instances.
    Server is in passthrough mode for data endpoints (any bearer
    accepted, identity = sha256(token)[:6])."""

    def test_alice_cannot_delete_bobs_env(self, env_task):
        env_id, task_id = env_task
        alice_token = unique_live_token("edge_alice")
        bob_token = unique_live_token("edge_bob")
        session_id = unique_live_scope("edge_cross_token")
        with httpx.Client() as client:
            # Bob creates an env.
            r_b = _create(client, f"{env_id}@{task_id}",
                          token=bob_token, session_id=session_id)
            assert r_b.status_code == 201
            inst_b = r_b.json()["id"]
            try:
                # Alice tries to delete it — server.py uses
                # _require_own_env which 403s on cross-token access.
                r_a = _delete(client, inst_b, token=alice_token)
                assert r_a.status_code in (403, 404), r_a.text
                # Bob can still delete his own.
                r_b2 = _delete(client, inst_b, token=bob_token)
                assert r_b2.status_code == 200
            except Exception:
                _delete(client, inst_b, token=bob_token)
                raise

    def test_list_instances_scoped_to_caller_token(self, env_task):
        env_id, task_id = env_task
        alice_token = unique_live_token("edge_list_alice")
        bob_token = unique_live_token("edge_list_bob")
        session_id = unique_live_scope("edge_list_instances")
        with httpx.Client() as client:
            # Alice creates 2, Bob creates 3.
            a_ids = []
            b_ids = []
            for _ in range(2):
                r = _create(client, f"{env_id}@{task_id}",
                            token=alice_token, session_id=session_id)
                if r.status_code == 201:
                    a_ids.append(r.json()["id"])
            for _ in range(3):
                r = _create(client, f"{env_id}@{task_id}",
                            token=bob_token, session_id=session_id)
                if r.status_code == 201:
                    b_ids.append(r.json()["id"])
            try:
                # Alice's list shows only her envs.
                alice_list = client.get(
                    f"{LIVE_SERVER_URL}/instances",
                    headers=_bearer(alice_token), timeout=5.0,
                ).json()["instances"]
                bob_list = client.get(
                    f"{LIVE_SERVER_URL}/instances",
                    headers=_bearer(bob_token), timeout=5.0,
                ).json()["instances"]
                alice_ids = {i["id"] for i in alice_list}
                bob_ids = {i["id"] for i in bob_list}
                assert all(iid in alice_ids for iid in a_ids)
                assert all(iid not in alice_ids for iid in b_ids)
                assert all(iid in bob_ids for iid in b_ids)
                assert all(iid not in bob_ids for iid in a_ids)
            finally:
                for iid in a_ids:
                    _delete(client, iid, token=alice_token)
                for iid in b_ids:
                    _delete(client, iid, token=bob_token)

    def test_admin_sees_all_tokens(self):
        r = httpx.get(
            f"{LIVE_SERVER_URL}/admin/tokens",
            headers=_bearer(LIVE_ADMIN_TOKEN), timeout=5.0,
        )
        body = r.json()
        # The exact token set depends on previous live tests and server mode.
        tokens_seen = [t.get("token") or "anonymous" for t in body["tokens"]]
        # Just verify the shape — content depends on test order.
        assert isinstance(tokens_seen, list)
        for row in body["tokens"]:
            assert "token_hash" in row
            assert "instances_active" in row


# ---------------------------------------------------------------------------
# Multi-session within one token
# ---------------------------------------------------------------------------

class TestMultiSession:
    def test_distinct_sessions_under_same_token(self, env_task):
        env_id, task_id = env_task
        session_x = unique_live_scope("edge_session_x")
        session_y = unique_live_scope("edge_session_y")
        with httpx.Client() as client:
            inst_x = _create(client, f"{env_id}@{task_id}",
                             session_id=session_x).json()["id"]
            inst_y = _create(client, f"{env_id}@{task_id}",
                             session_id=session_y).json()["id"]
            try:
                # List filtered by session_id.
                rx = client.get(
                    f"{LIVE_SERVER_URL}/instances",
                    params={"session_id": session_x},
                    headers=_bearer(LIVE_SERVER_TOKEN), timeout=5.0,
                ).json()["instances"]
                ry = client.get(
                    f"{LIVE_SERVER_URL}/instances",
                    params={"session_id": session_y},
                    headers=_bearer(LIVE_SERVER_TOKEN), timeout=5.0,
                ).json()["instances"]
                assert any(i["id"] == inst_x for i in rx)
                assert not any(i["id"] == inst_x for i in ry)
                assert any(i["id"] == inst_y for i in ry)
                assert not any(i["id"] == inst_y for i in rx)
            finally:
                _delete(client, inst_x)
                _delete(client, inst_y)


# ---------------------------------------------------------------------------
# Weird concurrency
# ---------------------------------------------------------------------------

class TestConcurrencyEdges:
    def test_concurrent_delete_same_instance(self, env_task):
        """Concurrent DELETEs on the same id must not crash. Some get
        200 (the winner) and the rest 200/404 depending on whether the
        server's lock interleaves; no 500s."""
        env_id, task_id = env_task
        with httpx.Client() as client:
            inst = _create(client, f"{env_id}@{task_id}",
                           session_id=unique_live_scope("edge_concurrent_delete")).json()["id"]
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(_delete, client, inst)
                           for _ in range(4)]
                results = [f.result() for f in futures]
            codes = Counter(r.status_code for r in results)
            assert codes.get(500, 0) == 0, codes
            # At least one DELETE succeeded.
            assert codes.get(200, 0) >= 1
            # All responses are 200 or 404 (no 5xx).
            assert sum(codes.values()) == codes.get(200, 0) + codes.get(404, 0)

    def test_concurrent_resets_on_same_instance(self, env_task):
        """Two threads /reset the same instance. Should both succeed
        (resets are idempotent for grounding env)."""
        env_id, task_id = env_task
        with httpx.Client() as client:
            inst = _create(client, f"{env_id}@{task_id}",
                           session_id=unique_live_scope("edge_concurrent_reset")).json()["id"]
            try:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = [pool.submit(_reset, client, inst)
                               for _ in range(4)]
                    results = [f.result() for f in futures]
                for r in results:
                    assert r.status_code in (200, 500), r.text
                # At least one should succeed.
                assert any(r.status_code == 200 for r in results)
            finally:
                _delete(client, inst)

    def test_concurrent_create_under_cap(self, env_task):
        """50 concurrent creates with cap=64. All should succeed. Counters
        consistent."""
        env_id, task_id = env_task
        before = httpx.get(
            f"{LIVE_SERVER_URL}/admin/budget",
            headers=_bearer(LIVE_ADMIN_TOKEN), timeout=5.0,
        ).json()
        created: list[str] = []
        session_id = unique_live_scope("edge_concurrent_create")

        async def burst():
            async with httpx.AsyncClient() as ac:
                async def one():
                    r = await ac.post(
                        f"{LIVE_SERVER_URL}/instances",
                        json={"env_key": f"{env_id}@{task_id}",
                              "env_kwargs": {}, "session_id": session_id},
                        headers=_bearer(LIVE_SERVER_TOKEN), timeout=30.0,
                    )
                    return r
                rs = await asyncio.gather(*(one() for _ in range(50)))
                return rs

        results = asyncio.run(burst())
        for r in results:
            if r.status_code == 201:
                created.append(r.json()["id"])
        try:
            # All 50 should have made it in (cap is 64 - residual).
            slots_free = before["max_live_envs"] - before["in_flight"]
            assert len(created) == min(50, slots_free), (
                f"expected min(50, {slots_free}) admits, got {len(created)}"
            )
            # /admin/budget shows the increment.
            mid = httpx.get(
                f"{LIVE_SERVER_URL}/admin/budget",
                headers=_bearer(LIVE_ADMIN_TOKEN), timeout=5.0,
            ).json()
            assert mid["in_flight"] >= before["in_flight"] + len(created)
        finally:
            with httpx.Client() as client:
                for iid in created:
                    _delete(client, iid)


# ---------------------------------------------------------------------------
# Token registry growth (auth burst)
# ---------------------------------------------------------------------------

class TestTokenRegistry:
    def test_many_distinct_tokens_registered(self):
        """100 distinct one-shot tokens. Registry grows by 100 (or more)
        but server stays responsive."""
        # Take a baseline count.
        r = httpx.get(
            f"{LIVE_SERVER_URL}/admin/tokens",
            headers=_bearer(LIVE_ADMIN_TOKEN), timeout=5.0,
        )
        before = len(r.json()["tokens"])

        async def hit_with_token(i: int):
            async with httpx.AsyncClient() as ac:
                # Cheap probe — /envs is auth-required.
                return await ac.get(
                    f"{LIVE_SERVER_URL}/envs",
                    headers=_bearer(unique_live_token(f"edge_registry_{i:04d}")),
                    timeout=5.0,
                )

        async def burst():
            return await asyncio.gather(*(hit_with_token(i)
                                          for i in range(100)))

        results = asyncio.run(burst())
        assert all(r.status_code == 200 for r in results)
        r2 = httpx.get(
            f"{LIVE_SERVER_URL}/admin/tokens",
            headers=_bearer(LIVE_ADMIN_TOKEN), timeout=5.0,
        )
        after = len(r2.json()["tokens"])
        assert after >= before + 100, f"registry didn't grow: {before} -> {after}"


# ---------------------------------------------------------------------------
# in_flight invariant under load
# ---------------------------------------------------------------------------

class TestInFlightInvariant:
    def test_in_flight_drains_after_burst(self, env_task):
        """After a create+delete burst, in_flight returns to baseline."""
        env_id, task_id = env_task
        before = httpx.get(
            f"{LIVE_SERVER_URL}/admin/budget",
            headers=_bearer(LIVE_ADMIN_TOKEN), timeout=5.0,
        ).json()["in_flight"]

        with httpx.Client() as client:
            ids: list[str] = []
            session_id = unique_live_scope("edge_drain")
            for _ in range(20):
                r = _create(client, f"{env_id}@{task_id}",
                            session_id=session_id)
                if r.status_code == 201:
                    ids.append(r.json()["id"])
            for iid in ids:
                _delete(client, iid)

        # Give the server a moment to settle.
        time.sleep(1.0)
        after = httpx.get(
            f"{LIVE_SERVER_URL}/admin/budget",
            headers=_bearer(LIVE_ADMIN_TOKEN), timeout=5.0,
        ).json()["in_flight"]
        assert after <= before + 2, (
            f"in_flight leaked: {before} -> {after}"
        )
