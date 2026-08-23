"""Tests for shared-backend isolation — the conflict gate.

Three layers, all deterministic, no Docker / no live server:
  1. ``ConflictKeyTable`` unit tests — the RW-lock semantics + release.
  2. ``_close_quietly`` unit tests — writer restores + key released on close.
  3. Gate integration via FastAPI ``TestClient`` — POST /instances admit /
     503 matrix, Retry-After, keyless no-op, failure-path key release.

Run::
    uv run pytest tests/gym/remote/test_conflict.py -v
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
from unittest.mock import patch

from fastapi.testclient import TestClient

import lite.gym as gym
from lite.core.metadata import LiteCUAMetadata
from lite.gym.base import LiteBaseEnv
from lite.gym.remote.conflict import ConflictKeyTable
from lite.gym.remote.server import State, _close_quietly, _EnvSession, make_app
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult

# ===========================================================================
# 1. ConflictKeyTable — RW-lock semantics
# ===========================================================================

class TestConflictKeyTable:
    def test_writer_is_exclusive(self):
        t = ConflictKeyTable()
        assert t.try_acquire("w1", ("k",), mutating=True)
        # another writer, a reader → both blocked
        assert not t.try_acquire("w2", ("k",), mutating=True)
        assert not t.try_acquire("r1", ("k",), mutating=False)

    def test_readers_share(self):
        t = ConflictKeyTable()
        assert t.try_acquire("r1", ("k",), mutating=False)
        assert t.try_acquire("r2", ("k",), mutating=False)
        # writer blocked while readers hold it
        assert not t.try_acquire("w1", ("k",), mutating=True)

    def test_release_frees_for_writer(self):
        t = ConflictKeyTable()
        assert t.try_acquire("w1", ("k",), mutating=True)
        t.release("w1")
        assert t.try_acquire("w2", ("k",), mutating=True)

    def test_writer_waits_for_all_readers(self):
        t = ConflictKeyTable()
        t.try_acquire("r1", ("k",), mutating=False)
        t.try_acquire("r2", ("k",), mutating=False)
        assert not t.try_acquire("w1", ("k",), mutating=True)
        t.release("r1")
        assert not t.try_acquire("w1", ("k",), mutating=True)  # r2 still holds
        t.release("r2")
        assert t.try_acquire("w1", ("k",), mutating=True)  # now idle

    def test_release_idempotent(self):
        t = ConflictKeyTable()
        t.try_acquire("w1", ("k",), mutating=True)
        t.release("w1")
        t.release("w1")  # no error, no effect
        t.release("never-held")  # unknown id → no error
        assert t.try_acquire("w2", ("k",), mutating=True)

    def test_empty_keys_no_op(self):
        t = ConflictKeyTable()
        assert t.try_acquire("x", (), mutating=True)
        assert t.holder("x") is None          # nothing recorded
        assert t.snapshot() == {}
        # never blocks anyone
        assert t.try_acquire("y", (), mutating=True)

    def test_multikey_all_or_nothing(self):
        t = ConflictKeyTable()
        assert t.try_acquire("w1", ("a", "b"), mutating=True)
        # a writer wanting (b, c) conflicts on b → acquires NOTHING (c free)
        assert not t.try_acquire("w2", ("b", "c"), mutating=True)
        assert "c" not in t.snapshot()        # c was not partially taken
        # after w1 releases, (b, c) succeeds
        t.release("w1")
        assert t.try_acquire("w2", ("b", "c"), mutating=True)

    def test_gc_empty_keys(self):
        t = ConflictKeyTable()
        t.try_acquire("w1", ("k",), mutating=True)
        assert "k" in t.snapshot()
        t.release("w1")
        assert t.snapshot() == {}             # key GC'd once unheld

    def test_holder_reports_keys_and_mutating(self):
        t = ConflictKeyTable()
        t.try_acquire("w1", ("a", "b"), mutating=True)
        assert t.holder("w1") == (("a", "b"), True)
        t.try_acquire("r1", ("c",), mutating=False)
        assert t.holder("r1") == (("c",), False)
        assert t.holder("absent") is None

    def test_different_keys_independent(self):
        t = ConflictKeyTable()
        assert t.try_acquire("w1", ("a",), mutating=True)
        assert t.try_acquire("w2", ("b",), mutating=True)  # different key → ok


# ===========================================================================
# 2. _close_quietly — reset-in-close (writer restores, then key released)
# ===========================================================================

class _NoopEnv(LiteBaseEnv):
    def __init__(self, env_id: str = "test.shared"):
        self._env_id = env_id
        self.closed = False

    def _runtime_metadata(self) -> LiteCUAMetadata:
        return LiteCUAMetadata(dims=("browser", "use"))

    async def reset(self) -> LiteEnvObservation:
        return LiteEnvObservation(image=None, text="ok")

    async def step(self, actions: list) -> LiteEnvStepResult:
        return LiteEnvStepResult()

    async def close(self) -> None:
        self.closed = True


def _state() -> State:
    from lite.gym.remote.admission import AdmissionConfig, AdmissionGate
    return State(
        admission=AdmissionGate(AdmissionConfig(
            max_live_envs=100, emergency_ram_pct=99.9,
            emergency_ram_free_min_bytes=1, emergency_load_per_cpu=9999.0,
            docker_create_concurrency=8,
        )),
        idle_ttl_sec=3600.0,
    )


def _session(env_id: str = "test.shared") -> _EnvSession:
    return _EnvSession(
        env=_NoopEnv(env_id), env_key=f"{env_id}@t", env_id=env_id,
        session_id="s", token_hash="th", created_at=0.0, last_active_at=0.0,
    )


class TestCloseQuietly:
    def test_reader_close_releases_no_restore(self):
        state = _state()
        state.conflict.try_acquire("id1", ("k",), mutating=False)
        s = _session()
        with patch("lite.gym.remote.server.restore_backend_dispatch") as rb:
            asyncio.run(_close_quietly(state, s, "id1"))
        assert s.env.closed is True
        assert state.conflict.holder("id1") is None       # released
        rb.assert_not_called()                            # readers don't restore

    def test_writer_close_restores_then_releases(self):
        state = _state()
        state.conflict.try_acquire("id1", ("webarena",), mutating=True)
        s = _session("browsergym.webarena")
        calls = []

        async def fake_restore(env_id, key):
            calls.append((env_id, key))

        with patch("lite.gym.remote.server.restore_backend_dispatch", fake_restore):
            asyncio.run(_close_quietly(state, s, "id1"))
        assert calls == [("browsergym.webarena", "webarena")]  # restored before release
        assert state.conflict.holder("id1") is None            # released

    def test_keyless_close_is_noop_on_conflict(self):
        state = _state()
        s = _session()
        with patch("lite.gym.remote.server.restore_backend_dispatch") as rb:
            asyncio.run(_close_quietly(state, s, "no-keys"))
        assert s.env.closed is True
        rb.assert_not_called()

    def test_writer_restore_failure_keeps_key_held(self):
        # A2: a failed restore must NOT release the key — the next
        # claimant must never see a dirty stack. The key stays held and a
        # failure counter is bumped for operator visibility.
        state = _state()
        state.conflict.try_acquire("id1", ("webarena",), mutating=True)
        s = _session("browsergym.webarena")

        async def boom_restore(env_id, key):
            raise RuntimeError("reset service unreachable")

        with patch("lite.gym.remote.server.restore_backend_dispatch", boom_restore):
            asyncio.run(_close_quietly(state, s, "id1"))
        assert s.env.closed is True
        assert state.conflict.holder("id1") is not None    # key KEPT (poisoned)
        assert state.restore_fail_total == 1
        # the key is still exclusively held → a new writer is rejected
        assert state.conflict.try_acquire("id2", ("webarena",), mutating=True) is False


# ===========================================================================
# 3. Gate integration — POST /instances admit / 503 / release
# ===========================================================================

def _client_with_meta(meta_by_env_id: dict[str, LiteCUAMetadata], *, make_raises=False):
    """TestClient with gym.make stubbed + task_metadata stubbed so the gate
    sees controlled conflict_keys/mutating per env_id. Returns (client,
    state, patchers)."""
    state = _state()
    app = make_app(state, token=None)

    def fake_make(env_key, **kw):
        if make_raises:
            raise RuntimeError("boom")
        env_id = env_key.split("@", 1)[0]
        return _NoopEnv(env_id)

    def fake_meta(env_id, task_id):
        return meta_by_env_id.get(env_id, LiteCUAMetadata(dims=("browser", "use")))

    patchers = [
        patch("lite.gym.remote.server.gym.make", fake_make),
        patch.object(gym.registry, "task_metadata", fake_meta),
    ]
    for p in patchers:
        p.start()
    return TestClient(app), state, patchers


def _create(client, env_key, token="alice", session_id="s1"):
    return client.post(
        "/instances",
        json={"env_key": env_key, "env_kwargs": {}, "session_id": session_id},
        headers={"Authorization": f"Bearer {token}"},
    )


WRITER = LiteCUAMetadata(
    dims=("browser", "use"),
    others={"conflict_keys": ("webarena",), "mutating": True},
)
READER = LiteCUAMetadata(
    dims=("browser", "use"),
    others={"conflict_keys": ("webarena",), "mutating": False},
)


class TestGate:
    def test_writer_then_writer_same_key_503(self):
        client, state, ps = _client_with_meta({"browsergym.webarena": WRITER})
        try:
            r1 = _create(client, "browsergym.webarena@0")
            assert r1.status_code == 201, r1.text
            r2 = _create(client, "browsergym.webarena@1")
            assert r2.status_code == 503
            assert "Retry-After" in r2.headers
            assert int(r2.headers["Retry-After"]) >= 1
            # the conflict 503 is counted (observability — distinct from the
            # admission layers; exported as cua_lite_conflict_503_total)
            assert state.conflict_503_total == 1
            m = client.get("/metrics").text
            assert "cua_lite_conflict_503_total 1" in m
        finally:
            for p in ps:
                p.stop()

    def test_writer_blocks_reader(self):
        client, state, ps = _client_with_meta(
            {"browsergym.webarena": WRITER, "wa.reader": READER})
        try:
            assert _create(client, "browsergym.webarena@0").status_code == 201
            # a reader on the SAME key is blocked by the live writer
            r = _create(client, "wa.reader@0")
            # wa.reader maps to READER (same key "webarena") → blocked
            assert r.status_code == 503
        finally:
            for p in ps:
                p.stop()

    def test_readers_share_same_key(self):
        client, state, ps = _client_with_meta({"browsergym.webarena": READER})
        try:
            assert _create(client, "browsergym.webarena@0").status_code == 201
            assert _create(client, "browsergym.webarena@1").status_code == 201
            assert _create(client, "browsergym.webarena@2").status_code == 201
            snap = state.conflict.snapshot()
            assert snap["webarena"]["readers"] == 3
            assert snap["webarena"]["writer"] is None
        finally:
            for p in ps:
                p.stop()

    def test_keyless_env_never_gated(self):
        client, state, ps = _client_with_meta({})  # default meta → conflict_keys=()
        try:
            for i in range(5):
                assert _create(client, f"miniwob@{i}").status_code == 201
            assert state.conflict.snapshot() == {}   # gate never engaged
        finally:
            for p in ps:
                p.stop()

    def test_cold_spawn_failure_releases_key(self):
        # gym.make raises → 4xx/5xx, but the gate-acquired key must be
        # released in the finally so the backend isn't wedged.
        client, state, ps = _client_with_meta(
            {"browsergym.webarena": WRITER}, make_raises=True)
        try:
            r = _create(client, "browsergym.webarena@0")
            assert r.status_code >= 400
            assert state.conflict.snapshot() == {}   # key released on failure
            # and a subsequent writer can now claim
        finally:
            for p in ps:
                p.stop()

    def test_admission_503_releases_preacquired_conflict_key(self):
        # The conflict key is acquired before L2 admission. If admission rejects,
        # no env is inserted and /instances has no holder to close, so the key
        # must be released on the rejection path itself.
        client, state, ps = _client_with_meta({"browsergym.webarena": WRITER})
        try:
            from lite.gym.remote.admission import AdmissionGate

            state.admission = AdmissionGate(
                replace(state.admission.config, max_live_envs=0)
            )

            r = _create(client, "browsergym.webarena@0")

            assert r.status_code == 503
            assert state.conflict.snapshot() == {}
            assert state.admission.snapshot()["in_flight"] == 0
        finally:
            for p in ps:
                p.stop()

    def test_503_then_success_after_release(self):
        client, state, ps = _client_with_meta({"browsergym.webarena": WRITER})
        try:
            r1 = _create(client, "browsergym.webarena@0")
            assert r1.status_code == 201
            inst_id = r1.json()["id"]
            assert _create(client, "browsergym.webarena@1").status_code == 503
            # close the holder; drain the detached _close_quietly so the key
            # is released (reset-in-close happens in the background task).
            d = client.delete(f"/instances/{inst_id}",
                              headers={"Authorization": "Bearer alice"})
            assert d.status_code == 200
            # drain pending background closes on the app loop
            _drain_pending(client, state)
            assert _create(client, "browsergym.webarena@2").status_code == 201
        finally:
            for p in ps:
                p.stop()


def _drain_pending(client, state, timeout_s: float = 5.0):
    """Block until the server's detached close tasks finish (the key is
    released at the end of _close_quietly). Polls the conflict snapshot."""
    import time
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not state.conflict.snapshot():
            return
        time.sleep(0.05)


def test_others_key_constants_are_the_wire_literals():
    """section D1/section 6.E: the constants MUST equal the wire literals byte-for-byte —
    /tasks metadata and --filter "m.others.get('mutating')" read these."""
    from lite.gym.remote.conflict import OTHERS_CONFLICT_KEYS, OTHERS_MUTATING
    assert OTHERS_CONFLICT_KEYS == "conflict_keys"
    assert OTHERS_MUTATING == "mutating"
