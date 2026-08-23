"""Dual-mode metadata parity (same-source invariant, metadata contract): the env-server
serves EXACTLY the registered copy for every enumerable env, and a real
server-side create returns the same metadata as a direct ``gym.make``.

In-process ``TestClient`` against ``make_app`` — no live server, no Docker.
The create leg uses captcha (probe-free construction). Together with the
direct-mode parity tests (tests/gym/metadata/test_metadata_invariant.py: T1 +
direct-construction rows), this closes the loop: registered == direct-live
== server-served, for every env on every host.

Size guard: envs whose full catalog would blow up the (unpaginated) /tasks
response (webgym ~293k rows) are skipped — the serving code path
(``task_metadata`` → ``to_dict`` per task) is identical for all envs, so the
covered envs pin it.

Run: uv run pytest tests/gym/remote/test_metadata_dual_mode.py -v
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from gym.remote.conftest import bearer_header as _h
from gym.remote.conftest import make_test_admission

import lite.gym as gym
from lite.core.metadata import LiteCUAMetadata, LiteGenericMetadata
from lite.core.tools import make_tool_schema
from lite.gym.base import LiteBaseEnv
from lite.gym.errors import EnvDepsMissingError
from lite.gym.registry import (
    _env_make_kwargs,
    _imported,
    _specs,
    _splits,
    register,
    registry,
)
from lite.gym.remote.server import State, make_app
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult

_TOKEN = "dual-mode-test"
_MAX_TASKS_SERVED = 3000  # response-size guard (see module docstring)


@pytest.fixture(autouse=True)
def _direct_registry(monkeypatch):
    # The comparison baseline is the DIRECT registry — a set env-server URL
    # would reroute local registration to a remote server.
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_URL", raising=False)


@pytest.fixture()
def client():
    state = State(
        admission=make_test_admission(max_live_envs=4),
        idle_ttl_sec=3600.0,
    )
    app = make_app(state, token=_TOKEN)
    with TestClient(app) as c:
        yield c


def _jsonable(d: dict) -> dict:
    """JSON-normalize (tuples → lists) to match the wire format."""
    return json.loads(json.dumps(d))


def test_served_task_metadata_equals_registered(client):
    """GET /envs/{env_id}/tasks serves ``to_dict`` of the registered copy for
    EVERY task of EVERY enumerable env — server mode can never drift from the
    same-source registry. Non-vacuous: fails outright if the sweep covered
    almost nothing (the silent-skip failure mode)."""
    checked_envs = 0
    for env_id in sorted(registry.env_ids()):
        try:
            splits = registry.task_ids(env_id)
        except Exception:  # noqa: BLE001 — dep/data-gated env
            continue
        ids = [t for tids in splits.values() for t in tids]
        if not ids or len(ids) > _MAX_TASKS_SERVED:
            continue
        r = client.get(f"/envs/{env_id}/tasks", headers=_h(_TOKEN))
        assert r.status_code == 200, (env_id, r.text)
        served = r.json()["metadata"]
        assert set(served) == set(ids), env_id
        for tid in ids:
            direct = _jsonable(registry.task_metadata(env_id, tid).to_dict())
            assert served[tid] == direct, (env_id, tid)
        checked_envs += 1
    assert checked_envs >= 3, (
        f"dual-mode sweep covered only {checked_envs} envs (vacuous)"
    )


def test_server_create_metadata_equals_direct_make(client):
    """POST /instances (REAL construction — no gym.make stub) returns the
    same metadata as a direct ``gym.make`` for the same key. captcha:
    probe-free ctor, no containers."""
    try:
        tid = registry.task_ids("captcha")["eval"][0]
        key = f"captcha@{tid}"
        direct = _jsonable(gym.make(key).metadata.to_dict())
    except EnvDepsMissingError as e:
        pytest.skip(f"captcha unavailable (deps/assets; run its install.sh): {e}")
    r = client.post(
        "/instances",
        headers=_h(_TOKEN),
        json={"env_key": key, "env_kwargs": {}, "session_id": "dual-mode"},
    )
    assert r.status_code == 201, r.text
    assert r.json()["metadata"] == direct
    client.delete(
        "/instances",
        headers=_h(_TOKEN),
        params={"session_id": "dual-mode", "env_id": "captcha"},
    )


class _KwargMetadataEnv(LiteBaseEnv):
    def __init__(
        self,
        *,
        valid_actions: list[str] | None = None,
        extra_tools: list[str] | None = None,
        cursor: bool = True,
    ) -> None:
        self._valid_actions = valid_actions
        self._extra_tools = list(extra_tools or [])
        self._cursor = cursor

    def _runtime_metadata(self) -> LiteCUAMetadata:
        identity = getattr(self, "identity", None)
        return LiteCUAMetadata(
            dims=("browser", "use"),
            valid_actions=self._valid_actions,
            extra_tool_schemas=[
                make_tool_schema(
                    name,
                    description=f"{name} tool",
                    parameters={"type": "object", "properties": {}},
                )
                for name in self._extra_tools
            ],
            others={
                "cursor": self._cursor,
                "session_id": (
                    identity.resolved_session_id() if identity is not None else None
                ),
            },
        )

    async def reset(self) -> LiteEnvObservation:
        return LiteEnvObservation(image=None, text="ready")

    async def step(self, actions) -> LiteEnvStepResult:
        return LiteEnvStepResult()

    async def close(self) -> None:
        pass


class _GenericMetadataEnv(LiteBaseEnv):
    def _runtime_metadata(self) -> LiteGenericMetadata:
        identity = getattr(self, "identity", None)
        return LiteGenericMetadata(
            extra_tool_schemas=[
                make_tool_schema(
                    "answer",
                    description="answer tool",
                    parameters={"type": "object", "properties": {}},
                )
            ],
            others={
                "session_id": (
                    identity.resolved_session_id() if identity is not None else None
                ),
            },
        )

    async def reset(self) -> LiteEnvObservation:
        return LiteEnvObservation(image=None, text="ready")

    async def step(self, actions) -> LiteEnvStepResult:
        return LiteEnvStepResult()

    async def close(self) -> None:
        pass


def test_server_create_metadata_equals_direct_make_with_nonempty_kwargs(client):
    """Live create metadata parity must cover non-empty env/make kwargs too."""
    env_id = "_dual_mode_kwargs"
    task_id = "t1"
    key = f"{env_id}@{task_id}"
    env_kwargs = {
        "valid_actions": ["click"],
        "extra_tools": ["response"],
        "cursor": False,
    }
    try:
        _imported[env_id] = "local"
        register(
            key,
            entry_point=_KwargMetadataEnv,
            split="eval",
            metadata=LiteCUAMetadata(dims=("browser", "use")),
        )
        registry.set_env_make_kwargs(env_id, {"cursor": True})

        direct_env = gym.make(key, session_id="dual-mode-kwargs", **env_kwargs)
        try:
            direct = _jsonable(direct_env.metadata.to_dict())
        finally:
            import asyncio

            asyncio.run(direct_env.close())

        r = client.post(
            "/instances",
            headers=_h(_TOKEN),
            json={
                "env_key": key,
                "env_kwargs": env_kwargs,
                "session_id": "dual-mode-kwargs",
            },
        )
        assert r.status_code == 201, r.text
        assert r.json()["metadata"] == direct
    finally:
        client.delete(
            "/instances",
            headers=_h(_TOKEN),
            params={"session_id": "dual-mode-kwargs", "env_id": env_id},
        )
        _specs.pop(key, None)
        _splits.pop(env_id, None)
        _env_make_kwargs.pop(env_id, None)
        _imported.pop(env_id, None)


def test_server_create_metadata_parses_generic_metadata(client):
    env_id = "_dual_mode_generic"
    task_id = "t1"
    key = f"{env_id}@{task_id}"
    try:
        _imported[env_id] = "local"
        register(
            key,
            entry_point=_GenericMetadataEnv,
            split="eval",
            metadata=LiteGenericMetadata(),
        )

        direct_env = gym.make(key, session_id="dual-mode-generic")
        try:
            direct = _jsonable(direct_env.metadata.to_dict())
        finally:
            import asyncio

            asyncio.run(direct_env.close())

        r = client.post(
            "/instances",
            headers=_h(_TOKEN),
            json={
                "env_key": key,
                "env_kwargs": {},
                "session_id": "dual-mode-generic",
            },
        )
        assert r.status_code == 201, r.text
        assert r.json()["metadata"] == direct
        assert r.json()["metadata"]["metadata_kind"] == "generic"
        assert r.json()["metadata"]["dims"] == []
    finally:
        client.delete(
            "/instances",
            headers=_h(_TOKEN),
            params={"session_id": "dual-mode-generic", "env_id": env_id},
        )
        _specs.pop(key, None)
        _splits.pop(env_id, None)
        _imported.pop(env_id, None)


def test_server_tasks_metadata_emits_generic_metadata(client):
    env_id = "_dual_mode_generic_tasks"
    key = f"{env_id}@t1"
    try:
        _imported[env_id] = "local"
        register(
            key,
            entry_point=_GenericMetadataEnv,
            split="eval",
            metadata=LiteGenericMetadata(others={"source": "registry"}),
        )

        r = client.get(f"/envs/{env_id}/tasks", headers=_h(_TOKEN))
        assert r.status_code == 200, r.text
        metadata = r.json()["metadata"]["t1"]
        assert metadata["metadata_kind"] == "generic"
        assert metadata["dims"] == []
        assert metadata["others"] == {
            "source": "registry",
            "env_id": env_id,
            "task_id": "t1",
        }
    finally:
        _specs.pop(key, None)
        _splits.pop(env_id, None)
        _imported.pop(env_id, None)
