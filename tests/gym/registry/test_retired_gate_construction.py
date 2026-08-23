"""Construction guards for the retired pre-env availability gate.

Run:
    uv run pytest tests/gym/registry/test_retired_gate_construction.py -q
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import lite.gym as gym
from lite.core.metadata import LiteCUAMetadata
from lite.gym.base import LiteBaseEnv
from lite.gym.remote.admission import AdmissionConfig, AdmissionGate
from lite.gym.remote.server import State, make_app
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult


class _StubEnv(LiteBaseEnv):
    def _runtime_metadata(self) -> LiteCUAMetadata:
        return LiteCUAMetadata(dims=("desktop", "use"))

    async def reset(self) -> LiteEnvObservation:
        return LiteEnvObservation(image=None, text="ok")

    async def step(self, actions: list) -> LiteEnvStepResult:
        return LiteEnvStepResult()

    async def close(self) -> None:
        pass


def _wrapper_chain(env: object) -> list[str]:
    chain: list[str] = []
    cur = env
    while True:
        chain.append(type(cur).__name__)
        inner = getattr(cur, "__dict__", {}).get("env")
        if inner is None:
            return chain
        cur = inner


def _register_stub_env(monkeypatch, env_id: str, key: str) -> None:
    import sys

    from lite.gym.registry import register

    registry_module = sys.modules["lite.gym.registry"]
    register(key, entry_point=lambda **_kw: _StubEnv())
    monkeypatch.setattr(
        registry_module, "_import_env", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(registry_module, "ensure_services", lambda _env_id: None)


def _cleanup_stub_env(env_id: str, key: str) -> None:
    import sys

    registry_module = sys.modules["lite.gym.registry"]
    registry_module._specs.pop(key, None)
    registry_module._splits.pop(env_id, None)
    registry_module._env_make_kwargs.pop(env_id, None)
    registry_module._imported.pop(env_id, None)
    registry_module._services_started.discard(env_id)
    registry_module._tasks_registered.discard(env_id)


def _admission() -> AdmissionGate:
    return AdmissionGate(
        AdmissionConfig(
            max_live_envs=100,
            emergency_ram_pct=99.9,
            emergency_ram_free_min_bytes=1,
            emergency_load_per_cpu=9999.0,
            docker_create_concurrency=8,
        )
    )


def test_direct_make_installs_only_current_wrapper_chain(monkeypatch):
    env_id = "faketest_retired_gate_direct"
    key = f"{env_id}@t1"
    try:
        _register_stub_env(monkeypatch, env_id, key)

        env = gym.make(
            key,
            env_server_url=None,
            step_timeout=1.0,
            reset_timeout=1.0,
            loop_detect=2,
        )

        assert _wrapper_chain(env) == [
            "LoopDetectWrapper",
            "StepTimeoutWrapper",
            "_StubEnv",
        ]
    finally:
        _cleanup_stub_env(env_id, key)


def test_envserver_create_installs_only_current_wrapper_chain(monkeypatch):
    import lite.gym.utils.server.routing as routing

    env_id = "faketest_retired_gate_server"
    key = f"{env_id}@t1"
    try:
        _register_stub_env(monkeypatch, env_id, key)
        monkeypatch.setattr(routing, "_serve_locally", routing._serve_locally)
        state = State(admission=_admission(), idle_ttl_sec=3600.0)
        app = make_app(state, token=None)

        with TestClient(app) as client:
            response = client.post(
                "/instances",
                json={
                    "env_key": key,
                    "env_kwargs": {
                        "step_timeout": 1.0,
                        "reset_timeout": 1.0,
                        "loop_detect": 2,
                    },
                    "session_id": "s",
                },
            )
            assert response.status_code == 201, response.text

            sessions = list(state.envs.values())
            assert len(sessions) == 1
            assert _wrapper_chain(sessions[0].env) == [
                "LoopDetectWrapper",
                "StepTimeoutWrapper",
                "_StubEnv",
            ]
    finally:
        _cleanup_stub_env(env_id, key)
