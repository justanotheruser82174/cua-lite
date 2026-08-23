from __future__ import annotations

from fastapi.testclient import TestClient
from gym.remote.conftest import bearer_header as _h
from gym.remote.conftest import make_test_admission

from lite.gym.remote.server import State, make_app

_TOKEN = "catalog-namespace-token"


def test_single_env_catalog_uses_exact_known_env_id_for_multi_level_ids(monkeypatch):
    env_id = "cua.bench.webtop.s02fam"
    seen: list[str] = []

    def is_known_env_id(candidate: str) -> bool:
        seen.append(candidate)
        return candidate == env_id

    monkeypatch.setattr(
        "lite.gym.remote.server.gym.registry.is_known_env_id",
        is_known_env_id,
    )
    monkeypatch.setattr(
        "lite.gym.remote.server.gym.registry.registered_env_ids",
        lambda: (_ for _ in ()).throw(AssertionError("must not enumerate all envs")),
    )
    monkeypatch.setattr(
        "lite.gym.remote.server.gym.registry.task_ids",
        lambda candidate: {"eval": ["task_0"]} if candidate == env_id else KeyError(candidate),
    )
    monkeypatch.setattr(
        "lite.gym.services.health_check",
        lambda candidate: None,
    )

    state = State(
        admission=make_test_admission(max_live_envs=4),
        idle_ttl_sec=3600.0,
        allowed_env_ids={env_id},
    )
    with TestClient(make_app(state, token=_TOKEN)) as client:
        response = client.get(f"/envs/{env_id}", headers=_h(_TOKEN))

    assert response.status_code == 200, response.text
    assert response.json() == {
        "available": True,
        "n_tasks": 1,
        "splits": ["eval"],
    }
    assert seen == [env_id]
