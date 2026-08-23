"""Hermetic CUA bench remote ``/tasks`` catalog/readback coverage."""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from gym.remote.conftest import bearer_header as _h
from gym.remote.conftest import make_test_admission

from lite.core.metadata import LiteCUAMetadata
from lite.gym.remote.server import State, make_app

_TOKEN = "cua-bench-tasks-test"
_ENV_ID = "cua.bench.webtop.s02fam"
_TASK_IDS = ["webtop_task/0", "webtop_task/1"]
_EXPECTED_VALID_ACTIONS = [
    "click",
    "type",
    "key",
    "mouse_move",
    "drag",
    "scroll",
    "wait",
]
_EXPECTED_ENV_SUPPORTED_KWARGS = [
    "env_path",
    "extra_tools",
    "height",
    "max_steps",
    "post_action_delay",
    "valid_actions",
    "variant_idx",
    "width",
]


class _FakeCuaBenchEnv:
    def __init__(self, env_path: str) -> None:
        self._env_path = Path(env_path)

    def tasks_config_fn(self) -> list[types.SimpleNamespace]:
        return [
            types.SimpleNamespace(
                description=f"{self._env_path.name} variant {idx}",
                computer={"provider": "webtop", "setup_config": {}},
            )
            for idx in range(len(_TASK_IDS))
        ]


def _fake_cua_bench_module() -> types.ModuleType:
    module = types.ModuleType("cua_bench")
    module.make = lambda env_path: _FakeCuaBenchEnv(env_path)
    return module


def _write_fake_dataset(tmp_path: Path) -> Path:
    env_dir = tmp_path / "cua-bench-s02fam" / "webtop_task"
    env_dir.mkdir(parents=True)
    (env_dir / "main.py").write_text("# fake cua-bench task module\n")
    return env_dir.parent


def _isolate_registry(monkeypatch) -> Any:
    registry_mod = importlib.import_module("lite.gym.registry")
    for name in (
        "_env_make_kwargs",
        "_env_supported_kwargs",
        "_families",
        "_import_locks",
        "_imported",
        "_imported_remote_urls",
        "_services",
        "_specs",
        "_splits",
    ):
        monkeypatch.setattr(registry_mod, name, {})
    for name in ("_declared_env_ids", "_services_started", "_tasks_registered"):
        monkeypatch.setattr(registry_mod, name, set())
    return registry_mod


@pytest.fixture()
def isolated_registry(monkeypatch: pytest.MonkeyPatch):
    registry_mod = _isolate_registry(monkeypatch)
    import lite.gym.utils.server.routing as routing
    from lite.gym.utils import config as env_config

    monkeypatch.delenv("CUA_LITE_ENV_SERVER_URL", raising=False)
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_TOKEN", raising=False)
    monkeypatch.delenv("CUA_BENCH_CONFIG", raising=False)
    monkeypatch.delitem(sys.modules, "lite.gym.envs.cua.bench.main", raising=False)
    monkeypatch.setattr(routing, "_serve_locally", False)
    env_config.load.cache_clear()
    yield registry_mod
    sys.modules.pop("lite.gym.envs.cua.bench.main", None)
    env_config.load.cache_clear()


def _make_client() -> TestClient:
    state = State(
        admission=make_test_admission(max_live_envs=4),
        idle_ttl_sec=3600.0,
    )
    return TestClient(make_app(state, token=_TOKEN, port=30997))


class _CatalogResponse:
    status_code = 200

    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body
        self.text = json.dumps(body)

    def json(self) -> dict[str, Any]:
        return self._body

    def raise_for_status(self) -> None:
        pass


def _metadata_payload(task_id: str) -> dict[str, Any]:
    return {
        "metadata_kind": "cua",
        "dims": ["desktop", "use"],
        "extra_tool_schemas": [],
        "others": {"env_id": _ENV_ID, "task_id": task_id},
        "valid_actions": _EXPECTED_VALID_ACTIONS,
    }


def _catalog_payload() -> dict[str, Any]:
    return {
        "splits": {"eval": list(_TASK_IDS)},
        "metadata": {task_id: _metadata_payload(task_id) for task_id in _TASK_IDS},
        "kwargs": {},
        "env_make_kwargs": {"cursor": True},
        "env_supported_kwargs": list(_EXPECTED_ENV_SUPPORTED_KWARGS),
    }


def _assert_catalog_payload(catalog: dict[str, Any]) -> None:
    expected = _catalog_payload()
    assert catalog["splits"] == expected["splits"]
    assert catalog["kwargs"] == expected["kwargs"]
    assert catalog["env_make_kwargs"] == expected["env_make_kwargs"]
    assert catalog["env_supported_kwargs"] == expected["env_supported_kwargs"]
    assert catalog["metadata"] == expected["metadata"]


def test_cua_bench_webtop_tasks_catalog_endpoint(
    isolated_registry,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _ = isolated_registry
    monkeypatch.setenv("CUA_BENCH_DATASET_ROOT", str(_write_fake_dataset(tmp_path)))
    monkeypatch.setitem(sys.modules, "cua_bench", _fake_cua_bench_module())

    with _make_client() as client:
        response = client.get(f"/envs/{_ENV_ID}/tasks", headers=_h(_TOKEN))

    assert response.status_code == 200, response.text
    _assert_catalog_payload(response.json())


def test_cua_bench_remote_client_projects_tasks_catalog(
    isolated_registry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_mod = isolated_registry
    catalog = _catalog_payload()

    import lite.gym.utils.server.routing as routing

    monkeypatch.setattr(routing, "_serve_locally", False)

    from lite.gym.remote import client as remote_client

    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> _CatalogResponse:
        assert url == f"http://fake-server/envs/{_ENV_ID}/tasks"
        assert headers == {"Authorization": f"Bearer {_TOKEN}"}
        assert timeout == 30.0
        return _CatalogResponse(catalog)

    monkeypatch.setenv("CUA_LITE_ENV_SERVER_URL", "http://fake-server")
    monkeypatch.setenv("CUA_LITE_ENV_SERVER_TOKEN", _TOKEN)
    monkeypatch.setattr(remote_client.httpx, "get", fake_get)

    registry_mod._import_env(_ENV_ID)

    assert registry_mod._imported[_ENV_ID] == "remote"
    assert registry_mod._imported_remote_urls[_ENV_ID] == "http://fake-server"
    assert registry_mod.registry.task_ids(_ENV_ID) == {"eval": _TASK_IDS}
    assert registry_mod.registry.env_make_kwargs(_ENV_ID) == {"cursor": True}
    assert registry_mod.registry.env_supported_kwargs(_ENV_ID) == catalog[
        "env_supported_kwargs"
    ]
    readback_metadata = registry_mod.registry.task_metadata(_ENV_ID, _TASK_IDS[0])
    assert isinstance(readback_metadata, LiteCUAMetadata)
    assert readback_metadata.dims == ("desktop", "use")
    assert readback_metadata.others == {
        "env_id": _ENV_ID,
        "task_id": _TASK_IDS[0],
    }
