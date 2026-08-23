"""CUA bench direct catalog registration coverage."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from lite.core.metadata import LiteCUAMetadata

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


def _isolate_registry(monkeypatch: pytest.MonkeyPatch) -> Any:
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
def isolated_cua_bench_catalog(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    registry_mod = _isolate_registry(monkeypatch)
    import lite.gym.utils.server.routing as routing
    from lite.gym.utils import config as env_config

    monkeypatch.delenv("CUA_LITE_ENV_SERVER_URL", raising=False)
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_TOKEN", raising=False)
    monkeypatch.delenv("CUA_BENCH_CONFIG", raising=False)
    monkeypatch.setenv("CUA_BENCH_DATASET_ROOT", str(_write_fake_dataset(tmp_path)))
    monkeypatch.setitem(sys.modules, "cua_bench", _fake_cua_bench_module())
    monkeypatch.delitem(sys.modules, "lite.gym.envs.cua.bench.main", raising=False)
    monkeypatch.setattr(routing, "_serve_locally", False)
    env_config.load.cache_clear()
    yield registry_mod
    sys.modules.pop("lite.gym.envs.cua.bench.main", None)
    env_config.load.cache_clear()


def test_cua_bench_webtop_catalog_registers_tasks_metadata_and_defaults(
    isolated_cua_bench_catalog,
) -> None:
    _ = isolated_cua_bench_catalog
    cb_main = importlib.import_module("lite.gym.envs.cua.bench.main")

    assert cb_main.registry.task_ids(_ENV_ID) == {"eval": _TASK_IDS}
    assert cb_main.registry.env_make_kwargs(_ENV_ID) == {"cursor": True}
    assert cb_main.registry.env_supported_kwargs(_ENV_ID) == (
        _EXPECTED_ENV_SUPPORTED_KWARGS
    )
    assert cb_main.registry.task_kwargs(_ENV_ID, _TASK_IDS[0]) == {}

    for task_id in _TASK_IDS:
        task_metadata = cb_main.registry.task_metadata(_ENV_ID, task_id)
        assert isinstance(task_metadata, LiteCUAMetadata)
        assert task_metadata.dims == ("desktop", "use")
        assert task_metadata.valid_actions == _EXPECTED_VALID_ACTIONS
        assert task_metadata.others == {"env_id": _ENV_ID, "task_id": task_id}
