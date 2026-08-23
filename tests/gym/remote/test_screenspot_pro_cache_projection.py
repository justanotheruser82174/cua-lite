"""ScreenSpot-Pro HF cache discovery parity.

These tests do not use the real HuggingFace cache. They install a tiny fake
``huggingface_hub.snapshot_download`` module and prove that both direct registry
discovery and env-server discovery resolve the ScreenSpot-Pro task catalog from
the same server-process cache path.

Run:
    uv run pytest tests/gym/remote/test_screenspot_pro_cache_projection.py -q
"""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from huggingface_hub import hf_hub_download as _real_hf_hub_download

import lite.gym as gym
import lite.gym.utils.server.routing as routing
from lite.gym.errors import EnvDepsMissingError
from lite.gym.registry import _imported, _specs, _splits, _tasks_registered
from lite.gym.remote.admission import AdmissionConfig, AdmissionGate
from lite.gym.remote.server import State, make_app

ENV_ID = "screenspot_pro"

TOKEN = "screenspot-cache-test"


@pytest.fixture()
def isolated_screenspot_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    import lite.gym.envs.screenspot_pro.main as screenspot

    saved_module_registered = screenspot._tasks_registered
    saved_registry_registered = ENV_ID in _tasks_registered
    saved_specs = {key: value for key, value in _specs.items() if key.startswith(f"{ENV_ID}@")}
    saved_splits = {key: list(value) for key, value in (_splits.get(ENV_ID) or {}).items()}
    had_splits = ENV_ID in _splits
    saved_imported = _imported.get(ENV_ID)
    had_imported = ENV_ID in _imported
    saved_serve_locally = routing._serve_locally

    def clear() -> None:
        for key in [key for key in _specs if key.startswith(f"{ENV_ID}@")]:
            _specs.pop(key, None)
        _splits.pop(ENV_ID, None)
        _tasks_registered.discard(ENV_ID)
        screenspot._tasks_registered = False
        _imported[ENV_ID] = "local"

    clear()
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_URL", raising=False)
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_TOKEN", raising=False)
    yield screenspot

    clear()
    _specs.update(saved_specs)
    if had_splits:
        _splits[ENV_ID] = saved_splits
    if saved_registry_registered:
        _tasks_registered.add(ENV_ID)
    screenspot._tasks_registered = saved_module_registered
    if had_imported:
        _imported[ENV_ID] = saved_imported  # type: ignore[assignment]
    else:
        _imported.pop(ENV_ID, None)
    routing._serve_locally = saved_serve_locally


def _fake_snapshot(root: Path) -> Path:
    snapshot = root / "snapshot"
    annotations = snapshot / "annotations"
    images = snapshot / "images" / "linux"
    annotations.mkdir(parents=True)
    images.mkdir(parents=True)
    (annotations / "linux_common_linux.json").write_text(
        json.dumps(
            [
                {
                    "instruction": "Click the OK button.",
                    "bbox": [10, 20, 30, 40],
                    "img_size": [100, 100],
                    "img_filename": "linux/ok.png",
                    "application": "demo",
                    "group": "linux_common",
                    "ui_type": "button",
                    "platform": "linux",
                }
            ]
        ),
        encoding="utf-8",
    )
    return snapshot


def _install_fake_hub(
    monkeypatch: pytest.MonkeyPatch,
    snapshot_download,
    hf_hub_download=_real_hf_hub_download,
) -> None:
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download
    # Also stub ``hf_hub_download``: while this patch is installed, ANY
    # first-time import of a module that reads it off ``huggingface_hub``
    # resolves against this stub. ``lite.gym.envs.lite.cuagym.src.utils.dataset``
    # does exactly that, and gets pulled in transitively by the lite env
    # umbrella. Without this line the test passes only when some earlier test
    # already imported that chain -- i.e. it passes in a full run and fails when
    # run alone, which is the worst failure mode a test can have.
    hub.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)


def _clear_screenspot_tasks(screenspot: Any) -> None:
    for key in [key for key in _specs if key.startswith(f"{ENV_ID}@")]:
        _specs.pop(key, None)
    _splits.pop(ENV_ID, None)
    _tasks_registered.discard(ENV_ID)
    screenspot._tasks_registered = False
    _imported[ENV_ID] = "local"


def _server_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Stub BOTH membership accessors: the catalog endpoints reach for
    # ``registered_env_ids`` (unscoped list) or ``is_known_env_id`` (single-env
    # probe). Leaving either real would register screenspot_pro here and add a
    # snapshot_download probe pair that the call counts below do not expect —
    # this fixture is measuring the probes ``task_ids`` makes, not membership.
    monkeypatch.setattr(gym.registry, "registered_env_ids", lambda: [ENV_ID])
    monkeypatch.setattr(gym.registry, "is_known_env_id", lambda env_id: env_id == ENV_ID)
    state = State(
        admission=AdmissionGate(
            AdmissionConfig(
                max_live_envs=10,
                emergency_ram_pct=99.9,
                emergency_ram_free_min_bytes=1,
                emergency_load_per_cpu=9999.0,
            )
        ),
        idle_ttl_sec=3600.0,
        allowed_env_ids={ENV_ID},
    )
    return TestClient(make_app(state, token=TOKEN))


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def test_screenspot_cached_hf_snapshot_is_discovered_in_direct_and_server_mode(
    isolated_screenspot_registry: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    screenspot = isolated_screenspot_registry
    snapshot = _fake_snapshot(tmp_path)
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(
        repo_id: str,
        *,
        repo_type: str,
        allow_patterns: list[str],
        local_files_only: bool = False,
    ) -> str:
        calls.append(
            {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "allow_patterns": list(allow_patterns),
                "local_files_only": local_files_only,
            }
        )
        assert repo_id == "likaixin/ScreenSpot-Pro"
        assert repo_type == "dataset"
        assert allow_patterns == ["annotations/*.json", "images/**/*.png"]
        assert local_files_only is True
        return str(snapshot)

    _install_fake_hub(monkeypatch, fake_snapshot_download)

    direct_splits = gym.registry.task_ids(ENV_ID)
    assert direct_splits == {"eval": ["linux_common_linux_0"]}

    _clear_screenspot_tasks(screenspot)
    with _server_client(monkeypatch) as client:
        env_meta = client.get(f"/envs/{ENV_ID}", headers=_auth())
        assert env_meta.status_code == 200, env_meta.text
        assert env_meta.json() == {
            "available": True,
            "n_tasks": 1,
            "splits": ["eval"],
        }
        tasks = client.get(f"/envs/{ENV_ID}/tasks", headers=_auth())
        assert tasks.status_code == 200, tasks.text
        payload = tasks.json()

    assert payload["splits"] == direct_splits
    assert set(payload["metadata"]) == {"linux_common_linux_0"}
    assert payload["metadata"]["linux_common_linux_0"]["others"]["bbox"] == [10, 20, 30, 40]
    assert [call["local_files_only"] for call in calls] == [True, True]


def test_screenspot_server_reports_missing_hf_cache_from_server_process(
    isolated_screenspot_registry: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screenspot = isolated_screenspot_registry
    calls: list[bool] = []

    def fake_snapshot_download(
        _repo_id: str,
        *,
        repo_type: str,
        allow_patterns: list[str],
        local_files_only: bool = False,
    ) -> str:
        calls.append(local_files_only)
        assert repo_type == "dataset"
        assert allow_patterns == ["annotations/*.json", "images/**/*.png"]
        if local_files_only:
            raise FileNotFoundError("fake cache miss")
        raise RuntimeError("fake offline server process")

    _install_fake_hub(monkeypatch, fake_snapshot_download)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    with pytest.raises(EnvDepsMissingError):
        gym.registry.task_ids(ENV_ID)
    assert calls == [True, False]

    _clear_screenspot_tasks(screenspot)
    calls.clear()
    with _server_client(monkeypatch) as client:
        env_meta = client.get(f"/envs/{ENV_ID}", headers=_auth())
        assert env_meta.status_code == 200, env_meta.text
        body = env_meta.json()
        assert body["available"] is False
        assert "ScreenSpot-Pro annotation data not downloaded" in body["error"]

        tasks = client.get(f"/envs/{ENV_ID}/tasks", headers=_auth())
        assert tasks.status_code == 501, tasks.text
        assert "ScreenSpot-Pro annotation data not downloaded" in tasks.text

    assert calls == [True, False, True, False]
