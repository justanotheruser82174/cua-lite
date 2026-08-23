"""CUAWorld tests split from _cuaworld_support.py: materials registration."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import lite.gym as gym
from lite.gym.envs.lite.cuaworld.src import software
from lite.gym.envs.lite.cuaworld.src.adapter import _final_reward
from lite.gym.errors import EnvDepsMissingError
from lite.gym.sandbox import lookup_task
from lite.gym.services import BackendFamily, family_of, services_for
from tests.gym.envs.lite._cuaworld_support import _GOOD, _materials, _stamp_materials


def test_register_with_faked_materials(tmp_path, monkeypatch):
    monkeypatch.setattr(software, "_CUAWORLD_DIR", tmp_path)
    _materials(tmp_path, "swalpha", "swalpha_env",
               tasks={"t1": _GOOD, "t2": _GOOD},
               registered={"eval": ["t1"], "train": ["t2"]})
    software.register_cuaworld_software("swalpha", "swalpha_env")
    splits = gym.registry.task_ids("lite.cuaworld.swalpha")
    assert splits["eval"] == ["t1"]
    assert splits["train"] == ["t2"]
    # DEDICATED (per-trajectory container, pooled/warmable) — like lite.osworld.
    assert family_of("lite.cuaworld.swalpha") is BackendFamily.DEDICATED


@pytest.mark.asyncio
async def test_registration_preserves_upstream_limits_and_hook_timeout(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(software, "_CUAWORLD_DIR", tmp_path)
    spec = {
        **_GOOD,
        "id": "official-task@1",
        "env_id": "official_env@0.1",
        "init": {
            "max_steps": 17,
            "timeout_sec": 83,
            "reward_type": "continuous",
        },
        "hooks": {
            "pre_task": "/workspace/tasks/limited/setup_task.sh",
            "pre_task_timeout": 47,
        },
    }
    _materials(
        tmp_path,
        "swlimits",
        "swlimits_env",
        tasks={"limited": spec},
        registered={"eval": ["limited"]},
    )
    task_dir = (
        tmp_path / ".cache" / "swlimits" / "swlimits_env" / "tasks" / "limited"
    )
    (task_dir / "setup_task.sh").write_text("true\n")
    _stamp_materials(task_dir.parents[1], "swlimits_env")
    captured = {}

    async def fake_setup(computer, path, **kwargs):
        captured.update(path=path, **kwargs)

    monkeypatch.setattr(software, "run_cuaworld_setup", fake_setup)
    software.register_cuaworld_software("swlimits", "swlimits_env")

    task = lookup_task("lite.cuaworld.swlimits", "limited")
    assert task.max_steps == 17
    assert task.metadata["timeout_sec"] == 83
    assert task.metadata["others"] == {
        "source": "gym-anything/swlimits_env",
        "upstream_task_id": "official-task@1",
        "upstream_env_id": "swlimits_env@official",
        "reward_type": "continuous",
    }
    await task.setup_fn(task, SimpleNamespace())
    assert captured["path"] == task_dir
    assert captured["task_spec"] == spec
    assert captured["timeout"] == 47.0


def test_registration_preserves_unknown_upstream_reward_type(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(software, "_CUAWORLD_DIR", tmp_path)
    spec = {
        **_GOOD,
        "init": {
            "max_steps": 80,
            "timeout_sec": 600,
            "reward_type": "weighted",
        },
    }
    _materials(
        tmp_path,
        "swweighted",
        "swweighted_env",
        tasks={"weighted": spec},
        registered={"eval": ["weighted"]},
    )

    software.register_cuaworld_software("swweighted", "swweighted_env")

    task = lookup_task("lite.cuaworld.swweighted", "weighted")
    assert task.metadata["others"]["reward_type"] == "weighted"
    # An unknown/dense/weighted type is worth 1.0 on a PASS (like sparse) and the
    # verifier's score/100 on a FAILURE, as shaped partial credit. It used to return
    # 0.0 unconditionally, which threw the verdict away: 113 `dense` + 1 `weighted`
    # tasks scored 0 however well the agent did (openrocket/
    # transonic_drag_optimization recorded reward 0.0 while its verifier returned
    # `passed: true, raw_score: 85`). Dense reward would come from upstream
    # `reward_shaping`, and the locked materials declare none — so there is nothing
    # to accumulate. `score/100` on the PASSING side would be wrong too: every
    # registered non-excluded dense/weighted verifier passes below 100 (thresholds
    # 17-75), and three cap at max_score 32/46/52 even on a perfect run.
    assert _final_reward(
        spec,
        raw_score=100.0,
        passed=True,
    ) == (1.0, "weighted")
    assert _final_reward(spec, raw_score=85.0, passed=True) == (1.0, "weighted")
    assert _final_reward(spec, raw_score=85.0, passed=False) == (0.85, "weighted")
    assert _final_reward(spec, raw_score=0.0, passed=False) == (0.0, "weighted")


def test_malformed_task_json_skipped_not_fatal(tmp_path, monkeypatch):
    """M1: an unsalvageable task — missing ``success.spec.program`` (KeyError) or a
    null ``success`` (TypeError on ``None["spec"]``) — is skipped, NOT crashing the
    whole env's registration (which would take down every ``lite.cuaworld.*`` import).
    A merely-null ``init``/``natural_language`` is tolerated (``or {}`` guards)."""
    monkeypatch.setattr(software, "_CUAWORLD_DIR", tmp_path)
    _materials(tmp_path, "swbeta", "swbeta_env",
               tasks={"ok": _GOOD,
                      "bad_missing": {"description": "no success key"},
                      "bad_null_success": {"success": None},
                      "ok_null_init": {**_GOOD, "init": None}},  # salvaged, kept
               registered={"eval": ["ok", "bad_missing", "bad_null_success",
                                    "ok_null_init"]})
    software.register_cuaworld_software("swbeta", "swbeta_env")
    assert gym.registry.task_ids("lite.cuaworld.swbeta")["eval"] == ["ok", "ok_null_init"]


def test_nested_additional_splits_flattened(tmp_path, monkeypatch):
    """M13: a split nested under ``additional_splits`` is registered, not dropped."""
    monkeypatch.setattr(software, "_CUAWORLD_DIR", tmp_path)
    _materials(tmp_path, "swgamma", "swgamma_env",
               tasks={"a": _GOOD, "b": _GOOD},
               registered={"eval": ["a"],
                           "additional_splits": {"long_horizon": ["b"]}})
    software.register_cuaworld_software("swgamma", "swgamma_env")
    splits = gym.registry.task_ids("lite.cuaworld.swgamma")
    assert splits.get("long_horizon") == ["b"]


def test_import_safe_without_materials():
    """The catalog module imports before install without touching Docker or HF."""
    import importlib
    importlib.import_module("lite.gym.envs.lite.cuaworld.main")


def test_missing_materials_remain_discoverable_and_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(software, "_CUAWORLD_DIR", tmp_path)
    env_id = "lite.cuaworld.swlazy"
    software.register_cuaworld_software("swlazy", "swlazy_env")

    assert family_of(env_id) is BackendFamily.DEDICATED
    assert isinstance(services_for(env_id), software.CUAWorldServices)
    registered = gym.registry.registered_env_ids()
    assert env_id in registered
    assert "lite.cuaworld" not in registered
    with pytest.raises(EnvDepsMissingError, match="task materials are missing"):
        gym.registry.task_ids(env_id)

    _materials(
        tmp_path,
        "swlazy",
        "swlazy_env",
        tasks={"ready": _GOOD},
        registered={"eval": ["ready"]},
    )
    assert gym.registry.task_ids(env_id)["eval"] == ["ready"]


def test_incomplete_materials_do_not_poison_lazy_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(software, "_CUAWORLD_DIR", tmp_path)
    env_id = "lite.cuaworld.swincomplete"
    cache = tmp_path / ".cache" / "swincomplete" / "swincomplete_env"
    (cache / "tasks").mkdir(parents=True)
    software.register_cuaworld_software("swincomplete", "swincomplete_env")

    with pytest.raises(EnvDepsMissingError, match="missing or incomplete"):
        gym.registry.task_ids(env_id)

    _materials(
        tmp_path,
        "swincomplete",
        "swincomplete_env",
        tasks={"ready": _GOOD},
        registered={"eval": ["ready"]},
    )
    assert gym.registry.task_ids(env_id)["eval"] == ["ready"]


def test_stale_materials_identity_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(software, "_CUAWORLD_DIR", tmp_path)
    env_id = "lite.cuaworld.swstale"
    _materials(
        tmp_path,
        "swstale",
        "swstale_env",
        tasks={"ready": _GOOD},
        registered={"eval": ["ready"]},
    )
    cache = tmp_path / ".cache" / "swstale" / "swstale_env"
    (cache / ".materials_revision").write_text("stale\n")
    software.register_cuaworld_software("swstale", "swstale_env")
    with pytest.raises(EnvDepsMissingError, match="stale or from a different"):
        gym.registry.task_ids(env_id)


def test_stale_materials_digest_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(software, "_CUAWORLD_DIR", tmp_path)
    env_id = "lite.cuaworld.swdigest"
    _materials(
        tmp_path,
        "swdigest",
        "swdigest_env",
        tasks={"ready": _GOOD},
        registered={"eval": ["ready"]},
    )
    cache = tmp_path / ".cache" / "swdigest" / "swdigest_env"
    (cache / "tasks" / "ready" / "verifier.py").write_text("def verify(*_): return 0\n")
    software.register_cuaworld_software("swdigest", "swdigest_env")
    with pytest.raises(EnvDepsMissingError, match="content digest is stale"):
        gym.registry.task_ids(env_id)


@pytest.mark.parametrize("filename", [".materials_revision", "registered.json"])
def test_unreadable_catalog_files_are_actionable(
    tmp_path, monkeypatch, filename
):
    monkeypatch.setattr(software, "_CUAWORLD_DIR", tmp_path)
    software_name = f"swunreadable{filename[0].isalpha()}"
    env_id = f"lite.cuaworld.{software_name}"
    _materials(
        tmp_path,
        software_name,
        "unreadable_env",
        tasks={"ready": _GOOD},
        registered={"eval": ["ready"]},
    )
    target = tmp_path / ".cache" / software_name / "unreadable_env" / filename
    original_read_text = type(target).read_text

    def fail_read(self, *args, **kwargs):
        if self == target:
            raise PermissionError("read only")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(type(target), "read_text", fail_read)
    software.register_cuaworld_software(software_name, "unreadable_env")
    with pytest.raises(EnvDepsMissingError, match="read|registered.json"):
        gym.registry.task_ids(env_id)


def test_unreadable_task_json_does_not_crash_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(software, "_CUAWORLD_DIR", tmp_path)
    _materials(
        tmp_path,
        "swunreadabletask",
        "unreadabletask_env",
        tasks={"ready": _GOOD, "blocked": _GOOD},
        registered={"eval": ["ready", "blocked"]},
    )
    target = (
        tmp_path
        / ".cache"
        / "swunreadabletask"
        / "unreadabletask_env"
        / "tasks"
        / "blocked"
        / "task.json"
    )
    original_read_text = type(target).read_text

    def fail_read(self, *args, **kwargs):
        if self == target:
            raise PermissionError("read only")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(type(target), "read_text", fail_read)
    software.register_cuaworld_software(
        "swunreadabletask", "unreadabletask_env"
    )
    assert gym.registry.task_ids("lite.cuaworld.swunreadabletask")["eval"] == [
        "ready"
    ]


@pytest.mark.parametrize("registered", [[], {"eval": [1]}, {"additional_splits": []}])
def test_invalid_registered_schema_is_actionable(tmp_path, monkeypatch, registered):
    monkeypatch.setattr(software, "_CUAWORLD_DIR", tmp_path)
    env_id = f"lite.cuaworld.swschema{abs(hash(repr(registered)))}"
    _materials(
        tmp_path,
        env_id.rsplit(".", 1)[-1],
        "schema_env",
        tasks={"ready": _GOOD},
        registered=registered,
    )
    software.register_cuaworld_software(env_id.rsplit(".", 1)[-1], "schema_env")
    with pytest.raises(EnvDepsMissingError, match="registered.json"):
        gym.registry.task_ids(env_id)


def test_bad_manifest_and_empty_catalog_can_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(software, "_CUAWORLD_DIR", tmp_path)
    env_id = "lite.cuaworld.swretry"
    _materials(
        tmp_path,
        "swretry",
        "swretry_env",
        tasks={"ready": {"description": "invalid"}},
        registered={"eval": ["ready"]},
    )
    cache = tmp_path / ".cache" / "swretry" / "swretry_env"
    (cache / "registered.json").write_text("{")
    _stamp_materials(cache, "swretry_env")
    software.register_cuaworld_software("swretry", "swretry_env")
    with pytest.raises(EnvDepsMissingError, match="invalid registered.json"):
        gym.registry.task_ids(env_id)

    (cache / "registered.json").write_text(json.dumps({"eval": ["ready"]}))
    _stamp_materials(cache, "swretry_env")
    with pytest.raises(EnvDepsMissingError, match="no valid registered tasks"):
        gym.registry.task_ids(env_id)

    task = cache / "tasks" / "ready" / "task.json"
    task.write_text(json.dumps(_GOOD))
    _stamp_materials(cache, "swretry_env")
    assert gym.registry.task_ids(env_id)["eval"] == ["ready"]


def test_registration_declares_total_step_timeout(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(software, "_CUAWORLD_DIR", tmp_path)
    _materials(
        tmp_path,
        "swtimeout",
        "swtimeout_env",
        tasks={"ready": _GOOD},
        registered={"eval": ["ready"]},
    )
    env_id = "lite.cuaworld.swtimeout"
    try:
        software.register_cuaworld_software("swtimeout", "swtimeout_env")

        assert gym.registry.env_make_kwargs(env_id)["step_timeout"] == (
            software.CFG.make_kwargs["step_timeout"]
        )
    finally:
        import importlib

        registry_module = importlib.import_module("lite.gym.registry")
        from lite.gym.sandbox.register import _TASKS

        keys = [
            key
            for key in registry_module._specs
            if key.startswith(f"{env_id}@")
        ]
        for key in keys:
            registry_module._specs.pop(key, None)
        for mapping in (
            registry_module._splits,
            registry_module._env_make_kwargs,
            registry_module._families,
            registry_module._services,
            _TASKS,
        ):
            mapping.pop(env_id, None)
        registry_module._declared_env_ids.discard(env_id)
        registry_module._tasks_registered.discard(env_id)
        registry_module._services_started.discard(env_id)


def test_registered_task_path_escape_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(software, "_CUAWORLD_DIR", tmp_path)
    _materials(
        tmp_path,
        "swescape",
        "escape_env",
        tasks={"ready": _GOOD},
        registered={"eval": ["../outside", "ready"]},
    )

    software.register_cuaworld_software("swescape", "escape_env")

    assert gym.registry.task_ids("lite.cuaworld.swescape")["eval"] == ["ready"]


def test_missing_registered_json_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setattr(software, "_CUAWORLD_DIR", tmp_path)
    _materials(
        tmp_path,
        "swunreadabledir",
        "unreadabledir_env",
        tasks={"ready": _GOOD},
        registered={"eval": ["ready"]},
    )
    cache = tmp_path / ".cache" / "swunreadabledir" / "unreadabledir_env"
    (cache / "registered.json").unlink()
    software.register_cuaworld_software(
        "swunreadabledir", "unreadabledir_env"
    )
    with pytest.raises(EnvDepsMissingError, match="missing registered.json"):
        gym.registry.task_ids("lite.cuaworld.swunreadabledir")


def test_missing_local_materials_path_is_actionable(tmp_path, monkeypatch):
    from lite.gym.envs.lite.cuaworld.src import image_spec

    missing = tmp_path / "missing"
    monkeypatch.setenv("LITE_CUAWORLD_MATERIALS_REPO", str(missing))
    monkeypatch.setitem(
        image_spec._REGISTERED_UPSTREAM_ENVS, "swalpha", "swalpha_env"
    )
    svc = software.CUAWorldServices(
        "cua-lite/lite.cuaworld.swalpha:latest",
        "install swalpha",
        lambda: None,
    )
    with pytest.raises(EnvDepsMissingError, match="local materials subtree"):
        svc.ensure("lite.cuaworld.swalpha")
