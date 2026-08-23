"""Provenance tests for the rollout pipeline (issue #120).

Covers:
  * ``build_provenance`` — key set + ``commit`` shape (``lite.utils.git``) +
    ``config_path=None`` passthrough;
  * ``save_run_info`` idempotency — exact-LINE matching (a recorded
    ``--sample 10`` must not swallow a later ``--sample 1``, the C2 prefix
    collision);
  * end-to-end: a stub env registered in the real registry → real
    ``gym.make`` → fake agent → real ``TrajectoryLogger`` →
    ``trajectory.parquet`` whose ``metadata.others`` carries the full
    identity (from the logger's rollout-spec write) + provenance + outcome
    schema.

Run: uv run pytest tests/infer/rollout/test_provenance.py -v
"""
from __future__ import annotations

import asyncio
import json
import re
import sys

import pandas as pd

from lite.core.metadata import LiteCUAMetadata
from lite.data.staging import coerce_meta
from lite.gym.base import LiteBaseEnv
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult
from lite.infer.rollout import build_provenance, build_runtime_stamp, save_run_info

# ---------------------------------------------------------------------------
# build_provenance
# ---------------------------------------------------------------------------

def test_build_provenance_schema():
    p = build_provenance("gpt-5.5", "gpt.teacher", "scripts/configs/x.yaml")
    # EXACT, because every key here lands in ``metadata.others`` of every
    # published row: the runtime stamp used to be spread in and nothing off the
    # row read it, so a set-equality is the thing that keeps it out.
    assert set(p) == {
        "model_id",
        "agent_id",
        "config_path",
        "commit",
        "command",
    }
    assert p["model_id"] == "gpt-5.5"
    assert p["agent_id"] == "gpt.teacher"
    assert p["config_path"] == "scripts/configs/x.yaml"
    # This test runs inside the repo → a real short hash, optionally -dirty.
    assert re.fullmatch(r"[0-9a-f]{6,}(-dirty)?", p["commit"])
    assert p["command"].startswith("uv run python ")


def test_build_runtime_stamp_redacts_server_url(monkeypatch):
    """The stamp goes to ``run_info.txt`` / ``summary.json``, never to a row."""
    monkeypatch.setenv("CUA_LITE_ENV_SERVER_URL", "http://127.0.0.1:30100")
    monkeypatch.setenv("CUA_LITE_ENV_SERVER_TOKEN", "secret-token")
    monkeypatch.setenv("SESSION_ID", "sess-a")

    stamp = build_runtime_stamp()

    assert stamp["runtime_mode"] == "server"
    assert stamp["env_server_url_present"] is True
    assert stamp["env_server_url_redacted"] == "<set>"
    assert "30100" not in str(stamp)
    assert "secret-token" not in str(stamp)
    assert stamp["env_server_token_present"] is True
    assert stamp["session_id"] == "sess-a"


def test_build_provenance_config_path_none_passthrough():
    assert build_provenance("m", "a", None)["config_path"] is None


# ---------------------------------------------------------------------------
# save_run_info idempotency (C2: exact-line match, no prefix collision)
# ---------------------------------------------------------------------------

def test_run_info_prefix_argv_is_not_swallowed(tmp_path, monkeypatch):
    info = tmp_path / "run_info.txt"

    monkeypatch.setattr(sys, "argv", ["scripts/rollout.py", "--sample", "10"])
    save_run_info(tmp_path)
    assert "--sample 10" in info.read_text()

    # A strict PREFIX of the recorded argv — substring matching would skip it.
    monkeypatch.setattr(sys, "argv", ["scripts/rollout.py", "--sample", "1"])
    save_run_info(tmp_path)
    lines = info.read_text().splitlines()
    assert "uv run python scripts/rollout.py --sample 1" in lines

    # Re-invoking with an already-recorded argv stays a no-op.
    n_lines = len(lines)
    save_run_info(tmp_path)
    assert len(info.read_text().splitlines()) == n_lines


# ---------------------------------------------------------------------------
# End-to-end: registry → gym.make → fake agent →
# TrajectoryLogger → trajectory.parquet
# ---------------------------------------------------------------------------

_ENV_ID = "_provtest"
_TASK_ID = "ptask"


class _ProvStubEnv(LiteBaseEnv):
    """Minimal env whose self-reported identity is WRONG on purpose — the
    persisted row must carry the key-derived identity from the wrapper."""

    def __init__(self, **kwargs):  # absorbs seed/max_steps/... make() kwargs
        pass

    def _runtime_metadata(self) -> LiteCUAMetadata:
        return LiteCUAMetadata(
            dims=("browser", "use"),
            others={"domain": "search", "task_id": "env-authored-WRONG"},
        )

    async def reset(self) -> LiteEnvObservation:
        return LiteEnvObservation(image=None, text="ready")

    async def step(self, actions) -> LiteEnvStepResult:
        return LiteEnvStepResult()

    async def close(self) -> None:
        pass


class _FakeAgent:
    """Reads the (wrapped) env's metadata into a LiteSample and drives the
    hook lifecycle exactly like AdapterBasedAgent.sample does."""

    def __init__(self, env):
        self._env = env

    async def sample(self, env, hooks=()):
        from lite.core import (
            LiteRLSample,
            LiteSample,
        )

        lite_rl_sample = LiteRLSample(
            lite_sample=LiteSample(
                metadata=env.metadata,
                images=[],
                messages=[{"role": "user", "content": [{"type": "text", "text": "go"}]}],
            ),
            processed_images=[],
            steps=[],
            episode_return=1.0,
            terminated=True,
            truncated=False,
        )
        for hook in hooks:
            hook.on_complete(lite_rl_sample)
        return lite_rl_sample


def test_run_rollout_persists_identity_and_provenance(tmp_path, monkeypatch):
    import lite.agents.factory as agent_factory

    # NOTE: ``lite.gym.registry`` (module) is shadowed by the ``registry``
    # facade instance re-exported from ``lite.gym`` — from-import the module
    # globals directly (same pattern as tests/gym/registry/test_gym_registry.py).
    from lite.gym.registry import _imported, _specs, _splits, register
    from lite.infer.rollout import run_rollout

    key = f"{_ENV_ID}@{_TASK_ID}"
    # The stub env_id has no importable module — mark it imported so
    # _import_env / task_ids fast-path to the entries registered below.
    monkeypatch.setitem(_imported, _ENV_ID, "local")
    register(
        key=key, entry_point=_ProvStubEnv, split="train",
        metadata=LiteCUAMetadata(dims=("browser", "use"), others={"domain": "search"}),
    )
    monkeypatch.setattr(agent_factory, "make", lambda model_id, *, env, **kw: _FakeAgent(env))

    model_id = "gpt-5.5"  # any AGENTS key works — the factory is faked
    expected_agent_id = agent_factory.AGENTS[model_id]["agent_id"]

    try:
        all_done, log_root = asyncio.run(run_rollout(
            model_id=model_id, model_path=model_id, env_id=_ENV_ID,
            agent_kwargs={}, env_kwargs={}, seed=1,
            log_root=tmp_path / "root",
        ))
    finally:
        _specs.pop(key, None)
        _splits.pop(_ENV_ID, None)

    assert all_done

    pq = log_root / "train" / _TASK_ID / "sample_00" / "trajectory.parquet"
    row = pd.read_parquet(pq).iloc[0]
    metadata = coerce_meta(row["metadata"])
    others = dict(metadata["others"])

    # EXACT key set for ``others``: domain + provenance + rollout identity/outcome.
    assert set(others) == {
        "domain",
        "model_id", "agent_id", "config_path", "commit", "command",  # provenance
        "env_id", "task_id", "episode_return", "terminated", "truncated",
    }
    # Identity: the logger writes the rollout spec's key-derived values under
    # ``metadata.others``,
    # overriding the env's self-reported form.
    assert others["env_id"] == _ENV_ID
    assert others["task_id"] == _TASK_ID
    assert "env_id" not in metadata and "task_id" not in metadata
    assert others["domain"] == "search"
    # Provenance: flat keys from build_provenance.
    assert others["model_id"] == model_id
    assert others["agent_id"] == expected_agent_id
    assert others["config_path"] is None
    assert re.fullmatch(r"[0-9a-f]{6,}(-dirty)?", others["commit"])
    assert others["command"].startswith("uv run python ")
    # Outcome.
    assert float(others["episode_return"]) == 1.0

    summary = json.loads((log_root / "summary.json").read_text())
    config = summary["config"]
    assert config["runtime_mode"] == "direct"
    assert config["env_server_url_present"] is False
    assert config["env_server_url_redacted"] is None
    assert config["config_path"] is None
    assert config["env_kwargs"] == {}


def test_run_rollout_yaml_env_kwargs_image_reaches_make_kwargs(tmp_path, monkeypatch):
    import lite.agents.factory as agent_factory
    from lite.gym.registry import _imported, _specs, _splits, register
    from lite.infer.rollout import run_rollout

    env_id = "_provimage"
    task_id = "image-task"
    key = f"{env_id}@{task_id}"
    captured_kwargs: list[dict] = []

    class _CaptureEnv(_ProvStubEnv):
        def __init__(self, **kwargs):
            captured_kwargs.append(dict(kwargs))

    config = tmp_path / "rollout.yaml"
    config.write_text(
        "\n".join([
            f"env_id: {env_id}",
            "env_kwargs:",
            "  image: cua-lite/lite.osworld:mine",
            "  extra_tools: [bash, terminate]",
        ])
    )

    monkeypatch.delenv("CUA_LITE_ENV_SERVER_URL", raising=False)
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_TOKEN", raising=False)
    monkeypatch.setitem(_imported, env_id, "local")
    register(
        key=key, entry_point=_CaptureEnv, split="train",
        metadata=LiteCUAMetadata(dims=("browser", "use"), others={"domain": "search"}),
    )
    monkeypatch.setattr(agent_factory, "make", lambda model_id, *, env, **kw: _FakeAgent(env))

    try:
        all_done, log_root = asyncio.run(run_rollout(
            model_id="gpt-5.5", model_path="gpt-5.5", env_id=env_id,
            agent_kwargs={}, env_kwargs={"max_steps": 2}, seed=1,
            log_root=tmp_path / "image-root", config_path=str(config),
        ))
    finally:
        _specs.pop(key, None)
        _splits.pop(env_id, None)

    assert all_done
    assert captured_kwargs
    assert captured_kwargs[0]["image"] == "cua-lite/lite.osworld:mine"
    assert captured_kwargs[0]["extra_tools"] == ["bash", "terminate"]
    assert captured_kwargs[0]["max_steps"] == 2

    summary = json.loads((log_root / "summary.json").read_text())
    assert summary["config"]["config_path"] == str(config)
    assert summary["config"]["env_kwargs"] == {
        "image": "cua-lite/lite.osworld:mine",
        "extra_tools": ["bash", "terminate"],
        "max_steps": 2,
    }
