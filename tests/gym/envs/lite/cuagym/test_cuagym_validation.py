from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lite.gym.envs.lite.cuagym import main as M
from lite.gym.envs.lite.cuagym.src.utils import dataset
from lite.gym.errors import EnvDepsMissingError

ROOT = Path(__file__).resolve().parents[5]


def _require_fresh_task_cache() -> None:
    try:
        M._register_tasks()
    except EnvDepsMissingError as exc:
        pytest.skip(str(exc))


def test_validation_lock_matches_the_pinned_catalog() -> None:
    from lite.gym.envs.lite.cuagym.scripts.utils import validation_sweep

    lock = validation_sweep.audit_lock(validation_sweep._rows())
    assert lock["_meta"]["total"] == 209
    assert sum(
        finding.get("layer") == "duplicate_bundle_audit"
        for task_id, finding in lock.items()
        if task_id != "_meta"
    ) == 128


def test_validation_lock_is_consumed_by_the_catalogs() -> None:
    _require_fresh_task_cache()
    catalogs = [
        ROOT / "lite/gym/envs/lite/cuagym/.cache/web/lite.cuagym_tasks/train.jsonl",
        ROOT
        / "lite/gym/envs/lite/cuagym/.cache/desktop/"
        "lite.cuagym_desktop_tasks/train.jsonl",
    ]
    rows = {
        row["task_id"]: row
        for catalog in catalogs
        for row in map(json.loads, catalog.read_text().splitlines())
    }
    locked = set(dataset.validation_excludes())
    assert locked <= rows.keys()
    assert {
        (rows[task_id]["metadata"].get("others") or {}).get("exclude_reason")
        for task_id in locked
    } == {
        "broken_mock:blank_render",
        "broken_reward:instruction_mismatch",
    }


def test_live_validation_rechecks_prior_live_findings() -> None:
    from lite.gym.envs.lite.cuagym.scripts.utils import validation_sweep

    rows = {
        "clean": {"metadata": {"others": {}}},
        "prior-live": {
            "metadata": {
                "others": {"exclude_reason": "broken_reward:nonzero_baseline"}
            }
        },
        "static": {
            "metadata": {
                "others": {"exclude_reason": "broken_mock:blank_render"}
            }
        },
    }
    lock = {
        "_meta": {},
        "prior-live": {
            "layer": "live_noop",
            "reason": "broken_reward:nonzero_baseline",
        },
        "static": {
            "layer": "mock_runtime_audit",
            "reason": "broken_mock:blank_render",
        },
    }

    assert validation_sweep._live_ids(rows, lock) == ["clean", "prior-live"]


def test_live_validation_only_excludes_terminal_task_errors() -> None:
    from lite.gym.envs.lite.cuagym.scripts.utils import validation_sweep
    from lite.gym.errors import CuaGymTaskError

    terminal = validation_sweep._task_failure(
        "task",
        "setup",
        CuaGymTaskError("bad bundle", phase="setup", kind="command_failed"),
    )
    transient = validation_sweep._task_failure(
        "task",
        "setup",
        ConnectionError("docker unavailable"),
    )

    assert terminal["reason"] == "broken_setup:live_validation_error"
    assert transient["outcome"] == "transient_setup_error"
    assert "reason" not in transient


def test_live_validation_bypasses_exclusion_on_unwrapped_env() -> None:
    from lite.gym.envs.lite.cuagym.scripts.utils import validation_sweep
    from lite.gym.envs.lite.cuagym.src.browser import scripts as browser

    inner = SimpleNamespace(
        _task=SimpleNamespace(metadata={"others": {"apps": ["slack"]}}),
        _setup_fn=object(),
    )
    wrapper = SimpleNamespace(unwrapped=inner, _setup_fn=object())

    validation_sweep._bypass_exclusion_guard(wrapper)

    assert inner._setup_fn is browser.setup_fn
    assert wrapper._setup_fn is not browser.setup_fn


@pytest.mark.asyncio
async def test_live_validation_steps_with_canonical_terminate(monkeypatch) -> None:
    import lite.gym as gym
    from lite.core.tools import make_tool_call
    from lite.gym.envs.lite.cuagym.scripts.utils import validation_sweep

    seen = {}

    class Env:
        async def reset(self):
            seen["reset"] = True

        async def step(self, actions):
            seen["actions"] = actions
            return SimpleNamespace(reward=0.0)

        async def close(self):
            seen["closed"] = True

    monkeypatch.setattr(gym, "make", lambda *_args, **_kwargs: Env())

    result = await validation_sweep._live_once("task", bypass_exclude=False)

    assert result == {"task_id": "task", "outcome": "clean_zero", "reward": 0.0}
    assert seen["actions"] == [make_tool_call("terminate", {})]
    assert seen["closed"] is True


def test_validation_checkpoint_rejects_different_run(tmp_path: Path) -> None:
    from lite.gym.envs.lite.cuagym.scripts.utils import validation_sweep

    path = tmp_path / "validation.jsonl"
    ids = ["a", "b"]
    path.write_text(
        json.dumps({"_meta": validation_sweep._checkpoint_meta(ids, 2)}) + "\n"
        + json.dumps({"task_id": "a", "outcome": "clean_zero"}) + "\n"
    )

    assert validation_sweep._load_checkpoint(path, ids, 2)[0]["task_id"] == "a"
    with pytest.raises(RuntimeError, match="does not match"):
        validation_sweep._load_checkpoint(path, ["a"], 2)


@pytest.mark.asyncio
async def test_live_validation_transient_does_not_make_task_unstable(
    monkeypatch,
) -> None:
    from lite.gym.envs.lite.cuagym.scripts.utils import validation_sweep

    outcomes = iter(("transient_reward_error", "clean_zero"))

    async def live_once(task_id, *, bypass_exclude):
        return {"task_id": task_id, "outcome": next(outcomes), "reward": 0}

    monkeypatch.setattr(validation_sweep, "_live_once", live_once)
    result = await validation_sweep._live_one(
        "task", asyncio.Semaphore(1), attempts=2, bypass_exclude=False
    )

    assert result["outcome"] == "clean_zero"
