"""lite.cuaworld setup budget vs. reset deadline — the anti-drift guard.

``SandboxBaseEnv.reset()`` runs boot + ``run_cuaworld_post_start`` + the task's
``setup_fn``, and ``StepTimeoutWrapper`` bounds the whole thing with
``reset_timeout``. A task's ``hooks.pre_task_timeout`` used to be honoured up to
1800 s while ``reset_timeout`` stayed at configs/default.yaml's 600 s, so every
long setup died with a generic ``EnvTimeoutError`` before its own budget expired
(130 such tasks on disk, all slicer3d: 1800 s x63, 1200 s x24, 900 s x43).

The fix makes the two numbers ONE number: ``MAX_PRE_TASK_TIMEOUT`` both caps the
declared setup budget (clamp + WARN) and is added into the registered
``reset_timeout``. These tests pin that coupling from both ends.

Run: uv run pytest tests/gym/envs/lite/cuaworld/test_cuaworld_reset_budget.py -q
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lite.gym.envs.lite.cuaworld.src import software


def _boot_budget() -> float:
    """configs/default.yaml's ``reset_timeout`` — the boot/post_start share."""
    return float(software.CFG.make_kwargs["reset_timeout"])


# --------------------------------------------------------------------------- #
# the coupling itself
# --------------------------------------------------------------------------- #

def test_reset_timeout_covers_the_longest_declarable_setup():
    assert software.MAKE_KWARGS["reset_timeout"] == (
        _boot_budget() + software.MAX_PRE_TASK_TIMEOUT
    )
    # ...and is therefore strictly larger than any setup budget that can be run.
    assert software.MAKE_KWARGS["reset_timeout"] > software.MAX_PRE_TASK_TIMEOUT


def test_make_kwargs_only_overrides_reset_timeout():
    assert set(software.MAKE_KWARGS) == set(software.CFG.make_kwargs)
    for k, v in software.CFG.make_kwargs.items():
        if k != "reset_timeout":
            assert software.MAKE_KWARGS[k] == v


def test_post_start_keeps_the_yaml_boot_budget():
    """The yaml value stays the boot/post_start share — only the WRAPPER
    deadline grows. Guards against "fixing" this by editing the yaml number,
    which would silently hand post_start a 2400 s budget too."""
    src = Path(software.__file__).read_text()
    assert 'timeout=float(CFG.make_kwargs["reset_timeout"])' in src


# --------------------------------------------------------------------------- #
# _make_setup_fn — honour up to the ceiling, clamp + WARN above it
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
@pytest.mark.parametrize("declared,expected", [
    (None, 600.0),                                  # upstream default
    (900, 900.0),                                   # slicer3d tier 1
    (1200, 1200.0),                                 # slicer3d tier 2
    (1800, 1800.0),                                 # slicer3d tier 3 — the ceiling
])
async def test_declared_budget_is_honoured_up_to_the_ceiling(
    declared, expected, monkeypatch, tmp_path,
):
    captured = {}

    async def fake_setup(computer, path, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(software, "run_cuaworld_setup", fake_setup)
    spec = {"hooks": {}} if declared is None else {"hooks": {"pre_task_timeout": declared}}
    await software._make_setup_fn(tmp_path, spec)(None, SimpleNamespace())
    assert captured["timeout"] == expected
    assert captured["timeout"] <= software.MAX_PRE_TASK_TIMEOUT


@pytest.mark.asyncio
async def test_over_ceiling_declaration_is_clamped_with_a_warning(
    monkeypatch, tmp_path, caplog,
):
    captured = {}

    async def fake_setup(computer, path, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(software, "run_cuaworld_setup", fake_setup)
    task_dir = tmp_path / "greedy_task"
    with caplog.at_level("WARNING", logger=software.__name__):
        await software._make_setup_fn(
            task_dir, {"hooks": {"pre_task_timeout": 3600}},
        )(None, SimpleNamespace())

    assert captured["timeout"] == software.MAX_PRE_TASK_TIMEOUT
    assert "greedy_task" in caplog.text
    assert "clamping" in caplog.text


@pytest.mark.asyncio
async def test_non_dict_hooks_fall_back_to_the_upstream_default(monkeypatch, tmp_path):
    captured = {}

    async def fake_setup(computer, path, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(software, "run_cuaworld_setup", fake_setup)
    await software._make_setup_fn(tmp_path, {"hooks": "not-a-dict"})(None, SimpleNamespace())
    assert captured["timeout"] == software._UPSTREAM_PRE_TASK_TIMEOUT


# --------------------------------------------------------------------------- #
# the real corpus — no on-disk task may out-declare the ceiling
# --------------------------------------------------------------------------- #

def test_installed_corpus_declares_nothing_above_the_ceiling():
    """If materials are installed, every declared ``pre_task_timeout`` must fit
    under the ceiling — otherwise the clamp would silently shorten real setups
    and MAX_PRE_TASK_TIMEOUT needs raising (which raises reset_timeout with it)."""
    root = software._materials_dir()
    task_files = list(root.glob("*/*/tasks/*/task.json"))
    if not task_files:
        pytest.skip("lite.cuaworld materials are not installed")
    over: dict[str, float] = {}
    for tj in task_files:
        try:
            spec = json.loads(tj.read_text())
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue  # a few upstream task.json files are malformed; registration skips them too
        hooks = spec.get("hooks") or {}
        if not isinstance(hooks, dict):
            continue
        declared = float(hooks.get("pre_task_timeout", software._UPSTREAM_PRE_TASK_TIMEOUT))
        if declared > software.MAX_PRE_TASK_TIMEOUT:
            over[str(tj.parent.name)] = declared
    assert not over, f"declarations above MAX_PRE_TASK_TIMEOUT: {over}"
