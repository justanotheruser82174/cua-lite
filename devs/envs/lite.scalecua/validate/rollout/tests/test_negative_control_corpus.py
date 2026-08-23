"""Durable guard for the pinned negative-control corpus.

The negative-control corpus covers wrong-setting, window-state, wrong-content,
and site-block trajectories that are correctly reward=0. It is pinned as fixed
``task_id`` rows with expected non-recovery, and is guarded here so that a future
checker change that "recovers" a negative control wrongly flips it to a pass.

This is a pure static/structural + behavioral test — it reads catalog JSONL and
the corpus only, spins up no container, so it is NOT marked ``live``.

Run:
    uv run pytest devs/envs/lite.scalecua/validate/rollout/tests/test_negative_control_corpus.py -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from lite.gym.envs.lite.scalecua.src.utils import dataset

_CORPUS_PATH = Path(__file__).resolve().parents[2] / "samples/negative_control_corpus.jsonl"
_GATE_SCRIPT = Path(__file__).resolve().parents[1] / "vision_gate.py"


def _load_vision_gate():
    spec = importlib.util.spec_from_file_location("scalecua_vision_gate_nc", _GATE_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_vision_gate()


def _catalog_ids() -> dict[str, set[str]]:
    return {
        split: {row["task_id"] for _, row in dataset.iter_jsonl(dataset.catalog_path(split))}
        for split in dataset.RUNTIME_SPLITS
    }


def test_corpus_is_pinned_wellformed_and_resolves_in_catalog():
    # The loader itself enforces the hard structural invariants (non-empty, closed
    # loosening set, expected_negative_reward==0, expected_recovered is False,
    # is_fn_recovery_target is False, closed class set, no dupes, N>=3 per bucket).
    rows = gate.load_negative_control_corpus(_CORPUS_PATH)
    assert len(rows) >= 3 * len(gate.REQUIRED_NEGATIVE_CONTROL_LOOSENINGS)

    # Every listed loosening bucket is present with >= 3 rows.
    by_loosening: dict[str, int] = {}
    for row in rows:
        by_loosening[row["loosening"]] = by_loosening.get(row["loosening"], 0) + 1
    assert set(by_loosening) == gate.REQUIRED_NEGATIVE_CONTROL_LOOSENINGS
    assert min(by_loosening.values()) >= gate.MIN_ROWS_PER_LOOSENING

    # Class labels stay inside the closed negative-control vocabulary.
    assert {row["negative_control_class"] for row in rows} <= gate.NEGATIVE_CONTROL_CLASSES

    # No row is an FN-recovery target and every row pins NON-recovery.
    for row in rows:
        assert row["expected_recovered"] is False
        assert row["is_fn_recovery_target"] is False
        assert float(row["expected_negative_reward"]) == 0.0

    # Every pinned task_id resolves in the runtime catalog for its split.
    catalog = _catalog_ids()
    for row in rows:
        assert row["split"] in catalog, row["split"]
        assert row["task_id"] in catalog[row["split"]], row["task_id"]


def test_gate_passes_when_all_controls_stay_reward_zero():
    rows = gate.load_negative_control_corpus(_CORPUS_PATH)
    rescored = {row["task_id"]: 0.0 for row in rows}
    report = gate.check_negative_controls_non_recovery(rescored, corpus=rows)
    assert report["passed"] is True
    assert report["recovered"] == []
    assert report["missing"] == []
    assert report["total"] == len(rows)


def test_gate_fails_when_a_negative_control_is_wrongly_recovered():
    """The durability contract: a checker change that flips a pinned negative
    control to a pass MUST turn the gate red."""
    rows = gate.load_negative_control_corpus(_CORPUS_PATH)
    rescored = {row["task_id"]: 0.0 for row in rows}
    flipped = rows[0]["task_id"]
    rescored[flipped] = 1.0  # a loosened checker wrongly "recovers" this control

    report = gate.check_negative_controls_non_recovery(rescored, corpus=rows)
    assert report["passed"] is False
    assert [entry["task_id"] for entry in report["recovered"]] == [flipped]
    assert report["recovered"][0]["loosening"] == rows[0]["loosening"]


def test_gate_fails_when_a_negative_control_is_missing_from_rescore():
    """A silently-dropped negative control must not pass the gate by omission."""
    rows = gate.load_negative_control_corpus(_CORPUS_PATH)
    rescored = {row["task_id"]: 0.0 for row in rows}
    dropped = rows[-1]["task_id"]
    del rescored[dropped]

    report = gate.check_negative_controls_non_recovery(rescored, corpus=rows)
    assert report["passed"] is False
    assert dropped in {entry["task_id"] for entry in report["missing"]}


def test_loader_rejects_a_recovered_negative_control(tmp_path):
    """A corpus row that declares recovery (expected_recovered true) is rejected."""
    import json

    good = gate.load_negative_control_corpus(_CORPUS_PATH)[0]
    bad = dict(good)
    bad["expected_recovered"] = True
    bogus = tmp_path / "bad.jsonl"
    bogus.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected_recovered must be false"):
        gate.load_negative_control_corpus(bogus)
