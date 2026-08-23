from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_scalecua_oracle_validate_module():
    path = Path(__file__).resolve().parents[1] / "validate.py"
    spec = importlib.util.spec_from_file_location("scalecua_oracle_validate", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_scalecua_oracle_validate_positive_int_rejects_zero():
    validate = _load_scalecua_oracle_validate_module()

    assert validate._positive_int("1") == 1
    with pytest.raises(argparse.ArgumentTypeError):
        validate._positive_int("0")
    with pytest.raises(argparse.ArgumentTypeError):
        validate._positive_int("-1")


@pytest.mark.asyncio
async def test_scalecua_oracle_validate_amain_rejects_zero_retries(tmp_path):
    validate = _load_scalecua_oracle_validate_module()
    args = SimpleNamespace(
        fixtures=tmp_path / "fixtures.jsonl",
        artifacts=tmp_path / "artifacts",
        filter=None,
        limit=None,
        concurrency=1,
        retries=0,
        reset_timeout=1.0,
        oracle_timeout=1.0,
        report=None,
        resume_from=None,
        session_id="unit_test_session",
        require_rl_flush_fired=False,
    )

    with pytest.raises(ValueError, match="must be >= 1"):
        await validate._amain(args)


def test_scalecua_oracle_validate_summary_counts_weak_gate4_rl_flush_only():
    validate = _load_scalecua_oracle_validate_module()
    summary = validate._build_summary(
        [
            {"passed": True, "replay": {"split": "rl", "flush_fired_counts": {"vlc": 1}}},
            {
                "passed": True,
                "replay": {"split": "rl", "flush_fired_counts": {"thunderbird": 1}},
            },
            {
                "passed": True,
                "replay": {"split": "train", "flush_fired_counts": {"thunderbird": 5}},
            },
            {"passed": True, "replay": {"split": "rl", "flush_fired_counts": {}}},
        ],
        require_rl_flush_fired=True,
    )

    assert summary["total"] == 4
    assert summary["failed"] == 0
    assert summary["flush_fired_counts"] == {"thunderbird": 6, "vlc": 1}
    assert summary["weak_gate_4"] == {
        "required": True,
        "passed": True,
        "missing": [],
        "rl_vlc_flush_fired": 1,
        "rl_thunderbird_flush_fired": 1,
    }
    missing = validate._build_summary(
        [{"passed": True, "replay": {"split": "rl", "flush_fired_counts": {"vlc": 1}}}],
        require_rl_flush_fired=True,
    )
    assert missing["weak_gate_4"]["passed"] is False
    assert missing["weak_gate_4"]["missing"] == ["rl_thunderbird_flush_fired"]
    assert validate._summary_failed(missing) is True
    train_only_does_not_satisfy_rl_gate = validate._build_summary(
        [
            {"passed": True, "replay": {"split": "rl", "flush_fired_counts": {"vlc": 1}}},
            {
                "passed": True,
                "replay": {"split": "train", "flush_fired_counts": {"thunderbird": 5, "vlc": 5}},
            },
        ],
        require_rl_flush_fired=True,
    )
    assert train_only_does_not_satisfy_rl_gate["weak_gate_4"]["passed"] is False
    assert train_only_does_not_satisfy_rl_gate["weak_gate_4"]["missing"] == [
        "rl_thunderbird_flush_fired"
    ]
    assert validate._summary_failed(train_only_does_not_satisfy_rl_gate) is True


@pytest.mark.asyncio
async def test_scalecua_oracle_validate_amain_fails_when_required_rl_flush_missing(
    tmp_path,
    monkeypatch,
):
    validate = _load_scalecua_oracle_validate_module()
    fixtures = [
        {
            "fixture_id": "rl_vlc",
            "task_id": "task_rl_vlc",
            "split": "rl",
            "oracle_actions": [{"type": "execute", "parameters": {"command": "true"}}],
        },
        {
            "fixture_id": "train_tb",
            "task_id": "task_train_tb",
            "split": "train",
            "oracle_actions": [{"type": "execute", "parameters": {"command": "true"}}],
        },
    ]

    async def fake_validate_one(fixture, *, artifacts_root, reset_timeout, oracle_timeout):
        counts = {"vlc": 1} if fixture["split"] == "rl" else {"thunderbird": 1, "vlc": 1}
        return {
            "fixture_id": fixture["fixture_id"],
            "task_id": fixture["task_id"],
            "split": fixture["split"],
            "passed": True,
            "replay": {
                "split": fixture["split"],
                "flush_fired_counts": counts,
            },
            "flush_fired_counts": counts,
        }

    monkeypatch.setattr(validate, "_load_fixtures", lambda path, wanted: fixtures)
    monkeypatch.setattr(validate, "_validate_one", fake_validate_one)
    monkeypatch.setattr(validate, "_sweep_own_containers", lambda session_id: 0)
    monkeypatch.setattr(validate, "_install_signal_cleanup", lambda session_id: None)
    args = SimpleNamespace(
        fixtures=tmp_path / "fixtures.jsonl",
        artifacts=tmp_path / "artifacts",
        filter=None,
        limit=None,
        concurrency=2,
        retries=1,
        reset_timeout=1.0,
        oracle_timeout=1.0,
        report=tmp_path / "report.jsonl",
        resume_from=None,
        session_id="unit_test_session",
        require_rl_flush_fired=True,
    )

    rc = await validate._amain(args)

    summary = json.loads((args.artifacts / "summary.json").read_text(encoding="utf-8"))
    assert rc == 1
    assert summary["failed"] == 0
    assert summary["weak_gate_4"]["passed"] is False
    assert summary["weak_gate_4"]["missing"] == ["rl_thunderbird_flush_fired"]

