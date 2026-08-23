"""Tests for lite.osworld cross-replay bridge helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


class TestCrossReplayBridge:
    """Static tests for the lite.osworld ↔ osworld bridge replay helper."""

    @staticmethod
    def _load_bridge_module():
        repo = Path(__file__).resolve().parents[5]
        path = repo / "devs/envs/lite.osworld/bridge/cross_replay.py"
        spec = importlib.util.spec_from_file_location(
            "_lite_osworld_cross_replay_under_test",
            path,
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_load_kept_turns_reads_nested_canonical_calls(self, tmp_path):
        from lite.core.tools import make_tool_call

        mod = self._load_bridge_module()
        skipped = tmp_path / "turn_0000"
        skipped.mkdir()
        (skipped / "03_actions.json").write_text(
            json.dumps({"lite_message": {"tool_calls": []}}),
            encoding="utf-8",
        )
        turn = tmp_path / "turn_0001"
        turn.mkdir()
        call = make_tool_call(
            "click",
            {"coordinate": [100, 200]},
            call_id="call_click",
        )
        (turn / "03_actions.json").write_text(
            json.dumps({"lite_message": {"tool_calls": [call]}}),
            encoding="utf-8",
        )
        (turn / "05_timing.json").write_text(
            json.dumps({"predict": 1.25}),
            encoding="utf-8",
        )

        assert mod._load_kept_turns(tmp_path) == [("turn_0001", [call], 1.25)]

    def test_load_kept_turns_rejects_legacy_flat_calls(self, tmp_path):
        mod = self._load_bridge_module()
        turn = tmp_path / "turn_0000"
        turn.mkdir()
        (turn / "03_actions.json").write_text(
            json.dumps(
                {
                    "lite_message": {
                        "tool_calls": [{"name": "click", "arguments": {}}],
                    },
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="bare model-function projection"):
            mod._load_kept_turns(tmp_path)

    def test_dry_run_prints_nested_call_names(self, tmp_path, monkeypatch, capsys):
        from lite.core.tools import make_tool_call

        mod = self._load_bridge_module()
        turn = tmp_path / "turn_0000"
        turn.mkdir()
        (turn / "03_actions.json").write_text(
            json.dumps(
                {
                    "lite_message": {
                        "tool_calls": [
                            make_tool_call(
                                "click",
                                {"coordinate": [100, 200]},
                                call_id="call_click",
                            )
                        ],
                    },
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            mod,
            "pair_for",
            lambda _task_id: SimpleNamespace(
                lite_key="lite.osworld@test",
                osworld_key="osworld@test",
                uuid="uuid",
                domain="chrome",
                exclude_reason=None,
            ),
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "cross_replay.py",
                "task-id",
                "--lite-rollout",
                str(tmp_path),
                "--dry-run",
            ],
        )

        mod._main()

        assert "first=['click']" in capsys.readouterr().out
