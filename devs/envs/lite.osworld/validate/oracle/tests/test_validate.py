"""Tests for lite.osworld oracle validator helpers."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[6]

class TestOracleValidator:
    """Guards for the dev oracle validator itself."""

    def test_after_postconfig_oracle_does_not_mark_task_metadata_done(self):
        """The validator may run postconfig before oracle replay, but final
        evaluation must still take the production path and run postconfig again.

        Regression caught here: writing ``_postconfig_done`` into the task
        metadata made sheet_print oracles remove CSV sidecars and then skip the
        final CSV export, causing false reward=0 failures.
        """
        src = (_REPO / "devs/envs/lite.osworld/validate/oracle/validate.py").read_text()
        assert 'evaluator = copy.deepcopy(meta.get("evaluator", {}))' in src
        oracle_block = src.split("if oracle_actions:", 1)[1].split(
            "# Kill apps that hold files the oracle needs to modify.", 1
        )[0]
        assert "_postconfig_done" not in oracle_block
