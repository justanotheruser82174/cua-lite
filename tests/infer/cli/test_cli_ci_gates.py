"""Tests for rollout summary CI gates and debug log checks.

Validates the opt-in CI assertions wired into the rollout entrypoints
(``scripts/rollout.py``) via
``--min-valid-frac``, ``--min-mean-return``, and ``--debug``.

Run::

    uv run pytest tests/infer/cli/test_cli_ci_gates.py -v
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from lite.core.tools.calls import (
    make_tool_call,
)
from lite.utils.parquet import write_records_to_parquet


def _write_summary(tmp_path: Path, stats: dict) -> Path:
    p = tmp_path / "summary.json"
    p.write_text(json.dumps({"config": {}, "stats": stats, "tasks": []}))
    return p


def _metadata(
    *,
    task_type: str = "use",
    platform: str = "desktop",
    extra_tool_schemas: list[dict] | None = None,
    terminated: bool | None = None,
    truncated: bool | None = None,
) -> dict:
    metadata = {
        "metadata_kind": "cua",
        "dims": [platform, task_type],
        "extra_tool_schemas": extra_tool_schemas or [],
        "valid_actions": None,
        "others": {
            "env_id": "test.env",
            "task_id": "task_0",
        },
    }
    if terminated is not None:
        metadata["others"]["terminated"] = terminated
    if truncated is not None:
        metadata["others"]["truncated"] = truncated
    return metadata


def _write_trajectory(
    sample_dir: Path,
    messages: list[dict],
    *,
    metadata: dict | None = None,
    images: list[str] | None = None,
    json_fields: tuple[str, ...] | None = None,
) -> Path:
    sample_dir.mkdir(parents=True, exist_ok=True)
    path = sample_dir / "trajectory.parquet"
    write_records_to_parquet([{
        "images": images or [],
        "messages": messages,
        "metadata": metadata or _metadata(),
    }], path, json_fields=json_fields)
    return path


def _text(text: str) -> list[dict]:
    return [{"type": "text", "text": text}]


def _computer_call(call_id: str = "call_0") -> dict:
    return make_tool_call(
        "computer",
        {"actions": [{"action": "click", "coordinate": [10, 20]}]},
        call_id=call_id,
    )


def _messages_with_tool_result(tool_text: str, *, call_id: str = "call_0") -> list[dict]:
    return [
        {"role": "user", "content": _text("task")},
        {"role": "assistant", "content": [], "tool_calls": [_computer_call(call_id)]},
        {"role": "tool", "tool_call_id": call_id, "content": _text(tool_text)},
        {"role": "assistant", "content": _text("Done.")},
    ]


def _write_turn_debug_payload(
    sample_dir: Path,
    *,
    call_id: str = "call_0",
    tool_name: str = "computer",
    arguments: dict | None = None,
    text: str | None = "obs",
    error: str | None = "bad",
) -> Path:
    turn = sample_dir / "turn_0000"
    turn.mkdir(parents=True, exist_ok=True)
    (turn / "01_prompt.txt").write_text("prompt")
    (turn / "02_response.txt").write_text("response")
    if arguments is None:
        arguments = {"actions": [{"action": "click", "coordinate": [10, 20]}]}
    (turn / "03_actions.json").write_text(json.dumps({
        "lite_message": {
            "role": "assistant",
            "content": [],
            "tool_calls": [make_tool_call(tool_name, arguments, call_id=call_id)],
        },
    }))
    result = {
        "tool_call_id": call_id,
        "images": [],
        "text": text,
        "metadata": None,
        "error": error,
    }
    (turn / "04_results.json").write_text(json.dumps({
        "reward": 0.0,
        "terminated": True,
        "truncated": False,
        "results": [result],
        "info": {},
    }))
    return turn


def _write_empty_turn_debug_payload(sample_dir: Path) -> Path:
    turn = sample_dir / "turn_0000"
    turn.mkdir(parents=True, exist_ok=True)
    (turn / "01_prompt.txt").write_text("prompt")
    (turn / "02_response.txt").write_text("unparseable output")
    (turn / "03_actions.json").write_text(json.dumps({"lite_message": {}}))
    (turn / "04_results.json").write_text(json.dumps({
        "reward": 0.0,
        "terminated": True,
        "truncated": False,
        "results": [],
        "info": {},
    }))
    return turn


def _write_result_image(
    turn: Path,
    name: str = "0000_0000_from_call_0.png",
    data: bytes = b"png",
) -> dict:
    path = turn / "env_result_images" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "path": str(Path("env_result_images") / name),
        "source": "env_result_images",
        "bytes": len(data),
        "sha1": hashlib.sha1(data).hexdigest(),
    }


def _params() -> dict:
    return {
        "type": "object",
        "properties": {
            "app_name": {"type": "string"},
            "command": {"type": "string"},
            "coordinate": {"type": "array"},
            "reason": {"type": "string"},
            "status": {"type": "string"},
            "text": {"type": "string"},
            "url": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": [],
    }


_PROJECTED_NATIVE_MISMATCH = "does not match projected native text/error"


def test_rollout_cli_runs_log_contract_gate(monkeypatch, tmp_path: Path):
    from argparse import Namespace

    from lite.infer import cli

    rollout_debug_values: list[bool] = []

    async def fake_run_rollout(**kwargs):
        rollout_debug_values.append(kwargs["debug"])
        return True, tmp_path

    checked_roots: list[Path] = []

    def fake_check_log_contract(log_root: Path) -> list[str]:
        checked_roots.append(log_root)
        return [f"{log_root}: contract drift"]

    fake_debug_module = types.ModuleType("lite.infer.debug.log_contract")
    fake_debug_module.check_log_contract = fake_check_log_contract

    args = Namespace(
        model_id="gpt-5.5",
        model_path=None,
        env_id="lite.demo",
        env_kwargs={},
        seed=42,
        group_size=1,
        concurrency=1,
        log_root=None,
        prompt_data=None,
        splits=None,
        head=None,
        sample=None,
        filter_expr=None,
        task_id=None,
        config_path=None,
        save_data=True,
        save_video=False,
        save_gif=False,
        render_instruction_banner=True,
        group_shared_seed=True,
        max_attempts=1,
        min_valid_frac=None,
        min_mean_return=None,
        debug=True,
    )
    monkeypatch.setattr(cli, "run_rollout", fake_run_rollout)
    monkeypatch.setitem(sys.modules, "lite.infer.debug.log_contract", fake_debug_module)

    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(cli._rollout_with_retries(args, {}))

    assert exc_info.value.code == 1
    assert checked_roots == [tmp_path]
    assert rollout_debug_values == [True]


def test_rollout_cli_scopes_log_contract_gate_to_selected_splits(monkeypatch, tmp_path: Path):
    from argparse import Namespace

    from lite.infer import cli

    async def fake_run_rollout(**_kwargs):
        return True, tmp_path

    checked_roots: list[Path] = []

    def fake_check_log_contract(log_root: Path) -> list[str]:
        checked_roots.append(log_root)
        return []

    fake_debug_module = types.ModuleType("lite.infer.debug.log_contract")
    fake_debug_module.check_log_contract = fake_check_log_contract

    args = Namespace(
        model_id="gpt-5.5",
        model_path=None,
        env_id="lite.demo",
        env_kwargs={},
        seed=42,
        group_size=1,
        concurrency=1,
        log_root=None,
        prompt_data=None,
        splits=["train.synth", "train.perturb"],
        head=None,
        sample=None,
        filter_expr=None,
        task_id=None,
        config_path=None,
        save_data=True,
        save_video=False,
        save_gif=False,
        render_instruction_banner=True,
        group_shared_seed=True,
        max_attempts=1,
        min_valid_frac=None,
        min_mean_return=None,
        debug=True,
    )
    monkeypatch.setattr(cli, "run_rollout", fake_run_rollout)
    monkeypatch.setitem(sys.modules, "lite.infer.debug.log_contract", fake_debug_module)

    asyncio.run(cli._rollout_with_retries(args, {}))

    assert checked_roots == [tmp_path / "train.synth", tmp_path / "train.perturb"]

