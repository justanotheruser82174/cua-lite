"""Tests for rollout summary CI gates and debug log checks.

Validates the opt-in CI assertions wired into the rollout entrypoints
(``scripts/rollout.py``) via
``--min-valid-frac``, ``--min-mean-return``, and ``--debug``.

Run::

    uv run pytest tests/infer/rollout/test_rollout_ci_gates.py -v
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from lite.core.tools.calls import (
    make_tool_call,
)
from lite.infer.rollout import (
    check_ci_gates,
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


class TestCheckCiGates:
    def test_no_gates_set_returns_empty(self, tmp_path: Path):
        p = _write_summary(
            tmp_path,
            {"num_samples": 10, "num_valid": 5, "mean_episode_return": 0.5},
        )
        assert check_ci_gates(p) == []

    def test_missing_summary_reports(self, tmp_path: Path):
        reasons = check_ci_gates(
            tmp_path / "nope.json", min_valid_frac=1.0,
        )
        assert len(reasons) == 1
        assert "summary.json not found" in reasons[0]

    def test_valid_frac_pass(self, tmp_path: Path):
        p = _write_summary(
            tmp_path,
            {"num_samples": 10, "num_valid": 10, "mean_episode_return": 0.5},
        )
        assert check_ci_gates(p, min_valid_frac=1.0) == []
        assert check_ci_gates(p, min_valid_frac=0.5) == []

    def test_valid_frac_fail(self, tmp_path: Path):
        p = _write_summary(
            tmp_path,
            {"num_samples": 10, "num_valid": 6, "mean_episode_return": 0.5},
        )
        reasons = check_ci_gates(p, min_valid_frac=0.8)
        assert len(reasons) == 1
        assert "valid fraction" in reasons[0]
        assert "0.600" in reasons[0]
        assert "0.800" in reasons[0]

    def test_valid_frac_denominator_is_num_samples_not_num_tasks(self, tmp_path: Path):
        # Grouped rollout: num_samples = num_tasks * group_size > num_tasks.
        # The gate must use num_samples as the denominator, not num_tasks —
        # falling back to num_tasks would deflate the denominator and could
        # pass a run that should fail.
        p = _write_summary(tmp_path, {
            "num_tasks": 5, "num_samples": 10, "num_valid": 10,
            "mean_episode_return": 0.5,
        })
        assert check_ci_gates(p, min_valid_frac=1.0) == []

    def test_valid_frac_zero_samples(self, tmp_path: Path):
        p = _write_summary(
            tmp_path,
            {"num_samples": 0, "num_valid": 0, "mean_episode_return": 0.0},
        )
        reasons = check_ci_gates(p, min_valid_frac=1.0)
        # 0/0 → frac=0.0 → fails any positive threshold
        assert len(reasons) == 1

    def test_valid_frac_missing_num_samples_fails_loud(self, tmp_path: Path):
        # A summary without num_samples (e.g. hand-written or pre-migration)
        # must not silently pass by falling back to a smaller denominator
        # such as num_tasks — it must fail loud instead.
        p = _write_summary(
            tmp_path,
            {"num_tasks": 10, "num_valid": 10, "mean_episode_return": 0.5},
        )
        reasons = check_ci_gates(p, min_valid_frac=1.0)
        assert len(reasons) == 1
        assert "0/0" in reasons[0]

    def test_mean_return_pass(self, tmp_path: Path):
        p = _write_summary(
            tmp_path,
            {"num_samples": 10, "num_valid": 10, "mean_episode_return": 0.7},
        )
        assert check_ci_gates(p, min_mean_return=0.5) == []
        assert check_ci_gates(p, min_mean_return=0.7) == []

    def test_mean_return_fail(self, tmp_path: Path):
        p = _write_summary(
            tmp_path,
            {"num_samples": 10, "num_valid": 10, "mean_episode_return": 0.3},
        )
        reasons = check_ci_gates(p, min_mean_return=0.5)
        assert len(reasons) == 1
        assert "mean_episode_return" in reasons[0]
        assert "0.3000" in reasons[0]
        assert "0.5000" in reasons[0]

    def test_both_gates_independent(self, tmp_path: Path):
        # Both fail → two reasons. AND semantics.
        p = _write_summary(
            tmp_path,
            {"num_samples": 10, "num_valid": 4, "mean_episode_return": 0.1},
        )
        reasons = check_ci_gates(p, min_valid_frac=0.8, min_mean_return=0.5)
        assert len(reasons) == 2

    def test_both_gates_pass(self, tmp_path: Path):
        p = _write_summary(
            tmp_path,
            {"num_samples": 10, "num_valid": 10, "mean_episode_return": 0.9},
        )
        assert check_ci_gates(p, min_valid_frac=0.8, min_mean_return=0.5) == []


def test_rollout_parser_exposes_debug_gate():
    from lite.infer.rollout import make_rollout_parser

    default_args = make_rollout_parser().parse_args(["--env-id", "lite.demo"])
    enabled_args = make_rollout_parser().parse_args([
        "--env-id", "lite.demo",
        "--debug",
    ])
    disabled_args = make_rollout_parser().parse_args([
        "--env-id", "lite.demo",
        "--debug", "false",
    ])

    assert default_args.debug is False
    assert enabled_args.debug is True
    assert disabled_args.debug is False


class _RolloutDebugFakeEnv:
    async def close(self) -> None:
        pass


class _RolloutDebugFakeAgent:
    async def sample(self, env, hooks=()):
        from lite.agents.core.agent.hooks import SampleStepData
        from lite.agents.types import PredictResult
        from lite.core import LiteCUAMetadata, LiteSample
        from lite.core.samples import LiteRLSample, LiteRLStep
        from lite.core.tools.results import LiteToolResult
        from lite.gym.types import LiteEnvStepResult

        image = Image.new("RGB", (100, 80), "white")
        actions = [
            make_tool_call(
                "click",
                {"coordinate": [500, 250], "button": "left"},
                call_id="call_0",
            )
        ]
        step = LiteRLStep(
            prompt="prompt text",
            image_indices=(0,),
            response="raw response",
            reward=1.0,
        )
        for hook in hooks:
            hook.on_step(SampleStepData(
                step_idx=0,
                image=image,
                predict_result=PredictResult(
                    lite_message={
                        "role": "assistant",
                        "content": [],
                        "tool_calls": actions,
                    },
                    agent_message={"role": "assistant", "content": "agent"},
                    step=step,
                ),
                step_result=LiteEnvStepResult(
                    results=[LiteToolResult(tool_call_id="call_0", text="ok")],
                    reward=1.0,
                    terminated=True,
                    truncated=False,
                    info={
                        "executed_actions": [
                            {
                                "name": "click",
                                "arguments": {"coordinate": [50, 20]},
                            },
                        ],
                    },
                ),
                actions=actions,
            ))

        sample = LiteRLSample(
            lite_sample=LiteSample(
                metadata=LiteCUAMetadata(
                    dims=("desktop", "use"),
                    others={"env_id": "lite.demo", "task_id": "debug_task"},
                ),
                images=[image],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "index": 0},
                            {"type": "text", "text": "task"},
                        ],
                    },
                    {"role": "assistant", "content": [], "tool_calls": actions},
                    {"role": "tool", "tool_call_id": "call_0", "content": _text("ok")},
                    {"role": "assistant", "content": _text("Done.")},
                ],
            ),
            processed_images=[image],
            steps=[step],
            episode_return=1.0,
            terminated=True,
            truncated=False,
        )
        for hook in hooks:
            hook.on_complete(sample)
        return sample


def _run_rollout_debug_artifact_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    debug: bool,
    save_gif: bool = False,
    save_video: bool = False,
) -> Path:
    import lite.agents.factory as agent_factory
    from lite.infer import rollout as rollout_module

    monkeypatch.setattr(
        rollout_module.gym,
        "make",
        lambda *args, **kwargs: _RolloutDebugFakeEnv(),
    )
    monkeypatch.setattr(
        agent_factory,
        "make",
        lambda *args, **kwargs: _RolloutDebugFakeAgent(),
    )

    all_done, log_root = asyncio.run(
        rollout_module.run_rollout(
            model_id="gpt-5.5",
            model_path="gpt-5.5",
            env_id="lite.demo",
            agent_kwargs={},
            env_kwargs={},
            seed=1,
            concurrency=1,
            log_root=tmp_path / f"debug-{debug}-gif-{save_gif}-video-{save_video}",
            task_id="debug_task",
            save_gif=save_gif,
            save_video=save_video,
            render_instruction_banner=False,
            debug=debug,
        )
    )

    assert all_done is True
    return log_root / "task" / "debug_task" / "sample_00"


def test_rollout_debug_false_keeps_prompt_image_artifacts_off_with_gif(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    sample_dir = _run_rollout_debug_artifact_case(
        monkeypatch,
        tmp_path,
        debug=False,
        save_gif=True,
    )
    turn_dir = sample_dir / "turn_0000"

    assert (sample_dir / "trajectory.gif").is_file()
    assert not (sample_dir / "trajectory.mp4").exists()
    assert not (turn_dir / "prompt_images").exists()
    assert not (turn_dir / "prompt_images_annotated").exists()


def test_rollout_debug_true_emits_prompt_image_artifacts_without_media(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    sample_dir = _run_rollout_debug_artifact_case(
        monkeypatch,
        tmp_path,
        debug=True,
    )
    turn_dir = sample_dir / "turn_0000"

    assert (turn_dir / "prompt_images" / "0000_reset.png").is_file()
    assert (turn_dir / "prompt_images_annotated" / "0000_reset.png").is_file()
    assert not (sample_dir / "trajectory.gif").exists()
    assert not (sample_dir / "trajectory.mp4").exists()

