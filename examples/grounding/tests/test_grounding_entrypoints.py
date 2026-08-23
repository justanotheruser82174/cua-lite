from __future__ import annotations

import importlib
import os
import runpy
import subprocess
import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]

_OLD_CONFIG_ARG = (
    '--custom-config-path "'
    "${CONFIG_PATH:-${CUA_LITE_ROOT}/scripts/configs/"
    '${MODEL_FAMILY}/compact/${ENV_ID}.yaml}"'
)


def _write(path: Path, text: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if executable:
        path.chmod(0o755)


def _canonical_grpo_text(*, parameterised: bool = True) -> str:
    """A stand-in for the canonical GRPO launcher used by the wrapper tests."""
    # These are interpolated INTO the f-string below, so their braces are literal.
    module_ref = "${ROLLOUT_MODULE}" if parameterised else "lite.train.rollout.grpo"
    knob = (
        "ROLLOUT_MODULE=${ROLLOUT_MODULE:-lite.train.rollout.grpo}\n"
        if parameterised
        else ""
    )
    return f"""
#!/usr/bin/env bash
set -euo pipefail
CUA_LITE_ROOT="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")/../.." &>/dev/null && pwd)"
MODEL_FAMILY="${{MODEL_FAMILY:-qwen3_vl}}"
RUN_REL="${{RUN_REL:-fake/run}}"
source "$(dirname "${{BASH_SOURCE[0]}}")/utils/cleanup.sh"
source "$(dirname "${{BASH_SOURCE[0]}}")/utils/ray.sh"
{knob}ROLLOUT_ARGS=(
  --custom-generate-function-path "{module_ref}.generate"
  --custom-convert-samples-to-train-data-path "{module_ref}.convert_samples_to_train_data"
  --rollout-function-path "{module_ref}.generate_rollout"
  {_OLD_CONFIG_ARG}
)
WANDB_ARGS=(--wandb-group "${{WANDB_GROUP_OVERRIDE:-${{RUN_REL}}${{WANDB_GROUP_SUFFIX:-}}}}")
printf 'root=%s\\n' "$CUA_LITE_ROOT" > "$FAKE_CAPTURE"
printf 'model_id=%s\\n' "$MODEL_ID" >> "$FAKE_CAPTURE"
printf 'train_gpus=%s\\n' "$NUM_TRAIN_GPUS" >> "$FAKE_CAPTURE"
printf 'rollout_gpus=%s\\n' "$NUM_ROLLOUT_GPUS" >> "$FAKE_CAPTURE"
printf 'rollout_batch_size=%s\\n' "$ROLLOUT_BATCH_SIZE" >> "$FAKE_CAPTURE"
printf 'n_samples_per_prompt=%s\\n' "$N_SAMPLES_PER_PROMPT" >> "$FAKE_CAPTURE"
printf 'num_steps_per_rollout=%s\\n' "$NUM_STEPS_PER_ROLLOUT" >> "$FAKE_CAPTURE"
printf 'env_concurrency=%s\\n' "$ENV_CONCURRENCY" >> "$FAKE_CAPTURE"
printf 'env_server_url=%s\\n' "${{CUA_LITE_ENV_SERVER_URL:-unset}}" >> "$FAKE_CAPTURE"
printf 'env_server_token=%s\\n' "${{CUA_LITE_ENV_SERVER_TOKEN:-unset}}" >> "$FAKE_CAPTURE"
printf '%s\\n' "${{ROLLOUT_ARGS[@]}}" "${{WANDB_ARGS[@]}}" >> "$FAKE_CAPTURE"
""".lstrip()


def _stage_regionfocus_root(tmp_path: Path, *, parameterised: bool = True) -> Path:
    root = tmp_path / "cua-lite"
    _write(
        root / "examples" / "grounding" / "scripts" / "run_grpo.sh",
        (_ROOT / "examples" / "grounding" / "scripts" / "run_grpo.sh").read_text(),
        executable=True,
    )
    _write(
        root / "scripts" / "train" / "run_grpo.sh",
        _canonical_grpo_text(parameterised=parameterised),
        executable=True,
    )
    _write(
        root / "scripts" / "train" / "utils" / "cleanup.sh",
        "echo cleanup >> \"$FAKE_TRACE\"\n",
    )
    _write(
        root / "scripts" / "train" / "utils" / "ray.sh",
        "echo ray >> \"$FAKE_TRACE\"\n",
    )
    _write(
        root / "scripts" / "train" / "utils" / "models.sh",
        'case "$MODEL_ID" in\n'
        '  Qwen/Qwen3.5*) MODEL_FAMILY=qwen3_5 ;;\n'
        '  *) MODEL_FAMILY=qwen3_vl ;;\n'
        "esac\n",
    )
    _write(
        root / "examples" / "grounding" / "configs" / "qwen3_vl" / "osworld_g.regionfocus.yaml",
        "agent_id: qwen3_vl.regionfocus\n",
    )
    _write(
        root / "examples" / "grounding" / "configs" / "qwen3_5" / "osworld_g.regionfocus.yaml",
        "agent_id: qwen3_5.regionfocus\n",
    )
    _write(root / "prompt.parquet", "fake prompt")
    return root


def _run_regionfocus_wrapper(
    root: Path,
    *,
    include_gpu_counts: bool = True,
    model_id: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "CUA_LITE_ROOT": str(root.parent / "stale-root"),
            "PROMPT_DATA": str(root / "prompt.parquet"),
            "FAKE_CAPTURE": str(root / "capture.txt"),
            "FAKE_TRACE": str(root / "trace.txt"),
            "CUA_LITE_ENV_SERVER_URL": "http://127.0.0.1:30100",
            "CUA_LITE_ENV_SERVER_TOKEN": "fake-token",
        }
    )
    if model_id is not None:
        env["MODEL_ID"] = model_id
    if include_gpu_counts:
        env.update({
            "NUM_TRAIN_GPUS": "2",
            "NUM_ROLLOUT_GPUS": "2",
        })
    else:
        env.pop("NUM_TRAIN_GPUS", None)
        env.pop("NUM_ROLLOUT_GPUS", None)
    return subprocess.run(
        ["bash", str(root / "examples" / "grounding" / "scripts" / "run_grpo.sh")],
        cwd=_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )


def test_regionfocus_rollout_forwards_debug_artifact_flags() -> None:
    text = (_ROOT / "examples" / "grounding" / "rollout.py").read_text()

    assert 'default="Qwen/Qwen3-VL-2B-Instruct"' in text
    assert "save_gif=args.save_gif" in text
    assert "debug=args.debug" in text
    assert "render_instruction_banner=args.render_instruction_banner" in text


def test_regionfocus_grpo_wrapper_execs_the_canonical_with_its_own_rollout_module(
    tmp_path: Path,
) -> None:
    """The wrapper contract is what reaches the canonical launcher."""
    root = _stage_regionfocus_root(tmp_path)

    result = _run_regionfocus_wrapper(root)

    assert result.returncode == 0, result.stderr
    capture = (root / "capture.txt").read_text()
    assert f"root={root}" in capture
    assert "model_id=Qwen/Qwen3-VL-2B-Instruct" in capture
    assert "train_gpus=2" in capture
    assert "rollout_gpus=2" in capture
    assert "env_server_url=unset" in capture
    assert "env_server_token=unset" in capture
    for entry in ("generate", "convert_samples_to_train_data", "generate_rollout"):
        assert f"examples.grounding.rollout_grpo.{entry}" in capture
    assert "lite.train.rollout.grpo." not in capture
    config_path = (
        root
        / "examples"
        / "grounding"
        / "configs"
        / "qwen3_vl"
        / "osworld_g.regionfocus.yaml"
    )
    assert str(config_path) in capture
    assert "--wandb-group\nfake/run_regionfocus" in capture
    assert (root / "trace.txt").read_text().splitlines() == ["cleanup", "ray"]


def test_regionfocus_grpo_wrapper_selects_qwen35_config_from_model_id(
    tmp_path: Path,
) -> None:
    root = _stage_regionfocus_root(tmp_path)

    result = _run_regionfocus_wrapper(root, model_id="Qwen/Qwen3.5-2B")

    assert result.returncode == 0, result.stderr
    capture = (root / "capture.txt").read_text()
    assert "model_id=Qwen/Qwen3.5-2B" in capture
    config_path = (
        root
        / "examples"
        / "grounding"
        / "configs"
        / "qwen3_5"
        / "osworld_g.regionfocus.yaml"
    )
    assert str(config_path) in capture


def test_regionfocus_grpo_wrapper_trains_the_wrong_module_if_canonical_drops_the_knob(
    tmp_path: Path,
) -> None:
    """The example wrapper depends on the canonical rollout-module knob."""
    root = _stage_regionfocus_root(tmp_path, parameterised=False)

    result = _run_regionfocus_wrapper(root)

    assert result.returncode == 0, result.stderr
    capture = (root / "capture.txt").read_text()
    assert "lite.train.rollout.grpo.generate" in capture
    assert "examples.grounding.rollout_grpo.generate" not in capture


def test_regionfocus_grpo_wrapper_requires_explicit_gpu_counts(tmp_path: Path) -> None:
    root = _stage_regionfocus_root(tmp_path)

    result = _run_regionfocus_wrapper(root, include_gpu_counts=False)

    assert result.returncode == 1
    assert "Grounding GRPO requires explicit GPU counts" in result.stderr


def test_regionfocus_training_entrypoints_import_without_slime() -> None:
    code = (
        "import examples.grounding.rollout_grpo\n"
        "import examples.grounding.rollout_reinforce\n"
        "print('ok')\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_regionfocus_reinforce_entrypoint_drains_and_logs_metrics(monkeypatch) -> None:
    calls: list[tuple[int, object, object, bool]] = []
    fake_reinforce = types.ModuleType("lite.train.rollout.reinforce")
    fake_reinforce.generate = object()
    fake_reinforce.convert_samples_to_train_data = object()

    def base_generate_rollout(args, rollout_id, data_source, evaluation=False):
        calls.append((rollout_id, args, data_source, evaluation))
        return "base-result"

    fake_reinforce.generate_rollout = base_generate_rollout
    monkeypatch.setitem(sys.modules, "lite.train.rollout.reinforce", fake_reinforce)
    sys.modules.pop("examples.grounding.rollout_reinforce", None)
    module = importlib.import_module("examples.grounding.rollout_reinforce")
    logged: list[tuple[object, bool]] = []
    monkeypatch.setattr(
        module.rf_metrics,
        "drain_and_log_after_rollout",
        lambda args, *, evaluation: logged.append((args, evaluation)),
    )

    args = object()
    result = module.generate_rollout(args, 7, "data", evaluation=True)

    assert result == "base-result"
    assert calls == [(7, args, "data", True)]
    assert logged == [(args, True)]


def test_regionfocus_local_rollout_entry_smoke_debug_false_true(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import transformers

    import lite.infer.rollout as rollout_mod
    import lite.infer.serving as serving_mod

    config_path = tmp_path / "regionfocus.yaml"
    config_path.write_text(
        "agent_kwargs:\n"
        "  judge_initial_point: false\n"
        "env_kwargs:\n"
        "  instruction_style: refined\n"
    )
    calls: list[dict] = []

    async def fake_run_rollout(**kwargs):
        assert "CUA_LITE_ENV_SERVER_URL" not in os.environ
        assert "CUA_LITE_ENV_SERVER_TOKEN" not in os.environ
        calls.append(kwargs)
        log_root = Path(kwargs["log_root"])
        log_root.mkdir(parents=True, exist_ok=True)
        (log_root / "summary.json").write_text(
            '{"num_tasks": 1, "num_valid": 1, "mean_episode_return": 1.0}'
        )
        sample_dir = log_root / "eval" / "task" / "sample_0000"
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "summary.json").write_text(
            '{"regionfocus": {"judge_records": []}}'
        )
        return True, log_root

    monkeypatch.setattr(rollout_mod, "run_rollout", fake_run_rollout)
    monkeypatch.setattr(
        serving_mod,
        "make_server_generate_fn",
        lambda url, sampling_kwargs: object(),
    )
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda model_path, trust_remote_code=True: object(),
    )
    monkeypatch.setenv("CUA_LITE_ENV_SERVER_URL", "http://127.0.0.1:30100")
    monkeypatch.setenv("CUA_LITE_ENV_SERVER_TOKEN", "fake-token")

    for debug_value, expected in (("false", False), ("true", True)):
        log_root = tmp_path / f"run-{debug_value}"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "examples.grounding.rollout",
                "--model-id",
                "Qwen/Qwen3-VL-2B-Instruct",
                "--env-id",
                "osworld_g",
                "--config-path",
                str(config_path),
                "--sglang-server-url",
                "http://127.0.0.1:1",
                "--log-root",
                str(log_root),
                "--head",
                "1",
                "--debug",
                debug_value,
                "--save-video",
                "false",
                "--save-gif",
                "true",
                "--render-instruction-banner",
                "false",
            ],
        )

        runpy.run_module("examples.grounding.rollout", run_name="__main__")

        call = calls[-1]
        assert call["debug"] is expected
        assert call["save_video"] is False
        assert call["save_gif"] is True
        assert call["render_instruction_banner"] is False
        assert call["config_path"] == str(config_path)
        assert call["agent_kwargs"]["judge_initial_point"] is False
        assert "processor" in call["agent_kwargs"]
        assert "generate_fn" in call["agent_kwargs"]
        assert call["env_kwargs"] == {}
        assert (log_root / "judge_summary.json").exists()
