from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]


def _write(path: Path, text: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if executable:
        path.chmod(0o755)


def _stage_repo(tmp_path: Path, *, config: bool = True) -> Path:
    root = tmp_path / "repo"
    train_dir = root / "scripts" / "train"
    utils_dir = train_dir / "utils"
    model_dir = train_dir / "models"
    bin_dir = root / "bin"
    train_dir.mkdir(parents=True)
    utils_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    bin_dir.mkdir(parents=True)

    shutil.copy(_ROOT / "scripts" / "train" / "run_grpo.sh", train_dir / "run_grpo.sh")
    shutil.copy(
        _ROOT / "scripts" / "train" / "utils" / "runtime_env.sh",
        utils_dir / "runtime_env.sh",
    )

    _write(utils_dir / "cleanup.sh", 'echo cleanup >> "$FAKE_TRACE"\n')
    _write(utils_dir / "preflight.sh", 'echo preflight >> "$FAKE_TRACE"\n')
    _write(utils_dir / "nvlink.sh", 'echo nvlink >> "$FAKE_TRACE"\nHAS_NVLINK=0\n')
    _write(
        utils_dir / "ckpt_args.sh",
        """
echo ckpt_args >> "$FAKE_TRACE"
CKPT_ARGS=(--fake-ckpt "$SAVE_DIR" "$SAVE_HF_DIR" "$HF_CKPT")
""".lstrip(),
    )
    _write(
        utils_dir / "models.sh",
        """
echo models >> "$FAKE_TRACE"
MODEL_ARGS_FILE=fake_model
MODEL_FAMILY=qwen3_vl
MODEL_SLUG=qwen3_vl_2b
resolve_tp() {
  if [ -n "$1" ]; then echo "$1"; else echo "1"; fi
}
resolve_num_engines() {
  if [ "$2" -lt "$1" ]; then
    echo "bad rollout engine shape" >&2
    exit 1
  fi
  echo "$(( $2 / $1 ))"
}
""".lstrip(),
    )
    _write(
        utils_dir / "ray.sh",
        'start_ray() { echo start_ray >> "$FAKE_TRACE"; RAY_PORT=18080; }\n',
    )
    _write(
        model_dir / "fake_model.sh",
        'echo model_args >> "$FAKE_TRACE"\nMODEL_ARGS=(--fake-model-arg)\n',
    )

    if config:
        _write(
            root / "scripts" / "configs" / "qwen3_vl" / "compact" / "lite.fake.yaml",
            "env_id: lite.fake\n",
        )

    _write(
        bin_dir / "ray",
        """
#!/usr/bin/env bash
printf '%s\\n' "$@" > "$FAKE_RAY_CAPTURE"
exit "${FAKE_RAY_EXIT:-44}"
""".lstrip(),
        executable=True,
    )
    return root


def _base_env(root: Path) -> dict[str, str]:
    hf_ckpt = root / "hf_ckpt"
    hf_ckpt.mkdir()
    prompt = root / "prompt.parquet"
    prompt.write_text("fake")
    parent_path = os.environ.get("PATH", "")
    pythonpath = os.environ.get("PYTHONPATH")
    env = {
        "ENV_ID": "lite.fake",
        "PROMPT_DATA": str(prompt),
        "NUM_TRAIN_GPUS": "2",
        "CUA_LITE_ENV_SERVER_URL": "http://env-server.invalid:30100",
        "CUA_LITE_ENV_SERVER_TOKEN": "token-for-test",
        "SESSION_ID": "grpo-smoke-test",
        "HF_CKPT": str(hf_ckpt),
        "FAKE_TRACE": str(root / "trace.log"),
        "FAKE_RAY_CAPTURE": str(root / "ray.args"),
        "FAKE_RAY_EXIT": "44",
        "PATH": f"{root / 'bin'}:{parent_path}",
    }
    if pythonpath is not None:
        env["PYTHONPATH"] = pythonpath
    return env


def _run_grpo(root: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(root / "scripts" / "train" / "run_grpo.sh")],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )


def test_run_grpo_reaches_ray_with_resolved_runtime_env_and_argv(tmp_path: Path) -> None:
    root = _stage_repo(tmp_path)
    env = _base_env(root)
    env.update({"DUMP": "1", "WANDB_API_KEY": "wandb-token", "NUM_ROLLOUT": "7"})
    env["CUA_LITE_REGISTRATION_MODULES"] = "examples.geo3k.registration"

    result = _run_grpo(root, env)

    assert result.returncode == 44
    assert (root / "trace.log").read_text().splitlines() == [
        "cleanup",
        "models",
        "preflight",
        "nvlink",
        "ckpt_args",
        "model_args",
        "start_ray",
    ]
    assert "token-for-test" not in result.stdout

    ray_args = (root / "ray.args").read_text()
    assert "--runtime-env-json=" in ray_args
    assert '"SESSION_ID": "grpo-smoke-test"' in ray_args
    assert '"CUA_LITE_ENV_SERVER_URL": "http://env-server.invalid:30100"' in ray_args
    assert '"CUA_LITE_ENV_SERVER_TOKEN": "token-for-test"' in ray_args
    assert '"CUA_LITE_REGISTRATION_MODULES": "examples.geo3k.registration"' in ray_args
    assert "--fake-model-arg" in ray_args
    assert "--custom-config-path\n" in ray_args
    assert "/scripts/configs/qwen3_vl/compact/lite.fake.yaml" in ray_args
    assert "--num-rollout\n7" in ray_args
    assert "--sglang-server-concurrency\n16" in ray_args
    assert "--use-wandb" in ray_args
    assert "--dump-details\n/tmp/cua-lite/grpo/qwen3_vl_2b/lite.fake/" in ray_args


def test_run_grpo_retired_tis_fails_before_preflight(tmp_path: Path) -> None:
    root = _stage_repo(tmp_path)
    env = _base_env(root)
    env["USE_TIS"] = "1"

    result = _run_grpo(root, env)

    assert result.returncode == 1
    assert "USE_TIS has been retired" in result.stderr
    assert (root / "trace.log").read_text().splitlines() == ["cleanup"]
    assert not (root / "ray.args").exists()


def test_run_grpo_requires_token_when_env_server_url_is_set(tmp_path: Path) -> None:
    root = _stage_repo(tmp_path)
    env = _base_env(root)
    env.pop("CUA_LITE_ENV_SERVER_TOKEN")

    result = _run_grpo(root, env)

    assert result.returncode == 1
    assert "CUA_LITE_ENV_SERVER_TOKEN is required" in result.stderr
    assert (root / "trace.log").read_text().splitlines() == ["cleanup"]
    assert not (root / "ray.args").exists()


def test_run_grpo_rejects_sync_rollout_gpu_mismatch_before_preflight(tmp_path: Path) -> None:
    root = _stage_repo(tmp_path)
    env = _base_env(root)
    env["NUM_ROLLOUT_GPUS"] = "1"

    result = _run_grpo(root, env)

    assert result.returncode == 1
    assert "sync/colocate mode requires NUM_ROLLOUT_GPUS=NUM_TRAIN_GPUS" in result.stderr
    assert (root / "trace.log").read_text().splitlines() == ["cleanup"]
    assert not (root / "ray.args").exists()


def test_run_grpo_rejects_missing_default_config_before_ray(tmp_path: Path) -> None:
    root = _stage_repo(tmp_path, config=False)
    env = _base_env(root)

    result = _run_grpo(root, env)

    assert result.returncode == 1
    assert "CONFIG_PATH does not exist" in result.stderr
    assert (root / "trace.log").read_text().splitlines() == [
        "cleanup",
        "models",
    ]
    assert not (root / "ray.args").exists()


def test_run_grpo_routes_the_rollout_module_through_a_knob() -> None:
    """The canonical GRPO launcher exposes the rollout module as a wrapper knob."""
    text = (_ROOT / "scripts" / "train" / "run_grpo.sh").read_text()
    assert "ROLLOUT_MODULE=${ROLLOUT_MODULE:-lite.train.rollout.grpo}" in text
    for entry in ("generate", "convert_samples_to_train_data", "generate_rollout"):
        assert f'"${{ROLLOUT_MODULE}}.{entry}"' in text
