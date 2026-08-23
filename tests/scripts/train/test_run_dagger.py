from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]


def _write(path: Path, text: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if executable:
        path.chmod(0o755)


def _stage_repo(
    tmp_path: Path,
    *,
    preflight: str = "pass",
    models: str = "pass",
) -> Path:
    root = tmp_path / "repo"
    train_dir = root / "scripts" / "train"
    utils_dir = train_dir / "utils"
    model_dir = train_dir / "models"
    bin_dir = root / "bin"
    train_dir.mkdir(parents=True)
    utils_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    bin_dir.mkdir(parents=True)

    shutil.copy(_ROOT / "scripts" / "train" / "run_dagger.sh", train_dir / "run_dagger.sh")
    shutil.copy(
        _ROOT / "scripts" / "train" / "utils" / "runtime_env.sh",
        utils_dir / "runtime_env.sh",
    )

    _write(utils_dir / "cleanup.sh", 'echo cleanup >> "$FAKE_TRACE"\n')
    if preflight == "fail":
        _write(
            utils_dir / "preflight.sh",
            'echo preflight >> "$FAKE_TRACE"\nexit 42\n',
        )
    else:
        _write(utils_dir / "preflight.sh", 'echo preflight >> "$FAKE_TRACE"\n')

    _write(
        utils_dir / "serve_teacher.sh",
        """
serve_teacher() {
  echo "serve_teacher $*" >> "$FAKE_TRACE"
  sleep 300 &
  TEACHER_PID=$!
  printf '%s\\n' "$TEACHER_PID" > "$FAKE_TEACHER_PID_FILE"
  TEACHER_URL="http://127.0.0.1:29999/generate"
  export TEACHER_PID TEACHER_URL
}
""".lstrip(),
    )

    if models == "fail":
        _write(
            utils_dir / "models.sh",
            'echo models >> "$FAKE_TRACE"\nexit 33\n',
        )
    else:
        _write(
            utils_dir / "models.sh",
            """
echo models >> "$FAKE_TRACE"
MODEL_ARGS_FILE=fake_model
MODEL_FAMILY=qwen3_vl
MODEL_SLUG="${MODEL_ID//\\//_}"
resolve_tp() {
  if [ -n "$1" ]; then echo "$1"; else echo "1"; fi
}
resolve_num_engines() {
  echo "$(( $2 / $1 ))"
}
""".lstrip(),
        )

    _write(
        root / "scripts" / "configs" / "qwen3_vl" / "recipes" / "dagger" / "lite.fake.yaml",
        "env_id: lite.fake\n",
    )
    _write(utils_dir / "nvlink.sh", "HAS_NVLINK=0\n")
    _write(utils_dir / "ckpt_args.sh", "CKPT_ARGS=()\n")
    _write(utils_dir / "ray.sh", "start_ray() { RAY_PORT=18080; }\n")
    _write(model_dir / "fake_model.sh", "MODEL_ARGS=(--fake-model-arg)\n")

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
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "ENV_ID": "lite.fake",
            "PROMPT_DATA": str(prompt),
            "CUA_LITE_ENV_SERVER_URL": "http://env-server.invalid:30100",
            "CUA_LITE_ENV_SERVER_TOKEN": "token-for-test",
            "SESSION_ID": "dagger-smoke-test",
            "HF_CKPT": str(hf_ckpt),
            "FAKE_TRACE": str(root / "trace.log"),
            "FAKE_TEACHER_PID_FILE": str(root / "teacher.pid"),
            "FAKE_RAY_CAPTURE": str(root / "ray.args"),
            "FAKE_RAY_EXIT": "44",
            "PATH": f"{root / 'bin'}:{env.get('PATH', '')}",
        }
    )
    return env


def _run_dagger(root: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(root / "scripts" / "train" / "run_dagger.sh")],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _assert_pid_stops(pid: int) -> None:
    try:
        for _ in range(20):
            if not _pid_alive(pid):
                return
            time.sleep(0.05)
        raise AssertionError(f"process {pid} is still alive")
    finally:
        if _pid_alive(pid):
            os.kill(pid, signal.SIGKILL)


def test_run_dagger_gpu_partition_failure_happens_before_preflight_or_teacher(
    tmp_path: Path,
) -> None:
    root = _stage_repo(tmp_path)
    env = _base_env(root)
    env.update({"CUDA_VISIBLE_DEVICES": "0", "NUM_TEACHER_GPUS": "1"})

    result = _run_dagger(root, env)

    assert result.returncode == 1
    assert "0 student GPUs" in result.stderr
    assert (root / "trace.log").read_text().splitlines() == ["cleanup"]
    assert not (root / "teacher.pid").exists()


def test_run_dagger_rejects_malformed_teacher_gpu_count_before_preflight(
    tmp_path: Path,
) -> None:
    root = _stage_repo(tmp_path)
    env = _base_env(root)
    env["NUM_TEACHER_GPUS"] = "abc"

    result = _run_dagger(root, env)

    assert result.returncode == 1
    assert "NUM_TEACHER_GPUS=abc must be a non-negative integer" in result.stderr
    assert (root / "trace.log").read_text().splitlines() == ["cleanup"]
    assert not (root / "teacher.pid").exists()


def test_run_dagger_requires_token_when_env_server_url_is_set(tmp_path: Path) -> None:
    root = _stage_repo(tmp_path)
    env = _base_env(root)
    env.pop("CUA_LITE_ENV_SERVER_TOKEN")

    result = _run_dagger(root, env)

    assert result.returncode == 1
    assert "CUA_LITE_ENV_SERVER_TOKEN is required" in result.stderr
    assert (root / "trace.log").read_text().splitlines() == ["cleanup"]
    assert not (root / "teacher.pid").exists()


def test_run_dagger_remote_teacher_url_skips_local_teacher(tmp_path: Path) -> None:
    root = _stage_repo(tmp_path)
    env = _base_env(root)
    env.update(
        {
            "DAGGER_TEACHER_URL": "http://remote-teacher.invalid/generate",
            "DUMP": "1",
        }
    )

    result = _run_dagger(root, env)

    assert result.returncode == 44
    trace = (root / "trace.log").read_text().splitlines()
    assert "serve_teacher --gpus 0" not in trace
    assert not (root / "teacher.pid").exists()

    ray_args = (root / "ray.args").read_text()
    assert '"DAGGER_TEACHER_URL": "http://remote-teacher.invalid/generate"' in ray_args
    assert "remote-teacher.invalid" not in result.stdout
    assert "teacher=remote-env" in result.stdout


def test_run_dagger_rejects_conflicting_remote_and_local_teacher(tmp_path: Path) -> None:
    root = _stage_repo(tmp_path)
    env = _base_env(root)
    env.update(
        {
            "DAGGER_TEACHER_URL": "http://remote-teacher.invalid/generate",
            "NUM_TEACHER_GPUS": "1",
        }
    )

    result = _run_dagger(root, env)

    assert result.returncode == 1
    assert "DAGGER_TEACHER_URL was provided" in result.stderr
    assert (root / "trace.log").read_text().splitlines() == ["cleanup"]
    assert not (root / "teacher.pid").exists()


def test_run_dagger_requires_teacher_source_when_local_teacher_disabled(
    tmp_path: Path,
) -> None:
    root = _stage_repo(tmp_path)
    env = _base_env(root)
    env["NUM_TEACHER_GPUS"] = "0"

    result = _run_dagger(root, env)

    assert result.returncode == 1
    assert "NUM_TEACHER_GPUS=0 requires DAGGER_TEACHER_URL or dagger.teacher_url" in result.stderr
    assert (root / "trace.log").read_text().splitlines() == ["cleanup", "models"]
    assert not (root / "teacher.pid").exists()


def test_run_dagger_rejects_sync_rollout_gpu_mismatch_before_preflight(
    tmp_path: Path,
) -> None:
    root = _stage_repo(tmp_path)
    env = _base_env(root)
    env["NUM_ROLLOUT_GPUS"] = "2"

    result = _run_dagger(root, env)

    assert result.returncode == 1
    assert "sync/colocate mode requires NUM_ROLLOUT_GPUS=NUM_TRAIN_GPUS" in result.stderr
    assert (root / "trace.log").read_text().splitlines() == ["cleanup"]
    assert not (root / "teacher.pid").exists()


def test_run_dagger_rejects_async_gpu_overcommit_before_preflight(
    tmp_path: Path,
) -> None:
    root = _stage_repo(tmp_path)
    env = _base_env(root)
    env.update({"CUDA_VISIBLE_DEVICES": "0,1,2,3", "ASYNC": "1"})

    result = _run_dagger(root, env)

    assert result.returncode == 1
    assert "ASYNC=1 DAgger needs NUM_TRAIN_GPUS+NUM_ROLLOUT_GPUS" in result.stderr
    assert (root / "trace.log").read_text().splitlines() == ["cleanup"]
    assert not (root / "teacher.pid").exists()


def test_run_dagger_rejects_student_gpu_count_above_remaining_before_preflight(
    tmp_path: Path,
) -> None:
    root = _stage_repo(tmp_path)
    env = _base_env(root)
    env.update({"CUDA_VISIBLE_DEVICES": "0,1", "NUM_TEACHER_GPUS": "1", "NUM_TRAIN_GPUS": "2"})

    result = _run_dagger(root, env)

    assert result.returncode == 1
    assert "NUM_TRAIN_GPUS=2 exceeds student-visible GPUs" in result.stderr
    assert (root / "trace.log").read_text().splitlines() == ["cleanup"]
    assert not (root / "teacher.pid").exists()


def test_run_dagger_env_preflight_failure_happens_before_teacher_start(tmp_path: Path) -> None:
    root = _stage_repo(tmp_path, preflight="fail")
    env = _base_env(root)

    result = _run_dagger(root, env)

    assert result.returncode == 42
    assert (root / "trace.log").read_text().splitlines() == [
        "cleanup",
        "models",
        "preflight",
    ]
    assert not (root / "teacher.pid").exists()


def test_run_dagger_missing_config_fails_before_preflight_or_teacher(tmp_path: Path) -> None:
    root = _stage_repo(tmp_path)
    config = root / "scripts" / "configs" / "qwen3_vl" / "recipes" / "dagger" / "lite.fake.yaml"
    config.unlink()
    env = _base_env(root)

    result = _run_dagger(root, env)

    assert result.returncode == 1
    assert "CONFIG_PATH does not exist" in result.stderr
    assert (root / "trace.log").read_text().splitlines() == ["cleanup", "models"]
    assert not (root / "teacher.pid").exists()


def test_run_dagger_grounding_env_fails_before_preflight_or_teacher(tmp_path: Path) -> None:
    root = _stage_repo(tmp_path)
    env = _base_env(root)
    env["ENV_ID"] = "screenspot_pro"

    result = _run_dagger(root, env)

    assert result.returncode == 2
    assert "does not support single-step grounding env 'screenspot_pro'" in result.stderr
    assert (root / "trace.log").read_text().splitlines() == ["cleanup"]
    assert not (root / "teacher.pid").exists()


def test_run_dagger_ray_failure_cleans_owned_teacher_and_builds_runtime_env(tmp_path: Path) -> None:
    root = _stage_repo(tmp_path)
    env = _base_env(root)
    config_path = root / "custom-dagger.yaml"
    config_path.write_text("env_id: lite.fake\n")
    env.update({"CONFIG_PATH": str(config_path), "DUMP": "1"})

    result = _run_dagger(root, env)

    assert result.returncode == 44
    trace = (root / "trace.log").read_text().splitlines()
    assert trace[:4] == [
        "cleanup",
        "models",
        "preflight",
        "serve_teacher --gpus 0",
    ]
    pid = int((root / "teacher.pid").read_text())
    _assert_pid_stops(pid)

    ray_args = (root / "ray.args").read_text()
    assert "--runtime-env-json=" in ray_args
    assert '"CUA_LITE_ENV_SERVER_URL": "http://env-server.invalid:30100"' in ray_args
    assert '"CUA_LITE_ENV_SERVER_TOKEN": "token-for-test"' in ray_args
    assert '"DAGGER_TEACHER_URL": "http://127.0.0.1:29999/generate"' in ray_args
    assert f"--custom-config-path\n{config_path}" in ray_args
    assert "--dump-details\n/tmp/cua-lite/dagger/Qwen_Qwen3-VL-2B-Instruct/lite.fake/" in ray_args
