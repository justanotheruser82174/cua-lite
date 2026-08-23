from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]


_COMMON_RAY_WORKER_ENV_KEYS = {
    "PYTHONPATH",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "NCCL_NVLS_ENABLE",
    "MEGATRON_CONFIG_LOCK_DIR",
    "SESSION_ID",
}


_SEGMENTER_WORKER_ENV_KEYS = {
    "CUA_LITE_DISABLE_RADIX",
    "CUA_LITE_MULTIMODAL_LAZY_EXPAND",
    "CUA_LITE_MULTIMODAL_FP32",
    "CUA_LITE_ROLLOUT_PROC_WORKERS",
}


_ONLINE_ROLLOUT_WORKER_ENV_KEYS = {
    *_COMMON_RAY_WORKER_ENV_KEYS,
    *_SEGMENTER_WORKER_ENV_KEYS,
    "CUA_LITE_ENV_SERVER_URL",
    "CUA_LITE_ENV_SERVER_TOKEN",
    "CUA_LITE_REGISTRATION_MODULES",
    "ROLLOUT_STALL_TIMEOUT_S",
}


def _script(name: str) -> str:
    return (_ROOT / "scripts" / "train" / name).read_text()


def _runtime_env_block(script_name: str) -> str:
    text = _script(script_name)
    start = text.index('RUNTIME_ENV_JSON="$(build_runtime_env_json')
    end = text.index('\n)"', start) + len('\n)"')
    return text[start:end]


def _runtime_env_keys(script_name: str) -> set[str]:
    block = _runtime_env_block(script_name)
    return set(re.findall(r'^\s+"([^"=]+)=', block, re.M))


def _ray_submit_arrays(script_name: str) -> list[str]:
    text = _script(script_name)
    block = text[text.index("ray job submit ") :]
    return re.findall(r'"\$\{([A-Z_]+)\[@\]\}"', block)


def _assert_markers_in_order(script_name: str, markers: list[str]) -> None:
    text = _script(script_name)
    positions = [text.index(marker) for marker in markers]
    assert positions == sorted(positions), script_name


def test_online_training_scripts_forward_rollout_worker_env_knobs() -> None:
    for script_name in ("run_grpo.sh", "run_reinforce.sh", "run_dagger.sh"):
        text = _script(script_name)
        block = _runtime_env_block(script_name)

        assert 'UTILS_DIR="${CUA_LITE_ROOT}/scripts/train/utils"' in text
        assert 'source "${UTILS_DIR}/runtime_env.sh"' in text
        assert '--runtime-env-json="${RUNTIME_ENV_JSON}"' in text
        assert 'RUNTIME_ENV_JSON="{\n' not in text
        missing = [key for key in _ONLINE_ROLLOUT_WORKER_ENV_KEYS if key not in block]
        assert not missing, f"{script_name} missing Ray runtime env keys: {missing}"


def test_online_training_scripts_use_shared_rollout_engine_guard() -> None:
    for script_name in ("run_grpo.sh", "run_reinforce.sh", "run_dagger.sh"):
        text = _script(script_name)

        assert (
            'NUM_ENGINES=$(resolve_num_engines "${GPUS_PER_ENGINE}" "${NUM_ROLLOUT_GPUS}")'
        ) in text
        assert "NUM_ENGINES=$(( ${NUM_ROLLOUT_GPUS} / GPUS_PER_ENGINE ))" not in text


def test_online_training_scripts_share_env_server_and_sync_preflights() -> None:
    for script_name in ("run_grpo.sh", "run_reinforce.sh", "run_dagger.sh"):
        text = _script(script_name)

        assert "sync/colocate mode requires NUM_ROLLOUT_GPUS=NUM_TRAIN_GPUS" in text
        assert "USE_TIS has been retired from this launcher" in text
        assert "CUA_LITE_ENV_SERVER_TOKEN is required when CUA_LITE_ENV_SERVER_URL is set" in text
        assert 'if [ -n "${CUA_LITE_ENV_SERVER_URL:-}" ]' in text


def test_online_training_scripts_share_rollout_module_and_wandb_suffix_knobs() -> None:
    expected_modules = {
        "run_grpo.sh": "lite.train.rollout.grpo",
        "run_reinforce.sh": "lite.train.rollout.reinforce",
        "run_dagger.sh": "lite.train.rollout.dagger",
    }
    for script_name, module in expected_modules.items():
        text = _script(script_name)

        assert f"ROLLOUT_MODULE=${{ROLLOUT_MODULE:-{module}}}" in text
        for entrypoint in ("generate", "convert_samples_to_train_data", "generate_rollout"):
            assert f'"${{ROLLOUT_MODULE}}.{entrypoint}"' in text
        assert "WANDB_GROUP=${WANDB_GROUP_OVERRIDE:-${RUN_REL}${WANDB_GROUP_SUFFIX:-}}" in text


@pytest.mark.parametrize(
    ("script_name", "expected_keys"),
    [
        ("run_grpo.sh", _ONLINE_ROLLOUT_WORKER_ENV_KEYS | {"DROP_ZERO_STD_GROUP"}),
        ("run_reinforce.sh", _ONLINE_ROLLOUT_WORKER_ENV_KEYS),
        ("run_dagger.sh", _ONLINE_ROLLOUT_WORKER_ENV_KEYS | {"DAGGER_TEACHER_URL"}),
        (
            "run_sft.sh",
            _COMMON_RAY_WORKER_ENV_KEYS | _SEGMENTER_WORKER_ENV_KEYS | {"PYTORCH_CUDA_ALLOC_CONF"},
        ),
    ],
)
def test_training_scripts_runtime_env_key_sets_are_exact(
    script_name: str, expected_keys: set[str]
) -> None:
    assert _runtime_env_keys(script_name) == expected_keys


def test_dagger_script_preflights_before_starting_owned_teacher() -> None:
    text = _script("run_dagger.sh")

    student_gpu_guard = '_require_positive_int "NUM_TRAIN_GPUS" "${NUM_TRAIN_GPUS}"'
    grounding_guard = "screenspot_pro|osworld_g)"
    models = 'source "${UTILS_DIR}/models.sh"'
    config_path = 'CONFIG_PATH=${CONFIG_PATH:-${DEFAULT_CONFIG_PATH}}'
    env_preflight = 'source "${UTILS_DIR}/preflight.sh"'
    teacher_start = 'serve_teacher --gpus "$(IFS=,; echo "${_GPU[*]:0:NUM_TEACHER_GPUS}")"'

    assert text.index(student_gpu_guard) < text.index(models)
    assert text.index(grounding_guard) < text.index(env_preflight)
    assert text.index(grounding_guard) < text.index(teacher_start)
    assert text.index(student_gpu_guard) < text.index(teacher_start)
    assert text.index(models) < text.index(config_path)
    assert text.index(config_path) < text.index(env_preflight)
    assert text.index(env_preflight) < text.index(teacher_start)
    assert "trap _cleanup_dagger_teacher EXIT" in text
    assert '_DAGGER_TEACHER_PID="${TEACHER_PID:-}"' in text


def test_dagger_script_forwards_teacher_url_and_debug_dump_knobs() -> None:
    text = _script("run_dagger.sh")
    block = _runtime_env_block("run_dagger.sh")

    assert '"DAGGER_TEACHER_URL=${DAGGER_TEACHER_URL:-}"' in block
    assert (
        'DEFAULT_CONFIG_PATH="${CUA_LITE_ROOT}/scripts/configs/${MODEL_FAMILY}/recipes/'
        'dagger/${ENV_ID}.yaml"'
    ) in text
    assert 'CONFIG_PATH=${CONFIG_PATH:-${DEFAULT_CONFIG_PATH}}' in text
    assert '--custom-config-path "${CONFIG_PATH}"' in text
    assert 'DUMP_ARGS=(--dump-details "${DUMP_DIR}")' in text
    assert '"${DUMP_ARGS[@]}"' in text


def test_sft_script_forwards_worker_env_for_local_source_segmenter() -> None:
    text = _script("run_sft.sh")
    block = _runtime_env_block("run_sft.sh")
    expected_keys = _COMMON_RAY_WORKER_ENV_KEYS | _SEGMENTER_WORKER_ENV_KEYS

    assert 'SLIME_DIR="${CUA_LITE_ROOT}/slime"' in text
    assert 'UTILS_DIR="${CUA_LITE_ROOT}/scripts/train/utils"' in text
    assert 'source "${UTILS_DIR}/runtime_env.sh"' in text
    assert '"PYTHONPATH=/root/Megatron-LM/:${CUA_LITE_ROOT}:${SLIME_DIR}"' in block
    assert '--runtime-env-json="${RUNTIME_ENV_JSON}"' in text
    assert 'RUNTIME_ENV_JSON="{\n' not in text
    missing = [key for key in expected_keys if key not in block]
    assert not missing, f"run_sft.sh missing Ray runtime env keys: {missing}"


def test_sft_script_pins_the_actual_offline_batch_and_backend_shape() -> None:
    text = _script("run_sft.sh")
    executable = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )

    assert "_N_TRAJ=${ROLLOUT_BATCH_SIZE}" in text
    assert "ROLLOUT_BATCH_SIZE*N_SAMPLES_PER_PROMPT" not in executable
    assert '_require_positive_int "GLOBAL_BATCH_SIZE" "${GLOBAL_BATCH_SIZE}"' in text
    assert '_require_positive_int "ROLLOUT_BATCH_SIZE" "${ROLLOUT_BATCH_SIZE}"' in text
    assert "PP_SIZE=${PP_SIZE:-1}" in text
    assert "CP_SIZE=${CP_SIZE:-1}" in text
    assert "--pipeline-model-parallel-size \"${PP_SIZE}\"" in text
    assert "--context-parallel-size \"${CP_SIZE}\"" in text


def test_sft_script_keeps_offline_data_and_eval_contracts() -> None:
    text = _script("run_sft.sh")

    for expected in (
        "--input-key steps",
        "--metadata-key processed_images",
        "--global-batch-size \"${GLOBAL_BATCH_SIZE}\"",
        "--debug-train-only",
        "EVAL_PROMPT_DATA is not supported by run_sft.sh",
    ):
        assert expected in text


def test_train_launchers_quote_final_ray_argv_arrays() -> None:
    for script_name in ("run_grpo.sh", "run_reinforce.sh", "run_dagger.sh", "run_sft.sh"):
        text = _script(script_name)
        submit = text[text.index("ray job submit ") :]

        assert '"${MODEL_ARGS[@]}"' in submit
        assert '"${CKPT_ARGS[@]}"' in submit
        assert '"${BACKEND_ARGS[@]}"' in submit
        assert "${MODEL_ARGS[@]}" not in submit.replace('"${MODEL_ARGS[@]}"', "")
        assert "${CKPT_ARGS[@]}" not in submit.replace('"${CKPT_ARGS[@]}"', "")
        assert "${BACKEND_ARGS[@]}" not in submit.replace('"${BACKEND_ARGS[@]}"', "")


@pytest.mark.parametrize(
    ("script_name", "expected_arrays"),
    [
        (
            "run_grpo.sh",
            [
                "MODEL_ARGS",
                "CKPT_ARGS",
                "ROLLOUT_ARGS",
                "EVAL_ARGS",
                "GRPO_ARGS",
                "OPTIMIZER_ARGS",
                "SGLANG_ARGS",
                "WANDB_ARGS",
                "BACKEND_ARGS",
                "MISC_ARGS",
                "DUMP_ARGS",
            ],
        ),
        (
            "run_reinforce.sh",
            [
                "MODEL_ARGS",
                "CKPT_ARGS",
                "ROLLOUT_ARGS",
                "EVAL_ARGS",
                "REINFORCE_ARGS",
                "OPTIMIZER_ARGS",
                "SGLANG_ARGS",
                "WANDB_ARGS",
                "BACKEND_ARGS",
                "MISC_ARGS",
                "DUMP_ARGS",
            ],
        ),
        (
            "run_dagger.sh",
            [
                "MODEL_ARGS",
                "CKPT_ARGS",
                "ROLLOUT_ARGS",
                "EVAL_ARGS",
                "SFT_LOSS_ARGS",
                "OPTIMIZER_ARGS",
                "SGLANG_ARGS",
                "WANDB_ARGS",
                "BACKEND_ARGS",
                "MISC_ARGS",
                "DUMP_ARGS",
            ],
        ),
        (
            "run_sft.sh",
            [
                "MODEL_ARGS",
                "CKPT_ARGS",
                "SFT_ARGS",
                "EVAL_ARGS",
                "OPTIMIZER_ARGS",
                "WANDB_ARGS",
                "BACKEND_ARGS",
                "DUMP_ARGS",
            ],
        ),
    ],
)
def test_train_launchers_keep_final_ray_argv_array_order(
    script_name: str, expected_arrays: list[str]
) -> None:
    assert _ray_submit_arrays(script_name) == expected_arrays


@pytest.mark.parametrize("script_name", ["run_grpo.sh", "run_reinforce.sh"])
def test_online_training_launcher_service_order(script_name: str) -> None:
    _assert_markers_in_order(
        script_name,
        [
            'source "${UTILS_DIR}/cleanup.sh"',
            'source "${UTILS_DIR}/models.sh"',
            'source "${UTILS_DIR}/preflight.sh"',
            'source "${UTILS_DIR}/nvlink.sh"',
            'source "${UTILS_DIR}/ckpt_args.sh"',
            'source "${CUA_LITE_ROOT}/scripts/train/models/${MODEL_ARGS_FILE}.sh"',
            'source "${UTILS_DIR}/ray.sh"',
            'source "${UTILS_DIR}/runtime_env.sh"',
            "start_ray",
            'RUNTIME_ENV_JSON="$(build_runtime_env_json',
            "ray job submit ",
        ],
    )


def test_dagger_training_launcher_service_order() -> None:
    _assert_markers_in_order(
        "run_dagger.sh",
        [
            'source "${UTILS_DIR}/cleanup.sh"',
            'source "${UTILS_DIR}/models.sh"',
            'CONFIG_PATH=${CONFIG_PATH:-${DEFAULT_CONFIG_PATH}}',
            'source "${UTILS_DIR}/preflight.sh"',
            'serve_teacher --gpus "$(IFS=,; echo "${_GPU[*]:0:NUM_TEACHER_GPUS}")"',
            'source "${UTILS_DIR}/nvlink.sh"',
            'source "${UTILS_DIR}/ckpt_args.sh"',
            'source "${CUA_LITE_ROOT}/scripts/train/models/${MODEL_ARGS_FILE}.sh"',
            'source "${UTILS_DIR}/ray.sh"',
            'source "${UTILS_DIR}/runtime_env.sh"',
            "start_ray",
            'RUNTIME_ENV_JSON="$(build_runtime_env_json',
            "ray job submit ",
        ],
    )


def test_sft_training_launcher_service_order() -> None:
    _assert_markers_in_order(
        "run_sft.sh",
        [
            'source "${UTILS_DIR}/cleanup.sh"',
            'PROMPT_DATA=${PROMPT_DATA:?"PROMPT_DATA is required"}',
            'source "${UTILS_DIR}/models.sh"',
            'source "${UTILS_DIR}/nvlink.sh"',
            "PREFLIGHT FAIL: no visible GPU is usable.",
            'source "${UTILS_DIR}/ckpt_args.sh"',
            'source "${CUA_LITE_ROOT}/scripts/train/models/${MODEL_ARGS_FILE}.sh"',
            'source "${UTILS_DIR}/ray.sh"',
            'source "${UTILS_DIR}/runtime_env.sh"',
            "start_ray",
            'RUNTIME_ENV_JSON="$(build_runtime_env_json',
            "ray job submit ",
        ],
    )


def test_runtime_env_helper_json_escapes_shell_values() -> None:
    helper = _ROOT / "scripts" / "train" / "utils" / "runtime_env.sh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; shift; build_runtime_env_json "$@"',
            "bash",
            str(helper),
            'SESSION_ID=bad"session',
            r"CUA_LITE_ENV_SERVER_TOKEN=tok\en",
            "MULTILINE=line\nsecond",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["env_vars"] == {
        "SESSION_ID": 'bad"session',
        "CUA_LITE_ENV_SERVER_TOKEN": r"tok\en",
        "MULTILINE": "line\nsecond",
    }


def test_runtime_env_helper_forwards_named_launcher_env_vars() -> None:
    helper = _ROOT / "scripts" / "train" / "utils" / "runtime_env.sh"
    env = os.environ.copy()
    env.update({
        "CUA_LITE_RAY_ENV_VARS": "GEO3K_SOURCE,EXTRA_VALUE MISSING_VALUE",
        "GEO3K_SOURCE": "/root/datasets/geo3k_imgurl/train.parquet",
        "EXTRA_VALUE": "kept",
    })

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; shift; build_runtime_env_json "$@"',
            "bash",
            str(helper),
            "SESSION_ID=run-1",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["env_vars"] == {
        "SESSION_ID": "run-1",
        "GEO3K_SOURCE": "/root/datasets/geo3k_imgurl/train.parquet",
        "EXTRA_VALUE": "kept",
    }


def test_slime_container_init_uses_bind_mounted_slime_submodule_not_baked_image() -> None:
    text = _script("slime/init.sh")

    submodule_preflight = '[ ! -f "${CUA_LITE_ROOT}/slime/pyproject.toml" ]'
    drop_baked_slime = "rm -rf /root/slime"
    editable_lite = 'pip install -q --no-deps -e "${CUA_LITE_ROOT}"'
    editable_slime = 'pip install -q --no-deps -e "${CUA_LITE_ROOT}/slime"'

    for expected in (submodule_preflight, drop_baked_slime, editable_lite, editable_slime):
        assert expected in text
    assert text.index(drop_baked_slime) < text.index(editable_slime)
    assert text.index(editable_lite) < text.index(editable_slime)

