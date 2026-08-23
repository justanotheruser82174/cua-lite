from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "scripts" / "train" / "utils" / "models.sh"


def _run_models(model_id: str, extra: str = "") -> subprocess.CompletedProcess[str]:
    command = "\n".join(
        [
            'MODEL_ID="$2"',
            'source "$1"',
            'printf "MODEL_ARGS_FILE=%s\\n" "$MODEL_ARGS_FILE"',
            'printf "MODEL_FAMILY=%s\\n" "$MODEL_FAMILY"',
            'printf "MODEL_SLUG=%s\\n" "$MODEL_SLUG"',
            extra,
        ]
    )
    return subprocess.run(
        ["bash", "-c", command, "bash", str(HELPER), model_id],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        (
            "Qwen/Qwen3-VL-4B-Instruct",
            {
                "MODEL_ARGS_FILE": "Qwen3-VL-4B",
                "MODEL_FAMILY": "qwen3_vl",
                "MODEL_SLUG": "Qwen_Qwen3-VL-4B-Instruct",
            },
        ),
        (
            "Qwen/Qwen2.5-VL-3B-Instruct",
            {
                "MODEL_ARGS_FILE": "Qwen2.5-VL-3B",
                "MODEL_FAMILY": "qwen3_vl",
                "MODEL_SLUG": "Qwen_Qwen2.5-VL-3B-Instruct",
            },
        ),
        (
            "Tongyi-MAI/MAI-UI-2B",
            {
                "MODEL_ARGS_FILE": "Qwen3-VL-2B",
                "MODEL_FAMILY": "qwen3_vl",
                "MODEL_SLUG": "Tongyi-MAI_MAI-UI-2B",
            },
        ),
        (
            "Qwen/Qwen3.5-9B",
            {
                "MODEL_ARGS_FILE": "Qwen3.5-9B",
                "MODEL_FAMILY": "qwen3_5",
                "MODEL_SLUG": "Qwen_Qwen3.5-9B",
            },
        ),
    ],
)
def test_models_exports_public_training_model_contract(
    model_id: str, expected: dict[str, str]
) -> None:
    proc = _run_models(model_id)

    assert proc.returncode == 0, proc.stderr
    actual = dict(line.split("=", 1) for line in proc.stdout.splitlines())
    assert actual == expected


def test_models_rejects_unknown_model_id() -> None:
    proc = _run_models("Unknown/Model")

    assert proc.returncode != 0
    assert "MODEL_ID must be one of:" in proc.stderr


def test_resolve_tp_defaults_to_one() -> None:
    proc = _run_models(
        "Qwen/Qwen3-VL-4B-Instruct",
        'printf "TP=%s\\n" "$(resolve_tp "" 8)"',
    )

    assert proc.returncode == 0, proc.stderr
    assert "TP=1" in proc.stdout.splitlines()


def test_resolve_tp_accepts_divisible_request() -> None:
    proc = _run_models(
        "Qwen/Qwen3.5-9B",
        'printf "TP=%s\\n" "$(resolve_tp 4 8)"',
    )

    assert proc.returncode == 0, proc.stderr
    assert "TP=4" in proc.stdout.splitlines()


def test_resolve_tp_rejects_non_divisible_request() -> None:
    proc = subprocess.run(
        [
            "bash",
            "-c",
            'MODEL_ID="$2"; source "$1"; resolve_tp 3 8',
            "bash",
            str(HELPER),
            "Qwen/Qwen3.5-9B",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert proc.returncode != 0
    assert "TP_SIZE=3 does not divide NUM_TRAIN_GPUS=8" in proc.stderr


@pytest.mark.parametrize(
    ("tp", "num_train_gpus", "expected"),
    [
        ("0", "8", "TP_SIZE=0 must be a positive integer"),
        ("bad", "8", "TP_SIZE=bad must be a positive integer"),
        ("2", "0", "NUM_TRAIN_GPUS=0 must be a positive integer"),
        ("2", "bad", "NUM_TRAIN_GPUS=bad must be a positive integer"),
    ],
)
def test_resolve_tp_rejects_non_positive_or_non_numeric_values(
    tp: str, num_train_gpus: str, expected: str
) -> None:
    proc = subprocess.run(
        [
            "bash",
            "-c",
            'MODEL_ID="$2"; source "$1"; resolve_tp "$3" "$4"',
            "bash",
            str(HELPER),
            "Qwen/Qwen3.5-9B",
            tp,
            num_train_gpus,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert proc.returncode != 0
    assert expected in proc.stderr


def test_resolve_num_engines_accepts_flooring_spare_rollout_gpus() -> None:
    proc = _run_models(
        "Qwen/Qwen3-VL-4B-Instruct",
        'printf "ENGINES=%s\\n" "$(resolve_num_engines 2 5)"',
    )

    assert proc.returncode == 0, proc.stderr
    assert "ENGINES=2" in proc.stdout.splitlines()


@pytest.mark.parametrize(
    ("gpus_per_engine", "num_rollout_gpus", "expected"),
    [
        ("0", "8", "GPUS_PER_ENGINE=0 must be a positive integer"),
        ("bad", "8", "GPUS_PER_ENGINE=bad must be a positive integer"),
        ("2", "0", "NUM_ROLLOUT_GPUS=0 must be a positive integer"),
        ("4", "2", "GPUS_PER_ENGINE=4 exceeds NUM_ROLLOUT_GPUS=2"),
    ],
)
def test_resolve_num_engines_rejects_zero_engine_cases(
    gpus_per_engine: str, num_rollout_gpus: str, expected: str
) -> None:
    proc = subprocess.run(
        [
            "bash",
            "-c",
            'MODEL_ID="$2"; source "$1"; resolve_num_engines "$3" "$4"',
            "bash",
            str(HELPER),
            "Qwen/Qwen3.5-9B",
            gpus_per_engine,
            num_rollout_gpus,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert proc.returncode != 0
    assert expected in proc.stderr
