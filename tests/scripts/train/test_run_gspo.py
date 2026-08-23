from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]


def test_run_gspo_delegates_to_grpo_with_gspo_defaults(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    train_dir = root / "scripts" / "train"
    train_dir.mkdir(parents=True)
    shutil.copy(_ROOT / "scripts" / "train" / "run_gspo.sh", train_dir / "run_gspo.sh")

    capture = root / "capture.txt"
    grpo = train_dir / "run_grpo.sh"
    grpo.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' "
        "\"$ALGO\" \"$ADV_ESTIMATOR\" \"$EPS_CLIP\" \"$EPS_CLIP_HIGH\" "
        "\"$NUM_STEPS_PER_ROLLOUT\" \"$CUA_LITE_DISABLE_RADIX\" "
        "\"$1\" \"$2\" > \"$FAKE_CAPTURE\"\n",
        encoding="utf-8",
    )
    grpo.chmod(0o755)

    env = os.environ.copy()
    env["FAKE_CAPTURE"] = str(capture)

    result = subprocess.run(
        ["bash", str(train_dir / "run_gspo.sh"), "arg-one", "arg-two"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "gspo",
        "gspo",
        "1e-4",
        "2e-4",
        "1",
        "1",
        "arg-one",
        "arg-two",
    ]


def test_run_gspo_respects_explicit_overrides(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    train_dir = root / "scripts" / "train"
    train_dir.mkdir(parents=True)
    shutil.copy(_ROOT / "scripts" / "train" / "run_gspo.sh", train_dir / "run_gspo.sh")

    capture = root / "capture.txt"
    grpo = train_dir / "run_grpo.sh"
    grpo.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' "
        "\"$ALGO\" \"$ADV_ESTIMATOR\" \"$EPS_CLIP\" \"$EPS_CLIP_HIGH\" "
        "\"$NUM_STEPS_PER_ROLLOUT\" \"$CUA_LITE_DISABLE_RADIX\" > \"$FAKE_CAPTURE\"\n",
        encoding="utf-8",
    )
    grpo.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "FAKE_CAPTURE": str(capture),
            "ALGO": "custom",
            "ADV_ESTIMATOR": "custom_adv",
            "EPS_CLIP": "0.3",
            "EPS_CLIP_HIGH": "0.4",
            "NUM_STEPS_PER_ROLLOUT": "3",
            "CUA_LITE_DISABLE_RADIX": "0",
        }
    )

    result = subprocess.run(
        ["bash", str(train_dir / "run_gspo.sh")],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "custom",
        "custom_adv",
        "0.3",
        "0.4",
        "3",
        "0",
    ]


def test_run_gspo_forces_per_turn_radix_setting_before_grpo_exec() -> None:
    text = (_ROOT / "scripts" / "train" / "run_gspo.sh").read_text()

    assert "export CUA_LITE_DISABLE_RADIX=${CUA_LITE_DISABLE_RADIX:-1}" in text
    assert "export NUM_STEPS_PER_ROLLOUT=${NUM_STEPS_PER_ROLLOUT:-1}" in text
    assert 'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"' in text
    assert 'exec bash "${SCRIPT_DIR}/run_grpo.sh" "$@"' in text


def test_run_gspo_inherits_the_grpo_dotted_paths_rather_than_declaring_its_own() -> None:
    text = (_ROOT / "scripts" / "train" / "run_gspo.sh").read_text()

    for flag in (
        "--custom-generate-function-path",
        "--custom-convert-samples-to-train-data-path",
        "--rollout-function-path",
    ):
        assert flag not in text
