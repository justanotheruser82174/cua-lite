from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _shell_scripts() -> list[Path]:
    patterns = [
        "scripts/train/run_*.sh",
        "scripts/train/slime/*.sh",
        "scripts/train/utils/*.sh",
        "scripts/train/models/*.sh",
    ]
    scripts: list[Path] = []
    for pattern in patterns:
        matches = sorted(ROOT.glob(pattern))
        assert matches, f"shell-script glob matched nothing: {pattern}"
        scripts.extend(matches)
    return sorted(scripts)


@pytest.mark.parametrize("script", _shell_scripts(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_train_shell_script_syntax(script: Path) -> None:
    proc = subprocess.run(
        ["bash", "-n", str(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
