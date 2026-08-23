from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_TOOLS = ("bash", "tar", "zip", "unzip")
pytestmark = pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in _TOOLS),
    reason="raw archive checks require bash, zip, and unzip",
)


def _run(script: Path, raw_root: Path) -> None:
    subprocess.run(
        ["bash", str(script)],
        check=True,
        cwd=_ROOT,
        env={**os.environ, "CUA_LITE_RAW_DATASETS_ROOT": str(raw_root)},
        capture_output=True,
        text=True,
    )


def _make_zip(directory: Path, name: str, payload: str) -> None:
    (directory / payload).write_bytes(b"payload")
    subprocess.run(
        ["zip", "-q", name, payload],
        check=True,
        cwd=directory,
    )


def test_opencua_raw_rebuilds_truncated_merged_zip(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    images = raw_root / "xlangai" / "AgentNet" / "ubuntu_images"
    images.mkdir(parents=True)
    _make_zip(images, "images.zip", "sample.png")
    (images / "images-full.zip").write_bytes(b"interrupted")

    _run(
        _ROOT / "lite/data/preproc/opencua/scripts/process_raw_data.sh",
        raw_root,
    )

    subprocess.run(
        ["unzip", "-tq", str(images / "images-full.zip")],
        check=True,
        capture_output=True,
    )
    assert (images / "images-full.zip.extracted").is_file()
    assert (images / "sample.png").read_bytes() == b"payload"


def test_ui_genie_raw_rebuilds_truncated_merged_zip(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    ui = raw_root / "HanXiao1999" / "UI-Genie-Agent-16k"
    (ui / "data" / "screenshots").mkdir(parents=True)
    (ui / "ui_genie_agent_16k.jsonl").write_text("")

    archive_dir = raw_root / "Yuxiang007" / "AMEX" / "AMEX"
    archive_dir.mkdir(parents=True)
    payload = os.urandom(100_000)
    (archive_dir / "sample.png").write_bytes(payload)
    subprocess.run(
        ["zip", "-q", "screenshot.zip", "sample.png"],
        check=True,
        cwd=archive_dir,
    )
    for part in range(1, 9):
        shutil.copyfile(
            archive_dir / "screenshot.zip",
            archive_dir / f"screenshot.z{part:02d}",
        )
    (archive_dir / "screenshot_merged.zip").write_bytes(b"interrupted")

    _run(
        _ROOT / "lite/data/preproc/ui_genie_agent/scripts/process_raw_data.sh",
        raw_root,
    )

    subprocess.run(
        ["unzip", "-tq", str(archive_dir / "screenshot_merged.zip")],
        check=True,
        capture_output=True,
    )
    assert (archive_dir / "screenshot_merged.zip.extracted").is_file()
    assert (ui / "AMEX" / "screenshot" / "sample.png").read_bytes() == payload


def test_scalecua_raw_keeps_parts_when_archive_has_no_files(tmp_path: Path) -> None:
    dataset = tmp_path / "ScaleCUA-Data"
    data_dir = dataset / "data" / "data_test"
    data_dir.mkdir(parents=True)
    subprocess.run(
        ["tar", "-czf", "empty.tar.gz", "--files-from", "/dev/null"],
        check=True,
        cwd=data_dir,
    )
    archive = data_dir / "empty.tar.gz"
    part = data_dir / "empty.tar.gz.part-000"
    archive.replace(part)

    result = subprocess.run(
        [
            "bash",
            str(_ROOT / "lite/data/preproc/scalecua/scripts/process_raw_data.sh"),
            "--dataset-dir",
            str(dataset),
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert part.is_file()
    assert not (data_dir / "empty.tar.gz.extracted").exists()
