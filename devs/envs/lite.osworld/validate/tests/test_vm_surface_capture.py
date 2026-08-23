from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PINNED_REVISION = "a" * 40
QCOW2_SHA256 = "b" * 64


def _load_capture_module():
    path = Path(__file__).resolve().parents[1] / "vm_surface_capture.py"
    spec = importlib.util.spec_from_file_location(
        "_lite_osworld_vm_surface_capture_under_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _base_args(qcow2: Path, output: Path) -> list[str]:
    return [
        "--qcow2",
        str(qcow2),
        "--output",
        str(output),
        "--hf-revision",
        PINNED_REVISION,
        "--sha256",
        QCOW2_SHA256,
    ]


def test_vm_surface_agent_shell_uses_scrubbed_path_and_non_login_bash():
    mod = _load_capture_module()

    command = mod.agent_shell("compgen -c")

    assert f"PATH={mod.AGENT_PATH}" in command
    assert "env -i" in command
    assert "bash -c" in command
    assert "bash -lc" not in command


def test_vm_surface_capture_defaults_to_private_osworld_image(tmp_path):
    mod = _load_capture_module()
    qcow2 = tmp_path / "Ubuntu.qcow2"
    qcow2.write_bytes(b"qcow2")

    args = mod.parse_args(_base_args(qcow2, tmp_path / "vm_surface.json"))

    assert args.image == "cua-lite/osworld:mine"
    assert args.hf_revision == PINNED_REVISION
    assert args.sha256 == QCOW2_SHA256


def test_vm_surface_capture_rejects_latest_before_boot(monkeypatch, tmp_path):
    mod = _load_capture_module()
    qcow2 = tmp_path / "Ubuntu.qcow2"
    qcow2.write_bytes(b"qcow2")
    output = tmp_path / "vm_surface.json"

    def fail_capture(_args):
        raise AssertionError("capture_manifest should not run for :latest")

    monkeypatch.setattr(mod, "capture_manifest", fail_capture)

    with pytest.raises(SystemExit, match="not :latest"):
        mod.main(["--image", "cua-lite/osworld:latest", *_base_args(qcow2, output)])


def test_vm_surface_capture_rejects_implicit_latest_before_boot(monkeypatch, tmp_path):
    mod = _load_capture_module()
    qcow2 = tmp_path / "Ubuntu.qcow2"
    qcow2.write_bytes(b"qcow2")
    output = tmp_path / "vm_surface.json"

    def fail_capture(_args):
        raise AssertionError("capture_manifest should not run for untagged images")

    monkeypatch.setattr(mod, "capture_manifest", fail_capture)

    with pytest.raises(SystemExit, match="not :latest"):
        mod.main(["--image", "cua-lite/osworld", *_base_args(qcow2, output)])


def test_vm_surface_capture_rejects_unpinned_hf_revision_before_boot(monkeypatch, tmp_path):
    mod = _load_capture_module()
    qcow2 = tmp_path / "Ubuntu.qcow2"
    qcow2.write_bytes(b"qcow2")
    output = tmp_path / "vm_surface.json"

    def fail_capture(_args):
        raise AssertionError("capture_manifest should not run for branch revisions")

    monkeypatch.setattr(mod, "capture_manifest", fail_capture)

    args = _base_args(qcow2, output)
    args[args.index(PINNED_REVISION)] = "main"
    with pytest.raises(SystemExit, match="pinned 40-character commit SHA"):
        mod.main(args)


def test_vm_surface_capture_rejects_bad_sha256_before_boot(monkeypatch, tmp_path):
    mod = _load_capture_module()
    qcow2 = tmp_path / "Ubuntu.qcow2"
    qcow2.write_bytes(b"qcow2")
    output = tmp_path / "vm_surface.json"

    def fail_capture(_args):
        raise AssertionError("capture_manifest should not run for bad sha256")

    monkeypatch.setattr(mod, "capture_manifest", fail_capture)

    args = _base_args(qcow2, output)
    args[args.index(QCOW2_SHA256)] = "not-a-sha"
    with pytest.raises(SystemExit, match="64-character hex"):
        mod.main(args)
