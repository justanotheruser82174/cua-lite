from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from lite.gym.remote.fleet import make_provider

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "serve_fleet.py"


def _load_serve_fleet():
    module_name = "_cua_lite_serve_fleet_under_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_serve_fleet_help_documents_static_nodes_file() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--provider" in proc.stdout
    assert "--nodes-file" in proc.stdout
    assert "--node-token" in proc.stdout
    assert "--timeout-keep-alive" in proc.stdout


def test_real_static_provider_requires_nodes_file() -> None:
    with pytest.raises(SystemExit, match="--provider static requires --nodes-file"):
        make_provider("static", None)


def test_main_passes_keep_alive_to_uvicorn(monkeypatch, tmp_path) -> None:
    mod = _load_serve_fleet()
    nodes_file = tmp_path / "nodes.txt"
    nodes_file.write_text("127.0.0.1:30100\n")
    calls = []

    def fake_run(app, **kwargs):
        calls.append((app, kwargs))

    monkeypatch.setattr(mod.uvicorn, "run", fake_run)
    mod.main(
        argparse.Namespace(
            host="127.0.0.1",
            port=30301,
            provider="static",
            nodes_file=str(nodes_file),
            node_token="fleet-router",
            poll_interval=0.5,
            catalog_interval=10.0,
            timeout_keep_alive=123.0,
        )
    )

    _app, kwargs = calls[0]
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 30301
    assert kwargs["timeout_keep_alive"] == 123.0


def test_main_rejects_non_positive_intervals(tmp_path) -> None:
    mod = _load_serve_fleet()
    nodes_file = tmp_path / "nodes.txt"
    nodes_file.write_text("127.0.0.1:30100\n")

    with pytest.raises(SystemExit, match="must be > 0"):
        mod.main(
            argparse.Namespace(
                host="127.0.0.1",
                port=30301,
                provider="static",
                nodes_file=str(nodes_file),
                node_token="fleet-router",
                poll_interval=0.0,
                catalog_interval=10.0,
                timeout_keep_alive=None,
            )
        )
