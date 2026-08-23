from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "serve_sglang.py"


def _load_serve_sglang():
    module_name = "_cua_lite_serve_sglang_under_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    values = {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "model_path": None,
        "host": "0.0.0.0",
        "port": 30000,
        "engine_kwargs": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _capture_run(monkeypatch, mod):
    calls: list[tuple[list[str], bool]] = []

    def fake_run(cmd, *, check, **_kwargs):
        calls.append((list(cmd), check))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    return calls


def test_main_uses_factory_tp_and_visible_gpu_count(monkeypatch, capsys) -> None:
    mod = _load_serve_sglang()
    calls = _capture_run(monkeypatch, mod)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    mod.main(_args(model_id="Qwen/Qwen3-VL-32B-Instruct", port=30001))

    cmd, check = calls[0]
    assert check is True
    assert cmd[cmd.index("--tp-size") + 1] == "2"
    assert cmd[cmd.index("--dp-size") + 1] == "2"
    out = capsys.readouterr().out
    assert out == "PORT=30001\n"


def test_engine_kwargs_override_and_render_cli_tokens(monkeypatch) -> None:
    mod = _load_serve_sglang()
    calls = _capture_run(monkeypatch, mod)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

    mod.main(
        _args(
            model_id="Qwen/Qwen3-VL-32B-Instruct",
            model_path="/models/qwen",
            host="127.0.0.1",
            port=30002,
            engine_kwargs={
                "tp_size": 4,
                "cuda_graph_bs": [1, 2, 4],
                "trust_remote_code": True,
                "disable_cuda_graph": False,
                "context_length": 8192,
            },
        )
    )

    cmd, _check = calls[0]
    assert cmd[:3] == [sys.executable, "-m", "sglang.launch_server"]
    assert cmd[cmd.index("--model-path") + 1] == "/models/qwen"
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--port") + 1] == "30002"
    assert cmd[cmd.index("--tp-size") + 1] == "4"
    assert cmd[cmd.index("--dp-size") + 1] == "2"
    idx = cmd.index("--cuda-graph-bs")
    assert cmd[idx + 1 : idx + 4] == ["1", "2", "4"]
    assert "--trust-remote-code" in cmd
    assert "--disable-cuda-graph" not in cmd
    assert cmd[cmd.index("--context-length") + 1] == "8192"


def test_non_divisible_visible_gpu_count_keeps_floor_behavior(monkeypatch) -> None:
    mod = _load_serve_sglang()
    calls = _capture_run(monkeypatch, mod)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")

    mod.main(_args(engine_kwargs={"tp_size": 2}))

    cmd, _check = calls[0]
    assert cmd[cmd.index("--tp-size") + 1] == "2"
    assert cmd[cmd.index("--dp-size") + 1] == "1"


def test_rejects_tp_size_larger_than_visible_gpus(monkeypatch) -> None:
    mod = _load_serve_sglang()
    _capture_run(monkeypatch, mod)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    with pytest.raises(SystemExit, match="exceeds visible GPUs"):
        mod.main(_args(engine_kwargs={"tp_size": 2}))


def test_rejects_dp_size_engine_kwarg(monkeypatch) -> None:
    mod = _load_serve_sglang()
    _capture_run(monkeypatch, mod)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    with pytest.raises(SystemExit, match="dp_size is derived"):
        mod.main(_args(engine_kwargs={"dp_size": 2}))


def test_rejects_invalid_tp_size(monkeypatch) -> None:
    mod = _load_serve_sglang()
    _capture_run(monkeypatch, mod)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    with pytest.raises(SystemExit, match="positive integer"):
        mod.main(_args(engine_kwargs={"tp_size": "2"}))


def test_choices_exclude_hf_only_models(monkeypatch) -> None:
    """``backends`` narrows both the ``--model-id`` choices and ``main``.

    No shipped ``LOCAL_AGENTS`` entry declares ``backends`` today, so the
    hf-only arm is pinned through an injected entry: the gate is live CLI
    code (``scripts/serve_sglang.py``) and would otherwise go unexercised.
    """
    mod = _load_serve_sglang()

    assert "Qwen/Qwen3-VL-8B-Instruct" in mod._sglang_model_ids()

    monkeypatch.setitem(mod.LOCAL_AGENTS, "acme/HF-Only-7B", {"backends": ("hf",)})
    assert "acme/HF-Only-7B" not in mod._sglang_model_ids()


def test_main_rejects_hf_only_model_even_if_called_directly(monkeypatch) -> None:
    mod = _load_serve_sglang()
    _capture_run(monkeypatch, mod)
    monkeypatch.setitem(mod.LOCAL_AGENTS, "acme/HF-Only-7B", {"backends": ("hf",)})

    with pytest.raises(SystemExit, match="does not support sglang"):
        mod.main(_args(model_id="acme/HF-Only-7B"))


def test_engine_kwargs_cli_value_must_be_json_object() -> None:
    mod = _load_serve_sglang()

    assert mod._json_object('{"tp_size": 2}') == {"tp_size": 2}
    with pytest.raises(argparse.ArgumentTypeError, match="JSON object"):
        mod._json_object("[]")
    with pytest.raises(argparse.ArgumentTypeError, match="invalid JSON"):
        mod._json_object("{")


def test_port_zero_uses_free_port(monkeypatch, capsys) -> None:
    mod = _load_serve_sglang()
    calls = _capture_run(monkeypatch, mod)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(mod, "_find_free_port", lambda: 30999)

    mod.main(_args(port=0))

    cmd, _check = calls[0]
    assert cmd[cmd.index("--port") + 1] == "30999"
    assert "PORT=30999" in capsys.readouterr().out


def test_help_documents_port_zero_contract() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--port" in proc.stdout
    assert "PORT=<actual>" in proc.stdout
