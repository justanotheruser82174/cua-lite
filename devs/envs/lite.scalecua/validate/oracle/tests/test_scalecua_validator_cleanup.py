from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

VALIDATE = Path(__file__).resolve().parents[1] / "validate.py"
SWEEP_FN = "_sweep_own_containers"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_lite_scalecua_oracle_validate", VALIDATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_docker(
    module: ModuleType,
    monkeypatch,
    *,
    found: list[str],
    echoed: list[str],
    stderr: str = "",
) -> list[list[str]]:
    calls: list[list[str]] = []

    def _run(argv, **kwargs):
        calls.append(list(argv))
        if argv[1] == "ps":
            return SimpleNamespace(stdout="\n".join(found) + "\n", stderr="", returncode=0)
        assert argv[:4] == ["docker", "rm", "-f", "-v"], f"unexpected argv: {argv}"
        return SimpleNamespace(
            stdout=("\n".join(echoed) + "\n") if echoed else "",
            stderr=stderr,
            returncode=0,
        )

    monkeypatch.setattr(module.subprocess, "run", _run)
    return calls


def test_sweep_counts_removals_not_matches(monkeypatch, caplog):
    module = _load()
    stderr = (
        "Error response from daemon: removal of container c3 is already in progress\n"
        "Error response from daemon: removal of container c4 is already in progress"
    )
    _fake_docker(
        module,
        monkeypatch,
        found=["c1", "c2", "c3", "c4"],
        echoed=["c1", "c2"],
        stderr=stderr,
    )

    with caplog.at_level(logging.WARNING):
        n = module.__dict__[SWEEP_FN]("sess")

    assert n == 2
    assert "already in progress" in caplog.text
    assert "c3" in caplog.text and "c4" in caplog.text


def test_sweep_total_failure_reports_zero_not_the_match_count(monkeypatch, caplog):
    module = _load()
    _fake_docker(
        module,
        monkeypatch,
        found=["a", "b", "c"],
        echoed=[],
        stderr="Error response from daemon: removal of container a is already in progress",
    )

    with caplog.at_level(logging.WARNING):
        n = module.__dict__[SWEEP_FN]("sess")

    assert n == 0
    assert "SURVIVED" in caplog.text


def test_sweep_clean_removal_is_silent_and_exact(monkeypatch, caplog):
    module = _load()
    _fake_docker(module, monkeypatch, found=["x", "y"], echoed=["y", "x"])

    with caplog.at_level(logging.WARNING):
        n = module.__dict__[SWEEP_FN]("sess")

    assert n == 2
    assert "SURVIVED" not in caplog.text


def test_sweep_ignores_containers_that_appeared_after_the_ps(monkeypatch):
    module = _load()
    _fake_docker(module, monkeypatch, found=["a"], echoed=["a", "late-arrival"])

    assert module.__dict__[SWEEP_FN]("sess") == 1


def test_sweep_no_containers_is_zero(monkeypatch):
    module = _load()

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda argv, **kw: SimpleNamespace(stdout="\n", stderr="", returncode=0),
    )
    assert module.__dict__[SWEEP_FN]("sess") == 0


@pytest.mark.parametrize(
    "boom",
    [
        OSError("docker binary missing"),
        subprocess.TimeoutExpired(cmd="docker", timeout=15),
        subprocess.SubprocessError("spawn failed"),
    ],
    ids=["oserror", "timeout", "subprocess-error"],
)
def test_sweep_never_raises_and_unknown_is_none(boom, monkeypatch):
    module = _load()

    def _run(argv, **kwargs):
        raise boom

    monkeypatch.setattr(module.subprocess, "run", _run)
    assert module.__dict__[SWEEP_FN]("sess") is None


def test_sweep_rm_timeout_is_unknown_not_a_count(monkeypatch):
    module = _load()

    def _run(argv, **kwargs):
        if argv[1] == "ps":
            return SimpleNamespace(stdout="c1\nc2\n", stderr="", returncode=0)
        raise subprocess.TimeoutExpired(cmd=argv, timeout=60)

    monkeypatch.setattr(module.subprocess, "run", _run)
    assert module.__dict__[SWEEP_FN]("sess") is None


def test_sweep_removal_keeps_the_dash_v(monkeypatch):
    module = _load()
    calls = _fake_docker(module, monkeypatch, found=["a"], echoed=["a"])

    module.__dict__[SWEEP_FN]("sess")

    rm = [c for c in calls if c[1] == "rm"]
    assert rm and rm[0][:4] == ["docker", "rm", "-f", "-v"], rm
