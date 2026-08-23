"""Unit tests for lite/utils/git.py (metadata contract section 11.P P1/P7).

Run: uv run pytest tests/utils/test_git.py -v
"""
from __future__ import annotations

import re

import lite.utils.git as git_mod
from lite.utils.git import git_commit, run_git


def test_run_git_returns_repo_state_regardless_of_cwd(tmp_path, monkeypatch):
    """Anchored at project_root(), not the process CWD."""
    monkeypatch.chdir(tmp_path)  # a non-repo directory
    head = run_git("rev-parse --short HEAD")
    assert re.fullmatch(r"[0-9a-f]{6,}", head)


def test_git_commit_shape():
    commit = git_commit()
    assert re.fullmatch(r"[0-9a-f]{6,}(-dirty)?", commit)


def test_p7_degradation_outside_checkout(monkeypatch):
    """project_root() raising RuntimeError (installed-wheel deployment) must
    degrade to "" — upload.py's default-tag path relies on it (logs + skips)."""
    def _boom() -> None:
        raise RuntimeError("no source checkout")
    monkeypatch.setattr(git_mod, "project_root", _boom)
    assert run_git("rev-parse --short HEAD") == ""
    assert git_commit() == ""
