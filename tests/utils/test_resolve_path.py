"""
Tests for lite.utils.path: resolve_path.

Covers path that exists as-is, relative path with env var set, and missing path/env.

Run: uv run pytest tests/utils/test_resolve_path.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lite.utils.path import resolve_path


def test_resolve_path_existing_absolute(tmp_path):
    """resolve_path returns resolved path when given path exists (absolute)."""
    d = tmp_path / "sub"
    d.mkdir()
    (d / "file.txt").write_text("x")
    result = resolve_path(str(d / "file.txt"), "CUA_LITE_DATASETS_ROOT")
    assert result == str((d / "file.txt").resolve())
    assert Path(result).exists()

def test_resolve_path_existing_relative_from_cwd(tmp_path, monkeypatch):
    """resolve_path returns resolved path when relative path exists from cwd."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "local.txt").write_text("x")
    result = resolve_path("local.txt", "CUA_LITE_DATASETS_ROOT")
    assert result == str((tmp_path / "local.txt").resolve())

def test_resolve_path_relative_with_env_set(tmp_path, monkeypatch):
    """resolve_path with relative path uses env var as base when path does not exist as-is."""
    monkeypatch.setenv("CUA_LITE_DATASETS_ROOT", str(tmp_path))
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "x.parquet").write_text("")
    result = resolve_path("data/x.parquet", "CUA_LITE_DATASETS_ROOT")
    assert result == str((tmp_path / "data" / "x.parquet").resolve())

def test_resolve_path_relative_env_unset_raises(monkeypatch):
    """resolve_path with relative path and env unset raises FileNotFoundError."""
    monkeypatch.delenv("CUA_LITE_DATASETS_ROOT", raising=False)
    with pytest.raises(FileNotFoundError, match="must be set"):
        resolve_path("nonexistent/relative/path.parquet", "CUA_LITE_DATASETS_ROOT")

def test_resolve_path_relative_env_empty_raises(monkeypatch):
    """resolve_path with relative path and empty env raises FileNotFoundError."""
    monkeypatch.setenv("CUA_LITE_DATASETS_ROOT", "")
    with pytest.raises(FileNotFoundError, match="must be set"):
        resolve_path("any/path", "CUA_LITE_DATASETS_ROOT")

def test_resolve_path_relative_env_set_but_path_missing_raises(tmp_path, monkeypatch):
    """resolve_path when env points to base but full path does not exist raises."""
    monkeypatch.setenv("CUA_LITE_DATASETS_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_path("missing/file.parquet", "CUA_LITE_DATASETS_ROOT")
