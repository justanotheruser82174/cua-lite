"""Shared active-source scanner for retired-path static gates."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import PurePosixPath

from lite.utils.path import project_root

ACTIVE_SOURCE_ROOTS: tuple[str, ...] = (
    "README.md",
    "docs",
    "devs",
    "examples",
    "lite",
    "scripts",
    "tests",
)

_CACHE_DIRS = frozenset({"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"})


def active_source_files(roots: Sequence[str] = ACTIVE_SOURCE_ROOTS) -> list[str]:
    """Repo-relative tracked and untracked-not-ignored files under active roots."""
    root = project_root()
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *roots],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    return [
        rel
        for rel in listing.split("\0")
        if rel and not (_CACHE_DIRS & set(PurePosixPath(rel).parts)) and (root / rel).is_file()
    ]


def scan_active_sources(
    patterns: Iterable[str],
    *,
    exclude: Iterable[str] = (),
    literal: bool = False,
    roots: Sequence[str] = ACTIVE_SOURCE_ROOTS,
) -> list[str]:
    """Return ``path:lineno: line`` for active-source lines matching any pattern."""
    root = project_root()
    skipped = set(exclude)
    combined = re.compile(
        "|".join(
            f"(?:{re.escape(pattern) if literal else pattern})"
            for pattern in patterns
        ),
        re.MULTILINE,
    )

    offenders: list[str] = []
    for rel in active_source_files(roots):
        if rel in skipped:
            continue
        data = (root / rel).read_bytes()
        if b"\x00" in data:
            continue
        text = data.decode("utf-8", errors="replace")
        if not combined.search(text):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if combined.search(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    return offenders
