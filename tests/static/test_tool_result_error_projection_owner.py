"""Static guard for tool-result error projection ownership.

Run:
    uv run pytest tests/static/test_tool_result_error_projection_owner.py -q
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TARGET_OWNER = REPO / "lite" / "core" / "tools" / "results.py"
GRAMMAR_ROOTS = (
    REPO / "lite" / "agents",
    REPO / "lite" / "core",
    REPO / "lite" / "data",
    REPO / "lite" / "infer",
    REPO / "lite" / "train",
    REPO / "lite" / "utils",
)
ERROR_HEADER_LITERAL = "## Error from previous action:"
PROJECTION_HELPERS = frozenset(
    {
        "project_tool_result_text",
        "text_has_projected_tool_result_error",
        "extract_projected_tool_result_error",
    }
)
SKIPPED_PARTS = {
    "__pycache__",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "_vendor",
}


def _iter_python_files(roots: tuple[Path, ...] = GRAMMAR_ROOTS) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if root.is_file():
            candidates = [root]
        else:
            candidates = list(root.rglob("*.py"))
        for path in candidates:
            parts = path.relative_to(REPO).parts
            if any(part in SKIPPED_PARTS for part in parts):
                continue
            paths.append(path)
    # Discovery canary — see ``test_zero_variance_subclass._iter_source_files``:
    # an emptied walk would make every gate below pass vacuously.
    assert paths, f"discovery found no .py files under {[str(r) for r in roots]}"
    return sorted(set(paths))


def _defined_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_tool_result_error_projection_canonical_owner_exists() -> None:
    text = TARGET_OWNER.read_text(encoding="utf-8")

    assert ERROR_HEADER_LITERAL in text
    assert PROJECTION_HELPERS <= _defined_functions(TARGET_OWNER)


def test_tool_result_error_header_grammar_has_single_canonical_owner() -> None:
    offenders: list[str] = []
    for path in _iter_python_files(GRAMMAR_ROOTS):
        if path == TARGET_OWNER:
            continue
        text = path.read_text(encoding="utf-8")
        if ERROR_HEADER_LITERAL in text:
            offenders.append(
                f"{path.relative_to(REPO)} contains {ERROR_HEADER_LITERAL!r}"
            )
        duplicate_helpers = PROJECTION_HELPERS & _defined_functions(path)
        if duplicate_helpers:
            offenders.append(
                f"{path.relative_to(REPO)} defines projection helper(s) "
                f"{sorted(duplicate_helpers)}"
            )

    assert not offenders, (
        "Tool-result error text/error projection grammar is owned by "
        f"{TARGET_OWNER.relative_to(REPO)}. Consumers should call the "
        "owner's projection/read helpers.\n" + "\n".join(offenders)
    )
