"""Static residue guards for canonical Lite tool-call shape."""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

PUBLIC_CONTRACT_ROOTS = (
    REPO / "lite" / "core" / "tools",
    REPO / "lite" / "data" / "utils" / "rows.py",
    REPO / "lite" / "data" / "hf" / "stage.py",
    REPO / "lite" / "infer" / "debug" / "log_contract.py",
    REPO / "lite" / "train" / "export" / "export_sft.py",
)

TEXT_SUFFIXES = {".md", ".py", ".yaml", ".yml"}
SKIPPED_PARTS = {
    "__pycache__",
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tmp",
    ".data",
    ".venv",
    "_vendor",
    "build",
    "logs",
    "node_modules",
    "run_records",
    "runs",
    "slime",
}
LEGACY_OWNER_PREFIXES = ("devs/migration/",)

CALL_ID_TOKEN = re.compile(r"(?<!tool_)(?<![A-Za-z0-9_])call_id(?![A-Za-z0-9_])")
NAME_TOKEN = re.compile(r"(?<![A-Za-z0-9_])name(?![A-Za-z0-9_])")
ARGUMENTS_TOKEN = re.compile(r"(?<![A-Za-z0-9_])arguments(?![A-Za-z0-9_])")


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def _is_scanned_path(path: Path) -> bool:
    rel = path.relative_to(REPO)
    if path == Path(__file__).resolve():
        return False
    if path.suffix not in TEXT_SUFFIXES:
        return False
    if set(rel.parts) & SKIPPED_PARTS:
        return False
    return not ("src" in rel.parts and "gen" in rel.parts)


def _is_pruned_dir(path: Path) -> bool:
    rel = path.relative_to(REPO)
    return path.name in SKIPPED_PARTS or (
        len(rel.parts) >= 2 and rel.parts[-2:] == ("src", "gen")
    )


def _is_legacy_owner_path(path: Path) -> bool:
    rel = _rel(path)
    return any(rel.startswith(prefix) for prefix in LEGACY_OWNER_PREFIXES)


def _iter_scanned_files() -> list[Path]:
    files: list[Path] = []
    for root in PUBLIC_CONTRACT_ROOTS:
        if root.is_file():
            if _is_scanned_path(root):
                files.append(root)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            base = Path(dirpath)
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if not _is_pruned_dir(base / dirname)
            ]
            for filename in filenames:
                path = base / filename
                if _is_scanned_path(path):
                    files.append(path)
    assert files, "canonical shape residue scan discovered no files"
    return sorted(files)


def _constant_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_none(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _dict_values_by_string_key(node: ast.AST) -> dict[str, ast.AST]:
    if not isinstance(node, ast.Dict):
        return {}
    return {
        key.value: value
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }


def _is_under_tool_calls_key(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    current = node
    while current in parents:
        parent = parents[current]
        if isinstance(parent, ast.Dict):
            for key, value in zip(parent.keys, parent.values, strict=True):
                if value is current and _constant_string(key) == "tool_calls":
                    return True
        current = parent
    return False


def _is_nullable_nested_padding(values_by_key: dict[str, ast.AST]) -> bool:
    """Parquet can leave nullable retired columns beside the nested call."""
    return (
        {"id", "type", "function"} <= set(values_by_key)
        and _constant_string(values_by_key["type"]) == "function"
        and all(
            _is_none(values_by_key.get(key))
            for key in ("call_id", "name", "arguments")
        )
    )


def _is_retired_flat_lite_dict(node: ast.AST) -> bool:
    values_by_key = _dict_values_by_string_key(node)
    if not {"call_id", "name", "arguments"} <= set(values_by_key):
        return False
    if _constant_string(values_by_key.get("type")) == "function_call":
        return False
    return not _is_nullable_nested_padding(values_by_key)


def _python_flat_lite_offenders_from_tree(path: Path, tree: ast.AST) -> list[str]:
    parents = _parent_map(tree)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        if not _is_retired_flat_lite_dict(node):
            continue
        if not _is_under_tool_calls_key(node, parents):
            continue
        offenders.append(
            f"{_rel(path)}:{node.lineno}: retired flat Lite tool_calls literal"
        )
    return offenders


def _python_flat_lite_offenders(path: Path) -> list[str]:
    if _is_legacy_owner_path(path):
        return []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return _python_flat_lite_offenders_from_tree(path, tree)


def _text_flat_lite_offenders(path: Path, text: str) -> list[str]:
    if path.suffix == ".py" or _is_legacy_owner_path(path):
        return []
    lines = text.splitlines()
    offenders: list[str] = []
    for index, line in enumerate(lines):
        if not CALL_ID_TOKEN.search(line):
            continue
        start = max(0, index - 2)
        end = min(len(lines), index + 8)
        window = "\n".join(lines[start:end])
        lowered_window = window.lower()
        if "function_call" in lowered_window or "make_tool_call" in lowered_window:
            continue
        if "{" not in window or "}" not in window:
            continue
        if not (NAME_TOKEN.search(window) and ARGUMENTS_TOKEN.search(window)):
            continue
        offenders.append(
            f"{_rel(path)}:{index + 1}: retired flat Lite call_id/name/arguments example"
        )
    return offenders


def test_no_retired_flat_canonical_lite_tool_call_residue() -> None:
    offenders: list[str] = []
    for path in _iter_scanned_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        offenders.extend(_text_flat_lite_offenders(path, text))
        if path.suffix == ".py":
            offenders.extend(_python_flat_lite_offenders(path))

    assert not offenders, (
        "retired bare Lite tool-call examples should not reappear; use nested "
        "Lite calls for canonical examples, and keep provider-flat or env-wire "
        "discussion scoped to its boundary:\n  " + "\n  ".join(offenders)
    )


def test_python_detector_rejects_retired_flat_tool_calls_literal() -> None:
    tree = ast.parse("""
message = {
    "role": "assistant",
    "tool_calls": [
        {"call_id": "call_1", "name": "click", "arguments": {"x": 1, "y": 2}},
    ],
}
""")

    offenders = _python_flat_lite_offenders_from_tree(
        REPO / "lite" / "core" / "tools" / "_planted.py",
        tree,
    )

    assert offenders == [
        "lite/core/tools/_planted.py:5: retired flat Lite tool_calls literal"
    ]


def test_text_detector_rejects_retired_flat_contract_examples() -> None:
    offenders = _text_flat_lite_offenders(
        REPO / "lite" / "core" / "tools" / "README.md",
        """
```json
{"tool_calls": [{"call_id": "call_1", "name": "click", "arguments": {"x": 1}}]}
```
""",
    )

    assert offenders == [
        "lite/core/tools/README.md:3: retired flat Lite call_id/name/arguments example"
    ]


def test_residue_gate_ignores_private_wording_without_contract_shape() -> None:
    offenders = _text_flat_lite_offenders(
        REPO / "lite" / "core" / "tools" / "README.md",
        "A private note may mention flat canonical Lite wording without "
        "publishing a call_id/name/arguments example.",
    )

    assert offenders == []
