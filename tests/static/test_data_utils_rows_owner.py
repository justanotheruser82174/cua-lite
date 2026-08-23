"""Static guards for data row validator ownership."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_ROWS = REPO / "lite" / "data" / "utils" / "rows.py"
DATA_UTILS_ROOT = REPO / "lite" / "data" / "utils" / "__init__.py"
MIGRATION_DIR = REPO / "devs" / "migration"
VALIDATOR_MODULES = (
    DATA_ROWS,
    REPO / "lite" / "data" / "hf" / "stage.py",
    REPO / "lite" / "data" / "hf" / "upload.py",
)


def _imports_from(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    return modules


def test_data_row_validation_is_not_owned_by_package_root_or_alias_file() -> None:
    assert not (REPO / "lite" / "data" / "utils" / "tool_calls.py").exists()
    assert not (REPO / "lite" / "data" / "utils" / "tools.py").exists()

    root_source = DATA_UTILS_ROOT.read_text(encoding="utf-8")
    assert "def clean_nones" not in root_source
    assert "validate_tool_calls" not in root_source
    assert "validate_publish_rows" not in root_source


def test_data_row_validators_do_not_import_agents_helpers() -> None:
    offenders: list[str] = []
    for path in VALIDATOR_MODULES:
        rel = path.relative_to(REPO)
        for module in _imports_from(path):
            if module == "lite.agents" or module.startswith("lite.agents."):
                offenders.append(f"{rel}: imports {module}")

    assert not offenders, "\n".join(offenders)


def test_production_data_boundary_imports_only_public_data_row_validators() -> None:
    offenders: list[str] = []
    for path in VALIDATOR_MODULES:
        rel = path.relative_to(REPO)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "lite.data.utils.rows":
                continue
            for alias in node.names:
                if alias.name.startswith("_"):
                    offenders.append(f"{rel}: imports private {alias.name}")

    assert not offenders, "\n".join(offenders)


def test_import_data_rows_does_not_initialize_agents() -> None:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = (
        str(REPO)
        if not env.get("PYTHONPATH")
        else f"{REPO}{os.pathsep}{env['PYTHONPATH']}"
    )
    code = """
import sys
import lite.data.utils.rows  # noqa: F401

offenders = sorted(
    name for name in sys.modules
    if name == "lite.agents" or name.startswith("lite.agents.")
)
if offenders:
    print("\\n".join(offenders))
    raise SystemExit(1)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_migration_does_not_import_private_data_validator_hooks() -> None:
    offenders: list[str] = []
    migration_files = sorted(MIGRATION_DIR.glob("*.py"))
    # Discovery canary — see ``test_zero_variance_subclass._iter_source_files``:
    # an emptied glob would make this gate pass vacuously.
    assert migration_files, f"discovery found no .py files under {MIGRATION_DIR}"
    for path in migration_files:
        rel = path.relative_to(REPO)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "lite.data.utils.rows":
                continue
            for alias in node.names:
                if alias.name.startswith("_"):
                    offenders.append(f"{rel}: imports private {alias.name}")

    assert not offenders, "\n".join(offenders)


def test_production_data_validators_have_no_migration_compatibility_hook() -> None:
    source = DATA_ROWS.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(DATA_ROWS))
    function_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }

    assert "validate_publish_rows" not in function_names
    assert "repair_legacy_tool_call_args" not in function_names
    assert "upgrade_parquet_row" not in function_names
    assert "schema_free_names_for_metadata" not in source
    assert "migration-compatible" not in source


def test_validate_publish_rows_public_alias_is_removed() -> None:
    source = DATA_ROWS.read_text(encoding="utf-8")
    assert "def validate_publish_rows" not in source
    assert "validate_publish_rows" not in source
