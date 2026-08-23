"""Local orchestration for one-time LiteSample forward migration.

The pure schema transform lives in ``upgrade.py``. This module handles file
iteration, local artifact writing, and optional invariant verification. It
deliberately does not mutate remote HuggingFace repositories.

``--dry-run`` migrates and verifies entirely in memory and writes nothing --
the same meaning the sibling ``devs/data/*/filter.py --dry-run`` flags carry.

Run:
    uv run python -m devs.migration.run <input> -o <output> [--verify]
    uv run python -m devs.migration.run <input> --dry-run --verify
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from lite.core.metadata import LiteCUAMetadata
from lite.data.staging import CANONICAL_SPLITS as STAGING_CANONICAL_SPLITS


def _load_sibling_module(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"cua_lite_migration_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:  # Package import, e.g. ``python -m devs.migration.run``.
    from .upgrade import upgrade_lite_sample, upgrade_parquet_row
    from .verify import verify_lite_sample
except ImportError:  # Direct script/importlib execution.
    _upgrade = _load_sibling_module("upgrade")
    _verify = _load_sibling_module("verify")
    upgrade_lite_sample = _upgrade.upgrade_lite_sample
    upgrade_parquet_row = _upgrade.upgrade_parquet_row
    verify_lite_sample = _verify.verify_lite_sample


SUPPORTED_SUFFIXES = {".json", ".jsonl", ".parquet"}
_DEV_DATASET_COMPONENTS = {
    "lite.osworld": "Lite.OSWorld",
    "lite.cuagym": "Lite.CUAGym",
    "lite.cuaworld": "Lite.CUAWorld",
    "lite.scalecua": "Lite.ScaleCUA",
    "webgym": "WebGym",
}
ALLOWED_MIGRATION_DATASETS = frozenset(_DEV_DATASET_COMPONENTS.values())
_CANONICAL_PLATFORMS = frozenset(str(platform) for platform in LiteCUAMetadata.Platform)
_LEGACY_WEB_PLATFORM = "web"
_BROWSER_PLATFORM = LiteCUAMetadata.Platform.BROWSER.value
_MIGRATION_INPUT_PLATFORMS = _CANONICAL_PLATFORMS | frozenset({_LEGACY_WEB_PLATFORM})
_CANONICAL_TASK_TYPES = frozenset(str(task_type) for task_type in LiteCUAMetadata.TaskType)
_CANONICAL_SPLITS = frozenset(STAGING_CANONICAL_SPLITS)


class MigrationSummary:
    def __init__(
        self,
        *,
        input_path: str,
        output_path: str | None,
        dry_run: bool,
        files: int = 0,
        rows: int = 0,
        verified: int = 0,
    ) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.dry_run = dry_run
        self.files = files
        self.rows = rows
        self.verified = verified

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "dry_run": self.dry_run,
            "files": self.files,
            "rows": self.rows,
            "verified": self.verified,
        }


def migrate_rows(
    rows: list[dict[str, Any]],
    *,
    parquet_rows: bool = False,
    verify: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Upgrade rows in memory and optionally verify each migrated row."""
    upgraded: list[dict[str, Any]] = []
    verified = 0
    for row in rows:
        migrated = upgrade_parquet_row(row) if parquet_rows else upgrade_lite_sample(row)
        if verify:
            verify_lite_sample(migrated)
            verified += 1
        upgraded.append(migrated)
    return upgraded, verified


def migrate_path(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    dry_run: bool = False,
    verify: bool = False,
) -> MigrationSummary:
    """Upgrade one supported file or a tree of supported files.

    ``dry_run`` migrates (and, with ``verify``, verifies) every row in memory and
    writes nothing; any ``output_path`` is ignored. A real run writes, and to avoid
    accidental source mutation it requires an explicit output path.
    """
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(src)
    _require_allowed_lite_dataset_path(src)
    if dry_run:
        # In-memory only: never write, whatever ``output_path`` says.
        dst = None
    elif output_path is None:
        raise ValueError("refusing to migrate in place; pass --output")
    else:
        dst = Path(output_path)

    summary = MigrationSummary(
        input_path=str(src),
        output_path=str(dst) if dst is not None else None,
        dry_run=dry_run,
    )

    files = _iter_input_files(src)
    for file_path in files:
        _require_allowed_lite_dataset_path(file_path)

    for file_path in files:
        rows, mode = _read_rows(file_path)
        upgraded, verified_count = migrate_rows(
            rows,
            parquet_rows=file_path.suffix == ".parquet",
            verify=verify,
        )
        summary.files += 1
        summary.rows += len(upgraded)
        summary.verified += verified_count

        if dst is not None:
            _write_rows(_target_path(src, file_path, dst), upgraded, mode)

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSON, JSONL, parquet, or directory tree to migrate")
    parser.add_argument(
        "-o",
        "--output",
        help="local output file or directory (required unless --dry-run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "migrate and verify entirely in memory and write NOTHING; "
            "--output is ignored"
        ),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="run devs/migration/verify.py invariants on migrated rows",
    )
    args = parser.parse_args(argv)

    summary = migrate_path(
        args.input,
        args.output,
        dry_run=args.dry_run,
        verify=args.verify,
    )
    print(json.dumps(summary.to_dict(), sort_keys=True))
    return 0


def _iter_input_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"unsupported migration input suffix: {path.suffix}")
        return [path]
    files = [p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix in SUPPORTED_SUFFIXES]
    if not files:
        raise ValueError(f"no supported migration inputs under {path}")
    return files


def _require_allowed_lite_dataset_path(path: Path) -> str:
    dataset = _allowed_lite_dataset_from_path(path)
    if dataset is not None:
        return dataset
    allowed = ", ".join(sorted(ALLOWED_MIGRATION_DATASETS))
    raise ValueError(
        "default migration only accepts the allow-listed published datasets "
        f"{allowed}; refusing out-of-scope input path: {path}"
    )


def _allowed_lite_dataset_from_path(path: Path) -> str | None:
    parts = path.parts

    if _has_out_of_scope_dataset_alias(parts):
        return None

    dataset = _dev_data_lite_dataset_from_parts(parts)
    if dataset is not None:
        return dataset

    return _lite_dataset_from_ancestor_parts(path)


def _has_out_of_scope_dataset_alias(parts: tuple[str, ...]) -> bool:
    return any(
        part.startswith(dataset) and part != dataset
        for part in parts
        for dataset in ALLOWED_MIGRATION_DATASETS
    )


def _dev_data_lite_dataset_from_parts(parts: tuple[str, ...]) -> str | None:
    for index in range(len(parts) - 2):
        if parts[index] == "devs" and parts[index + 1] == "data":
            dataset = _DEV_DATASET_COMPONENTS.get(parts[index + 2])
            if dataset is not None:
                return dataset
    return None


def _lite_dataset_from_ancestor_parts(path: Path) -> str | None:
    parts = path.parts
    ancestor_parts = parts if path.is_dir() else parts[:-1]
    for index, part in reversed(list(enumerate(ancestor_parts))):
        dataset = _lite_dataset_from_dir_name(part)
        if dataset is not None:
            rel_parts = parts[index + 1 :]
            if _is_valid_lite_dataset_relative_path(rel_parts, path.is_dir()):
                return dataset
    return None


def _lite_dataset_from_dir_name(name: str) -> str | None:
    for dataset in ALLOWED_MIGRATION_DATASETS:
        if name == dataset:
            return dataset
    return None


def _is_valid_lite_dataset_relative_path(
    rel_parts: tuple[str, ...],
    is_dir: bool,
) -> bool:
    if not rel_parts:
        return True

    if is_dir:
        return _is_valid_lite_dataset_directory_relpath(rel_parts)

    if len(rel_parts) == 1:
        return Path(rel_parts[0]).suffix in SUPPORTED_SUFFIXES
    if len(rel_parts) == 3:
        platform, task_type, filename = rel_parts
        return (
            platform in _MIGRATION_INPUT_PLATFORMS
            and task_type in _CANONICAL_TASK_TYPES
            and Path(filename).suffix in SUPPORTED_SUFFIXES
        )
    if len(rel_parts) == 4:
        platform, task_type, split, filename = rel_parts
        return (
            platform in _MIGRATION_INPUT_PLATFORMS
            and task_type in _CANONICAL_TASK_TYPES
            and split in _CANONICAL_SPLITS
            and Path(filename).suffix in SUPPORTED_SUFFIXES
        )
    return False


def _is_valid_lite_dataset_directory_relpath(rel_parts: tuple[str, ...]) -> bool:
    if len(rel_parts) == 1:
        return rel_parts[0] in _MIGRATION_INPUT_PLATFORMS
    if len(rel_parts) == 2:
        platform, task_type = rel_parts
        return platform in _MIGRATION_INPUT_PLATFORMS and task_type in _CANONICAL_TASK_TYPES
    if len(rel_parts) == 3:
        platform, task_type, split = rel_parts
        return (
            platform in _MIGRATION_INPUT_PLATFORMS
            and task_type in _CANONICAL_TASK_TYPES
            and split in _CANONICAL_SPLITS
        )
    return False


def _target_path(root: Path, file_path: Path, output_root: Path) -> Path:
    if root.is_file():
        if output_root.exists() and output_root.is_dir():
            return output_root / root.name
        return output_root
    return output_root / _canonical_output_relpath(file_path.relative_to(root))


def _canonical_output_relpath(relpath: Path) -> Path:
    parts = list(relpath.parts)
    for index in range(len(parts) - 1):
        if (
            parts[index] == _LEGACY_WEB_PLATFORM
            and parts[index + 1] in _CANONICAL_TASK_TYPES
        ):
            parts[index] = _BROWSER_PLATFORM
    return Path(*parts)


def _read_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return rows, "jsonl"
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data, "json-list"
        if isinstance(data, dict):
            return [data], "json-single"
        raise ValueError(f"{path} must contain a JSON object or list")
    if path.suffix == ".parquet":
        pd = _load_pandas()
        return pd.read_parquet(path).to_dict(orient="records"), "parquet"
    raise ValueError(f"unsupported migration input suffix: {path.suffix}")


def _write_rows(path: Path, rows: list[dict[str, Any]], mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, separators=(",", ":")))
                f.write("\n")
        return
    if mode == "json-single":
        if len(rows) != 1:
            raise ValueError("json-single output received multiple rows")
        with path.open("w", encoding="utf-8") as f:
            json.dump(rows[0], f, separators=(",", ":"))
        return
    if mode == "json-list":
        with path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, separators=(",", ":"))
        return
    if mode == "parquet":
        from lite.data.staging import write_partition

        write_partition(rows, path)
        return
    raise ValueError(f"unsupported output mode: {mode}")


def _load_pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on optional local env.
        raise RuntimeError("pandas is required to migrate parquet files") from exc
    return pd


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["MigrationSummary", "main", "migrate_path", "migrate_rows"]
