"""The migration tool's input-path gate: which published datasets it will touch.

Run:
    uv run pytest devs/migration/tests/test_path_gate.py -q

``devs/migration`` exists to migrate old HF-uploaded rollout rows from a fixed
allow-list. Its gate decides that by path, and until now **nothing pinned it**
— the allow-list could gain or lose an entry with no test noticing.

The gate is deliberately conservative and exact-name based. The migration scope
is the five HF-uploaded repos we are rewriting: ``Lite.OSWorld``,
``Lite.CUAGym``, ``Lite.CUAWorld``, ``Lite.ScaleCUA``, and ``WebGym``.
Similar-looking names such as ``WebGymRT`` and fresh-preproc ``ScaleCUA`` are
still out of scope.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_module():
    path = _REPO_ROOT / "devs" / "migration" / "run.py"
    spec = importlib.util.spec_from_file_location("_migration_run_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("relpath", "expected"),
    [
        # Every component of devs/data/ that holds generated data.
        ("devs/data/lite.osworld/x.parquet", "Lite.OSWorld"),
        ("devs/data/lite.cuagym/x.parquet", "Lite.CUAGym"),
        ("devs/data/lite.cuaworld/x.parquet", "Lite.CUAWorld"),
        ("devs/data/lite.scalecua/x.parquet", "Lite.ScaleCUA"),
        ("devs/data/webgym/x.parquet", "WebGym"),
        # Historical legacy-source WebGym inputs may still arrive under old
        # web/use partitions; upgrade.py owns rewriting row metadata to
        # browser/use on output.
        ("devs/data/webgym/out/web/use/train/x.parquet", "WebGym"),
        ("WebGym/web/use/train/x.parquet", "WebGym"),
        ("WebGym/browser/use/train/x.parquet", "WebGym"),
        # Non-target WebGym-like names and dotted scratch aliases remain out of scope.
        ("WebGymRT/devs/data/lite.cuagym/x.parquet", None),
        ("WebGymRT/Lite.OSWorld/old.jsonl", None),
        ("WebGymRT/Lite.OSWorld/desktop/use/old.jsonl", None),
        ("WebGymTest/Lite.OSWorld/old.jsonl", None),
        ("WebGymRT/x.parquet", None),
        ("WebGym.RT/x.parquet", None),
        ("WebGym.copy/x.parquet", None),
        # Canonical partition layout under a dataset-named ancestor.
        ("Lite.OSWorld/desktop/use/train/x.parquet", "Lite.OSWorld"),
        ("Lite.OSWorld/browser/use/train/x.parquet", "Lite.OSWorld"),
        ("Lite.OSWorld.copy/desktop/use/train/x.parquet", None),
        # ScaleCUA without the Lite.* published-route name is fresh preproc, not migration.
        ("ScaleCUA/desktop/use/train/x.parquet", None),
        ("cua-lite/ScaleCUA/desktop/use/train/x.parquet", None),
        ("cua-lite/Lite.ScaleCUA/desktop/use/train/x.parquet", "Lite.ScaleCUA"),
        # Not ours to migrate: upstream corpora and unrelated paths.
        ("devs/data/aguvis/x.parquet", None),
        ("devs/data/scalecua/x.parquet", None),
        ("devs/data/utils.py", None),
        ("/tmp/somewhere/x.parquet", None),
    ],
)
def test_path_gate_resolves_exactly_the_datasets_we_generate(
    relpath: str, expected: str | None
) -> None:
    module = _run_module()

    assert module._allowed_lite_dataset_from_path(Path(relpath)) == expected


def test_dev_data_component_map_is_the_published_migration_scope() -> None:
    """The migration scope is a fixed published-dataset allow-list, not a devs/data sweep."""
    module = _run_module()
    expected = {
        "lite.osworld": "Lite.OSWorld",
        "lite.cuagym": "Lite.CUAGym",
        "lite.cuaworld": "Lite.CUAWorld",
        "lite.scalecua": "Lite.ScaleCUA",
        "webgym": "WebGym",
    }

    assert module._DEV_DATASET_COMPONENTS == expected
    assert module.ALLOWED_MIGRATION_DATASETS == frozenset(expected.values())


def test_allow_list_and_component_map_agree() -> None:
    """Two copies of one fact; nothing else keeps them in step.

    A component mapped to a dataset name absent from the allow-list resolves to a name
    the gate then rejects downstream — a silent refusal with a confusing message.
    """
    module = _run_module()

    assert set(module._DEV_DATASET_COMPONENTS.values()) <= set(module.ALLOWED_MIGRATION_DATASETS)


def test_path_gate_keeps_legacy_web_local_to_migration_input() -> None:
    module = _run_module()

    assert "browser" in module._CANONICAL_PLATFORMS
    assert "web" not in module._CANONICAL_PLATFORMS
    assert module._MIGRATION_INPUT_PLATFORMS == module._CANONICAL_PLATFORMS | {"web"}
