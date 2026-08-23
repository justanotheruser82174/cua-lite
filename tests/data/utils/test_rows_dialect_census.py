"""Shared dialect-finish vocabulary census for row and migration validators.

The row validator and the one-time migration verifier intentionally redeclare
the same dialect-only finish spellings at separate boundaries. This test stays
under ``tests/data/utils`` because it pins the shared row-validator vocabulary
against the migration copy without teaching either owner to import the other's
private table.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from lite.core.tools import extra_tools as extra_tools_module
from lite.core.tools.extra_tools import LiteFinishToolSet
from lite.data.utils import rows as rows_module


def _extra_tool_names_table(action_space: object) -> dict:
    """Read a family's provider-value -> Lite extra-tool-name table by declaration."""
    for name in dir(action_space):
        if name.endswith("_TO_EXTRA_TOOL_NAMES"):
            return getattr(action_space, name) or {}
    return {}


def _native_finish_tool_aliases_from_families() -> dict[str, frozenset[str]]:
    """Recompute dialect finish spellings from the model families that declare them."""
    from lite.agents.bootstrap import register_all
    from lite.agents.core.action_space.base import BaseActionSpace

    register_all()
    finish_names = LiteFinishToolSet.get_tool_names()
    aliases: dict[str, set[str]] = {}
    pending = [BaseActionSpace]
    seen: set[type] = set()
    while pending:
        action_space = pending.pop()
        if action_space in seen:
            continue
        seen.add(action_space)
        pending.extend(action_space.__subclasses__())
        for native, canonicals in _extra_tool_names_table(action_space).items():
            if native in finish_names:
                continue
            targets = frozenset(canonicals) & finish_names
            if targets:
                aliases.setdefault(native, set()).update(targets)
    return {native: frozenset(targets) for native, targets in aliases.items()}


def test_native_finish_tool_aliases_are_derived_from_every_model_family():
    assert (
        rows_module._NATIVE_FINISH_TOOL_ALIASES
        == _native_finish_tool_aliases_from_families()
    )
    assert set(rows_module._NATIVE_FINISH_TOOL_ALIASES) == {
        "ABORT",
        "COMPLETE",
        "INFO",
        "answer",
        "call_user",
        "finished",
    }
    assert rows_module._NATIVE_FINISH_TOOL_ALIASES["finished"] == frozenset(
        {"response", "terminate"}
    )


def _import_migration_verify():
    path = Path(__file__).resolve().parents[3] / "devs" / "migration" / "verify.py"
    spec = importlib.util.spec_from_file_location("_migration_verify_census", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_verifier_dialect_census_matches_every_model_family():
    verify = _import_migration_verify()
    families = _native_finish_tool_aliases_from_families()

    assert verify._DIALECT_ONLY_TOOL_ALIASES == families
    assert set(verify._DIALECT_ONLY_TOOL_ALIASES) == {
        "ABORT",
        "COMPLETE",
        "INFO",
        "answer",
        "call_user",
        "finished",
    }
    for name in sorted(verify._DIALECT_ONLY_TOOL_ALIASES):
        with pytest.raises(ValueError, match="is dialect-only"):
            verify._reject_dialect_only_tool_name(name, "tool_call")


def test_dialect_finish_spellings_are_disjoint_from_core_finish_tools():
    aliases = rows_module._NATIVE_FINISH_TOOL_ALIASES
    assert set().union(*aliases.values()) <= LiteFinishToolSet.get_tool_names()
    assert not (frozenset(aliases) & LiteFinishToolSet.get_tool_names())
    assert not hasattr(extra_tools_module, "DIALECT_TOOL_ALIASES")
    assert "DIALECT_TOOL_ALIASES" not in Path(
        extra_tools_module.__file__
    ).read_text()
