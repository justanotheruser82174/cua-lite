"""Env metadata boundary emits canonical nested Lite tool schemas."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lite.core.tools.schemas import tool_schema_name, validate_extra_tool_schemas
from lite.gym.errors import EnvDepsMissingError
from lite.gym.sandbox import SandboxTaskConfig


def _assert_nested_wire_metadata(metadata) -> None:
    wire = metadata.to_dict()
    schemas = wire["extra_tool_schemas"]
    validate_extra_tool_schemas(schemas)
    for schema in schemas:
        assert set(schema) <= {"type", "function"}
        assert {"name", "description", "parameters"}.isdisjoint(schema)


def test_mobilegym_opt_in_extra_tool_metadata_serializes_nested() -> None:
    from lite.gym.envs.mobilegym.main import RemoteMobileGymEnv, _load_tasks_json

    task_id, meta = next(iter(_load_tasks_json().items()))
    env = RemoteMobileGymEnv(
        task_id=task_id,
        difficulty=meta["difficulty"],
        scope=meta["scope"],
        objective=meta["objective"],
        composition=meta["composition"],
        capabilities=meta["capabilities"],
        task_apps=meta["apps"],
        answer_fields=meta["answer_fields"],
        base_max_steps=meta["max_steps"],
        extra_tools=["open_app", "response"],
    )

    _assert_nested_wire_metadata(env.metadata)
    assert [tool_schema_name(schema) for schema in env.metadata.extra_tool_schemas] == [
        "open_app",
        "response",
    ]


def test_browsergym_action_subset_metadata_serializes_nested() -> None:
    try:
        from lite.gym.envs.browsergym.main import BrowserGymEnv
    except EnvDepsMissingError as exc:
        pytest.skip(f"browsergym unavailable: {exc}")

    metadata = BrowserGymEnv._task_metadata(
        "webarena",
        "webarena.0",
        action_subsets=("coord", "chat", "infeas", "nav", "tab"),
        viewport=(1280, 720),
    )

    _assert_nested_wire_metadata(metadata)
    assert {"response", "terminate", "goto", "back"} <= {
        tool_schema_name(schema) for schema in metadata.extra_tool_schemas
    }


def test_lite_osworld_checked_catalog_metadata_serializes_nested() -> None:
    try:
        from lite.gym.envs.lite.osworld.main import (
            _DATA_DIR,
            LiteOsworldEnv,
            _catalog_lock_entries,
            _check_catalog_entry,
        )
    except EnvDepsMissingError as exc:
        pytest.skip(f"lite.osworld unavailable: {exc}")

    entries = _catalog_lock_entries()
    _check_catalog_entry("eval", entries["eval"])
    row = json.loads((_DATA_DIR / Path(entries["eval"]["path"])).read_text().splitlines()[0])
    task = SandboxTaskConfig(
        task_id=row["task_id"],
        instruction=row["instruction"],
        computer={},
        max_steps=row.get("max_steps", 15),
        metadata=row["metadata"],
        platform="desktop",
    )
    try:
        env = LiteOsworldEnv(task=task, extra_tools=["report_infeasible"])
    except EnvDepsMissingError as exc:
        pytest.skip(f"lite.osworld runtime deps unavailable: {exc}")

    _assert_nested_wire_metadata(env.metadata)
    assert [tool_schema_name(schema) for schema in env.metadata.extra_tool_schemas] == [
        "report_infeasible",
    ]
