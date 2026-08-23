"""Tests for migration upgrade.py legacy input repair."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from lite.core.tools.calls import make_tool_call, tool_call_name

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_migration_module(filename: str):
    path = _PROJECT_ROOT / "devs" / "migration" / filename
    spec = importlib.util.spec_from_file_location(f"cua_lite_migration_{filename}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clean_nones(item):
    if isinstance(item, dict):
        return {k: _clean_nones(v) for k, v in item.items() if v is not None}
    if isinstance(item, list):
        return [_clean_nones(i) for i in item]
    return item


def _old_jsonl_use_row() -> dict:
    return {
        "images": ["screen0.png", "screen1.png"],
        "metadata": {
            "platform": "desktop",
            "task_type": "use",
            "valid_actions": ["click", "response"],
        },
        "messages": [
            {"role": "user", "content": [{"type": "image", "index": 0}]},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "click",
                            "arguments": {"coordinate": [10, 20]},
                        },
                    }
                ],
            },
            {"role": "user", "content": [{"type": "image", "index": 1}]},
        ],
    }


def test_upgrade_coerces_legacy_null_and_missing_tool_call_arguments():
    """Issue #46: old rows stored no-arg tools as ``arguments: null``, and
    ``clean_nones`` then dropped the key outright. The live repair is the
    upgrader's own ``_nest_call`` / ``_coerce_arguments``, which also parse the
    JSON-string form that the old standalone helper could not.
    """
    upgrade = _load_migration_module("upgrade.py")

    null_call = {"type": "function", "function": {"name": "back", "arguments": None}}
    stripped = _clean_nones([{"role": "assistant", "tool_calls": [null_call]}])
    assert "arguments" not in stripped[0]["tool_calls"][0]["function"]

    assert upgrade._nest_call(null_call, platform="desktop") == make_tool_call("back", {})
    assert upgrade._nest_call(
        stripped[0]["tool_calls"][0],
        platform="desktop",
    ) == make_tool_call("back", {})
    assert upgrade._nest_call({"function": {"name": "noop"}}, platform="desktop") == make_tool_call(
        "noop",
        {},
    )
    assert upgrade._nest_call(
        {"function": {"name": "goto", "arguments": '{"url": "x"}'}},
        platform="desktop",
    ) == make_tool_call("goto", {"url": "x"})

    with_id = upgrade._nest_call(
        {
            "call_id": "legacy_0",
            "name": "goto",
            "arguments": {"url": "https://example.com"},
        },
        platform="desktop",
    )
    assert list(with_id) == ["id", "type", "function"]
    assert with_id == make_tool_call(
        "goto",
        {"url": "https://example.com"},
        call_id="legacy_0",
    )
    assert not hasattr(upgrade, "_is_already_canonical")


def test_upgrade_maps_historical_web_platform_to_browser_output():
    upgrade = _load_migration_module("upgrade.py")
    verify = _load_migration_module("verify.py")
    row = _old_jsonl_use_row()
    row["metadata"]["platform"] = "web"

    migrated = upgrade.upgrade_lite_sample(row)

    assert migrated["metadata"]["dims"] == ["browser", "use"]
    assert "platform" not in migrated["metadata"]
    assert tool_call_name(migrated["messages"][1]["tool_calls"][0]) == "computer"
    verify.verify_lite_sample(migrated)


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        (
            {
                "name": "response",
                "description": "Submit an answer.",
                "parameters": {"type": "object", "properties": {}, "required": []},
                "strict": True,
            },
            True,
        ),
        (
            {
                "type": "function",
                "function": {
                    "name": "response",
                    "description": "Submit an answer.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                    "strict": False,
                },
                "strict": True,
            },
            False,
        ),
    ],
)
def test_upgrade_moves_legacy_schema_strict_to_canonical_function_scope(
    schema: dict,
    expected: bool,
):
    upgrade = _load_migration_module("upgrade.py")

    nested = upgrade._nest_schema(schema)

    assert nested["function"]["strict"] is expected
    assert "strict" not in nested


def test_migration_coerces_legacy_hf_materialized_message_padding():
    upgrade = _load_migration_module("upgrade.py")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": None, "text": None},
                {"type": "text", "text": "finish", "index": None},
            ],
            "tool_calls": None,
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Done.", "index": None}],
            "tool_calls": [],
        },
    ]

    assert upgrade.coerce_legacy_materialized_messages(messages) == [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": None},
                {"type": "text", "text": "finish"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
    ]
