"""Common preproc guards for wrapper tool/result contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from lite.core.tools.calls import tool_call_arguments, tool_call_id, tool_call_name
from lite.core.tools.schemas import tool_schema_name
from lite.data.utils.rows import validate_canonical_rows

_ROOT = Path(__file__).resolve().parents[3]


def _load_preproc_script(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relpath)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_DONE = [{"type": "text", "text": "Done."}]


def _assert_canonical_call_ids(messages: list[dict]) -> None:
    seen: set[str] = set()
    for message in messages:
        for call in message.get("tool_calls") or []:
            assert call["type"] == "function"
            call_id = tool_call_id(call)
            assert call_id is not None
            assert call_id not in seen
            seen.add(call_id)


def _assert_first_action_result_is_tool(row: dict) -> None:
    messages = row["messages"]
    first_call = messages[1]["tool_calls"][0]
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": tool_call_id(first_call),
        "content": [{"type": "image", "index": 1}],
    }
    _assert_canonical_call_ids(messages)


def _all_calls(row: dict) -> list[dict]:
    return [
        call
        for message in row["messages"]
        for call in (message.get("tool_calls") or [])
    ]


def _assert_structural_done_row(row: dict) -> None:
    """Assert the structural content-only completion form.

    Used when the source supplied a terminal marker/status but no final
    executable label. EOF executable labels use :func:`_assert_final_action_row`
    instead.
    """
    assert row["messages"][-1] == {"role": "assistant", "content": _DONE}
    assert all(tool_call_name(call) != "terminate" for call in _all_calls(row))
    assert all(
        tool_schema_name(schema) != "terminate"
        for schema in row["metadata"]["extra_tool_schemas"]
    )
    validate_canonical_rows([row], "structural-done")


def _assert_final_action_row(row: dict) -> None:
    """Assert a row ending on an unobserved final assistant tool-call label."""
    final = row["messages"][-1]
    assert final["role"] == "assistant"
    assert final.get("tool_calls")
    assert final.get("content") != _DONE
    _assert_canonical_call_ids(row["messages"])
    validate_canonical_rows([row], "final-action-eof")


def _assert_terminate_outcome(row: dict, *, status: str, reason: str | None = None) -> None:
    """The dropped ``terminate``'s payload, in the one place it still survives.

    ``terminate_status`` is emitted only for a non-success status, and
    ``terminate_reason`` only for non-blank authored text.
    """
    others = row["metadata"]["others"]
    assert others["terminate_status"] == status
    assert others.get("terminate_reason") == reason


def _assert_no_terminate_outcome(row: dict) -> None:
    """A success (or absent) terminator asserts nothing ``Done.`` does not."""
    others = row["metadata"]["others"]
    assert "terminate_status" not in others
    assert "terminate_reason" not in others


def _actions(calls: list[dict]) -> list[dict]:
    return [
        action
        for call in calls
        for action in tool_call_arguments(call).get("actions", [])
    ]
