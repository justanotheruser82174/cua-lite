"""Tests for zero-arg tool-call parquet and debug-log checks.

Validates that empty tool-call arguments remain present as ``{}`` through
trajectory parquet storage and debug log validation.

Run::

    uv run pytest tests/utils/test_parquet_zero_arg_tool_calls.py -v
"""
from __future__ import annotations

from pathlib import Path

from lite.core.tools.calls import (
    make_tool_call,
    tool_call_arguments,
    tool_call_name,
)
from lite.core.tools.schemas import make_tool_schema
from lite.utils.parquet import _EMPTY_STRUCT_CHILD, write_records_to_parquet


def _metadata(
    *,
    task_type: str = "use",
    platform: str = "desktop",
    extra_tool_schemas: list[dict] | None = None,
    terminated: bool | None = None,
    truncated: bool | None = None,
) -> dict:
    metadata = {
        "metadata_kind": "cua",
        "dims": [platform, task_type],
        "extra_tool_schemas": extra_tool_schemas or [],
        "valid_actions": None,
        "others": {
            "env_id": "test.env",
            "task_id": "task_0",
        },
    }
    if terminated is not None:
        metadata["others"]["terminated"] = terminated
    if truncated is not None:
        metadata["others"]["truncated"] = truncated
    return metadata


def _write_trajectory(
    sample_dir: Path,
    messages: list[dict],
    *,
    metadata: dict | None = None,
    images: list[str] | None = None,
    json_fields: tuple[str, ...] | None = None,
) -> Path:
    sample_dir.mkdir(parents=True, exist_ok=True)
    path = sample_dir / "trajectory.parquet"
    write_records_to_parquet([{
        "images": images or [],
        "messages": messages,
        "metadata": metadata or _metadata(),
    }], path, json_fields=json_fields)
    return path


def _text(text: str) -> list[dict]:
    return [{"type": "text", "text": text}]


def _browser_nav_metadata() -> dict:
    return _metadata(
        platform="browser",
        extra_tool_schemas=[
            make_tool_schema(
                "back",
                description="back",
                parameters={"type": "object", "properties": {}, "required": []},
            ),
            make_tool_schema(
                "goto",
                description="goto",
                parameters={
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            ),
        ],
        terminated=True,
    )


def _zero_arg_trajectory_messages() -> list[dict]:
    return [
        {"role": "user", "content": _text("go back")},
        {"role": "assistant", "content": [], "tool_calls": [
            make_tool_call(
                "goto",
                {"url": "https://example.com"},
                call_id="call_0000",
            ),
        ]},
        {"role": "tool", "tool_call_id": "call_0000", "content": _text("ok")},
        {"role": "assistant", "content": [], "tool_calls": [
            make_tool_call("back", {}, call_id="call_0001"),
        ]},
        {"role": "tool", "tool_call_id": "call_0001", "content": _text("ok")},
        {"role": "assistant", "content": _text("Done.")},
    ]


class TestZeroArgToolCallRoundTrip:
    def test_zero_arg_call_passes_the_log_contract(self, tmp_path: Path):
        """End-to-end reproduction of the field defect and its fix.

        Pre-fix this produced exactly the observed pair of violations per
        zero-arg call:
          ``...tool_calls[0].arguments must be an object``  (``{}`` -> ``null``)
          ``...orphan role:"tool" ...``                     (the cascade)
        Post-fix: zero violations."""
        from lite.infer.debug.log_contract import check_log_contract

        sample_dir = tmp_path / "sample_00"
        _write_trajectory(sample_dir, _zero_arg_trajectory_messages(),
                          metadata=_browser_nav_metadata())

        assert check_log_contract(sample_dir) == []

    def test_zero_arg_call_reads_back_as_a_dict_not_none(self, tmp_path: Path):
        """The representation requirement itself: ``{}`` must never come back as
        ``None``. It is stored as a PRESENT struct, which pandas materialises as
        a dict — distinguishable from a null struct, which materialises as
        ``None``."""
        import pandas as pd

        sample_dir = tmp_path / "sample_00"
        path = _write_trajectory(sample_dir, _zero_arg_trajectory_messages(),
                                 metadata=_browser_nav_metadata())

        messages = pd.read_parquet(path).to_dict("records")[0]["messages"]
        calls = [c for m in messages for c in (m.get("tool_calls") or [])]
        by_name = {tool_call_name(c): c for c in calls}

        assert tool_call_arguments(by_name["back"]) is not None
        assert isinstance(tool_call_arguments(by_name["back"]), dict)
        assert tool_call_arguments(by_name["goto"])["url"] == "https://example.com"

    def test_all_zero_arg_calls_still_write(self, tmp_path: Path):
        """The genuine ``struct<>`` case — every value at the ``arguments`` path
        is ``{}``, so PyArrow has no child field to infer and ``pq.write_table``
        raises ``ArrowNotImplementedError``. This is the constraint the old
        blanket rewrite existed for, so the fix must still handle it: a single
        reserved null child is injected, and the value still reads back as a
        dict rather than ``None``."""
        import pandas as pd

        from lite.infer.debug.log_contract import check_log_contract

        sample_dir = tmp_path / "sample_00"
        path = _write_trajectory(sample_dir, [
            {"role": "user", "content": _text("go back")},
            {"role": "assistant", "content": [], "tool_calls": [
                make_tool_call("back", {}, call_id="call_0000"),
            ]},
            {"role": "tool", "tool_call_id": "call_0000", "content": _text("ok")},
            {"role": "assistant", "content": _text("Done.")},
        ], metadata=_browser_nav_metadata())

        messages = pd.read_parquet(path).to_dict("records")[0]["messages"]
        call = [c for m in messages for c in (m.get("tool_calls") or [])][0]

        assert isinstance(tool_call_arguments(call), dict)
        assert tool_call_arguments(call) == {_EMPTY_STRUCT_CHILD: None}
        assert check_log_contract(sample_dir) == []
