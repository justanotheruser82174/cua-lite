"""Tests for the WebGym data quality report.

This is a standalone dev script test. ``quality_check.py`` imports its
co-located ``filter`` module by bare name, so the loader temporarily puts this
parent directory on ``sys.path``.

Run: uv run pytest devs/data/webgym/tests/test_webgym_quality_check.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from lite.core.tools.action_space import LiteDesktopActionSet
from lite.core.tools.calls import make_tool_call, tool_call_arguments

_HERE = Path(__file__).resolve().parents[1]


def _load_quality_check() -> ModuleType:
    path = _HERE / "quality_check.py"
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("webgym_quality_check", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(path.parent))


def test_webgym_quality_noop_set_is_derived_from_the_canonical_catalog() -> None:
    """The webgym no-op set is the canonical desktop catalog minus what webgym's
    action translator executes, never a hand-maintained parallel list.
    """
    quality = _load_quality_check()

    assert quality._NOOP <= LiteDesktopActionSet.get_action_names()
    assert quality._NOOP == {
        "cursor_position",
        "hold_key",
        "key_down",
        "key_up",
        "mouse_down",
        "mouse_up",
        "screenshot",
        "wait",
    }
    # Pre-refactor names that can never match a canonical action must be gone
    # (a double/triple click is ``click`` with ``clicks``; scroll direction is an
    # argument), and executed verbs must not be counted as wasted steps.
    for gone in (
        "double_click",
        "triple_click",
        "right_click",
        "middle_click",
        "hscroll",
        "mouse_move",
        "click",
        "type",
        "key",
        "scroll",
        "drag",
    ):
        assert gone not in quality._NOOP


def test_webgym_quality_action_stream_descends_only_into_action_batches() -> None:
    quality = _load_quality_check()
    msg = {
        "role": "assistant",
        "tool_calls": [
            make_tool_call(
                "computer",
                {
                    "actions": [
                        {"action": "click", "coordinate": [10, 20]},
                        {"action": "type", "text": "query"},
                    ],
                },
            ),
            make_tool_call("goto", {"url": "https://example.com"}),
            make_tool_call("response", {"text": "done"}),
        ],
    }

    assert quality._iter_action_items(msg) == [
        ("click", {"coordinate": [10, 20]}),
        ("type", {"text": "query"}),
        ("goto", {"url": "https://example.com"}),
        ("response", {"text": "done"}),
    ]
    # Standalone-tool detection reads the top level only.
    assert quality._iter_top_level_calls(msg) == [
        ("computer", tool_call_arguments(msg["tool_calls"][0])),
        ("goto", {"url": "https://example.com"}),
        ("response", {"text": "done"}),
    ]


def test_webgym_quality_rejects_standalone_tool_nested_in_an_action_batch() -> None:
    quality = _load_quality_check()
    msg = {
        "role": "assistant",
        "tool_calls": [
            make_tool_call(
                "computer",
                {"actions": [{"action": "response", "text": "x"}]},
            ),
        ],
    }
    with pytest.raises(ValueError, match="must not be nested"):
        quality._iter_action_items(msg)


def test_webgym_quality_success_trajs_accepts_untagged_metadata(tmp_path: Path) -> None:
    quality = _load_quality_check()
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "goal"}]},
        {"role": "assistant", "content": [{"type": "action_description", "text": "Done."}]},
    ]
    metadata = {"others": {"episode_return": 1.0, "task_id": "task_ok"}}
    pd.DataFrame(
        {
            "messages": [json.dumps(messages)],
            "metadata": [json.dumps(metadata)],
        }
    ).to_parquet(tmp_path / "trajectory.parquet", index=False)

    got = list(quality._success_trajs(tmp_path))

    assert got == [(messages, metadata["others"])]


def test_webgym_quality_success_trajs_reads_episode_return_from_others(
    tmp_path: Path,
) -> None:
    quality = _load_quality_check()
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "goal"}]},
        {"role": "assistant", "content": [{"type": "action_description", "text": "Done."}]},
    ]
    metadata = {
        "metadata_kind": "cua",
        "dims": ["browser", "use"],
        "extra_tool_schemas": [],
        "valid_actions": None,
        "others": {"episode_return": 1.0, "difficulty": 2},
    }
    pd.DataFrame(
        {
            "messages": [json.dumps(messages)],
            "metadata": [json.dumps(metadata)],
        }
    ).to_parquet(tmp_path / "trajectory.parquet", index=False)

    got = list(quality._success_trajs(tmp_path))

    assert got == [(messages, metadata["others"])]
