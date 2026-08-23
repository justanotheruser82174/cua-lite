from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from lite.core.tools import make_tool_call

ROOT = Path(__file__).resolve().parents[3]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _batched_computer_call() -> dict:
    return make_tool_call(
        "computer",
        {
            "actions": [
                {"action": "click", "coordinate": [1, 2]},
                {"action": "wait", "duration": 1},
            ],
        },
        call_id="call_0000",
    )


def test_devs_data_utils_reads_canonical_nested_calls_and_batches() -> None:
    data_utils = _load_module(
        ROOT / "devs/data/utils.py",
        "devs_data_utils_canonical_under_test",
    )

    message = {
        "role": "assistant",
        "tool_calls": [
            _batched_computer_call(),
            make_tool_call("response", {"text": "done"}, call_id="call_0001"),
        ],
    }

    assert data_utils._iter_action_items(message) == [
        ("click", {"coordinate": [1, 2]}),
        ("wait", {"duration": 1}),
        ("response", {"text": "done"}),
    ]


def test_devs_data_utils_rejects_bare_agent_wire_calls() -> None:
    data_utils = _load_module(
        ROOT / "devs/data/utils.py",
        "devs_data_utils_bare_under_test",
    )

    message = {
        "role": "assistant",
        "tool_calls": [{"name": "click", "arguments": {}}],
    }

    with pytest.raises(KeyError, match="function"):
        data_utils._iter_action_items(message)


def test_cuaworld_analyzer_reads_canonical_nested_batches() -> None:
    analyze = _load_module(
        ROOT / "devs/data/lite.cuaworld/analyze.py",
        "lite_cuaworld_analyze_canonical_under_test",
    )

    assert analyze._wait_pattern([
        {"role": "assistant", "tool_calls": [_batched_computer_call()]},
    ]) == (0, 1)


def test_cuaworld_analyzer_rejects_bare_agent_wire_calls() -> None:
    analyze = _load_module(
        ROOT / "devs/data/lite.cuaworld/analyze.py",
        "lite_cuaworld_analyze_bare_under_test",
    )
    messages = [{
        "role": "assistant",
        "tool_calls": [{"name": "wait", "arguments": {}}],
    }]

    with pytest.raises(KeyError, match="function"):
        analyze._wait_pattern(messages)
