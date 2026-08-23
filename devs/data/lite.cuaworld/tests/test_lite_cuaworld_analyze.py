"""Tests for the Lite.CUAWorld collection analyzer.

Run: uv run pytest devs/data/lite.cuaworld/tests/test_lite_cuaworld_analyze.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from lite.core import LiteCUAMetadata
from lite.core.tools.calls import make_tool_call

_HERE = Path(__file__).resolve().parents[1]


def _load_analyze() -> ModuleType:
    path = _HERE / "analyze.py"
    spec = importlib.util.spec_from_file_location("lite_cuaworld_analyze", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cuaworld_analyzer_wait_pattern_descends_into_batches() -> None:
    analyze = _load_analyze()

    msgs = [
        {
            "role": "assistant",
            "tool_calls": [make_tool_call("wait", {"duration": 1})],
        },
        {
            "role": "assistant",
            "tool_calls": [
                make_tool_call(
                    "computer",
                    {
                        "actions": [
                            {"action": "click", "coordinate": [1, 2]},
                            {"action": "wait", "duration": 1},
                        ],
                    },
                )
            ],
        },
    ]

    assert analyze._wait_pattern(msgs) == (1, 1)


def _write_sample(
    sample_dir: Path,
    *,
    metadata: dict,
    messages: list[dict] | None = None,
) -> None:
    sample_dir.mkdir(parents=True)
    (sample_dir / "summary.json").write_text(
        json.dumps(
            {
                "episode_return": 1.0,
                "n_turns": 1,
                "terminated": True,
                "truncated": False,
            }
        )
    )
    if messages is None:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    make_tool_call("wait", {"duration": 1}),
                ],
            },
        ]
    pd.DataFrame(
        {
            "messages": [json.dumps(messages)],
            "metadata": [json.dumps(metadata)],
        }
    ).to_parquet(sample_dir / "trajectory.parquet", index=False)


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param(
            LiteCUAMetadata(
                dims=("desktop", "use"),
                extra_tool_schemas=[],
                valid_actions=None,
                others={"env_id": "lite.cuaworld.writer"},
            ).to_dict(),
            id="canonical-tagged",
        ),
        pytest.param(
            {"others": {"env_id": "lite.cuaworld.writer"}},
            id="historical-untagged-others",
        ),
    ],
)
def test_cuaworld_analyzer_accepts_json_string_parquet_cells(
    tmp_path: Path,
    capsys,
    metadata: dict,
) -> None:
    analyze = _load_analyze()
    sample_dir = tmp_path / "task_ok" / "sample_00"
    _write_sample(sample_dir, metadata=metadata)

    analyze.analyze(tmp_path)

    out = capsys.readouterr().out
    assert "samples: 1" in out
    assert "writer" in out


def test_cuaworld_analyzer_uses_metadata_others_env_id(tmp_path: Path, capsys) -> None:
    analyze = _load_analyze()
    sample_dir = tmp_path / "task_ok" / "sample_00"
    _write_sample(
        sample_dir,
        metadata={
            "env_id": "lite.cuaworld.stale_top_level",
            "others": {"env_id": "lite.cuaworld.writer"},
        },
    )

    analyze.analyze(tmp_path)

    out = capsys.readouterr().out
    assert "writer" in out
    assert "stale_top_level" not in out
