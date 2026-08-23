"""Provenance keys survive the filter → stage pipeline (metadata contract section 11.P P6).

A trajectory row's ``metadata.others`` carries the five run-provenance keys
(``model_id``/``agent_id``/``config_path``/``commit``/``command``); this pins
that both post-collection steps pass them through verbatim:

  * ``devs/data/lite.osworld/filter.py`` — ``_process_file`` copies the
    metadata column untouched (only ``messages`` is rewritten);
  * ``lite.data.hf.stage`` — serializes the metadata verbatim into the staged
    parquet's metadata column.

Run: uv run pytest tests/data/hf/test_provenance_roundtrip.py -v
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

from lite.core import LiteCUAMetadata
from lite.core.tools import make_tool_call
from lite.data.hf.stage import stage
from lite.utils.parquet import write_records_to_parquet
from lite.utils.path import project_root

PROVENANCE = {
    "model_id": "gpt-5.5",
    "agent_id": "gpt.teacher",
    "config_path": "scripts/configs/x.yaml",
    "commit": "abc1234",
    "command": "uv run python scripts/rollout.py --model-id gpt-5.5",
}


def _load_filter_module():
    path = (
        project_root()
        / "devs"
        / "data"
        / "lite.osworld"
        / "filter.py"
    )
    spec = importlib.util.spec_from_file_location("losworld_filter", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["losworld_filter"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_fixture_row(sample_dir: Path) -> None:
    record = {
        "images": [],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "go"}], "tool_calls": []},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}],
             "tool_calls": [make_tool_call(
                 "computer",
                 {"actions": [{"action": "click", "coordinate": [5, 5]}]},
                 call_id="call_0000",
             )]},
            {"role": "tool", "tool_call_id": "call_0000",
             "content": [{"type": "text", "text": "clicked"}]},
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            others={
                "domain": "calc",
                "env_id": "lite.osworld",
                "task_id": "t1",
                "episode_return": 1.0,
                "terminated": True,
                "truncated": False,
                **PROVENANCE,
            },
        ).to_dict(),
    }
    sample_dir.mkdir(parents=True, exist_ok=True)
    write_records_to_parquet([record], sample_dir / "trajectory.parquet")


def test_provenance_survives_filter_then_stage(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    _write_fixture_row(raw_root / "train" / "t1" / "sample_00")

    # --- filter leg: the real _process_file (metadata must pass through) ---
    fmod = _load_filter_module()
    clean_root = tmp_path / "clean"
    src = raw_root / "train" / "t1" / "sample_00" / "trajectory.parquet"
    dst = clean_root / "train" / "t1" / "sample_00" / "trajectory.parquet"
    *_, wrote = fmod._process_file(
        src, dst, noop=frozenset(fmod.DEFAULT_NOOP_ACTIONS), strip_noop_save=False,
        output_root=clean_root,
        drop_loops=True, drop_undo_storm=True, drop_no_submit=False,
        undo_storm_min=4, ensure_terminate=False, collapse_reasoning=False,
    )
    assert wrote
    raw_md = pd.read_parquet(dst).iloc[0]["metadata"]
    filtered = json.loads(raw_md) if isinstance(raw_md, str) else dict(raw_md)
    f_others = dict(filtered["others"])
    for key, value in PROVENANCE.items():
        assert f_others[key] == value, f"filter dropped/changed {key}"

    # --- stage leg: real lite.data.hf.stage over the filtered root ---
    out_dir = tmp_path / "staged"
    stage([clean_root], name="ProvTest", out_dir=out_dir, filter_expr=None)
    staged_files = list(out_dir.rglob("*.parquet"))
    assert staged_files, "stage produced no parquet"
    row = pd.read_parquet(staged_files[0]).iloc[0]
    md = row["metadata"]
    md = json.loads(md) if isinstance(md, str) else dict(md)
    s_others = dict(md["others"])
    for key, value in PROVENANCE.items():
        assert s_others[key] == value, f"stage dropped/changed {key}"
    assert s_others["task_id"] == "t1"
    assert s_others["env_id"] == "lite.osworld"
    assert float(s_others["episode_return"]) == 1.0
