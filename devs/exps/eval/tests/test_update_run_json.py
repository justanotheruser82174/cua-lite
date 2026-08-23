from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
UPDATE_RUN_JSON = ROOT / "devs" / "exps" / "eval" / "utils" / "update_run_json.py"
SPEC = importlib.util.spec_from_file_location("update_run_json_under_test", UPDATE_RUN_JSON)
assert SPEC is not None and SPEC.loader is not None
update_run_json = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(update_run_json)
result_from_summary = update_run_json.result_from_summary


def _summary(path: Path, *, model: str = "Qwen/Qwen3.5-4B") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    summary = path / "summary.json"
    summary.write_text(
        json.dumps({
            "config": {"model": model},
            "stats": {
                "num_valid": 3,
                "num_tasks": 3,
                "num_samples": 3,
                "mean_episode_return": 0.5,
            },
        }),
        encoding="utf-8",
    )
    return summary


def test_update_run_json_uses_slug_suffix_before_run_id(tmp_path: Path) -> None:
    slug = tmp_path / "Qwen_Qwen3.5-4B__som"

    row = result_from_summary(slug, _summary(slug), run_id="run_0_mixed")

    assert row["config_id"] == "som"


def test_update_run_json_falls_back_to_mode_specific_run_id(tmp_path: Path) -> None:
    slug = tmp_path / "Qwen_Qwen3.5-4B"

    row = result_from_summary(slug, _summary(slug), run_id="run_0_som")

    assert row["config_id"] == "som"


def test_update_run_json_unrecognized_run_id_stays_default(tmp_path: Path) -> None:
    slug = tmp_path / "Qwen_Qwen3.5-4B"

    row = result_from_summary(slug, _summary(slug), run_id="run_custom")

    assert row["config_id"] == "default"
