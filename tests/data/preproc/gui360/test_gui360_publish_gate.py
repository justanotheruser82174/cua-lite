"""GUI360 producer publish-gate regressions."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from lite.core.tools.calls import tool_call_arguments, tool_call_id, tool_call_name
from lite.data.utils.rows import validate_canonical_rows

_ROOT = Path(__file__).resolve().parents[4]
_PREPROC = _ROOT / "lite" / "data" / "preproc"


def _load_preproc_script(path: Path):
    """Import a hyphenated preproc script (not importable as a module name)."""
    name = f"cua_lite_pg_{path.parent.name}_{path.stem.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gui360_grounding_point_row_passes_publish_gate(tmp_path, monkeypatch) -> None:
    """Drive gui360 ``grounding-point`` end-to-end over a synthetic subset."""
    pytest.importorskip("PIL")
    from PIL import Image

    module = _load_preproc_script(_PREPROC / "gui360" / "grounding-point.py")

    subset = tmp_path / module.SUBSET_REL
    (subset / "images").mkdir(parents=True)
    Image.new("RGB", (800, 600), "white").save(subset / "images" / "step1.png")
    prompt = (
        "<image>\nThe instruction is:\nClick the Save button\n\n"
        "Output the coordinate as <coordinate> [x, y] </coordinate>"
    )
    (subset / "training_data.json").write_text(
        json.dumps(
            [
                {
                    "id": "excel_1",
                    "images": ["images\\step1.png"],
                    "conversation": [
                        {"from": "human", "value": prompt},
                        {"from": "gpt", "value": "<coordinate> [400, 300] </coordinate>"},
                    ],
                }
            ]
        )
    )
    monkeypatch.setenv("CUA_LITE_RAW_DATASETS_ROOT", str(tmp_path))

    rows = list(module.process_records(
        str(subset / "training_data.json"), head=None, verbose=False
    ))
    assert len(rows) == 1
    call = rows[0]["messages"][1]["tool_calls"][0]
    assert tool_call_name(call) == "point"
    assert tool_call_id(call), "grounding.point row must carry a canonical id"
    assert tool_call_arguments(call)["coordinate"] == [500, 500]
    validate_canonical_rows(rows, "gui360/grounding.point")


def test_gui360_grounding_missing_images_field_fails_loudly(tmp_path) -> None:
    module = _load_preproc_script(_PREPROC / "gui360" / "grounding-point.py")
    source = tmp_path / "training_data.json"
    source.write_text(json.dumps([{"id": "broken"}]))

    with pytest.raises(ValueError, match="No image field"):
        list(module.process_records(str(source), head=None, verbose=False))
