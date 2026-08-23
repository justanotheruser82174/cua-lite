"""Cohort-level regressions for the ScaleCUA preprocessors.

Complements ``tests/data/preproc/common/test_common_publish_gate.py`` (which owns
the tree-wide inventory guard) with the behaviour that is
specific to these two dataset dirs and was found by running them:

* the three ScaleCUA ``grounding-*`` scripts built their tool call from the
  action space alone, so every emitted row failed the publish gate with
  ``messages[1].tool_calls[0] missing non-empty call_id``;
* ``scalecua/understanding.py`` returned a bare list from its missing-annotation
  early exit while ``main()`` unpacks a ``(results, n_empty)`` pair;
* ``scalecua/use.py`` had no bound at all, so a from-scratch run processed tens
  of GB and had never been smoke-tested. ``--head`` must truncate whole
  records/trajectories, never a partial row.

Run:
    uv run --extra data --extra dev pytest -q \
        tests/data/preproc/scalecua/test_preproc_scalecua_cohorts.py
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lite.core.tools.calls import tool_call_arguments, tool_call_id, tool_call_name
from lite.data.staging import iter_parquet_rows, write_partition
from lite.data.utils.rows import validate_canonical_rows

_ROOT = Path(__file__).resolve().parents[4]
_PREPROC = _ROOT / "lite" / "data" / "preproc"


def test_scalecua_android_20250407_understanding_roots_match_archive_layout() -> None:
    meta = json.loads((_PREPROC / "scalecua" / "meta.json").read_text())
    names = (
        "android_20250407_dense_caption_20250712",
        "android_20250407_screen_transition_20250712",
        "android_20250407_user_intention_20250703",
    )
    assert {meta[name]["root"] for name in names} == {"data/data_20250407/android"}


def test_scalecua_windows_20250616_paste_roots_match_archive_layout() -> None:
    meta = json.loads((_PREPROC / "scalecua" / "meta.json").read_text())
    names = tuple(f"windows_aug_action_grounding_20250616_{i}" for i in range(1, 9))
    assert {meta[name]["root"] for name in names} == {
        "data/data_20250616/windows_pure_paste/images"
    }


def _load_preproc_script(path: Path):
    """Import a hyphenated preproc script (not importable as a module name)."""
    name = f"cua_lite_cohort_{path.parent.name}_{path.stem.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Source records, shaped like the real ScaleCUA annotation lines.
# ---------------------------------------------------------------------------

_POINT_RECORD = {
    "image": "img/step_1.png",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nLocate the point contained in: <ref>Quick & Easy</ref>",
        },
        {"from": "gpt", "value": "<ref>Quick & Easy</ref><point>[[125, 261]]</point>"},
    ],
    "width": 1000,
    "height": 1000,
}
_BBOX_RECORD = {
    "image": "img/step_1.png",
    "conversations": [
        {
            "from": "human",
            "value": (
                "<image>\nLocate the object referred to by <ref>Quick & Easy</ref> "
                "and output its bbox."
            ),
        },
        {"from": "gpt", "value": "<ref>Quick & Easy</ref><box>[[125, 261, 199, 312]]</box>"},
    ],
    "width": 1000,
    "height": 1000,
}
_ACTION_RECORD = {
    "image": "img/step_1.png",
    "conversations": [
        {"from": "human", "value": "<image>\nClick on 'Quick & Easy' in the dropdown menu."},
        {"from": "gpt", "value": "<action>\nclick(x=0.1624, y=0.2869)\n</action>"},
    ],
    "width": 1000,
    "height": 1000,
}
_UNDERSTANDING_RECORD = {
    "image": "img/step_1.png",
    "conversations": [
        {
            "from": "human",
            "value": (
                "<image>\nEvaluate the image and the action "
                "'click(x=0.07, y=0.94)'."
            ),
        },
        {"from": "gpt", "value": "The user likely intended to add content to the application."},
    ],
    "width": 1000,
    "height": 1000,
}


def _scalecua_source(tmp_path: Path, record: dict, *, copies: int = 1) -> tuple[Path, Path]:
    """Lay out a minimal ``OpenGVLab/ScaleCUA-Data`` tree; return (raw_root, subset)."""
    raw_root = tmp_path / "raw"
    subset = raw_root / "OpenGVLab" / "ScaleCUA-Data"
    (subset / "annotations").mkdir(parents=True)
    (subset / "data" / "img").mkdir(parents=True)
    # These scripts read width/height from the record and never open the image,
    # so existence is all that resolve_path needs.
    (subset / "data" / "img" / "step_1.png").write_bytes(b"")
    (subset / "annotations" / "a.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for _ in range(copies))
    )
    return raw_root, subset


def _run_point(module, **kw) -> list[dict[str, Any]]:
    return module.process_point_grounding_entry(**kw)[0]


def _run_bbox(module, **kw) -> list[dict[str, Any]]:
    return module.process_bbox_grounding_entry(**kw)[0]


def _run_action(module, **kw) -> list[dict[str, Any]]:
    return module.process_action_grounding_entry(**kw)[0]


def _run_understanding(module, **kw) -> list[dict[str, Any]]:
    return module.process_understanding_entry(task_type="user_intention", **kw)[0]


def _assert_persisted_rows_pass_publish_gate(
    rows: list[dict[str, Any]],
    tmp_path: Path,
    label: str,
) -> list[dict[str, Any]]:
    parquet_path = tmp_path / "persisted" / f"{label.replace('/', '_')}.parquet"
    write_partition(rows, parquet_path)
    persisted = list(iter_parquet_rows(parquet_path))
    assert len(persisted) == len(rows)
    for row in persisted:
        assert isinstance(row["messages"], str)
        assert isinstance(row["metadata"], str)
        metadata = json.loads(row["metadata"])
        assert "extra_tool_schemas" in metadata
        assert "valid_actions" in metadata
    validate_canonical_rows(persisted, label)
    return [
        {
            **row,
            "messages": json.loads(row["messages"]),
            "metadata": json.loads(row["metadata"]),
        }
        for row in persisted
    ]


def _assert_grounding_label_has_no_tool_result(row: dict[str, Any]) -> None:
    """Grounding rows are supervised labels, not executable action/result turns."""
    assert row["messages"][-1]["role"] == "assistant"
    assert row["messages"][-1]["tool_calls"]
    assert all(message.get("role") != "tool" for message in row["messages"])


@pytest.mark.parametrize(
    "action",
    [
        "scroll(x=0.5, y=0.5, direction=down)",
        "key(keys=['enter'])",
    ],
)
def test_scalecua_mobile_grounding_rejects_desktop_only_actions(action: str) -> None:
    module = _load_preproc_script(_PREPROC / "scalecua" / "grounding-action.py")

    with pytest.raises(ValueError, match="Mobile .* grounding"):
        module.parse_action(
            f"<action>{action}</action>", "mobile", "android_unit", 1
        )


def test_scalecua_desktop_scroll_grounding_supplies_required_amount() -> None:
    module = _load_preproc_script(_PREPROC / "scalecua" / "grounding-action.py")
    call = module.parse_action(
        "<action>scroll(x=0.5, y=0.5, direction=down)</action>",
        "desktop",
        "desktop_unit",
        1,
    )[0]

    assert tool_call_arguments(call)["actions"] == [{
        "action": "scroll",
        "coordinate": [500, 500],
        "direction": "down",
        "amount": 3,
    }]


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("key(keys=['ctrl', '+'])", ["ctrl", "+"]),
        ("key(keys=['plus', 'minus', 'equal'])", ["+", "-", "="]),
        ("key(keys=[','])", [","]),
    ],
)
def test_scalecua_desktop_grounding_key_actions_emit_canonical_glyphs(
    action: str, expected: list[str]
) -> None:
    module = _load_preproc_script(_PREPROC / "scalecua" / "grounding-action.py")
    call = module.parse_action(
        f"<action>{action}</action>",
        "desktop",
        "desktop_unit",
        1,
    )[0]

    assert tool_call_arguments(call)["actions"] == [
        {"action": "key", "keys": expected}
    ]


@pytest.mark.parametrize(
    "action",
    [
        "key(keys=['Insert a blank line'])",
        "key(keys=['ctrl+s'])",
        "key(keys=[])",
        "key(keys=[',', 1])",
        "key(keys=[' '])",
    ],
)
def test_scalecua_desktop_grounding_rejects_bad_key_tokens(action: str) -> None:
    module = _load_preproc_script(_PREPROC / "scalecua" / "grounding-action.py")

    with pytest.raises(ValueError, match="key"):
        module.parse_action(f"<action>{action}</action>", "desktop", "desktop_unit", 1)


# (script relpath, meta key, source record, runner, expected top-level call names)
_CASES: list[tuple[str, str, dict, Callable, list[str]]] = [
    ("grounding-point.py", "web_internvl_grounding", _POINT_RECORD, _run_point, ["point"]),
    ("grounding-bbox.py", "web_internvl_grounding", _BBOX_RECORD, _run_bbox, ["bbox"]),
    ("grounding-action.py", "web_action_grounding", _ACTION_RECORD, _run_action, ["computer"]),
    ("understanding.py", "web_user_intention", _UNDERSTANDING_RECORD, _run_understanding, []),
]


@pytest.mark.parametrize(
    "script,key,record,runner,expected_calls",
    _CASES,
    ids=[c[0] for c in _CASES],
)
def test_scalecua_entry_processor_rows_pass_publish_gate(
    tmp_path: Path,
    monkeypatch,
    script: str,
    key: str,
    record: dict,
    runner: Callable,
    expected_calls: list[str],
) -> None:
    """Every ScaleCUA cohort emits canonical rows from a real-shaped source line.

    The ``grounding-*`` cases are the direct regression: they used to yield rows
    whose only tool call had keys ``['arguments', 'name']``.
    """
    raw_root, subset = _scalecua_source(tmp_path, record)
    monkeypatch.setenv("CUA_LITE_RAW_DATASETS_ROOT", str(raw_root))

    module = _load_preproc_script(_PREPROC / "scalecua" / script)
    rows = runner(
        module,
        key=key,
        value={"annotation": "annotations/a.jsonl", "root": "data"},
        base_dir=str(subset),
        base_datasets_dir=str(raw_root),
        platform_type="browser",
    )

    assert len(rows) == 1
    [row] = _assert_persisted_rows_pass_publish_gate(
        rows,
        tmp_path,
        f"scalecua/{script}",
    )

    calls = [c for m in row["messages"] for c in (m.get("tool_calls") or [])]
    assert [tool_call_name(c) for c in calls] == expected_calls
    assert all(set(c) == {"id", "type", "function"} for c in calls)
    assert len({tool_call_id(c) for c in calls}) == len(calls)
    if script.startswith("grounding"):
        _assert_grounding_label_has_no_tool_result(row)


@pytest.mark.parametrize(
    "script,record,runner",
    [(c[0], c[2], c[3]) for c in _CASES],
    ids=[c[0] for c in _CASES],
)
def test_scalecua_head_truncates_whole_records(
    tmp_path: Path, monkeypatch, script: str, record: dict, runner: Callable
) -> None:
    """``--head`` bounds the record loop, so kept rows equal the unbounded prefix."""
    raw_root, subset = _scalecua_source(tmp_path, record, copies=5)
    monkeypatch.setenv("CUA_LITE_RAW_DATASETS_ROOT", str(raw_root))

    module = _load_preproc_script(_PREPROC / "scalecua" / script)
    kwargs = dict(
        key="web_internvl_grounding" if "grounding" in script else "web_user_intention",
        value={"annotation": "annotations/a.jsonl", "root": "data"},
        base_dir=str(subset),
        base_datasets_dir=str(raw_root),
        platform_type="browser",
    )

    bounded = runner(module, head=2, **kwargs)
    full = runner(module, **kwargs)

    assert len(full) == 5
    assert len(bounded) == 2
    assert [r["messages"] for r in bounded] == [r["messages"] for r in full[:2]]


def test_scalecua_understanding_missing_annotation_returns_a_pair(tmp_path: Path) -> None:
    """A missing annotation must not break the caller's tuple unpack.

    ``main()`` does ``results, n_records, skips = process_understanding_entry(...)``
    — the accounting travels with the rows so a run cannot report the rows
    without it. The early return handed back a bare list, so the first absent
    annotation file would raise ``not enough values to unpack``.
    """
    module = _load_preproc_script(_PREPROC / "scalecua" / "understanding.py")

    results, n_records, skips = module.process_understanding_entry(
        key="web_user_intention",
        value={"annotation": "annotations/missing.jsonl", "root": "data"},
        base_dir=str(tmp_path),
        base_datasets_dir=str(tmp_path.parent),
        platform_type="browser",
        task_type="user_intention",
    )

    assert results == []
    # Zero records were read, so there is no record-level loss to attribute:
    # main() reports this as an entry that read nothing.
    assert n_records == 0
    assert skips == {}


def test_scalecua_understanding_fails_loud_on_malformed_jsonl(tmp_path: Path) -> None:
    module = _load_preproc_script(_PREPROC / "scalecua" / "understanding.py")
    annotation = tmp_path / "annotations" / "broken.jsonl"
    annotation.parent.mkdir()
    annotation.write_text("{\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"broken\.jsonl line 1"):
        module.process_understanding_entry(
            key="web_user_intention",
            value={"annotation": "annotations/broken.jsonl", "root": "data"},
            base_dir=str(tmp_path),
            base_datasets_dir=str(tmp_path.parent),
            platform_type="browser",
            task_type="user_intention",
        )


def test_scalecua_understanding_fails_loud_without_an_image_field(tmp_path: Path) -> None:
    module = _load_preproc_script(_PREPROC / "scalecua" / "understanding.py")
    annotation = tmp_path / "annotations" / "missing-image.jsonl"
    annotation.parent.mkdir()
    annotation.write_text('{"conversations": []}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"missing-image\.jsonl line 1"):
        module.process_understanding_entry(
            key="web_user_intention",
            value={"annotation": "annotations/missing-image.jsonl", "root": "data"},
            base_dir=str(tmp_path),
            base_datasets_dir=str(tmp_path.parent),
            platform_type="browser",
            task_type="user_intention",
        )


def test_scalecua_demux_drops_gaps_and_keeps_contiguous_eof_partials(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A missing decision drops its lane; an open contiguous lane still publishes."""
    module = _load_preproc_script(_PREPROC / "scalecua" / "use.py")
    annotation = tmp_path / "a.jsonl"

    def record(task: str, source: str, previous: str, action: str) -> dict[str, Any]:
        return {
            "image": f"{source}/images/unused.png",
            "conversations": [
                {"from": "human", "value": (
                    f"<image>\nTask: {task}\n\nPrevious operations:\n{previous}"
                )},
                {"from": "gpt", "value": (
                    f"<operation>act</operation><action>{action}</action>"
                )},
            ],
        }

    rows = [
        record("Task A", "source-a", "None", "click(x=0.1, y=0.2)"),
        record("Task B", "source-b", "None", "click(x=0.3, y=0.4)"),
        record(
            "Task A",
            "source-a",
            'Step 1: click the button next to "Step 1 of 2"',
            "click(x=0.5, y=0.6)",
        ),
        record(
            "Task A",
            "source-a",
            'Step 1: click the button next to "Step 1 of 2"',
            "click(x=0.5, y=0.6)",
        ),
        record(
            "Task A",
            "source-a",
            "Step 1: click\nStep 2: focus\nStep 3: type",
            "terminate(status='success')",
        ),
        record(
            "Task A",
            "source-a",
            "Step 1: click\nStep 2: focus\nStep 3: type\nStep 4: done",
            "click(x=0.7, y=0.8)",
        ),
        record(
            "Task A",
            "source-c",
            "Step 1: click",
            "terminate(status='success')",
        ),
    ]
    annotation.write_text("".join(json.dumps(row) + "\n" for row in rows))

    captured: list[list[str]] = []

    def capture(traj_records, *_args, **_kwargs):
        captured.append([
            module.extract_task_from_prompt(record["conversations"][0]["value"])
            for record in traj_records
        ])
        return {"messages": []}

    monkeypatch.setattr(module, "merge_trajectory_steps", capture)
    output, n_records, skips = module.process_trajectory_entry(
        key="web_navigation",
        value={"annotation": annotation.name, "root": "."},
        base_dir=str(tmp_path),
        base_datasets_dir=str(tmp_path),
        platform_type="browser",
    )

    assert len(output) == 1
    assert captured == [["Task B"]]
    assert n_records == 7
    assert skips == {
        "duplicate_step": 1,
        "post_terminal_step": 1,
        "unmatched_step": 1,
        "history_gap": 3,
    }


# ---------------------------------------------------------------------------
# The smoke bounds themselves: without them these scripts are untestable.
#
# That every script OFFERS ``--head`` is a tree-wide inventory fact, so it lives
# next to the other one, in test_common_publish_gate.py
# (``test_script_accepts_a_head_smoke_bound``): glob-discovered over all 19
# scripts, and derived by parsing ``--head 3`` with each script's real argparse
# parser. It used to live here as ``assert '"--head"' in source`` over a
# hand-typed list of 8 -- a source-text pin that passed when the flag was
# deleted but the literal survived in a comment. What stays here is the
# cohort-specific SEMANTICS of the bound.
# ---------------------------------------------------------------------------
