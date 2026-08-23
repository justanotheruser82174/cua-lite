from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from lite.core import LiteCUAMetadata
from lite.core.tools.calls import tool_call_arguments, tool_call_id, tool_call_name
from lite.core.tools.schemas import tool_schema_name
from lite.data.preproc.ui_genie_agent import use as ui_genie_use
from lite.data.preproc.ui_genie_agent import utils as ui_genie_utils
from lite.data.staging import coerce_messages, coerce_meta, iter_parquet_rows, write_partition
from lite.data.utils.messages import extra_tool_schemas_for_messages
from lite.data.utils.rows import (
    iter_canonical_actions,
    validate_action_batches,
    validate_canonical_rows,
)

_IMAGE_PREFIX = "cua-lite/UI-Genie-Agent/images/"
_HASHED_IMAGE_RE = re.compile(r"^[0-9a-f]{2}/[0-9a-f]{64}\.png$")


def _write_png(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), color=color).save(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_ui_genie_grouping_fails_loud_on_malformed_source_record(tmp_path: Path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text('{"images": [}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"bad\.jsonl line 1"):
        ui_genie_use.group_records_by_uid(str(source), "ui_genie")


def _amex_record(step: int, args: dict[str, Any]) -> dict[str, Any]:
    progress = "\n".join(f"Step{i}: prior" for i in range(1, step + 1)) or "none"
    return {
        "images": [f"AMEX/screenshot/checkout-{step}.png"],
        "messages": [
            {"content": "system prompt; resolution is 100x100"},
            {"content": f"The user query: submit the checkout form\nTask progress: {progress}"},
            {"content": f"<tool_call>{json.dumps({'arguments': args})}</tool_call>"},
        ],
    }


def _iter_tool_calls(messages: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for message in messages:
        yield from message.get("tool_calls") or []


def _assert_canonical_row(
    row: dict[str, Any],
    *,
    key: tuple[str, str, str, str],
    out_dir: Path,
) -> None:
    assert set(row) == {"images", "messages", "metadata"}

    images = row["images"]
    messages = row["messages"]
    metadata = row["metadata"]
    assert isinstance(images, list) and images
    assert isinstance(messages, list) and messages
    assert isinstance(metadata, dict)

    lite_meta = LiteCUAMetadata.from_dict(metadata)
    assert key[0] == lite_meta.platform.value == "mobile"
    assert key[1] == lite_meta.task_type.value == "use"
    assert "platform" not in metadata
    assert "task_type" not in metadata
    assert key[2] in {"train", "validation"}
    assert key[3] == "amex"

    assert lite_meta.valid_actions is None
    assert "split" not in metadata
    assert "split" not in metadata.get("others", {})

    for image_rel in images:
        assert image_rel.startswith(_IMAGE_PREFIX)
        hashed_rel = image_rel.removeprefix(_IMAGE_PREFIX)
        assert _HASHED_IMAGE_RE.match(hashed_rel)
        assert (out_dir / "images" / hashed_rel).is_file()

    referenced: set[int] = set()
    for message in messages:
        for part in message.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "image":
                index = part.get("index")
                assert isinstance(index, int)
                assert 0 <= index < len(images)
                referenced.add(index)
    assert referenced == set(range(len(images)))

    for message in messages:
        if message.get("role") == "tool":
            assert isinstance(message.get("tool_call_id"), str)
        else:
            assert "tool_call_id" not in message

    calls = list(_iter_tool_calls(messages))
    assert calls
    assert [tool_call_id(call) for call in calls] == [
        f"call_{i:04d}" for i in range(len(calls))
    ]
    for call in calls:
        assert set(call) == {"id", "type", "function"}
        assert isinstance(tool_call_arguments(call), dict)

    assert [tool_call_name(call) for call in calls] == ["open_app", "mobile"]
    assert {tool_schema_name(schema) for schema in metadata["extra_tool_schemas"]} == {"open_app"}
    assert metadata["extra_tool_schemas"] == extra_tool_schemas_for_messages(messages)

    validate_action_batches(messages)
    mobile_actions = [
        (name, arguments)
        for message in messages
        for name, arguments in iter_canonical_actions(message)
        if name == "tap"
    ]
    assert len(mobile_actions) == 1
    assert mobile_actions[0][1]["coordinate"] == [250, 350]

    for message in messages:
        for _name, arguments in iter_canonical_actions(message):
            for field in ("coordinate", "start_coordinate"):
                coord = arguments.get(field)
                if coord is None:
                    continue
                assert isinstance(coord, list) and len(coord) == 2
                assert all(isinstance(v, int | float) and 0 <= v <= 1000 for v in coord)
            keys = arguments.get("keys")
            if keys is not None:
                assert isinstance(keys, list)
                assert all(isinstance(k, str) and not k.startswith("VK_") for k in keys)

    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    seen_call_ids: set[str] = set()
    paired_call_ids: set[str] = set()
    for message in messages:
        if message["role"] == "assistant":
            for call in message.get("tool_calls") or []:
                call_id = tool_call_id(call)
                assert call_id is not None
                seen_call_ids.add(call_id)
        if message["role"] == "tool":
            assert message["tool_call_id"] in seen_call_ids
            assert message["tool_call_id"] not in paired_call_ids
            paired_call_ids.add(message["tool_call_id"])
            assert message["content"] and message["content"][0]["type"] == "image"
    assert paired_call_ids == {"call_0000", "call_0001"}

    assert messages[-1] == {"role": "assistant", "content": [{"type": "text", "text": "done"}]}
    assert messages[-1]["content"][0]["text"].strip() not in {"", "<|im_end|>", "</s>"}

    validate_canonical_rows([row], "ui_genie_amex")


def test_ui_genie_amex_raw_entrypoint_to_canonical_roundtrip(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    out_root = tmp_path / "out"
    source_root = raw_root / "HanXiao1999" / "UI-Genie-Agent-16k"
    image_dir = source_root / "AMEX" / "screenshot"
    for step, color in enumerate(((210, 40, 40), (40, 180, 80), (50, 90, 210))):
        _write_png(image_dir / f"checkout-{step}.png", color=color)
    _write_jsonl(
        source_root / "AMEX_Agent_34K.jsonl",
        [
            _amex_record(0, {"action": "open", "text": "Chrome", "action_desc": "open browser"}),
            _amex_record(
                1,
                {"action": "click", "coordinate": [25, 35], "action_desc": "tap submit"},
            ),
            _amex_record(2, {"action": "terminate", "status": "success", "action_info": "done"}),
        ],
    )
    monkeypatch.setenv("CUA_LITE_RAW_DATASETS_ROOT", str(raw_root))
    monkeypatch.setenv("CUA_LITE_DATASETS_ROOT", str(out_root))

    rows, n_traj, n_steps, skips = ui_genie_use.process_subset(
        "amex",
        ui_genie_use.SUBSETS["amex"],
        str(raw_root),
        head=1,
        verbose=False,
    )
    assert len(rows) == 1
    # The ledger travels with the rows and closes in trajectories.
    assert n_traj == len(rows) + sum(skips.values())
    assert n_steps == 3
    assert dict(skips) == {}
    variant, entry = "amex", rows[0]

    out_dir = ui_genie_utils.out_dir_for(root=out_root)
    key, staged = ui_genie_utils.stage_entry(
        copy.deepcopy(entry),
        store=ui_genie_utils.make_image_store(out_dir),
        splitter=ui_genie_utils.make_splitter(),
        variant=variant,
    )
    _assert_canonical_row(staged, key=key, out_dir=out_dir)

    parquet_path = tmp_path / "roundtrip" / "ui_genie_amex.parquet"
    write_partition([staged], parquet_path)
    raw_rows = list(iter_parquet_rows(parquet_path))
    assert len(raw_rows) == 1
    raw_row = raw_rows[0]
    assert set(raw_row) == {"images", "messages", "metadata"}
    assert isinstance(raw_row["messages"], str)
    assert isinstance(raw_row["metadata"], str)

    roundtripped = {
        "images": raw_row["images"],
        "messages": coerce_messages(raw_row["messages"]),
        "metadata": coerce_meta(raw_row["metadata"]),
    }
    assert roundtripped == staged
    _assert_canonical_row(roundtripped, key=key, out_dir=out_dir)


# The set-of-mark system prompt, verbatim from ``ui_genie_agent_16k.jsonl``'s
# second modality (2,416 of its 16,698 step records). It declares no resolution.
_SOM_SYSTEM_PROMPT = (
    "You are a helpful assistant.\n\n# Tools\n\nThe interactive UI elements on "
    "the screenshot are labeled with numeric tags starting from 1."
)


def _ui_genie_som_record(step: int, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "images": [f"data/screenshots/u/screenshot-{step}.png"],
        "messages": [
            {"content": _SOM_SYSTEM_PROMPT},
            {"content": "The user query: play a song\nTask progress: none"},
            {"content": f"<tool_call>{json.dumps({'arguments': args})}</tool_call>"},
        ],
    }


def test_ui_genie_som_variant_is_skipped_as_itself_not_as_a_bad_resolution():
    """The 417 dropped ``ui_genie`` trajectories must name their real cause.

    The set-of-mark modality declares no resolution *by design*, so reading the
    resolution first reports it as malformed source data — which is what sent
    18.9% of the subset to a skip reason that misdescribed it.
    """
    try:
        ui_genie_use.build_trajectory(
            "u",
            [
                (0, _ui_genie_som_record(0, {"action": "click", "som": 21,
                                             "action_desc": "tap element 21"})),
                (1, _ui_genie_som_record(1, {"action": "terminate", "status": "success",
                                             "action_info": "done"})),
            ],
            ui_genie_use.SUBSETS["ui_genie"],
        )
    except ui_genie_use.SkipTrajectoryError as e:
        assert e.reason == "som_annotation_variant"
        assert "resolution" not in str(e)
    else:
        raise AssertionError("SOM-modality trajectory was not skipped")


def test_ui_genie_point_prompt_without_a_resolution_is_its_own_bucket():
    """The residual bucket is real but empty on today's snapshot (0 of 51,786 steps)."""
    rec = _ui_genie_som_record(0, {"action": "click", "coordinate": [1, 2]})
    rec["messages"][0]["content"] = "You are a helpful assistant.\n\n# Tools\n"
    try:
        ui_genie_use.build_trajectory("u", [(0, rec)], ui_genie_use.SUBSETS["ui_genie"])
    except ui_genie_use.SkipTrajectoryError as e:
        assert e.reason == "malformed_resolution"
    else:
        raise AssertionError("resolution-less point prompt was not skipped")
