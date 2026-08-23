from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from lite.core import LiteCUAMetadata
from lite.core.tools.calls import tool_call_arguments, tool_call_id, tool_call_name
from lite.data.preproc.gui360 import use as gui360_use
from lite.data.preproc.gui360 import utils as gui360_utils
from lite.data.staging import coerce_messages, coerce_meta, iter_parquet_rows, write_partition
from lite.data.utils.rows import (
    iter_canonical_actions,
    validate_action_batches,
    validate_canonical_rows,
)

_IMAGE_PREFIX = "cua-lite/GUI-360/images/"
_HASHED_IMAGE_RE = re.compile(r"^[0-9a-f]{2}/[0-9a-f]{64}\.png$")


def _write_png(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), color=color).save(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _step(
    *,
    execution_id: str,
    screenshot: str,
    status: str,
    action: dict[str, Any],
    subtask: str,
) -> dict[str, Any]:
    return {
        "execution_id": execution_id,
        "request": "Copy the title text in Word.",
        "evaluation": {"complete": "yes"},
        "step": {
            "status": status,
            "thought": f"Need to {subtask}.",
            "subtask": subtask,
            "screenshot_clean": screenshot,
            "action": action,
        },
    }


def _tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [call for message in messages for call in (message.get("tool_calls") or [])]


def test_gui360_use_reports_missing_terminal_reason(tmp_path) -> None:
    path = tmp_path / "unfinished.jsonl"
    _write_jsonl(
        path,
        [_step(
            execution_id="unfinished",
            screenshot="shot.png",
            status="RUNNING",
            subtask="keep working",
            action={"function": "click", "coordinate_x": 1, "coordinate_y": 1},
        )],
    )

    with pytest.raises(gui360_use.SkipTrajectory, match="missing final OVERALL_FINISH"):
        gui360_use.build_trajectory(str(path), "word", "forms")


@pytest.mark.parametrize("wheel_dist", [None, True, "3", 3.5])
def test_gui360_rejects_malformed_wheel_distance(wheel_dist) -> None:
    action = {"function": "wheel_mouse_input", "args": {}}
    if wheel_dist is not None:
        action["args"]["wheel_dist"] = wheel_dist

    with pytest.raises(gui360_use.SkipTrajectory, match="integer wheel_dist"):
        gui360_use.map_action(action, 100, 100, "bad-wheel")


def _assert_gui360_use_row(
    row: dict[str, Any],
    *,
    key: tuple[str, str, str, str],
    out_dir: Path,
) -> None:
    assert set(row) == {"images", "messages", "metadata"}
    images = row["images"]
    messages = row["messages"]
    metadata = row["metadata"]

    lite_meta = LiteCUAMetadata.from_dict(metadata)
    assert key[0] == lite_meta.platform.value == "desktop"
    assert key[1] == lite_meta.task_type.value == "use"
    assert "platform" not in metadata
    assert "task_type" not in metadata
    assert key[2] in {"train", "validation"}
    assert key[3] == "use"
    assert lite_meta.valid_actions is None
    assert metadata["extra_tool_schemas"] == []
    assert "split" not in metadata
    assert "split" not in metadata.get("others", {})

    assert len(images) == 3
    for rel_image in images:
        assert rel_image.startswith(_IMAGE_PREFIX)
        hashed_rel = rel_image.removeprefix(_IMAGE_PREFIX)
        assert _HASHED_IMAGE_RE.match(hashed_rel)
        assert (out_dir / "images" / hashed_rel).is_file()

    referenced = {
        part["index"]
        for message in messages
        for part in (message.get("content") or [])
        if isinstance(part, dict) and part.get("type") == "image"
    }
    assert referenced == {0, 1, 2}

    for message in messages:
        if message.get("role") == "tool":
            assert isinstance(message.get("tool_call_id"), str)
        else:
            assert "tool_call_id" not in message
        for call in message.get("tool_calls") or []:
            assert set(call) == {"id", "type", "function"}
            assert tool_call_name(call) == "computer"
            assert isinstance(tool_call_arguments(call), dict)

    calls = _tool_calls(messages)
    assert [tool_call_id(call) for call in calls] == ["call_0000", "call_0001"]
    validate_action_batches(messages)

    actions = [
        (name, arguments)
        for message in messages
        for name, arguments in iter_canonical_actions(message)
    ]
    assert actions[0] == ("click", {"coordinate": [500, 400]})
    assert actions[1] == ("key", {"keys": ["ctrl", "c"]})
    assert all("VK_" not in key for _name, args in actions for key in args.get("keys", []))

    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert messages[2]["tool_call_id"] == "call_0000"
    assert messages[2]["content"] == [{"type": "image", "index": 1}]
    assert messages[4]["tool_call_id"] == "call_0001"
    assert messages[4]["content"] == [{"type": "image", "index": 2}]
    assert messages[-1] == {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}
    assert messages[-1]["content"][0]["text"].strip() not in {"", "<|im_end|>", "</s>"}

    validate_canonical_rows([row], "gui360_use")


def test_gui360_use_raw_entrypoint_to_canonical_roundtrip(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    out_root = tmp_path / "out"
    app = "word"
    category = "forms"
    execution_id = "gui360_exec_001"

    image_dir = raw_root / gui360_use.TRAIN_REL / "image" / app / category
    for idx, color in enumerate(((220, 220, 40), (40, 140, 220), (180, 60, 180))):
        _write_png(image_dir / f"shot{idx}.png", color=color)

    jsonl_path = (
        raw_root
        / gui360_use.TRAIN_REL
        / "data"
        / app
        / category
        / "success"
        / f"{execution_id}.jsonl"
    )
    _write_jsonl(
        jsonl_path,
        [
            _step(
                execution_id=execution_id,
                screenshot="shot0.png",
                status="RUNNING",
                subtask="select the title",
                action={
                    "function": "click",
                    "coordinate_x": 50,
                    "coordinate_y": 40,
                    "args": {"button": "left"},
                },
            ),
            _step(
                execution_id=execution_id,
                screenshot="shot1.png",
                status="RUNNING",
                subtask="copy the title",
                action={"function": "type", "args": {"keys": "{VK_CONTROL}c"}},
            ),
            _step(
                execution_id=execution_id,
                screenshot="shot2.png",
                status="OVERALL_FINISH",
                subtask="finish",
                action={"function": "", "args": {}},
            ),
        ],
    )
    monkeypatch.setenv("CUA_LITE_RAW_DATASETS_ROOT", str(raw_root))
    monkeypatch.setenv("CUA_LITE_DATASETS_ROOT", str(out_root))

    files = list(gui360_use.iter_trajectory_files(str(raw_root)))
    assert files == [(str(jsonl_path), app, category)]
    entry = gui360_use.build_trajectory(*files[0])
    assert entry is not None

    out_dir = gui360_utils.out_dir_for(root=out_root)
    key, staged = gui360_utils.stage_entry(
        copy.deepcopy(entry),
        store=gui360_utils.make_image_store(out_dir),
        splitter=gui360_utils.make_splitter(),
        variant=gui360_use.VARIANT,
    )
    _assert_gui360_use_row(staged, key=key, out_dir=out_dir)

    parquet_path = tmp_path / "roundtrip" / "gui360_use.parquet"
    write_partition([staged], parquet_path)
    raw_row = next(iter_parquet_rows(parquet_path))
    assert isinstance(raw_row["messages"], str)
    assert isinstance(raw_row["metadata"], str)
    roundtripped = {
        "images": raw_row["images"],
        "messages": coerce_messages(raw_row["messages"]),
        "metadata": coerce_meta(raw_row["metadata"]),
    }
    assert roundtripped == staged
    _assert_gui360_use_row(roundtripped, key=key, out_dir=out_dir)
