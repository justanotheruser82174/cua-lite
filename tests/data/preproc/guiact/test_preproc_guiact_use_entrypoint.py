from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from lite.core import LiteCUAMetadata
from lite.core.tools.calls import tool_call_arguments, tool_call_id, tool_call_name
from lite.data.preproc.guiact import use as guiact_use
from lite.data.preproc.guiact import utils as guiact_utils
from lite.data.staging import coerce_messages, coerce_meta, iter_parquet_rows, write_partition
from lite.data.utils.rows import (
    iter_canonical_actions,
    validate_action_batches,
    validate_canonical_rows,
)

_IMAGE_PREFIX = "cua-lite/GUIAct/images/"
_HASHED_IMAGE_RE = re.compile(r"^[0-9a-f]{2}/[0-9a-f]{64}\.jpg$")


def _write_jpg(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), color=color).save(path, format="JPEG")


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def _assert_guiact_web_multi_use_row(
    row: dict[str, Any],
    *,
    key: tuple[str, str, str, str],
    out_dir: Path,
) -> None:
    assert set(row) == {"images", "messages", "metadata"}
    images = row["images"]
    messages = row["messages"]
    metadata = row["metadata"]

    assert key == ("browser", "use", "train", "use")
    lite_meta = LiteCUAMetadata.from_dict(metadata)
    assert lite_meta.platform.value == "browser"
    assert lite_meta.task_type.value == "use"
    assert lite_meta.valid_actions is None
    assert "platform" not in metadata
    assert "task_type" not in metadata
    assert metadata["extra_tool_schemas"] == []
    assert "split" not in metadata
    assert "split" not in metadata.get("others", {})

    assert len(images) == 2
    for image_rel in images:
        assert image_rel.startswith(_IMAGE_PREFIX)
        hashed_rel = image_rel.removeprefix(_IMAGE_PREFIX)
        assert _HASHED_IMAGE_RE.match(hashed_rel)
        assert (out_dir / "images" / hashed_rel).is_file()

    referenced = {
        part["index"]
        for message in messages
        for part in (message.get("content") or [])
        if isinstance(part, dict) and part.get("type") == "image"
    }
    assert referenced == {0, 1}

    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert messages[0]["content"] == [
        {"type": "image", "index": 0},
        {"type": "text", "text": "Click the checkout button."},
    ]
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "call_0000",
        "content": [{"type": "image", "index": 1}],
    }
    # The trajectory ends on an ANSWER, so the final turn carries the answer
    # text itself -- ``Done.`` is only the no-information end signal.
    assert messages[-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "the checkout button is at the top right"}],
    }

    for message in messages:
        if message.get("role") == "tool":
            assert isinstance(message.get("tool_call_id"), str)
        else:
            assert "tool_call_id" not in message
        for call in message.get("tool_calls") or []:
            assert set(call) == {"id", "type", "function"}
            assert tool_call_name(call) == "computer"
            assert isinstance(tool_call_arguments(call), dict)

    calls = [call for message in messages for call in (message.get("tool_calls") or [])]
    assert [tool_call_id(call) for call in calls] == ["call_0000"]
    validate_action_batches(messages)
    actions = [
        (name, arguments)
        for message in messages
        for name, arguments in iter_canonical_actions(message)
    ]
    assert actions == [("click", {"coordinate": [250, 350]})]

    assert metadata["others"] == {
        "id": "guiact_web_multi_train_42",
        "resolution": [100, 100],
        "os": None,
        "source": "yiye2023/GUIAct",
        "source_id": "guiact_web_multi_train_42",
    }
    validate_canonical_rows([row], "guiact_web_multi_use")


def test_guiact_use_raw_entrypoint_to_canonical_roundtrip(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    out_root = tmp_path / "out"
    data_path = raw_root / "yiye2023" / "GUIAct" / "web-multi_train_data.json"
    image_dir = raw_root / "yiye2023" / "GUIAct" / "images" / "web-multi" / "train"

    _write_jpg(image_dir / "web42_0.jpg", color=(210, 80, 80))
    _write_jpg(image_dir / "web42_1.jpg", color=(80, 160, 220))
    _write_json(
        data_path,
        [
            {
                "uid": "uid_record_42_step_0",
                "image_id": "web42_0",
                "image_size": {"width": 100, "height": 100},
                "question": "Click the checkout button.",
                "actions_label": [
                    {"name": "click", "point": {"related": "<point>0.25, 0.35</point>"}},
                ],
            },
            {
                "uid": "uid_record_42_step_1",
                "image_id": "web42_1",
                "image_size": {"width": 100, "height": 100},
                "question": "Click the checkout button.",
                "actions_label": [
                    {"name": "answer", "text": "the checkout button is at the top right"},
                ],
            },
        ],
    )
    monkeypatch.setenv("CUA_LITE_RAW_DATASETS_ROOT", str(raw_root))
    monkeypatch.setenv("CUA_LITE_DATASETS_ROOT", str(out_root))

    config = guiact_use.SUBSETS["web_multi"]
    with data_path.open(encoding="utf-8") as f:
        data = json.load(f)
    episodes = guiact_use._group_episodes(data, config["uid_re"])
    assert sorted(episodes) == ["42"]

    entry = guiact_use.episode_to_example("42", episodes["42"], config=config)
    entry["metadata"]["others"]["split"] = "train"

    out_dir = guiact_utils.out_dir_for(root=out_root)
    key, staged = guiact_utils.stage_entry(
        copy.deepcopy(entry),
        store=guiact_utils.make_image_store(out_dir),
        splitter=guiact_utils.make_splitter(),
        variant=guiact_use.VARIANT,
    )
    _assert_guiact_web_multi_use_row(staged, key=key, out_dir=out_dir)

    parquet_path = tmp_path / "roundtrip" / "guiact_web_multi_use.parquet"
    write_partition([staged], parquet_path)
    raw_rows = list(iter_parquet_rows(parquet_path))
    assert len(raw_rows) == 1
    raw_row = raw_rows[0]
    assert isinstance(raw_row["messages"], str)
    assert isinstance(raw_row["metadata"], str)
    roundtripped = {
        "images": raw_row["images"],
        "messages": coerce_messages(raw_row["messages"]),
        "metadata": coerce_meta(raw_row["metadata"]),
    }
    assert roundtripped == staged
    _assert_guiact_web_multi_use_row(roundtripped, key=key, out_dir=out_dir)


def test_guiact_reused_episode_id_starts_a_new_contiguous_run() -> None:
    data = [
        {"uid": "uid_record_42_step_0", "question": "first"},
        {"uid": "uid_record_42_step_1", "question": "first"},
        {"uid": "uid_record_7_step_0", "question": "middle"},
        {"uid": "uid_record_42_step_0", "question": "second"},
        {"uid": "uid_record_42_step_1", "question": "second"},
    ]

    episodes = guiact_use._group_episodes(
        data,
        guiact_use.SUBSETS["web_multi"]["uid_re"],
    )

    assert list(episodes) == ["42", "7", "42_run_2"]
    assert {row["question"] for row in episodes["42"]} == {"first"}
    assert {row["question"] for row in episodes["42_run_2"]} == {"second"}

    corrected = guiact_use._group_episodes(
        [
            {"uid": "uid_record_9_step_0", "question": "same", "answer": "wrong"},
            {"uid": "uid_record_7_step_0", "question": "middle"},
            {"uid": "uid_record_9_step_0", "question": "same", "answer": "correct"},
        ],
        guiact_use.SUBSETS["web_multi"]["uid_re"],
    )
    assert list(corrected) == ["9", "7"]
    assert corrected["9"][0]["answer"] == "correct"


def test_guiact_smartphone_use_raw_entrypoint_to_canonical_roundtrip(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    out_root = tmp_path / "out"
    data_path = raw_root / "yiye2023" / "GUIAct" / "smartphone_train_data.json"
    image_dir = raw_root / "yiye2023" / "GUIAct" / "images" / "smartphone"

    _write_jpg(image_dir / "phone7_0.jpg", color=(210, 80, 80))
    _write_jpg(image_dir / "phone7_1.jpg", color=(80, 160, 220))
    _write_json(
        data_path,
        [
            {
                "uid": "uid_episode_7_step_0",
                "image_id": "phone7_0",
                "image_size": {"width": 100, "height": 100},
                "question": "Tap the checkout button.",
                "actions_label": {"name": "tap", "point": {"related": "<point>0.25, 0.35</point>"}},
            },
            {
                "uid": "uid_episode_7_step_1",
                "image_id": "phone7_1",
                "image_size": {"width": 100, "height": 100},
                "question": "Tap the checkout button.",
                "actions_label": {"name": "answer", "text": "the cart holds 3 items"},
            },
        ],
    )
    monkeypatch.setenv("CUA_LITE_RAW_DATASETS_ROOT", str(raw_root))
    monkeypatch.setenv("CUA_LITE_DATASETS_ROOT", str(out_root))

    config = guiact_use.SUBSETS["smartphone"]
    with data_path.open(encoding="utf-8") as f:
        data = json.load(f)
    episodes = guiact_use._group_episodes(data, config["uid_re"])
    assert sorted(episodes) == ["7"]

    entry = guiact_use.episode_to_example("7", episodes["7"], config=config)
    entry["metadata"]["others"]["split"] = "train"

    out_dir = guiact_utils.out_dir_for(root=out_root)
    key, staged = guiact_utils.stage_entry(
        copy.deepcopy(entry),
        store=guiact_utils.make_image_store(out_dir),
        splitter=guiact_utils.make_splitter(),
        variant=guiact_use.VARIANT,
    )

    assert key == ("mobile", "use", "train", "use")
    assert staged["metadata"]["others"] == {
        "id": "guiact_smartphone_7",
        "resolution": [100, 100],
        "os": "android",
        "source": "yiye2023/GUIAct",
        "source_id": "guiact_smartphone_7",
    }
    assert len(staged["images"]) == 2
    for image_rel in staged["images"]:
        assert image_rel.startswith(_IMAGE_PREFIX)
        hashed_rel = image_rel.removeprefix(_IMAGE_PREFIX)
        assert _HASHED_IMAGE_RE.match(hashed_rel)
        assert (out_dir / "images" / hashed_rel).is_file()

    messages = staged["messages"]
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    call = messages[1]["tool_calls"][0]
    assert tool_call_name(call) == "mobile"
    assert tool_call_arguments(call)["actions"] == [
        {"action": "tap", "coordinate": [250, 350], "clicks": 1}
    ]
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": tool_call_id(call),
        "content": [{"type": "image", "index": 1}],
    }
    assert messages[-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "the cart holds 3 items"}],
    }
    validate_action_batches(messages)
    validate_canonical_rows([staged], "guiact_smartphone_use")

    parquet_path = tmp_path / "roundtrip" / "guiact_smartphone_use.parquet"
    write_partition([staged], parquet_path)
    raw_row = next(iter_parquet_rows(parquet_path))
    roundtripped = {
        "images": raw_row["images"],
        "messages": coerce_messages(raw_row["messages"]),
        "metadata": coerce_meta(raw_row["metadata"]),
    }
    assert roundtripped == staged
    validate_canonical_rows([roundtripped], "guiact_smartphone_use")
