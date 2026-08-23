from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from lite.core import LiteCUAMetadata
from lite.core.tools.calls import tool_call_arguments, tool_call_id, tool_call_name
from lite.data.preproc.opencua import use as opencua_use
from lite.data.preproc.opencua import utils as opencua_utils
from lite.data.staging import coerce_messages, coerce_meta, iter_parquet_rows, write_partition
from lite.data.utils.rows import (
    iter_canonical_actions,
    validate_action_batches,
    validate_canonical_rows,
)

_IMAGE_PREFIX = "cua-lite/OpenCUA/images/"
_HASHED_IMAGE_RE = re.compile(r"^[0-9a-f]{2}/[0-9a-f]{64}\.png$")


def _write_png(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), color=color).save(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _step(image: str, code: str) -> dict[str, Any]:
    return {"image": image, "value": {"thought": "t", "action": "a", "code": code}}


def _write_os_metadata(raw_root: Path, system_by_task_id: dict[str, str]) -> None:
    """AgentNet's ``meta_data_merged.jsonl`` sidecar — where ``others.os`` comes
    from. Written per-fixture so the entrypoint tests exercise the real loader
    rather than a stubbed table."""
    _write_jsonl(
        raw_root / opencua_use.OS_METADATA_JSONL,
        [
            {"task_id": task_id, "system": system}
            for task_id, system in system_by_task_id.items()
        ],
    )


def _assert_opencua_win_mac_row(
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
    assert key[3] == "win_mac"
    assert lite_meta.valid_actions is None
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
    assert set(messages[1]) == {"role", "content", "tool_calls"}
    assert messages[1]["content"] == [
        {"type": "inline_reasoning", "text": "Click the title field."},
        {"type": "action_description", "text": "click title"},
        {
            "type": "metadata",
            "data": {
                "opencua_step": {
                    "observation": "A TextEdit window is visible.",
                    "reflection": "The prior step opened the editor.",
                    "last_step_correct": True,
                    "last_step_redundant": False,
                }
            },
        },
    ]
    assert all(
        part["type"] in {
            "text",
            "image",
            "metadata",
            "action_description",
            "inline_reasoning",
            "history_summary",
        }
        for message in messages
        for part in (message.get("content") or [])
    )
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "call_0000",
        "content": [{"type": "image", "index": 1}],
    }
    assert messages[-1] == {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}

    others = metadata["others"]
    assert others["id"] == "agentnet_win_mac_mac_task_001"
    assert others["source"] == "xlangai/AgentNet"
    assert others["source_id"] == "mac_task_001"
    assert others["resolution"] == [100, 100]
    assert others["os"] == "macos"
    assert others["domain"] == "desktop"
    assert others["task_completed"] is True
    assert "terminate_status" not in others
    validate_canonical_rows([row], "opencua_win_mac")


def test_opencua_win_mac_raw_entrypoint_to_canonical_roundtrip(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    out_root = tmp_path / "out"
    config = opencua_use.DATASETS["win_mac"]
    jsonl_path = raw_root / config["jsonl"]
    images_dir = raw_root / config["images"]

    _write_png(images_dir / "macos_step0.png", color=(180, 80, 80))
    _write_png(images_dir / "macos_step1.png", color=(80, 180, 120))
    _write_jsonl(
        jsonl_path,
        [
            {
                "task_id": "mac_task_001",
                "instruction": "Use TextEdit on macOS to confirm the title.",
                "domain": "desktop",
                "task_completed": True,
                "alignment_score": 1.0,
                "efficiency_score": 0.9,
                "task_difficulty": "easy",
                "traj": [
                    {
                        "image": "macos_step0.png",
                        "value": {
                            "thought": "Click the title field.",
                            "action": "click title",
                            "code": "pyautogui.click(x=0.25, y=0.35)",
                            "observation": "A TextEdit window is visible.",
                            "reflection": "The prior step opened the editor.",
                            "last_step_correct": True,
                            "last_step_redundant": False,
                        },
                    },
                    {
                        "image": "macos_step1.png",
                        "value": {
                            "thought": "The title is confirmed.",
                            "action": "finish",
                            "code": "computer.terminate(status='success')",
                        },
                    },
                ],
            }
        ],
    )
    _write_os_metadata(raw_root, {"mac_task_001": "Darwin"})
    monkeypatch.setenv("CUA_LITE_RAW_DATASETS_ROOT", str(raw_root))
    monkeypatch.setenv("CUA_LITE_DATASETS_ROOT", str(out_root))

    rows = list(
        opencua_use.iter_examples(
            jsonl_path=jsonl_path,
            images_dir=images_dir,
            relative_root=config["images"],
            dataset_type="win_mac",
            os_by_task_id=opencua_use.load_os_by_task_id(),
            verbose=False,
            head=1,
        )
    )
    assert len(rows) == 1

    out_dir = opencua_utils.out_dir_for(root=out_root)
    key, staged = opencua_utils.stage_entry(
        copy.deepcopy(rows[0]),
        store=opencua_utils.make_image_store(out_dir),
        splitter=opencua_utils.make_splitter(),
        variant="win_mac",
    )
    _assert_opencua_win_mac_row(staged, key=key, out_dir=out_dir)

    parquet_path = tmp_path / "roundtrip" / "opencua_win_mac.parquet"
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
    _assert_opencua_win_mac_row(roundtripped, key=key, out_dir=out_dir)


def test_opencua_ubuntu_raw_entrypoint_to_canonical_roundtrip(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    out_root = tmp_path / "out"
    config = opencua_use.DATASETS["ubuntu"]
    jsonl_path = raw_root / config["jsonl"]
    images_dir = raw_root / config["images"]

    _write_png(images_dir / "ubuntu_step0.png", color=(180, 80, 80))
    _write_png(images_dir / "ubuntu_step1.png", color=(80, 180, 120))
    _write_jsonl(
        jsonl_path,
        [
            {
                "task_id": "ubuntu_task_001",
                "instruction": "Use Ubuntu settings to confirm the title.",
                "domain": "desktop",
                "task_completed": True,
                "alignment_score": 1.0,
                "efficiency_score": 0.9,
                "task_difficulty": "easy",
                "traj": [
                    {
                        "image": "ubuntu_step0.png",
                        "value": {
                            "thought": "Click the title field.",
                            "action": "click title",
                            "code": "pyautogui.click(x=0.25, y=0.35)",
                        },
                    },
                    {
                        "image": "ubuntu_step1.png",
                        "value": {
                            "thought": "The title is confirmed.",
                            "action": "finish",
                            "code": "computer.terminate(status='success')",
                        },
                    },
                ],
            }
        ],
    )
    _write_os_metadata(raw_root, {"ubuntu_task_001": "Ubuntu"})
    monkeypatch.setenv("CUA_LITE_RAW_DATASETS_ROOT", str(raw_root))
    monkeypatch.setenv("CUA_LITE_DATASETS_ROOT", str(out_root))

    rows = list(
        opencua_use.iter_examples(
            jsonl_path=jsonl_path,
            images_dir=images_dir,
            relative_root=config["images"],
            dataset_type="ubuntu",
            os_by_task_id=opencua_use.load_os_by_task_id(),
            verbose=False,
            head=1,
        )
    )
    assert len(rows) == 1

    out_dir = opencua_utils.out_dir_for(root=out_root)
    key, staged = opencua_utils.stage_entry(
        copy.deepcopy(rows[0]),
        store=opencua_utils.make_image_store(out_dir),
        splitter=opencua_utils.make_splitter(),
        variant="ubuntu",
    )

    assert key[0] == "desktop"
    assert key[1] == "use"
    assert key[2] in {"train", "validation"}
    assert key[3] == "ubuntu"
    assert staged["metadata"]["others"]["id"] == "agentnet_ubuntu_ubuntu_task_001"
    assert staged["metadata"]["others"]["source_id"] == "ubuntu_task_001"
    assert staged["metadata"]["others"]["os"] == "ubuntu"
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
    assert tool_call_name(call) == "computer"
    assert tool_call_arguments(call)["actions"] == [
        {"action": "click", "coordinate": [250, 350]}
    ]
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": tool_call_id(call),
        "content": [{"type": "image", "index": 1}],
    }
    assert messages[-1] == {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}
    validate_action_batches(messages)
    validate_canonical_rows([staged], "opencua_ubuntu")

    parquet_path = tmp_path / "roundtrip" / "opencua_ubuntu.parquet"
    write_partition([staged], parquet_path)
    raw_row = next(iter_parquet_rows(parquet_path))
    roundtripped = {
        "images": raw_row["images"],
        "messages": coerce_messages(raw_row["messages"]),
        "metadata": coerce_meta(raw_row["metadata"]),
    }
    assert roundtripped == staged
    validate_canonical_rows([roundtripped], "opencua_ubuntu")


def test_opencua_duplicate_task_ids_get_record_indexed_row_ids(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    config = opencua_use.DATASETS["win_mac"]
    jsonl_path = raw_root / config["jsonl"]
    images_dir = raw_root / config["images"]

    for i in range(4):
        _write_png(images_dir / f"dup_step{i}.png", color=(40 * i, 100, 160))
    terminate = "computer.terminate(status='success')"
    _write_jsonl(
        jsonl_path,
        [
            {
                "task_id": "shared_task",
                "instruction": "First task body.",
                "domain": "desktop",
                "traj": [
                    _step("dup_step0.png", "pyautogui.click(x=0.1, y=0.2)"),
                    _step("dup_step1.png", terminate),
                ],
            },
            {
                "task_id": "shared_task",
                "instruction": "Second task body.",
                "domain": "desktop",
                "traj": [
                    _step("dup_step2.png", "pyautogui.click(x=0.7, y=0.8)"),
                    _step("dup_step3.png", terminate),
                ],
            },
        ],
    )
    _write_os_metadata(raw_root, {"shared_task": "Darwin"})
    monkeypatch.setenv("CUA_LITE_RAW_DATASETS_ROOT", str(raw_root))

    rows = list(
        opencua_use.iter_examples(
            jsonl_path=jsonl_path,
            images_dir=images_dir,
            relative_root=config["images"],
            dataset_type="win_mac",
            os_by_task_id=opencua_use.load_os_by_task_id(),
            verbose=False,
        )
    )

    assert [row["metadata"]["others"]["id"] for row in rows] == [
        "agentnet_win_mac_shared_task",
        "agentnet_win_mac_shared_task_record_2",
    ]
    assert [row["metadata"]["others"]["source_id"] for row in rows] == [
        "shared_task",
        "shared_task",
    ]
    assert rows[0]["messages"][0]["content"][1]["text"] == "First task body."
    assert rows[1]["messages"][0]["content"][1]["text"] == "Second task body."
    validate_canonical_rows(rows, "opencua_duplicate_ids")


def test_opencua_head_bounds_duplicate_prescan_and_episode_loop(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    config = opencua_use.DATASETS["win_mac"]
    jsonl_path = raw_root / config["jsonl"]
    images_dir = raw_root / config["images"]

    _write_png(images_dir / "head0.png", color=(20, 80, 160))
    _write_png(images_dir / "head1.png", color=(60, 120, 200))
    _write_jsonl(
        jsonl_path,
        [{
            "task_id": "head_task",
            "instruction": "Only process the first record.",
            "domain": "desktop",
            "traj": [
                _step("head0.png", "pyautogui.click(x=0.1, y=0.2)"),
                _step("head1.png", "computer.terminate(status='success')"),
            ],
        }],
    )
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write("{not json outside head}\n")
    _write_os_metadata(raw_root, {"head_task": "Darwin"})
    monkeypatch.setenv("CUA_LITE_RAW_DATASETS_ROOT", str(raw_root))

    rows = list(
        opencua_use.iter_examples(
            jsonl_path=jsonl_path,
            images_dir=images_dir,
            relative_root=config["images"],
            dataset_type="win_mac",
            os_by_task_id=opencua_use.load_os_by_task_id(),
            verbose=False,
            head=1,
        )
    )

    assert len(rows) == 1
    assert rows[0]["metadata"]["others"]["id"] == "agentnet_win_mac_head_task"
    assert rows[0]["metadata"]["others"]["source_id"] == "head_task"


def test_unterminated_trajectory_publishes_final_action(tmp_path, monkeypatch):
    """A trajectory whose final emitted call is not ``terminate`` keeps it as EOF."""
    raw_root = tmp_path / "raw"
    out_root = tmp_path / "out"
    config = opencua_use.DATASETS["ubuntu"]
    jsonl_path = raw_root / config["jsonl"]
    images_dir = raw_root / config["images"]
    for i in range(2):
        _write_png(images_dir / f"step{i}.png", color=(40 * i, 90, 120))

    click = "pyautogui.click(x=0.25, y=0.35)"
    terminate = "computer.terminate(status='success')"
    _write_jsonl(
        jsonl_path,
        [
            {
                "task_id": "terminates",
                "instruction": "Confirm the title.",
                "traj": [_step("step0.png", click), _step("step1.png", terminate)],
            },
            {
                "task_id": "never_terminates",
                "instruction": "Confirm the title.",
                "traj": [_step("step0.png", click), _step("step1.png", click)],
            },
            {
                "task_id": "terminate_not_last",
                "instruction": "Confirm the title.",
                "traj": [_step("step0.png", terminate), _step("step1.png", click)],
            },
        ],
    )
    _write_os_metadata(raw_root, {
        "terminates": "Ubuntu",
        "never_terminates": "Ubuntu",
        "terminate_not_last": "Ubuntu",
    })
    monkeypatch.setenv("CUA_LITE_RAW_DATASETS_ROOT", str(raw_root))
    monkeypatch.setenv("CUA_LITE_DATASETS_ROOT", str(out_root))

    kwargs: dict[str, Any] = {
        "images_dir": images_dir,
        "relative_root": config["images"],
        "dataset_type": "ubuntu",
        "os_by_task_id": opencua_use.load_os_by_task_id(),
    }
    rows = list(opencua_use.iter_examples(jsonl_path=jsonl_path, verbose=True, **kwargs))
    assert [row["metadata"]["others"]["source_id"] for row in rows] == [
        "terminates",
        "never_terminates",
        "terminate_not_last",
    ]
    validate_canonical_rows(rows, "opencua_terminate_guard")

    records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    for record in records[1:]:
        row = opencua_use.record_to_example(record, record_idx=1, **kwargs)
        assert row["messages"][-1]["role"] == "assistant"
        assert row["messages"][-1].get("tool_calls")
