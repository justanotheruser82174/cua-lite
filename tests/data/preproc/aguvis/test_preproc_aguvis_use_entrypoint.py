from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from lite.core import LiteCUAMetadata
from lite.core.tools.calls import tool_call_arguments, tool_call_id, tool_call_name
from lite.core.tools.schemas import tool_schema_name
from lite.data.preproc.aguvis import use as aguvis_use
from lite.data.preproc.aguvis import utils as aguvis_utils
from lite.data.staging import coerce_messages, coerce_meta, iter_parquet_rows, write_partition
from lite.data.utils.messages import extra_tool_schemas_for_messages
from lite.data.utils.rows import (
    iter_canonical_actions,
    validate_action_batches,
    validate_canonical_rows,
)

_IMAGE_PREFIX = "cua-lite/Aguvis/images/"
_HASHED_IMAGE_RE = re.compile(r"^[0-9a-f]{2}/[0-9a-f]{64}\.png$")


def _write_png(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), color=color).save(path)


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def _record(image: str, *, desc: str, code: str) -> dict[str, Any]:
    return {
        "image": image,
        "conversations": [
            {
                "from": "human",
                "value": "Instruction: open settings and tap Wi-Fi\n\nPrevious actions:",
            },
            {"from": "gpt", "value": f"Action: {desc}"},
            {"from": "gpt", "value": code},
        ],
    }


def _tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [call for message in messages for call in (message.get("tool_calls") or [])]


def _assert_aguvis_use_row(
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
    assert key[0] == lite_meta.platform.value == "mobile"
    assert key[1] == lite_meta.task_type.value == "use"
    assert "platform" not in metadata
    assert "task_type" not in metadata
    assert key[2] in {"train", "validation"}
    assert key[3] == "android_control"
    assert lite_meta.valid_actions is None
    assert metadata["others"]["id"] == "aguvis_android_control_episode"
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
            assert call["type"] == "function"
            assert isinstance(tool_call_arguments(call), dict)

    calls = _tool_calls(messages)
    assert [tool_call_id(call) for call in calls] == ["call_0000", "call_0001"]
    assert [tool_call_name(call) for call in calls] == ["open_app", "mobile"]
    assert metadata["extra_tool_schemas"] == extra_tool_schemas_for_messages(messages)
    assert {tool_schema_name(schema) for schema in metadata["extra_tool_schemas"]} == {"open_app"}
    validate_action_batches(messages)

    actions = [
        (name, arguments)
        for message in messages
        for name, arguments in iter_canonical_actions(message)
    ]
    assert actions == [
        ("open_app", {"app_name": "Settings"}),
        ("tap", {"coordinate": [250, 350], "clicks": 1}),
    ]

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

    validate_canonical_rows([row], "aguvis_use")


def test_aguvis_use_raw_entrypoint_to_canonical_roundtrip(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    out_root = tmp_path / "out"
    source_root = raw_root / aguvis_utils.SOURCE_STAGE2
    image_dir = source_root / "android_control" / "images"
    for idx, color in enumerate(((240, 80, 80), (80, 180, 90), (70, 110, 220))):
        _write_png(image_dir / f"episode_{idx}.png", color=color)
    _write_json(
        source_root / "android_control.json",
        [
            _record("episode_0.png", desc="open Settings", code="open('Settings')"),
            _record("episode_1.png", desc="tap Wi-Fi", code="pyautogui.click(x=0.25, y=0.35)"),
            _record("episode_2.png", desc="finish", code="terminate(status='success')"),
        ],
    )
    monkeypatch.setenv("CUA_LITE_RAW_DATASETS_ROOT", str(raw_root))
    monkeypatch.setenv("CUA_LITE_DATASETS_ROOT", str(out_root))

    rows, n_records, skips = aguvis_use.process_subset(
        "android_control",
        aguvis_use.SUBSETS["android_control"],
        str(raw_root),
        head=1,
        verbose=False,
    )
    assert len(rows) == 1
    variant, entry, n_ep_records = rows[0]
    assert variant == "android_control"
    # The ledger is denominated in source records and closes on this window.
    assert (n_records, n_ep_records, dict(skips)) == (3, 3, {})

    out_dir = aguvis_utils.out_dir_for(root=out_root)
    key, staged = aguvis_utils.stage_entry(
        copy.deepcopy(entry),
        store=aguvis_utils.make_image_store(out_dir),
        splitter=aguvis_utils.make_splitter(),
        variant=variant,
    )
    _assert_aguvis_use_row(staged, key=key, out_dir=out_dir)

    parquet_path = tmp_path / "roundtrip" / "aguvis_use.parquet"
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
    _assert_aguvis_use_row(roundtripped, key=key, out_dir=out_dir)


def test_aguvis_use_terminatorless_subset_still_yields_rows(tmp_path, monkeypatch):
    """A subset whose episodes carry NO terminator must still publish rows.

    ``android_control``, ``coat``, ``guide`` and ``miniwob`` have no explicit
    terminator step, so a whole-episode skip on "missing terminal terminate"
    emptied all four (150/150 episodes skipped each). The source cannot support
    the LAST action's ``role:"tool"`` result, so that action remains as the EOF
    supervised label.
    """
    raw_root = tmp_path / "raw"
    out_root = tmp_path / "out"
    source_root = raw_root / aguvis_utils.SOURCE_STAGE2
    image_dir = source_root / "android_control" / "images"
    for idx, color in enumerate(((240, 80, 80), (80, 180, 90), (70, 110, 220))):
        _write_png(image_dir / f"episode_{idx}.png", color=color)
    _write_json(
        source_root / "android_control.json",
        [
            _record("episode_0.png", desc="open Settings", code="open('Settings')"),
            _record("episode_1.png", desc="tap Wi-Fi", code="pyautogui.click(x=0.25, y=0.35)"),
            _record("episode_2.png", desc="tap Save", code="pyautogui.click(x=0.5, y=0.5)"),
        ],
    )
    monkeypatch.setenv("CUA_LITE_RAW_DATASETS_ROOT", str(raw_root))
    monkeypatch.setenv("CUA_LITE_DATASETS_ROOT", str(out_root))

    rows, _n_records, _skips = aguvis_use.process_subset(
        "android_control",
        aguvis_use.SUBSETS["android_control"],
        str(raw_root),
        head=1,
        verbose=False,
    )
    assert len(rows) == 1
    variant, entry, _n_ep_records = rows[0]

    out_dir = aguvis_utils.out_dir_for(root=out_root)
    _key, staged = aguvis_utils.stage_entry(
        copy.deepcopy(entry),
        store=aguvis_utils.make_image_store(out_dir),
        splitter=aguvis_utils.make_splitter(),
        variant=variant,
    )
    assert len(staged["images"]) == 3
    assert [message["role"] for message in staged["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert staged["messages"][-1]["role"] == "assistant"
    assert tool_call_name(staged["messages"][-1]["tool_calls"][0]) == "mobile"
    assert staged["messages"][-1]["content"] == [
        {"type": "action_description", "text": "tap Save"}
    ]
    validate_canonical_rows([staged], "aguvis_use_no_terminator")


def test_aguvis_use_ledger_names_every_drop_and_closes_in_records(tmp_path, monkeypatch, capsys):
    """Every dropped source record is behind a named reason, and the ledger closes.

    Before this, ``process_subset`` printed one undifferentiated ``skip=N`` in
    EPISODES, which is why ``guide.json`` losing 365 of its 973 episodes to a
    single unrenderable call shape went unnoticed. The ledger is denominated in
    source records because one row is a whole episode and consumes many records,
    so records are the only unit in which ``read == kept + sum(skips)`` holds.
    ``missing_image`` is also re-keyed when this host has no image directory for
    the subset at all -- the host lacking data is a different fact from the
    adapter dropping a row.
    """
    raw_root = tmp_path / "raw"
    source_root = raw_root / aguvis_utils.SOURCE_STAGE2
    image_dir = source_root / "android_control" / "images"
    for name in ("keep_0.png", "keep_1.png"):
        _write_png(image_dir / name, color=(10, 20, 30))
    _write_json(
        source_root / "android_control.json",
        [
            _record("keep_0.png", desc="tap Wi-Fi", code="pyautogui.click(x=0.25, y=0.35)"),
            _record("keep_1.png", desc="tap Save", code="pyautogui.click(x=0.5, y=0.5)"),
            # never written to disk -> the whole episode goes, counted in records
            _record("gone_0.png", desc="tap", code="pyautogui.click(x=0.4, y=0.4)"),
            _record("gone_1.png", desc="tap", code="pyautogui.click(x=0.1, y=0.1)"),
            # never groups into an episode at all
            {"image": "", "conversations": []},
        ],
    )
    monkeypatch.setenv("CUA_LITE_RAW_DATASETS_ROOT", str(raw_root))

    rows, n_records, skips = aguvis_use.process_subset(
        "android_control",
        aguvis_use.SUBSETS["android_control"],
        str(raw_root),
        head=None,
        verbose=False,
    )
    kept = sum(n for _variant, _entry, n in rows)
    assert (n_records, kept) == (5, 2)
    assert dict(skips) == {"no_image_name": 1, "missing_image": 2}
    assert n_records == kept + sum(skips.values())
    assert "records skipped" in capsys.readouterr().out

    # A host with no image directory for the subset is a distinct cause.
    monkeypatch.setattr(aguvis_use.os.path, "isdir", lambda _p: False)
    _rows, _n, host_skips = aguvis_use.process_subset(
        "android_control",
        aguvis_use.SUBSETS["android_control"],
        str(raw_root),
        head=None,
        verbose=False,
    )
    assert host_skips["missing_image_dir_absent"] == 2
    assert "missing_image" not in host_skips
