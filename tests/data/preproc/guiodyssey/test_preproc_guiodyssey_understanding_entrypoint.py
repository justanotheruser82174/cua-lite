from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from lite.core import LiteCUAMetadata
from lite.data.preproc.guiodyssey import understanding as guiodyssey_understanding
from lite.data.preproc.guiodyssey import utils as guiodyssey_utils
from lite.data.staging import coerce_messages, coerce_meta, iter_parquet_rows, write_partition
from lite.data.utils.rows import validate_canonical_rows

_IMAGE_PREFIX = "cua-lite/GUIOdyssey/images/"
_HASHED_IMAGE_RE = re.compile(r"^[0-9a-f]{2}/[0-9a-f]{64}\.png$")


def test_guiodyssey_loader_fails_loud_on_malformed_annotation(tmp_path: Path) -> None:
    annotations = tmp_path / "hflqf88888" / "GUIOdyssey" / "annotations"
    annotations.mkdir(parents=True)
    (annotations / "broken.json").write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="broken.json"):
        list(guiodyssey_utils.iter_episodes(str(tmp_path)))


def _write_png(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), color=color).save(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _assert_guiodyssey_understanding_row(
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
    assert key[1] == lite_meta.task_type.value == "understanding"
    assert "platform" not in metadata
    assert "task_type" not in metadata
    assert key[2] in {"train", "validation"}
    assert key[3] == "understanding"
    assert lite_meta.valid_actions is None
    assert metadata["extra_tool_schemas"] == []
    assert "split" not in metadata
    assert "split" not in metadata.get("others", {})

    assert len(images) == 1
    image_rel = images[0]
    assert image_rel.startswith(_IMAGE_PREFIX)
    hashed_rel = image_rel.removeprefix(_IMAGE_PREFIX)
    assert _HASHED_IMAGE_RE.match(hashed_rel)
    assert (out_dir / "images" / hashed_rel).is_file()

    assert len(messages) == 2
    assert messages[0] == {
        "role": "user",
        "content": [
            {"type": "image", "index": 0},
            {"type": "text", "text": guiodyssey_understanding._pick_prompt("ep0001", 0)},
        ],
    }
    assert messages[1] == {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "The home screen shows a search box and app shortcuts.",
            }
        ],
    }
    assert not any("tool_calls" in message for message in messages)
    assert not any("tool_call_id" in message for message in messages)

    assert metadata["others"] == {
        "id": "guiodyssey_ep0001_0",
        "resolution": [1080, 2400],
        "os": "android",
        "source": "hflqf88888/GUIOdyssey",
        "source_id": "ep0001_0",
        "category": "Web_Shopping",
        "apps": ["Chrome", "Maps"],
        "device_name": "Pixel 7 Pro",
    }
    validate_canonical_rows([row], "guiodyssey_understanding")


def test_guiodyssey_understanding_raw_entrypoint_to_canonical_roundtrip(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    out_root = tmp_path / "out"
    episode_id = "ep0001"

    _write_png(
        raw_root / guiodyssey_utils.SCREENSHOTS_REL / f"{episode_id}_0.png",
        color=(80, 150, 220),
    )
    _write_json(
        raw_root / guiodyssey_utils.ANNOTATIONS_REL / f"{episode_id}.json",
        {
            "device_info": {
                "w": 1080,
                "h": 2400,
                "device_name": "Pixel 7 Pro",
            },
            "task_info": {
                "category": "Web_Shopping",
                "app": ["Chrome", "Maps"],
            },
            "steps": [
                {
                    "step": 0,
                    "description": "The home screen shows a search box and app shortcuts.",
                }
            ],
        },
    )
    monkeypatch.setenv("CUA_LITE_RAW_DATASETS_ROOT", str(raw_root))
    monkeypatch.setenv("CUA_LITE_DATASETS_ROOT", str(out_root))

    episodes = list(guiodyssey_utils.iter_episodes(str(raw_root), head=1))
    assert len(episodes) == 1
    ep_id, episode = episodes[0]
    assert ep_id == episode_id
    row = guiodyssey_understanding.step_to_record(
        ep_id,
        episode["steps"][0],
        episode["device_info"],
        episode["task_info"],
    )
    assert row is not None

    out_dir = guiodyssey_utils.out_dir_for(root=out_root)
    key, staged = guiodyssey_utils.stage_entry(
        copy.deepcopy(row),
        store=guiodyssey_utils.make_image_store(out_dir),
        splitter=guiodyssey_utils.make_splitter(),
        variant=guiodyssey_understanding.VARIANT,
    )
    _assert_guiodyssey_understanding_row(staged, key=key, out_dir=out_dir)

    parquet_path = tmp_path / "roundtrip" / "guiodyssey_understanding.parquet"
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
    _assert_guiodyssey_understanding_row(roundtripped, key=key, out_dir=out_dir)
