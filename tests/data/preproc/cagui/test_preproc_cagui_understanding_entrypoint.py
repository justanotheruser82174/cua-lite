from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from lite.core import LiteCUAMetadata
from lite.data.preproc.cagui import understanding as cagui_understanding
from lite.data.preproc.cagui import utils as cagui_utils
from lite.data.staging import coerce_messages, coerce_meta, iter_parquet_rows, write_partition
from lite.data.utils.rows import validate_canonical_rows

_IMAGE_PREFIX = "cua-lite/CAGUI/images/"
_HASHED_IMAGE_RE = re.compile(r"^[0-9a-f]{2}/[0-9a-f]{64}\.jpeg$")


def _write_jpeg(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), color=color).save(path, format="JPEG")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _assert_cagui_understanding_row(
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
    assert key[3] == "cap"
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
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0] == {"type": "image", "index": 0}
    assert "边界框 [100, 200, 500, 600]" in messages[0]["content"][1]["text"]
    assert messages[1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "打开 Wi-Fi 设置"}],
    }
    assert not any("tool_calls" in message for message in messages)
    assert not any("tool_call_id" in message for message in messages)

    assert metadata["others"] == {
        "id": "cagui_cap_1",
        "resolution": [100, 100],
        "os": "android",
        "source": "OpenBMB/CAGUI",
        "source_id": "cap/1.jpeg",
        "language": "zh",
    }
    validate_canonical_rows([row], "cagui_understanding")


def test_cagui_understanding_raw_entrypoint_to_canonical_roundtrip(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    out_root = tmp_path / "out"
    base = raw_root / "OpenBMB" / "CAGUI" / "CAGUI_grounding"
    _write_jpeg(base / "images" / "cap" / "1.jpeg", color=(90, 160, 220))
    _write_jsonl(
        base / "code" / "cap.jsonl",
        [
            {
                "task": "bbox2function",
                "image": "grounding_eval/dataset/images/ignored_prefix/1.jpeg",
                "id": 999,
                "abs_position": "<10, 20, 50, 60>",
                "rel_position": "<0.1, 0.2, 0.5, 0.6>",
                "text": "打开 Wi-Fi 设置",
            }
        ],
    )
    monkeypatch.setenv("CUA_LITE_RAW_DATASETS_ROOT", str(raw_root))
    monkeypatch.setenv("CUA_LITE_DATASETS_ROOT", str(out_root))

    rows, total_seen, skipped_image, skipped_text = cagui_understanding.iter_examples(
        "cap",
        str(raw_root),
        head=1,
        verbose=False,
    )
    assert total_seen == 1
    assert skipped_image == skipped_text == 0
    assert len(rows) == 1

    out_dir = cagui_utils.out_dir_for(root=out_root)
    key, staged = cagui_utils.stage_entry(
        copy.deepcopy(rows[0]),
        store=cagui_utils.make_image_store(out_dir),
        splitter=cagui_utils.make_splitter(),
        variant="cap",
    )
    _assert_cagui_understanding_row(staged, key=key, out_dir=out_dir)

    parquet_path = tmp_path / "roundtrip" / "cagui_understanding.parquet"
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
    _assert_cagui_understanding_row(roundtripped, key=key, out_dir=out_dir)
