from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from lite.core import LiteCUAMetadata
from lite.data.preproc.gui360 import understanding as gui360_understanding
from lite.data.preproc.gui360 import utils as gui360_utils
from lite.data.staging import coerce_messages, coerce_meta, iter_parquet_rows, write_partition
from lite.data.utils.rows import validate_canonical_rows

_IMAGE_PREFIX = "cua-lite/GUI-360/images/"
_HASHED_IMAGE_RE = re.compile(r"^[0-9a-f]{2}/[0-9a-f]{64}\.png$")


def _write_png(path: Path, *, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_gui360_streaming_array_loader_rejects_truncated_input(tmp_path: Path) -> None:
    source = tmp_path / "truncated.json"
    source.write_text('[{"id": 1}', encoding="utf-8")

    with pytest.raises(ValueError, match="Truncated JSON array"):
        list(gui360_utils.iter_json_array(source))


def _assert_gui360_understanding_row(
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
    assert key[1] == lite_meta.task_type.value == "understanding"
    assert "platform" not in metadata
    assert "task_type" not in metadata
    assert key[2] in {"train", "validation"}
    assert key[3] == "screen_parsing"
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

    assert messages[0] == {
        "role": "user",
        "content": [
            {"type": "image", "index": 0},
            {"type": "text", "text": gui360_understanding.PROMPT},
        ],
    }
    assert messages[1]["role"] == "assistant"
    assert set(messages[1]) == {"role", "content"}
    controls = json.loads(messages[1]["content"][0]["text"])
    # Two of the four source controls survive: the in-bounds one verbatim, and
    # the 2-px resize overflow clipped to the image edge (1038 -> 1036, so the
    # normalized right edge lands on 1000 instead of origin/main's 1002). The
    # row below the viewport and the zero-area phantom are gone.
    assert controls == [
        {"control_text": "Save", "control_rect": [39, 27, 154, 137]},
        {"control_text": "Formula Bar", "control_rect": [941, 356, 1000, 383]},
    ]
    for control in controls:
        assert all(0 <= v <= 1000 for v in control["control_rect"]), control
    assert not any("tool_calls" in message for message in messages)
    assert not any("tool_call_id" in message for message in messages)

    assert metadata["others"] == {
        "id": "gui360_screen_1",
        "resolution": [1036, 728],
        "os": "windows",
        "source": "vyokky/GUI-360",
        "source_id": "screen_parsing_train_resize",
    }
    validate_canonical_rows([row], "gui360_understanding")


def _screen_parsing_record(
    controls: list[dict[str, Any]], *, record_id: str = "gui360_screen_1"
) -> dict[str, Any]:
    return {
        "id": record_id,
        "images": ["images\\screen.png"],
        "conversation": [
            {"from": "human", "value": "Describe the screen controls."},
            {"from": "gpt", "value": json.dumps(controls)},
        ],
    }


# The real ``screen_parsing_train_resize`` geometry. Upstream ``control_rect``
# values are UIA bounds captured on the 1040x736 screenshot and shipped against
# the 1036x728 resize, so a 2-px right-edge overflow is the CORPUS-TYPICAL shape,
# not an edge case: it is 24,898 of the 37,818 out-of-bounds rects in a
# 801,796-control sample, and the very first record of the file carries one.
_IMAGE_SIZE = (1036, 728)
_SOURCE_CONTROLS = [
    # in bounds -- passes through untouched
    {
        "control_text": "Save",
        "control_rect": [40, 20, 160, 100],
        "control_type": "Button",
        "source": "uia",
        "label": "ignored",
    },
    # the record-#1 rect verbatim: right=1038 on a 1036-px-wide image, 2 px of
    # resize rounding. Clipped, kept.
    {"control_text": "Formula Bar", "control_rect": [975, 259, 1038, 279], "source": "uia"},
    # an Excel row scrolled below the viewport: top == height, nothing on screen.
    {"control_text": "19", "control_rect": [8, 728, 84, 748], "source": "uia"},
    # the in-bounds zero-area UIA phantom (98 of them in the same sample).
    {"control_text": "", "control_rect": [8, 8, 8, 8], "source": "uia"},
]


def test_gui360_understanding_raw_entrypoint_to_canonical_roundtrip(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    out_root = tmp_path / "out"
    subset = raw_root / gui360_understanding.SUBSET_REL
    image_path = subset / "images" / "screen.png"
    json_path = subset / "training_data.json"

    _write_png(image_path, size=_IMAGE_SIZE, color=(80, 140, 220))
    _write_json(json_path, [_screen_parsing_record(_SOURCE_CONTROLS)])
    monkeypatch.setenv("CUA_LITE_RAW_DATASETS_ROOT", str(raw_root))
    monkeypatch.setenv("CUA_LITE_DATASETS_ROOT", str(out_root))

    rows = list(gui360_understanding.process_records(str(json_path), head=1, verbose=False))
    assert len(rows) == 1

    out_dir = gui360_utils.out_dir_for(root=out_root)
    key, staged = gui360_utils.stage_entry(
        copy.deepcopy(rows[0]),
        store=gui360_utils.make_image_store(out_dir),
        splitter=gui360_utils.make_splitter(),
        variant=gui360_understanding.VARIANT,
    )
    _assert_gui360_understanding_row(staged, key=key, out_dir=out_dir)

    parquet_path = tmp_path / "roundtrip" / "gui360_understanding.parquet"
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
    _assert_gui360_understanding_row(roundtripped, key=key, out_dir=out_dir)


@pytest.mark.parametrize(
    ("rect", "expected_rect"),
    [
        # The corpus-typical case: 2 px of resize rounding on the real geometry.
        # Clipped, not dropped -- and the clip is what keeps the answer inside the
        # [0, 1000] range its own prompt promises (origin/main emitted 1002 here).
        ([975, 259, 1038, 279], [941, 356, 1000, 383]),
        # 1 px over each edge, the smallest overflow expressible.
        ([40, 20, 1037, 100], [39, 27, 1000, 137]),
        ([40, 20, 160, 729], [39, 27, 154, 1000]),
        # Negative origin, the mirror-image rounding overflow.
        ([-1, -1, 160, 100], [0, 0, 154, 137]),
        # Fully in bounds -- byte-identical passthrough, nothing to clip.
        ([40, 20, 160, 100], [39, 27, 154, 137]),
    ],
)
def test_gui360_understanding_clips_overflowing_control_rect(rect, expected_rect):
    controls, n_clipped, n_offscreen = gui360_understanding.build_controls(
        [{"control_text": "Save", "control_rect": rect}],
        width=_IMAGE_SIZE[0],
        height=_IMAGE_SIZE[1],
        record_id="gui360_clip",
    )
    assert controls == [{"control_text": "Save", "control_rect": expected_rect}]
    assert n_offscreen == 0
    assert n_clipped == (1 if list(rect) != [40, 20, 160, 100] else 0)


@pytest.mark.parametrize(
    "rect",
    [
        [8, 728, 84, 748],  # top == height: an Excel row below the viewport
        [304, 2931, 507, 3181],  # a PowerPoint thumbnail 2,200 px below the screen
        [1036, 259, 1099, 279],  # left == width: off the right edge entirely
        [8, 8, 8, 8],  # in-bounds zero-area UIA phantom
        [160, 20, 40, 100],  # right < left
        [40, 100, 160, 20],  # bottom < top
    ],
)
def test_gui360_understanding_drops_control_with_no_visible_area(rect):
    """A control with no on-screen area is dropped and COUNTED -- never raised.

    The prompt asks for controls "visible in this screenshot", so a rect whose
    intersection with the image is empty has no answer to give. Upstream ships
    these at every scale (20 px to 2,453 px of overflow), which is why no pixel
    tolerance can be the rule and why refusing the record cannot be: 96.4% of
    records carry at least one out-of-bounds rect.
    """
    controls, n_clipped, n_offscreen = gui360_understanding.build_controls(
        [{"control_text": "19", "control_rect": rect}],
        width=_IMAGE_SIZE[0],
        height=_IMAGE_SIZE[1],
        record_id="gui360_offscreen",
    )
    assert controls == []
    assert (n_clipped, n_offscreen) == (0, 1)


@pytest.mark.parametrize("rect", [None, [40, 20, 160], [40, "bad", 160, 100], "40,20,160,100"])
def test_gui360_understanding_rejects_control_rect_that_is_not_four_numbers(rect):
    """The one boundary check kept: an annotation that is not a rect at all.

    Where a rect SITS is a geometric question (clip or drop). Whether the
    upstream JSON even carries four numbers is a parse question about an external
    file, and origin/main refused it too -- measured 0 occurrences in 801,796
    controls, so this never fires on the real corpus.
    """
    with pytest.raises(ValueError, match="Malformed control_rect"):
        gui360_understanding.build_controls(
            [{"control_text": "Save", "control_rect": rect}],
            width=_IMAGE_SIZE[0],
            height=_IMAGE_SIZE[1],
            record_id="gui360_bad_rect",
        )


def test_gui360_understanding_reports_clip_and_drop_tallies_without_verbose(
    tmp_path, monkeypatch, capsys
):
    """Both tallies are printed on a plain run, not only under ``--verbose``.

    A per-control drop visible only in verbose mode is how a partition loses rows
    under a clean success line.
    """
    raw_root = tmp_path / "raw"
    subset = raw_root / gui360_understanding.SUBSET_REL
    _write_png(subset / "images" / "screen.png", size=_IMAGE_SIZE, color=(80, 140, 220))
    json_path = subset / "training_data.json"
    _write_json(
        json_path,
        [
            _screen_parsing_record(_SOURCE_CONTROLS, record_id="gui360_screen_1"),
            # every control off-screen -> the record itself is skipped, counted,
            # and NOT raised
            _screen_parsing_record(_SOURCE_CONTROLS[2:], record_id="gui360_screen_2"),
        ],
    )
    monkeypatch.setenv("CUA_LITE_RAW_DATASETS_ROOT", str(raw_root))

    rows = list(gui360_understanding.process_records(str(json_path), head=None, verbose=False))
    assert [row["metadata"]["others"]["id"] for row in rows] == ["gui360_screen_1"]

    out = capsys.readouterr().out
    assert "Records: 2 read, 1 emitted" in out
    assert "1 no on-screen controls" in out
    assert "Controls: 1 clipped to image bounds, 4 dropped (no visible area)" in out
