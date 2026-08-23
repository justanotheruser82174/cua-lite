"""Tests for the HF transport-layer image-dedup fold/unfold.

Folding groups same-image grounding/understanding rows into one row (image
embedded once on HF) and reverses exactly on download. The canonical local
format (one instruction per row) must round-trip byte-identically.

Run: uv run pytest tests/data/hf/test_fold.py -v
"""

from __future__ import annotations

import json

from lite.core import LiteCUAMetadata
from lite.core.tools import make_tool_call
from lite.data.hf.fold import FOLD_COL, fold_rows, should_fold, unfold_rows


def _row(img: str, instr: str, rid: str) -> dict:
    """A canonical single-image grounding row (messages/metadata as JSON strings)."""
    return {
        "images": [f"cua-lite/X/images/{img}.png"],
        "messages": json.dumps(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "index": 0},
                        {"type": "text", "text": instr},
                    ],
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        make_tool_call(
                            "computer",
                            {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                            call_id="call_0000",
                        )
                    ],
                },
            ]
        ),
        "metadata": json.dumps(
            LiteCUAMetadata(
                dims=("browser", "grounding.action"),
                others={"id": rid, "source_id": rid},
            ).to_dict()
        ),
    }


def _sig(r: dict) -> tuple:
    return (tuple(r["images"]), r["messages"], r["metadata"])


def test_should_fold():
    assert should_fold("grounding.action")
    assert should_fold("grounding.bbox")
    assert should_fold("grounding.point")
    assert should_fold("understanding")
    assert not should_fold("use")


def test_fold_groups_by_image():
    rows = [
        _row("imgA", "click login", "a1"),
        _row("imgA", "click signup", "a2"),
        _row("imgA", "select menu", "a3"),
        _row("imgB", "press ok", "b1"),
    ]
    folded = fold_rows(rows)
    assert len(folded) == 2  # two unique images
    assert all(FOLD_COL in f and len(f["images"]) == 1 for f in folded)
    # the 3-instruction image carries 3 members
    big = next(f for f in folded if f["images"][0].endswith("imgA.png"))
    assert len(json.loads(big[FOLD_COL])) == 3


def test_roundtrip_lossless():
    rows = [
        _row("imgA", "click login", "a1"),
        _row("imgA", "click signup", "a2"),
        _row("imgB", "press ok", "b1"),
        _row("imgA", "select menu", "a3"),
        _row("imgC", "scroll down", "c1"),
    ]
    out = unfold_rows(fold_rows(rows))
    # multiset-equal (fold groups by image so order changes, content must not)
    assert sorted(map(_sig, out)) == sorted(map(_sig, rows))


def test_roundtrip_preserves_hf_image_dicts():
    """After upload embeds bytes, the folded row's `images` is an HF Image dict;
    unfold must copy that entry verbatim into every member."""
    folded = fold_rows([_row("imgA", "i1", "a1"), _row("imgA", "i2", "a2")])
    # simulate HF embedding: replace path with an Image dict
    folded[0]["images"] = [{"bytes": b"\x89PNG\r\n\x1a\n", "path": None}]
    out = unfold_rows(folded)
    assert len(out) == 2
    assert all(out[i]["images"][0]["bytes"] == b"\x89PNG\r\n\x1a\n" for i in range(2))
    assert json.loads(out[0]["metadata"])["others"]["id"] == "a1"
    assert json.loads(out[1]["metadata"])["others"]["id"] == "a2"


def test_roundtrip_preserves_role_tool_and_batched_calls():
    row = _row("imgA", "batched", "batched")
    messages = json.loads(row["messages"])
    messages[1]["tool_calls"] = [make_tool_call(
        "computer",
        {"actions": [
            {"action": "click", "coordinate": [1, 2]},
            {"action": "type", "text": "hello"},
        ]},
        call_id="call_0000",
    )]
    messages.append({
        "role": "tool",
        "tool_call_id": "call_0000",
        "content": [{"type": "text", "text": "ok"}],
    })
    row["messages"] = json.dumps(messages)

    [out] = unfold_rows(fold_rows([row]))
    assert json.loads(out["messages"]) == messages


def test_single_image_unique_no_false_merge():
    rows = [_row(f"img{i}", f"instr{i}", str(i)) for i in range(5)]
    folded = fold_rows(rows)
    assert len(folded) == 5  # all distinct images, nothing merged
    assert sorted(map(_sig, unfold_rows(folded))) == sorted(map(_sig, rows))
