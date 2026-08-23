"""Parquet round-trip of the action-batch ``computer`` call shape.

A cheap regression lock proving the leaf parquet I/O does NOT pin a struct
schema that would block the action-batch tool-call shape. A turn's GUI actions
are stored inside ONE ``computer`` tool_call whose payload is a nested
``arguments.actions: [...]`` array, and each
env tool result rides as a ``role:"tool"`` message. Because the staging writer
serializes ``messages`` to a JSON STRING before handing rows to PyArrow
(``lite/data/staging.py::serialize_opaque_json_fields``), the nested/variadic shape
never touches Arrow's struct inference — so it should round-trip
byte-identically. This freezes that property so a future "optimization" that
stores ``messages`` as a typed struct column
(re-introducing a schema pin) trips here.

The JSON-string message path means the action-batch shape survives
write→read unchanged. If this ever fails, the writer started pinning a struct
schema — exactly the footgun this lock guards.

Hermetic: pure ``pyarrow`` write + ``iter_parquet_rows`` read; no model, no env.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/data/staging/test_opaque_json_parquet_roundtrip.py -p no:cacheprovider -q
"""

from __future__ import annotations

import json

import numpy as np

from lite.core import LiteCUAMetadata
from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_arguments
from lite.data.staging import coerce_messages, coerce_meta, iter_parquet_rows, write_partition
from lite.utils.parquet import _EMPTY_STRUCT_CHILD

# ---------------------------------------------------------------------------
# A single SFT-ready row whose ``messages`` carry BOTH canonical features:
#   * a nested action-batch ``computer`` call with ``arguments.actions: [...]``
#   * a per-call ``role:"tool"`` result message
# ---------------------------------------------------------------------------

_BATCHED_MESSAGES = [
    {"role": "user", "content": [{"type": "image", "index": 0},
                                 {"type": "text", "text": "Open the editor and type hi."}]},
    {
        "role": "assistant",
        "content": [{"type": "action_description", "text": "Click the editor, then type."}],
        "tool_calls": [
            make_tool_call("computer", {
                "actions": [
                {"action": "left_click", "coordinate": [120, 240]},
                {"action": "type", "text": "hi"},
                ],
            }, call_id="call_0000")
        ],
    },
    # Per-call env result as a role:"tool" message (new shape).
    {
        "role": "tool",
        "tool_call_id": "call_0000",
        "content": [{"type": "text", "text": "ok"}],
    },
]


def _row() -> dict:
    return {
        "messages": _BATCHED_MESSAGES,
        "metadata": LiteCUAMetadata(dims=("desktop", "use")).to_dict(),
        "images": [],
    }


def test_batched_messages_roundtrip_byte_identical(tmp_path) -> None:
    """Write an action-batch ``computer`` call plus a ``role:"tool"`` message.

    ``write_partition`` JSON-serializes ``messages``; ``coerce_messages`` parses
    the stored JSON string on read. The nested ``actions`` array and the
    ``role:"tool"`` message must survive the trip with no struct-schema coercion.
    """
    out = tmp_path / "train.parquet"
    write_partition([_row()], out)

    rows = list(iter_parquet_rows(out))
    assert len(rows) == 1

    got = coerce_messages(rows[0]["messages"])
    assert got == _BATCHED_MESSAGES

    # The action-batch payload specifically survives (no struct flattening / null-fill).
    tool_calls = got[1]["tool_calls"]
    assert tool_calls[0] == make_tool_call(
        "computer",
        {
            "actions": [
                {"action": "left_click", "coordinate": [120, 240]},
                {"action": "type", "text": "hi"},
            ]
        },
        call_id="call_0000",
    )
    actions = tool_call_arguments(tool_calls[0])["actions"]
    assert actions == [
        {"action": "left_click", "coordinate": [120, 240]},
        {"action": "type", "text": "hi"},
    ]
    # The role:"tool" message survives verbatim.
    assert got[2]["role"] == "tool"


def test_messages_stored_as_json_string_not_struct(tmp_path) -> None:
    """The stored ``messages`` cell is a JSON STRING (not an Arrow struct) — the
    exact property that keeps the variadic action-batch shape schema-pin-free.
    This is the regression lock: if a future writer stores a typed struct column,
    the raw cell stops being ``str`` and this fails."""
    out = tmp_path / "train.parquet"
    write_partition([_row()], out)

    raw = next(iter_parquet_rows(out))["messages"]
    assert isinstance(raw, str), (
        "messages must be stored as a JSON string; a struct column would pin a "
        "schema that blocks the action-batch computer{actions:[...]} shape (D5)."
    )
    # It parses back to the original structure.
    assert json.loads(raw) == _BATCHED_MESSAGES


def test_opaque_fields_roundtrip_zero_arg_dicts_and_numpy_scalars(tmp_path) -> None:
    """Canonical staging write/read must not expose Arrow's empty-struct sentinel.

    ``messages`` and ``metadata`` are the data/export opaque fields. Even when a
    row contains a zero-argument tool call (``arguments == {}``) and numpy scalar
    values from pandas/preproc code, the persisted cells should be JSON strings
    that read back as plain python objects.
    """
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "go back"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "metadata",
                    "data": {
                        "enabled": np.bool_(True),
                        "label": np.str_("same"),
                    },
                }
            ],
            "tool_calls": [make_tool_call("back", {}, call_id="call_0000")],
        },
    ]
    metadata = LiteCUAMetadata(
        dims=("browser", "use"),
        others={
            "success": np.bool_(True),
            "attempt": np.int64(2),
        },
    ).to_dict()
    out = tmp_path / "train.parquet"
    write_partition([{"images": [], "messages": messages, "metadata": metadata}], out)

    row = next(iter_parquet_rows(out))
    raw_messages = row["messages"]
    raw_metadata = row["metadata"]

    assert isinstance(raw_messages, str)
    assert isinstance(raw_metadata, str)
    assert _EMPTY_STRUCT_CHILD not in raw_messages
    assert _EMPTY_STRUCT_CHILD not in raw_metadata

    got_messages = coerce_messages(raw_messages)
    got_metadata = coerce_meta(raw_metadata)

    assert tool_call_arguments(got_messages[1]["tool_calls"][0]) == {}
    assert got_messages[1]["content"][0]["data"] == {
        "enabled": True,
        "label": "same",
    }
    assert got_metadata["others"] == {"success": True, "attempt": 2}
