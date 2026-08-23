from __future__ import annotations

import json

import pytest

from lite.core import LiteCUAMetadata
from lite.core.tools import make_tool_call
from lite.data.utils.rows import validate_canonical_rows


def _coord_row(action: dict) -> dict:
    return {
        "images": [],
        "messages": json.dumps(
            [
                {"role": "user", "content": [{"type": "text", "text": "target"}]},
                {
                    "role": "assistant",
                    "tool_calls": [
                        make_tool_call(
                            "computer",
                            {"actions": [action]},
                            call_id="call_0000",
                        )
                    ],
                },
            ]
        ),
        "metadata": json.dumps(
            LiteCUAMetadata(
                dims=("desktop", "grounding.action"),
                extra_tool_schemas=[],
                valid_actions=None,
                others={},
            ).to_dict()
        ),
    }


def test_validate_canonical_rows_accepts_in_range_coordinates() -> None:
    validate_canonical_rows(
        [
            _coord_row({"action": "click", "coordinate": [0, 0]}),
            _coord_row({"action": "click", "coordinate": [500, 500]}),
            _coord_row({"action": "click", "coordinate": [1000, 1000]}),
        ],
        "ok",
    )


@pytest.mark.parametrize("coord", [[449, 1060], [-3, 50], [790, 179, 808, 1009]])
def test_validate_canonical_rows_rejects_oob_coordinates(coord: list[int]) -> None:
    with pytest.raises(ValueError, match="out-of-range"):
        validate_canonical_rows([_coord_row({"action": "click", "coordinate": coord})], "bad")


def test_validate_canonical_rows_allows_actions_without_coordinates() -> None:
    row = _coord_row({"action": "type", "text": "hello"})
    validate_canonical_rows([row], "ok")
