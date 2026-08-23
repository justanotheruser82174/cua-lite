from __future__ import annotations

import pytest

from lite.core.tools import make_tool_call
from lite.data.preproc import common as preproc_common


def _oob_row(wrapper: str, coord: list[int]) -> dict:
    return {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    make_tool_call(
                        wrapper,
                        {
                            "actions": [
                                {
                                    "action": "tap" if wrapper == "mobile" else "click",
                                    "coordinate": coord,
                                },
                            ]
                        },
                        call_id="call_0000",
                    )
                ],
            }
        ],
    }


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"coordinate": [1500, 20]}, True),
        ({"start_coordinate": [20, -1], "coordinate": [500, 20]}, True),
        ({"coordinate": [0, 1000]}, False),
    ],
)
def test_preproc_oob_gate_scans_direct_tool_calls(
    arguments: dict,
    expected: bool,
) -> None:
    row = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [make_tool_call("click", arguments, call_id="call_0000")],
            }
        ],
    }

    assert preproc_common.has_oob_coordinate(row) is expected


@pytest.mark.parametrize("wrapper", ["computer", "mobile"])
@pytest.mark.parametrize(
    ("coord_key", "coord"),
    [
        ("coordinate", [1500, 20]),
        ("start_coordinate", [20, -1]),
    ],
)
def test_preproc_oob_gate_scans_action_batches(
    wrapper: str,
    coord_key: str,
    coord: list[int],
) -> None:
    action = "tap" if wrapper == "mobile" else "click"
    row = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    make_tool_call(
                        wrapper,
                        {"actions": [{"action": action, coord_key: coord}]},
                        call_id="call_0000",
                    )
                ],
            }
        ],
    }

    assert preproc_common.has_oob_coordinate(row) is True


@pytest.mark.parametrize("wrapper", ["computer", "mobile"])
def test_preproc_oob_gate_keeps_in_range_action_batches(
    wrapper: str,
) -> None:
    action = "tap" if wrapper == "mobile" else "click"
    row = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    make_tool_call(
                        wrapper,
                        {
                            "actions": [
                                {
                                    "action": action,
                                    "start_coordinate": [0, 1000],
                                    "coordinate": [500, 20],
                                }
                            ],
                        },
                        call_id="call_0000",
                    )
                ],
            }
        ],
    }

    assert preproc_common.has_oob_coordinate(row) is False


def test_oob_gate_descends_into_mobile_batches() -> None:
    assert preproc_common.has_oob_coordinate(_oob_row("mobile", [1500, 20])) is True
    assert preproc_common.has_oob_coordinate(_oob_row("computer", [1500, 20])) is True
    assert preproc_common.has_oob_coordinate(_oob_row("mobile", [500, 20])) is False
    assert preproc_common.has_oob_coordinate(_oob_row("computer", [500, 20])) is False


def test_oob_gate_scans_tuple_coordinates() -> None:
    row = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    make_tool_call(
                        "click",
                        {"coordinate": (1500, 20)},
                        call_id="call_0000",
                    )
                ],
            }
        ],
    }

    assert preproc_common.has_oob_coordinate(row) is True



def test_preproc_oob_gate_descends_into_batched_computer_actions() -> None:
    row = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    make_tool_call(
                        "computer",
                        {"actions": [{"action": "click", "coordinate": [1500, 5]}]},
                        call_id="call_computer",
                    )
                ],
            }
        ],
    }

    assert preproc_common.has_oob_coordinate(row) is True
