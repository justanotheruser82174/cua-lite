"""Qwen3-VL history windows budget images, not text-only tool turns."""

from __future__ import annotations

import pytest

from lite.agents.core.protocol import BaseProtocol
from lite.agents.models.qwen3_vl.protocol import Qwen3VLHistoryProtocol
from lite.core.messages import group_into_turns
from lite.core.tools import make_tool_call
from lite.core.tools.results import TOOL_RESULT_ERROR_SECTION_HEADER


def _action(desc: str, i: int) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "action_description", "text": desc}],
        "tool_calls": [
            make_tool_call(
                "computer",
                {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                call_id=f"g{i}",
            )
        ],
    }


def _bash(cmd: str, i: int) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "action_description", "text": f"run {cmd}"}],
        "tool_calls": [make_tool_call("bash", {"command": cmd}, call_id=f"b{i}")],
    }


def _img_result(idx: int) -> dict:
    return {
        "role": "tool",
        "tool_call_id": "x",
        "content": [{"type": "image", "index": idx}],
    }


def _txt_result(text: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": "x",
        "content": [{"type": "text", "text": text}],
    }


def _img_error_result(idx: int, text: str, *, metadata: bool = True) -> dict:
    content = [
        {"type": "image", "index": idx},
        {"type": "text", "text": text},
    ]
    if metadata:
        content.append({"type": "metadata", "data": {"is_error": True}})
    return {"role": "tool", "tool_call_id": "x", "content": content}


def _turn0() -> dict:
    return {
        "role": "user",
        "content": [{"type": "image", "index": 0}, {"type": "text", "text": "goal"}],
    }


def _action_only(n_turns: int) -> list[dict]:
    msgs = [_turn0()]
    for i in range(n_turns):
        msgs += [_action(f"click {i}", i), _img_result(i + 1)]
    return msgs


def _rendered_text(messages: list[dict]) -> str:
    out = []
    for message in messages:
        for part in message.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "text":
                out.append(part["text"])
    return "\n".join(out)


@pytest.mark.parametrize("n_turns", [3, 5, 9, 12, 20])
def test_action_only_qwen3_vl_window_start_matches_the_legacy_expression(
    n_turns: int,
) -> None:
    proto = Qwen3VLHistoryProtocol()
    turns = group_into_turns(_action_only(n_turns))
    legacy = max(0, len(turns) - proto.full_history_size)
    image_indices = proto.image_turn_indices(turns)
    new = (
        image_indices[-proto.full_history_size]
        if len(image_indices) > proto.full_history_size
        else 0
    )

    assert new == legacy


def test_qwen3_vl_bash_turns_do_not_push_screenshots_out_of_the_window() -> None:
    proto = Qwen3VLHistoryProtocol(full_history_size=4)
    turns = group_into_turns(
        [_turn0()]
        + [
            message
            for i in range(4)
            for message in (
                _action(f"click {i}", i),
                _img_result(i + 1),
                _bash(f"echo {i}", i),
                _txt_result(f"OUT-{i}"),
            )
        ]
    )
    image_indices = proto.image_turn_indices(turns)
    window_start = image_indices[-4] if len(image_indices) > 4 else 0
    kept = turns[window_start:]

    assert len([turn for turn in kept if proto.turn_has_image(turn)]) == 4


def test_qwen3_vl_summary_carries_an_evicted_bash_result_verbatim() -> None:
    msgs = [_turn0()]
    msgs += [_bash("ls -l", 0), _txt_result("total 12\nconfig.yaml")]
    for i in range(6):
        msgs += [_action(f"click {i}", i), _img_result(i + 2)]

    out = Qwen3VLHistoryProtocol(full_history_size=2).process_messages(msgs)
    text = _rendered_text(out)

    assert "total 12" in text and "config.yaml" in text


def test_qwen3_vl_action_only_summary_has_no_result_text() -> None:
    out = Qwen3VLHistoryProtocol(full_history_size=2).process_messages(_action_only(8))
    text = _rendered_text(out)

    assert "Step 1: click 0" in text
    assert "\nOUT-" not in text


def test_mixed_action_and_bash_in_one_turn_keeps_the_bash_text() -> None:
    turn = group_into_turns(
        [
            {"role": "user", "content": [{"type": "text", "text": "goal"}]},
            _action("click", 0),
            _img_result(1),
            _txt_result("BASH-OUT"),
            _action("click again", 1),
        ]
    )[1]

    assert BaseProtocol.turn_has_image(turn) is True
    assert Qwen3VLHistoryProtocol()._tool_result_summary_text(turn) == "BASH-OUT"


def test_last_turn_has_no_following_result() -> None:
    assert Qwen3VLHistoryProtocol()._tool_result_summary_text(None) == ""


def test_a_result_with_no_text_part_yields_nothing() -> None:
    turn = group_into_turns(
        [
            {
                "role": "tool",
                "tool_call_id": "x",
                "content": [{"type": "metadata", "data": {"returncode": 0}}],
            },
            _action("click", 0),
        ]
    )[0]

    assert Qwen3VLHistoryProtocol()._tool_result_summary_text(turn) == ""


def test_an_errored_bash_result_is_still_carried() -> None:
    turn = group_into_turns(
        [
            {
                "role": "tool",
                "tool_call_id": "x",
                "content": [
                    {"type": "text", "text": "bash failed: agent bash shell exited"},
                    {"type": "metadata", "data": {"is_error": True}},
                ],
            },
            _action("click", 0),
        ]
    )[0]

    assert "bash failed" in Qwen3VLHistoryProtocol()._tool_result_summary_text(turn)


def test_image_error_summary_uses_structured_metadata_or_own_header() -> None:
    turn = group_into_turns(
        [
            _img_error_result(
                1,
                f"## AXTree:\nbutton\n\n{TOOL_RESULT_ERROR_SECTION_HEADER}\nclick failed",
                metadata=False,
            ),
            _action("click", 0),
        ]
    )[0]

    assert "click failed" in Qwen3VLHistoryProtocol()._tool_result_summary_text(turn)

    marked_turn = group_into_turns(
        [
            _img_error_result(
                1,
                f"## AXTree:\nbutton\n\n{TOOL_RESULT_ERROR_SECTION_HEADER}\nclick failed",
                metadata=True,
            ),
            _action("click", 0),
        ]
    )[0]
    assert "click failed" in Qwen3VLHistoryProtocol()._tool_result_summary_text(marked_turn)


def test_image_error_summary_does_not_parse_inline_error_phrases() -> None:
    turn = group_into_turns(
        [
            _img_error_result(
                1,
                f"AXTree text mentions {TOOL_RESULT_ERROR_SECTION_HEADER}\nnot a protocol header",
                metadata=False,
            ),
            _action("click", 0),
        ]
    )[0]

    assert Qwen3VLHistoryProtocol()._tool_result_summary_text(turn) == ""


def test_qwen3_vl_evicted_image_tool_result_with_own_error_header_survives() -> None:
    error_text = f"{TOOL_RESULT_ERROR_SECTION_HEADER}\nSENTINEL_click_failed_without_metadata"
    msgs = [
        _turn0(),
        _action("bad click", 0),
        _img_error_result(1, error_text, metadata=False),
        _action("recover", 1),
        _img_result(2),
        _action("finish", 2),
        _img_result(3),
    ]

    text = _rendered_text(Qwen3VLHistoryProtocol(full_history_size=1).process_messages(msgs))

    assert error_text in text


def test_a_user_observation_is_not_treated_as_a_tool_result() -> None:
    turn = group_into_turns(
        [
            {"role": "user", "content": [{"type": "text", "text": "the goal text"}]},
            _action("click", 0),
        ]
    )[0]

    assert Qwen3VLHistoryProtocol()._tool_result_summary_text(turn) == ""


def test_consecutive_evicted_bash_turns_each_keep_their_output() -> None:
    msgs = [_turn0()]
    for i in range(3):
        msgs += [_bash(f"echo {i}", i), _txt_result(f"OUT-{i}")]
    for i in range(6):
        msgs += [_action(f"click {i}", i), _img_result(i + 1)]

    text = _rendered_text(Qwen3VLHistoryProtocol(full_history_size=2).process_messages(msgs))

    for i in range(3):
        assert f"OUT-{i}" in text


def test_qwen3_vl_single_turn_and_empty_input_do_not_raise() -> None:
    proto = Qwen3VLHistoryProtocol()

    assert proto.process_messages([]) == []
    assert proto.process_messages([_turn0()]) is not None
    proto.process_messages([_turn0(), _action("click", 0)])


def test_a_turn_with_no_assistant_is_skipped_by_the_summary() -> None:
    msgs = [_turn0()]
    for i in range(6):
        msgs += [_action(f"click {i}", i), _img_result(i + 1)]
    msgs.append(_txt_result("TRAILING"))

    out = Qwen3VLHistoryProtocol(full_history_size=2).process_messages(msgs)

    assert out


def test_compact_qwen3_vl_keeps_exactly_one_image_and_still_summarizes_bash() -> None:
    msgs = [_turn0()]
    for i in range(4):
        msgs += [_bash(f"echo {i}", i), _txt_result(f"OUT-{i}")]
        msgs += [_action(f"click {i}", i), _img_result(i + 1)]

    out = Qwen3VLHistoryProtocol(full_history_size=1).process_messages(msgs)
    text = _rendered_text(out)
    image_count = sum(
        1
        for message in out
        for part in (message.get("content") or [])
        if isinstance(part, dict) and part.get("type") == "image"
    )

    assert image_count == 1
    for i in range(4):
        assert f"OUT-{i}" in text


def test_qwen3_vl_summary_history_size_still_caps_old_bash_output() -> None:
    msgs = [_turn0()]
    msgs += [_bash("ancient", 0), _txt_result("ANCIENT-OUT")]
    for i in range(12):
        msgs += [_action(f"click {i}", i), _img_result(i + 1)]

    proto = Qwen3VLHistoryProtocol(full_history_size=2, summary_history_size=3)
    text = _rendered_text(proto.process_messages(msgs))
    generous = Qwen3VLHistoryProtocol(full_history_size=2, summary_history_size=100)

    assert "ANCIENT-OUT" not in text
    assert "ANCIENT-OUT" in _rendered_text(generous.process_messages(msgs))


def test_qwen3_vl_subclasses_do_not_override_the_windowing_method() -> None:
    import lite.agents.extensions  # noqa: F401
    from lite.agents.bootstrap import register_all

    register_all()

    stack = list(Qwen3VLHistoryProtocol.__subclasses__())
    while stack:
        subclass = stack.pop()
        stack.extend(subclass.__subclasses__())
        assert "_compute_summary_and_window" not in vars(subclass), (
            f"{subclass.__name__} overrides _compute_summary_and_window and would "
            "miss the image-budget window and text-result summary"
        )


def test_a_turn_with_no_observations_at_all_is_benign() -> None:
    msgs = [
        _turn0(),
        _action("a1", 1),
        _action("a2", 2),
        _img_result(1),
        _action("a3", 3),
    ]
    turns = group_into_turns(msgs)

    assert BaseProtocol.turn_has_image(turns[1]) is False
    assert turns[1]["observations"] == []
    assert Qwen3VLHistoryProtocol()._tool_result_summary_text(turns[1]) == ""
    assert BaseProtocol.image_turn_indices(turns) == [0, 2]
