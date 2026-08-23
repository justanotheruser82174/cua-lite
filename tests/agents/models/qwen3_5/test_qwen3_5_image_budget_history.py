"""Qwen3.5 history windows budget images, not text-only tool turns."""

from __future__ import annotations

import pytest

from lite.agents.core.protocol import BaseProtocol
from lite.agents.models.qwen3_5.protocol import Qwen3_5HistoryProtocol
from lite.core.messages import group_into_turns
from lite.core.tools import make_tool_call
from lite.core.tools.results import TOOL_RESULT_ERROR_SECTION_HEADER

COLLAPSE = "This screenshot has been collapsed."


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
def test_action_only_qwen3_5_folds_exactly_the_legacy_prefix(n_turns: int) -> None:
    proto = Qwen3_5HistoryProtocol()
    turns = group_into_turns(_action_only(n_turns))
    legacy_folded = proto._compute_folded_count(len(turns))
    image_indices = proto.image_turn_indices(turns)

    assert proto._compute_folded_count(len(image_indices)) == legacy_folded
    assert set(image_indices[:legacy_folded]) == set(range(legacy_folded))


def test_qwen3_5_never_collapses_a_text_tool_result() -> None:
    msgs = [_turn0()]
    for i in range(8):
        msgs += [_bash(f"echo {i}", i), _txt_result(f"OUTPUT-{i}")]

    out = Qwen3_5HistoryProtocol(image_max=1, fold_size=1).process_messages(msgs)

    assert COLLAPSE not in _rendered_text(out)
    for i in range(8):
        assert f"OUTPUT-{i}" in _rendered_text(out)


def test_qwen3_5_still_collapses_image_turns() -> None:
    out = Qwen3_5HistoryProtocol(image_max=1, fold_size=1).process_messages(_action_only(8))

    assert COLLAPSE in _rendered_text(out)


def test_qwen3_5_text_turns_do_not_consume_the_fold_budget() -> None:
    proto = Qwen3_5HistoryProtocol(image_max=2, fold_size=2)
    pure = _rendered_text(proto.process_messages(_action_only(6)))

    mixed_msgs = [_turn0()]
    for i in range(6):
        mixed_msgs += [_action(f"click {i}", i), _img_result(i + 1)]
        mixed_msgs += [_bash(f"echo {i}", i), _txt_result(f"OUT-{i}")]
    mixed = _rendered_text(proto.process_messages(mixed_msgs))

    assert mixed.count(COLLAPSE) == pure.count(COLLAPSE)


def test_qwen3_5_folded_mixed_turn_keeps_text_only_tool_result() -> None:
    msgs = [
        _turn0(),
        _action("click and inspect", 0),
        _img_result(1),
        _txt_result("BASH-OUT"),
    ]
    for i in range(4):
        msgs += [_action(f"click {i}", i + 1), _img_result(i + 2)]

    text = _rendered_text(Qwen3_5HistoryProtocol(image_max=1, fold_size=1).process_messages(msgs))

    assert COLLAPSE in text
    assert "BASH-OUT" in text


def test_qwen3_5_folded_role_tool_image_error_keeps_projected_text() -> None:
    error_text = f"{TOOL_RESULT_ERROR_SECTION_HEADER}\nelement not visible"
    msgs = [_turn0(), _action("click", 0), _img_error_result(1, error_text)]
    for i in range(4):
        msgs += [_action(f"click {i}", i + 1), _img_result(i + 2)]

    text = _rendered_text(Qwen3_5HistoryProtocol(image_max=1, fold_size=1).process_messages(msgs))

    assert COLLAPSE in text
    assert "element not visible" in text


def test_qwen3_5_evicted_image_tool_result_with_own_error_header_survives() -> None:
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

    text = _rendered_text(Qwen3_5HistoryProtocol(history_n=1).process_messages(msgs))

    assert error_text in text


def test_an_all_bash_trajectory_folds_and_evicts_nothing() -> None:
    msgs = [{"role": "user", "content": [{"type": "text", "text": "goal"}]}]
    for i in range(12):
        msgs += [_bash(f"echo {i}", i), _txt_result(f"OUT-{i}")]
    turns = group_into_turns(msgs)

    assert BaseProtocol.image_turn_indices(turns) == []

    q35 = Qwen3_5HistoryProtocol(image_max=1, fold_size=1)
    assert q35._compute_folded_count(0) == 0
    text = _rendered_text(q35.process_messages(msgs))
    assert COLLAPSE not in text
    for i in range(12):
        assert f"OUT-{i}" in text


def test_qwen3_5_single_turn_and_empty_input_do_not_raise() -> None:
    proto = Qwen3_5HistoryProtocol()

    assert proto.process_messages([]) == []
    assert proto.process_messages([_turn0()]) is not None
    proto.process_messages([_turn0(), _action("click", 0)])


def test_compact_qwen3_5_history_n_1_keeps_bash_output_in_the_summary() -> None:
    msgs = [_turn0()]
    for i in range(4):
        msgs += [_bash(f"echo {i}", i), _txt_result(f"OUT-{i}")]
        msgs += [_action(f"click {i}", i), _img_result(i + 1)]

    text = _rendered_text(Qwen3_5HistoryProtocol(history_n=1).process_messages(msgs))

    for i in range(4):
        assert f"OUT-{i}" in text


def test_qwen3_5_has_no_summary_cap_so_the_oldest_bash_output_survives() -> None:
    msgs = [_turn0()]
    msgs += [_bash("ancient", 0), _txt_result("ANCIENT-OUT")]
    for i in range(12):
        msgs += [_action(f"click {i}", i), _img_result(i + 1)]

    text = _rendered_text(Qwen3_5HistoryProtocol(history_n=2).process_messages(msgs))

    assert "ANCIENT-OUT" in text


def test_qwen3_5_subclasses_do_not_override_the_windowing_method() -> None:
    import lite.agents.extensions  # noqa: F401
    from lite.agents.bootstrap import register_all

    register_all()

    stack = list(Qwen3_5HistoryProtocol.__subclasses__())
    while stack:
        subclass = stack.pop()
        stack.extend(subclass.__subclasses__())
        assert "_compute_summary_and_window" not in vars(subclass), (
            f"{subclass.__name__} overrides _compute_summary_and_window and would "
            "miss the image-budget window and text-result summary"
        )
