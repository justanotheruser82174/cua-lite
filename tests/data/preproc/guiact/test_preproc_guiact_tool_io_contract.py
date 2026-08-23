"""GUIAct preproc tool/result contract tests."""

from __future__ import annotations

import pytest
from data.preproc._tool_io_helpers import (
    _actions,
    _all_calls,
    _assert_final_action_row,
    _assert_first_action_result_is_tool,
    _assert_structural_done_row,
    _load_preproc_script,
)

from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.schemas import tool_schema_name
from lite.data.preproc.guiact import use as guiact_use
from lite.data.preproc.guiact import utils as guiact_utils
from lite.data.utils.rows import validate_canonical_rows

guiact_grounding = _load_preproc_script(
    "lite/data/preproc/guiact/grounding-action.py",
    "cua_lite_test_guiact_grounding_action",
)


def test_guiact_grounding_action_derives_response_schema(monkeypatch) -> None:
    monkeypatch.setattr(guiact_grounding, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    row = guiact_grounding.record_to_example(
        {
            "uid": "u1",
            "image_id": "img1",
            "image_size": {"width": 1000, "height": 1000},
            "question": "Answer the question",
            "actions_label": [{"name": "answer", "text": "final answer"}],
        },
        split="train",
    )

    assert [tool_call_name(call) for call in _all_calls(row)] == ["response"]
    assert [
        tool_schema_name(schema)
        for schema in row["metadata"]["extra_tool_schemas"]
    ] == ["response"]


def test_guiact_post_action_screenshot_is_tool(monkeypatch):
    monkeypatch.setattr(guiact_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")

    def step(image_id: str, point: str) -> dict:
        return {
            "image_id": image_id,
            "image_size": {"width": 1000, "height": 1000},
            "question": "finish",
            "actions_label": [{"name": "click", "point": {"related": point}}],
        }

    row = guiact_use.episode_to_example(
        "ep",
        [
            step("uid_record_ep_step_00", "<point>0.1, 0.2</point>"),
            step("uid_record_ep_step_01", "<point>0.3, 0.4</point>"),
            {
                "image_id": "uid_record_ep_step_02",
                "image_size": {"width": 1000, "height": 1000},
                "question": "finish",
                "actions_label": [{"name": "answer", "text": ""}],
            },
        ],
        config=guiact_use.SUBSETS["web_multi"],
    )
    _assert_first_action_result_is_tool(row)
    _assert_structural_done_row(row)


def _guiact_answer_step(image_id: str, text: str, *, before: list[dict] | None = None) -> dict:
    return {
        "image_id": image_id,
        "image_size": {"width": 1000, "height": 1000},
        "question": "who wrote it?",
        "actions_label": [*(before or []), {"name": "answer", "text": text}],
    }


def _guiact_click_step(image_id: str, point: str) -> dict:
    return {
        "image_id": image_id,
        "image_size": {"width": 1000, "height": 1000},
        "question": "who wrote it?",
        "actions_label": [{"name": "click", "point": {"related": point}}],
    }


@pytest.fixture
def guiact_env(monkeypatch):
    monkeypatch.setattr(guiact_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")


def test_guiact_terminal_answer_text_is_the_final_turn(guiact_env):
    """The final turn CARRIES the answer; ``Done.`` would destroy it.

    GUIAct's terminal ``answer`` is the QA answer, and it survives nowhere else:
    ``terminate_outcome_others`` maps a success outcome to ``{}``. Rewriting the
    final turn to the structural ``Done.`` marker lost the text on every kept
    web-multi and smartphone row.
    """
    row = guiact_use.episode_to_example(
        "ep",
        [
            _guiact_click_step("uid_record_ep_step_00", "<point>0.1, 0.2</point>"),
            _guiact_answer_step("uid_record_ep_step_01", "Ada Lovelace"),
        ],
        config=guiact_use.SUBSETS["web_multi"],
    )

    assert row["messages"][-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Ada Lovelace"}],
    }
    _assert_first_action_result_is_tool(row)
    validate_canonical_rows([row], "guiact-answer")


def _guiact_phone_tap(image_id: str) -> dict:
    return {
        "image_id": image_id,
        "image_size": {"width": 1000, "height": 1000},
        "question": "who wrote it?",
        "actions_label": {"name": "tap", "point": {"related": "<point>0.1, 0.2</point>"}},
    }


def _guiact_phone_answer(image_id: str, text: str) -> dict:
    return {
        "image_id": image_id,
        "image_size": {"width": 1000, "height": 1000},
        "question": "who wrote it?",
        "actions_label": {"name": "answer", "text": text},
    }


def test_guiact_smartphone_terminal_answer_text_is_the_final_turn(guiact_env):
    """An AUTHORED smartphone answer is kept, and lone-dict ``actions_label`` works.

    Smartphone stores ``actions_label`` as a lone dict rather than a list, so this
    is the mobile twin of the web test above. It no longer pins "the same rule
    applies to every smartphone answer" -- the sibling test below shows the canned
    markers that make up 100% of the real corpus are terminators, not answers --
    but it does pin the half that survives: a string the marker vocabulary does
    not contain is authored prose, and prose is never destroyed. That is what
    keeps the classification total over a source this adapter does not own.
    """
    row = guiact_use.episode_to_example(
        "ep",
        [
            _guiact_phone_tap("uid_episode_ep_step_00"),
            _guiact_phone_answer("uid_episode_ep_step_01", "Ada Lovelace"),
        ],
        config=guiact_use.SUBSETS["smartphone"],
    )

    assert row["messages"][-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Ada Lovelace"}],
    }
    assert "terminate_status" not in row["metadata"]["others"]
    validate_canonical_rows([row], "guiact-smartphone-answer")


def test_guiact_smartphone_canned_completion_marker_is_not_an_answer(guiact_env):
    """``"task complete"`` is a terminator, so the final turn is ``Done.``.

    This is 7,466 of the 7,476 smartphone episodes -- the whole subset is one of
    two canned strings -- so publishing the text as the model's answer trained
    every mobile row to end on the visible words "task complete" instead of the
    structural marker. ``status="success"`` asserts nothing past ``Done.``, so
    ``terminate_outcome_others`` records nothing (the cagui code-10 rule).
    """
    row = guiact_use.episode_to_example(
        "ep",
        [
            _guiact_phone_tap("uid_episode_ep_step_00"),
            _guiact_phone_answer("uid_episode_ep_step_01", "task complete"),
        ],
        config=guiact_use.SUBSETS["smartphone"],
    )

    _assert_structural_done_row(row)
    assert "terminate_status" not in row["metadata"]["others"]
    validate_canonical_rows([row], "guiact-smartphone-complete")


def test_guiact_smartphone_canned_impossible_marker_records_failure(guiact_env):
    """``"task impossible"`` is a self-reported FAILURE, not an ordinary success.

    Published as the final turn, such an episode's only trace of impossibility
    was that visible string, so it read as a success. It routes exactly as
    cagui's code 11 and guiodyssey's ``INCOMPLETE`` do: ``Done.`` final plus
    ``others.terminate_status="failure"``.
    """
    row = guiact_use.episode_to_example(
        "ep",
        [
            _guiact_phone_tap("uid_episode_ep_step_00"),
            _guiact_phone_answer("uid_episode_ep_step_01", "task impossible"),
        ],
        config=guiact_use.SUBSETS["smartphone"],
    )

    _assert_structural_done_row(row)
    assert row["metadata"]["others"]["terminate_status"] == "failure"
    assert row["metadata"]["extra_tool_schemas"] == []
    validate_canonical_rows([row], "guiact-smartphone-impossible")


def test_guiact_web_answer_reading_like_a_marker_is_still_an_answer(guiact_env):
    """The marker vocabulary is scoped to the subset that declares it.

    web-multi is ``TERMINAL_ANSWER_AUTHORED_ONLY``, so an authored answer whose
    words happen to read "task complete" is published verbatim. A global string
    test would demote it -- which is why the distinction is a declared property
    of the subset rather than a match anywhere on the path.
    """
    row = guiact_use.episode_to_example(
        "ep",
        [
            _guiact_click_step("uid_record_ep_step_00", "<point>0.1, 0.2</point>"),
            _guiact_answer_step("uid_record_ep_step_01", "task complete"),
        ],
        config=guiact_use.SUBSETS["web_multi"],
    )

    assert row["messages"][-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "task complete"}],
    }
    assert "terminate_status" not in row["metadata"]["others"]
    validate_canonical_rows([row], "guiact-web-marker-wording")


def test_guiact_smartphone_lone_canned_marker_publishes_nothing(guiact_env):
    """A single-step episode whose only step is the marker has no content.

    All 10 ``"task impossible"`` episodes in the corpus are single-step, plus 28
    ``"task complete"`` ones. With the marker no longer standing in as an answer
    there is no action and no answer left, so the episode is skipped rather than
    published as a goal, a screenshot and a bare ``Done.``.
    """
    with pytest.raises(guiact_utils.SkipTrajectoryError, match="nothing publishable"):
        guiact_use.episode_to_example(
            "ep",
            [_guiact_phone_answer("uid_episode_ep_step_00", "task impossible")],
            config=guiact_use.SUBSETS["smartphone"],
        )


def test_guiact_terminal_answer_beside_executable_actions_still_publishes(guiact_env):
    """``select_text+copy+answer`` is the single most common terminal shape.

    Skipping every terminal step that is not a LONE ``answer`` dropped 45.7% of
    web-multi train and 54.9% of test -- and those episodes carry an answer. The
    executable action and answer text now share the final assistant turn.
    """
    row = guiact_use.episode_to_example(
        "ep",
        [
            _guiact_click_step("uid_record_ep_step_00", "<point>0.1, 0.2</point>"),
            _guiact_click_step("uid_record_ep_step_01", "<point>0.3, 0.4</point>"),
            _guiact_answer_step(
                "uid_record_ep_step_02",
                "Ada Lovelace",
                before=[{"name": "copy"}],
            ),
        ],
        config=guiact_use.SUBSETS["web_multi"],
    )

    assert len(row["images"]) == 3
    assert [m["role"] for m in row["messages"]] == [
        "user", "assistant", "tool", "assistant", "tool", "assistant",
    ]
    assert row["messages"][-1]["content"] == [{"type": "text", "text": "Ada Lovelace"}]
    assert tool_call_name(row["messages"][-1]["tool_calls"][0]) == "computer"
    validate_canonical_rows([row], "guiact-answer-with-actions")


def test_guiact_terminal_step_without_answer_still_publishes(guiact_env):
    """No answer and no terminator still preserves the terminal action label."""
    row = guiact_use.episode_to_example(
        "ep",
        [
            _guiact_click_step("uid_record_ep_step_00", "<point>0.1, 0.2</point>"),
            _guiact_click_step("uid_record_ep_step_01", "<point>0.3, 0.4</point>"),
        ],
        config=guiact_use.SUBSETS["web_multi"],
    )

    assert len(row["images"]) == 2
    assert [m["role"] for m in row["messages"]] == ["user", "assistant", "tool", "assistant"]
    _assert_final_action_row(row)


def test_guiact_skips_nonterminal_answer_it_cannot_pair(guiact_env):
    """A mid-trajectory ``answer`` is a standalone ``response`` with no result.

    A screenshot answers at most one call and the action-batch call beside it takes
    it, so such a row leaves the ``response`` unpaired -- 7 published web-multi
    train rows are invalid for exactly this reason.
    """
    with pytest.raises(guiact_utils.SkipTrajectoryError, match="non-terminal answer"):
        guiact_use.episode_to_example(
            "ep",
            [
                _guiact_answer_step(
                    "uid_record_ep_step_00",
                    "mid",
                    before=[{"name": "click", "point": {"related": "<point>0.1, 0.2</point>"}}],
                ),
                _guiact_click_step("uid_record_ep_step_01", "<point>0.3, 0.4</point>"),
                _guiact_answer_step("uid_record_ep_step_02", "Ada Lovelace"),
            ],
            config=guiact_use.SUBSETS["web_multi"],
        )


def test_guiact_single_step_episode_publishes_final_action(guiact_env):
    """A lone executable action is a valid EOF target."""
    row = guiact_use.episode_to_example(
        "ep",
        [_guiact_click_step("uid_record_ep_step_00", "<point>0.1, 0.2</point>")],
        config=guiact_use.SUBSETS["web_multi"],
    )

    assert [m["role"] for m in row["messages"]] == ["user", "assistant"]
    _assert_final_action_row(row)


def test_guiact_single_step_web_answer_publishes_final_text(guiact_env):
    """A terminal answer is publishable text even without an executable step."""
    row = guiact_use.episode_to_example(
        "ep",
        [_guiact_answer_step("uid_record_ep_step_00", "Ada Lovelace")],
        config=guiact_use.SUBSETS["web_multi"],
    )

    assert row["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "who wrote it?"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Ada Lovelace"}],
        },
    ]
    validate_canonical_rows([row], "guiact-single-step-answer")


def test_guiact_web_batches_wrapped_helpers_without_nested_computer():
    calls = guiact_utils.convert_web_actions(
        [
            {"name": "click", "point": {"related": "<point>0.1, 0.2</point>"}},
            {
                "name": "input",
                "point": {"related": "<point>0.3, 0.4</point>"},
                "text": "hello",
            },
        ],
        is_terminal=False,
    )

    assert [tool_call_name(call) for call in calls] == ["computer"]
    actions = tool_call_arguments(calls[0])["actions"]
    assert [action["action"] for action in actions] == ["click", "click", "type"]
    assert all(action["action"] != "computer" for action in actions)


def test_guiact_rejects_related_oob_before_normalization_clamp() -> None:
    with pytest.raises(guiact_utils.SkipTrajectoryError, match="outside \\[0, 1\\]"):
        guiact_utils.convert_web_actions(
            [{"name": "click", "point": {"related": "<point>1.2, 0.5</point>"}}],
            is_terminal=False,
        )
    with pytest.raises(guiact_utils.SkipTrajectoryError, match="outside \\[0, 1\\]"):
        guiact_utils.convert_mobile_action(
            {"name": "tap", "point": {"related": "<point>1.2, 0.5</point>"}},
            is_terminal=False,
        )

    assert guiact_utils.rel_to_1000(1 + 1e-7, -1e-7) == [1000, 0]


def test_guiact_only_removes_synthesized_input_focus_duplicate() -> None:
    click = {"name": "click", "point": {"related": "<point>0.2, 0.3</point>"}}
    input_action = {
        "name": "input",
        "point": {"related": "<point>0.2, 0.3</point>"},
        "text": "hello",
    }
    assert [action["action"] for action in _actions(
        guiact_utils.convert_web_actions([click, input_action], is_terminal=False)
    )] == ["click", "type"]
    assert [action["action"] for action in _actions(
        guiact_utils.convert_web_actions([click, click], is_terminal=False)
    )] == ["click", "click"]


def test_guiact_uses_absolute_scroll_sign_and_rejects_true_noop() -> None:
    up = {
        "name": "scroll",
        "scroll": {
            "related": {"down": "-0.00", "right": "0.00"},
            "absolute": {"down": -1, "right": 0},
        },
    }
    assert _actions(guiact_utils.convert_web_actions([up], is_terminal=False)) == [
        {"action": "scroll", "direction": "up", "amount": 1}
    ]
    noop = {
        "name": "scroll",
        "scroll": {
            "related": {"down": "0.00", "right": "0.00"},
            "absolute": {"down": 0, "right": 0},
        },
    }
    with pytest.raises(guiact_utils.SkipTrajectoryError, match="zero-distance"):
        guiact_utils.convert_web_actions([noop], is_terminal=False)


def test_guiact_rejects_unexecutable_select_instead_of_faking_click() -> None:
    with pytest.raises(guiact_utils.SkipTrajectoryError, match="no executable"):
        guiact_utils.convert_web_actions(
            [{"name": "select", "text": "French", "point": {"related": "<point>0.2, 0.3</point>"}}],
            is_terminal=False,
        )


def test_guiact_rejects_whole_trajectory_on_logical_history_gap() -> None:
    data = [
        {"uid": "uid_record_1_step_0", "question": "q", "actions_history": ""},
        {
            "uid": "uid_record_1_step_2",
            "question": "q",
            "actions_history": "step 0: hover\nstep 1: click",
        },
    ]
    episodes = guiact_use._group_episodes(data, guiact_use.SUBSETS["web_multi"]["uid_re"])
    assert list(episodes) == ["1"]
    with pytest.raises(guiact_utils.SkipTrajectoryError, match="missing logical action"):
        guiact_use.episode_to_example(
            "1", episodes["1"], config=guiact_use.SUBSETS["web_multi"]
        )

    prefix_only = [{
        "uid": "uid_record_2_step_1",
        "question": "q",
        "actions_history": "step 0: click",
    }]
    episodes = guiact_use._group_episodes(
        prefix_only, guiact_use.SUBSETS["web_multi"]["uid_re"]
    )
    with pytest.raises(guiact_utils.SkipTrajectoryError, match="does not start at step 0"):
        guiact_use.episode_to_example(
            "2", episodes["2"], config=guiact_use.SUBSETS["web_multi"]
        )


def test_guiact_duplicate_source_step_fails_instead_of_silent_dedupe() -> None:
    row = {"uid": "uid_record_1_step_0", "question": "q", "actions_history": ""}
    with pytest.raises(ValueError, match="Duplicate source step"):
        guiact_use._group_episodes(
            [row, dict(row)], guiact_use.SUBSETS["web_multi"]["uid_re"]
        )
