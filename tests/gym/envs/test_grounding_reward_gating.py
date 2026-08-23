"""What the grounding reward is allowed to score, and what must not zero it.

The rule these envs implement: **the reward scores the ANSWER.** It is 0.0 when
the answer is missing, ambiguous, or malformed -- and only then. An error on a
call that is not an answer (an unknown tool, a ``terminate`` the task never
advertised) is reported to the model as feedback but must leave the score alone.

Nothing distinguished these cases before: the gate was
``had_model_action_error or action_errors or unsupported_reasons``, so ANY
errored call zeroed a correct, unambiguous click. Both halves are pinned here
because both are silent -- a wrong reward looks exactly like a right one in
aggregate.

These fixtures construct the env classes directly and seed the screenshot bytes
instead of calling ``reset()``, so they need no ScreenSpot-Pro / OSWorld-G data.

This file stays shallow because it asserts the shared pure-grounding reward
contract across ``screenspot_pro`` and ``osworld_g``; each test name keeps the
env-specific half explicit.

Run:
    uv run pytest tests/gym/envs/test_grounding_reward_gating.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lite.core.tools import make_tool_call

# bbox [10, 10, 20, 20] on a 100x100 image -> normalised [0.1, 0.1, 0.2, 0.2].
# cua-lite points are [0, 1000], so 150 -> 0.15 (inside) and 900 -> 0.9 (outside).
_INSIDE = [150, 150]
_OUTSIDE = [900, 900]


def _screenspot_env():
    from lite.gym.envs.screenspot_pro.main import ScreenSpotProEnv

    env = ScreenSpotProEnv(
        annotation={
            "img_filename": "unused.png",
            "instruction": "click the target",
            "bbox": [10, 10, 20, 20],
            "img_size": [100, 100],
        },
        images_dir=Path("/tmp"),
    )
    env._screenshot = b"shot"
    return env


def _osworld_g_env(box_type: str = "bbox"):
    from lite.gym.envs.osworld_g.main import OSWorldGEnv

    annotation = {
        "instruction": "click the target",
        "image_path": "unused.png",
        "image_size": [100, 100],
        "box_type": box_type,
        "box_coordinates": [10, 10, 20, 20],
        "GUI_types": [],
    }
    env = OSWorldGEnv(
        annotation_original=annotation,
        annotation_refined=annotation,
        images_dir=Path("/tmp"),
        task_id="fixture",
    )
    env._screenshot = b"shot"
    return env


# ---------------------------------------------------------------------------
# Baseline: a lone correct / incorrect point still scores the obvious way.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_screenspot_single_point_scores_normally() -> None:
    inside = await _screenspot_env().step(
        [make_tool_call("point", {"coordinate": _INSIDE}, call_id="a")]
    )
    outside = await _screenspot_env().step(
        [make_tool_call("point", {"coordinate": _OUTSIDE}, call_id="a")]
    )
    assert inside.reward == 1.0
    assert outside.reward == 0.0


# ---------------------------------------------------------------------------
# Half A -- ambiguity. Two points is not an answer, even if one of them hits.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_screenspot_two_points_score_zero_even_when_the_first_hits() -> None:
    """Hedging must not pay.

    Taking the first point (the pre-refactor behaviour) lets a model emit
    several candidate clicks and rely on the grader to find the hit, which
    turns a single-shot grounding benchmark into a multiple-choice one.
    """
    result = await _screenspot_env().step([
        make_tool_call("point", {"coordinate": _INSIDE}, call_id="first"),
        make_tool_call("point", {"coordinate": _OUTSIDE}, call_id="second"),
    ])
    assert result.reward == 0.0


@pytest.mark.asyncio
async def test_osworld_g_two_points_score_zero_even_when_the_first_hits() -> None:
    result = await _osworld_g_env().step([
        make_tool_call("point", {"coordinate": _INSIDE}, call_id="first"),
        make_tool_call("point", {"coordinate": _OUTSIDE}, call_id="second"),
    ])
    assert result.reward == 0.0


@pytest.mark.asyncio
async def test_screenspot_no_point_scores_zero() -> None:
    """A model that only calls an unadvertised finish tool has not answered."""
    result = await _screenspot_env().step(
        [make_tool_call("terminate", {"status": "success"}, call_id="t")]
    )
    assert result.reward == 0.0


# ---------------------------------------------------------------------------
# Half B -- a NON-answer error must not zero a correct, unambiguous click.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_screenspot_inactive_finish_tool_does_not_zero_a_correct_point() -> None:
    """``terminate`` is not advertised here (``extra_tools`` defaults to []).

    A model that clicks correctly AND emits a stray ``terminate`` gets feedback
    about the terminate, but the click is still the single well-formed answer.
    Zeroing it grades a correct click as wrong because of an unrelated call.
    """
    env = _screenspot_env()
    result = await env.step([
        make_tool_call("point", {"coordinate": _INSIDE}, call_id="answer"),
        make_tool_call("terminate", {"status": "success"}, call_id="stray"),
    ])

    assert result.reward == 1.0, (
        "an unrelated rejected call must not zero a correct, unambiguous answer"
    )
    errors = {r.tool_call_id: r.error for r in result.results}
    assert "terminate is not available" in (errors["stray"] or ""), (
        "the stray call is still reported to the model"
    )


@pytest.mark.asyncio
async def test_screenspot_unknown_tool_does_not_zero_a_correct_point() -> None:
    env = _screenspot_env()
    result = await env.step([
        make_tool_call("point", {"coordinate": _INSIDE}, call_id="answer"),
        make_tool_call("not_a_real_tool", {}, call_id="bogus"),
    ])

    assert result.reward == 1.0
    errors = {r.tool_call_id: r.error for r in result.results}
    assert "unknown tool" in (errors["bogus"] or "")


@pytest.mark.asyncio
async def test_osworld_g_inactive_finish_tool_does_not_zero_a_correct_point() -> None:
    env = _osworld_g_env()
    result = await env.step([
        make_tool_call("point", {"coordinate": _INSIDE}, call_id="answer"),
        make_tool_call("terminate", {"status": "success"}, call_id="stray"),
    ])

    assert result.reward == 1.0
    errors = {r.tool_call_id: r.error for r in result.results}
    assert "terminate is not available" in (errors["stray"] or "")


@pytest.mark.asyncio
async def test_screenspot_non_answer_error_still_zeroes_an_INCORRECT_point() -> None:
    """The relaxation must not accidentally reward a miss."""
    result = await _screenspot_env().step([
        make_tool_call("point", {"coordinate": _OUTSIDE}, call_id="answer"),
        make_tool_call("terminate", {"status": "success"}, call_id="stray"),
    ])
    assert result.reward == 0.0


# ---------------------------------------------------------------------------
# ...but an error on the ANSWER itself still zeroes.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_screenspot_malformed_point_alongside_a_good_one_scores_zero() -> None:
    """Two point ATTEMPTS is still an ambiguous answer.

    The malformed one is dropped before ``_evaluate``, so without an explicit
    answer-error gate the surviving good point would look like a lone answer
    and score 1.0 -- paying a model for spraying malformed candidates.
    """
    result = await _screenspot_env().step([
        make_tool_call("point", {"coordinate": "abc"}, call_id="bad"),
        make_tool_call("point", {"coordinate": _INSIDE}, call_id="good"),
    ])
    assert result.reward == 0.0


@pytest.mark.asyncio
async def test_osworld_g_malformed_point_alongside_a_good_one_scores_zero() -> None:
    result = await _osworld_g_env().step([
        make_tool_call("point", {"coordinate": "abc"}, call_id="bad"),
        make_tool_call("point", {"coordinate": _INSIDE}, call_id="good"),
    ])
    assert result.reward == 0.0


# ---------------------------------------------------------------------------
# The crux: both cases are "a correct point plus one rejected call", and they
# must score DIFFERENTLY. Keep them adjacent -- the whole gate is this contrast.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stray_finish_tool_and_wrong_answer_wrapper_score_differently() -> None:
    """Same shape, opposite scores, because only one is an attempt to ANSWER.

    ``terminate`` is not an answer -- the point stands alone, so it scores.
    ``computer(actions=[{"action": "point", ...}])`` IS an answer, routed
    through a wrapper this task does not accept: the model made two answer
    attempts, so the surviving point is not a lone answer and must not score.
    """
    stray_finish = await _screenspot_env().step([
        make_tool_call("point", {"coordinate": _INSIDE}, call_id="answer"),
        make_tool_call("terminate", {"status": "success"}, call_id="stray"),
    ])
    wrong_wrapper = await _screenspot_env().step([
        make_tool_call("point", {"coordinate": _INSIDE}, call_id="answer"),
        make_tool_call(
            "computer",
            {"actions": [{"action": "point", "coordinate": _OUTSIDE}]},
            call_id="second_attempt",
        ),
    ])

    assert stray_finish.reward == 1.0, "a non-answer call must not poison the score"
    assert wrong_wrapper.reward == 0.0, "a second answer attempt must poison it"
