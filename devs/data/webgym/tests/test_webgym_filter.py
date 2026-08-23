"""Tests for the webgym data-collection filter (``devs/data/webgym/filter.py``).

Focus: the turn-drop + chat-template alternation invariants, especially the
LEADING no-op-only turn carry-forward (dropping its paired user obs would
otherwise strand the goal / produce two consecutive user messages); plus the
footgun drops (F1 search-goto, F2 cross-domain goto, F3 loops, F6 search-start,
serp_only, search_flail, captcha).

This is a standalone dev script test (not a package module), so it loads the
parent ``filter.py`` module directly. Not in the default ``tests/`` suite.

The basename is component-qualified on purpose: ``devs/data/*/`` dirs are not
importable packages, so pytest names each test module by its bare basename --
two ``test_filter.py`` files would collide and ``pytest devs/data`` would
silently drop one of them. Keep every test basename under ``devs/`` unique.

Run:
    uv run pytest devs/data/webgym/tests/test_webgym_filter.py -v
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.calls import (
    make_tool_call,
    tool_call_arguments,
    tool_call_id,
    tool_call_name,
)
from lite.core.tools.extra_tools import LiteFinishToolSet
from lite.data.staging import coerce_messages, coerce_meta

_spec = importlib.util.spec_from_file_location(
    "webgym_filter", Path(__file__).resolve().parents[1] / "filter.py"
)
assert _spec is not None and _spec.loader is not None
web_filter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(web_filter)

DEFAULT_NOOP_ACTIONS = web_filter.DEFAULT_NOOP_ACTIONS
_registered_domain = web_filter._registered_domain
_traj_footguns = web_filter._traj_footguns
collapse_inline_reasoning = web_filter.collapse_inline_reasoning
has_oob_coordinate = web_filter.has_oob_coordinate
strip_noop_actions = web_filter.strip_noop_actions


def _call(name, **args):
    return make_tool_call(name, args, call_id=f"call_{name}")


def _computer(*actions: dict):
    return make_tool_call(
        "computer",
        {"actions": list(actions)},
        call_id="call_computer",
    )


def _init(host):
    """A first user message carrying webgym's ``Initial website:`` line (start-host source)."""
    return {"role": "user", "content": [{"type": "text", "text": f"do a thing\n\nInitial website: https://{host}/"}]}


def _asst(*calls):
    return {"role": "assistant", "tool_calls": list(calls)}

NOOP = frozenset(DEFAULT_NOOP_ACTIONS)  # {"screenshot", "wait"}


def _user(*parts: dict) -> dict:
    return {"role": "user", "content": list(parts)}


def _assistant(*names: str) -> dict:
    return {
        "role": "assistant",
        "tool_calls": [make_tool_call(n, {}, call_id=f"call_{n}") for n in names],
    }


def _browser_metadata(others: dict) -> dict:
    return LiteCUAMetadata(dims=("browser", "use"), others=others).to_dict()


def _img(tag: str) -> dict:
    return {"type": "image", "image": tag}


def _text(s: str) -> dict:
    return {"type": "text", "text": s}


def _meta(data: dict) -> dict:
    return {"type": "metadata", "data": data}


def _roles(msgs: list[dict]) -> list[str]:
    return [m["role"] for m in msgs]


def _assert_alternates(msgs: list[dict]) -> None:
    """No two consecutive same-role messages (chat-template contract)."""
    for a, b in zip(msgs, msgs[1:]):
        assert a["role"] != b["role"], f"non-alternating: {_roles(msgs)}"


def test_strips_noop_from_mixed_turn_keeps_others():
    """A turn with [screenshot, type] keeps only `type`; turn survives."""
    msgs = [
        _user(_text("goal")),
        _assistant("click"),
        _user(_img("i1")),
        _assistant("screenshot", "type"),
    ]
    out, n_stripped, n_dropped = strip_noop_actions(msgs, NOOP)
    assert n_stripped == 1 and n_dropped == 0
    assert _roles(out) == ["user", "assistant", "user", "assistant"]
    kept = [tool_call_name(tc) for tc in out[-1]["tool_calls"]]
    assert kept == ["type"]


def test_strips_noop_from_mixed_batched_computer_keeps_others():
    """An action-batch ``computer`` turn strips nested no-ops without dropping real GUI actions."""
    msgs = [
        _user(_text("goal")),
        _asst(_computer(
            {"action": "click", "coordinate": [10, 20]},
            {"action": "screenshot"},
            {"action": "type", "text": "hello"},
        )),
    ]
    out, n_stripped, n_dropped = strip_noop_actions(msgs, NOOP)
    assert n_stripped == 1 and n_dropped == 0
    assert _roles(out) == ["user", "assistant"]
    assert out[1]["tool_calls"] == [
        make_tool_call(
            "computer",
            {
                "actions": [
                    {"action": "click", "coordinate": [10, 20]},
                    {"action": "type", "text": "hello"},
                ],
            },
            call_id="call_computer",
        )
    ]


def test_strip_noop_rewrite_drops_stale_raw_response():
    msgs = [
        _user(_text("goal")),
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [
                _computer(
                    {"action": "click", "coordinate": [10, 20]},
                    {"action": "screenshot"},
                ),
            ],
            "raw_response": {
                "adapter_key": "qwen3_vl@desktop@use",
                "text": "old raw action",
            },
        },
    ]

    out, n_stripped, n_dropped = strip_noop_actions(msgs, NOOP)

    assert (n_stripped, n_dropped) == (1, 0)
    assert "raw_response" not in out[1]


@pytest.mark.parametrize("name", ["response", "terminate", "goto", "open_app", "bash"])
def test_standalone_extra_rejected_inside_batched_computer(name: str):
    msgs = [
        _user(_text("goal")),
        _asst(_computer({"action": name, "text": "x"})),
    ]
    with pytest.raises(ValueError, match="must not be nested"):
        strip_noop_actions(msgs, NOOP)


def test_drops_noop_only_midturn_with_preceding_user():
    """A mid-trajectory no-op-only turn drops with its paired user obs; the
    surviving sequence still alternates and keeps the real work."""
    msgs = [
        _user(_text("goal")),
        _assistant("click"),
        _user(_img("i1")),
        _assistant("screenshot"),  # no-op only → drop it + the i1 user obs
        _user(_img("i2")),
        _assistant("type"),
    ]
    out, n_stripped, n_dropped = strip_noop_actions(msgs, NOOP)
    assert n_stripped == 1 and n_dropped == 1
    _assert_alternates(out)
    assert _roles(out) == ["user", "assistant", "user", "assistant"]
    # The dropped user's image (i1) is gone; i2 survives.
    assert out[2]["content"] == [_img("i2")]


def test_drops_noop_only_batched_turn_with_preceding_user():
    """A computer batch containing only no-op actions drops like no-op-only turns."""
    msgs = [
        _user(_text("goal")),
        _assistant("click"),
        _user(_img("i1")),
        _asst(_computer({"action": "wait"}, {"action": "screenshot"})),
        _user(_img("i2")),
        _assistant("type"),
    ]
    out, n_stripped, n_dropped = strip_noop_actions(msgs, NOOP)
    assert n_stripped == 2 and n_dropped == 1
    _assert_alternates(out)
    assert _roles(out) == ["user", "assistant", "user", "assistant"]
    assert out[2]["content"] == [_img("i2")]


def test_leading_noop_only_carries_goal_forward():
    """Trajectory opens with a no-op-only turn: popping its paired user would
    strand the goal. The goal's NON-image parts must carry into the next user
    obs (alternation preserved), and the stale pre-no-op screenshot is dropped."""
    msgs = [
        _user(_text("goal"), _img("i0")),
        _assistant("screenshot"),  # FIRST turn, no-op only
        _user(_img("i1")),
        _assistant("click"),
    ]
    out, n_stripped, n_dropped = strip_noop_actions(msgs, NOOP)
    assert n_stripped == 1 and n_dropped == 1
    _assert_alternates(out)
    assert _roles(out) == ["user", "assistant"]
    # goal text merged into the surviving user obs; stale i0 dropped, i1 kept.
    assert out[0]["content"] == [_text("goal"), _img("i1")]
    assert tool_call_name(out[1]["tool_calls"][0]) == "click"


def test_leading_noop_preserves_reference_image_parts_not_stale_screenshot():
    metadata = _meta({"source": "reference-image"})
    msgs = [
        _user(
            {"type": "image", "index": 1},
            {"type": "image", "index": 0},
            _text("goal"),
            metadata,
        ),
        _assistant("screenshot"),
        _user({"type": "image", "index": 2}),
        _assistant("click"),
    ]

    out, n_stripped, n_dropped = strip_noop_actions(msgs, NOOP)

    assert n_stripped == 1 and n_dropped == 1
    assert out[0]["content"] == [
        {"type": "image", "index": 1},
        _text("goal"),
        metadata,
        {"type": "image", "index": 2},
    ]


def test_consecutive_leading_noops_preserve_goal_once():
    """Two consecutive leading no-op-only turns: the goal must survive exactly
    once (re-carried when the intermediate carrying-user is itself popped)."""
    msgs = [
        _user(_text("goal"), _img("i0")),
        _assistant("wait"),        # no-op only
        _user(_img("i1")),
        _assistant("screenshot"),  # no-op only
        _user(_img("i2")),
        _assistant("type"),
    ]
    out, n_stripped, n_dropped = strip_noop_actions(msgs, NOOP)
    assert n_stripped == 2 and n_dropped == 2
    _assert_alternates(out)
    assert _roles(out) == ["user", "assistant"]
    # exactly one goal text, merged into the final surviving obs (i2).
    assert out[0]["content"] == [_text("goal"), _img("i2")]
    assert tool_call_name(out[1]["tool_calls"][0]) == "type"


def test_no_noops_is_identity():
    msgs = [
        _user(_text("goal")),
        _assistant("click"),
        _user(_img("i1")),
        _assistant("type"),
    ]
    out, n_stripped, n_dropped = strip_noop_actions(msgs, NOOP)
    assert n_stripped == 0 and n_dropped == 0
    assert _roles(out) == ["user", "assistant", "user", "assistant"]


def test_strip_does_not_mutate_input():
    """``strip_noop_actions`` must not mutate its input in place — the invariant
    that lets the caller drop the redundant defensive copy. A turn that gets
    edited (noop stripped from a mixed [screenshot, type] turn) is the sharpest
    case for an accidental in-place write."""
    import copy

    msgs = [
        _user(_text("goal")),
        _assistant("click"),
        _user(_img("i1")),
        _assistant("screenshot", "type"),
    ]
    before = copy.deepcopy(msgs)
    strip_noop_actions(msgs, NOOP)
    assert msgs == before  # input list + its dicts are untouched


def test_strip_batched_noop_does_not_mutate_input():
    import copy

    msgs = [
        _user(_text("goal")),
        _asst(_computer({"action": "click", "coordinate": [1, 1]}, {"action": "screenshot"})),
    ]
    before = copy.deepcopy(msgs)
    strip_noop_actions(msgs, NOOP)
    assert msgs == before


# --- footgun detection (F1 search-engine goto, F3 loops) ---

def test_footgun_search_engine_goto_flagged():
    msgs = [_user(_text("g")), _asst(_call("goto", url="https://www.google.com/search?q=site%3Ax.org+foo"))]
    assert "search_goto" in _traj_footguns(msgs, drop_search_goto=True, drop_loops=True)


def test_footgun_nested_goto_in_computer_is_not_valid_nav():
    msgs = [_user(_text("g")), _asst(_computer({"action": "goto", "url": "https://www.google.com/search?q=x"}))]
    with pytest.raises(ValueError, match="must not be nested"):
        _traj_footguns(msgs, drop_search_goto=True, drop_loops=True)


def test_footgun_site_own_search_NOT_flagged():
    # a site's own /search?q= endpoint is legitimate grounded navigation, not a footgun
    msgs = [_user(_text("g")), _asst(_call("goto", url="https://wyso.org/search?q=kettering"))]
    assert _traj_footguns(msgs, drop_search_goto=True, drop_loops=True) == set()


def test_footgun_loop_needs_3_consecutive():
    # 2 consecutive identical (double-click / two same-position scrolls) is NOT a loop
    two = [_asst(_call("click", coordinate=[5, 5])), _asst(_call("click", coordinate=[5, 5]))]
    assert "loop" not in _traj_footguns(two, drop_search_goto=True, drop_loops=True)
    # 3 consecutive identical IS a stall
    three = two + [_asst(_call("click", coordinate=[5, 5]))]
    assert "loop" in _traj_footguns(three, drop_search_goto=True, drop_loops=True)
    # 3 scrolls at DIFFERENT positions (page-scan) is NOT a loop
    scan = [_asst(_call("scroll", coordinate=[1, 1])), _asst(_call("scroll", coordinate=[2, 2])),
            _asst(_call("scroll", coordinate=[3, 3]))]
    assert "loop" not in _traj_footguns(scan, drop_search_goto=True, drop_loops=True)


def test_footgun_loop_descends_into_batched_computer_actions():
    msgs = [
        _asst(_computer({"action": "click", "coordinate": [5, 5]})),
        _asst(_computer({"action": "click", "coordinate": [5, 5]})),
        _asst(_computer({"action": "click", "coordinate": [5, 5]})),
    ]
    assert "loop" in _traj_footguns(msgs, drop_search_goto=False, drop_loops=True)


def test_footgun_flags_off_returns_empty():
    msgs = [_asst(_call("goto", url="https://www.google.com/search?q=x"))]
    assert _traj_footguns(msgs, drop_search_goto=False, drop_loops=False) == set()


# --- search-flail: multi-engine bounce / >=3-search chain (non-reproducible for a 2B) ---

def test_footgun_search_flail_multi_engine_bounce():
    # google -> bing -> duckduckgo = 3 distinct engines (126677 shape) -> flail
    msgs = [_user(_text("g")),
            _asst(_call("goto", url="https://www.google.com/search?q=x")),
            _asst(_call("goto", url="https://www.bing.com/search?q=x")),
            _asst(_call("goto", url="https://duckduckgo.com/?q=x"))]
    assert "search_flail" in _traj_footguns(msgs, False, False, drop_search_flail=True)


def test_footgun_search_flail_long_single_engine_chain():
    # 3 searches on ONE engine = a research chain a small student can't reproduce -> flail
    msgs = [_user(_text("g"))] + [
        _asst(_call("goto", url=f"https://duckduckgo.com/?q=q{i}")) for i in range(3)]
    assert "search_flail" in _traj_footguns(msgs, False, False, drop_search_flail=True)


def test_footgun_single_search_NOT_flail():
    # one search-then-navigate is the learnable skill -> kept
    msgs = [_user(_text("g")), _asst(_call("goto", url="https://www.google.com/search?q=x"))]
    assert "search_flail" not in _traj_footguns(msgs, False, False, drop_search_flail=True)


def test_footgun_two_same_engine_searches_NOT_flail():
    # refine a query once on one engine = 2 searches, 1 engine -> kept
    msgs = [_user(_text("g")),
            _asst(_call("goto", url="https://duckduckgo.com/?q=a")),
            _asst(_call("goto", url="https://duckduckgo.com/?q=a+refined"))]
    assert "search_flail" not in _traj_footguns(msgs, False, False, drop_search_flail=True)


def test_footgun_site_own_search_NOT_flail():
    # a site's own /search?q= is not a search-engine -> 3 of them are not a flail
    msgs = [_user(_text("g"))] + [
        _asst(_call("goto", url=f"https://wyso.org/search?q=q{i}")) for i in range(3)]
    assert "search_flail" not in _traj_footguns(msgs, False, False, drop_search_flail=True)


# --- serp-only: searched but never clicked through (snippet-scrape) — model-size-independent ---

def test_footgun_serp_only_no_clickthrough_flagged():
    # search 3 engines, screenshot each, never click a result -> snippet-scrape (126677 shape)
    msgs = [_user(_text("g")),
            _asst(_call("goto", url="https://www.google.com/search?q=x")),
            _asst(_call("screenshot")),
            _asst(_call("goto", url="https://www.bing.com/search?q=x")),
            _asst(_call("response", answer="guess"))]
    assert "serp_only" in _traj_footguns(msgs, False, False, drop_serp_only=True)


def test_footgun_search_then_click_NOT_serp_only():
    # search once -> click a result (productive research) -> kept, even for the strictest reading
    msgs = [_user(_text("g")),
            _asst(_call("goto", url="https://duckduckgo.com/?q=x")),
            _asst(_call("click", coordinate=[100, 200])),
            _asst(_call("response", answer="from the page"))]
    assert "serp_only" not in _traj_footguns(msgs, False, False, drop_serp_only=True)


def test_footgun_search_then_goto_real_page_NOT_serp_only():
    # search -> goto a real (non-search) result page -> productive research -> kept
    msgs = [_user(_text("g")),
            _asst(_call("goto", url="https://www.bing.com/search?q=garmin")),
            _asst(_call("goto", url="https://www.garmin.com/")),
            _asst(_call("response", answer="from garmin.com"))]
    assert "serp_only" not in _traj_footguns(msgs, False, False, drop_serp_only=True)


def test_footgun_no_search_NOT_serp_only():
    # purely on-page (no search at all) is never serp_only
    msgs = [_user(_text("g")), _asst(_call("click", coordinate=[1, 1])), _asst(_call("response"))]
    assert "serp_only" not in _traj_footguns(msgs, False, False, drop_serp_only=True)


# --- captcha: trajectory hit a bot-verification wall (teacher text shows it) ---

def _asst_desc(text: str) -> dict:
    """Assistant turn whose content carries an action_description (where the block surfaces)."""
    return {"role": "assistant",
            "content": [{"type": "action_description", "text": text}],
            "tool_calls": [_call("click")]}


def test_footgun_captcha_flagged():
    msgs = [_user(_text("g")), _asst_desc("Click the human verification checkbox.")]
    assert "captcha" in _traj_footguns(msgs, False, False, drop_captcha=True)


def test_footgun_captcha_cloudflare_flagged():
    msgs = [
        _user(_text("g")),
        _asst_desc("The page shows a Cloudflare 'Just a moment...' challenge."),
    ]
    assert "captcha" in _traj_footguns(msgs, False, False, drop_captcha=True)


def test_footgun_captcha_off_by_default():
    msgs = [_user(_text("g")), _asst_desc("Click the human verification checkbox.")]
    assert _traj_footguns(msgs, drop_search_goto=False, drop_loops=False) == set()


def test_footgun_verify_recipe_NOT_captcha():
    # precise markers: "verify" in normal reasoning must NOT trip the captcha flag
    msgs = [_user(_text("g")), _asst_desc("Click the result, then verify the recipe has ratings.")]
    assert "captcha" not in _traj_footguns(msgs, False, False, drop_captcha=True)


# --- F2 cross-domain goto (eTLD+1) + F6 search-engine start website ---

def test_registered_domain_etld_plus_one():
    assert _registered_domain("podcasts.apple.com") == "apple.com"      # same-brand subdomain
    assert _registered_domain("www.amazon.de") == "amazon.de"
    assert _registered_domain("lesen.amazon.de") == "amazon.de"
    assert _registered_domain("cfd.direct") == "cfd.direct"
    assert _registered_domain("en.wikipedia.org") == "wikipedia.org"
    assert _registered_domain("amazon.co.uk") == "amazon.co.uk"          # multi-label suffix
    assert _registered_domain("cardinalscholar.bsu.edu") == "bsu.edu"


def test_footgun_xdomain_goto_flagged():
    # youtube.com task that goes to wikipedia.org → cross registered-domain (F2, d7/276212 shape)
    msgs = [_init("youtube.com"), _asst(_call("goto", url="https://en.wikipedia.org/wiki/X"))]
    assert "xdomain_goto" in _traj_footguns(msgs, False, False, drop_xdomain_goto=True)


def test_footgun_same_brand_subdomain_NOT_flagged():
    # apple.com → podcasts.apple.com (charts) is a same-brand subdomain → legit grounded nav
    msgs = [_init("apple.com"), _asst(_call("goto", url="https://podcasts.apple.com/us/charts"))]
    assert "xdomain_goto" not in _traj_footguns(msgs, False, False, drop_xdomain_goto=True)


def test_footgun_same_domain_deeplink_NOT_flagged():
    # amazon.de bestsellers constructed category-id URL stays within amazon.de (d6/45042 shape)
    msgs = [_init("lesen.amazon.de"),
            _asst(_call("goto", url="https://www.amazon.de/gp/bestsellers/digital-text/530886031"))]
    assert _traj_footguns(msgs, True, True, drop_xdomain_goto=True) == set()


def test_footgun_search_start_flagged():
    # task whose START website is a search engine (F6, d7/276760 shape)
    msgs = [_init("google.com"), _asst(_call("click", coordinate=[5, 5]))]
    assert "search_start" in _traj_footguns(msgs, False, False, drop_search_start=True)


def test_footgun_xdomain_and_search_off_by_default():
    # F2/F6 default OFF: a cross-domain goto from a search start must NOT be flagged unless asked
    msgs = [_init("google.com"), _asst(_call("goto", url="https://en.wikipedia.org/wiki/X"))]
    assert _traj_footguns(msgs, drop_search_goto=False, drop_loops=False) == set()


# --- unsubmitted: answer never submitted via the `response` tool ---

def test_footgun_unsubmitted_no_response_flagged():
    msgs = [_user(_text("g")), _asst(_call("click", coordinate=[1, 1]))]
    assert "unsubmitted" in _traj_footguns(msgs, False, False, drop_unsubmitted=True)


def test_footgun_unsubmitted_empty_response_flagged():
    # a `response` call with blank text is still no answer for the grader
    msgs = [_user(_text("g")), _asst(_call("response", text="   "))]
    assert "unsubmitted" in _traj_footguns(msgs, False, False, drop_unsubmitted=True)


def test_footgun_unsubmitted_text_final_done_is_not_response():
    msgs = [
        _user(_text("g")),
        {"role": "assistant", "content": [_text("Done.")], "tool_calls": []},
    ]
    assert "unsubmitted" in _traj_footguns(msgs, False, False, drop_unsubmitted=True)


def test_footgun_nested_response_in_computer_is_not_submitted():
    msgs = [_user(_text("g")), _asst(_computer({"action": "response", "text": "the answer"}))]
    with pytest.raises(ValueError, match="must not be nested"):
        _traj_footguns(msgs, False, False, drop_unsubmitted=True)


def test_footgun_submitted_response_NOT_flagged():
    msgs = [_user(_text("g")), _asst(_call("response", text="the answer"))]
    assert "unsubmitted" not in _traj_footguns(msgs, False, False, drop_unsubmitted=True)


def test_footgun_unsubmitted_off_by_default():
    msgs = [_user(_text("g")), _asst(_call("click", coordinate=[1, 1]))]
    assert _traj_footguns(msgs, drop_search_goto=False, drop_loops=False) == set()


# --- illposed_task: degenerate underspecified instruction template (>=3 "a specific <noun>") ---

def test_footgun_illposed_task_flagged():
    txt = ("Find papers related to a specific paper ID within a specific category "
           "with a specific keyword in the abstract for a specific date range")
    msgs = [_user(_text(txt)), _asst(_call("response", text="ok"))]
    assert "illposed_task" in _traj_footguns(msgs, False, False, drop_illposed=True)


def test_footgun_normal_task_NOT_illposed():
    txt = "List the seven countries where the NUS Executive MBA offers immersive segments."
    msgs = [_user(_text(txt)), _asst(_call("response", text="ok"))]
    assert "illposed_task" not in _traj_footguns(msgs, False, False, drop_illposed=True)


def test_footgun_two_slot_task_NOT_illposed():
    # a legit 2-slot task is below the >=3 threshold → kept (avoids false positives)
    txt = "Find a specific product with a specific feature on the page."
    msgs = [_user(_text(txt)), _asst(_call("response", text="ok"))]
    assert "illposed_task" not in _traj_footguns(msgs, False, False, drop_illposed=True)


def test_footgun_illposed_off_by_default():
    txt = "a specific keyword for a specific category in a specific date range"
    msgs = [_user(_text(txt)), _asst(_call("response", text="ok"))]
    assert _traj_footguns(msgs, drop_search_goto=False, drop_loops=False) == set()


# --- has_oob_coordinate: unconditional out-of-[0,1000] coordinate drop ---

def test_oob_coordinate_flagged():
    assert has_oob_coordinate([_asst(_call("click", coordinate=[1500, 5]))])   # x > 1000
    assert has_oob_coordinate([_asst(_call("click", coordinate=[5, -3]))])     # y < 0


def test_oob_coordinate_descends_into_batched_computer_actions():
    assert has_oob_coordinate([_asst(_computer({"action": "click", "coordinate": [1500, 5]}))])


def test_oob_start_coordinate_descends_into_batched_computer_actions():
    assert has_oob_coordinate([
        _asst(_computer({"action": "drag", "start_coordinate": [5, -3], "coordinate": [10, 10]}))
    ])


def test_oob_tuple_coordinate_uses_shared_range_owner():
    assert has_oob_coordinate([_asst(_call("click", coordinate=(1001, 5)))])


def test_in_range_coordinate_NOT_flagged():
    # the [0, 1000] endpoints are valid, not OOB
    assert not has_oob_coordinate([_asst(_call("click", coordinate=[0, 0])),
                                   _asst(_call("click", coordinate=[1000, 1000]))])


def test_no_coordinate_NOT_flagged():
    # back / response carry no coordinate → nothing to flag
    assert not has_oob_coordinate([_asst(_call("back")), _asst(_call("response", text="x"))])


def test_null_args_rejected_by_canonical_filter():
    msgs = [{"role": "assistant", "tool_calls": [
        {
            "id": "call_back",
            "type": "function",
            "function": {"name": "back", "arguments": None},
        },
    ]}]

    with pytest.raises(ValueError, match="arguments must be a dict"):
        has_oob_coordinate(msgs)


def test_bare_agent_wire_call_rejected_by_canonical_filter():
    msgs = [{"role": "assistant", "tool_calls": [
        {"name": "back", "arguments": {}},
    ]}]

    with pytest.raises(KeyError, match="function"):
        has_oob_coordinate(msgs)


def test_filter_refuses_stale_output_without_overwrite(tmp_path, monkeypatch):
    import json
    import sys

    import pandas as pd

    log_root = tmp_path / "logs"
    task_dir = log_root / "train" / "task_clean"
    task_dir.mkdir(parents=True)
    pd.DataFrame({
        "messages": [[_user(_text("goal"))]],
        "metadata": [json.dumps(_browser_metadata({"episode_return": 1.0}))],
    }).to_parquet(task_dir / "trajectory.parquet", index=False)

    out = tmp_path / "out"
    out.mkdir()
    stale = out / "old" / "trajectory.parquet"
    stale.parent.mkdir()
    stale.write_text("old")

    monkeypatch.setattr(sys, "argv", ["filter.py", "--log-root", str(log_root), "--out", str(out)])
    with pytest.raises(SystemExit, match="--overwrite"):
        web_filter.main()

    monkeypatch.setattr(sys, "argv", [
        "filter.py", "--log-root", str(log_root), "--out", str(out), "--overwrite",
    ])
    web_filter.main()
    assert not stale.exists()
    kept = {p.parent.name for p in out.rglob("trajectory.parquet")}
    assert kept == {"task_clean"}, kept


def test_process_file_accepts_json_string_messages_and_metadata(tmp_path):
    import json

    import pandas as pd

    from lite.data.staging import coerce_messages

    log_root = tmp_path / "logs"
    task_dir = log_root / "train" / "task_json"
    task_dir.mkdir(parents=True)
    messages = [
        _user(_text("goal")),
        _asst(_call("click", coordinate=[1, 2])),
    ]
    pd.DataFrame({
        "messages": [json.dumps(messages)],
        "metadata": [json.dumps(_browser_metadata({"episode_return": 1.0}))],
    }).to_parquet(task_dir / "trajectory.parquet", index=False)

    dst = tmp_path / "out" / "train" / "task_json" / "trajectory.parquet"
    *_, wrote = web_filter._process_file(
        task_dir / "trajectory.parquet",
        dst,
        noop=NOOP,
        output_root=tmp_path / "out",
        drop_failed=True,
        drop_loops=False,
        collapse_reasoning=False,
    )

    assert wrote
    row = pd.read_parquet(dst).iloc[0]
    assert coerce_messages(row["messages"]) == messages


def test_process_file_preserves_canonical_others_durable_metadata(tmp_path):
    import json

    import pandas as pd

    log_root = tmp_path / "logs"
    task_dir = log_root / "train" / "task_canonical"
    task_dir.mkdir(parents=True)
    messages = [
        _user(_text("goal")),
        _asst(_call("click", coordinate=[1, 2])),
    ]
    expected_others = {
        "task_id": "task_canonical",
        "env_id": "webgym",
        "episode_return": 1.0,
        "terminated": True,
        "truncated": False,
        "difficulty": "easy",
    }
    pd.DataFrame({
        "messages": [json.dumps(messages)],
        "metadata": [json.dumps(_browser_metadata(expected_others))],
    }).to_parquet(task_dir / "trajectory.parquet", index=False)

    dst = tmp_path / "out" / "train" / "task_canonical" / "trajectory.parquet"
    *_, wrote = web_filter._process_file(
        task_dir / "trajectory.parquet",
        dst,
        noop=NOOP,
        output_root=tmp_path / "out",
        drop_failed=True,
        collapse_reasoning=False,
    )

    assert wrote
    metadata = coerce_meta(pd.read_parquet(dst).iloc[0]["metadata"])
    assert metadata == _browser_metadata(expected_others)
    for key in ("task_id", "env_id", "episode_return", "terminated", "truncated"):
        assert key not in metadata


def test_filter_rebases_images_so_filtered_root_stages_without_raw_root(
    tmp_path,
    monkeypatch,
):
    import shutil
    import sys

    import pandas as pd
    from PIL import Image

    from lite.data.hf.stage import stage
    from lite.data.staging import write_partition
    from lite.utils.image import load_images

    raw_root = tmp_path / "raw"
    task_dir = raw_root / "train" / "task_clean"
    image = task_dir / "trajectory_images" / "000000.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (1, 1), color=(1, 2, 3)).save(image)
    write_partition([{
        "images": [str(image)],
        "messages": [
            _user({"type": "image", "index": 0}, _text("goal")),
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        ],
        "metadata": LiteCUAMetadata(
            dims=("browser", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={
                "task_id": "task_clean",
                "env_id": "webgym",
                "episode_return": 1.0,
                "terminated": True,
                "truncated": False,
            },
        ).to_dict(),
    }], task_dir / "trajectory.parquet")

    filtered = tmp_path / "filtered"
    monkeypatch.setattr(sys, "argv", [
        "filter.py", "--log-root", str(raw_root), "--out", str(filtered),
    ])
    web_filter.main()

    filtered_parquet = filtered / "train" / "task_clean" / "trajectory.parquet"
    row = pd.read_parquet(filtered_parquet).iloc[0]
    assert list(row["images"]) == ["train/task_clean/images/000000.png"]
    assert isinstance(row["messages"], str)
    idx = coerce_messages(row["messages"])[0]["content"][0]["index"]
    assert idx == 0
    assert isinstance(idx, int)
    assert (filtered_parquet.parent / "images" / "000000.png").read_bytes() == image.read_bytes()
    assert load_images(list(row["images"]), image_root=str(filtered))[0].size == (1, 1)

    shutil.rmtree(raw_root)
    name = "WebGymFilteredPortable"
    staged = tmp_path / "staged" / "cua-lite" / name
    stage([filtered], name=name, out_dir=staged, filter_expr=None)
    assert list(staged.rglob("*.parquet"))


def test_current_shape_success_fixture_filters_and_stages(
    tmp_path,
    monkeypatch,
):
    """Fixture-only smoke for the current WebGym collect row shape.

    This is not collect-side evidence: it proves the cleaner/stager keep a
    successful current row. Release evidence still needs a real raw rollout root
    with a surviving trajectory.
    """
    import sys

    import pandas as pd
    from PIL import Image

    from lite.data.hf.stage import stage
    from lite.data.staging import write_partition

    raw_root = tmp_path / "raw"
    task_dir = raw_root / "train" / "task_current"
    (task_dir / "images").mkdir(parents=True)
    images = []
    for i, color in enumerate([(1, 0, 0), (0, 2, 0)]):
        path = task_dir / "images" / f"{i:06d}.png"
        Image.new("RGB", (1, 1), color=color).save(path)
        images.append(path)

    write_partition([{
        "images": [str(path) for path in images],
        "messages": [
            _user(
                {"type": "image", "index": 0},
                _text(
                    "Find the visible price.\n\n"
                    "Initial website: https://example.com\n\n"
                    "When you have found the answer, submit it to finish the task."
                ),
            ),
            _rt_asst(
                "call_0000",
                {"action": "screenshot"},
                {"action": "click", "coordinate": [10, 20]},
            ),
            {
                "role": "tool",
                "tool_call_id": "call_0000",
                "content": [{"type": "image", "index": 1}],
            },
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": "Submit the answer."}],
                "tool_calls": [
                    make_tool_call("response", {"text": "$19.99"}, call_id="call_0001")
                ],
            },
        ],
        "metadata": LiteCUAMetadata(
            dims=("browser", "use"),
            extra_tool_schemas=[LiteFinishToolSet.get_tool_schema("response")],
            valid_actions=["click", "type", "key", "scroll", "wait"],
            others={
                "task_id": "task_current",
                "env_id": "webgym",
                "episode_return": 1.0,
                "terminated": True,
                "truncated": False,
            },
        ).to_dict(),
    }], task_dir / "trajectory.parquet")

    filtered = tmp_path / "filtered"
    monkeypatch.setattr(sys, "argv", [
        "filter.py",
        "--log-root",
        str(raw_root),
        "--out",
        str(filtered),
        "--drop-failed",
        "--drop-loops",
        "--drop-serp-only",
        "--drop-captcha",
        "--drop-unsubmitted",
        "--drop-illposed-task",
    ])
    web_filter.main()

    filtered_row = pd.read_parquet(
        filtered / "train" / "task_current" / "trajectory.parquet"
    ).iloc[0]
    filtered_messages = coerce_messages(filtered_row["messages"])
    computer_args = tool_call_arguments(filtered_messages[1]["tool_calls"][0])
    assert computer_args["actions"] == [{"action": "click", "coordinate": [10, 20]}]

    name = "WebGymCurrentShapeFixture"
    staged = tmp_path / "staged" / "cua-lite" / name
    stage(
        [filtered],
        name=name,
        out_dir=staged,
        filter_expr=None,
        config_names=["browser.use.train"],
        overwrite=True,
    )
    staged_files = list(staged.rglob("*.parquet"))
    assert [path.name for path in staged_files] == ["browser.use.train.parquet"]


# --- collapse_inline_reasoning: flatten GPT inline_reasoning to one line ---

def _asst_reasoning(text: str) -> dict:
    return {"role": "assistant",
            "content": [{"type": "inline_reasoning", "text": text},
                        {"type": "action_description", "text": "Click."}],
            "tool_calls": [_call("click", coordinate=[1, 1])]}


def test_collapse_inline_reasoning_flattens_to_one_line():
    msgs = [_user(_text("g")), _asst_reasoning("**Header**\n\nLine one.\nLine two.  extra")]
    out, n = collapse_inline_reasoning(msgs)
    assert n == 1
    assert out[1]["content"][0]["text"] == "**Header** Line one. Line two. extra"
    assert out[1]["content"][1]["text"] == "Click."        # action_description untouched
    assert "\n" in msgs[1]["content"][0]["text"]           # input not mutated


def test_collapse_inline_reasoning_rewrite_drops_stale_raw_response():
    msgs = [_user(_text("g")), _asst_reasoning("line 1\nline 2")]
    msgs[1]["raw_response"] = {
        "adapter_key": "qwen3_vl@desktop@use",
        "text": "old raw reasoning",
    }

    out, n = collapse_inline_reasoning(msgs)

    assert n == 1
    assert "raw_response" not in out[1]


def test_collapse_inline_reasoning_noop_on_clean_text():
    msgs = [_user(_text("g")), _asst_reasoning("already one line")]
    out, n = collapse_inline_reasoning(msgs)
    assert n == 0 and out[1]["content"][0]["text"] == "already one line"


# ---------------------------------------------------------------------------
# role:"tool" result layout — the observation for turn N+1 is the tool result
# FOLLOWING assistant turn N. These guard that branch; the preceding-observation
# fixtures above never enter it. NOTE: _assert_alternates is deliberately NOT used —
# a correct role:"tool" sequence is [user, assistant, tool, assistant, tool],
# which alternates by design but not by the user/assistant rule that helper encodes.
# ---------------------------------------------------------------------------


def _rt_asst(call_id: str, *actions: dict) -> dict:
    """An assistant turn issuing ONE batched GUI call with the given actions."""
    return {
        "role": "assistant",
        "tool_calls": [
            make_tool_call("computer", {"actions": list(actions)}, call_id=call_id)
        ],
    }


def _rt_result(call_id: str, tag: str = "obs") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": [_img(tag)]}


def test_role_tool_all_noop_turn_drops_its_own_result_not_the_observation():
    """The result FOLLOWS the turn, so the dropped turn takes its own result with it."""
    msgs = [
        _user(_text("goal")),
        _rt_asst("c1", {"action": "screenshot"}, {"action": "wait"}),
        _rt_result("c1"),
        _rt_asst("c2", {"action": "click", "coordinate": [3, 4]}),
        _rt_result("c2"),
    ]
    out, n_stripped, n_dropped = strip_noop_actions(msgs, NOOP)
    assert n_stripped == 2 and n_dropped == 1
    assert _roles(out) == ["user", "assistant", "tool"]
    # the SURVIVING turn's result is kept and the dropped turn's is gone
    assert [m["tool_call_id"] for m in out if m["role"] == "tool"] == ["c2"]
    # the goal is untouched — under this result layout it is not a paired observation
    assert any(p.get("text") == "goal" for p in out[0]["content"])


def test_role_tool_leading_noop_turn_keeps_the_goal():
    """Regression: the preceding-observation branch would strand the task here."""
    msgs = [
        _user(_text("find the price")),
        _rt_asst("c1", {"action": "wait"}),
        _rt_result("c1"),
        _rt_asst("c2", {"action": "type", "text": "shoes"}),
        _rt_result("c2"),
    ]
    out, _, n_dropped = strip_noop_actions(msgs, NOOP)
    assert n_dropped == 1
    assert _roles(out) == ["user", "assistant", "tool"]
    assert any(p.get("text") == "find the price" for p in out[0]["content"])


def test_role_tool_partial_batch_keeps_the_turn_and_its_result():
    msgs = [
        _user(_text("goal")),
        _rt_asst("c1", {"action": "screenshot"}, {"action": "click", "coordinate": [1, 2]}),
        _rt_result("c1"),
    ]
    out, n_stripped, n_dropped = strip_noop_actions(msgs, NOOP)
    assert n_stripped == 1 and n_dropped == 0
    assert _roles(out) == ["user", "assistant", "tool"]
    kept = tool_call_arguments(out[1]["tool_calls"][0])["actions"]
    assert [a["action"] for a in kept] == ["click"]


def test_role_tool_mixed_top_level_noop_drops_only_its_result():
    msgs = [
        _user(_text("goal")),
        {
            "role": "assistant",
            "tool_calls": [
                make_tool_call("screenshot", {}, call_id="c_noop"),
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                    call_id="c_real",
                ),
            ],
        },
        _rt_result("c_noop"),
        _rt_result("c_real"),
    ]
    out, n_stripped, n_dropped = strip_noop_actions(msgs, NOOP)
    assert n_stripped == 1 and n_dropped == 0
    live = {tool_call_id(tc) for m in out if m["role"] == "assistant" for tc in m["tool_calls"]}
    assert live == {"c_real"}
    assert [m["tool_call_id"] for m in out if m["role"] == "tool"] == ["c_real"]


def test_role_tool_noop_only_batch_with_response_keeps_response_and_result():
    msgs = [
        _user(_text("goal")),
        {
            "role": "assistant",
            "tool_calls": [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "screenshot"}]},
                    call_id="call_computer",
                ),
                make_tool_call("response", {"text": "ok"}, call_id="call_response"),
            ],
        },
        _rt_result("call_computer"),
        _rt_result("call_response"),
    ]

    out, n_stripped, n_dropped = strip_noop_actions(msgs, NOOP)

    assert (n_stripped, n_dropped) == (1, 0)
    assert [tool_call_id(tc) for m in out if m["role"] == "assistant" for tc in m["tool_calls"]] == [
        "call_response",
    ]
    assert [m["tool_call_id"] for m in out if m["role"] == "tool"] == ["call_response"]


def test_no_orphan_tool_result_survives_a_dropped_turn():
    """The defect this branch fixes: a role:"tool" whose assistant turn is gone."""
    msgs = [
        _user(_text("goal")),
        _rt_asst("c1", {"action": "click", "coordinate": [1, 1]}),
        _rt_result("c1"),
        _rt_asst("c2", {"action": "wait"}),
        _rt_result("c2"),
        _rt_asst("c3", {"action": "type", "text": "x"}),
        _rt_result("c3"),
    ]
    out, _, n_dropped = strip_noop_actions(msgs, NOOP)
    assert n_dropped == 1
    live = {tool_call_id(tc) for m in out if m["role"] == "assistant" for tc in m["tool_calls"]}
    orphans = [
        m["tool_call_id"]
        for m in out
        if m["role"] == "tool" and m["tool_call_id"] not in live
    ]
    assert orphans == [], f"orphaned tool results: {orphans}"


def test_result_layout_is_chosen_by_role_tool_presence_not_by_call_id():
    """Assistant call id is not the discriminator for result layout."""
    preceding_observation_with_ids = [
        _user(_text("goal")),
        _rt_asst("c1", {"action": "wait"}),  # has an id, but no role:"tool" anywhere
        _user(_img("i1")),
        _rt_asst("c2", {"action": "click", "coordinate": [1, 1]}),
    ]
    out, _, n_dropped = strip_noop_actions(preceding_observation_with_ids, NOOP)
    assert n_dropped == 1
    # Preceding-observation layout: pop the paired user obs and carry the goal forward.
    assert _roles(out) == ["user", "assistant"]
    assert any(p.get("text") == "goal" for p in out[0]["content"])
    _assert_alternates(out)


# =============================================================================
# Image compaction — dropping a TURN drops no PICTURE, so the filtered row would
# otherwise publish an image no message shows and leave a hole in the index
# sequence. compact_row_images (devs/data/utils.py) closes both, in one step.
# =============================================================================

def test_filter_compacts_the_orphan_image_a_dropped_noop_turn_leaves(tmp_path, monkeypatch):
    """End-to-end through ``web_filter.main()``: 4 turns, the MIDDLE one a no-op.

    The no-op turn and its paired result are dropped, so the picture that result
    showed is referenced by nothing. The written row must carry 3 images, not 4,
    with indices ``0..2`` — and each surviving reference must still resolve to
    the SAME picture (checked by pixel bytes, since the rebased file NAMES are
    positional and would look right either way).
    """
    import sys

    import pandas as pd
    from PIL import Image

    from lite.data.staging import write_partition

    raw_root = tmp_path / "raw"
    task_dir = raw_root / "train" / "task_noop"
    (task_dir / "trajectory_images").mkdir(parents=True)
    images = []
    for i, color in enumerate([(10, 0, 0), (20, 0, 0), (30, 0, 0), (40, 0, 0)]):
        path = task_dir / "trajectory_images" / f"{i:06d}.png"
        Image.new("RGB", (1, 1), color=color).save(path)
        images.append(path)

    write_partition([{
        "images": [str(p) for p in images],
        "messages": [
            _user({"type": "image", "index": 0}, _text("do a thing")),
            _rt_asst("c0", {"action": "click", "coordinate": [1, 2]}),
            {"role": "tool", "tool_call_id": "c0", "content": [{"type": "image", "index": 1}]},
            _rt_asst("c1", {"action": "screenshot"}),          # <-- the no-op turn
            {"role": "tool", "tool_call_id": "c1", "content": [{"type": "image", "index": 2}]},
            _rt_asst("c2", {"action": "click", "coordinate": [3, 4]}),
            {"role": "tool", "tool_call_id": "c2", "content": [{"type": "image", "index": 3}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        ],
        "metadata": LiteCUAMetadata(
            dims=("browser", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={
                "task_id": "task_noop",
                "env_id": "webgym",
                "episode_return": 1.0,
                "terminated": True,
                "truncated": False,
            },
        ).to_dict(),
    }], task_dir / "trajectory.parquet")

    filtered = tmp_path / "filtered"
    monkeypatch.setattr(sys, "argv", [
        "filter.py", "--log-root", str(raw_root), "--out", str(filtered),
    ])
    web_filter.main()

    filtered_parquet = filtered / "train" / "task_noop" / "trajectory.parquet"
    row = pd.read_parquet(filtered_parquet).iloc[0]
    msgs = coerce_messages(row["messages"])
    indices = [
        part["index"]
        for m in msgs
        for part in (m.get("content") or [])
        if part.get("type") == "image"
    ]

    assert len(list(row["images"])) == 3
    assert indices == [0, 1, 2]
    assert sorted(set(indices)) == list(range(len(list(row["images"]))))
    # By CONTENT: the survivors are the goal screen, the first result and the
    # last result — the no-op's screen is gone, not merely renumbered.
    written = [(filtered / rel).read_bytes() for rel in row["images"]]
    assert written == [images[0].read_bytes(), images[1].read_bytes(), images[3].read_bytes()]
    assert images[2].read_bytes() not in written
