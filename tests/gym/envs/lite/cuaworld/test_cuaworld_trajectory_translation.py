"""CUAWorld tests split from _cuaworld_support.py: trajectory translation."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_id
from lite.core.tools.results import LiteToolResult
from lite.gym.envs.lite.cuaworld.src import software
from lite.gym.envs.lite.cuaworld.src.adapter import (
    _load_trajectory,
    create_episode_dir,
    initialize_episode,
    record_episode_step,
    remove_episode_dir,
    write_episode_frame,
)
from lite.gym.types import LiteEnvStepResult
from tests.gym.envs.lite._cuaworld_support import _GOOD, _png_bytes


def test_a_rejected_slot_is_never_reported_to_the_reward_evaluator():
    """A rejected action is one the agent never took; scoring it is a reward leak.

    Ingress FORWARDS a rejected child so the env can answer it per slot and
    still frame it (R2a + R3). ``_accepted_actions_for_final_eval`` feeds the
    upstream evaluator, which must see only what actually reached the guest --
    so it has to drop the very actions R3 keeps. The two requirements pull in
    opposite directions, which is exactly why this needs pinning.
    """
    from lite.core.metadata import LiteCUAMetadata

    kept = software._accepted_actions_for_final_eval(
        [make_tool_call(
            "computer",
            {"actions": [
                {"action": "click", "coordinate": [10, 10]},
                {"action": "bogus"},
            ]},
            call_id="c1",
        )],
        LiteCUAMetadata(),
    )

    assert [a.get("name") for a in kept] == ["click"]


def test_lite_actions_translate_to_upstream_trajectory_shape():
    actions = [
        {
            "name": "click",
            "arguments": {"coordinate": [500, 250], "button": "left"},
        },
        {
            "name": "type",
            "arguments": {"text": "hello"},
        },
        {
            "name": "key",
            "arguments": {"keys": ["ctrl", "s"]},
        },
        {
            "name": "terminate",
            "arguments": {},
        },
    ]

    assert software._upstream_trajectory_actions(actions, "1920x1080") == [
        {"mouse": {"left_click": [960, 270]}},
        {"keyboard": {"text": "hello"}},
        {"keyboard": {"keys": ["ctrl", "s"]}},
    ]


def test_trajectory_translation_projects_plus_glyph_to_upstream_key_name():
    actions = [
        {"name": "key", "arguments": {"keys": ["ctrl", "+"]}},
        {"name": "key_down", "arguments": {"keys": ["ctrl", "+"]}},
        {"name": "key_up", "arguments": {"keys": ["ctrl", "+"]}},
        {"name": "hold_key", "arguments": {"keys": ["ctrl", "+"], "duration": 0.5}},
    ]

    assert software._upstream_trajectory_actions(actions, "1000x1000") == [
        {"keyboard": {"keys": ["ctrl", "plus"]}},
        {"keyboard": {"keys_down": ["ctrl", "plus"]}},
        {"keyboard": {"keys_up": ["ctrl", "plus"]}},
        {"keyboard": {"keys_down": ["ctrl", "plus"]}},
        {"action": "wait", "time": 0.5},
        {"keyboard": {"keys_up": ["ctrl", "plus"]}},
    ]


def test_trajectory_translation_stops_at_termination():
    actions = [
        {
            "name": "click",
            "arguments": {"coordinate": [500, 500]},
        },
        {"name": "terminate", "arguments": {}},
        {
            "name": "click",
            "arguments": {"coordinate": [250, 250]},
        },
    ]

    assert software._upstream_trajectory_actions(actions, "1000x1000") == [
        {"mouse": {"left_click": [500, 500]}}
    ]


def test_trajectory_translation_matches_click_and_horizontal_scroll_execution():
    actions = [
        {
            "name": "click",
            "arguments": {
                "coordinate": [500, 500],
                "button": "right",
                "clicks": 3,
            },
        },
        {
            "name": "scroll",
            "arguments": {"direction": "left", "amount": 4},
        },
    ]

    # Three RIGHT clicks, not a left double-click plus a left single: the upstream
    # vocabulary's ``double_click`` is left-only, and ``multi_click`` executes
    # this as three right clicks. Translation and execution must agree.
    assert software._upstream_trajectory_actions(actions, "1000x1000") == [
        {"mouse": {"right_click": [500, 500]}},
        {"mouse": {"right_click": [500, 500]}},
        {"mouse": {"right_click": [500, 500]}},
        {"mouse": {"horizontal_scroll": -4}},
    ]


def test_trajectory_translation_records_middle_click_coordinate():
    actions = [{
        "name": "click",
        "arguments": {
            "coordinate": [250, 750],
            "button": "middle",
        },
    }]

    assert software._upstream_trajectory_actions(actions, "1000x1000") == [
        {"mouse": {"middle_click": [250, 750]}}
    ]


@pytest.mark.asyncio
async def test_step_keeps_canonical_batch_call_for_parent_env(
    tmp_path,
    monkeypatch,
):
    from lite.gym.sandbox.base import SandboxBaseEnv
    from lite.gym.sandbox.types import SandboxTaskConfig

    seen = {}

    async def step_base(env, actions):
        seen["actions"] = actions
        seen["pending_actions"] = env._computer._cuaworld_pending_actions
        return LiteEnvStepResult(
            reward=0.0,
            terminated=False,
            truncated=False,
            info={},
            results=[
                LiteToolResult(tool_call_id=tool_call_id(action), images=[_png_bytes()])
                for action in actions
                if tool_call_id(action)
            ],
        )

    monkeypatch.setattr(SandboxBaseEnv, "step", step_base)
    env_cls = software._make_env_class(
        "lite.cuaworld.test",
        {"image": "test", "display": "1000x1000"},
    )
    env = object.__new__(env_cls)
    env._cuaworld_episode_dir = tmp_path
    env._cuaworld_finalized = False
    env._computer = SimpleNamespace()
    env._task = SandboxTaskConfig("test", "test", {})
    env._debug = False
    env._step_count = 0
    env._cuaworld_timeout_sec = None
    env._cuaworld_episode_started_at = None
    env._evaluate_final_fn = None

    actions = [make_tool_call(
        "computer",
        {
            "actions": [{"action": "click", "coordinate": [500, 500]}],
        },
        call_id="call_0",
    )]

    await env.step(actions)

    assert seen["actions"] == actions
    assert seen["pending_actions"] == [{"mouse": {"left_click": [500, 500]}}]


@pytest.mark.asyncio
async def test_step_records_standalone_canonical_call_for_upstream_trajectory(
    tmp_path,
    monkeypatch,
):
    from lite.gym.sandbox.base import SandboxBaseEnv
    from lite.gym.sandbox.types import SandboxTaskConfig

    seen = {}

    async def step_base(env, actions):
        seen["actions"] = actions
        seen["pending_actions"] = env._computer._cuaworld_pending_actions
        return LiteEnvStepResult(
            reward=0.0,
            terminated=False,
            truncated=False,
            info={},
            results=[
                LiteToolResult(tool_call_id=tool_call_id(action), images=[_png_bytes()])
                for action in actions
                if tool_call_id(action)
            ],
        )

    monkeypatch.setattr(SandboxBaseEnv, "step", step_base)
    env_cls = software._make_env_class(
        "lite.cuaworld.test",
        {"image": "test", "display": "1000x1000"},
    )
    env = object.__new__(env_cls)
    env._cuaworld_episode_dir = tmp_path
    env._cuaworld_finalized = False
    env._computer = SimpleNamespace()
    env._task = SandboxTaskConfig("test", "test", {})
    env._debug = False
    env._step_count = 0
    env._cuaworld_timeout_sec = None
    env._cuaworld_episode_started_at = None
    env._evaluate_final_fn = None

    actions = [
        make_tool_call(
            "click",
            {"coordinate": [500, 500]},
            call_id="call_click",
        )
    ]

    await env.step(actions)

    assert seen["actions"] == actions
    assert seen["pending_actions"] == [{"mouse": {"left_click": [500, 500]}}]


@pytest.mark.asyncio
async def test_step_trajectory_projection_respects_valid_actions(
    tmp_path,
    monkeypatch,
):
    from lite.gym.sandbox.base import SandboxBaseEnv
    from lite.gym.sandbox.types import SandboxTaskConfig

    seen = {}

    async def step_base(env, actions):
        seen["actions"] = actions
        seen["pending_actions"] = env._computer._cuaworld_pending_actions
        return LiteEnvStepResult(
            reward=0.0,
            terminated=False,
            truncated=False,
            info={},
            results=[],
        )

    monkeypatch.setattr(SandboxBaseEnv, "step", step_base)
    env_cls = software._make_env_class(
        "lite.cuaworld.test",
        {"image": "test", "display": "1000x1000"},
    )
    env = object.__new__(env_cls)
    env._cuaworld_episode_dir = tmp_path
    env._cuaworld_finalized = False
    env._computer = SimpleNamespace()
    env._task = SandboxTaskConfig("test", "test", {}, platform="desktop")
    env._debug = False
    env._step_count = 0
    env._cuaworld_timeout_sec = None
    env._cuaworld_episode_started_at = None
    env._evaluate_final_fn = None
    env._valid_actions = ["type"]

    actions = [make_tool_call(
        "computer",
        {
            "actions": [{"action": "click", "coordinate": [500, 500]}],
        },
        call_id="call_0",
    )]

    await env.step(actions)

    assert seen["actions"] == actions
    assert seen["pending_actions"] == []


def test_episode_recorder_matches_upstream_trajectory_shape():
    episode_dir = create_episode_dir()
    try:
        initialize_episode(
            episode_dir,
            env_id="lite.cuaworld.contract",
            lite_env_id="lite.cuaworld.contract",
            task_id="trajectory-contract",
            container_name="fake",
            screenshot=_png_bytes(),
        )
        record_episode_step(episode_dir, index=0, actions=[{"action": "first"}])
        write_episode_frame(
            episode_dir,
            0,
            _png_bytes(),
        )
        record_episode_step(episode_dir, index=1, actions=[{"action": "second"}])
        write_episode_frame(
            episode_dir,
            1,
            _png_bytes(),
        )
        (episode_dir / "post_verification.png").write_bytes(_png_bytes())
        (episode_dir / "final.png").write_bytes(_png_bytes())

        trajectory = _load_trajectory(episode_dir)
        assert [event["event"] for event in trajectory["steps"]] == [
            "reset",
            "session",
            "step",
            "step",
        ]
        assert len(trajectory["frames"]) == 2
        assert sorted(trajectory["step_frames"]) == [0, 1]
        assert trajectory["first_frame"] == trajectory["frames"][0]
        assert trajectory["last_frame"] == trajectory["frames"][-1]
        assert Path(trajectory["final_screenshot"]).is_file()
        assert Path(trajectory["post_verification_screenshot"]).is_file()
    finally:
        remove_episode_dir(episode_dir)


@pytest.mark.asyncio
async def test_terminal_evaluator_records_the_terminal_step_frame(
    tmp_path, monkeypatch
):
    episode_dir = tmp_path / "episode"
    episode_dir.mkdir()
    seen = {}

    async def verify(*args, **kwargs):
        seen["episode_dir"] = kwargs["episode_dir"]
        return 1.0, {"passed": True}

    monkeypatch.setattr(software, "run_cuaworld_verify", verify)
    evaluate = software._make_eval_fn(
        tmp_path,
        "verifier.py::verify",
        {},
        _GOOD,
        "lite.cuaworld.test",
        "test_env",
    )
    computer = SimpleNamespace(
        interface=SimpleNamespace(screenshot=lambda: asyncio.sleep(0, result=_png_bytes())),
        _cuaworld_episode_dir=episode_dir,
        _cuaworld_pending_step_index=2,
        _cuaworld_pending_frame_index=2,
        _cuaworld_pending_actions=[{"keyboard": {"text": "done"}}],
    )

    reward, info = await evaluate(SimpleNamespace(), computer)

    assert (reward, info) == (1.0, {"passed": True})
    assert seen["episode_dir"] == episode_dir
    assert (episode_dir / "frame_00002.png").read_bytes() == _png_bytes()
    step = json.loads((episode_dir / "traj.jsonl").read_text().splitlines()[-1])
    assert step["idx"] == 2
