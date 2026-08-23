"""WebGym browser surface tests shared by cross-agent browser flows.

Run:
    uv run pytest tests/gym/envs/webgym/test_webgym_crossagent.py -v
"""

from __future__ import annotations

from lite.core.tools.extra_tools import LiteBrowserNavToolSet
from lite.core.tools.schemas import tool_schema_name
from lite.gym.envs.webgym.main import WebGymEnv


def test_ns2_webgym_valid_actions_are_grounded_no_nav() -> None:
    """WebGym's default valid_actions are grounded GUI verbs only."""

    env = WebGymEnv(task={"task_id": "t", "difficulty": 2})
    assert not (set(env.metadata.valid_actions) & LiteBrowserNavToolSet.get_tool_names())
    assert {"response", "terminate"}.isdisjoint(env.metadata.valid_actions)
    assert env.metadata.extra_tool_schemas == []


def test_ns2_nav_extra_tools_single_source() -> None:
    """WebGym resolves browser navigation schemas from LiteBrowserNavToolSet."""

    env = WebGymEnv(
        task={"task_id": "t", "difficulty": 2},
        extra_tools=["goto", "back", "response"],
    )

    names = [tool_schema_name(t) for t in env.metadata.extra_tool_schemas]
    assert names == ["goto", "back", "response"]

