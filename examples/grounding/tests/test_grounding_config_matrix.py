from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import MagicMock

import pytest
import yaml

from lite.agents.bootstrap import register_all
from lite.agents.core.adapter.base import AgentAdapterRegistry
from lite.agents.core.agent.base import AgentRegistry, BaseAgent
from lite.agents.factory import AGENTS, LOCAL_AGENTS, make
from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import (
    LiteDesktopActionSet,
    LiteMobileActionSet,
    LitePointActionSet,
)
from lite.utils.registry import compose_key

register_all()
importlib.import_module("examples.grounding.adapter")

_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_ROOT = _ROOT / "examples" / "grounding" / "configs"

_EXPECTED_CONFIGS = (
    "examples/grounding/configs/qwen3_5/osworld_g.regionfocus.yaml",
    "examples/grounding/configs/qwen3_vl/osworld_g.regionfocus.yaml",
)

_ALLOWED_YAML_VALID_ACTIONS = (
    LiteDesktopActionSet.get_action_names()
    | LiteMobileActionSet.get_action_names()
    | LitePointActionSet.get_action_names()
)

_FAMILY_TO_MODEL: dict[str, str] = {}
for _model_id, _cfg in AGENTS.items():
    _FAMILY_TO_MODEL.setdefault(_cfg["agent_id"], _model_id)


class ConfigRow(NamedTuple):
    rel: str
    agent_id: str
    env_id: str | None
    agent_kwargs: dict
    env_kwargs: dict


def _rows() -> list[ConfigRow]:
    rows: list[ConfigRow] = []
    for path in sorted(_CONFIG_ROOT.rglob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        assert isinstance(data, dict), path
        rows.append(
            ConfigRow(
                rel=path.relative_to(_ROOT).as_posix(),
                agent_id=data["agent_id"],
                env_id=data.get("env_id"),
                agent_kwargs=data.get("agent_kwargs") or {},
                env_kwargs=data.get("env_kwargs") or {},
            )
        )
    return rows


ROWS = _rows()


def _model_for(agent_id: str) -> str:
    family = agent_id.split(".")[0]
    return _FAMILY_TO_MODEL[family]


def _grounding_metadata() -> LiteCUAMetadata:
    return LiteCUAMetadata(dims=("desktop", "grounding.point"))


def _build_agent(row: ConfigRow, metadata: LiteCUAMetadata) -> BaseAgent:
    model_id = _model_for(row.agent_id)
    kwargs = {}
    if model_id in LOCAL_AGENTS:
        kwargs = {"processor": MagicMock(), "generate_fn": lambda *a, **k: None}
    env = SimpleNamespace(metadata=metadata)
    return make(model_id, env=env, agent_id=row.agent_id, **kwargs, **row.agent_kwargs)


def test_grounding_config_matrix_enumerates_every_example_yaml() -> None:
    assert [row.rel for row in ROWS] == list(_EXPECTED_CONFIGS)
    assert all(row.env_id == "osworld_g" for row in ROWS)
    assert all(row.agent_kwargs == {"judge_initial_point": True} for row in ROWS)


def test_grounding_config_valid_actions_are_known_actions() -> None:
    offenders: list[str] = []
    for row in ROWS:
        value = row.env_kwargs.get("valid_actions")
        if value is None:
            continue
        assert isinstance(value, list), f"{row.rel}: valid_actions must be list|null"
        unknown = sorted(set(value) - _ALLOWED_YAML_VALID_ACTIONS)
        if unknown:
            offenders.append(f"{row.rel}: unknown GUI actions {unknown}")
    assert not offenders, "\n".join(offenders)


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row.rel)
def test_grounding_config_matrix_constructs_regionfocus_agent(row: ConfigRow) -> None:
    metadata = _grounding_metadata()

    key = compose_key(row.agent_id, *metadata.dims)
    assert AgentRegistry.contains(key)
    assert AgentAdapterRegistry.contains(key)
    assert isinstance(_build_agent(row, metadata), BaseAgent)
