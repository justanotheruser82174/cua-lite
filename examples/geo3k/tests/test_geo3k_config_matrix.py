from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import MagicMock

import pytest
import yaml

from examples.geo3k.env import ENV_ID, Geo3KEnv
from lite.agents.bootstrap import register_all
from lite.agents.core.agent.base import AgentRegistry, BaseAgent
from lite.agents.factory import AGENTS, LOCAL_AGENTS, make
from lite.core.metadata import LiteGenericMetadata
from lite.core.tools.action_space import (
    LiteDesktopActionSet,
    LiteMobileActionSet,
    LitePointActionSet,
)
from lite.utils.registry import compose_key

register_all()

_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_ROOT = _ROOT / "examples" / "geo3k" / "configs"

_EXPECTED_CONFIGS = (
    "examples/geo3k/configs/qwen3_5/geo3k.mt.yaml",
    "examples/geo3k/configs/qwen3_5/geo3k.yaml",
    "examples/geo3k/configs/qwen3_vl/geo3k.mt.yaml",
    "examples/geo3k/configs/qwen3_vl/geo3k.yaml",
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


def _build_agent(row: ConfigRow, metadata: LiteGenericMetadata) -> BaseAgent:
    model_id = _model_for(row.agent_id)
    kwargs = {}
    if model_id in LOCAL_AGENTS:
        kwargs = {"processor": MagicMock(), "generate_fn": lambda *a, **k: None}
    env = SimpleNamespace(metadata=metadata)
    return make(model_id, env=env, agent_id=row.agent_id, **kwargs, **row.agent_kwargs)


def test_geo3k_config_matrix_enumerates_every_example_yaml() -> None:
    assert [row.rel for row in ROWS] == list(_EXPECTED_CONFIGS)
    assert all(row.env_id == ENV_ID for row in ROWS)


def test_geo3k_config_valid_actions_are_known_actions() -> None:
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
def test_geo3k_config_matrix_constructs_agent_from_env_kwargs(row: ConfigRow) -> None:
    env = Geo3KEnv(**row.env_kwargs)
    metadata = env._runtime_metadata()

    assert isinstance(metadata, LiteGenericMetadata)
    assert metadata.dims == ()
    key = compose_key(row.agent_id, *metadata.dims)
    assert AgentRegistry.contains(key)
    assert isinstance(_build_agent(row, metadata), BaseAgent)
