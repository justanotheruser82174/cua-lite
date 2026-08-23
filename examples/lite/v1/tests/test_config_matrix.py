from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import MagicMock

import pytest
import yaml

from lite.agents.bootstrap import register_all
from lite.agents.core.agent.base import AgentRegistry, BaseAgent
from lite.agents.factory import AGENTS, LOCAL_AGENTS, make
from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import (
    LiteDesktopActionSet,
    LiteMobileActionSet,
    LitePointActionSet,
)
from lite.core.tools.extra_tools import LiteBrowserNavToolSet, LiteFinishToolSet
from lite.core.tools.schemas import tool_schema_name
from lite.utils.registry import compose_key

register_all()

_ROOT = Path(__file__).resolve().parents[4]
_CONFIG_ROOT = _ROOT / "examples" / "lite" / "v1" / "configs"

_EXPECTED_CONFIGS = (
    "examples/lite/v1/configs/qwen3_5/default/desktop.use.yaml",
    "examples/lite/v1/configs/qwen3_5/default/webgym.yaml",
    "examples/lite/v1/configs/qwen3_5/reasoning/desktop.use.yaml",
    "examples/lite/v1/configs/qwen3_5/reasoning/webgym.yaml",
)
_ENVLESS_DESKTOP_CONFIGS = (
    "examples/lite/v1/configs/qwen3_5/default/desktop.use.yaml",
    "examples/lite/v1/configs/qwen3_5/reasoning/desktop.use.yaml",
)

_ALLOWED_YAML_VALID_ACTIONS = (
    LiteDesktopActionSet.get_action_names()
    | LiteMobileActionSet.get_action_names()
    | LitePointActionSet.get_action_names()
)

_BROWSER_EXTRA_TOOL_SCHEMAS = {
    tool_schema_name(schema): schema
    for schema in (LiteBrowserNavToolSet.get_tool_schemas() + LiteFinishToolSet.get_tool_schemas())
}

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


def _browser_extra_tool_schemas(names: list[str]) -> list[dict]:
    unknown = sorted(set(names) - set(_BROWSER_EXTRA_TOOL_SCHEMAS))
    assert not unknown, f"unknown browser extra_tools: {unknown}"
    return [_BROWSER_EXTRA_TOOL_SCHEMAS[name] for name in names]


def _metadata_for(row: ConfigRow) -> LiteCUAMetadata:
    if row.env_id == "webgym":
        extra_tools = row.env_kwargs.get("extra_tools") or []
        return LiteCUAMetadata(
            dims=("browser", "use"),
            extra_tool_schemas=_browser_extra_tool_schemas(extra_tools),
        )
    if row.env_id is None and row.rel in _ENVLESS_DESKTOP_CONFIGS:
        return LiteCUAMetadata(dims=("desktop", "use"))
    raise AssertionError(f"{row.rel}: no Lite v1 metadata owner for env_id={row.env_id!r}")


def _build_agent(row: ConfigRow, metadata: LiteCUAMetadata) -> BaseAgent:
    model_id = _model_for(row.agent_id)
    kwargs = {}
    if model_id in LOCAL_AGENTS:
        kwargs = {"processor": MagicMock(), "generate_fn": lambda *a, **k: None}
    agent_kwargs = dict(row.agent_kwargs)
    env = SimpleNamespace(metadata=metadata)
    return make(model_id, env=env, agent_id=row.agent_id, **kwargs, **agent_kwargs)


def test_lite_v1_config_matrix_enumerates_every_example_yaml() -> None:
    assert [row.rel for row in ROWS] == list(_EXPECTED_CONFIGS)
    assert [row.rel for row in ROWS if row.env_id is None] == list(_ENVLESS_DESKTOP_CONFIGS)


def test_lite_v1_config_valid_actions_are_known_actions() -> None:
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
def test_lite_v1_config_matrix_constructs_agent_for_owned_metadata(row: ConfigRow) -> None:
    metadata = _metadata_for(row)

    key = compose_key(row.agent_id, *metadata.dims)
    assert AgentRegistry.contains(key)
    if row.env_id == "webgym":
        requested = row.env_kwargs["extra_tools"]
        assert [tool_schema_name(schema) for schema in metadata.extra_tool_schemas] == requested
    assert isinstance(_build_agent(row, metadata), BaseAgent)
