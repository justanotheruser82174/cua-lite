from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_package_root_import_has_no_registration_side_effects() -> None:
    result = _run_python(
        """
import json
import sys

import lite.agents as agents

print(json.dumps({
    "core_action_space_loaded": "lite.agents.core.action_space.base" in sys.modules,
    "has_make_export": "make" in agents.__all__,
    "models_loaded": any(
        name == "lite.agents.models" or name.startswith("lite.agents.models.")
        for name in sys.modules
    ),
}))
"""
    )
    payload = json.loads(result.stdout)

    assert payload == {
        "core_action_space_loaded": False,
        "has_make_export": True,
        "models_loaded": False,
    }


def test_package_root_surface_is_make_register_registry() -> None:
    """The root is a three-name facade (mirroring ``lite.gym``), not a
    re-export table: every other name is imported from its owning module."""
    result = _run_python(
        """
import json

import lite.agents as agents
from lite.agents import make, register, registry

print(json.dumps({
    "all": sorted(agents.__all__),
    "make_callable": callable(make),
    "register_callable": callable(register),
    "registry": registry.__name__,
    "has_adapter_reexport": hasattr(agents, "Qwen3VLDesktopUseAdapter"),
    "has_registry_reexport": hasattr(agents, "AgentAdapterRegistry"),
}))
"""
    )
    payload = json.loads(result.stdout)

    assert payload == {
        "all": ["make", "register", "registry"],
        "make_callable": True,
        "register_callable": True,
        "registry": "AgentRegistry",
        "has_adapter_reexport": False,
        "has_registry_reexport": False,
    }


def test_bootstrap_registers_builtins_idempotently() -> None:
    from lite.agents.bootstrap import register_all
    from lite.agents.core.action_space.base import ActionSpaceRegistry
    from lite.agents.core.adapter import AgentAdapterRegistry
    from lite.agents.core.agent import AgentRegistry
    from lite.agents.core.protocol import ProtocolRegistry

    register_all()
    register_all()

    assert ActionSpaceRegistry.contains("qwen3_vl@desktop")
    assert AgentAdapterRegistry.contains("qwen3_vl@desktop@use")
    assert AgentRegistry.contains("qwen3_vl@desktop@use")
    assert ProtocolRegistry.contains("qwen3_vl.history")


def test_agent_core_facade_exposes_only_runtime_surface() -> None:
    """Message builders live under their owner module, not the package facade.

    A closure gate, not implementation coupling: it names retired/relocated
    helpers only to assert they stay OFF the facade. ``build_observation_message``
    has no owner module left either — it was deleted outright — so removing this
    test would let any of the three drift back onto ``lite.agents.core.agent``.
    """
    import lite.agents.core.agent as agent

    assert set(agent.__all__) == {
        "AdapterBasedAgent",
        "AgentRegistry",
        "AutoAdapterAgent",
        "BaseAgent",
    }
    for helper in {
        "build_initial_user_message",
        "build_observation_message",
        "build_tool_result_message",
    }:
        assert not hasattr(agent, helper)


def test_factory_make_bootstraps_builtins_in_fresh_process() -> None:
    result = _run_python(
        """
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from lite.agents.core.agent import AgentRegistry
from lite.agents.factory import make
from lite.core import LiteCUAMetadata

before = AgentRegistry.contains("qwen3_vl@desktop@use")
env = SimpleNamespace(metadata=LiteCUAMetadata(dims=("desktop", "use")))
agent = make(
    "Qwen/Qwen3-VL-2B-Instruct",
    env=env,
    processor=MagicMock(),
    generate_fn=lambda *args, **kwargs: None,
)
print(json.dumps({
    "before": before,
    "after": AgentRegistry.contains("qwen3_vl@desktop@use"),
    "agent_class": type(agent).__name__,
}))
"""
    )
    payload = json.loads(result.stdout)

    assert payload == {
        "before": False,
        "after": True,
        "agent_class": "Qwen3VLDesktopUseAgent",
    }


def test_python_m_agents_lists_bootstrapped_registries() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "lite.agents"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "CUA-Lite Agents Module" in result.stdout
    assert "Registered action spaces:" in result.stdout
    assert "qwen3_vl@mobile: Qwen3VLMobileActionSpace" in result.stdout
    assert "qwen3_vl@mobile@use: Qwen3VLMobileUseAdapter" in result.stdout
    assert "qwen3_vl.history: Qwen3VLHistoryProtocol" in result.stdout
