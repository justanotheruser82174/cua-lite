"""Model B port auto-pick must degrade gracefully on range exhaustion.

Regression for the bug where the Model B autopick path (browsergym's
WebArena ports) caught ``except RuntimeError`` around ``allocate_ports`` /
``allocate_ports``, and those allocators raise
``lite.gym.errors.CapacityExhausted`` — which is NOT a ``RuntimeError``
subclass. So on a fully-exhausted fallback range the exception propagated out
of ``ensure()`` as a 503 instead of the intended "log + fail loudly"
degradation. The auto-pick below forces the exhaustion path and asserts it
returns instead of raising.

(webgym + mobilegym are now container-only — their host autopick helpers were
removed, so only browsergym keeps a host-side WebArena port plan.)

Run: uv run pytest tests/gym/envs/browsergym/test_port_autopick_exhaustion.py
"""
from __future__ import annotations

import pytest

import lite.gym.utils.backend.ports as port_mod
from lite.gym.errors import CapacityExhausted, EnvDepsMissingError

try:
    from lite.gym.envs.browsergym import main as browsergym_main
except EnvDepsMissingError as e:  # browsergym absent until the env's install.sh runs
    pytest.skip(f"browsergym env deps not installed: {e}", allow_module_level=True)


def test_capacity_exhausted_is_not_runtimeerror():
    # The whole bug hinges on this: the old `except RuntimeError` could never
    # catch what the allocator actually raises.
    assert not issubclass(CapacityExhausted, RuntimeError)


@pytest.fixture
def _force_exhaustion(monkeypatch):
    """Make every preferred port look busy and every allocator raise
    CapacityExhausted, so the auto-pick functions must hit their except clause."""
    monkeypatch.setattr(port_mod, "_is_port_free", lambda *a, **k: False)

    def _boom(*a, **k):
        raise CapacityExhausted(
            what="synthetic: fallback range exhausted", retry_after_s=30.0,
        )

    monkeypatch.setattr(port_mod, "allocate_ports", _boom)


def test_browsergym_autopick_degrades(monkeypatch, _force_exhaustion):
    for row in browsergym_main._WA_PORT_PLAN:
        monkeypatch.delenv(row["port_var"], raising=False)
    browsergym_main._auto_pick_webarena_ports()
