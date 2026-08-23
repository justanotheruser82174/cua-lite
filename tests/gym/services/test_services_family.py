"""Unit tests for the BackendFamily declaration + derivations.

Pins the "declare family once, derive the rest" contract: register_family / family_of
(umbrella-aware), and the three pure derivations reconcile_mode / warm_strategy /
has_lifecycle. No docker / no env-server.

Run: uv run pytest tests/gym/services/test_services_family.py -q
"""
from __future__ import annotations

import sys

import pytest

from lite.gym.services import (
    BackendFamily,
    EnvServices,
    ReconcileMode,
    WarmStrategy,
    family_of,
    has_lifecycle,
    reconcile_mode,
    register_family,
    register_services,
    services_for,
    warm_strategy,
)

# The state dicts live on the registry MODULE, not the ``registry`` singleton that
# ``lite.gym.__init__`` rebinds onto the attribute (see services.py _registry_module).
registry = sys.modules["lite.gym.registry"]


@pytest.fixture
def clean_reg(monkeypatch):
    """Fresh family/services registries so tests don't leak into real envs."""
    monkeypatch.setattr(registry, "_families", {})
    monkeypatch.setattr(registry, "_services", {})
    yield


class _Services(EnvServices):
    pass


# --------------------------------------------------------------------------- #
# register_family / family_of
# --------------------------------------------------------------------------- #

def test_register_and_lookup(clean_reg):
    register_family("webgym", BackendFamily.SINGLETON)
    assert family_of("webgym") is BackendFamily.SINGLETON


def test_family_of_umbrella_fallback(clean_reg):
    # browsergym registers the umbrella; leaves resolve to it.
    register_family("browsergym", BackendFamily.SINGLETON)
    assert family_of("browsergym.webarena") is BackendFamily.SINGLETON
    assert family_of("browsergym.visualwebarena") is BackendFamily.SINGLETON
    assert family_of("browsergym.miniwob") is BackendFamily.SINGLETON


def test_multi_level_family_exact_registration_wins_over_umbrella(clean_reg):
    register_family("cua", BackendFamily.REMOTE)
    register_family("cua.bench.webtop.s02fam", BackendFamily.PURE)

    assert family_of("cua.bench.webtop.s02fam") is BackendFamily.PURE
    assert family_of("cua.bench.other") is BackendFamily.REMOTE


def test_family_of_undeclared_is_none(clean_reg):
    assert family_of("nope") is None


def test_register_family_rejects_non_enum(clean_reg):
    with pytest.raises(TypeError, match="must be a BackendFamily"):
        register_family("x", "singleton")  # type: ignore[arg-type]


def test_multi_level_services_exact_registration_wins_over_umbrella(clean_reg):
    umbrella = _Services()
    exact = _Services()

    register_services("cua", umbrella)
    register_services("cua.bench.webtop.s02fam", exact)

    assert services_for("cua.bench.webtop.s02fam") is exact
    assert services_for("cua.bench.other") is umbrella


# --------------------------------------------------------------------------- #
# reconcile_mode (pure function of family)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("family,expected", [
    (BackendFamily.DEDICATED, ReconcileMode.PER_INSTANCE),
    (BackendFamily.SINGLETON, ReconcileMode.BOOT_ONLY),
    (BackendFamily.PURE, ReconcileMode.NONE),
    (BackendFamily.REMOTE, ReconcileMode.NONE),
    (None, ReconcileMode.NONE),
])
def test_reconcile_mode(family, expected):
    assert reconcile_mode(family) is expected


# --------------------------------------------------------------------------- #
# has_lifecycle (pure function of family)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("family,expected", [
    (BackendFamily.DEDICATED, True),
    (BackendFamily.SINGLETON, True),
    (BackendFamily.PURE, False),
    (BackendFamily.REMOTE, False),
    (None, False),
])
def test_has_lifecycle(family, expected):
    assert has_lifecycle(family) is expected


# --------------------------------------------------------------------------- #
# warm_strategy
# --------------------------------------------------------------------------- #

def test_warm_strategy_singleton_is_eager(clean_reg):
    register_family("webgym", BackendFamily.SINGLETON)
    assert warm_strategy("webgym") is WarmStrategy.EAGER


def test_warm_strategy_dedicated_is_none(clean_reg):
    register_family("androidworld", BackendFamily.DEDICATED)
    assert warm_strategy("androidworld") is WarmStrategy.NONE
    register_family("captcha", BackendFamily.DEDICATED)
    assert warm_strategy("captcha") is WarmStrategy.NONE


def test_warm_strategy_pure_remote_none(clean_reg):
    register_family("screenspot_pro", BackendFamily.PURE)
    register_family("osworld", BackendFamily.REMOTE)
    assert warm_strategy("screenspot_pro") is WarmStrategy.NONE
    assert warm_strategy("osworld") is WarmStrategy.NONE


def test_warm_strategy_undeclared_is_none(clean_reg):
    assert warm_strategy("nope") is WarmStrategy.NONE
