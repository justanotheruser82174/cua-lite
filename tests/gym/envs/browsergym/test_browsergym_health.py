"""Unit tests for BrowserGymServices.health() — the per-benchmark readiness probe
that makes `GET /envs/browsergym.<bench>` report the truth.
Mocks the HTTP probes; no docker.

Run: uv run pytest tests/gym/envs/browsergym/test_browsergym_health.py -q
"""
from __future__ import annotations

import pytest

pytest.importorskip("browsergym.core", reason="browsergym not installed")
from lite.gym.envs.browsergym import main as bg
from lite.gym.errors import EnvDepsMissingError


@pytest.fixture
def svc():
    return bg.BrowserGymServices()


def test_health_passes_when_stack_up(svc, monkeypatch):
    monkeypatch.setattr(bg, "_service_up", lambda b: True)
    monkeypatch.setattr(bg, "_homepage_up", lambda: True)
    monkeypatch.setattr(bg, "_classifieds_up", lambda: True)
    svc.health("browsergym.webarena")  # no raise
    svc.health("browsergym.visualwebarena")
    svc.health("browsergym.miniwob")


def test_health_raises_when_stack_cold(svc, monkeypatch):
    monkeypatch.setattr(bg, "_service_up", lambda b: False)
    with pytest.raises(EnvDepsMissingError, match="webarena"):
        svc.health("browsergym.webarena")


def test_health_vwa_needs_classifieds(svc, monkeypatch):
    # WA stack up but classifieds down ⇒ visualwebarena not healthy, webarena is.
    monkeypatch.setattr(bg, "_service_up", lambda b: True)
    monkeypatch.setattr(bg, "_homepage_up", lambda: True)
    monkeypatch.setattr(bg, "_classifieds_up", lambda: False)
    svc.health("browsergym.webarena")  # WA fine
    with pytest.raises(EnvDepsMissingError, match="visualwebarena"):
        svc.health("browsergym.visualwebarena")


def test_health_per_leaf(svc, monkeypatch):
    # webarena up, visualwebarena down (its classifieds missing) — only VWA fails.
    monkeypatch.setattr(bg, "_service_up", lambda b: True)
    monkeypatch.setattr(bg, "_homepage_up", lambda: True)
    monkeypatch.setattr(bg, "_classifieds_up", lambda: False)
    svc.health("browsergym.webarena")
    with pytest.raises(EnvDepsMissingError):
        svc.health("browsergym.visualwebarena")


def test_health_raises_when_homepage_down(svc, monkeypatch):
    # I3's needle probe is a health conjunct of its own: services can be "up"
    # (ports listening) while the WA homepage serves an error page.
    monkeypatch.setattr(bg, "_service_up", lambda b: True)
    monkeypatch.setattr(bg, "_classifieds_up", lambda: True)
    monkeypatch.setattr(bg, "_homepage_up", lambda: False)
    with pytest.raises(EnvDepsMissingError, match="webarena"):
        svc.health("browsergym.webarena")


def test_health_bare_umbrella_is_noop(svc, monkeypatch):
    # Bare umbrella has no single backend to probe — must not raise even if cold.
    monkeypatch.setattr(bg, "_service_up", lambda b: False)
    svc.health("browsergym")  # no raise


def test_health_never_boots(svc, monkeypatch):
    # health must be probe-only — assert it never calls the boot path.
    monkeypatch.setattr(bg, "_service_up", lambda b: True)
    monkeypatch.setattr(bg, "_homepage_up", lambda: True)
    monkeypatch.setattr(bg, "_classifieds_up", lambda: True)
    def _boom(*a, **k):
        raise AssertionError("health() must not boot (_ensure_services called)")
    monkeypatch.setattr(bg, "_ensure_services", _boom)
    svc.health("browsergym.webarena")


# --------------------------------------------------------------------------- #
# all-sites readiness: _service_up(WA/VWA) gates on EVERY local site, not just
# shopping — so available:true can't lie while gitlab is still booting.
# --------------------------------------------------------------------------- #

def _only_gitlab_down(url, needle, ok_codes):
    # Resolve the gitlab port the SAME way _wa_stack_up does (env override → default)
    # so the match holds even when GITLAB_PORT is exported in the runner's env.
    import os
    gitlab_port = os.environ.get("GITLAB_PORT", str(bg._GITLAB_PORT_DEFAULT))
    return f":{gitlab_port}/" not in url  # every site up except gitlab


def test_service_up_false_when_only_gitlab_down(monkeypatch):
    monkeypatch.setattr(bg, "_wa_site_up", _only_gitlab_down)
    assert bg._service_up("webarena") is False  # the whole stack is "not ready"


def test_service_up_true_when_all_sites_up(monkeypatch):
    monkeypatch.setattr(bg, "_wa_site_up", lambda url, needle, ok: True)
    assert bg._service_up("webarena") is True


def test_down_sites_names_gitlab(monkeypatch):
    monkeypatch.setattr(bg, "_wa_site_up", _only_gitlab_down)
    assert bg._wa_down_sites("webarena") == ["gitlab"]


def test_vwa_down_sites_includes_classifieds(monkeypatch):
    monkeypatch.setattr(bg, "_wa_site_up", lambda url, needle, ok: True)  # shared sites up
    monkeypatch.setattr(bg, "_classifieds_up", lambda: False)            # classifieds down
    assert bg._wa_down_sites("visualwebarena") == ["classifieds"]
    # webarena (no classifieds dependency) is fully up in the same state
    assert bg._wa_down_sites("webarena") == []
