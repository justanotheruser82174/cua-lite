"""Unit tests for WebGym host-aware pool sizing.

Run: uv run pytest tests/gym/envs/webgym/test_pool_sizing.py
"""
from __future__ import annotations

import pytest

from lite.gym.envs.webgym.pool_sizing import (
    _INSTANCE_RAM_GB,
    _INSTANCES_PER_VCPU,
    _MAX_INSTANCES,
    _MIN_INSTANCES,
    _RAM_BUDGET_FRAC,
    PoolSize,
    derive_pool_size,
    derive_pool_size_from_env,
)


class TestDerivePoolSize:
    def test_default_nodes_is_one(self):
        # N>1 double-counts on a shared-redis single host; default must be 1.
        assert derive_pool_size(8, 32.0).nodes == 1

    def test_big_host_caps_instances_at_max(self):
        # 384 vCPU / 1505 GB would size to ~256 by both caps; bounded to _MAX.
        ps = derive_pool_size(384, 1505.0)
        assert ps.instances == _MAX_INSTANCES
        assert ps.nodes == 1

    def test_tiny_host_floors_instances_at_min(self):
        ps = derive_pool_size(1, 1.0)
        assert ps.instances == _MIN_INSTANCES

    def test_ram_is_the_binding_cap(self):
        # Lots of vCPU but little RAM → RAM cap wins.
        ram_gb = 40.0
        ps = derive_pool_size(1024, ram_gb)
        expected = int((ram_gb * _RAM_BUDGET_FRAC) / _INSTANCE_RAM_GB)
        assert ps.instances == expected

    def test_cpu_is_the_binding_cap(self):
        # Lots of RAM but few vCPU → vCPU cap wins.
        vcpu = 10
        ps = derive_pool_size(vcpu, 100000.0)
        expected = int(vcpu * _INSTANCES_PER_VCPU)
        assert ps.instances == expected

    def test_instances_override_bypasses_caps(self):
        ps = derive_pool_size(8, 32.0, instances_override=200)
        assert ps.instances == 200  # override is trusted, not clamped to _MAX

    def test_nonpositive_instances_override_ignored(self):
        ps = derive_pool_size(8, 32.0, instances_override=0)
        assert ps.instances == derive_pool_size(8, 32.0).instances

    def test_nodes_override_applied(self):
        assert derive_pool_size(8, 32.0, nodes_override=3).nodes == 3

    def test_nonpositive_nodes_override_falls_back_to_one(self):
        assert derive_pool_size(8, 32.0, nodes_override=0).nodes == 1

    def test_concurrency_hint_caps_overprovisioned_pool(self):
        # Big host would size to _MAX (128); a conc hint of 56 caps it to
        # ceil(56*1.3)=73 so the pool isn't 2.3x over-provisioned vs leases held.
        import math
        ps = derive_pool_size(384, 1505.0, concurrency_hint=56)
        assert ps.instances == math.ceil(56 * 1.3) == 73

    def test_concurrency_hint_does_not_inflate_a_small_host(self):
        # If the host-derived size is ALREADY below the hint cap, keep it
        # (the hint is a CEILING, never a floor).
        small = derive_pool_size(10, 100000.0)  # cpu-bound to 15
        capped = derive_pool_size(10, 100000.0, concurrency_hint=56)
        assert capped.instances == small.instances  # 15, not inflated to 73

    def test_concurrency_hint_never_below_min(self):
        ps = derive_pool_size(384, 1505.0, concurrency_hint=1)
        assert ps.instances == _MIN_INSTANCES  # ceil(1.3)=2 < 8 floor → 8

    def test_zero_concurrency_hint_is_legacy_host_sizing(self):
        assert (derive_pool_size(384, 1505.0, concurrency_hint=0).instances
                == derive_pool_size(384, 1505.0).instances == _MAX_INSTANCES)

    def test_explicit_instances_override_ignores_concurrency_hint(self):
        # A pinned M is trusted verbatim — the conc hint must not shrink it.
        ps = derive_pool_size(384, 1505.0, instances_override=200, concurrency_hint=56)
        assert ps.instances == 200

    def test_returns_frozen_dataclass(self):
        ps = derive_pool_size(8, 32.0)
        assert isinstance(ps, PoolSize)
        with pytest.raises(Exception):
            ps.instances = 5  # type: ignore[misc]


class TestDerivePoolSizeFromEnv:
    """M comes from the yaml (server_kwargs.instances; 0 = auto-derive).
    WEBGYM_INSTANCES is RETIRED as a config knob — the yaml is the single source
    (change it there or via WEBGYM_CONFIG); `ensure` passes the derived M to the
    container as `-e WEBGYM_INSTANCES`."""

    def test_env_instances_var_is_ignored(self, monkeypatch):
        # A stray WEBGYM_INSTANCES is NOT honoured — default.yaml is instances: 0,
        # so M is host-derived, not the env value.
        from lite.gym.utils import config as env_config
        monkeypatch.setenv("WEBGYM_INSTANCES", "42")
        monkeypatch.delenv("WEBGYM_CONFIG", raising=False)
        env_config.load.cache_clear()
        ps = derive_pool_size_from_env()
        assert ps.instances != 42
        assert _MIN_INSTANCES <= ps.instances <= _MAX_INSTANCES
        assert ps.nodes == 1
        env_config.load.cache_clear()

    def test_yaml_instances_pins_pool(self, monkeypatch, tmp_path):
        # The sanctioned way to pin M: a WEBGYM_CONFIG yaml (same flow `ensure` uses).
        from lite.gym.utils import config as env_config
        cfg = tmp_path / "pinned.yaml"
        cfg.write_text("env_var_prefix: WEBGYM\nserver_kwargs: {instances: 37}\n")
        monkeypatch.setenv("WEBGYM_CONFIG", str(cfg))
        monkeypatch.delenv("WEBGYM_INSTANCES", raising=False)
        env_config.load.cache_clear()
        ps = derive_pool_size_from_env()
        assert ps.instances == 37
        env_config.load.cache_clear()

    def test_no_config_uses_host_capacity(self, monkeypatch):
        from lite.gym.utils import config as env_config
        monkeypatch.delenv("WEBGYM_INSTANCES", raising=False)
        monkeypatch.delenv("WEBGYM_CONFIG", raising=False)
        env_config.load.cache_clear()
        ps = derive_pool_size_from_env()
        # Whatever the host probes to, it must obey the contract.
        assert _MIN_INSTANCES <= ps.instances <= _MAX_INSTANCES
        assert ps.nodes == 1
        env_config.load.cache_clear()
