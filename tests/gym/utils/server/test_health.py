"""Tests for shared env-server health helpers.

Run: uv run pytest tests/gym/utils/server/test_health.py -q
"""
from __future__ import annotations

import pytest

from lite.gym.errors import CapacityExhausted, EnvDepsMissingError
from lite.gym.utils.server.health import CachedEnvDepsHealthCheck


def test_cached_env_deps_health_check_caches_success_until_ttl() -> None:
    calls: list[str] = []
    clock = {"now": 100.0}
    health = CachedEnvDepsHealthCheck(
        lambda env_id: calls.append(env_id),
        clock=lambda: clock["now"],
    )

    health("android")
    health("android")

    assert calls == ["android"]


def test_cached_env_deps_health_check_refreshes_after_ttl() -> None:
    calls: list[str] = []
    clock = {"now": 100.0}
    health = CachedEnvDepsHealthCheck(
        lambda env_id: calls.append(env_id),
        clock=lambda: clock["now"],
    )

    health("android")
    clock["now"] = 105.0
    health("android")
    clock["now"] = 111.0
    health("android")

    assert calls == ["android", "android"]


def test_cached_env_deps_health_check_caches_dependency_failure() -> None:
    calls: list[str] = []

    def fail(env_id: str) -> None:
        calls.append(env_id)
        raise EnvDepsMissingError(
            what="missing android image",
            install="uv run --no-sync bash lite/gym/envs/android/scripts/install.sh",
            see="/lite/gym/envs/android/README.md",
        )

    health = CachedEnvDepsHealthCheck(fail, clock=lambda: 100.0)

    with pytest.raises(EnvDepsMissingError, match="missing android image") as first:
        health("android")
    with pytest.raises(EnvDepsMissingError, match="missing android image") as second:
        health("android")

    assert second.value is not first.value
    assert second.value.what == first.value.what
    assert second.value.install == first.value.install
    assert second.value.see == first.value.see
    assert calls == ["android"]


def test_cached_env_deps_health_check_does_not_cache_transient_capacity() -> None:
    calls: list[str] = []

    def fail(env_id: str) -> None:
        calls.append(env_id)
        raise CapacityExhausted(
            what="docker inspect busy",
            retry_after_s=1.0,
            layer="image_freshness",
        )

    health = CachedEnvDepsHealthCheck(fail, clock=lambda: 100.0)

    for _ in range(2):
        with pytest.raises(CapacityExhausted, match="docker inspect busy"):
            health("android")

    assert calls == ["android", "android"]
