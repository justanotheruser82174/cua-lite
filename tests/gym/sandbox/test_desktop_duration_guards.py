from __future__ import annotations

from types import SimpleNamespace

import pytest

import lite.gym.sandbox.base as sandbox_base
from lite.gym.sandbox.base import _dispatch_desktop_action


class _FakeInterface:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def key_down(self, key: str) -> None:
        self.calls.append(("key_down", key))

    async def key_up(self, key: str) -> None:
        self.calls.append(("key_up", key))


async def _fail_sleep(_duration: float) -> None:
    raise AssertionError("duration guard should run before asyncio.sleep")


@pytest.mark.asyncio
async def test_wait_rejects_huge_duration_before_sleep(monkeypatch):
    monkeypatch.setattr(sandbox_base.asyncio, "sleep", _fail_sleep)

    with pytest.raises(ValueError, match="wait.duration must be <= 30"):
        await _dispatch_desktop_action(
            SimpleNamespace(interface=_FakeInterface()),
            "wait",
            {"duration": 31},
            1000,
            1000,
        )


@pytest.mark.asyncio
async def test_hold_key_rejects_huge_duration_before_key_down(monkeypatch):
    monkeypatch.setattr(sandbox_base.asyncio, "sleep", _fail_sleep)
    interface = _FakeInterface()

    with pytest.raises(ValueError, match="hold_key.duration must be <= 5"):
        await _dispatch_desktop_action(
            SimpleNamespace(interface=interface),
            "hold_key",
            {"keys": ["a"], "duration": 6},
            1000,
            1000,
        )

    assert interface.calls == []
