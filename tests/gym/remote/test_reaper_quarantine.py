"""Integration tests for the low-level docker rm/quarantine helpers.

These tests spawn REAL docker containers (alpine, lightweight) in an
ISOLATED per-run scope so they cannot collide with any concurrently-running
env-server.

The reconcile-loop / drift behavior is covered by
``tests/gym/remote/test_reconcile.py`` and
``tests/gym/remote/test_reaper.py``. What remains here are the
``_try_orphan_rm`` quarantine-backoff tests for the surviving low-level
helper in :mod:`lite.gym.remote.reaper`.

Run:
    uv run pytest -m live tests/gym/remote/test_reaper_quarantine.py -v -s
"""

from __future__ import annotations

import hashlib
import subprocess

import pytest
from gym.remote.conftest import unique_live_scope, unique_live_token

from lite.gym.remote.reaper import _RM_QUARANTINE
from lite.gym.utils.config.naming import container_name_prefix, format_container_name

pytestmark = pytest.mark.live

TEST_TOKEN = unique_live_token("reaper_quarantine")
TEST_HASH = hashlib.sha256(TEST_TOKEN.encode()).hexdigest()[:6]
TEST_PORT = 40_000 + (int(TEST_HASH, 16) % 20_000)
TEST_ENV_ID = "androidworld"  # any registered docker env will do
TEST_IMAGE = "alpine:3.20"  # 7 MB; pre-pulled on first test
TEST_SESSION_ID = unique_live_scope("reaper_quarantine")


def _cleanup_scope() -> None:
    """`docker rm -f` any container in our test scope. Idempotent."""
    prefix = container_name_prefix(server_port=TEST_PORT, token_hash=TEST_HASH)
    r = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"name=^{prefix}"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    ids = [x for x in r.stdout.split() if x]
    if ids:
        subprocess.run(
            ["docker", "rm", "-f", *ids],
            capture_output=True,
            timeout=60,
        )


@pytest.fixture(autouse=True)
def _isolate_scope():
    """Wipe scope before + after each test so leftovers from a prior
    failed run don't leak across tests."""
    _RM_QUARANTINE.clear()
    _cleanup_scope()
    yield
    _RM_QUARANTINE.clear()
    _cleanup_scope()


# =============================================================================
# Quarantine backoff
# =============================================================================
# NOTE: the tier-escalation test (all-stubbed subprocess, no docker) moved to
# tests/gym/utils/backend/test_docker.py so it runs in the DEFAULT suite; only the
# real-docker clean-exit variant stays live-marked here.


def test_quarantine_clears_on_clean_exit():
    """A successful `docker rm -f` after a quarantine entry must clear
    the quarantine so the next iteration starts fresh."""
    from lite.gym.remote import reaper as dd

    name = format_container_name(
        env_id=TEST_ENV_ID,
        suffix="99",
        task_id="x",
        session_id=TEST_SESSION_ID,
        token_hash=TEST_HASH,
        server_port=TEST_PORT,
    )
    dd._RM_QUARANTINE[name] = (0.0, 1)  # pretend tier 1 quarantine (expired)

    # Spawn a real container with that name, then let _try_orphan_rm
    # remove it successfully.
    subprocess.run(
        ["docker", "run", "-d", "--name", name, TEST_IMAGE, "sleep", "60"],
        check=True,
        capture_output=True,
        timeout=90,
    )
    try:
        assert dd._try_orphan_rm(name, rm_timeout=60.0)
        assert name not in dd._RM_QUARANTINE
    finally:
        subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True,
            timeout=30,
        )
        dd._RM_QUARANTINE.clear()
