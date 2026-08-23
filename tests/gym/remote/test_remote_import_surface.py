"""Remote client import surface guards."""

from __future__ import annotations

import subprocess
import sys


def test_remote_client_facade_does_not_eager_import_server() -> None:
    code = """
import sys
import lite.gym.remote
from lite.gym.remote.client import LiteEnvClient
assert LiteEnvClient.__name__ == "LiteEnvClient"
assert "lite.gym.remote.server" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
