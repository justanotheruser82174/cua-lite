"""The env-server pre-flight script's expected wire version must track the module.

Run:
    uv run pytest tests/scripts/train/test_preflight.py -q
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from lite.gym.remote.frame import FRAME_MAGIC, FRAME_VERSION
from lite.utils.path import project_root

_SCRIPT = project_root() / "scripts/train/utils/preflight.sh"
_ROOT = project_root()


def _default(var: str) -> str:
    """The ``${VAR:-default}`` fallback the script hardcodes for *var*."""
    text = Path(_SCRIPT).read_text(encoding="utf-8")
    m = re.search(rf'\$\{{{var}:-([^}}]+)\}}', text)
    assert m, f"{var} not found in {_SCRIPT}"
    return m.group(1)


def test_preflight_expects_the_wire_the_module_emits():
    """A FRAME_MAGIC bump must redden here, not fail every training launch.

    The script probes ``/host_status.wire`` with a hardcoded expectation and is
    deliberately python-free, so nothing binds it to the module at runtime. Left
    unbound, bumping ``FRAME_MAGIC`` makes every pre-flight reject a CORRECT
    server as stale.
    """
    assert _default("CUA_LITE_EXPECTED_FRAME_MAGIC") == FRAME_MAGIC
    assert _default("CUA_LITE_EXPECTED_FRAME_VERSION") == str(FRAME_VERSION)


def _run_preflight_with_fake_curl(
    tmp_path: Path,
    host_status: str,
    *,
    session_id: str = "session",
    env_id: str = "browsergym.miniwob",
) -> tuple[subprocess.CompletedProcess[str], Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    curl_log = tmp_path / "curl.log"
    curl = fake_bin / "curl"
    curl.write_text(
        """#!/usr/bin/env bash
{
  printf 'CALL'
  for arg in "$@"; do
    printf '\\t%s' "$arg"
  done
  printf '\\n'
} >> "$CURL_LOG"

case " $* " in
  *"/host_status"*) printf '%s\\n' "$FAKE_HOST_STATUS" ;;
  *"/envs/"*) printf '%s\\n' '{"available":true,"n_tasks":1}' ;;
  *"dry_run=true"*) printf '%s\\n' '{"would_close":[]}' ;;
  *"/instances"*) printf '%s\\n' '{"instances":[]}' ;;
  *) echo "unexpected curl args: $*" >&2; exit 7 ;;
esac
""",
        encoding="utf-8",
    )
    curl.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CURL_LOG": str(curl_log),
        "FAKE_HOST_STATUS": host_status,
        "CUA_LITE_ENV_SERVER_URL": "http://server",
        "CUA_LITE_ENV_SERVER_TOKEN": "token",
        "CUA_LITE_EXPECTED_COMMIT": "abc123",
        "ENV_ID": env_id,
        "SESSION_ID": session_id,
    }
    proc = subprocess.run(
        ["bash", "-c", f"source {str(_SCRIPT)!r}"],
        cwd=_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    return proc, curl_log


def test_preflight_rejects_frame_version_prefix_match(tmp_path: Path) -> None:
    host_status = f'{{"frame_magic":"{FRAME_MAGIC}","frame_version":60,"commit":"abc123"}}'

    proc, _ = _run_preflight_with_fake_curl(tmp_path, host_status)

    assert proc.returncode != 0
    assert "frame_version mismatch" in proc.stderr


def test_preflight_urlencodes_bulk_instance_scope(tmp_path: Path) -> None:
    host_status = (
        f'{{"frame_magic":"{FRAME_MAGIC}","frame_version":{FRAME_VERSION},"commit":"abc123"}}'
    )
    session_id = "manual+a b&c%25"

    proc, curl_log = _run_preflight_with_fake_curl(
        tmp_path,
        host_status,
        session_id=session_id,
    )

    assert proc.returncode == 0, proc.stderr
    calls = curl_log.read_text(encoding="utf-8").splitlines()
    instance_calls = [line for line in calls if "\thttp://server/instances" in line]
    assert len(instance_calls) == 2
    assert all("?session_id=" not in line for line in instance_calls)
    assert all("\t-G\t" in line for line in instance_calls)
    assert all(f"\t--data-urlencode\tsession_id={session_id}" in line for line in instance_calls)
    assert all("\t--data-urlencode\tenv_id=browsergym.miniwob" in line for line in instance_calls)
    assert any("\t--data-urlencode\tdry_run=true" in line for line in instance_calls)
