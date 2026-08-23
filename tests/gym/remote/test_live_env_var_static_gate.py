"""Static gate for live/stress remote env-server variable ownership."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REMOTE_TEST_ROOT = Path(__file__).resolve().parent
REPO_ROOT = REMOTE_TEST_ROOT.parents[2]
ROOT_ENV_SERVER_VARS = ("CUA_LITE_ENV_SERVER_URL", "CUA_LITE_ENV_SERVER_TOKEN")
GENERIC_CLEANUP_SESSION_IDS = (
    "stress",
    "bulk",
    "dry",
    "test",
    "burst",
    "h6",
    "h7",
    "h9",
    "s1",
    "s2",
)
GENERIC_CLEANUP_SESSION_RE = "|".join(
    re.escape(session_id) for session_id in GENERIC_CLEANUP_SESSION_IDS
)
FORBIDDEN_LIVE_DEFAULT_PATTERNS = (
    re.compile(r"\bLIVE_SERVER_URL\s*=\s*['\"]http"),
    re.compile(r"\bLIVE_SERVER_TOKEN\s*=\s*['\"]"),
    re.compile(r"\bLIVE_ADMIN_TOKEN\s*=\s*['\"]"),
    re.compile(r"\bDEV_URL\s*="),
    re.compile(r"\bDEV_TOKEN\s*="),
    re.compile(r"\bADMIN_TOKEN\s*="),
    re.compile(r"zzh-dev"),
    re.compile(r"zzh-admin"),
)
FORBIDDEN_UNSCOPED_CLEANUP_PATTERNS = (
    re.compile(
        rf"session_id[\"']?\s*[:=]\s*['\"](?:{GENERIC_CLEANUP_SESSION_RE})['\"]"
    ),
    re.compile(rf"session_id=(?:{GENERIC_CLEANUP_SESSION_RE})(?:[&'\"\s]|$)"),
    re.compile(r"['\"]force['\"]\s*:\s*['\"]true['\"]"),
    re.compile(r"['\"]force['\"]\s*:\s*True"),
    re.compile(r"force=true"),
    re.compile(r"force=True"),
)
FORBIDDEN_UNSCOPED_TOKEN_PATTERNS = (
    re.compile(r"['\"]burst-token-[^'\"]*['\"]"),
    re.compile(r"\btoken\s*=\s*['\"](?:alice|bob|carol|dave|probe)['\"]"),
    re.compile(r"_bearer\(['\"](?:alice|bob|carol|dave|probe)['\"]\)"),
)
FORBIDDEN_SHARED_CONTAINER_SCOPE_PATTERNS = (
    re.compile(r"\b[A-Z_]*PORT\s*=\s*30200\b"),
    re.compile(r"server_port\s*=\s*30200\b"),
    re.compile(r"container_name_prefix\([^)]*server_port\s*=\s*30200"),
    re.compile(r"lite-env-30200"),
)
FORBIDDEN_PATTERNS = (
    FORBIDDEN_LIVE_DEFAULT_PATTERNS
    + FORBIDDEN_UNSCOPED_CLEANUP_PATTERNS
    + FORBIDDEN_UNSCOPED_TOKEN_PATTERNS
    + FORBIDDEN_SHARED_CONTAINER_SCOPE_PATTERNS
)


def _collect_live_stress_nodeids() -> list[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ROOT_ENV_SERVER_VARS
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-qq",
            "-p",
            "no:cacheprovider",
            "-m",
            "live or stress",
            str(REMOTE_TEST_ROOT.relative_to(REPO_ROOT)),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.startswith("tests/") and ".py::" in line
    ]


def _nodeid_paths(nodeids: list[str]) -> list[Path]:
    return sorted({REPO_ROOT / nodeid.split("::", 1)[0] for nodeid in nodeids})


def _remote_live_stress_files() -> list[Path]:
    return [path for path in _nodeid_paths(_collect_live_stress_nodeids()) if path != THIS_FILE]


def _forbidden_lines(path: Path, text: str) -> list[str]:
    offenders: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for env_var in ROOT_ENV_SERVER_VARS:
            if env_var in line:
                rel_path = path.relative_to(REPO_ROOT)
                offenders.append(f"{rel_path}:{line_no}: {env_var}")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(line):
                rel_path = path.relative_to(REPO_ROOT)
                offenders.append(f"{rel_path}:{line_no}: {pattern.pattern}")
    return offenders


def test_live_stress_files_do_not_read_root_env_server_vars_or_token_defaults() -> None:
    live_stress_files = _remote_live_stress_files()
    offenders: list[str] = []

    for path in live_stress_files:
        text = path.read_text(encoding="utf-8")
        offenders.extend(_forbidden_lines(path, text))

    assert live_stress_files, "static gate found no live/stress remote test files"
    assert not offenders, (
        "live/stress remote tests must use live-only env vars, not root "
        "env-server pytest bootstrap vars, hard-coded dev token defaults, "
        "generic cleanup scopes, fixed cleanup tokens, or shared 30200 "
        "container cleanup scopes:\n"
        + "\n".join(offenders)
    )


def test_live_stress_gate_rejects_forbidden_live_defaults_and_cleanup() -> None:
    sample = "\n".join(
        [
            "LIVE_SERVER_URL = \"http://127.0.0.1:30200\"",
            "LIVE_SERVER_TOKEN = \"zzh-dev\"",
            "LIVE_ADMIN_TOKEN = \"zzh-admin\"",
            "params={\"session_id\": \"stress\", \"force\": \"true\"}",
            "params={\"session_id\": \"bulk\", \"env_id\": env_id}",
            "await _bulk_cleanup_session(ac, \"burst-token-A\", env_id, \"h6\")",
            "TEST_PORT = 30200",
        ]
    )
    hits = _forbidden_lines(REPO_ROOT / "tests/gym/remote/test_sample.py", sample)
    assert any("LIVE_SERVER_URL" in hit for hit in hits)
    assert any("zzh-dev" in hit for hit in hits)
    assert any("session_id" in hit for hit in hits)
    assert any("force" in hit for hit in hits)
    assert any("burst-token" in hit for hit in hits)
    assert any("30200" in hit for hit in hits)


def test_live_stress_gate_accepts_final_scoped_sweep() -> None:
    sample = "\n".join(
        [
            "session_id = unique_live_scope(\"round\")",
            "token = unique_live_token(\"round\")",
            "await _bulk_cleanup_session(ac, token, env_id, session_id)",
            (
                "await client.delete('/instances', "
                "params={'session_id': session_id, 'env_id': env_id})"
            ),
        ]
    )

    hits = _forbidden_lines(REPO_ROOT / "tests/gym/remote/test_sample.py", sample)
    assert hits == []


def test_live_stress_file_discovery_uses_pytest_collection_nodeids() -> None:
    nodeids = [
        "tests/gym/remote/test_live_server.py::TestHealth::test_host_status_open",
        "tests/gym/remote/test_reaper_quarantine.py::test_quarantine_clears_on_clean_exit",
    ]

    assert _nodeid_paths(nodeids) == [
        REPO_ROOT / "tests/gym/remote/test_live_server.py",
        REPO_ROOT / "tests/gym/remote/test_reaper_quarantine.py",
    ]
