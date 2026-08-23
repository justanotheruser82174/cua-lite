"""Regression guards for lite.osworld Chrome cookie WAL checkpointing.

``osworld/src/gen/train/synth/chrome.py`` cookie postconfig must open the store
``mode=rw`` so an absent ``Cookies`` file is never fabricated as a 0-byte
database. The same rule is used by the shared flush in ``src/eval/runner.py``.

Run:
    uv run pytest tests/gym/envs/lite/osworld/test_lite_osworld_cookie_checkpoint.py -q
"""

from __future__ import annotations

import importlib.util
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

needs_desktop_env = pytest.mark.skipif(
    importlib.util.find_spec("desktop_env") is None,
    reason="desktop_env not installed (lite.osworld install.sh)",
)

_COOKIES_VM_PATH = "/home/user/chrome-data/Default/Cookies"


def _cookie_checkpoint_payload() -> str:
    """The `python3 -c` body the synth generator emits into `postconfig[0]`.

    Pulled from the generator (not from the committed JSONL) so this guard is
    green the moment the source is fixed, without waiting on a corpus regen.
    """
    try:
        from lite.gym.envs.lite.osworld.src.gen.train.synth.chrome import (
            _gold_cookie_delete_domains,
        )
    except Exception as exc:  # unseeded checkout: synth modules stage assets on import
        pytest.skip(f"synth chrome generator not importable: {exc}")

    _oracle, evaluator = _gold_cookie_delete_domains(domains=(".cnn.com",))
    command = evaluator["postconfig"][0]["parameters"]["command"]
    match = re.search(r'python3 -c "(.*?)" 2>/dev/null', command, re.S)
    assert match, f"postconfig[0] no longer wraps a `python3 -c` payload:\n{command}"
    return match.group(1)


def _run_checkpoint_against(store: Path) -> subprocess.CompletedProcess:
    """Run the emitted payload with the hardcoded VM path retargeted at `store`."""
    payload = _cookie_checkpoint_payload().replace(_COOKIES_VM_PATH, str(store))
    return subprocess.run([sys.executable, "-c", payload], capture_output=True, text=True)


def _seed_cookie_store(store: Path, host_key: str = ".cnn.com") -> None:
    conn = sqlite3.connect(store)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE cookies (creation_utc INTEGER, host_key TEXT, name TEXT)")
    conn.execute("INSERT INTO cookies VALUES (1, ?, 'sid')", (host_key,))
    conn.commit()
    conn.close()


def _read_cookies_from_main_db_only(store: Path, dest: Path) -> list:
    """Copy just the main .db (what `_download_from_container` fetches) and read it."""
    dest.write_bytes(store.read_bytes())
    conn = sqlite3.connect(dest)
    rows = conn.execute("SELECT * FROM cookies").fetchall()
    conn.close()
    return rows


def test_cookie_checkpoint_does_not_fabricate_an_absent_store(tmp_path):
    """THE GUARD. `sqlite3.connect(p)` opens rw-CREATE, so on a wholesale delete
    of `Cookies` it writes a 0-byte database where the evaluator expects to find
    "absent". `mode=rw` fails the open instead.
    """
    store = tmp_path / "Default" / "Cookies"
    store.parent.mkdir(parents=True)  # the profile dir always exists in the VM
    assert not store.exists()

    _run_checkpoint_against(store)

    assert not store.exists(), (
        "the cookie WAL-checkpoint postconfig fabricated an absent Cookies store; "
        "it must open `file:...?mode=rw` (see src/eval/runner.py for the same rule)"
    )


def test_cookie_checkpoint_still_flushes_a_real_store(tmp_path):
    """The guard must not cost the checkpoint its job — this is the reason the
    postconfig exists (agent deletes cookies via the UI -> the delete sits in the
    WAL -> `is_cookie_deleted` reads the main .db and false-fails).

    Models the real eval path: the runner downloads ONLY `/…/Default/Cookies`,
    never the `-wal`/`-shm` sidecars, so the delete has to be IN the main file.
    """
    store = tmp_path / "Default" / "Cookies"
    store.parent.mkdir(parents=True)
    _seed_cookie_store(store)

    # Delete in a child that exits WITHOUT sqlite cleanup (chrome getting pkill'd):
    # the commit lands in the WAL and no close-time checkpoint folds it in.
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sqlite3, os, sys\n"
            "c = sqlite3.connect(sys.argv[1])\n"
            "c.execute(\"DELETE FROM cookies WHERE host_key = '.cnn.com'\")\n"
            "c.commit()\n"
            "os._exit(0)\n",
            str(store),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    wal = store.parent / "Cookies-wal"
    assert wal.exists() and wal.stat().st_size > 0, "precondition: delete is WAL-only"
    assert _read_cookies_from_main_db_only(store, tmp_path / "pre") != [], (
        "precondition: the main .db still shows the pre-delete cookie"
    )

    proc = _run_checkpoint_against(store)
    assert proc.returncode == 0, proc.stderr

    assert store.exists()
    # TRUNCATE zeroes the WAL; the clean close then unlinks it.
    assert not wal.exists() or wal.stat().st_size == 0, "WAL was not checkpointed"
    assert _read_cookies_from_main_db_only(store, tmp_path / "post") == [], (
        "the checkpoint did not fold the agent's delete into the main .db"
    )


@needs_desktop_env
def test_a_fabricated_empty_store_is_one_getter_change_from_fail_open(tmp_path):
    """RECORD the blast radius, so nobody 'simplifies' the guard away.

    Today the fabrication fails CLOSED: the runner's `cookie_data` getter does
    `SELECT * FROM cookies`, which raises on the 0-byte file, so the getter
    returns None and `is_cookie_deleted(None, ...)` raises -> runner scores 0.0.
    It becomes a live fail-OPEN the moment that getter returns `[]` instead.
    """
    from desktop_env.evaluators.metrics.chrome import is_cookie_deleted

    rule = {"type": "domains", "domains": [".cnn.com"]}
    fabricated = tmp_path / "Cookies"
    sqlite3.connect(fabricated).close()
    assert fabricated.stat().st_size == 0

    conn = sqlite3.connect(fabricated)
    with pytest.raises(sqlite3.OperationalError):  # -> getter returns None
        conn.execute("SELECT * FROM cookies").fetchall()
    conn.close()

    with pytest.raises(TypeError):  # -> runner scores 0.0
        is_cookie_deleted(None, rule)
    assert is_cookie_deleted([], rule) == 1.0  # the latent fail-open
