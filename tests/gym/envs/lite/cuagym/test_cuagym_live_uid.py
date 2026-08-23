"""Durable live UID / dconf gates for a built ``lite.cuagym`` image.

These live tests cover two container identity invariants:

  1. GUI apps launch as the desktop user. cuagym's exec-stdio server runs AS the
     desktop ``user`` and ``op_run_command`` is a plain ``bash -c``: there is no uid
     drop, no ``runuser``, and no libreoffice-shim anywhere in production. So every
     GUI app the setup path launches must run as ``user``, NEVER root.
     :func:`test_cuagym_gui_apps_launch_as_user_not_root` launches the available B1
     apps, enumerates their PIDs and asserts ``ROOT_APP_COUNT == 0``.

  2. Rewards read the agent user's dconf. A reward that only reads the user's
     session (``gsettings get``) MUST run as the desktop user, else root reads
     root's dconf and scores the wrong value.
     :func:`test_cuagym_reward_gsettings_reads_agent_user_dconf` sets a gsettings
     key through the user session, then reads it back under the SAME identity
     ``reward.py`` runs under (uid=1000 + the user session bus) and asserts the
     reward read sees the agent-user's value, not root's.

Note on ``runuser`` in this file: it is TEST-HARNESS plumbing only. These tests
reach into the container with ``docker exec``, which lands as **root**, whereas the
exec-stdio server the production path uses is already the desktop user. ``_SESSION``
exists purely to put a probe command where the server already stands; it does not
mirror any production wrapper (``git grep runuser -- lite/`` returns nothing).

Both mirror the harness in
``tests/gym/envs/lite/cuagym/test_cuagym_parity.py`` exactly: they boot the shared
``cua-lite/lite.cuagym:latest`` container with ``docker run`` and SKIP cleanly
when the image is absent (they never hard-fail on a missing image).

Run (needs a fresh ``cua-lite/lite.cuagym:latest``)::

    uv run pytest -m live tests/gym/envs/lite/cuagym/test_cuagym_live_uid.py

Collect-only (no image / no boot required)::

    uv run pytest tests/gym/envs/lite/cuagym/test_cuagym_live_uid.py --collect-only -q
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid

import pytest

pytestmark = pytest.mark.live

_IMAGE = os.environ.get("LITE_CUAGYM_TEST_IMAGE", "cua-lite/lite.cuagym:latest")

# Test-harness prefix that reproduces the exec-stdio server's identity from a
# ``docker exec`` (which lands as root): drop to the desktop user and attach to its
# session bus. Production needs neither half -- the server IS that user and
# ``op_run_command`` injects ``DBUS_SESSION_BUS_ADDRESS`` itself. ``$bus`` must be
# defined by the surrounding script via ``bus="$(cat /tmp/dbus-session-bus-address)"``.
_SESSION = (
    'runuser -u user -- env HOME=/home/user DISPLAY=:1 '
    'DBUS_SESSION_BUS_ADDRESS="$bus"'
)

# B1-family GUI apps to drive, with a launch command + a pgrep pattern. Only the
# ones whose binary is present are launched (``command -v``), so the test adapts
# to whatever the image ships -- exactly like the parity test keys off available
# capabilities. gnome-terminal (205 bundles, highest severity: a root terminal is
# an integrity break) is mandatory; at least one more must also launch.
_GUI_APPS: tuple[tuple[str, str, str, str], ...] = (
    # name, binary, launch command, pgrep -f pattern
    ("gnome-terminal", "gnome-terminal", "gnome-terminal", "gnome-terminal"),
    ("gedit", "gedit", "gedit /home/user", "gedit"),
    ("soffice", "soffice", "soffice --writer --norestore --nologo", "soffice"),
    ("thunderbird", "thunderbird", "thunderbird", "thunderbird"),
)

# A root-owned process whose ``comm`` is one of these is part of the harness's own
# launch prefix (``_SESSION``'s runuser, setsid, the shell), NOT the GUI app itself
# -- filtered before the ROOT_APP_COUNT assert. Anything else root-owned is a real
# root GUI process.
_WRAPPER_COMMS = frozenset(
    {"runuser", "setsid", "env", "bash", "sh", "dash", "su", "sudo", "nohup"}
)

_GS_KEY = "org.gnome.desktop.interface color-scheme"
_GS_TARGET = "'prefer-dark'"


def _image_present() -> bool:
    return shutil.which("docker") is not None and subprocess.run(
        ["docker", "image", "inspect", _IMAGE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def _exec(name: str, script: str, *, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run a bash snippet inside the booted container (login shell, like parity)."""
    return subprocess.run(
        ["docker", "exec", name, "bash", "-lc", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.fixture(scope="module")
def booted_container() -> str:
    """Boot ``lite.cuagym`` and yield the container name once GNOME is up as user.

    Mirrors ``test_cuagym_parity.py``'s ``booted_desktop`` fixture: same
    ``docker run`` flags, same readiness gate (``/tmp/gnome-ready`` +
    ``/tmp/lite-cuagym-ready`` + ``pgrep -u user -f gnome-shell``), same
    force-remove teardown.
    """
    if not _image_present():
        pytest.skip(f"{_IMAGE} is not built")

    name = f"lite-cuagym-uid-{uuid.uuid4().hex[:12]}"
    started = subprocess.run(
        [
            "docker",
            "run",
            "--detach",
            "--name",
            name,
            # match the env's configs/default.yaml (8GB); a GEGL/office op
            # peaks past 4g and the container is SIGKILLed mid-test.
            "--memory",
            "8g",
            "--cpus",
            "2",
            "-e",
            "VNC_RESOLUTION=1920x1080",
            _IMAGE,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert started.returncode == 0, started.stderr

    try:
        deadline = time.monotonic() + 180
        last = subprocess.CompletedProcess([], 1, "", "desktop probe did not run")
        while time.monotonic() < deadline:
            last = _exec(
                name,
                (
                    "test -f /tmp/gnome-ready && "
                    "test -f /tmp/lite-cuagym-ready && "
                    "pgrep -u user -f gnome-shell >/dev/null"
                ),
                timeout=15,
            )
            if last.returncode == 0:
                break
            time.sleep(2)
        else:
            logs = subprocess.run(
                ["docker", "logs", name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            pytest.fail(
                "lite.cuagym desktop did not become ready:\n"
                f"{last.stdout}\n{last.stderr}\n{logs.stdout}\n{logs.stderr}"
            )
        yield name
    finally:
        subprocess.run(
            ["docker", "rm", "--force", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )


def _available_apps(name: str) -> list[tuple[str, str, str, str]]:
    """Return the subset of ``_GUI_APPS`` whose binary exists on the user PATH."""
    present: list[tuple[str, str, str, str]] = []
    for spec in _GUI_APPS:
        app, binary, _launch, _pat = spec
        probe = _exec(name, f'command -v {binary} >/dev/null 2>&1 && echo yes || echo no')
        if probe.stdout.strip() == "yes":
            present.append(spec)
    return present


def _enumerate_owners(name: str, app: str, pattern: str) -> list[tuple[str, str, str]]:
    """Return ``(pid, owner, comm)`` for every process matching ``pattern``."""
    script = (
        f"for pid in $(pgrep -f {pattern!r} 2>/dev/null); do "
        'line="$(ps -o user=,comm= -p $pid 2>/dev/null)"; '
        '[ -n "$line" ] && echo "$pid|$line"; '
        "done"
    )
    out = _exec(name, script).stdout
    rows: list[tuple[str, str, str]] = []
    for raw in out.splitlines():
        raw = raw.strip()
        if not raw or "|" not in raw:
            continue
        pid, _, rest = raw.partition("|")
        fields = rest.split()
        if not fields:
            continue
        owner = fields[0]
        comm = fields[-1]
        rows.append((pid.strip(), owner, comm))
    return rows


def test_cuagym_gui_apps_launch_as_user_not_root(booted_container: str) -> None:
    """B1 GUI apps launched under the server's identity run as ``user``, never root.

    Durable replacement for the ephemeral GUI-DRIVES-AS-USER probe: launch each
    available B1 app as the desktop user (the identity the exec-stdio server already
    has), enumerate its PIDs, filter the harness's own root-owned launch prefix, and
    require ``ROOT_APP_COUNT == 0``.
    """
    name = booted_container
    apps = _available_apps(name)
    app_names = [a[0] for a in apps]
    assert "gnome-terminal" in app_names, (
        f"gnome-terminal binary missing in {_IMAGE}; available: {app_names}"
    )
    assert len(apps) >= 2, (
        "need gnome-terminal plus at least one of gedit/soffice/thunderbird; "
        f"available: {app_names}"
    )

    patterns = "|".join(pat for _n, _b, _l, pat in apps)
    try:
        # Launch each app as uid=1000 on the user session bus -- the same identity
        # and session a cuagym setup command already has when it launches a GUI.
        for _n, _b, launch, _pat in apps:
            _exec(
                name,
                (
                    'bus="$(cat /tmp/dbus-session-bus-address)"; '
                    f"{_SESSION} setsid {launch} </dev/null >/dev/null 2>&1 &"
                ),
            )
        # Give the apps (esp. soffice / gnome-terminal-server dbus activation)
        # time to spawn their real processes.
        time.sleep(12)

        launched: dict[str, list[tuple[str, str, str]]] = {}
        root_offenders: list[str] = []
        for app, _b, _l, pat in apps:
            rows = _enumerate_owners(name, app, pat)
            # Drop the harness's own root-owned launch prefix (runuser/setsid/
            # shell/env); what remains root-owned is a genuine root GUI process ==
            # a regression.
            real = [
                (pid, owner, comm)
                for pid, owner, comm in rows
                if not (owner == "root" and comm in _WRAPPER_COMMS)
            ]
            launched[app] = real
            for pid, owner, comm in real:
                if owner == "root":
                    root_offenders.append(f"{app}: pid={pid} comm={comm} owner=root")

        # gnome-terminal + at least one more must actually have produced a real
        # (non-prefix) process, else the gate proved nothing.
        produced = [app for app, rows in launched.items() if rows]
        assert "gnome-terminal" in produced, (
            "gnome-terminal launched no GUI process; "
            f"enumeration: { {k: v for k, v in launched.items()} }"
        )
        assert len(produced) >= 2, (
            "expected >=2 B1 apps to launch a real process; "
            f"produced={produced} enumeration={ {k: v for k, v in launched.items()} }"
        )

        # Every non-shim GUI process must be owned by ``user``.
        assert not root_offenders, (
            "root-owned B1 GUI process(es) detected -- GUI-DRIVES-AS-USER "
            f"invariant broken (ROOT_APP_COUNT={len(root_offenders)}): "
            + "; ".join(root_offenders)
        )
        for app, rows in launched.items():
            for pid, owner, comm in rows:
                assert owner == "user", (
                    f"{app} pid={pid} comm={comm} owner={owner!r}, expected 'user'"
                )
    finally:
        _exec(
            name,
            "pkill -u user -f 'gnome-terminal|gedit|soffice|thunderbird' || true; "
            f"pkill -f {patterns!r} || true",
        )


def test_cuagym_reward_gsettings_reads_agent_user_dconf(booted_container: str) -> None:
    """A reward-style ``gsettings get`` reads the AGENT USER's dconf, not root's.

    Durable replacement for the ephemeral B4 probe: the agent sets
    ``color-scheme`` through the user session, then a reward-style read under the
    SAME identity reward.py runs under (uid=1000 + the user session bus) must see
    the agent's value.
    """
    name = booted_container

    # Capture the original value so we can restore it (self-cleaning).
    orig_res = _exec(
        name,
        f'bus="$(cat /tmp/dbus-session-bus-address)"; {_SESSION} gsettings get {_GS_KEY}',
    )
    assert orig_res.returncode == 0, (
        f"could not read original {_GS_KEY}: {orig_res.stdout}\n{orig_res.stderr}"
    )
    original = orig_res.stdout.strip()

    try:
        # Agent sets the key from inside the user session.
        set_res = _exec(
            name,
            (
                'bus="$(cat /tmp/dbus-session-bus-address)"; '
                f"{_SESSION} gsettings set {_GS_KEY} {_GS_TARGET}"
            ),
        )
        assert set_res.returncode == 0, f"gsettings set failed: {set_res.stderr}"

        # Prove the probe really is uid=1000 (the desktop user) — i.e. that it
        # stands exactly where the exec-stdio server already stands.
        id_res = _exec(
            name,
            f'bus="$(cat /tmp/dbus-session-bus-address)"; {_SESSION} id',
        )
        assert id_res.returncode == 0, id_res.stderr
        assert "uid=1000(user)" in id_res.stdout, (
            f"reward channel is not uid=1000(user): {id_res.stdout!r}"
        )

        # Reward-style read under the SAME identity: it must see the agent's value.
        reward_res = _exec(
            name,
            f'bus="$(cat /tmp/dbus-session-bus-address)"; {_SESSION} gsettings get {_GS_KEY}',
        )
        assert reward_res.returncode == 0, reward_res.stderr
        reward_value = reward_res.stdout.strip()
        assert reward_value == _GS_TARGET, (
            "reward-style read did not observe the agent-user's dconf value: "
            f"got {reward_value!r}, expected {_GS_TARGET!r}"
        )

        # Contrast: root (no user DBus) must NOT observe the agent's value -- this
        # is the exact wrong-reward bug that running the server AS the desktop user
        # prevents. Root's dconf is a separate profile, so it reads the schema
        # default, not 'prefer-dark'. Only assert when root produced a parseable value.
        root_res = _exec(
            name,
            f"env -u DBUS_SESSION_BUS_ADDRESS HOME=/root gsettings get {_GS_KEY}",
        )
        root_value = root_res.stdout.strip()
        if root_res.returncode == 0 and root_value:
            assert root_value != _GS_TARGET, (
                "root read the agent-user's dconf value -- running the reward as "
                "the desktop user would not have prevented the wrong-reward bug: "
                f"root read {root_value!r}"
            )
    finally:
        _exec(
            name,
            (
                'bus="$(cat /tmp/dbus-session-bus-address)"; '
                f"{_SESSION} gsettings set {_GS_KEY} {original}"
            ),
        )
