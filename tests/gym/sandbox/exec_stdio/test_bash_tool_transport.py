#!/usr/bin/env python3
"""Transport tests for the ``bash`` tool over the exec-stdio channel (section 3.10).

The ``bash`` tool is a persistent shell reached over the SAME sandbox-family
``docker exec -i`` transport as every other op: :class:`ExecStdioSession`
(``lite/gym/sandbox/exec_stdio/client.py``) frames each request/response as ONE
JSON line — so the trailing ``\\n`` IS the frame *sentinel* that delimits a
command's output+exit-status from the next frame. ``_roundtrip`` is the real
parse function; ``op_run_command`` (server side) is the shell backend.

This file is split into two layers, matching the repo's live convention
(``pyproject.toml`` de-selects ``-m 'not live and not stress'`` by default):

* HERMETIC (default, green now) — drive the REAL parse / timeout / respawn logic
  with fake procs + scripted ``StreamReader``s. No docker, no X, no sleeping.

* LIVE (``@pytest.mark.live`` on the class → excluded by default) — need a
  running sandbox container. They characterize the section 3.10 persistent-shell target
  (cd/export persistence) and the section 3.2 real-feedback contract. Point them at a
  container with ``LITE_BASH_TEST_CONTAINER=<name>``; without it they skip.

Run the HERMETIC layer (live deselected by the default marker filter):
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \\
        tests/gym/sandbox/exec_stdio/test_bash_tool_transport.py -p no:cacheprovider -q
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import signal

import pytest

from lite.gym.sandbox.exec_stdio import agent_shell as agent_shell_mod
from lite.gym.sandbox.exec_stdio import client, server
from lite.gym.sandbox.exec_stdio import session as session_mod
from lite.gym.sandbox.exec_stdio.client import (
    DEFAULT_CALL_TIMEOUT,
    DEFAULT_RUN_COMMAND_TIMEOUT,
    AgentShellSession,
    ExecStdioInterface,
    ExecStdioSession,
)

# =============================================================================
# HERMETIC test doubles — a live-looking proc backed by a scripted stdout, so
# ``_roundtrip``/``call`` run against real frames with no subprocess/docker/X.
# (Same shape the sibling test_exec_stdio.py uses for its binary-frame tests.)
# =============================================================================

class _FakeStdin:
    """Swallows the request line — these tests script the *responses*."""

    def write(self, _data: bytes) -> None:
        pass

    async def drain(self) -> None:
        pass


class _FakeProc:
    """A live-looking (``returncode is None``) proc over a scripted stdout."""

    returncode = None

    def __init__(self, stdout) -> None:
        self.stdin = _FakeStdin()
        self.stdout = stdout


def _reader_with(*lines: bytes) -> asyncio.StreamReader:
    r = asyncio.StreamReader()
    for ln in lines:
        r.feed_data(ln)
    r.feed_eof()
    return r


def _frame(rid: int, result) -> bytes:
    """The EXACT bytes server.py::main emits for a JSON-result op: one
    ``json.dumps`` object terminated by the ``\\n`` frame sentinel. Building the
    frame through ``json.dumps`` uses the same escaping the server does, so a
    round-trip through it exercises the real wire contract."""
    return (json.dumps({"id": rid, "ok": True, "result": result}) + "\n").encode()


# =============================================================================
# section 3.10 sentinel framing / parse — drive the REAL parser (``_roundtrip`` via
# ``call``). The ``\n`` line terminator is the frame sentinel; a run_command's
# {stdout, stderr, returncode} rides one line and must be recovered exactly,
# even when the command's OWN output contains the sentinel byte + a whole
# lookalike response frame (JSON escaping is the robustness contract).
# =============================================================================

async def test_frame_parse_extracts_output_and_exit_status():
    """A single run_command frame parses to exactly its {stdout, stderr,
    returncode} — the sentinel-delimited output and exit status, nothing else."""
    s = ExecStdioSession("fake", _local=True)
    s._proc = _FakeProc(_reader_with(
        _frame(1, {"stdout": "hello\n", "stderr": "oops\n", "returncode": 7}),
    ))

    res = await s.call("run_command", cmd="echo hello; echo oops 1>&2; exit 7")
    assert res == {"stdout": "hello\n", "stderr": "oops\n", "returncode": 7}


async def test_frame_survives_sentinel_like_output_no_desync():
    """A command whose stdout itself contains the ``\\n`` frame sentinel AND a
    complete lookalike response line must NOT split or spoof the frame: JSON
    escaping keeps it on one physical line, the parser returns the byte-exact
    output, and the NEXT op still reads its OWN frame (no desync / no spoof)."""
    s = ExecStdioSession("fake", _local=True)
    # Output packed with the sentinel byte and a forged "response" for id 1.
    poison = 'line1\n{"id": 1, "ok": true, "result": "SPOOFED"}\nline3\n'
    s._proc = _FakeProc(_reader_with(
        _frame(1, {"stdout": poison, "stderr": "", "returncode": 0}),
        _frame(2, {"stdout": "next\n", "stderr": "", "returncode": 0}),
    ))

    res = await s.call("run_command", cmd="printf ...")
    # Exact recovery — the embedded newlines/braces did not truncate the output,
    # and the forged inner frame was NOT mistaken for the response.
    assert res == {"stdout": poison, "stderr": "", "returncode": 0}
    # The forged frame left no stray bytes in the pipe: op #2 reads cleanly.
    res2 = await s.call("run_command", cmd="echo next")
    assert res2["stdout"] == "next\n"


# =============================================================================
# section 3.12 timeout-ordering invariant — the host readline budget MUST dominate the
# per-command server budget, or the client bails while the shell is still
# running, the reply lands on a dead pipe, and (run_command being non-retryable)
# the caller sees a transport failure for a command that actually ran. This is
# the exact hazard ``ExecStdioInterface.type_text`` already closes by passing an
# explicit dominating ``timeout`` to ``call()``.
# =============================================================================

#: client.py::_roundtrip waits ``readline`` for ``timeout + 10``.
_HOST_READLINE_CUSHION_S = 10.0
#: server.py::op_run_command's fallback subprocess budget.
_SERVER_RUN_COMMAND_DEFAULT_S = server.DEFAULT_RUN_COMMAND_TIMEOUT


async def test_default_run_command_host_budget_dominates_server_budget():
    """Drive the REAL ``run_command`` (mirroring the type_text budget test): the
    host readline budget it induces must be >= the server's per-command budget."""
    captured: dict = {}

    class _StubSession:
        _timeout = DEFAULT_CALL_TIMEOUT

        async def call(self, op, **kwargs):
            captured["op"] = op
            captured["kwargs"] = kwargs
            # run_command reads .stdout/.stderr/.returncode off the result.
            return {"stdout": "", "stderr": "", "returncode": 0}

    iface = ExecStdioInterface(_StubSession())  # type: ignore[arg-type]
    await iface.run_command("sleep 200")  # a legitimately long-but-bounded shell

    assert captured["op"] == "run_command"
    assert captured["kwargs"]["timeout"] == DEFAULT_RUN_COMMAND_TIMEOUT
    assert DEFAULT_RUN_COMMAND_TIMEOUT == _SERVER_RUN_COMMAND_DEFAULT_S
    # call(): timeout = float(kwargs.get("timeout") or self._timeout) — coerced
    # there, outside the retry try; _roundtrip's readline then waits that + the
    # fixed cushion.
    host_call_timeout = float(captured["kwargs"].get("timeout") or DEFAULT_CALL_TIMEOUT)
    host_readline_budget = host_call_timeout + _HOST_READLINE_CUSHION_S
    assert host_readline_budget >= _SERVER_RUN_COMMAND_DEFAULT_S, (
        f"host readline budget {host_readline_budget}s does not dominate the "
        f"server's {_SERVER_RUN_COMMAND_DEFAULT_S}s run_command budget — a "
        "command running longer than the host wait fails on a dead pipe"
    )


async def test_explicit_run_command_timeout_preserves_host_server_parity():
    """An explicit per-command timeout must be passed through to the server and
    still leave the host readline budget later than the server's subprocess
    budget."""
    captured: dict = {}

    class _StubSession:
        _timeout = DEFAULT_CALL_TIMEOUT

        async def call(self, op, **kwargs):
            captured["op"] = op
            captured["kwargs"] = kwargs
            return {"stdout": "", "stderr": "", "returncode": 0}

    iface = ExecStdioInterface(_StubSession())  # type: ignore[arg-type]
    await iface.run_command("sleep 4", timeout=4)

    assert captured["op"] == "run_command"
    assert captured["kwargs"]["timeout"] == 4
    server_budget = float(captured["kwargs"]["timeout"])
    host_readline_budget = server_budget + _HOST_READLINE_CUSHION_S
    assert host_readline_budget >= server_budget


async def test_run_command_rejects_non_positive_timeout():
    class _StubSession:
        async def call(self, op, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("run_command should reject before transport call")

    iface = ExecStdioInterface(_StubSession())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timeout must be positive"):
        await iface.run_command("sleep 0", timeout=0)


async def test_agent_shell_persists_cd_and_export_locally():
    shell = AgentShellSession("local", _local=True)
    try:
        await shell.run("cd /tmp")
        pwd = await shell.run("pwd")
        assert pwd.output.strip() == "/tmp"

        await shell.run("export LITE_AGENT_SHELL_TEST=ok")
        echo = await shell.run('printf %s "$LITE_AGENT_SHELL_TEST"')
        assert echo.output.strip() == "ok"
    finally:
        await shell.close()


async def test_agent_shell_returns_real_output_and_exit_status_locally():
    shell = AgentShellSession("local", _local=True)
    try:
        res = await shell.run("printf hello; false")
        assert res.output == "hello"
        assert res.returncode == 1
        assert res.output != "ok"
    finally:
        await shell.close()


async def test_agent_shell_timeout_closes_desynced_shell_locally():
    shell = AgentShellSession("local", _local=True)
    try:
        with pytest.raises(agent_shell_mod.AgentShellTimeout) as excinfo:
            await shell.run("sleep 1", timeout=0.01)
        assert excinfo.value.partial == ""
        res = await shell.run("printf ok")
        assert res.output == "ok"
        assert res.returncode == 0
    finally:
        await shell.close()


async def test_agent_shell_timeout_keeps_unterminated_partial_locally():
    shell = AgentShellSession("local", _local=True)
    try:
        with pytest.raises(agent_shell_mod.AgentShellTimeout) as excinfo:
            await shell.run("printf partial-timeout; sleep 30", timeout=1.0)
        assert excinfo.value.partial == "partial-timeout"
    finally:
        await shell.close()


async def test_agent_shell_immediate_timeout_does_not_reuse_previous_partial_locally():
    shell = AgentShellSession("local", _local=True)
    try:
        previous = await shell.run("printf previous")
        assert previous.output == "previous"

        with pytest.raises(agent_shell_mod.AgentShellTimeout) as excinfo:
            await shell.run("printf never-read", timeout=0)
        assert excinfo.value.partial == ""
    finally:
        await shell.close()


async def test_agent_shell_in_command_death_keeps_unterminated_partial_locally():
    shell = AgentShellSession("local", _local=True)
    try:
        with pytest.raises(agent_shell_mod.AgentShellDied) as excinfo:
            await shell.run("set -e; printf partial-death; false; echo unreachable")
        assert excinfo.value.partial == "partial-death"
        res = await shell.run("printf respawned")
        assert res.output == "respawned"
        assert res.returncode == 0
    finally:
        await shell.close()


async def test_agent_shell_shell_options_do_not_leak_between_calls_locally():
    shell = AgentShellSession("local", _local=True)
    try:
        first = await shell.run("set -u; set -o pipefail; printf armed")
        assert first.output == "armed"
        assert first.returncode == 0

        nounset = await shell.run('printf "%s" "$LITE_AGENT_UNSET_VAR"')
        assert nounset.output == ""
        assert nounset.returncode == 0

        pipefail = await shell.run('false | true; printf "rc=$?"')
        assert pipefail.output == "rc=0"
        assert pipefail.returncode == 0
    finally:
        await shell.close()


async def test_agent_shell_restore_state_cannot_be_overwritten_by_command_locally():
    shell = AgentShellSession("local", _local=True)
    try:
        res = await shell.run(
            "__lite_set_restore='exit 42'; "
            "__lite_shopt_restore='printf hacked'; "
            "set(){ printf fake-set; }; "
            "shopt(){ printf fake-shopt; }; "
            "printf body"
        )
        assert res.output == "body"
        assert res.returncode == 0

        after = await shell.run("unset -f set shopt; printf ok")
        assert after.output == "ok"
        assert after.returncode == 0
    finally:
        await shell.close()


async def test_agent_shell_set_options_survive_command_builtin_function_shadowing():
    shell = AgentShellSession("local", _local=True)
    try:
        res = await shell.run(
            "command(){ :; }; "
            "builtin(){ :; }; "
            "enable(){ :; }; "
            "set -u; "
            "printf body"
        )
        assert res.output == "body"
        assert res.returncode == 0

        after = await shell.run('printf "%s" "$LITE_AGENT_STILL_UNSET"')
        assert after.output == ""
        assert after.returncode == 0
    finally:
        await shell.close()


# =============================================================================
# section 3.10 AgentShell hardening — the verified holes, each pinned by a hermetic
# gate. The local layer runs REAL host bash (``--noprofile --norc``); the
# container layer fakes ``docker exec`` so the reap argv is observable offline.
# =============================================================================

async def test_agent_shell_oversized_single_line_truncates_and_survives():
    """One line longer than the output budget must TRUNCATE, not kill the
    session: reading with ``readline()`` bounded by the stream limit raises
    ``ValueError`` (and clears the buffer, eating the sentinel), which used to
    take the whole shell down with it."""
    shell = AgentShellSession("local", _local=True, output_limit=50_000)
    try:
        res = await shell.run("head -c 400000 /dev/zero | tr '\\0' 'x'")
        assert res.returncode == 0
        assert res.output.endswith("[truncated]\n")
        assert len(res.output) < 60_000, "output budget was not enforced"
        assert res.output.startswith("x" * 1000)
        # The session is still usable — and still stateful.
        await shell.run("export LITE_TRUNC_SURVIVED=yes")
        after = await shell.run('printf %s "$LITE_TRUNC_SURVIVED"')
        assert after.output == "yes"
        assert after.returncode == 0
    finally:
        await shell.close()


async def test_agent_shell_nonce_in_output_does_not_truncate(monkeypatch):
    """The per-command sentinel is matched LINE-ANCHORED. A command whose output
    replays the marker MID-LINE (nonce pinned here so the command can forge it)
    must not end the frame early: the full output and the real exit status
    survive, and the next command still reads its own frame."""
    monkeypatch.setattr(agent_shell_mod.secrets, "token_hex", lambda _n: "b" * 24)
    shell = AgentShellSession("local", _local=True)
    try:
        nonce = "__LITE_AGENT_SHELL_DONE_" + "b" * 24 + "__"
        res = await shell.run(f"printf 'lead {nonce}:0 trail\\nreal\\n'; (exit 3)")
        assert res.output == f"lead {nonce}:0 trail\nreal\n"
        assert res.returncode == 3, "a mid-line nonce hijacked the exit status"
        assert (await shell.run("printf next")).output == "next"
    finally:
        await shell.close()


async def test_agent_shell_cancellation_reaps_the_in_container_command():
    """``CancelledError`` is a ``BaseException``, so an ``except Exception`` arm
    would miss it and leave the command running. Cancellation arrives from
    ``StepTimeoutWrapper``'s ``wait_for``, env-server task cancellation and
    client disconnect — the common path, not an exotic one."""
    shell = AgentShellSession("local", _local=True)
    try:
        started = await shell.run("sleep 300 & echo $!")
        child = int(started.output.strip())
        os.kill(child, 0)  # alive before the cancellation

        task = asyncio.ensure_future(shell.run("sleep 300"))
        await asyncio.sleep(0.2)  # let the command reach the shell
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # the cleanup is shielded so a re-delivered cancel cannot abort the reap
        assert shell._cleanup_task is not None
        await asyncio.wait_for(asyncio.shield(shell._cleanup_task), timeout=10)

        for _ in range(50):
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                break
            await asyncio.sleep(0.1)
        else:
            os.kill(child, signal.SIGKILL)
            pytest.fail("the background child outlived the CANCELLED command — "
                        "the in-container process group was not reaped")
    finally:
        await shell.close()


async def test_agent_shell_after_cancellation_reads_its_own_output():
    """A session left parked mid-frame makes the NEXT run() return the cancelled
    command's residual output."""
    shell = AgentShellSession("local", _local=True)
    try:
        task = asyncio.ensure_future(shell.run("sleep 300; echo RESIDUAL"))
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(asyncio.shield(shell._cleanup_task), timeout=10)

        res = await shell.run("printf fresh")
        assert res.output == "fresh"
        assert res.returncode == 0
    finally:
        await shell.close()


async def test_agent_shell_bare_exit_is_reported_then_respawned():
    """RESIDUAL, pinned: the command runs in a brace group in THIS shell (a real
    subshell would isolate ``exit`` but lose ``cd``/``export`` persistence), so a
    bare ``exit`` still ends the session. The contract is that it surfaces as an
    error and the NEXT call transparently gets a fresh shell — with the previous
    shell state gone."""
    shell = AgentShellSession("local", _local=True)
    try:
        await shell.run("export LITE_PRE_EXIT=set")
        with pytest.raises(RuntimeError, match="exited before command completed"):
            await shell.run("exit 0")
        res = await shell.run('printf %s "${LITE_PRE_EXIT:-gone}"')
        assert res.output == "gone"
    finally:
        await shell.close()


async def test_agent_shell_spawn_banner_does_not_leak_into_first_result(monkeypatch):
    """``bash -l`` prints /etc/profile + MOTD bytes at spawn. The drain sentinel
    must consume them, or they prepend to the FIRST command's output."""
    real_exec = asyncio.create_subprocess_exec

    async def banner_exec(*_argv, **kwargs):
        # Same local shell, behind a login-style banner (last line deliberately
        # unterminated, the shape a shell prompt leaves behind).
        return await real_exec(
            "bash", "-c",
            "echo 'Welcome to Ubuntu'; printf 'no-newline-prompt> '; "
            "exec bash --noprofile --norc",
            **kwargs,
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", banner_exec)
    shell = AgentShellSession("local", _local=True)
    try:
        res = await shell.run("printf first-command-output")
        assert res.output == "first-command-output"
        assert "Ubuntu" not in res.output and "prompt" not in res.output
    finally:
        await shell.close()


async def test_agent_shell_scrubs_env_toolchain_off_login_path(monkeypatch):
    """section 3.10 PATH decision: keep ``bash -l`` (a real terminal) but never let the
    ENV exec-stdio toolchain (``/opt/env``, prepended to the exec-stdio
    ``server.py``'s own PATH) reach the agent. The
    login PATH is scrubbed in-shell and asserted host-side."""
    monkeypatch.setenv("PATH", "/opt/env/venv/bin:/opt/env/bin:" + os.environ["PATH"])
    shell = AgentShellSession("local", _local=True)
    try:
        res = await shell.run('printf %s "$PATH"')
        assert "/opt/env" not in res.output
        assert "/opt/env" not in shell.login_path
        assert "/usr/bin" in res.output, "the scrub must not empty PATH"
    finally:
        await shell.close()


async def test_agent_shell_timeout_reaps_local_process_group():
    """A timed-out command leaves a live process tree behind unless the shell's
    process group is reaped — killing only the host client never reaches it."""
    shell = AgentShellSession("local", _local=True)
    try:
        started = await shell.run("sleep 300 & echo $!")
        child = int(started.output.strip())
        os.kill(child, 0)  # alive before the timeout
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            await shell.run("sleep 300", timeout=0.05)
        # The pid may linger for a moment as a zombie until init reaps it.
        for _ in range(50):
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                break
            await asyncio.sleep(0.1)
        else:
            os.kill(child, signal.SIGKILL)
            pytest.fail("the background child outlived the timed-out command "
                        "— the in-container process group was not reaped")
    finally:
        await shell.close()


# --- container path: a fake ``docker exec`` so the reap argv is observable ----

class _FakeShellStdin:
    """Answers the spawn handshake the way a real login shell would, then goes
    silent — the exact shape of a hung/timed-out command."""

    def __init__(self, proc) -> None:
        self._proc = proc
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.writes.append(data)
        m = re.search(rb"__LITE_AGENT_SHELL_READY_[0-9a-f]+__", data)
        if m:
            self._proc.stdout.feed_data(
                b"Welcome to the container\n" + m.group(0)
                + f":{self._proc.shell_pid}:{self._proc.shell_pgid}"
                  f":{self._proc.shell_path}".encode() + b"\n"
            )
    def is_closing(self) -> bool:
        return False

    def close(self) -> None:
        pass

    async def drain(self) -> None:
        pass


class _FakeShellProc:
    def __init__(self, *, pid: int = 4242, pgid: int | None = 4242,
                 path: str = "/usr/bin:/bin") -> None:
        self.shell_pid = pid
        self.shell_pgid = pgid if pgid is not None else ""
        self.shell_path = path
        self.stdout = asyncio.StreamReader()
        self.stdin = _FakeShellStdin(self)
        self.returncode = None

    def terminate(self) -> None:
        self.returncode = -15
        self.stdout.feed_eof()

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode if self.returncode is not None else 0


class _ExitedProc:
    """A ``docker exec ... kill`` that already finished."""

    returncode = 0

    async def wait(self) -> int:
        return 0

    def kill(self) -> None:
        pass


def _fake_docker(monkeypatch, shell_proc):
    """Route every ``create_subprocess_exec`` through a recorder: the first call
    is the shell spawn, later ones are the reap execs."""
    argvs: list[list[str]] = []

    async def fake_exec(*argv, **_kwargs):
        argvs.append(list(argv))
        return shell_proc if len(argvs) == 1 else _ExitedProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    # Both channels share the reap on the session base, so the grace lives there.
    monkeypatch.setattr(session_mod, "_REAP_GRACE_S", 0.0)
    return argvs


async def test_agent_shell_timeout_issues_docker_kill_of_shell_pgid(monkeypatch):
    """Killing the HOST ``docker exec`` client does not signal anything inside
    the container. On timeout the shell's in-container process GROUP must be
    SIGTERM'd then SIGKILL'd over a separate ``docker exec``, or every timed-out
    command orphans a ``bash -l`` tree for the container's lifetime."""
    proc = _FakeShellProc(pid=4242, pgid=4242)
    argvs = _fake_docker(monkeypatch, proc)

    shell = AgentShellSession("cont-1", exec_user="user")
    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        await shell.run("sleep 300", timeout=0.05)

    spawn = argvs[0]
    assert spawn[:6] == ["docker", "exec", "-i", "-u", "user", "cont-1"]
    assert "setsid" in spawn[-1], "the shell must lead its own process group"
    kills = [a for a in argvs[1:] if a[0] == "docker"]
    assert len(kills) == 2, f"expected TERM then KILL, got {kills}"
    for expected, argv in zip(("TERM", "KILL"), kills):
        assert argv[:6] == ["docker", "exec", "-u", "user", "cont-1", "sh"]
        assert f"kill -{expected} -4242" in argv[-1], argv
    assert proc.returncode == -15, "the host client must be closed before the reap"


async def test_agent_shell_reap_never_group_kills_a_shared_group(monkeypatch):
    """If ``setsid`` is unavailable the shell is NOT its own group leader, and a
    ``kill -<pgid>`` would hit unrelated container processes. Degrade to a
    single-pid kill instead."""
    proc = _FakeShellProc(pid=4242, pgid=17)  # pgid != pid → not a leader
    argvs = _fake_docker(monkeypatch, proc)

    shell = AgentShellSession("cont-2", exec_user="user")
    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        await shell.run("sleep 300", timeout=0.05)

    kills = [a[-1] for a in argvs[1:] if a[0] == "docker"]
    assert kills == ["kill -TERM 4242 2>/dev/null || true",
                     "kill -KILL 4242 2>/dev/null || true"]


async def test_agent_shell_spawn_drains_container_banner(monkeypatch):
    """The container spawn handshake reports pid/pgid/PATH and swallows the
    login banner before any command runs."""
    proc = _FakeShellProc(pid=99, pgid=99)
    _fake_docker(monkeypatch, proc)

    shell = AgentShellSession("cont-3")
    await shell.start()
    assert (shell._pid, shell._pgid) == (99, 99)
    assert shell.login_path == "/usr/bin:/bin"
    setup = b"".join(proc.stdin.writes)
    assert b"exec 2>&1\n" in setup, "stderr must be merged IN-SHELL, not at the host pipe"
    assert b'cd "$HOME"' in setup


async def test_agent_shell_start_rejects_env_toolchain_on_agent_path(monkeypatch):
    """The section 3.10 PATH assertion: if the login shell still reports ``/opt/env``
    after the scrub, the env/agent separation is broken and the session must
    fail loudly instead of handing the agent the env toolchain."""
    proc = _FakeShellProc(path="/opt/env/venv/bin:/usr/bin")
    _fake_docker(monkeypatch, proc)

    shell = AgentShellSession("cont-4")
    with pytest.raises(RuntimeError, match="/opt/env"):
        await shell.start()
    assert not shell._alive()


# =============================================================================
# section 3.12 _respawn no-leak — if ``ExecStdioSession._respawn`` is just
# ``await self._spawn()`` it OVERWRITES ``self._proc`` without closing the old one
# first, orphaning the previous host ``docker exec`` process on every dead-pipe
# respawn. Inject a fake old proc and assert it was terminated before replacement.
# =============================================================================

class _RecordingProc:
    """A fake proc exposing exactly the surface ``close()`` touches, recording
    whether it was terminated (the leak signal).

    It also answers ONE ``ping`` frame, because a respawn re-identifies the fresh
    server to learn the pid/pgid the next reap aims at."""

    def __init__(self, events: list[str] | None = None, label: str = "proc",
                 *, pid: int = 4243) -> None:
        self._events = events
        self._label = label
        self.returncode = None
        self.terminated = False
        self.waited = False

        class _Stdin:
            def __init__(self) -> None:
                self.closed = False

            def is_closing(self) -> bool:
                return self.closed

            def close(self) -> None:
                self.closed = True

            def write(self, _data: bytes) -> None:
                pass

            async def drain(self) -> None:
                pass

        self.stdin = _Stdin()
        self.stdout = _reader_with(
            _frame(1, {"version": client.EXPECTED_SERVER_VERSION,
                       "pid": pid, "pgid": pid}),
        )

    def terminate(self) -> None:
        self.terminated = True
        if self._events is not None:
            self._events.append(f"{self._label}:terminate")

    def kill(self) -> None:
        self.terminated = True
        if self._events is not None:
            self._events.append(f"{self._label}:kill")

    async def wait(self) -> int:
        self.waited = True
        if self._events is not None:
            self._events.append(f"{self._label}:wait")
        self.returncode = -15
        return -15


async def test_respawn_closes_old_proc_before_replacement(monkeypatch):
    """The OLD proc must be terminated before ``_respawn`` installs the new one,
    otherwise every respawn orphans a host ``docker exec`` process."""
    s = ExecStdioSession("fake", _local=True)
    events: list[str] = []
    old = _RecordingProc(events, "old")
    s._proc = old

    async def fake_spawn():
        events.append("spawn")
        assert old.terminated
        assert old.waited
        # Simulate a real spawn: a fresh proc replaces self._proc.
        s._proc = _RecordingProc(events, "new", pid=7331)

    monkeypatch.setattr(s, "_spawn", fake_spawn)

    await s._respawn()

    assert s._proc is not old, "respawn must install a fresh proc"
    assert (s._pid, s._pgid) == (7331, 7331), (
        "the respawned server was not re-identified — its in-container tree "
        "would be unreapable for the rest of the session"
    )
    assert events[:3] == ["old:terminate", "old:wait", "spawn"]
    assert old.terminated, (
        "old docker-exec proc was overwritten without being terminated — "
        "host process leak on every respawn"
    )


async def test_respawn_retires_the_old_group_instead_of_reaping_it(monkeypatch):
    """A respawn must NOT kill the in-container process group; teardown must.

    ``call()`` respawns on ``asyncio.TimeoutError`` for EVERY op, and a host
    timeout means "no answer inside the margin", not "the work is wrong". Reaping
    on that path SIGKILLs a command that is merely SLOW — a ``setup_fn`` install,
    an evaluator getter, a chrome/sqlite config-flush mid-checkpoint — so a
    transport fault would surface as a wrong SCORE. The old identity is RETIRED,
    which is what keeps it reapable: dropping it would leave the old group alive
    for the container's lifetime with nothing left to aim at.
    """
    argvs: list[list[str]] = []
    procs: list[_RecordingProc] = []
    gens = iter(((8888, 1), (9999, 2)))  # (in-container pid, ping frame id)

    async def fake_exec(*argv, **_kwargs):
        argvs.append(list(argv))
        if "kill -" in argv[-1]:
            return _ExitedProc()
        pid, rid = next(gens)
        proc = _RecordingProc(pid=pid)
        proc.stdout = _reader_with(
            _frame(rid, {"version": client.EXPECTED_SERVER_VERSION,
                         "pid": pid, "pgid": pid}))
        procs.append(proc)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(session_mod, "_REAP_GRACE_S", 0.0)

    s = ExecStdioSession("cont-8", exec_user="user")
    await s.start()
    assert (s._pid, s._pgid) == (8888, 8888)

    await s._respawn()

    assert procs[0].terminated, "the old HOST docker exec client must still go"
    assert [a for a in argvs if "kill -" in a[-1]] == [], (
        "the respawn issued an in-container kill — the process group of a "
        "command that had not failed, only timed out, was destroyed"
    )
    assert (s._pid, s._pgid) == (9999, 9999)
    assert s._retired == [(8888, 8888)], (
        "the retired generation was dropped, not remembered — its group would "
        "be unreapable for the rest of the container's life"
    )

    await s.close()

    # ONE pair of signals covers both generations, so teardown does not get
    # slower per respawn.
    assert [a[-1] for a in argvs if "kill -" in a[-1]] == [
        "kill -TERM -8888 -9999 2>/dev/null || true",
        "kill -KILL -8888 -9999 2>/dev/null || true",
    ]
    assert s._retired == []


# =============================================================================
# The ENV channel's reap. Killing the host ``docker exec`` client signals nothing
# inside the container, so without a reap on ``close()`` every command the env
# side left running (a slow ``run_command``, a wedged ``_grab_xlib``) survives.
# Both channels drive the ONE reap on the session base; they differ only in how
# they LEARN the pid to aim it at.
# =============================================================================

async def test_exec_stdio_close_reaps_the_local_process_tree():
    """After ``close()``, nothing the session spawned is still running.

    The server puts itself in its own process group at startup, so the single
    group kill reaches the ``run_command`` child tree — which the host client's
    death never touches."""
    s = ExecStdioSession("local-reap", _local=True)
    await s.start()
    try:
        assert s._pid and s._pgid == s._pid, (
            "the server did not report an own-group identity — the reap would "
            "have nothing to aim at"
        )
        res = await s.call(
            "run_command", cmd="sleep 300 >/dev/null 2>&1 & echo $!")
        child = int(res["stdout"].strip())
        os.kill(child, 0)  # alive while the session is up
    finally:
        await s.close()

    # The pid may linger for a moment as a zombie until init reaps it.
    for _ in range(50):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.1)
    else:
        os.kill(child, signal.SIGKILL)
        pytest.fail("a command the session spawned outlived close() — the "
                    "in-container process group was not reaped")


async def test_exec_stdio_close_issues_docker_kill_of_server_pgid(monkeypatch):
    """Container path: the reap is a separate ``docker exec`` of
    ``kill -TERM/-KILL -<pgid>``, exactly as the agent shell's is, and it runs
    AFTER the host client is gone."""
    proc = _RecordingProc(pid=8888)
    argvs = _fake_docker(monkeypatch, proc)

    s = ExecStdioSession("cont-5", exec_user="user")
    await s.start()
    assert (s._pid, s._pgid) == (8888, 8888), "ping must carry the server identity"
    await s.close()

    assert proc.terminated, "the host client must be closed before the reap"
    kills = [a for a in argvs[1:] if a[0] == "docker"]
    assert len(kills) == 2, f"expected TERM then KILL, got {kills}"
    for expected, argv in zip(("TERM", "KILL"), kills):
        assert argv[:6] == ["docker", "exec", "-u", "user", "cont-5", "sh"]
        assert f"kill -{expected} -8888" in argv[-1], argv


async def test_exec_stdio_reap_is_skipped_when_the_server_reports_no_identity(
    monkeypatch,
):
    """An older baked server answers ``ping`` with the version alone. The reap
    then has no target and must be SKIPPED — never aimed at a guessed pid."""
    proc = _RecordingProc()
    proc.stdout = _reader_with(
        _frame(1, {"version": client.EXPECTED_SERVER_VERSION}))
    argvs = _fake_docker(monkeypatch, proc)

    s = ExecStdioSession("cont-6", exec_user="user")
    await s.start()
    assert (s._pid, s._pgid) == (None, None)
    await s.close()

    assert [a for a in argvs[1:] if a[0] == "docker"] == []


async def test_reap_refuses_a_nonpositive_pid_from_the_wire(monkeypatch):
    """``killpg(0, …)`` means "my OWN group" — the host harness. A 0 arriving
    from a foreign/garbage handshake body must be treated as no identity, not
    as a kill target."""
    proc = _RecordingProc()
    proc.stdout = _reader_with(
        _frame(1, {"version": client.EXPECTED_SERVER_VERSION, "pid": 0, "pgid": 0}))
    argvs = _fake_docker(monkeypatch, proc)

    s = ExecStdioSession("cont-7", exec_user="user")
    await s.start()
    await s.close()

    assert [a for a in argvs[1:] if a[0] == "docker"] == []


# =============================================================================
# LIVE layer — needs a running sandbox container. Class-scoped @pytest.mark.live
# de-selects these under the repo's default ``-m 'not live and not stress'``, so
# the file stays green offline while these characterize the real transport.
# Point them at a container with ``LITE_BASH_TEST_CONTAINER=<name>``.
# =============================================================================

_LIVE_CONTAINER = os.environ.get("LITE_BASH_TEST_CONTAINER")


@pytest.mark.live
class TestBashToolLive:
    """Real-container transport checks for the section 3.10 persistent bash shell.

    These characterize the redesign TARGET: cd/export must persist across
    sequential bash calls on one session, and a bash call must surface its real
    stdout rather than an ``"ok"`` placeholder (the section 3.2 ``info['tool_results']``
    gap). They will fail against the pre-redesign stateless ``run_command``,
    which is the point of pinning the target — and they never run by default.
    """

    async def _attach(self):
        if not _LIVE_CONTAINER:
            pytest.skip(
                "set LITE_BASH_TEST_CONTAINER=<running sandbox container> to run "
                "the live bash-transport tests"
            )
        return await client.attach(_LIVE_CONTAINER)

    async def test_cd_and_export_persist_across_calls(self):
        """Two sequential bash calls on ONE session share shell state: ``cd
        /tmp`` then ``pwd`` → ``/tmp``; ``export X=1`` then ``echo $X`` → ``1``."""
        handle = await self._attach()
        try:
            await handle.agent_shell.run("cd /tmp")
            pwd = await handle.agent_shell.run("pwd")
            assert pwd.output.strip() == "/tmp"

            await handle.agent_shell.run("export X=1")
            echo = await handle.agent_shell.run('printf %s "$X"')
            assert echo.output.strip() == "1"
        finally:
            await handle.stop()

    async def test_agent_shell_returns_real_stdout_not_ok_placeholder(self):
        """A bash call returns its ACTUAL stdout, not the ``"ok"`` placeholder —
        the real feedback the section 3.2 ``info['tool_results']`` contract owes."""
        handle = await self._attach()
        try:
            res = await handle.agent_shell.run("echo hello-from-bash")
            assert res.output.strip() == "hello-from-bash"
            assert res.output.strip() != "ok"
        finally:
            await handle.stop()
