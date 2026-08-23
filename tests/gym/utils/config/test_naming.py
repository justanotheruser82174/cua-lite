"""Container-name formatting + the direct-vs-server-mode name partition, and
the shell side of the same grammar (every ``envs/*/scripts/cleanup.sh``).

The cleanup.sh half EXECUTES each script under a stub ``docker`` (see the
SAFETY note below the first section) — it never asserts on shell source.

Run: uv run python -m pytest -n0 tests/gym/utils/config/test_naming.py
"""

from __future__ import annotations

import ast
import functools
import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from lite.gym.remote.reaper import _name_regex as reaper_name_regex
from lite.gym.utils.config.identity import EnvIdentity
from lite.gym.utils.config.naming import container_name_prefix, format_container_name


def test_config_identity_and_naming_current_owner_behavior() -> None:
    assert EnvIdentity(session_id="session-a").resolved_session_id() == "session-a"
    assert format_container_name(env_id="lite.osworld", suffix="x", session_id="s")


# ── cleanup.sh ⇄ container-name grammar ──────────────────────────────────────
#
# Every ``scripts/cleanup.sh`` that narrows by ``SESSION_ID`` must build its
# ``docker ps --filter name=`` segment the SAME way the producer builds the
# session segment of a real container name — otherwise the script silently
# reaps nothing (session contains a ``-`` or ``.``) or, worse, reaps across a
# forged ``-<port>-`` boundary. This used to be checked by asserting a literal
# shell substring against three hand-listed scripts; that gate could not see
# the other seven, and it pinned ONE spelling of the filter while the scripts
# in fact use three (``${x//…}``, ``sed``, ``tr -c``) — plus, until this gate
# landed, ``cua`` which lowercased and forgot to collapse at all.
#
# So: glob every cleanup.sh, EXECUTE it against an adversarial SESSION_ID with
# a stub ``docker`` on PATH, and compare the filter it really computed with
# what ``format_container_name`` really produces — both the session segment
# byte-for-byte AND, as a regex, against a whole container name the producer
# emits. No shell source is read.
#
# SAFETY: the scripts are destructive by design. They are run with a minimal
# environment whose PATH puts a stub dir FIRST and contains only ``os.defpath``
# after it, and every destructive binary a cleanup.sh can reach (docker, pkill,
# pgrep, kill, rm, rmdir, python) is shadowed by a no-op recorder. Text utils
# (sed/tr/printf/xargs/grep/realpath) stay real — they are what computes the
# sanitization under test. ``test_cleanup_script_sandbox_shadows_real_docker``
# asserts the shadowing actually took effect before any script is trusted.

#: A session id carrying every character class the producer must collapse: a
#: ``-`` (which would forge a ``lite-env-<port>-`` boundary), a ``.``, and
#: upper case (which some producers additionally lowercase).
_ADVERSARIAL_SESSION = "30100-Trial.x"
_CONTROL_SESSION = "controlsession"

#: cleanup.sh scripts that legitimately do NOT narrow by SESSION_ID — they scope
#: by ``CUA_LITE_ENV_SERVER_PORT`` (browsergym's per-scope service containers,
#: captcha's per-scope Flask procs + /tmp files) and own no session-named
#: container. Pinned so a NEW session-blind script has to be classified here.
_SESSION_BLIND_CLEANUP_SCRIPTS = frozenset({
    "lite/gym/envs/browsergym/scripts/cleanup.sh",
    "lite/gym/envs/captcha/scripts/cleanup.sh",
})

#: cleanup.sh scripts KNOWN not to sanitize SESSION_ID, exempted from the gate
#: below. **Empty, and that is the point**: the one entry this ever held --
#: ``lite/gym/envs/cua/scripts/cleanup.sh``, which lowercased but never collapsed
#: non-alnum, so ``SESSION_ID=my-run`` filtered ``my-run-...`` while the producer
#: had already emitted ``lite-env-my_run-...`` and it reaped NOTHING -- was FIXED
#: rather than grandfathered. The mechanism stays so a newly-discovered broken
#: script can be recorded here instead of silently deleted from the glob, but
#: ``test_no_cleanup_script_is_exempt_from_the_derivation`` below reddens the
#: moment it is used, so an entry has to be argued for rather than parked.
_UNSANITIZED_CLEANUP_SCRIPTS: frozenset[str] = frozenset()

#: The producer symbol every cleanup filter must agree with, and the fallback
#: producer for envs that mint no name of their own (lite.osworld / lite.cuagym
#: / lite.cuaworld all boot through the shared sandbox path).
_PRODUCER = "format_container_name"
_SHARED_PRODUCER = "lite/gym/sandbox/base.py"

#: Binaries a cleanup.sh may invoke that must never actually run here.
_STUBBED_BINARIES = ("docker", "pkill", "pgrep", "kill", "rm", "rmdir", "python", "python3")


def _cleanup_scripts() -> list[str]:
    repo = Path(__file__).resolve().parents[4]
    scripts = sorted(
        p.relative_to(repo).as_posix()
        for p in repo.glob("lite/gym/envs/**/scripts/cleanup.sh")
    )
    assert scripts, "discovery found no cleanup.sh — the glob or the env tree moved"
    return scripts


def _run_cleanup_script(script: str, session_id: str) -> tuple[str, ...]:
    """Execute ``script`` in the stub sandbox; return its ``docker ps`` filters.

    Returns every ``--filter name=<X>`` value the script handed to ``docker``,
    in call order. That is the script's real, computed intent — not a guess
    read off its source.
    """
    repo = Path(__file__).resolve().parents[4]
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        bindir = tmpdir / "bin"
        bindir.mkdir()
        log = tmpdir / "argv.log"
        # A stubbed ``python`` must still print SOMETHING: waa's cleanup does
        # `RUNTIME_ROOT="$(python config_value.py …)"` under `set -e`, so an
        # empty capture would abort the script before it reaches docker.
        fake_root = tmpdir / "fake-runtime-root"
        for name in _STUBBED_BINARIES:
            stub = bindir / name
            body = f'printf "%s\\n" "{name} $*" >> "$STUB_LOG"\n'
            if name.startswith("python"):
                body += f'printf "%s\\n" "{fake_root}"\n'
            stub.write_text(f"#!/bin/sh\n{body}exit 0\n")
            stub.chmod(0o755)
        env = {
            "PATH": f"{bindir}{os.pathsep}{os.defpath}",
            "HOME": str(tmpdir / "home"),
            "STUB_LOG": str(log),
            "SESSION_ID": session_id,
        }
        subprocess.run(
            ["bash", str(repo / script)],
            cwd=repo, env=env, capture_output=True, text=True, timeout=120,
        )
        calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return tuple(
        m.group(1)
        for line in calls if line.startswith("docker ps ")
        for m in re.finditer(r"--filter name=(\S+)", line)
    )


@functools.cache
def _cleanup_filters(script: str, session_id: str) -> tuple[str, ...]:
    return _run_cleanup_script(script, session_id)


def _producer_session_segment(session_id: str) -> str:
    """The session segment ``format_container_name`` puts in a real name.

    Read off the PRODUCER by building a name with no env_id/task_id and
    stripping the two segments we supplied — never by re-implementing the
    sanitizer here, which is precisely the drift the old literal-substring
    gate could not catch.
    """
    sentinel = "SUFFIXSENTINEL"
    name = format_container_name(env_id=None, suffix=sentinel, session_id=session_id)
    return name[len(container_name_prefix()):-len(f"-{sentinel}")]


@functools.cache
def _files_calling_producer(env_dir: Path) -> tuple[Path, ...]:
    """Every ``.py`` under ``env_dir`` that calls the producer. Dot-dirs are
    pruned: an env's ``.cache/`` holds downloaded task bundles (tens of
    thousands of files, hundreds of MB) that are not this repo's code."""
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(env_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__pycache__"]
        found.extend(
            path for name in filenames if name.endswith(".py")
            if _PRODUCER in (path := Path(dirpath) / name).read_text(
                encoding="utf-8", errors="ignore",
            )
        )
    return tuple(sorted(found))


def _lower_flags(path: Path) -> list[bool]:
    """For each producer call in ``path``: is its result ``.lower()``-ed?
    An ``ast`` walk, so an aliased import (``as _format_container_name``) and a
    call spread over six lines are seen exactly as the interpreter sees them."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {_PRODUCER} | {
        alias.asname
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        for alias in node.names if alias.name == _PRODUCER and alias.asname
    }
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in names
    ]
    lowered = {
        id(node.func.value) for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "lower" and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id in names
    }
    return [id(call) in lowered for call in calls]


@functools.cache
def _producer_lowercases(script: str) -> bool:
    """Does the env owning ``script`` lowercase the assembled container name?

    DERIVED from the producer's source, never pinned per env: osworld,
    osworld_2 and cua lowercase (a docker *image repository* must be lowercase
    and cua tags one), the other seven do not. A cleanup filter must follow its
    own env either way — lowercasing a producer that doesn't (or vice versa)
    makes the filter miss every mixed-case SESSION_ID, the same silent-no-op
    failure mode as forgetting to collapse ``-``.
    """
    repo = Path(__file__).resolve().parents[4]
    env_dir = (repo / script).parent.parent
    producers = _files_calling_producer(env_dir) or (repo / _SHARED_PRODUCER,)
    flags = {flag for path in producers for flag in _lower_flags(path)}
    assert len(flags) == 1, (
        f"cannot derive case-folding for {script}: producers {producers} "
        f"disagree ({flags}) — teach this helper which one names containers"
    )
    return flags.pop()


def _expected_session_segment(script: str, session_id: str) -> str:
    """The session segment a container of ``script``'s env really carries."""
    segment = _producer_session_segment(session_id)
    return segment.lower() if _producer_lowercases(script) else segment


def _filter_uses_producer_session_segment(
    script: str, filters: tuple[str, ...], session_id: str,
) -> bool:
    """Does every filter carry that segment as a WHOLE ``-``-delimited one?"""
    segment = _expected_session_segment(script, session_id)
    pattern = re.compile(rf"(?:^|-){re.escape(segment)}-")
    return bool(filters) and all(pattern.search(f) for f in filters)


def _env_id_from_filter(flt: str, segment: str) -> str:
    """The env_id a filter demands, read off the filter's own tail.

    ``docker ps --filter name=`` is a Go RE2 regex, so a literal ``.`` may be
    written ``[.]`` (cuaworld) or left bare (lite.osworld); unescape, then drop
    the segment separators. A tail ending in ``.`` names a FAMILY of ids the
    registry mints at runtime (``cua.bench.local.<dataset>``,
    ``lite.cuaworld.<software>``) — append a leaf so the reconstructed id is a
    real member of it.
    """
    env_id = flt.partition(segment)[2].replace("[.]", ".").strip("-")
    return f"{env_id}leaf" if env_id.endswith(".") else env_id


def test_cleanup_script_sandbox_shadows_real_docker() -> None:
    """The harness must actually intercept ``docker`` before any cleanup.sh is
    executed through it — otherwise every test below would be running real
    ``docker rm -fv`` against a shared daemon."""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "probe-cleanup.sh"
        # `rm -fv` on a name no container can have: if the stub is NOT in front,
        # this reaches the real daemon and matches nothing, and the assertion
        # below fails loudly rather than the suite quietly going unsandboxed.
        script.write_text(
            '#!/bin/bash\ndocker ps -aq --filter "name=SANDBOXPROBE-" '
            "| xargs -r docker rm -fv\n"
        )
        filters = _run_cleanup_script(str(script), _CONTROL_SESSION)
    assert filters == ("SANDBOXPROBE-",), (
        "the stub docker did not intercept the probe — cleanup.sh execution is "
        f"NOT sandboxed here (captured {filters!r})"
    )


@pytest.mark.parametrize("script", _cleanup_scripts())
def test_cleanup_script_session_scoping_is_classified(script: str) -> None:
    """Every cleanup.sh either narrows by SESSION_ID or is pinned as
    session-blind. Classification is DERIVED — run the script under two
    different SESSION_IDs and see whether its docker filters move — so a script
    that stops (or starts) honouring SESSION_ID reddens here."""
    moves = (
        _cleanup_filters(script, _ADVERSARIAL_SESSION)
        != _cleanup_filters(script, _CONTROL_SESSION)
    )
    assert moves == (script not in _SESSION_BLIND_CLEANUP_SCRIPTS), (
        f"{script}: SESSION_ID-sensitivity is {moves}, but the pin says "
        f"{script not in _SESSION_BLIND_CLEANUP_SCRIPTS}. Either the script "
        "changed, or _SESSION_BLIND_CLEANUP_SCRIPTS needs updating."
    )


@pytest.mark.parametrize(
    "script",
    [
        s for s in _cleanup_scripts()
        if s not in _SESSION_BLIND_CLEANUP_SCRIPTS
        and s not in _UNSANITIZED_CLEANUP_SCRIPTS
    ],
)
def test_cleanup_scripts_sanitize_session_id_like_container_naming(script: str) -> None:
    """The script's docker filter must carry the session segment
    :func:`format_container_name` actually produces for the same SESSION_ID.

    Derived end to end: the expectation comes from calling the producer, the
    observation comes from executing the script. Nothing here knows how the
    script spells its sanitizer (the ten scripts use ``${x//…}``, ``sed`` and
    ``tr -c`` interchangeably) and nothing knows how the producer spells its
    own — either side may be rewritten freely as long as they agree.

    Only the SESSION segment is checked. The env_id segment is a different
    contract (owned by ``match_pattern`` and each env's own tests), and two env
    families mint concrete ids the registry does not enumerate, so pinning the
    full name here would restate rather than derive.
    """
    filters = _cleanup_filters(script, _ADVERSARIAL_SESSION)
    assert _filter_uses_producer_session_segment(script, filters, _ADVERSARIAL_SESSION), (
        f"{script}: docker filters {filters!r} do not carry the session segment "
        f"{_expected_session_segment(script, _ADVERSARIAL_SESSION)!r} that "
        f"format_container_name produces for SESSION_ID={_ADVERSARIAL_SESSION!r}"
    )
    # The raw, unsanitized session must not survive: a literal '-' in the
    # session segment forges the `-<port>-` boundary a server's reaper scopes on.
    assert all(_ADVERSARIAL_SESSION.lower() not in f.lower() for f in filters), (
        f"{script}: the raw SESSION_ID reached the docker filter {filters!r}"
    )


def test_no_cleanup_script_is_exempt_from_the_derivation() -> None:
    """The exemption set must stay empty, so the real gate above covers the
    whole tree.

    An earlier version of this test LOOPED over ``_UNSANITIZED_CLEANUP_SCRIPTS``
    to re-assert that each exempted script still fails the derivation. The set
    is a hand-maintained literal that nothing in production can populate, so
    the loop body was unreachable: it executed a shell script per entry (8-16 s
    of subprocess) for zero assertions and passed with every production fact
    broken. The honest, and the only reachable, statement is that the exemption
    mechanism is unused -- and an entry added here now has to redden this line
    and argue for itself, instead of quietly shrinking the parametrize list at
    :307.
    """
    assert not _UNSANITIZED_CLEANUP_SCRIPTS, (
        f"cleanup scripts exempted from the SESSION_ID derivation: "
        f"{sorted(_UNSANITIZED_CLEANUP_SCRIPTS)} — fix the script instead, or "
        "justify the exemption here"
    )


def _server_reaper_regex(server_port: int, env_id: str) -> str:
    """The docker-ps scope a server on ``server_port`` uses to find its own
    ``env_id`` containers — read off the reaper itself
    (:func:`lite.gym.remote.reaper._name_regex`), never retyped, so
    a regex edit on the reaper side reddens the cross-mode mis-kill guard
    below instead of drifting silently past a private copy."""
    prefix = container_name_prefix(server_port=server_port, token_hash=None)
    return reaper_name_regex(prefix, env_id)


def test_session_id_hyphen_cannot_forge_a_server_port_boundary():
    """REGRESSION (cross-mode mis-kill): a DIRECT-mode container (no
    server_port segment) whose session_id starts ``"<port>-…"`` must NOT match
    a co-resident env-server's port-scoped reaper. Before the fix the raw
    session ``"30100-trial"`` produced ``lite-env-30100-trial-androidworld-…``,
    which a server on port 30100 would ``docker rm -f`` while it was live."""
    name = format_container_name(
        env_id="androidworld", suffix="9554",
        task_id="mytask", session_id="30100-trial", server_port=None,
    )
    # The dangerous hyphen is gone — session is one opaque token.
    assert "30100-trial" not in name
    assert "30100_trial" in name
    # And the server-on-30100 reaper does NOT match it.
    assert re.match(_server_reaper_regex(30100, "androidworld"), name) is None


def test_server_mode_name_still_matches_its_own_reaper():
    """The sanitization must not break the legitimate case: a server's own
    container (session in the wildcard middle) still matches its reaper."""
    name = format_container_name(
        env_id="androidworld", suffix="9554",
        task_id="mytask", session_id="batch-7", server_port=30100,
    )
    assert re.match(_server_reaper_regex(30100, "androidworld"), name)


def test_dotted_env_id_is_literal_in_reaper_regex():
    """``lite.osworld``'s ``.`` is re.escape-d, so a server scanning for
    ``lite.osworld`` does not also match a hypothetical ``lite-osworld``."""
    rgx = _server_reaper_regex(30100, "lite.osworld")
    good = format_container_name(
        env_id="lite.osworld", suffix="x", session_id="s", server_port=30100,
    )
    assert re.match(rgx, good + "-")  # trailing sep present in real names
    assert re.match(rgx, "lite-env-30100-s-liteXosworld-x-") is None


# ── section D2: producer/consumer share one name grammar ────────────────────────────

def test_match_pattern_matches_own_server_containers():
    import re

    from lite.gym.utils.config.naming import (
        container_name_prefix,
        format_container_name,
        match_pattern,
    )
    name = format_container_name(
        env_id="lite.osworld", suffix="abc123", task_id="t1",
        session_id="sess", server_port=30100,
    )
    pat = match_pattern(container_name_prefix(server_port=30100), "lite.osworld")
    assert re.search(pat, name), (pat, name)
    # '.' in env_id must not act as a wildcard: 'lite-osworld' names don't match
    assert not re.search(pat, name.replace("lite.osworld", "liteXosworld"))


def test_direct_mode_session_can_never_forge_a_server_scope():
    """The section 7.debunked regression guard: a DIRECT container (no server_port
    segment) whose session is numeric — or even carries a '-' that
    _sanitize_session_id must collapse — is structurally unmatchable by a
    co-resident server's scope pattern."""
    import re

    from lite.gym.utils.config.naming import (
        container_name_prefix,
        format_container_name,
        match_pattern,
    )
    pat = match_pattern(container_name_prefix(server_port=30100), "lite.osworld")
    for session in ("30100", "30100-x", "30100-", "-30100"):
        direct = format_container_name(
            env_id="lite.osworld", suffix="abc123", session_id=session,
        )
        assert not re.search(pat, direct), (pat, direct)
