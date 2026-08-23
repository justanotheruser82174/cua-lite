"""Guard: every ``sudo`` the lite.osworld **os** generators put into a command
WE run (setup / config / gold / oracle / evaluator) must be credential-primed.

Why this is a real invariant, not style. In the lite.osworld container the
desktop user is a **password-gated** sudoer (``useradd -G sudo`` + ``chpasswd``,
stock ``%sudo``, no ``NOPASSWD`` — see
``tests/gym/sandbox/test_separation_contract.py::test_sudo_is_password_gated_not_nopasswd``),
and ``exec_stdio``'s ``op_run_command`` runs ``bash -c`` with **no tty**. So a
bare ``sudo foo`` does not hang and does not succeed — it dies immediately with::

    sudo: a terminal is required to read the password; either use the -S option
          to read from standard input or configure an askpass helper

Verified in ``cua-lite/sandbox.linux:latest``. The working idiom, used by every
sibling row in these generators, feeds the password on stdin::

    echo user | sudo -S -v 2>/dev/null; sudo <cmd>          # prime, then reuse
    echo user | sudo -S <cmd>                               # or prime per call

Second, sharper rule (also verified in-container): the ``sudo -v`` timestamp is
keyed to the **parent pid** when no tty is present, so a primed credential is
reusable only by later ``sudo`` calls that are *direct children of the same
shell*. A ``sudo`` inside a ``$(...)`` command substitution or an explicit
``( ... )`` subshell gets a different parent and misses the cache. Each such
scope therefore needs its own priming, and the detector below treats it as an
independent scope.

Scope: the three ``os``-domain generators. Bare ``sudo`` also exists today in
``src/gen/eval/multi_apps.py`` (rows ``69acbb55`` / ``e1fc0df3`` / ``f5c13cdd``);
that file is owned elsewhere — widen ``_EVAL_MODULES`` once it is fixed.

Run:
    uv run --no-sync pytest tests/gym/envs/lite/osworld/test_lite_osworld_sudo_idiom.py -q
"""

from __future__ import annotations

import re

import pytest

# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

#: `sudo` as a word, not `/usr/bin/sudo-ish` or `no-sudo`.
_SUDO = re.compile(r"(?<![\w/-])sudo\b")

#: A priming call: the password arrives on stdin from a pipe — `... | sudo -S`.
_PRIMED = re.compile(r"\|\s*sudo\s+-S\b")


def _scopes(cmd: str) -> list[str]:
    """Split `cmd` into independent sudo-credential scopes.

    The top-level text is one scope; every ``$( ... )`` command substitution and
    every explicit ``( ... )`` subshell is its own scope, because a forked
    subshell does not inherit the parent shell's ppid-keyed sudo timestamp.
    Nested spans are returned as separate scopes and removed from their parent.
    """
    out: list[str] = []
    top: list[str] = []
    i, n = 0, len(cmd)
    while i < n:
        ch = cmd[i]
        is_subst = ch == "(" and (i == 0 or cmd[i - 1] == "$")
        is_group = ch == "(" and not is_subst
        if is_subst or is_group:
            depth, j = 1, i + 1
            while j < n and depth:
                if cmd[j] == "(":
                    depth += 1
                elif cmd[j] == ")":
                    depth -= 1
                j += 1
            inner = cmd[i + 1: j - 1]
            out.extend(_scopes(inner))
            top.append(" ")  # keep the parent scope's token boundaries
            i = j
            continue
        top.append(ch)
        i += 1
    out.append("".join(top))
    return out


def unprimed_sudo_scopes(cmd: str) -> list[str]:
    """Return the scopes of `cmd` whose FIRST ``sudo`` is not credential-primed.

    Empty list == the command is safe to run as the desktop user under
    password-gated, tty-less sudo.
    """
    bad: list[str] = []
    for scope in _scopes(cmd):
        m = _SUDO.search(scope)
        if not m:
            continue
        head, tail = scope[: m.start()], scope[m.start():]
        # `echo user | sudo -S ...`: the pipe must feed THIS first sudo.
        if re.search(r"\|\s*$", head) and re.match(r"sudo\s+-S\b", tail):
            continue
        # `... | sudo -S -v ...; sudo x` — priming earlier in the same scope.
        if _PRIMED.search(head):
            continue
        bad.append(scope.strip())
    return bad


# ---------------------------------------------------------------------------
# Object walkers
# ---------------------------------------------------------------------------

def _commands(obj, path: str = ""):
    """Yield ``(jsonpath, command_string)`` for every command in a step tree."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}"
            if k == "command":
                if isinstance(v, str):
                    yield p, v
                elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                    yield p, " ".join(v)
                continue
            yield from _commands(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _commands(v, f"{path}[{i}]")


def _violations(obj, label: str) -> list[str]:
    out = []
    for jp, cmd in _commands(obj, label):
        for scope in unprimed_sudo_scopes(cmd):
            out.append(f"{jp}: {scope[:200]}")
    return out


# ---------------------------------------------------------------------------
# 1. Detector self-test — proves the guard can actually fail.
# ---------------------------------------------------------------------------

class TestDetector:
    @pytest.mark.parametrize("cmd", [
        "sudo bash -c 'echo hi'",
        "sudo mkdir -p /a && sudo chown user:user /a && mv /b /a; true",
        "true; sudo tee /etc/timezone",
        # primed, but the reuse happens inside a forked command substitution,
        # which does NOT inherit the ppid-keyed timestamp
        "echo user | sudo -S -v 2>/dev/null; X=$(sudo cat /etc/shadow)",
        # primed, but the reuse happens in an explicit subshell
        "echo user | sudo -S -v 2>/dev/null; ( sudo id )",
        # a pipe that is not a password feed
        "cat pw.txt | grep x; sudo id",
    ])
    def test_flags_unprimed(self, cmd):
        assert unprimed_sudo_scopes(cmd), f"detector missed a bare sudo: {cmd!r}"

    @pytest.mark.parametrize("cmd", [
        "echo user | sudo -S apt-get install -y bc 2>&1 | tail -1",
        "echo user | sudo -S -v 2>/dev/null; sudo ln -sf /usr/share/zoneinfo/UTC /etc/localtime",
        "echo user | sudo -S -v 2>/dev/null; sudo tee /x >/dev/null && sudo chmod +x /x",
        "echo user | sudo -S -v 2>/dev/null; if [ -f /a ]; then sudo mkdir -p /b; fi",
        # priming and reuse both inside the same subshell scope
        "( echo user | sudo -S -v 2>/dev/null; sudo id )",
        # $( ) spans that contain no sudo must not be mistaken for violations
        "echo user | sudo -S -v 2>/dev/null; printf 'TZ=$(cat /etc/timezone)' | sudo tee /x",
        "mkdir -p $(dirname /a/b) && echo ok",
    ])
    def test_accepts_primed(self, cmd):
        assert not unprimed_sudo_scopes(cmd), f"detector false-positive: {cmd!r}"


# ---------------------------------------------------------------------------
# 2. The generators themselves.
# ---------------------------------------------------------------------------

class TestOsGeneratorsPrimeSudo:
    def test_eval_os_oracles(self):
        """`src/gen/eval/os.py::ORACLES` — curated actions / config_* / evaluator."""
        from lite.gym.envs.lite.osworld.src.gen.eval.os import ORACLES

        bad = []
        for uuid, entry in ORACLES.items():
            bad += _violations(entry, f"eval.os[{uuid}]")
        assert not bad, "unprimed sudo in eval/os.py ORACLES:\n  " + "\n  ".join(bad)

    def test_perturb_os_pre_config_steps(self):
        """`src/gen/train/perturb/os.py` — the module's own static step tables."""
        import lite.gym.envs.lite.osworld.src.gen.train.perturb.os as perturb_os

        bad = _violations(perturb_os._PARAPHRASE_PRE_CONFIG_STEPS,
                          "perturb.os._PARAPHRASE_PRE_CONFIG_STEPS")
        assert not bad, "unprimed sudo in perturb/os.py:\n  " + "\n  ".join(bad)

    def test_synth_os_emitted_rows(self):
        """`src/gen/train/synth/os.py` — every row the os templates actually emit.

        Data-gated exactly like ``test_train_jsonl_idempotent``: importing the
        synth package executes module-level ``_stage_asset`` calls that hard-fail
        on an unseeded checkout.
        """
        from lite.gym.envs.lite.osworld.src.utils.assets import asset_root

        root = asset_root()
        manifest = root / "MANIFEST.csv"
        if not manifest.is_file():
            pytest.skip(f"lite.osworld synth asset bundle not seeded at {root}")

        from lite.gym.envs.lite.osworld.src.gen.train.synth._utils import make_synth_row
        from lite.gym.envs.lite.osworld.src.gen.train.synth.catalog import TEMPLATES_BY_DOMAIN

        templates = TEMPLATES_BY_DOMAIN.get("os", [])
        assert templates, "no os synth templates registered — guard would be vacuous"

        bad, n_rows = [], 0
        for template in templates:
            for seed in range(1, template.n_rows + 1):
                params = template.param_fn(seed)
                if params.get("_skip"):
                    continue
                row = make_synth_row(template, seed, params)
                n_rows += 1
                bad += _violations(row["metadata"], row["task_id"])
        assert n_rows, "no os synth rows emitted — guard would be vacuous"
        assert not bad, "unprimed sudo in synth/os.py output:\n  " + "\n  ".join(bad)
