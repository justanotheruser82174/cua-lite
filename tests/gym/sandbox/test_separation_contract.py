"""Static source-contract + pure-python unit tests for the agent/env separation.

Post the *revert-to-default-user* refactor the separation is no longer a uid wall:
the exec-stdio server runs as the env's DESKTOP USER (``EXEC_USER = "user"``), and
``/opt/env`` + ``/opt/lite`` are world-READABLE (``a+rX``). The only remaining
boundary is a **PATH separation** — env-only CLIs (xdotool/jq/pandoc/…) are vendored
into ``/opt/env/bin``, which is off the agent's login PATH, so the agent doesn't see
them by default even though it *could* read the tree. Sudo is present but
**password-gated** (stock ``%sudo``), matching the real OSWorld VM, NOT ``NOPASSWD``.

These assert that the SOURCE (Dockerfiles + ``base.py``) encodes that default-user
contract — they do NOT build images or run containers, so they run in the normal
(non-``live``) suite and stay green offline. This file guards the invariants that
must hold in the source tree itself, so a refactor that silently re-walls the island
(or re-widens sudo to NOPASSWD) fails here first.

Run with:
    uv run --no-sync pytest tests/gym/sandbox/test_separation_contract.py -q
"""

from __future__ import annotations

import re
from pathlib import Path

from lite.gym.sandbox.base import SandboxBaseEnv
from lite.gym.sandbox.exec_stdio.server import _bracket_pkill

_REPO = Path(__file__).resolve().parents[3]
_DOCKERFILE_LINUX = _REPO / "lite/gym/sandbox/docker/Dockerfile.linux"
_DOCKERFILE_OSWORLD = _REPO / "lite/gym/envs/lite/osworld/docker/Dockerfile"
_CUAWORLD_SOFTWARE = _REPO / "lite/gym/envs/lite/cuaworld/src/software.py"
_TIMEDATECTL_SHIM = _REPO / "lite/gym/envs/lite/osworld/docker/bin/timedatectl"
_RELOCATE_HELPER = _REPO / "lite/gym/sandbox/docker/bin/relocate-clis.sh"


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def _noncomment_lines(path: Path) -> list[str]:
    """Dockerfile lines with pure-comment lines dropped, so prose that merely
    NAMES a token (e.g. a comment discussing ``NOPASSWD``) does not defeat a
    'token absent from the build' assertion."""
    return [ln for ln in _lines(path) if not ln.lstrip().startswith("#")]


# ---------------------------------------------------------------------------
# 1. Base default-user source-contract — Dockerfile.linux (base / sandbox.linux)
# ---------------------------------------------------------------------------
class TestBaseDockerfileDefaultUser:
    """The default-user contract is baked into the base image source: the agent
    IS the desktop user, IS a (password-gated) sudoer, and ``/opt/env`` + ``/opt/lite``
    are world-readable — the separation is PATH-based (env CLIs off the login PATH),
    not a uid-700 wall."""

    def test_useradd_line_has_sudo_group(self):
        # Dockerfile.linux:230 — the agent user is created `-G sudo`. This is the
        # load-bearing revert edit: the default-user contract makes the agent a
        # (password-gated) sudoer, matching the real OSWorld VM's `user`.
        build = _noncomment_lines(_DOCKERFILE_LINUX)
        useradd = [ln for ln in build if re.search(r"\buseradd\b", ln)]
        assert useradd, "no useradd line found in Dockerfile.linux"
        assert any("-G sudo" in ln for ln in useradd), (
            f"agent user is NOT in group sudo (revert dropped): {useradd!r}"
        )

    def test_sudo_is_password_gated_not_nopasswd(self):
        # The agent is a sudoer but PASSWORD-gated (Ubuntu's stock
        # `%sudo ALL=(ALL:ALL) ALL`, password baked via chpasswd) — NOT NOPASSWD.
        # A NOPASSWD grant would let the agent walk to root non-interactively; the
        # real OSWorld VM ships `echo ${USER} | sudo -S …`, so we mirror that.
        build = _noncomment_lines(_DOCKERFILE_LINUX)
        for ln in build:
            assert "NOPASSWD:ALL" not in ln.replace(" ", ""), (
                f"NOPASSWD:ALL sudoers grant present: {ln!r}"
            )
        # the baked password (`${USER}:${USER}` | chpasswd) is what makes the
        # stock password-gated %sudo actually usable.
        assert any(re.search(r"\bchpasswd\b", ln) for ln in build), (
            "no chpasswd line — password-gated sudo would be unusable"
        )

    def test_opt_env_is_world_readable_not_walled(self):
        # Dockerfile.linux:97 — /opt/env is `chmod -R a+rX` (world read + traverse),
        # NOT the old uid-700 `go-rwx` wall. Env CLIs still live under /opt/env/bin
        # (off the agent PATH), but the tree is READABLE — the wall is gone.
        # Assert on BUILD commands (comments dropped) so the header's contrast
        # phrasing ("not a uid wall") can't sway a text match either way.
        build_text = "\n".join(_noncomment_lines(_DOCKERFILE_LINUX))
        assert re.search(r"chmod\s+-R\s+a\+rX\s+/opt/env(?![\w/])", build_text), (
            "/opt/env not chmod'd a+rX (world-readable)"
        )
        assert "chmod -R go-rwx /opt/env" not in build_text, (
            "/opt/env still walled go-rwx — revert incomplete"
        )

    def test_terminal_wrappers_scrub_env_path(self):
        # `run_command` carries /opt/env for setup/eval. If that channel launches a
        # visible terminal, the terminal shell must not inherit env-only CLIs/venv.
        build = _lines(_DOCKERFILE_LINUX)

        for command in ("gnome-terminal", "x-terminal-emulator"):
            start = next(
                i
                for i, ln in enumerate(build)
                if ln
                == (
                    f"RUN cat > /usr/local/bin/{command} <<'EOF' && "
                    f"chmod +x /usr/local/bin/{command}"
                )
            )
            end = next(i for i in range(start + 1, len(build)) if build[i] == "EOF")
            wrapper = "\n".join(build[start : end + 1])
            assert (
                "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                in wrapper
            )
            assert f'exec /usr/bin/{command} "$@"' in wrapper

    def test_opt_lite_world_readable_after_stdio_server_copy(self):
        # Dockerfile.linux:770 COPY, :777 chmod: /opt/lite exists only after the
        # COPY of stdio_server.py, so its `a+rX` open is a SEPARATE later step. The
        # server source is now readable (no protocol-source wall) — only the install
        # LOCATION separation remains (/opt/lite is off nothing the agent runs).
        build = _lines(_DOCKERFILE_LINUX)
        copy_idx = next(
            i for i, ln in enumerate(build) if "COPY" in ln and "/opt/lite/stdio_server.py" in ln
        )
        lite_open = [
            i for i, ln in enumerate(build) if re.search(r"chmod\s+-R\s+a\+rX\s+/opt/lite\b", ln)
        ]
        assert lite_open, "no `chmod -R a+rX /opt/lite` step found"
        assert all(i > copy_idx for i in lite_open), (
            "/opt/lite chmod must come AFTER the stdio_server.py COPY"
        )
        build_text = "\n".join(_noncomment_lines(_DOCKERFILE_LINUX))
        assert "chmod -R go-rwx /opt/lite" not in build_text, (
            "/opt/lite still walled go-rwx — revert incomplete"
        )

    def test_system_dbus_root_program_baked_root_owned(self):
        # system-D-Bus bring-up lives in a root-run supervisor program so
        # start-gnome.sh (${USER}) never needs `sudo` at boot (password-gated sudo
        # would hang under `set -e`). The program is `user=root`, and its script/.conf
        # are created while the build is in a `USER root` context (→ baked root-owned).
        build = _lines(_DOCKERFILE_LINUX)
        conf_idx = next(i for i, ln in enumerate(build) if "[program:system-dbus]" in ln)
        # user=root within the program's conf block (until the next EOF/heredoc end)
        block = build[conf_idx : conf_idx + 12]
        assert any(ln.strip() == "user=root" for ln in block), (
            "[program:system-dbus] is not user=root"
        )
        # baked root-owned: the most recent `USER` directive before the script is
        # created must be `USER root` (not the ${USER} agent account). The script is
        # created by a heredoc (`cat > /usr/local/bin/system-dbus.sh <<'EOF'`) which,
        # post layer-squash, lives inside a mega-RUN block rather than its own `RUN`
        # line — match the heredoc create line, not a leading `RUN`.
        script_idx = next(
            i
            for i, ln in enumerate(build)
            if "system-dbus.sh" in ln and ("<<" in ln or ln.startswith("RUN"))
        )
        prior_user = [ln for ln in build[:script_idx] if ln.startswith("USER ")]
        assert prior_user and prior_user[-1].strip() == "USER root", (
            f"system-dbus script not created in a root USER context: {prior_user[-1:]!r}"
        )


# ---------------------------------------------------------------------------
# 1a. Base desktop defaults — Dockerfile.linux dconf contracts
# ---------------------------------------------------------------------------
class TestBaseDockerfileDesktopDefaults:
    def test_gnome_welcome_suppression_stays_out_of_dock_dconf_file(self):
        build = _lines(_DOCKERFILE_LINUX)

        def heredoc_block(target: str) -> str:
            start = next(i for i, ln in enumerate(build) if ln == target)
            end = next(i for i in range(start + 1, len(build)) if build[i] == "EOF")
            return "\n".join(build[start : end + 1])

        dock_target = "cat > /etc/dconf/db/local.d/03-shell-dock <<'EOF'"
        welcome_target = "cat > /etc/dconf/db/local.d/03-shell-welcome <<'EOF'"
        dock = heredoc_block(dock_target)
        welcome = heredoc_block(welcome_target)

        assert build.index(dock_target) < build.index(welcome_target)
        assert "welcome-dialog-last-shown-version" not in dock
        assert "welcome-dialog-last-shown-version='999999999'" in welcome


# ---------------------------------------------------------------------------
# 1b. osworld image — PATH-separated env CLIs, world-readable venv
# ---------------------------------------------------------------------------
class TestOsworldDockerfileEnvVendoring:
    def test_env_venv_is_world_readable_not_walled(self):
        # osworld Dockerfile:472 — the osworld layer EXTENDS /opt/env/venv with its
        # office/eval libs, then re-opens it `a+rX` (world-readable), NOT the old
        # uid-700 `go-rwx`. Under default-user, the desktop-user server reads this venv.
        build_text = "\n".join(_noncomment_lines(_DOCKERFILE_OSWORLD))
        assert re.search(r"chmod\s+-R\s+a\+rX\s+/opt/env/venv\b", build_text), (
            "/opt/env/venv not re-opened a+rX after the osworld pip install"
        )
        assert "chmod -R go-rwx /opt/env/venv" not in build_text, (
            "/opt/env/venv still walled go-rwx — revert incomplete"
        )

    def test_env_clis_vendored_off_agent_path(self):
        # The PATH separation is the REMAINING boundary: env-only CLIs' real binaries
        # are MOVED from /usr/bin into /opt/env/bin (off the agent's login PATH) and
        # de-registered from /usr/bin. Post the shared-helper refactor this mechanism
        # lives in ONE baked script — docker/bin/relocate-clis.sh — which the BASE calls
        # for its harness CLIs (xdotool + ImageMagick) and OSWORLD reuses for its eval
        # CLIs (jq/xclip/pandoc/pdftk/java). The exec_stdio/eval PATH carries
        # /opt/env/bin, so the env side still resolves them. Install-LOCATION separation.
        helper = _RELOCATE_HELPER.read_text()
        # the shared helper MOVES /usr/bin binaries into /opt/env/bin (not copied) …
        assert re.search(r'/usr/bin/\*\)\s*mv\s+"\$real"\s+"/opt/env/bin/\$b"', helper), (
            "relocate-clis.sh does not move /usr/bin CLIs into /opt/env/bin"
        )
        # … and de-registers them from /usr/bin so the agent PATH no longer sees them.
        assert re.search(r'rm\s+-f\s+"/usr/bin/\$c"', helper), (
            "relocate-clis.sh does not remove the CLI from the agent's /usr/bin"
        )
        # BASE invokes the helper for its own harness CLIs (xdotool + ImageMagick verbs),
        # then de-registers those command packages. (The helper call spans a `\`
        # continuation, so match against the joined non-comment build text — comments are
        # stripped, so xdotool/convert only appear in the relocation RUN.)
        base = _noncomment_lines(_DOCKERFILE_LINUX)
        base_text = "\n".join(base)
        assert (
            "relocate-clis.sh" in base_text and "xdotool" in base_text and "convert" in base_text
        ), (
            "base Dockerfile.linux does not relocate its harness CLIs "
            "(xdotool/convert) via relocate-clis.sh"
        )
        assert any("apt-get remove" in ln and "xdotool" in ln for ln in base), (
            "base does not de-register xdotool/imagemagick after relocation"
        )
        # OSWORLD reuses the SAME helper for its env-only eval CLIs (jq named), then removes them.
        osw = _noncomment_lines(_DOCKERFILE_OSWORLD)
        osw_text = "\n".join(osw)
        assert "relocate-clis.sh" in osw_text and "jq" in osw_text, (
            "osworld Dockerfile does not reuse relocate-clis.sh for its eval CLIs (jq)"
        )
        assert any("apt-get remove" in ln and "jq" in ln for ln in osw), (
            "osworld does not de-register its relocated eval CLIs"
        )
        # And osworld must NOT touch the BASE-relocated harness CLIs — installing OR
        # relocating them here re-leaks them onto the agent PATH after the base moved them
        # (osworld builds FROM the base, which KEEPS imagemagick-6-common+libmagickcore, so
        # an `apt-get install imagemagick` re-registers convert/import/… via alternatives).
        # The centerpiece negative control proved the xdotool case fails L1; guard both the
        # `xdotool` command and the `imagemagick` package name statically (both are
        # false-positive-free in osworld non-comment lines; the bare IM verbs like `convert`
        # collide with `libreoffice --convert-to` / python `import`, so we key on the package).
        for token in ("xdotool", "imagemagick"):
            assert not any(re.search(rf"\b{token}\b", ln) for ln in osw), (
                f"osworld names {token} in a non-comment line — the base owns it now; "
                "installing/relocating it here re-leaks the base-relocated "
                "CLIs onto the agent PATH"
            )


# ---------------------------------------------------------------------------
# 1c. base.py — exec-stdio server runs as the desktop user (default), cuaworld=root
# ---------------------------------------------------------------------------
class TestExecUserContract:
    def test_exec_user_default_is_user(self):
        # Default-user contract: the persistent exec-stdio backend runs as the env's
        # DESKTOP USER, so setup/getter/reward files are user-owned by construction and
        # /opt/env (now user-readable) is reached directly — no root, no run_as channel.
        assert SandboxBaseEnv.EXEC_USER == "user"

    def test_cuaworld_overrides_exec_user_to_root(self):
        # cuaworld's upstream gym-anything hooks self-drop via `su - ga -c` (GUI apps
        # must not run as root); passwordless `su` works only FROM root, so cuaworld
        # OVERRIDES the base default back to root. Assert on source (the class is
        # defined inside a factory fn, so it isn't importable directly).
        text = _CUAWORLD_SOFTWARE.read_text()
        assert re.search(r'EXEC_USER\s*:\s*ClassVar\[str\]\s*=\s*"root"', text), (
            "cuaworld no longer overrides EXEC_USER back to root"
        )


# ---------------------------------------------------------------------------
# 2a. Channel routing — pkill/pgrep self-match bracket rewrite (server.py)
# ---------------------------------------------------------------------------
class TestBracketPkill:
    """`pkill -f <pat>` matches the transport's OWN
    argv → self-terminates the backend / masks rc. Bracket the first char so it
    still matches the target but never our literal argv text."""

    def test_pkill_f_first_char_bracketed(self):
        assert _bracket_pkill("pkill -f chrome") == "pkill -f '[c]hrome'"

    def test_pgrep_f_first_char_bracketed(self):
        assert _bracket_pkill("pgrep -f soffice") == "pgrep -f '[s]office'"

    def test_trailing_operator_suffix_preserved(self):
        # the rewrite must not swallow a `; true` (or any) suffix after the pattern.
        assert _bracket_pkill("pkill -f chrome ; true") == "pkill -f '[c]hrome' ; true"

    def test_killall_unchanged(self):
        # killall matches `comm`, never self-matches → must be left alone.
        assert _bracket_pkill("killall chrome") == "killall chrome"

    def test_bare_pkill_without_f_unchanged(self):
        # bare `pkill <name>` (no -f) matches `comm`, never self-matches.
        assert _bracket_pkill("pkill chrome") == "pkill chrome"

    def test_already_bracketed_unchanged(self):
        assert _bracket_pkill("pkill -f '[c]hrome'") == "pkill -f '[c]hrome'"

    def test_metachar_leading_patterns_stay_valid_ere(self):
        # #154: bracketing the FIRST char corrupted a metachar-leading ERE
        # (`\[STUB\]`->`[\][STUB\]` unterminated, `^q`->`[^]q` bad class,
        # `.*x`->`[.]*x` loses `.*`). Bracket the first ALNUM char instead, and
        # skip patterns already carrying a `[...]` class (self-avoiding). Every
        # output must compile as an ERE and still break the self-match.
        import re

        cases = {
            "pkill -f '^qemu-system' || true": "pkill -f '^[q]emu-system' || true",
            "pkill -f '.*http.server'": "pkill -f '.*[h]ttp.server'",
            r"pkill -f '\[STUB VLC\]'": r"pkill -f '\[STUB VLC\]'",  # has [ -> untouched
            "pkill -f '[h]ttp.server 8766'": "pkill -f '[h]ttp.server 8766'",  # pre-bracketed
        }
        for cmd, want in cases.items():
            got = _bracket_pkill(cmd)
            assert got == want, (cmd, got)
            pat = re.search(r"pkill -f '([^']*)'", got).group(1)
            re.compile(pat)  # must be a valid ERE (raises otherwise)


# ---------------------------------------------------------------------------
# 3. timedatectl set-timezone traversal sanitize (source-contract)
# ---------------------------------------------------------------------------
class TestTimedatectlSanitize:
    """The osworld timedatectl shim's `set-timezone` guard
    must reject a leading `/` and any `..` path COMPONENT while ALLOWING internal
    `/` (else it breaks America/New_York and the default Asia/Hong_Kong). This is
    a shell script, so we assert on the source of the guard, not a python call."""

    def test_guard_rejects_leading_slash(self):
        text = _TIMEDATECTL_SHIM.read_text()
        # leading-slash rejection: a `case "$zone" in /*)` arm that errors out.
        assert re.search(r'case\s+"\$zone"\s+in\s*\n\s*/\*\)', text), (
            "no leading-'/' rejection arm in set-timezone guard"
        )

    def test_guard_rejects_dotdot_component(self):
        text = _TIMEDATECTL_SHIM.read_text()
        # `..`-component rejection via the `/`…`/` wrap + `*/../*` match.
        assert re.search(r'case\s+"/\$zone/"\s+in\s*\n\s*\*/\.\./\*\)', text), (
            "no '..'-component rejection arm in set-timezone guard"
        )

    def test_guard_allows_internal_slash(self):
        # the guard must NOT reject internal '/' wholesale (would break real zones
        # like America/New_York and the default Asia/Hong_Kong). Assert there is no
        # arm rejecting every internal slash (e.g. a `*/*)` → Invalid-time-zone arm).
        text = _TIMEDATECTL_SHIM.read_text()
        assert not re.search(r"\*/\*\)\s*printf\s+.Invalid time zone", text), (
            "guard rejects ALL internal '/' → real zones would break"
        )
