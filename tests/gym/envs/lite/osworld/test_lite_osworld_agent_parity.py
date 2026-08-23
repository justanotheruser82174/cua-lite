"""Anti-drift parity gate: lite.osworld's AGENT-FACING surface must stay 1:1 with the
official osworld guest VM, the env-facing half must stay OFF the agent, and the env venv
(/opt/env/venv, 3.12) must CARRY every package the in-container getters import.

WHY THIS EXISTS: the agent/env split lives in the Dockerfiles as imperative
install/vendor steps. Nothing structurally stopped it from drifting — a package
silently dropped or version-bumped, an env-only lib leaking onto the agent's python, or
a CLI re-leaking onto the agent PATH after an apt reinstall. Those bugs only surface at
train/eval time as a policy that worked on lite failing on real osworld. This test turns
the invariant into an enforced check.

GROUND TRUTH was read directly off the official osworld guest qcow2
(``docker_vm_data/Ubuntu.qcow2`` → Ubuntu 22.04.3, python 3.10.12) via libguestfs.
The reference below is lite's INTENDED agent set = the guest mirror WITH the three
documented, agent-immaterial deviations (numpy pin, MouseInfo/PyGetWindow/PyRect omitted,
single python-xlib 0.33). Any change to lite's agent surface must update BOTH the
Dockerfile and this reference in lockstep — that is the point.

Marked ``live`` (CLAUDE.md): needs Docker + the built image. Run with:
    uv run pytest tests/gym/envs/lite/osworld/test_lite_osworld_agent_parity.py -m live -q
Skips automatically if Docker or the image is unavailable. Override the tag (e.g. to
validate a freshly-built private tag) via LITE_OSWORLD_TEST_IMAGE.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.live

_IMAGE = os.environ.get("LITE_OSWORLD_TEST_IMAGE", "cua-lite/lite.osworld:latest")

# --- AGENT-FACING python (system 3.10 = /usr/bin/python3): dist-name -> expected version.
# Present + version-locked to the osworld guest, except the 3 deviations flagged inline.
AGENT_PY_REQUIRED = {
    "Flask": "3.0.0",
    "PyAutoGUI": "0.9.54",
    "PyScreeze": "0.1.30",
    "pytweening": "1.0.7",
    "PyMsgBox": "1.0.9",
    "pyperclip": "1.8.2",
    "pynput": "1.7.6",
    "python-xlib": "0.33",  # maintained Xlib; guest also ships colliding 0.15
    "Pillow": "10.1.0",
    "numpy": "1.24.4",  # scipy 1.8.0 caps numpy<1.25
    "scipy": "1.8.0",
    "matplotlib": "3.5.1",
    "sympy": "1.9",
    "reportlab": "3.6.8",
    "olefile": "0.46",
    "paramiko": "2.9.3",
    "pexpect": "4.8.0",
    "beautifulsoup4": "4.10.0",
    "lxml": "4.8.0",
    "PyYAML": "5.4.1",  # apt python3-yaml (NOT pip) — matches the guest's apt yaml exactly
    "requests": "2.25.1",
    "html5lib": "1.1",
    "chardet": "4.0.0",
}

# import-name checks (some libs don't carry a clean dist metadata version on the guest's apt build)
AGENT_IMPORT_REQUIRED = ["tkinter", "evdev"]

# env-only python — the guest's agent LACKS these, so ours must NOT import them on 3.10.
# (they live in /opt/env/venv). pygame is removed ENTIRELY (guest lacks it; snake eval uses a stub).
#
# ACCEPTED deviation for mouseinfo/pygetwindow/pyrect: the guest ships them as
# pyautogui deps, but lite installs pyautogui with --no-deps on purpose — the
# full-deps pull drags mouseinfo -> python3-Xlib 0.15 fork, clobbering the
# maintained python-xlib 0.33. Keeping them off the agent is the intended trade,
# so we assert their absence. NOTE: import names are lowercase (`mouseinfo`, not
# `MouseInfo`) — the capitalized spelling makes import_module raise ImportError
# unconditionally, making the "absent" assertion vacuous.
AGENT_PY_FORBIDDEN = [
    "openpyxl",
    "pandas",
    "docx",
    "pptx",
    "fitz",
    "mutagen",
    "xlsxwriter",
    "websocket",
    "pysrt",
    "pygame",
    "mouseinfo",
    "pygetwindow",
    "pyrect",
]

# env-facing python (/opt/env/venv, 3.12): the POSITIVE completeness half of the split.
# Every in-container getter/setup/eval runs here — config `execute` steps (the
# `python3 << 'PYEOF' … PYEOF` heredocs), run_command's `python3`, and the Flask server's
# shelled-out `python3` all resolve to /opt/env/venv/bin via the env PATH. So every package
# those getters import MUST be installed into this venv (the `uv pip install --python
# /opt/env/venv/bin/python` list in lite/gym/envs/lite/osworld/docker/Dockerfile). This is
# symmetric to AGENT_PY_FORBIDDEN above: that asserts these libs stay OFF the agent 3.10;
# this asserts they ARE in the env venv. A miss here is the exact ModuleNotFoundError a
# setup/eval getter would hit at runtime. IMPORT names (not dist names): pymupdf->fitz,
# python-pptx->pptx, python-docx->docx, Pillow->PIL, beautifulsoup4->bs4, python-xlib->Xlib,
# websocket-client->websocket, PyYAML->yaml, XlsxWriter->xlsxwriter.
ENV_IMPORT_REQUIRED = [
    "openpyxl",
    "pptx",
    "docx",
    "mutagen",
    "fitz",
    "xlsxwriter",
    "PIL",
    "numpy",
    "lxml",
    "bs4",
    "pandas",
    "pyautogui",
    "Xlib",
    "requests",
    "websocket",
    "yaml",
    "pysrt",
]

# CLIs the osworld guest LACKS → must be OFF the agent PATH (/usr/local/bin, /usr/bin, /bin).
# Includes ImageMagick's versioned real binaries + pdftk's real binary — the vendor step
# resolves each command to its real binary and moves that off /usr/bin, not just the
# bare-name symlinks.
IM_VERBS = [
    "convert",
    "import",
    "mogrify",
    "identify",
    "compare",
    "composite",
    "montage",
    "animate",
    "conjure",
    "display",
    "stream",
]
CLI_FORBIDDEN = [
    "xdotool",
    "xclip",
    "xsel",
    "jq",
    "pandoc",
    "pdftk",
    "pdftk.pdftk-java",
    *IM_VERBS,
    *(f"{verb}-im6" for verb in IM_VERBS),
    *(f"{verb}-im6.q16" for verb in IM_VERBS),
    "tesseract",
    "sox",
    "sqlite3",
    # java + the JRE sibling CLIs: the guest has NO java ecosystem, so the JRE
    # that rides in via pdftk-java's dep is an agent divergence — java is
    # vendored to /opt/env, the siblings are removed outright.
    # TODO(:mine): reconcile against enumerated JRE bin set
    "java",
    "keytool",
    "rmiregistry",
    "jjs",
    "jexec",
    "pack200",
    "unpack200",
    "rmid",
]

# CLIs the guest HAS → must be ON the agent PATH. Beyond the original KEEP set,
# this pins the 8 Q4 tools + the 8 build-toolchain names, all empirically
# VM-present — without them the "KEEP set all PRESENT" gate asserts nothing.
# (`expect` is intentionally absent: the VM lacks the expect CLI; it is now an
# accepted lite-only deviation covered by test_expect_not_env_vendored_into_opt_env.)
CLI_REQUIRED = [
    "wmctrl",
    "ffmpeg",
    "ffprobe",
    "pdftotext",
    "pdfinfo",
    "gs",
    "socat",
    "nc",
    "bc",
    "vim",
    "git",
    "python3",
    "soffice",
    "pdftoppm",
    "xdg-open",
    "xdg-mime",
    "gio",
    "unzip",
    "zip",
    "curl",
    "wget",
    "gcc",
    "g++",
    "cc",
    "cpp",
    "make",
    "ld",
    "as",
    "ar",
]


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _image_present() -> bool:
    return (
        subprocess.run(
            ["docker", "image", "inspect", _IMAGE],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


@pytest.fixture(scope="module")
def probe() -> dict:
    """Run ONE container that probes the whole agent/env surface, returns parsed JSON.

    Filesystem-based CLI checks (not $PATH) so the result is independent of shell init:
    a CLI is agent-reachable iff it exists in /usr/local/bin, /usr/bin, or /bin.
    """
    py_req = json.dumps(AGENT_PY_REQUIRED)
    py_imp = json.dumps(AGENT_IMPORT_REQUIRED)
    py_forbid = json.dumps(AGENT_PY_FORBIDDEN)
    cli_forbid = json.dumps(CLI_FORBIDDEN)
    cli_req = json.dumps(CLI_REQUIRED)
    snippet = f"""
import json, importlib, importlib.metadata as m, shutil, os

AGENT_DIRS = ["/usr/local/bin", "/usr/bin", "/bin"]
def on_agent_path(cmd):
    return any(os.path.exists(os.path.join(d, cmd)) for d in AGENT_DIRS)

# agent interpreter = system /usr/bin/python3 (this very interpreter, invoked below as 3.10)
py_ver = {{}}
for dist in json.loads('{py_req}'):
    try: py_ver[dist] = m.version(dist)
    except Exception: py_ver[dist] = None

py_import = {{}}
for name in json.loads('{py_imp}'):
    try: importlib.import_module(name); py_import[name] = True
    except Exception: py_import[name] = False

py_forbidden_present = []
for name in json.loads('{py_forbid}'):
    try: importlib.import_module(name); py_forbidden_present.append(name)
    except Exception: pass

cli_forbidden_leaked = [c for c in json.loads('{cli_forbid}') if on_agent_path(c)]
cli_required_missing = [c for c in json.loads('{cli_req}') if not on_agent_path(c)]

# structural invariants
try:
    import pkg_resources; pkg_resources_ok = True
except Exception:
    pkg_resources_ok = False
uv_on_agent_path = on_agent_path("uv")
expect_in_opt_env = os.path.exists("/opt/env/bin/expect")

print("PARITY_JSON:" + json.dumps(dict(
    py_ver=py_ver, py_import=py_import, py_forbidden_present=py_forbidden_present,
    cli_forbidden_leaked=cli_forbidden_leaked, cli_required_missing=cli_required_missing,
    pkg_resources_ok=pkg_resources_ok, uv_on_agent_path=uv_on_agent_path,
    expect_in_opt_env=expect_in_opt_env,
)))
"""
    if not _docker_available():
        pytest.skip("docker not available")
    if not _image_present():
        pytest.skip(f"{_IMAGE} not built (run scripts/install.sh)")
    out = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "/usr/bin/python3", _IMAGE, "-c", snippet],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert out.returncode == 0, f"probe failed:\n{out.stderr}\n{out.stdout}"
    line = next(ln for ln in out.stdout.splitlines() if ln.startswith("PARITY_JSON:"))
    return json.loads(line[len("PARITY_JSON:") :])


@pytest.fixture(scope="module")
def env_probe() -> dict:
    """Probe the ENV venv interpreter (/opt/env/venv/bin/python) — the one every
    in-container getter/setup/eval actually runs on. Entrypoint is the venv python
    directly (not bare `python3`, which would still resolve here via the env PATH, but
    the absolute path makes the target unambiguous). Returns {import_name: installed} +
    the interpreter version.

    Uses find_spec (LOCATE on sys.path), NOT import_module (EXECUTE): the drift this
    guards is "forgot to add the dist to the uv pip install list", i.e. is-it-installed.
    find_spec does not run the module body, so display-coupled packages don't false-fail
    in this headless bare `docker run` — e.g. `import pyautogui` runs `import mouseinfo`,
    which connects to $DISPLAY at import time and would raise DisplayConnectionError here
    (there is no X server on :1), even though pyautogui IS installed. Whether an installed
    package imports cleanly at runtime is the smoke rollout's job, not this gate's."""
    imp = json.dumps(ENV_IMPORT_REQUIRED)
    snippet = f"""
import json, importlib.util, sys
def installed(name):
    try: return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError: return False
env_import = {{name: installed(name) for name in json.loads('{imp}')}}
print("ENV_PARITY_JSON:" + json.dumps(dict(
    env_import=env_import, py=sys.version.split()[0])))
"""
    if not _docker_available():
        pytest.skip("docker not available")
    if not _image_present():
        pytest.skip(f"{_IMAGE} not built (run scripts/install.sh)")
    out = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/opt/env/venv/bin/python",
            _IMAGE,
            "-c",
            snippet,
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert out.returncode == 0, f"env probe failed:\n{out.stderr}\n{out.stdout}"
    line = next(ln for ln in out.stdout.splitlines() if ln.startswith("ENV_PARITY_JSON:"))
    return json.loads(line[len("ENV_PARITY_JSON:") :])


class TestEnvVenvImports:
    """/opt/env/venv (3.12) must carry every package the in-container setup/config/eval
    getters import — the POSITIVE half, symmetric to
    TestAgentPythonParity.test_env_only_packages_absent_from_agent (which asserts these
    same libs stay OFF the agent 3.10)."""

    def test_env_required_imports_available(self, env_probe):
        missing = [n for n, ok in env_probe["env_import"].items() if not ok]
        assert not missing, (
            f"/opt/env/venv is missing packages the getters import (not installed) — add "
            f"the dist to the `uv pip install --python /opt/env/venv/bin/python` list in "
            f"lite/gym/envs/lite/osworld/docker/Dockerfile: {missing}"
        )

    def test_env_venv_is_python_312(self, env_probe):
        # Sanity: the env getters must run on the 3.12 venv, not accidentally the agent
        # 3.10 — pins the interpreter the ENV_IMPORT_REQUIRED assertion was measured on.
        assert env_probe["py"].startswith("3.12"), (
            f"/opt/env/venv python is {env_probe['py']!r}, expected 3.12 "
            "(the env venv interpreter, not the agent 3.10)"
        )


class TestAgentPythonParity:
    """system 3.10 = the osworld guest mirror (versions locked)."""

    def test_required_packages_present_and_versioned(self, probe):
        wrong = {
            d: (exp, probe["py_ver"][d])
            for d, exp in AGENT_PY_REQUIRED.items()
            if probe["py_ver"][d] != exp
        }
        assert not wrong, f"agent-3.10 package drift (dist: (expected, actual)): {wrong}"

    def test_required_imports_available(self, probe):
        missing = [n for n, ok in probe["py_import"].items() if not ok]
        assert not missing, f"agent-3.10 missing imports (guest has them): {missing}"

    def test_env_only_packages_absent_from_agent(self, probe):
        assert not probe["py_forbidden_present"], (
            f"env-only libs LEAKED onto agent 3.10 (guest lacks them): "
            f"{probe['py_forbidden_present']}"
        )


class TestAgentCliParity:
    """default-PATH CLIs = the osworld guest mirror (VM-absent tools vendored into /opt/env)."""

    def test_env_only_clis_off_agent_path(self, probe):
        assert not probe["cli_forbidden_leaked"], (
            f"env-only CLIs LEAKED onto agent PATH (guest lacks them; incl. ImageMagick "
            f"versioned binaries / pdftk real binary): {probe['cli_forbidden_leaked']}"
        )

    def test_guest_clis_present_on_agent_path(self, probe):
        assert not probe["cli_required_missing"], (
            f"guest CLIs missing from agent PATH: {probe['cli_required_missing']}"
        )


class TestStructuralInvariants:
    """the boot + isolation guards this refactor added."""

    def test_pkg_resources_present_on_agent_310(self, probe):
        # supervisord (PID 1) imports pkg_resources; if a system-level pip clobbers the
        # shared apt setuptools, the container never boots. This is that regression's gate.
        assert probe["pkg_resources_ok"], "pkg_resources missing on 3.10 → supervisord can't boot"

    def test_uv_off_agent_path(self, probe):
        assert not probe["uv_on_agent_path"], "uv (env tooling) leaked onto the agent PATH"

    def test_expect_not_env_vendored_into_opt_env(self, probe):
        assert not probe["expect_in_opt_env"], "expect must stay agent-facing at /usr/bin/expect"


# --- North Star 3: the default-user RUNTIME contract (the posture dfd573fd broke) ------
# Probe the agent AS the desktop user in a LOGIN shell, so $PATH/$HOME/locale are exactly
# what a terminal the agent opens would see. This is the guarantee that the agent/env
# separation holds WITHOUT a uid wall: the agent IS the (password-gated) sudoer `user`,
# yet /opt/env is off its login PATH.
_NS3_LOGIN_PROBE = r"""
set +e
sudo -n true 2>/dev/null; NS3_SUDO_N=$?
echo user | sudo -S -k true 2>/dev/null; NS3_SUDO_PW=$?
export NS3_SUDO_N NS3_SUDO_PW
exec /usr/bin/python3 - <<'PY'
import os, json, subprocess
def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
pw = sh(["getent", "passwd", "user"]).split(":")
path = os.environ.get("PATH", "")
print("NS3_JSON:" + json.dumps(dict(
    uid=sh(["id", "-u"]),
    groups=sh(["id", "-Gn"]).split(),
    sudo_n_rc=int(os.environ.get("NS3_SUDO_N", "-1")),
    sudo_pw_rc=int(os.environ.get("NS3_SUDO_PW", "-1")),
    login_path=path,
    opt_env_on_path=any(p.startswith("/opt/env") for p in path.split(":")),
    home_env=os.environ.get("HOME", ""),
    home_getent=pw[5] if len(pw) > 6 else "",
    shell_getent=pw[6] if len(pw) > 6 else "",
    lang=os.environ.get("LANG", ""),
    lc_all=os.environ.get("LC_ALL", ""),
)))
PY
"""


@pytest.fixture(scope="module")
def agent_login() -> dict:
    """One container run AS `user` in a login shell (`bash -lc`), probing the runtime
    default-user contract. Returns parsed JSON."""
    if not _docker_available():
        pytest.skip("docker not available")
    if not _image_present():
        pytest.skip(f"{_IMAGE} not built (run scripts/install.sh)")
    out = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-u",
            "user",
            "--entrypoint",
            "/bin/bash",
            _IMAGE,
            "-lc",
            _NS3_LOGIN_PROBE,
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert out.returncode == 0, f"NS3 login probe failed:\n{out.stderr}\n{out.stdout}"
    line = next(ln for ln in out.stdout.splitlines() if ln.startswith("NS3_JSON:"))
    return json.loads(line[len("NS3_JSON:") :])


class TestNorthStar3Behavioral:
    """Runtime default-user guards: the agent IS a password-gated sudoer `user`, /opt/env
    is off its login PATH, and its identity/home/locale mirror the osworld VM."""

    def test_agent_user_in_group_sudo(self, agent_login):
        assert "sudo" in agent_login["groups"], (
            f"desktop user not in group sudo: {agent_login['groups']}"
        )

    def test_sudo_is_password_gated_not_nopasswd(self, agent_login):
        # VM-faithful: `sudo -n` (non-interactive) FAILS, password-piped `sudo -S` SUCCEEDS.
        assert agent_login["sudo_n_rc"] != 0, (
            "`sudo -n true` succeeded → NOPASSWD grant present (must be password-gated)"
        )
        assert agent_login["sudo_pw_rc"] == 0, (
            "`echo user | sudo -S true` failed → password-gated sudo is unusable"
        )

    def test_opt_env_off_agent_login_path(self, agent_login):
        assert not agent_login["opt_env_on_path"], (
            f"/opt/env leaked onto the agent login PATH (soft separation broken): "
            f"{agent_login['login_path']}"
        )

    def test_agent_identity_and_home(self, agent_login):
        assert agent_login["uid"] == "1000", f"desktop user uid != 1000: {agent_login['uid']}"
        assert agent_login["home_getent"] == "/home/user", agent_login["home_getent"]
        assert agent_login["home_env"] == "/home/user", agent_login["home_env"]
        assert agent_login["shell_getent"] == "/bin/bash", agent_login["shell_getent"]

    def test_agent_locale_matches_vm(self, agent_login):
        # osworld VM system locale is en_HK.UTF-8 (Dockerfile pins LANG + LC_ALL).
        assert agent_login["lang"] == "en_HK.UTF-8", agent_login["lang"]
        assert agent_login["lc_all"] == "en_HK.UTF-8", agent_login["lc_all"]
