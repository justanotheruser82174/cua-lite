"""Every third-party module a pinned cuagym bundle imports must be CLASSIFIED.

Run: ``uv run --no-sync pytest -q tests/gym/envs/lite/cuagym/test_cuagym_bundle_imports.py``

Why this exists — the `reward_judge` incident, which took months to surface:

45 bundles do ``from reward_judge import call_llm_judge``. Nothing in this repo ever
wrote ``/tmp/reward_judge.py`` (both ``scripts.py`` wrote only ``/tmp/reward.py``).
Upstream wraps every one of those 46 call sites in ``try/except``, so the import
failure produced **no error and no log** — the LLM component just scored 0, which is
byte-identical to "the agent did not earn those points". Three further layers hid it:

* the component is only 0.05-0.30 of the reward, so tasks still scored nonzero;
* the one guard that could have caught it (``cuagym/test_reward_import_presence.py``)
  scans the **desktop** bundle dir, and 44 of the 46 importers are **web**;
* that guard is ``skipif``-gated on an in-container interpreter, and this repo has no
  CI — so it never ran anywhere, ever.

The lesson generalised: **"nobody looked" must not be byte-identical to "it is fine."**
So this test refuses to let an import go unclassified. It runs on the DEV HOST with no
container and no skip — deliberately, because a guard that skips is not a guard.

It does NOT assert that a module is importable here: cuagym rewards execute in the
guest (``/opt/env/venv``), and asserting host importability would be the exact
wrong-environment error this campaign kept making. It asserts only that every module
has a recorded DISPOSITION. A materials bump that introduces an unknown import fails
here, loudly, on a laptop — instead of silently costing reward in a rollout.
"""

from __future__ import annotations

import ast
import collections
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[5]
_CUAGYM_DIR = _REPO / "lite/gym/envs/lite/cuagym"
_BUNDLE_ROOTS = sorted(
    path
    for path in (
        _CUAGYM_DIR / ".cache/web/lite.cuagym_tasks/bundles",
        _CUAGYM_DIR / ".cache/desktop/lite.cuagym_desktop_tasks/bundles",
    )
    if path.is_dir()
)

#: Supplied by the pinned image's ``/opt/env/venv`` (where cuagym rewards run).
#: PROBED against `cua-lite/lite.cuagym:latest`, not asserted from the Dockerfile —
#: `test_guest_venv_classification_matches_the_real_image` below re-derives it from
#: the running image, which is what makes this set falsifiable. The earlier hand-
#: maintained version listed `pyods` and `torch` here while both were
#: ABSENT; nothing could tell, because no test looked.
#: cv2/pdfminer/toml need the docker/Dockerfile ENV_PY layer that now PINS
#: opencv-python-headless / pdfminer.six / toml, so they require a rebuilt image. An
#: older build resolved them transitively and a later one did not — which is exactly
#: why this is probed against the running image and never read off the Dockerfile.
_GUEST_VENV = frozenset({
    "PIL", "PyPDF2", "Xlib", "bs4", "chardet", "cv2", "dateutil", "dbf", "distutils",
    "docx", "ezodf", "fastapi", "fitz", "fpdf", "gimpformats", "gymnasium", "lxml",
    "matplotlib", "msoffcrypto", "mutagen", "numpy", "odf", "openpyxl", "packaging",
    "pandas", "pdfminer", "piexif", "pikepdf", "pip", "playwright", "pptx",
    "pyautogui", "pyexcel_ods3", "pyhanko", "pymupdf", "pypdf", "pyperclip",
    "pytest", "reportlab", "requests", "scipy", "seaborn", "skimage", "toml",
    "urllib3", "yaml",
})

#: System-ABI modules deliberately NOT in the env venv: ``_python_for_source`` routes
#: a source importing these to the system interpreter (UNO_PY) instead. Absent from
#: ``/opt/env/venv`` **by design**, not by omission.
_SYSTEM_ABI = frozenset({"gi", "dbus", "uno", "unohelper", "com"})

#: WE must supply these — upstream's harness provides them and ours must too.
#: ``reward_judge`` is the incident above; ``install_reward_judge`` now writes it.
_SHIMMED = frozenset({"reward_judge"})

#: The module IS the deliverable: the task asks the agent to write or repair it
#: (e.g. bundle bc329a7f… is "fix the backtracking in my Sudoku solver"). None of
#: these exists in its bundle dir — verified. Here an ImportError legitimately means
#: the agent did not do the task, so upstream's try/except → 0 is CORRECT.
#: This is the one class where swallowing the import is right, which is precisely why
#: the pattern survived long enough to hide `reward_judge`.
_AGENT_ARTIFACT = frozenset({
    "alien", "board", "bullet", "deck", "game", "generator", "migrate", "solver",
    "student_db", "tracker", "utils", "validator",
})

#: Genuinely absent from the guest and NOT supplied by anyone — recorded rather than
#: silently tolerated. Promote to `_GUEST_VENV` if ever baked, or fix upstream.
#:   torch, stable_baselines3  2 reward.py (57e6a588 wants both, 392066aa wants torch)
#:     — ~2.5 GB onto a 6.7 GB image for 2 of the 10910 pinned bundles. Skipped on
#:     cost, deliberately, not overlooked.
#:   pyods  1 initial_setup.py (a5ed1cb4), and it is a DEAD import: `try: import pyods
#:     / except ImportError: pass`, after which the function builds the .ods with
#:     odfpy, which is installed. Nothing is lost.
#:   tomli  no importer left in the pinned corpus; kept only so a re-materialisation
#:     that brings it back does not read as new.
_KNOWN_MISSING = frozenset({
    "pyods", "stable_baselines3", "tomli", "torch",
})

_CLASSIFIED = (
    _GUEST_VENV | _SYSTEM_ABI | _SHIMMED | _AGENT_ARTIFACT | _KNOWN_MISSING
)


def _bundle_imports() -> dict[str, int]:
    """Top-level third-party module -> number of bundle files importing it."""
    counts: collections.Counter[str] = collections.Counter()
    for root in _BUNDLE_ROOTS:
        for path in root.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(errors="replace"))
            except (OSError, SyntaxError):
                continue  # broken bundles are their own (already-tagged) problem
            names: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names.add(node.module.split(".")[0])
            for name in names - sys.stdlib_module_names:
                counts[name] += 1
    return dict(counts)


pytestmark = pytest.mark.skipif(
    not _BUNDLE_ROOTS, reason="cuagym bundles not materialised on this host"
)


def test_every_bundle_import_is_classified():
    """An unknown import is a silent reward leak until proven otherwise."""
    found = _bundle_imports()
    unknown = sorted(set(found) - _CLASSIFIED)
    assert not unknown, (
        "cuagym bundles import modules with no recorded disposition: "
        + ", ".join(f"{m} ({found[m]} bundles)" for m in unknown)
        + ".\nClassify each in this file: _GUEST_VENV (baked in the image) / "
        "_SYSTEM_ABI (routed to UNO_PY) / _SHIMMED (we must write it) / "
        "_AGENT_ARTIFACT (the task asks the agent to create it) / _KNOWN_MISSING. "
        "Do NOT silence this by adding to _KNOWN_MISSING without checking the guest: "
        "reward_judge cost 45 bundles their LLM score precisely because an absent "
        "module and an unearned score are indistinguishable from the outside."
    )


def test_the_shimmed_modules_are_actually_written():
    """`_SHIMMED` is a promise; break it and we are back to the original incident."""
    from lite.gym.envs.lite.cuagym.src.utils import runtime

    src = runtime.reward_judge_source()
    assert "def call_llm_judge(" in src
    ast.parse(src)  # the generated module must at least be valid Python

    found = _bundle_imports()
    assert found.get("reward_judge", 0) >= 40, (
        "reward_judge importers vanished from the corpus — if upstream dropped it, "
        "retire the shim; do not leave a promise with nothing behind it."
    )


_IMAGE = "cua-lite/lite.cuagym:latest"
#: Where each class is expected to resolve. ENV_PY runs rewards/setup; UNO_PY exists
#: only because LibreOffice's bridge is Python-3.10-ABI (see docker/Dockerfile).
_ENV_PY = "/opt/env/venv/bin/python"
_UNO_PY = "/opt/env/uno-venv/bin/python"

#: One FRESH interpreter per module, deliberately. Checking them all in a single
#: process is contaminating: `find_spec("distutils")` fires setuptools'
#: `_distutils_hack`, which imports setuptools and puts its vendored `tomli` on
#: sys.path — after which `tomli` probes as PRESENT although a bundle that imports it
#: directly still gets ModuleNotFoundError. A false "present" is the failure direction
#: that hides gaps, so pay the ~30 ms per module.
_PROBE = """
import json, subprocess, sys
one = "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)"
print(json.dumps({
    m: subprocess.run([sys.executable, "-c", one, m], capture_output=True).returncode == 0
    for m in sys.argv[1].split(",")
}))
"""


def _probe(interpreter: str, modules: list[str]) -> dict[str, bool] | None:
    if shutil.which("docker") is None:
        return None
    if subprocess.run(["docker", "image", "inspect", _IMAGE],
                      capture_output=True).returncode != 0:
        return None
    proc = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", interpreter, _IMAGE,
         "-c", _PROBE, ",".join(sorted(modules))],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"dependency probe failed for {_IMAGE} via {interpreter} "
            f"(rc={proc.returncode}): {proc.stderr or proc.stdout}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_guest_venv_classification_matches_the_real_image():
    """`_GUEST_VENV` is a factual claim about the image; check it against the image.

    Without this the classification above is unfalsifiable — and it WAS wrong:
    cv2/pdfminer/pip/pyods/torch sat in `_GUEST_VENV` while absent from the venv, so
    five reward paths were scoring 0 with the test passing. `_KNOWN_MISSING` is
    checked in the same breath, so a package that quietly starts shipping gets
    promoted rather than left behind a stale excuse."""
    found = _bundle_imports()
    result = _probe(_ENV_PY, sorted(_GUEST_VENV | (_KNOWN_MISSING & set(found))))
    if result is None:
        pytest.skip(f"{_IMAGE} not built on this host")
    absent = sorted(m for m in _GUEST_VENV if not result.get(m, False))
    present = sorted(m for m in _KNOWN_MISSING & set(found) if result.get(m, False))
    assert not absent and not present, (
        f"ENV_PY ({_ENV_PY}) disagrees with this file.\n"
        f"  classified _GUEST_VENV but ABSENT: {absent}\n"
        f"  classified _KNOWN_MISSING but PRESENT: {present}\n"
        "Absent ⇒ a reward path is scoring 0 for an environment reason: bake it into "
        "the ENV_PY layer of lite/gym/envs/lite/cuagym/docker/Dockerfile (never into "
        "the agent's /usr/bin/python3). Present ⇒ move it to _GUEST_VENV."
    )


def test_system_abi_modules_resolve_under_uno_py():
    """`_SYSTEM_ABI` claims these are absent from ENV_PY *by design* and reachable via
    UNO_PY. `com` is the trap: `find_spec("com")` fails even in UNO_PY, yet
    `import uno; from com.sun.star.beans import PropertyValue` works, because the
    names come from uno.py's import hook and not a `com` package. Probing it the naive
    way reports a gap that is not there."""
    checkable = sorted(_SYSTEM_ABI - {"com"})
    result = _probe(_UNO_PY, checkable)
    if result is None:
        pytest.skip(f"{_IMAGE} not built on this host")
    absent = sorted(m for m in checkable if not result[m])
    assert not absent, (
        f"UNO_PY ({_UNO_PY}) cannot import {absent} — `_python_for_source` routes "
        "sources importing these there, so they would fail at runtime"
    )


def test_agent_artifact_modules_are_absent_from_their_bundles():
    """Pins the reasoning that makes `_AGENT_ARTIFACT` safe to swallow.

    If one of these ever ships inside its bundle, it stops being a deliverable and a
    failed import stops meaning "the agent did not do the task" — at which point the
    try/except is hiding a real defect again."""
    shipped = []
    for root in _BUNDLE_ROOTS:
        for path in root.rglob("*.py"):
            for name in _AGENT_ARTIFACT:
                if (path.parent / f"{name}.py").exists():
                    shipped.append(f"{path.parent.name}/{name}.py")
    assert not shipped, f"agent-artifact modules now ship in-bundle: {sorted(set(shipped))}"
