"""A module a hook imports IN THE CONTAINER must be present IN THAT CONTAINER.

Run: ``uv run --no-sync pytest -q tests/gym/envs/lite/cuaworld/test_cuaworld_container_deps.py``
(the image-probing test needs the per-software images; the manifest test does not)

Why: a missing dependency must ERROR, never become a silent 0. cuaworld hooks embed
``python3 <<'PY' … PY`` bodies and ``python3 -c "…"`` one-liners that run in the
container, and upstream wraps most of those imports in ``try/except`` — so an absent
module yields a partial or empty result JSON and the task scores low for a reason that
has nothing to do with the agent. In a gate whose whole job is separating "the env
broke" from "the agent failed", that is the one failure mode we cannot tolerate.

The trap this codifies: ``scripts/install.sh``'s ``install_python_deps`` already
installs ``pynrrd`` and ``pyshp`` — **on the HOST**, for the host-side ``verifier.py``.
The same module names are imported by hook bodies that run **in the container**, and
until the per-software layer in ``docker/Dockerfile`` nothing installed them there.
Fixing the host and missing the guest is the wrong-environment error this codebase
keeps repeating, so this test deliberately probes the image, not ``sys.path``.

Three things make a module NOT a gap, and each has its own recorded reason:
``_AGENT_ARTIFACT`` (the task asks the agent to write it), ``_RUNTIME_PROVIDED`` (the
hook itself installs it or puts it on ``sys.path``), and simply being in the image.
Anything else is a gap and fails here.
"""

from __future__ import annotations

import ast
import collections
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys

import pytest

from lite.gym.envs.lite.cuaworld.src.adapter import _apply_hook_execution_fixups

_REPO = pathlib.Path(__file__).resolve().parents[5]
_CUAWORLD = _REPO / "lite/gym/envs/lite/cuaworld"
_MATERIALS = _CUAWORLD / ".cache"
_SESSION_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

#: `python3 <<'PY' … PY` / `python3 - <<EOF … EOF`. The `pre` group must name a python
#: on the SAME line, which is what keeps `cat > foo.py << 'EOF'` — a heredoc that
#: WRITES python rather than running it — out of the manifest.
_HEREDOC = re.compile(
    r"""[^\n]*?\bpython3?\b[^\n<]*<<-?[ \t]*(?P<q>['"]?)(?P<tag>[A-Za-z_]\w*)(?P=q)[^\n]*\n"""
    r"""(?P<body>.*?)\n[ \t]*(?P=tag)[ \t]*$""",
    re.S | re.M,
)
#: `python3 -c "…"`. NOT optional: 6 of the gaps this test exists for live only in
#: `-c` bodies (gvsig `shapefile`, gretl/dbeaver `pandas`, pycharm `PIL`/`cv2`), so a
#: heredoc-only extractor reports them as clean.
_DASHC = re.compile(
    r"""\bpython3?\b[ \t]+(?:-[A-Za-z]+[ \t]+)*-c[ \t]+(?P<q>['"])(?P<body>.*?)(?<!\\)(?P=q)""",
    re.S,
)
#: `cat > /tmp/check.py <<'EOF' ... EOF` followed later by
#: `python3 /tmp/check.py`. These are executed hook Python too; missing them hid
#: astroimagej/generate_inverted_finding_chart's `cv2` dependency until a live
#: rollout produced `/tmp/task_result.json` missing from `ModuleNotFoundError`.
_GENERATED_PY = re.compile(
    r"""\bcat\b[^\n>]*>[ \t]*(?P<q>['"]?)(?P<path>[^\s'"]+\.py)(?P=q)"""
    r"""[^\n]*<<-?[ \t]*(?P<tq>['"]?)(?P<tag>[A-Za-z_]\w*)(?P=tq)[^\n]*\n"""
    r"""(?P<body>.*?)\n[ \t]*(?P=tag)[ \t]*$""",
    re.S | re.M,
)
_GENERATED_PY_EXEC = re.compile(
    r"""(?<![-A-Za-z0-9_])python3?\b[ \t]+(?:-[A-Za-z]+[ \t]+)*"""
    r"""(?P<q>['"]?)(?P<path>[^\s'"]+\.py)(?P=q)(?=\s|$)"""
)
#: Unquoted heredoc tags let the shell expand `$VAR` / `$(cmd)` into the body, which
#: is then not valid Python. Blank them out and re-parse rather than dropping the body.
_SHELL_EXPANSION = re.compile(
    r"\$\((?:[^()]|\([^()]*\))*\)|\$\{[^}]*\}|\$[A-Za-z_]\w*|`[^`]*`"
)

#: The module IS the agent's deliverable — the task asks for it, so ImportError there
#: legitimately means the agent did not do the work. Same class as cuagym's
#: `_AGENT_ARTIFACT`; see test_cuagym_bundle_imports.py. All three are reached through
#: a `sys.path.insert` into the agent's own workspace, never a package install.
_AGENT_ARTIFACT = frozenset({"models", "utils", "cif_parser"})

#: (software, module) that is absent from the image but reachable at RUNTIME, with the
#: mechanism that makes it so. Verified per entry against the pinned hooks and the
#: built image; re-verify before adding, because "the setup probably installs it" is
#: exactly the assumption that hides a real gap. Two of these were checked and turned
#: out FALSE — gvsig's `osgeo` (the tasks that apt-get gdal-bin are a disjoint set from
#: the tasks that import osgeo) and slicer3d's `pydicom` (self-healing per task, but
#: not in the two shared fixture builders) — and both are baked in the image instead.
_RUNTIME_PROVIDED: dict[tuple[str, str], str] = {
    ("sumo", "sumolib"): (
        "ships with SUMO; both hook bodies sys.path.insert($SUMO_HOME/tools) first "
        "and /usr/share/sumo/tools/sumolib imports cleanly there (verified in image)"
    ),
    ("slicer3d", "skimage"): (
        "both tasks run `pip3 install -q … scikit-image` in their own setup_task.sh"
    ),
    ("solvespace", "numpy"): "its one task apt-installs python3-numpy in setup_task.sh",
    ("solvespace", "trimesh"): "its one task pip-installs trimesh in setup_task.sh",
    ("pycharm", "cv2"): (
        "debug_document_scanner/setup_task.sh pip-installs opencv-python-headless"
    ),
    ("pycharm", "PIL"): (
        "fix_raytracer_prototype/setup_task.sh pip-installs Pillow; the other task "
        "(fix_steganography_tool) has no provider but both are in "
        "data/validation_excludes.json as nonzero_baseline, so ZERO live tasks"
    ),
    ("vscode", "ortools"): (
        "fix_delivery_routing_engine/setup_task.sh pip-installs it unconditionally "
        "under `set -e`, so a failure is loud; >100 MB for one excluded task"
    ),
    ("vscode", "api"): "workspace package created by setup_task.sh; export adds it to sys.path",
    ("vscode", "analytics"): "workspace package created by setup_task.sh",
    ("vscode", "camera"): "workspace module created by setup_task.sh",
    ("vscode", "config"): "workspace module created by setup_task.sh",
    ("vscode", "fastapi"): "debug_ml_model_api/setup_task.sh pip-installs it before export",
    ("vscode", "geometry"): "workspace module created by setup_task.sh",
    ("vscode", "joblib"): "debug_ml_model_api/setup_task.sh pip-installs it before export",
    ("vscode", "pipeline"): "workspace package created by setup_task.sh",
    ("vscode", "rasterizer"): "workspace module created by setup_task.sh",
    ("vscode", "sklearn"): (
        "debug_ml_model_api/setup_task.sh pip-installs scikit-learn before export"
    ),
    ("vscode", "spice_parser"): "workspace package created by setup_task.sh",
    ("vscode", "src"): "workspace package created by setup_task.sh",
    ("vscode", "transformers"): (
        "workspace package created by setup_task.sh; not HuggingFace transformers"
    ),
    ("coppeliasim", "coppeliasim_zmqremoteapi_client"): (
        "app ships it at /opt/CoppeliaSim/programming/zmqRemoteApi/clients/python/src "
        "and 3 of 5 sites sys.path.insert exactly that; the 2 that insert the parent "
        "without /src are broken, but all 4 tasks are excluded (missing_verifier) and "
        "the fifth site (task_utils.sh get_scene_object_count) has ZERO callers"
    ),
}


def _module_imports(body: str) -> set[str]:
    """Top-level third-party modules imported by one python body."""
    for text in (body, _SHELL_EXPANSION.sub("_SH_", body)):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
        return names - sys.stdlib_module_names
    return set()


def _hook_imports() -> dict[str, dict[str, set[str]]]:
    """software -> module -> {task dirs whose hook bodies import it}."""
    out: dict[str, dict[str, set[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(set)
    )
    for sh in _MATERIALS.glob("*/*/**/*.sh"):
        software = sh.relative_to(_MATERIALS).parts[0]
        text = sh.read_text(errors="replace")
        text = _apply_hook_execution_fixups(text, sh.parent.name, sh.name)
        bodies = [m.group("body") for m in _HEREDOC.finditer(text)]
        bodies += [m.group("body") for m in _DASHC.finditer(text)]
        generated = {m.group("path"): m.group("body") for m in _GENERATED_PY.finditer(text)}
        executed = {m.group("path") for m in _GENERATED_PY_EXEC.finditer(text)}
        bodies += [generated[path] for path in sorted(generated.keys() & executed)]
        for body in bodies:
            for name in _module_imports(body) - _AGENT_ARTIFACT:
                out[software][name].add(sh.parent.name)
    return {sw: dict(mods) for sw, mods in out.items()}


#: One FRESH interpreter per module. Checking them all in a single process is
#: contaminating — `find_spec("distutils")` fires setuptools' `_distutils_hack`,
#: which puts setuptools' vendored packages on sys.path and makes modules probe as
#: PRESENT that a hook importing them directly still cannot find. A false "present"
#: is the failure direction that hides gaps, so pay the ~30 ms per module.
_PROBE = """
import json, subprocess, sys
one = "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)"
print(json.dumps({
    m: subprocess.run([sys.executable, "-c", one, m], capture_output=True).returncode == 0
    for m in sys.argv[1].split(",")
}))
"""


def _probe(image: str, modules: list[str]) -> dict[str, bool] | None:
    """Ask the real image which of `modules` its hook-session python3 can import.

    Not /opt/env/venv: `adapter._run_hook` exports the desktop session PATH before
    running a hook, so a hook's bare `python3` is the session-path one.
    """
    if subprocess.run(["docker", "image", "inspect", image],
                      capture_output=True).returncode != 0:
        return None
    command = (
        f"export PATH={shlex.quote(_SESSION_PATH)}; "
        f"exec python3 -c {shlex.quote(_PROBE)} {shlex.quote(','.join(sorted(modules)))}"
    )
    proc = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "/bin/bash", image, "-lc", command],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"dependency probe failed for {image} (rc={proc.returncode}): "
            f"{proc.stderr or proc.stdout}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _required_images() -> set[str]:
    raw = os.environ.get("CUAWORLD_REQUIRE_IMAGES", "")
    return set(raw.replace(",", " ").split())


def _require_full_pinned_materials() -> None:
    libraries = sorted(_MATERIALS.glob("*/*/scripts/task_utils.sh"))
    if len(libraries) < 35:
        pytest.skip("cuaworld pinned materials not fully fetched on this host")


@pytest.mark.skipif(not list(_MATERIALS.glob("*")),
                    reason="cuaworld materials not fetched on this host")
def test_hook_import_manifest_is_derivable_without_a_container():
    """The requirement set must be computable on a laptop, so the probe below can be
    reasoned about even where images are absent. A guard nobody can run is not a
    guard — that is how `test_reward_import_presence.py` went years without firing."""
    _require_full_pinned_materials()
    need = _hook_imports()
    assert need, "no hook python bodies found — the extractor regexes have rotted"
    assert "slicer3d" in need and "nrrd" in need["slicer3d"], (
        "the known-worst case vanished; if the materials moved, re-derive this test's "
        "premise rather than deleting the assertion"
    )
    # The `-c` extractor is load-bearing: these three are invisible to a heredoc-only
    # scan, and two of them are UNGUARDED imports on the scoring path.
    for software, module in (("gretl", "pandas"), ("dbeaver", "pandas"),
                             ("pycharm", "cv2")):
        assert module in need.get(software, {}), (
            f"{software}/{module} disappeared from the manifest — if `_DASHC` stopped "
            f"matching, every `python3 -c` gap is now invisible"
        )
    assert "generate_inverted_finding_chart" in need["astroimagej"]["cv2"], (
        "generated-then-executed hook Python is invisible again; this live task "
        "needs cv2 in the AstroImageJ image or its export hook writes no result JSON"
    )


def test_runtime_image_marks_task_workspaces_git_safe():
    """Post-task hooks must be able to inspect repos the agent edited.

    Git's dubious-ownership guard otherwise turns a real commit into empty hook
    metadata, which under-scores tasks such as vscode/fix_aerospace_telemetry_parser
    for an environment reason.
    """
    dockerfile = (_CUAWORLD / "docker/Dockerfile").read_text()

    assert "git config --system --add safe.directory '/home/ga/workspace/*'" in dockerfile
    assert "git config --system --add safe.directory '*'" not in dockerfile


def test_runtime_image_dependency_repairs_fail_the_build():
    """Image dependency repair must not defer an import crash to reward time."""
    dockerfile = (_CUAWORLD / "docker/Dockerfile").read_text()

    assert "scipy repair unavailable" not in dockerfile
    assert "|| echo \"WARN" not in dockerfile


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker unavailable")
def test_every_hook_import_is_present_in_its_own_image():
    """Missing in-container dependency ⇒ loud failure here, never a silent 0 later."""
    need = _hook_imports()
    missing: dict[str, dict[str, int]] = {}
    missing_required: list[str] = []
    required = _required_images()
    probed = 0
    for software, mods in sorted(need.items()):
        wanted = sorted(m for m in mods if (software, m) not in _RUNTIME_PROVIDED)
        if not wanted:
            continue
        result = _probe(f"cua-lite/lite.cuaworld.{software}:latest", wanted)
        if result is None:
            if "*" in required or software in required:
                missing_required.append(software)
            continue  # image not built on this host; the manifest test still applies
        probed += 1
        gaps = {m: len(mods[m]) for m, ok in result.items() if not ok}
        if gaps:
            missing[software] = gaps
    assert not missing_required, (
        "CUAWORLD_REQUIRE_IMAGES requested images that are absent or not runnable: "
        + ", ".join(sorted(missing_required))
    )
    if not probed:
        pytest.skip("no cuaworld images built on this host")
    assert not missing, (
        "hook bodies import modules their own image cannot provide — these score "
        "partial/zero for an environment reason, not an agent reason:\n"
        + "\n".join(
            f"  {sw}: " + ", ".join(f"{m} ({n} tasks)" for m, n in sorted(g.items()))
            for sw, g in sorted(missing.items())
        )
        + "\nFix by baking the package into that software's branch of the per-SOFTWARE "
        "layer in lite/gym/envs/lite/cuaworld/docker/Dockerfile. Do NOT satisfy this "
        "by adding to install_python_deps — that installs on the HOST for verifier.py; "
        "these imports run in the CONTAINER. Confusing the two is the original defect. "
        "If the hook really does provide it at runtime, record the mechanism in "
        "_RUNTIME_PROVIDED — and verify it, do not assume it."
    )


@pytest.mark.skipif(not list(_MATERIALS.glob("*")),
                    reason="cuaworld materials not fetched on this host")
def test_exemptions_still_describe_a_real_import():
    """An exemption that no longer matches anything is dead weight that makes the next
    reader trust a claim nobody is checking. Both lists must stay grounded."""
    _require_full_pinned_materials()
    need = _hook_imports()
    stale = [f"{sw}/{m}" for (sw, m) in _RUNTIME_PROVIDED
             if m not in need.get(sw, {})]
    assert not stale, (
        f"_RUNTIME_PROVIDED entries no longer imported by any hook: {stale} — delete "
        "them, or the next gap gets waved through by a rule that stopped applying"
    )
