"""cuagym reward import-presence guard — every ENV_PY reward dep is installed.

Every cuagym setup/reward runs as the **desktop user** (``SandboxBaseEnv.EXEC_USER
= "user"``); the only per-script choice is WHICH env interpreter runs it, and
:func:`_python_for_source` makes it purely on library/ABI availability: the py3.10
uno-venv (``UNO_PY``) for a source importing the LibreOffice UNO / PyGObject
bridge, the py3.12 env-venv (``/opt/env/venv/bin/python`` = ``ENV_PY``) for
everything else. ``/opt/env`` is ``a+rX`` in the image, so the desktop user execs
``ENV_PY`` directly — no wrapper, no uid switch. That makes ``/opt/env/venv`` the
single place every non-gi/uno reward py-dep must be installed.

This guard imports the top-level modules of every materialized
``bundles/*/reward.py`` whose ``_python_for_source == ENV_PY`` under that exact
interpreter and reds on a ModuleNotFoundError — proving PRESENCE (a missing module
means "add it to ``/opt/env/venv``" per the install-to-``/opt`` policy). It says
nothing about *versions*: a dep that is present but semantically different from
what the upstream reward was authored against still passes here.

Known over-approximation: a reward's top-level imports also include modules the
TASK itself is expected to have produced in the reward's working directory
(``import board`` / ``import solver`` / ``import student_db`` …). Those are not
installable deps — a red on such a name means the probe's cwd is wrong, NOT
"pip-install a package by that name".

``/opt/env/venv`` exists only inside the image, so this test SKIPS on the dev host.
There is no CI in this repo (no ``.github/workflows``): it executes only when
pytest is run inside a lite.cuagym container with the repo mounted.

Run (in-container): pytest tests/gym/envs/lite/cuagym/test_reward_import_presence.py -q
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import warnings
from pathlib import Path

import pytest

from lite.gym.envs.lite.cuagym.src.desktop.scripts import _python_for_source
from lite.gym.envs.lite.cuagym.src.utils.runtime import ENV_PY

_REPO = Path(__file__).resolve().parents[5]
_BUNDLES = (
    _REPO / "lite/gym/envs/lite/cuagym/.cache/desktop"
    / "lite.cuagym_desktop_tasks/bundles"
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(ENV_PY),
    reason=f"{ENV_PY} absent — import-presence runs in-container, not on the dev host",
)


def _top_level_imports(source: str) -> set[str]:
    """Top-level module names an absolute ``import`` / ``from x import`` pulls in
    (``import a.b`` → ``a``; ``from x.y import z`` → ``x``). Relative imports and
    ``__future__`` are excluded."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except SyntaxError:
        return set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    modules.discard("__future__")
    return modules


def _env_py_reward_modules() -> dict[str, set[str]]:
    """{module: {reward_labels…}} over every ENV_PY (non-gi/uno) reward."""
    by_module: dict[str, set[str]] = {}
    for reward in sorted(_BUNDLES.rglob("reward.py")):
        source = reward.read_text()
        if _python_for_source(source) != ENV_PY:
            continue
        label = str(reward.relative_to(_REPO))
        for module in _top_level_imports(source):
            by_module.setdefault(module, set()).add(label)
    return by_module


@pytest.mark.skipif(not _BUNDLES.is_dir(), reason="cuagym bundles absent")
def test_env_py_reward_top_level_modules_are_importable():
    by_module = _env_py_reward_modules()
    assert by_module, "no ENV_PY reward modules found to check"
    modules = sorted(by_module)
    # One interpreter invocation: import each, collect the ModuleNotFound offenders.
    probe = (
        "import importlib, json, sys\n"
        f"mods = {modules!r}\n"
        "missing = []\n"
        "for m in mods:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "    except ModuleNotFoundError:\n"
        "        missing.append(m)\n"
        "    except Exception:\n"
        "        pass\n"  # import-time side effects are not a presence failure
        "print(json.dumps(missing))\n"
    )
    result = subprocess.run(
        [ENV_PY, "-c", probe], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, f"{ENV_PY} probe failed: {result.stderr}"
    missing = json.loads(result.stdout.strip().splitlines()[-1])
    detail = {m: sorted(by_module[m])[:3] for m in missing}
    assert not missing, (
        f"ENV_PY reward imports missing from {ENV_PY}: {detail} "
        "(install each into /opt/env/venv per the install-to-/opt policy)"
    )
