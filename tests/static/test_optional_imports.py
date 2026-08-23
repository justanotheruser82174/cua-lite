"""Optional third-party imports may suppress only a precise import failure.

This is not a shim guard. It combines static optional-import policy checks with
light subprocess import probes. An over-broad ``except`` around an optional
import makes a genuine failure inside ``lite`` look like "litellm is not
installed", and the registration silently disappears instead of raising.

``lite/agents/bootstrap.py`` gets this right today (it compares
``e.name.split(".")[0] != "litellm"``); this pins that the package facades do
not regress to string-matching the exception text.

Run:
    uv run pytest tests/static/test_optional_imports.py -q
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_OPTIONAL_IMPORT_FACADES = (
    "lite/agents/models/__init__.py",
    "lite/agents/extensions/__init__.py",
    "lite/agents/bootstrap.py",
)


def _run_import_probe(source: str) -> list[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(REPO) if not env.get("PYTHONPATH") else f"{REPO}{os.pathsep}{env['PYTHONPATH']}"
    )
    proc = subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert not proc.stderr, proc.stderr
    return proc.stdout.splitlines()


def test_optional_litellm_imports_do_not_match_exception_strings() -> None:
    offenders: list[str] = []
    for relpath in _OPTIONAL_IMPORT_FACADES:
        path = REPO / relpath
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if "litellm" not in text or "str(" not in text:
            continue
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            source = ast.get_source_segment(text, node) or ""
            if "litellm" in source and "str(" in source:
                offenders.append(f"{relpath}:{node.lineno}: {source}")

    assert not offenders, (
        "Optional API-agent imports may suppress only ModuleNotFoundError.name "
        "for litellm; string-matching the exception can hide unrelated import "
        "failures:\n  " + "\n  ".join(offenders)
    )


def test_gpt_agent_imports_do_not_load_litellm() -> None:
    lines = _run_import_probe(
        "import sys\n"
        "import lite.agents.models.gpt.agent\n"
        "print('litellm' in sys.modules)\n"
    )

    assert lines == ["False"]


def test_claude_agent_imports_do_not_load_litellm() -> None:
    lines = _run_import_probe(
        "import sys\n"
        "import lite.agents.models.claude.agent\n"
        "print('litellm' in sys.modules)\n"
    )

    assert lines == ["False"]


def test_gpt_convert_leaf_import_stays_light() -> None:
    lines = _run_import_probe(
        "import sys\n"
        "from lite.agents.models.gpt.utils.convert import "
        "lite_calls_from_gpt_computer_actions\n"
        "from lite.agents.models.gpt.utils.convert import "
        "lite_calls_from_gpt_mobile_function_calls\n"
        "from lite.agents.models.gpt.utils.convert import "
        "parse_provider_function_args\n"
        "print(callable(lite_calls_from_gpt_computer_actions))\n"
        "print(callable(lite_calls_from_gpt_mobile_function_calls))\n"
        "print(callable(parse_provider_function_args))\n"
        # ``parse`` is the module that imports this leaf; ``responses`` is the
        # rest of the GPT provider surface. Neither may ride along.
        "print('lite.agents.models.gpt.utils.parse' in sys.modules)\n"
        "print('lite.agents.models.gpt.utils.responses' in sys.modules)\n"
    )

    assert lines == ["True", "True", "True", "False", "False"]


def test_lite_data_import_does_not_load_dataset_backends() -> None:
    lines = _run_import_probe(
        "import sys\n"
        "import lite.data\n"
        "print('lite.data.load' in sys.modules)\n"
        "print('datasets' in sys.modules)\n"
        "from lite.data import discover_files_under_paths\n"
        "print(callable(discover_files_under_paths))\n"
        "print('lite.data.load' in sys.modules)\n"
    )

    assert lines == ["False", "False", "True", "True"]


def test_sandbox_task_types_do_not_load_base_env() -> None:
    lines = _run_import_probe(
        "import sys\n"
        "import lite.gym.sandbox\n"
        "print('lite.gym.sandbox.base' in sys.modules)\n"
        "from lite.gym.sandbox import SandboxTaskConfig\n"
        "print(SandboxTaskConfig.__name__)\n"
        "print('lite.gym.sandbox.base' in sys.modules)\n"
        "from lite.gym.sandbox import SandboxBaseEnv\n"
        "print(SandboxBaseEnv.__name__)\n"
        "print('lite.gym.sandbox.base' in sys.modules)\n"
    )

    assert lines == [
        "False",
        "SandboxTaskConfig",
        "False",
        "SandboxBaseEnv",
        "True",
    ]
