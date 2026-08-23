from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]

_DOTTED_PATH_FLAGS = (
    "--custom-generate-function-path",
    "--custom-convert-samples-to-train-data-path",
    "--rollout-function-path",
)


def _script(name: str) -> str:
    return (_ROOT / "scripts" / "train" / name).read_text()


def _dotted_paths_in_script(script_name: str) -> set[str]:
    """Every ``lite.train.rollout.*`` dotted path passed to a Slime path flag."""
    text = _script(script_name)
    defaults = dict(re.findall(r"^(\w+)=\$\{\1:-([\w.]+)\}$", text, re.M))
    found: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        parts = stripped.split()
        for flag in _DOTTED_PATH_FLAGS:
            if flag in parts:
                value = parts[parts.index(flag) + 1].strip('"')
                for name, default in defaults.items():
                    value = value.replace("${" + name + "}", default)
                if value.startswith("lite.train.rollout."):
                    found.add(value)
    return found


def _module_all(module_dotted: str) -> set[str]:
    module_path = _ROOT / module_dotted.replace(".", "/")
    path = module_path.with_suffix(".py")
    if not path.exists():
        path = module_path / "__init__.py"
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return {ast.literal_eval(e) for e in node.value.elts}
    raise AssertionError(f"{module_dotted} declares no module-level __all__")


@pytest.mark.parametrize(
    ("script_name", "module_dotted"),
    [
        ("run_grpo.sh", "lite.train.rollout.grpo"),
        ("run_reinforce.sh", "lite.train.rollout.reinforce"),
        ("run_dagger.sh", "lite.train.rollout.dagger"),
        ("run_sft.sh", "lite.train.rollout.sft"),
    ],
)
def test_shim_all_matches_the_dotted_paths_its_script_passes(
    script_name: str, module_dotted: str
) -> None:
    dotted = _dotted_paths_in_script(script_name)
    assert dotted, f"{script_name} passes no lite.train.rollout.* path flag"

    attrs = set()
    for path in dotted:
        module, _, attr = path.rpartition(".")
        assert module == module_dotted, f"{script_name} resolves {path}, not {module_dotted}.*"
        attrs.add(attr)

    assert _module_all(module_dotted) == attrs
