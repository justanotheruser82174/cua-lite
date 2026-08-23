"""Static guards for shared tool schema and annotation catalogs.

Run:
    uv run pytest tests/static/test_tool_schema_catalogs.py -q
"""

from __future__ import annotations

import ast
from pathlib import Path

from lite.agents.core.action_space.base import (
    LiteBBoxActionSpace,
    LiteDesktopActionSpace,
    LiteMobileActionSpace,
    LitePointActionSpace,
)
from lite.agents.core.agent.utils import annotations as annotation_catalog
from lite.core.tools import action_space as neutral_tools
from lite.core.tools.action_space import (
    LITE_MOBILE_ACTION_BATCH_TOOL_NAME,
    LiteDesktopActionSet,
    LiteMobileActionSet,
    LitePointActionSet,
)
from lite.core.tools.extra_tools import (
    LiteAppLaunchToolSet,
    LiteBrowserNavToolSet,
    LiteFinishToolSet,
    LiteInfeasibilityToolSet,
    LiteShellToolSet,
)

REPO = Path(__file__).resolve().parents[2]

SHARED_SCHEMA_NAMES = (
    LiteFinishToolSet.get_tool_names()
    | LiteBrowserNavToolSet.get_tool_names()
    | LiteAppLaunchToolSet.get_tool_names()
    | LiteShellToolSet.get_tool_names()
    | LiteInfeasibilityToolSet.get_tool_names()
    | LiteMobileActionSet.get_action_names()
    | frozenset({LITE_MOBILE_ACTION_BATCH_TOOL_NAME})
)

AGENT_ACTION_SPACE_BASE = (
    REPO / "lite" / "agents" / "core" / "action_space" / "base.py"
)
CORE_EXTRA_TOOLS = REPO / "lite" / "core" / "tools" / "extra_tools.py"

AGENT_BASE_CORE_CATALOG_CLASSES = frozenset({
    "LiteAppLaunchToolSet",
    "LiteBrowserNavToolSet",
    "LiteFinishToolSet",
})

# Core-owned catalog names that ``lite/agents/core/action_space/base.py`` must
# never grow a local copy of. It used to RE-EXPORT all of them; that barrel is
# gone (``__all__`` there is now exactly the ten names the file defines), so the
# set no longer describes an export surface — only a redeclaration ban.
AGENT_BASE_CORE_CATALOG_NAMES = frozenset({
    "LITE_COMPUTER_ACTION_BATCH_TOOL_NAME",
    "LITE_ACTION_BATCH_TOOL_NAMES",
    "LITE_MOBILE_ACTION_BATCH_TOOL_NAME",
    "LITE_VALID_ACTION_NAMES",
    "LITE_ACTION_SET_TOOL_NAMES",
    "lite_action_names_for_action_batch_tool",
    "lite_action_names_by_action_batch_tool",
    "is_lite_action_name_or_action_batch_tool_name",
    "make_open_app_tool",
    "lite_builtin_tool_names_for_metadata",
    "lite_action_set_tool_names_for_metadata",
})

ENV_SCHEMA_CONTRACT_FILES = tuple(
    sorted(
        path
        for path in (REPO / "lite" / "gym" / "envs").rglob("main.py")
        if "docker" not in path.relative_to(REPO).parts
    )
)

NON_ENV_SCHEMA_CONTRACT_FILES = (
    REPO / "lite" / "gym" / "sandbox" / "base.py",
    REPO / "lite" / "gym" / "utils" / "feedback" / "ingress.py",
    REPO / "lite" / "gym" / "utils" / "feedback" / "surface.py",
    REPO / "lite" / "gym" / "wrappers.py",
)

SCHEMA_CONTRACT_FILES = ENV_SCHEMA_CONTRACT_FILES + NON_ENV_SCHEMA_CONTRACT_FILES

ANNOTATION_RENDERERS = (
    REPO / "lite" / "agents" / "core" / "agent" / "logger.py",
)

ANNOTATION_CATALOG = (
    REPO / "lite" / "agents" / "core" / "agent" / "utils" / "annotations.py"
)

# Derived, not hand-listed: the catalog under test owns the definition, and a
# copy here would be the very duplication this module guards against.
ANNOTATION_VALID_ACTION_NAMES = annotation_catalog._COORDINATE_ANNOTATION_VALID_ACTION_NAMES

SKIPPED_PARTS = {
    "__pycache__",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "_vendor",
}


def _iter_python_files(roots: tuple[Path, ...]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if root.is_file():
            paths.append(root)
            continue
        for path in root.rglob("*.py"):
            parts = path.relative_to(REPO).parts
            if any(part in SKIPPED_PARTS for part in parts):
                continue
            if "src" in parts and "gen" in parts:
                continue
            paths.append(path)
    # Discovery canary — see ``test_zero_variance_subclass._iter_source_files``:
    # an emptied walk would make every gate below pass vacuously.
    assert paths, f"discovery found no .py files under {[str(r) for r in roots]}"
    return sorted(set(paths))


def _read_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _constant_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _dict_values_by_string_key(node: ast.AST | None) -> dict[str, ast.AST]:
    if not isinstance(node, ast.Dict):
        return {}
    return {
        key.value: value
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


def _assignment_names(node: ast.Assign | ast.AnnAssign) -> set[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names: set[str] = set()
    for target in targets:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            names.update(
                element.id for element in target.elts if isinstance(element, ast.Name)
            )
    return names


class _SchemaDuplicateVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.offenders: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        if _call_name(node.func) == "make_tool_schema":
            name = _constant_string(node.args[0]) if node.args else None
            for keyword in node.keywords:
                if keyword.arg == "name":
                    name = _constant_string(keyword.value)
            if name in SHARED_SCHEMA_NAMES:
                self.offenders.append(self._format(node, name, "make_tool_schema"))
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        values_by_key = _dict_values_by_string_key(node)
        schema_type = _constant_string(values_by_key.get("type"))
        function = _dict_values_by_string_key(values_by_key.get("function"))
        name = _constant_string(function.get("name"))
        if (
            schema_type == "function"
            and name in SHARED_SCHEMA_NAMES
            and "parameters" in function
        ):
            self.offenders.append(self._format(node, name, "nested schema literal"))
        flat_name = _constant_string(values_by_key.get("name"))
        if (
            schema_type == "function"
            and flat_name in SHARED_SCHEMA_NAMES
            and "parameters" in values_by_key
        ):
            self.offenders.append(
                self._format(node, flat_name, "flat schema literal")
            )
        self.generic_visit(node)

    def _format(self, node: ast.AST, name: str | None, kind: str) -> str:
        rel = self.path.relative_to(REPO)
        return f"{rel}:{node.lineno}: duplicate {kind} for shared tool {name!r}"


class _StringCatalogVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.offenders: list[str] = []

    def visit_List(self, node: ast.List) -> None:
        self._check_collection(node, node.elts)
        self.generic_visit(node)

    def visit_Set(self, node: ast.Set) -> None:
        self._check_collection(node, node.elts)
        self.generic_visit(node)

    def visit_Tuple(self, node: ast.Tuple) -> None:
        self._check_collection(node, node.elts)
        self.generic_visit(node)

    def _check_collection(self, node: ast.AST, elements: list[ast.AST]) -> None:
        values = {
            value
            for value in (_constant_string(element) for element in elements)
            if value is not None
        }
        overlap = values & ANNOTATION_VALID_ACTION_NAMES
        if len(overlap) >= 2:
            rel = self.path.relative_to(REPO)
            self.offenders.append(
                f"{rel}:{node.lineno}: local annotation action catalog {sorted(overlap)}"
            )


def test_shared_finish_nav_and_mobile_schemas_are_not_redeclared_in_gym_runtime():
    offenders: list[str] = []
    for path in _iter_python_files(SCHEMA_CONTRACT_FILES):
        visitor = _SchemaDuplicateVisitor(path)
        visitor.visit(_read_tree(path))
        offenders.extend(visitor.offenders)

    assert not offenders, "\n".join(offenders)


def test_shared_schema_scan_stays_on_public_declaration_surfaces():
    paths = _iter_python_files(SCHEMA_CONTRACT_FILES)
    relative_paths = {path.relative_to(REPO) for path in paths}

    assert Path("lite/gym/envs/webgym/main.py") in relative_paths
    assert Path("lite/gym/utils/feedback/surface.py") in relative_paths
    assert (
        Path("lite/gym/envs/webgym/docker/patches/_playwright_controller.py")
        not in relative_paths
    )
    assert all(
        (
            path.name == "main.py"
            and path.is_relative_to(REPO / "lite" / "gym" / "envs")
            and "docker" not in path.relative_to(REPO).parts
        )
        or path in NON_ENV_SCHEMA_CONTRACT_FILES
        for path in paths
    )


def test_schema_duplicate_detector_catches_nested_canonical_literals():
    visitor = _SchemaDuplicateVisitor(REPO / "lite" / "gym" / "envs" / "_planted.py")
    visitor.visit(ast.parse("""
DUPLICATE_GOTO_SCHEMA = {
    "type": "function",
    "function": {
        "name": "goto",
        "description": "Navigate.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}
"""))

    assert len(visitor.offenders) == 1
    assert "duplicate nested schema literal for shared tool 'goto'" in visitor.offenders[0]


def test_agent_action_space_base_does_not_redeclare_core_catalogs():
    offenders: list[str] = []
    tree = _read_tree(AGENT_ACTION_SPACE_BASE)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in AGENT_BASE_CORE_CATALOG_CLASSES:
            offenders.append(
                f"{AGENT_ACTION_SPACE_BASE.relative_to(REPO)}:{node.lineno}: "
                f"local class redefines core catalog {node.name}"
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in AGENT_BASE_CORE_CATALOG_NAMES:
                offenders.append(
                    f"{AGENT_ACTION_SPACE_BASE.relative_to(REPO)}:{node.lineno}: "
                    f"local function redefines core catalog {node.name}"
                )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for name in sorted(_assignment_names(node) & AGENT_BASE_CORE_CATALOG_NAMES):
                offenders.append(
                    f"{AGENT_ACTION_SPACE_BASE.relative_to(REPO)}:{node.lineno}: "
                    f"local assignment redefines core catalog {name}"
                    )

    assert not offenders, "\n".join(offenders)


def test_browser_rename_left_no_old_public_action_space_names():
    stale_tokens = ("LiteWebActionSpace", "LiteWebNavToolSet", "lite@web")
    offenders: list[str] = []
    for path in (AGENT_ACTION_SPACE_BASE, CORE_EXTRA_TOOLS):
        text = path.read_text(encoding="utf-8")
        offenders.extend(
            f"{path.relative_to(REPO)}: stale browser-platform token {token!r}"
            for token in stale_tokens
            if token in text
        )

    assert not offenders, "\n".join(offenders)


def test_neutral_gui_catalog_matches_action_space_classes():
    # In the provider-free Lite action sets, the emitted top-level tool and
    # action-batch envelope are the same canonical name.
    assert (
        neutral_tools.LITE_COMPUTER_ACTION_BATCH_TOOL_NAME
        == LiteDesktopActionSpace._ACTION_BATCH_TOOL_NAME
    )
    assert (
        neutral_tools.LITE_MOBILE_ACTION_BATCH_TOOL_NAME
        == LiteMobileActionSpace._ACTION_BATCH_TOOL_NAME
    )
    assert neutral_tools.LITE_ACTION_BATCH_TOOL_NAMES == frozenset({
        neutral_tools.LITE_COMPUTER_ACTION_BATCH_TOOL_NAME,
        LITE_MOBILE_ACTION_BATCH_TOOL_NAME,
    })
    # Core action catalogs answer the ACTION question; the batched spaces EMIT
    # only their action-batch tool, so ``get_action_names()`` is the right
    # accessor.
    assert (
        neutral_tools.LiteDesktopActionSet.get_action_names()
        == LiteDesktopActionSpace.get_action_names()
    )
    assert (
        neutral_tools.LiteMobileActionSet.get_action_names()
        == LiteMobileActionSpace.get_action_names()
    )
    assert (
        neutral_tools.LitePointActionSet.get_tool_names()
        == LitePointActionSpace.get_tool_names()
    )
    assert neutral_tools.LiteBBoxActionSet.get_tool_names() == LiteBBoxActionSpace.get_tool_names()
    # The TOOL layer of the same four sets, derived exactly the way
    # ``LITE_VALID_ACTION_NAMES`` derives the ACTION layer.
    assert neutral_tools.LITE_ACTION_SET_TOOL_NAMES == frozenset({
        "computer", "mobile", "point", "bbox",
    })


def test_annotation_catalog_is_derived_from_tool_signatures():
    """No hand-written per-platform name lists; the signatures are the source."""
    source = ANNOTATION_CATALOG.read_text(encoding="utf-8")

    assert "_DESKTOP_ANNOTATION_COORD_ACTIONS" not in source
    assert "_MOBILE_ANNOTATION_COORD_ACTIONS" not in source
    assert "inspect.signature(" in source

    assert ANNOTATION_VALID_ACTION_NAMES
    # ACTION layer on all three: annotation names are action names, so ``point``
    # is read with ``get_action_names()`` and not off its N=1 tool name.
    assert ANNOTATION_VALID_ACTION_NAMES <= (
        LiteDesktopActionSet.get_action_names()
        | LiteMobileActionSet.get_action_names()
        | LitePointActionSet.get_action_names()
    )


def test_annotation_renderers_use_shared_helper_without_local_action_catalogs():
    offenders: list[str] = []
    for path in ANNOTATION_RENDERERS:
        source = path.read_text(encoding="utf-8")
        assert "coordinate_annotation_records" in source, path.relative_to(REPO)
        visitor = _StringCatalogVisitor(path)
        visitor.visit(ast.parse(source, filename=str(path)))
        offenders.extend(visitor.offenders)

    assert not offenders, "\n".join(offenders)
