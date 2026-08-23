"""Pin the constants that are hand-copied across the docker image boundary.

WHY THE COPIES EXIST. The in-container servers under ``lite/gym/envs/*/docker/``
are ``COPY``'d into their images as standalone scripts. Inside the image there
is no ``lite`` package on ``sys.path`` — only the single file plus that image's
own dependencies. So those files physically cannot
``from lite.gym.utils.feedback.errors import MODEL_ACTION_ERROR_TYPES``; the
duplication is a property of the deployment, not an oversight.

WHY MERGING IS NOT THE FIX. "Just import the shared definition" breaks the
image build. "Just ship the ``lite`` package into every image" trades a
two-line copy for a whole-tier dependency in images that are deliberately
minimal (and, for mobilegym/webvoyager, built from upstream bases). What is
actually missing is not a merge — it is a CHECK. Each copy carried a
"keep in sync" comment and nothing enforced it. This module converts those
human promises into assertions.

HOW. The server modules are never imported here: their dependencies are heavy
or entirely absent outside their images. Each file is read as text and parsed
with ``ast``, and the literal values are compared against the host-side
definitions in ``lite/gym/utils/feedback/errors.py``.

Run:  uv run pytest tests/gym/utils/backend/test_docker_copy_parity.py -q
"""
from __future__ import annotations

import ast

import pytest

from lite.gym.utils.feedback.errors import (
    MODEL_ACTION_ERROR_TYPES,
    UNSUPPORTED_ACTION_PREFIX,
)
from lite.utils.path import project_root

ROOT = project_root()

# Image-side files carrying a ``_MODEL_ACTION_ERROR_TYPES`` copy.
_MODEL_ACTION_ERROR_COPIES = (
    "lite/gym/envs/online_mind2web/docker/server.py",
    "lite/gym/envs/mobilegym/docker/server.py",
    "lite/gym/envs/webharbor/webvoyager/docker/server.py",
)

# Image-side files that build the unsupported-action message inline.
_UNSUPPORTED_ACTION_COPIES = (
    "lite/gym/envs/online_mind2web/docker/server.py",
    "lite/gym/envs/mobilegym/docker/server.py",
)

# Image-side files carrying a ``_coerce_cursor_bool`` copy.
_CURSOR_BOOL_COPIES = (
    "lite/gym/envs/online_mind2web/docker/server.py",
    "lite/gym/envs/webharbor/webvoyager/docker/server.py",
    "lite/gym/envs/webgym/docker/patches/playwright_instance.py",
)


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def _module_assignment(text: str, name: str) -> ast.expr:
    """Return the value node of a module-level ``name = ...`` assignment."""
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            assert node.value is not None
            return node.value
    raise AssertionError(f"no module-level assignment to {name!r}")


def _function_source(text: str, name: str) -> str:
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            source = ast.get_source_segment(text, node)
            assert source is not None
            return source
    raise AssertionError(f"no module-level function {name!r}")


def _fstring_leading_prefixes(text: str) -> list[str]:
    """Every f-string's leading literal chunk, in source order."""
    prefixes: list[str] = []
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, ast.JoinedStr) or not node.values:
            continue
        head = node.values[0]
        if isinstance(head, ast.Constant) and isinstance(head.value, str):
            prefixes.append(head.value)
    return prefixes


@pytest.mark.parametrize("relpath", _MODEL_ACTION_ERROR_COPIES)
def test_model_action_error_types_copy_matches_host(relpath: str) -> None:
    """``_MODEL_ACTION_ERROR_TYPES`` in-image == ``MODEL_ACTION_ERROR_TYPES``."""
    value = _module_assignment(_read(relpath), "_MODEL_ACTION_ERROR_TYPES")

    assert isinstance(value, ast.Tuple), f"{relpath}: copy is not a tuple literal"
    copied = []
    for elt in value.elts:
        assert isinstance(elt, ast.Name), f"{relpath}: non-name exception entry"
        copied.append(elt.id)

    expected = [exc.__name__ for exc in MODEL_ACTION_ERROR_TYPES]
    assert copied == expected, (
        f"{relpath} drifted from lite/gym/utils/feedback/errors.py: "
        f"{copied} != {expected}"
    )


@pytest.mark.parametrize("relpath", _UNSUPPORTED_ACTION_COPIES)
def test_unsupported_action_prefix_copy_matches_host(relpath: str) -> None:
    """Inline ``f"unsupported action: {...}"`` == ``UNSUPPORTED_ACTION_PREFIX``.

    Collect every f-string whose leading literal looks like the unsupported
    action wording, and require the set to be exactly the host prefix — so a
    changed spelling, casing or spacing fails here rather than silently
    producing a message the host-side parser no longer recognises.
    """
    marker = UNSUPPORTED_ACTION_PREFIX.strip().lower().rstrip(":")
    found = {
        prefix
        for prefix in _fstring_leading_prefixes(_read(relpath))
        if prefix.strip().lower().startswith(marker)
    }

    assert found, f"{relpath}: no unsupported-action message literal found"
    assert found == {UNSUPPORTED_ACTION_PREFIX}, (
        f"{relpath} drifted from UNSUPPORTED_ACTION_PREFIX="
        f"{UNSUPPORTED_ACTION_PREFIX!r}: {sorted(found)}"
    )


def test_cursor_bool_parser_copies_are_byte_identical() -> None:
    """All three in-image ``_coerce_cursor_bool`` copies are the same source.

    There is no host-side definition to compare against — this one is a copy
    between images — so parity is pinned pairwise against the first file.
    """
    reference_path = _CURSOR_BOOL_COPIES[0]
    reference = _function_source(_read(reference_path), "_coerce_cursor_bool")

    drifted = [
        relpath
        for relpath in _CURSOR_BOOL_COPIES[1:]
        if _function_source(_read(relpath), "_coerce_cursor_bool") != reference
    ]

    assert drifted == [], f"_coerce_cursor_bool drifted from {reference_path}: {drifted}"


@pytest.mark.parametrize(
    "relpath", sorted(set(_MODEL_ACTION_ERROR_COPIES) | set(_CURSOR_BOOL_COPIES))
)
def test_image_side_files_do_not_import_lite(relpath: str) -> None:
    """The copies stay copies: no image-side file may import ``lite``.

    This is the standing reason the constants above are duplicated. If this
    ever passes trivially because someone imported the shared definition, the
    image build breaks — so the guard fails first.
    """
    offenders: list[str] = []
    for node in ast.walk(ast.parse(_read(relpath))):
        if isinstance(node, ast.Import):
            offenders += [
                a.name for a in node.names if (a.name == "lite" or a.name.startswith("lite."))
            ]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "lite" or node.module.startswith("lite."):
                offenders.append(node.module)

    assert offenders == [], f"{relpath} imports the host package: {offenders}"
