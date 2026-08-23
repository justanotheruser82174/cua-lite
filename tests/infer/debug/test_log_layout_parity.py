"""Parity lock: the debug-artifact names the writer cannot import.

``lite/infer/debug/log_layout.py`` declares the current per-turn trajectory
artifact names. One file still re-types them:

  * the WRITER, ``lite/agents/core/agent/logger.py`` — it *creates* the files
    and hardcodes ``"prompt_images"``, ``"04_results.json"``,
    ``"05_timing.json"``, ``"prompt_images_annotated"``,
    ``"env_result_images"`` and the sample-root ``"images"`` dir.

**Why the writer does not simply import the constants.** ``tests/static/test_tier_dag``
encodes ``train/infer > agents > gym > utils > core``, with ``lite/agents ->
lite.infer`` listed in ``FORBIDDEN_EDGES``. A ``from lite.infer.debug...``
import inside ``logger.py`` is therefore a layering inversion and would fail
that test. Until the name table moves DOWN to a tier both sides may import
(``lite/utils`` — the leaf every tier including ``gym`` may reach, and where a
dependency-free table of on-disk filenames belongs; ``lite/core`` is the
*protocol* tier for message/tool/sample shapes, not for filesystem layout), the
writer/declarer spellings can only be pinned statically, not merged by runtime
import.

This test does that pinning STATICALLY: it parses the writer and the devs tool
as text (``ast``), never importing ``lite.agents`` from an ``infer``-side
module, so the lock itself introduces no edge. Rename an artifact on one side
and exactly this test goes red, naming the file that drifted.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/infer/debug/test_log_layout_parity.py -p no:cacheprovider -q
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from lite.infer.debug import log_layout
from lite.utils.path import project_root

_ROOT = project_root()
WRITER = _ROOT / "lite/agents/core/agent/logger.py"
DECLARER = _ROOT / "lite/infer/debug/log_layout.py"

#: Current artifact names the writer must spell exactly as declared.
CURRENT_NAMES = (
    log_layout.RESULTS_JSON,
    log_layout.TIMING_JSON,
    log_layout.PROMPT_IMAGES_DIR,
    log_layout.PROMPT_IMAGES_ANNOTATED_DIR,
    log_layout.ENV_RESULT_IMAGES_DIR,
)

#: Legacy spellings. Current readers reject these; the writer must never emit one.
LEGACY_NAMES = (
    "00_screenshot.png",
    "04_screenshot_annotated.png",
    "05_results.json",
    "06_timing.json",
    "images",
    "annotated",
    "result_images",
)


def _string_constants(path: Path) -> set[str]:
    """Every ``str`` literal in a module, without importing it."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _turn_dir_literals(path: Path) -> set[str]:
    """Strings spelled as ``turn_dir / "<name>"`` — the PER-TURN artifacts.

    Depth matters, and a flat literal scan cannot see it: ``"images"`` is BOTH
    the legacy per-turn prompt-image dir (``turn_NNNN/images/``) and the CURRENT
    sample-root image store (``<sample>/images/``). Reading the ``/`` operand
    keeps the two apart.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module_literals = _module_level_literals(path)
    return {
        node.right.value if isinstance(node.right, ast.Constant) else module_literals[node.right.id]
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            and isinstance(node.left, ast.Name)
            and node.left.id == "turn_dir"
            and (
                (isinstance(node.right, ast.Constant) and isinstance(node.right.value, str))
                or (
                    isinstance(node.right, ast.Name)
                    and isinstance(module_literals.get(node.right.id), str)
                )
            )
        )
    }


def _module_level_literals(path: Path) -> dict[str, object]:
    """Module-level ``NAME = <literal>`` bindings, evaluated without import."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            out[target.id] = ast.literal_eval(node.value)
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------------
# Copy 2: the writer.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", CURRENT_NAMES)
def test_writer_spells_every_canonical_current_name(name: str) -> None:
    """``logger.py`` writes exactly the names ``log_layout`` reads back."""
    assert name in _turn_dir_literals(WRITER), (
        f"{WRITER.relative_to(_ROOT)} no longer writes turn_dir/{name!r}; "
        f"lite/infer/debug/log_layout.py still declares it"
    )


@pytest.mark.parametrize("legacy", LEGACY_NAMES)
def test_writer_never_emits_a_legacy_name(legacy: str) -> None:
    """Legacy spellings are rejected, not emitted."""
    assert legacy not in _turn_dir_literals(WRITER), (
        f"{WRITER.relative_to(_ROOT)} writes the legacy name turn_dir/{legacy!r}"
    )


def test_images_is_sample_root_only_not_a_turn_debug_dir() -> None:
    """The sample-root ``images/`` store is current; ``turn_NNNN/images/`` is not."""

    assert "images" not in _turn_dir_literals(WRITER)
    assert "images" in _string_constants(WRITER)
