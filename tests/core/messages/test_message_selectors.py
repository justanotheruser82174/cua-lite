"""Behavior + ownership tests for the list-level message selectors.

``instruction_text`` replaced hand-rolled "find the task instruction" loops
that disagreed on the same input. The measured pre-fix answers on a first
user message whose ``content`` is a plain ``str`` (a shape
``LiteUserMessage`` explicitly declares) were:

  * ``ui_tars/adapter.py``  -> ``AttributeError``
  * ``agent/logger.py``     -> ``""``      (silently empty)

Run:
    uv run pytest tests/core/messages/test_message_selectors.py -v
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from lite.core import messages as core_messages
from lite.core.messages import instruction_text
from lite.core.messages.selectors import first_user_message

ROOT = Path(__file__).resolve().parents[3]

#: The sites that used to hand-roll the selector.
SITES = (
    ROOT / "lite" / "agents" / "models" / "ui_tars" / "adapter.py",
    ROOT / "lite" / "agents" / "core" / "agent" / "logger.py",
)


#: The module paths that own the selector (``lite.core.messages`` re-exports the
#: ``selectors`` submodule, and sites import from either).
_SELECTOR_MODULES = ("lite.core.messages", "lite.core.messages.selectors")


def _calls_shared_selector(tree: ast.Module, name: str = "instruction_text") -> bool:
    """True iff the module imports ``name`` from the owning module and CALLS it.

    Alias-resolving, so ``from lite.core.messages import instruction_text as it``
    followed by ``it(messages)`` counts, while a bare mention in a comment or a
    docstring does not.
    """
    direct = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module in _SELECTOR_MODULES
        for alias in node.names
        if alias.name == name
    }
    # ``from lite.core import messages`` / ``import lite.core.messages as m``
    # -> the selector is reached as ``<binding>.instruction_text``.
    qualified = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "lite.core"
        for alias in node.names
        if alias.name == "messages"
    } | {
        alias.asname
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name in _SELECTOR_MODULES and alias.asname
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in direct:
            return True
        if (
            isinstance(func, ast.Attribute)
            and func.attr == name
            and isinstance(func.value, ast.Name)
            and func.value.id in qualified
        ):
            return True
    return False


class TestInstructionText:
    def test_str_content_is_the_instruction(self) -> None:
        """The shape that crashed ui_tars and silently emptied agent/logger.py."""
        messages = [{"role": "user", "content": "GOAL"}]
        assert instruction_text(messages) == "GOAL"

    def test_str_content_wins_over_a_later_user_message(self) -> None:
        messages = [
            {"role": "user", "content": "GOAL"},
            {"role": "user", "content": [{"type": "text", "text": "LATER"}]},
        ]
        assert instruction_text(messages) == "GOAL"

    def test_leading_text_part(self) -> None:
        messages = [{"role": "user", "content": [{"type": "text", "text": "GOAL"}]}]
        assert instruction_text(messages) == "GOAL"

    def test_image_before_text_in_the_first_user_message(self) -> None:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "SYS"}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "GOAL"},
                ],
            },
        ]
        assert instruction_text(messages) == "GOAL"

    def test_later_user_observation_is_never_promoted(self) -> None:
        """An image-only first turn yields "" -- not the next turn's text.

        Every user message after the first is an observation; returning its
        text would render a screenshot caption as the task goal.
        """
        messages = [
            {"role": "user", "content": [{"type": "image", "index": 0}]},
            {"role": "assistant", "content": [{"type": "text", "text": "A"}]},
            {"role": "user", "content": [{"type": "text", "text": "LATER"}]},
        ]
        assert instruction_text(messages) == ""

    def test_no_user_message(self) -> None:
        assert instruction_text([{"role": "assistant", "content": []}]) == ""

    def test_empty(self) -> None:
        assert instruction_text([]) == ""

    def test_none_is_not_an_empty_message_list(self) -> None:
        with pytest.raises(TypeError):
            instruction_text(None)


class TestFirstUserMessage:
    def test_returns_the_first_user_message_itself(self) -> None:
        first = {"role": "user", "content": "GOAL"}
        messages = [
            {"role": "system", "content": "SYS"},
            first,
            {"role": "user", "content": [{"type": "text", "text": "LATER"}]},
        ]
        assert first_user_message(messages) is first

    def test_skips_non_user_roles(self) -> None:
        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "assistant", "content": []},
        ]
        assert first_user_message(messages) is None
        assert first_user_message([]) is None

    def test_none_is_not_an_empty_message_list(self) -> None:
        with pytest.raises(TypeError):
            first_user_message(None)


class TestOwnership:
    def test_core_exports_the_selectors(self) -> None:
        assert core_messages.instruction_text is instruction_text
        assert "first_user_message" not in core_messages.__all__

    def test_no_site_hand_rolls_the_selector(self) -> None:
        """Each ex-hand-roller defines no private copy AND really calls the
        shared selector.

        The "really calls it" half used to be ``assert "instruction_text(" in
        source``, which is wrong in both directions: it passes on a module that
        only mentions the name in a comment or docstring, and it fails on a
        module that imports the selector under an alias and calls it correctly.
        Resolved through the AST instead -- import bindings (aliases included)
        collected first, then a call whose callee is one of them.
        """
        for path in SITES:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            defined = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            assert "_extract_instruction" not in defined, path
            assert _calls_shared_selector(tree), (
                f"{path} never calls lite.core.messages.instruction_text "
                "(hand-rolled again, or the import went dead)"
            )
