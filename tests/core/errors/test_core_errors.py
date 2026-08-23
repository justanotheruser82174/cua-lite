"""The core error model: one type, and the two taxonomies it must NOT absorb.

Run:
    uv run pytest tests/core/errors/test_core_errors.py -q
"""

from __future__ import annotations

import pytest

from lite.core.errors import LiteContractError, ToolCallValidationError
from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space.geometry import strict_norm_to_pixel
from lite.core.tools.calls import stamp_tool_call_list_ids
from lite.core.tools.schemas import validate_extra_tool_schemas
from lite.core.utils.filters import parse_filter


def test_contract_error_uses_the_named_type_only() -> None:
    """Core contract violations are not compatibility-shaped ``ValueError``s."""
    assert issubclass(LiteContractError, Exception)
    assert not issubclass(LiteContractError, ValueError)
    assert issubclass(ToolCallValidationError, LiteContractError)


def test_core_declares_no_gym_failure_category() -> None:
    """N4: core no longer carries a gym opinion, so there is nothing to keep in step.

    The slot itself is gone too, not just its value. ``failure_category()`` reads it with
    ``getattr(exc, "failure_category", None)``, which already yields ``None`` for a class
    that declares nothing — an empty slot would only invite the next hand-copied spelling.
    """
    assert not hasattr(LiteContractError, "failure_category")
    assert not hasattr(ToolCallValidationError, "failure_category")


def test_gym_classifies_the_tool_call_error_by_type() -> None:
    """The behaviour the deleted string was there to produce, now by ``isinstance``.

    This assertion is unchanged by N4 — it is what the previous test should have been in
    the first place. The two spellings it used to compare are now one.
    """
    from lite.gym.errors import FailureCategory, failure_category

    assert failure_category(ToolCallValidationError("x")) is FailureCategory.MODEL_OUTPUT_ERROR
    assert failure_category(LiteContractError("x")) is None


@pytest.mark.parametrize(
    "call",
    [
        lambda: parse_filter("not a lambda"),
        lambda: strict_norm_to_pixel("nope", 100, 100, clamp=True),
        lambda: validate_extra_tool_schemas([{"type": "function"}]),
        lambda: LiteCUAMetadata.from_dict({"metadata_kind": "cua", "dims": ["tablet", "use"]}),
        lambda: LiteCUAMetadata.from_dict({"metadata_kind": "cua", "dims": ["desktop", "browse"]}),
        lambda: LiteCUAMetadata.from_dict({"metadata_kind": "cua"}),
        lambda: LiteCUAMetadata(dims=("tablet", "use")),
    ],
)
def test_core_contract_violations_raise_the_named_type(call) -> None:
    with pytest.raises(LiteContractError):
        call()


def test_core_facade_keeps_filter_on_its_owner_and_exports_tool_result() -> None:
    import lite.core as core
    from lite.core.tools.results import LiteToolResult

    assert "parse_filter" not in core.__all__
    assert not hasattr(core, "parse_filter")
    assert "LiteToolResult" in core.__all__
    assert core.LiteToolResult is LiteToolResult


def test_tool_call_envelope_violation_raises_the_marked_subclass() -> None:
    with pytest.raises(ToolCallValidationError):
        stamp_tool_call_list_ids([{"function": {"name": "computer"}}], preserve=False)


def test_gym_and_agent_error_taxonomies_stay_where_they_are() -> None:
    """core/errors.py is not a dumping ground for everything error-shaped.

    ``lite.gym.errors`` owns the retry/HTTP failure taxonomy and
    ``lite.agents.core.action_space.errors`` owns model-output parse marking;
    neither is a statement about Lite data shape, so neither folds into core.
    """
    import lite.agents.core.action_space.errors as agent_errors
    import lite.core.errors as core_errors
    import lite.gym.errors as gym_errors

    for name in ("FailureCategory", "LiteGymError", "is_retryable"):
        assert hasattr(gym_errors, name), name
        assert not hasattr(core_errors, name), name
    for name in agent_errors.__all__:
        assert not hasattr(core_errors, name), name
