from __future__ import annotations

import inspect

import pytest

from lite.core.errors import LiteContractError
from lite.core.messages.image_refs import (
    IMAGE_REFERENCE_ROLES,
    iter_image_references,
    referenced_image_indices_in_message_order,
    referenced_images_in_message_order,
    validate_image_references,
)


def _assistant_message(step: int = 0) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "action_description", "text": f"step {step}"}],
    }


def test_image_references_follow_prompt_order_across_user_and_tool_roles() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 2},
                {"type": "text", "text": "task"},
                {"type": "image", "index": 0},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Action"}]},
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": [
                {"type": "text", "text": "after"},
                {"type": "image", "index": 1},
                {"type": "image", "index": 1},
            ],
        },
    ]

    refs = list(iter_image_references(messages))

    assert IMAGE_REFERENCE_ROLES == frozenset({"user", "tool"})
    assert [(ref.message_index, ref.part_index, ref.role, ref.index) for ref in refs] == [
        (0, 0, "user", 2),
        (0, 2, "user", 0),
        (2, 1, "tool", 1),
        (2, 2, "tool", 1),
    ]
    assert referenced_image_indices_in_message_order(messages) == (2, 0, 1, 1)
    assert referenced_images_in_message_order(messages, ["A", "B", "C"]) == [
        "C",
        "A",
        "B",
        "B",
    ]


def test_image_indices_legacy_user_characterization() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "hi"},
            ],
        },
        _assistant_message(),
        {"role": "user", "content": [{"type": "image", "index": 1}]},
    ]

    assert referenced_image_indices_in_message_order(messages) == (0, 1)


def test_image_indices_preserve_repeated_placeholders_for_positional_binding() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "image", "index": 0},
                {"type": "image", "index": 1},
            ],
        },
    ]

    assert referenced_image_indices_in_message_order(messages) == (0, 0, 1)
    assert referenced_images_in_message_order(messages, ["A", "B"]) == ["A", "A", "B"]


def test_image_indices_scan_user_or_tool() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "instruction"},
            ],
        },
        _assistant_message(),
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": [
                {"type": "image", "index": 1},
                {"type": "text", "text": "obs"},
            ],
        },
    ]

    assert referenced_image_indices_in_message_order(messages) == (0, 1)


def test_images_ordered_legacy_user_characterization() -> None:
    images = ["A", "B"]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 1},
                {"type": "image", "index": 0},
            ],
        },
    ]

    assert referenced_images_in_message_order(messages, images) == ["B", "A"]


def test_images_ordered_scan_user_or_tool() -> None:
    images = ["A", "B"]
    messages = [
        {"role": "user", "content": [{"type": "image", "index": 0}]},
        _assistant_message(),
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": [{"type": "image", "index": 1}],
        },
    ]

    assert referenced_images_in_message_order(messages, images) == ["A", "B"]


@pytest.mark.parametrize("role", ["assistant", "system", None])
def test_image_references_reject_image_parts_on_unbound_roles(role) -> None:
    messages = [{"role": role, "content": [{"type": "image", "index": 0}]}]

    with pytest.raises(LiteContractError, match="image parts are only valid"):
        referenced_image_indices_in_message_order(messages)

    with pytest.raises(LiteContractError, match="image parts are only valid"):
        validate_image_references(messages, ["image.png"])


def test_image_reference_roles_are_the_predicate() -> None:
    # The constant IS the predicate (no ``is_image_reference_role`` wrapper), so
    # membership has to answer for every role AND for non-role objects.
    assert "user" in IMAGE_REFERENCE_ROLES
    assert "tool" in IMAGE_REFERENCE_ROLES
    assert "assistant" not in IMAGE_REFERENCE_ROLES
    assert None not in IMAGE_REFERENCE_ROLES


@pytest.mark.parametrize("role", ["user", "tool"])
def test_image_reference_validation_accepts_runtime_bound_roles(role) -> None:
    messages = [{"role": role, "content": [{"type": "image", "index": 0}]}]

    validate_image_references(messages, ["image.png"])


def test_image_reference_iterator_has_no_permissive_unbound_role_mode() -> None:
    assert "reject_unbound_roles" not in inspect.signature(iter_image_references).parameters


@pytest.mark.parametrize("index", [-1, True, "0", 1.0, 1.5])
def test_image_indices_from_messages_rejects_invalid_index(index) -> None:
    messages = [{"role": "user", "content": [{"type": "image", "index": index}]}]

    with pytest.raises(LiteContractError, match="index"):
        referenced_image_indices_in_message_order(messages)


@pytest.mark.parametrize("index", [-1, True, "0", 1.5])
def test_role_agnostic_image_indices_rejects_invalid_index(index) -> None:
    messages = [{"role": "user", "content": [{"type": "image", "index": index}]}]

    with pytest.raises(LiteContractError, match="index"):
        referenced_image_indices_in_message_order(messages)


def test_image_reference_range_validation_allows_orphan_images() -> None:
    messages = [{"role": "user", "content": [{"type": "image", "index": 0}]}]

    validate_image_references(messages, ["used.png", "orphan.png"])
    validate_image_references(messages, ["used.png", None])
    assert referenced_images_in_message_order(messages, ["used.png", "orphan.png"]) == [
        "used.png"
    ]
    assert referenced_images_in_message_order(messages, ["used.png", None]) == [
        "used.png"
    ]


def test_image_reference_range_validation_rejects_out_of_range() -> None:
    messages = [{"role": "tool", "content": [{"type": "image", "index": 2}]}]

    with pytest.raises(LiteContractError, match="out of range"):
        validate_image_references(messages, ["0.png"])

    with pytest.raises(LiteContractError, match="out of range"):
        referenced_images_in_message_order(messages, ["0.png"])


def test_images_ordered_from_messages_rejects_referenced_none_slot() -> None:
    messages = [{"role": "tool", "content": [{"type": "image", "index": 1}]}]

    validate_image_references(messages, ["0.png", None])

    with pytest.raises(LiteContractError, match="unprocessed image slot"):
        referenced_images_in_message_order(messages, ["0.png", None])


def test_images_ordered_rejects_out_of_range_index() -> None:
    messages = [{"role": "user", "content": [{"type": "image", "index": 1}]}]

    with pytest.raises(LiteContractError, match="out of range"):
        referenced_images_in_message_order(messages, ["A"])


def test_images_ordered_allows_orphan_images() -> None:
    messages = [{"role": "user", "content": [{"type": "image", "index": 0}]}]

    assert referenced_images_in_message_order(messages, ["A", "ORPHAN"]) == ["A"]


def test_utils_image_does_not_export_image_reference_helpers() -> None:
    import lite.utils.image as image_utils

    assert not hasattr(image_utils, "coerce_image_index")
    assert not hasattr(image_utils, "coerce_image_indices")
    assert not hasattr(image_utils, "referenced_images_in_message_order")
