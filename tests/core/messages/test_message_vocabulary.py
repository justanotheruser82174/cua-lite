"""Pin the message part-type / role vocabulary to its single home.

``lite/core/messages/content.py`` owns the vocabulary: the ``Literal[...]``
fields on the content and message TypedDicts. Every exported constant is
*derived* from those fields, and every consumer imports the constants instead of
re-listing tags. These tests re-derive the vocabulary from the source AST --
independently of the runtime ``get_type_hints`` derivation -- so a divergence
between a ``Literal[...]`` and an exported name fails loudly.

Run:
    uv run pytest tests/core/messages/test_message_vocabulary.py -q
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from lite.core import messages as core_messages
from lite.core.errors import LiteContractError
from lite.core.messages import content as content_module
from lite.core.messages import final as final_module
from lite.core.messages import image_refs as image_refs_module
from lite.core.metadata import LiteCUAMetadata
from lite.core.samples import LiteSample
from lite.core.tools.calls import make_tool_call
from lite.data.utils.rows import validate_canonical_rows

ROOT = Path(__file__).resolve().parents[3]
MESSAGES_DIR = ROOT / "lite" / "core" / "messages"
CONTENT_PATH = MESSAGES_DIR / "content.py"
PROTOCOL_DIR = ROOT / "lite" / "agents" / "core" / "protocol"
WINDOW_PATH = PROTOCOL_DIR / "window.py"
AGENT_DIR = ROOT / "lite" / "agents" / "core" / "agent"
AGENT_UTILS_DIR = AGENT_DIR / "utils"
ADAPTER_DIR = ROOT / "lite" / "agents" / "core" / "adapter"

#: Canonical producer bodies: the core message modules, the provider-free
#: agent-loop protocol layer that rewrites message windows for one inference
#: call, the agent sampling loop and its helpers that assemble the Lite messages
#: a turn appends, and the adapter base that builds the raw-replay and
#: parse-failure assistant messages. All of them build canonical Lite messages,
#: so all of them import the tags rather than re-spelling them.
#: Provider/template vocabulary stays with its model family.
PRODUCER_PATHS = (
    sorted(MESSAGES_DIR.glob("*.py"))
    + sorted(PROTOCOL_DIR.glob("*.py"))
    + sorted(AGENT_DIR.glob("*.py"))
    + sorted(AGENT_UTILS_DIR.glob("*.py"))
    + sorted(ADAPTER_DIR.glob("*.py"))
)

#: Exported name -> (TypedDict name, field) it must mirror.
EXPORTED_TAGS = {
    "TEXT_PART": ("TextContent", "type"),
    "IMAGE_PART": ("ImageContent", "type"),
    "METADATA_PART": ("MetadataContent", "type"),
    "ACTION_DESCRIPTION_PART": ("ActionDescriptionContent", "type"),
    "INLINE_REASONING_PART": ("InlineReasoningContent", "type"),
    "HISTORY_SUMMARY_PART": ("HistorySummaryContent", "type"),
    "SYSTEM_ROLE": ("LiteSystemMessage", "role"),
    "USER_ROLE": ("LiteUserMessage", "role"),
    "ASSISTANT_ROLE": ("LiteAssistantMessage", "role"),
    "TOOL_ROLE": ("LiteToolMessage", "role"),
}


def _literal_fields(path: Path) -> dict[tuple[str, str], set[str]]:
    """Map (class name, field) -> its ``Literal[...]`` string members, from source."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: dict[tuple[str, str], set[str]] = {}
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for stmt in cls.body:
            if not isinstance(stmt, ast.AnnAssign) or stmt.annotation is None:
                continue
            if not ast.unparse(stmt.annotation).startswith("Literal["):
                continue
            field = stmt.target.id
            out[(cls.name, field)] = {
                node.value
                for node in ast.walk(stmt.annotation)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
    return out


def _module_constant_strings(path: Path) -> dict[str, set[str]]:
    """Map module-level SCREAMING_CASE constant -> the string literals it spells."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: dict[str, set[str]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        name = getattr(target, "id", None)
        if not name or not name.lstrip("_").isupper() or node.value is None:
            continue
        out[name] = {
            child.value
            for child in ast.walk(node.value)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
        }
    return out


def test_every_exported_tag_matches_its_literal_definition() -> None:
    literals = _literal_fields(CONTENT_PATH)
    for exported, key in EXPORTED_TAGS.items():
        assert key in literals, f"{key} no longer declares a Literal[...] field"
        assert literals[key] == {getattr(content_module, exported)}, (
            f"{exported} diverged from {key[0]}.{key[1]}"
        )


def test_vocabulary_sets_cover_exactly_the_literal_fields() -> None:
    literals = _literal_fields(CONTENT_PATH)
    part_tags: set[str] = set()
    role_tags: set[str] = set()
    for (_, field), members in literals.items():
        (part_tags if field == "type" else role_tags).update(members)

    assert content_module.CONTENT_PART_TYPES == part_tags
    assert core_messages.MESSAGE_ROLES == role_tags
    # ...and every exported single tag is a member of its own set.
    for exported, (_, field) in EXPORTED_TAGS.items():
        target = (
            content_module.CONTENT_PART_TYPES
            if field == "type"
            else core_messages.MESSAGE_ROLES
        )
        assert getattr(core_messages, exported) in target


def test_derived_subsets_claim_no_tag_the_vocabulary_lacks() -> None:
    assert content_module.MODEL_VISIBLE_PART_TYPES <= content_module.CONTENT_PART_TYPES
    assert image_refs_module.IMAGE_REFERENCE_ROLES <= core_messages.MESSAGE_ROLES
    assert set(final_module._DIAGNOSTIC_PART_TYPES) <= content_module.CONTENT_PART_TYPES


def test_video_is_not_a_core_message_part_or_filter_key() -> None:
    """Provider-native media keys belong to provider/template projection owners."""
    assert "video" not in content_module.CONTENT_PART_TYPES
    assert "video" not in content_module.MODEL_VISIBLE_PART_TYPES
    assert "_MODEL_VISIBLE_NATIVE_KEYS" not in vars(content_module)


def test_no_consumer_re_lists_the_vocabulary() -> None:
    """Only ``content.py`` may spell a tag inside a module-level constant."""
    vocabulary = content_module.CONTENT_PART_TYPES | core_messages.MESSAGE_ROLES
    consumers = [
        MESSAGES_DIR / "final.py",
        MESSAGES_DIR / "image_refs.py",
        MESSAGES_DIR / "__init__.py",
        WINDOW_PATH,
    ]
    for path in consumers:
        for name, strings in _module_constant_strings(path).items():
            leaked = strings & vocabulary
            assert not leaked, (
                f"{path.relative_to(ROOT)}:{name} re-lists {sorted(leaked)}; "
                f"import the constants from lite.core.messages.content instead"
            )


def _inline_tag_spellings(path: Path) -> list[str]:
    """Report tag strings a function body spells inline instead of importing.

    Two producer shapes carry the vocabulary: the ``type`` / ``role`` value of a
    content or message dict literal, and an equality comparison against one. A
    field NAME that happens to equal a tag (``{"type": TEXT_PART, "text": ...}``,
    ``"text" in item``, the grouped-turn dict's ``"assistant"`` key) is not
    vocabulary, so only value positions are read.
    """
    vocabulary = content_module.CONTENT_PART_TYPES | core_messages.MESSAGE_ROLES
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[str] = []
    for func in [
        n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]:
        for node in ast.walk(func):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value in {"type", "role"}
                        and isinstance(value, ast.Constant)
                        and value.value in vocabulary
                    ):
                        out.append(f"{func.name}: {{{key.value!r}: {value.value!r}}}")
            elif isinstance(node, ast.Compare) and all(
                isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops
            ):
                for operand in [node.left, *node.comparators]:
                    if isinstance(operand, ast.Constant) and operand.value in vocabulary:
                        out.append(f"{func.name}: == {operand.value!r}")
    return out


@pytest.mark.parametrize("path", PRODUCER_PATHS, ids=lambda p: str(p.relative_to(ROOT)))
def test_canonical_message_producers_build_from_the_owned_constants(path: Path) -> None:
    """Producer BODIES, not just module constants, use the owned vocabulary.

    ``test_no_consumer_re_lists_the_vocabulary`` catches a shadow constant; this
    catches the other drift, a builder that emits ``{"role": "assistant"}`` or
    compares against ``"text"`` inline while the module already imports the tag.
    """
    assert _inline_tag_spellings(path) == [], (
        f"{path.relative_to(ROOT)} spells canonical tags inline; import them "
        f"from lite.core.messages.content instead"
    )


def test_consumers_bind_the_shared_constants_not_copies() -> None:
    assert image_refs_module.IMAGE_REFERENCE_ROLES == frozenset(
        {content_module.USER_ROLE, content_module.TOOL_ROLE}
    )
    assert final_module.TEXT_PART is content_module.TEXT_PART
    assert final_module.ASSISTANT_ROLE is content_module.ASSISTANT_ROLE


def test_message_package_facade_exports_only_front_door_contracts() -> None:
    assert set(core_messages.__all__) == {
        "ACTION_DESCRIPTION_PART",
        "ASSISTANT_ROLE",
        "HISTORY_SUMMARY_PART",
        "IMAGE_PART",
        "INLINE_REASONING_PART",
        "MESSAGE_ROLES",
        "METADATA_PART",
        "SYSTEM_ROLE",
        "TEXT_PART",
        "TOOL_ROLE",
        "USER_ROLE",
        "ActionDescriptionContent",
        "HistorySummaryContent",
        "ImageContent",
        "InlineReasoningContent",
        "LiteAssistantMessage",
        "LiteMessage",
        "LiteSystemMessage",
        "LiteToolMessage",
        "LiteUserMessage",
        "MetadataContent",
        "TextContent",
        "extract_first_text",
        "get_action_description",
        "get_inline_reasoning",
        "group_into_turns",
        "instruction_text",
        "keep_model_visible_content",
        "make_assistant_content",
        "message_has_error_feedback",
        "message_has_image",
        "message_metadata",
        "no_tool_call_final_text",
        "peel_system_message",
        "set_or_append_text",
    }
    for private_name in {
        "CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY",
        "CONTENT_ONLY_FINAL_REASON",
        "CONTENT_ONLY_FINAL_TEXT",
        "CONTENT_PART_TYPES",
        "EMPTY_FINAL_REASON",
        "ImageReference",
        "MODEL_OUTPUT_ERROR_KEY",
        "NoToolCallFinalSummary",
        "RawAssistantTextSidecar",
        "TurnSpan",
        "canonicalize_no_tool_call_final_message",
        "classify_no_tool_call_final",
        "first_image_content_part",
        "first_user_message",
        "referenced_image_indices_in_message_order",
        "referenced_images_in_message_order",
        "is_canonical_no_tool_call_final_message",
        "is_image_reference_part",
        "make_no_tool_call_final_actions",
        "mark_model_output_error",
        "message_has_error_metadata",
        "no_tool_call_final_content",
        "no_tool_call_final_stop_reason",
        "pop_model_output_error",
        "summarize_no_tool_call_final",
        "tool_message_text_parts",
        "turn_spans",
        "validate_image_references",
    }:
        assert private_name not in core_messages.__all__
        assert not hasattr(core_messages, private_name)


def test_image_reference_owner_does_not_star_export_record_shape() -> None:
    assert "ImageReference" not in image_refs_module.__all__


# -----------------------------------------------------------------------------
# MESSAGE_ROLES has a caller: the boundary check that makes the vocabulary real
# -----------------------------------------------------------------------------
def test_validate_message_roles_accepts_every_role_in_the_vocabulary() -> None:
    messages = [{"role": role} for role in sorted(core_messages.MESSAGE_ROLES)]
    assert content_module.validate_message_roles(messages) is None


@pytest.mark.parametrize("role", ["unknown", "developer", "human", "", None, 0])
def test_validate_message_roles_rejects_anything_outside_it(role: object) -> None:
    with pytest.raises(LiteContractError, match="outside the Lite role vocabulary"):
        content_module.validate_message_roles([{"role": "user"}, {"role": role}])
    with pytest.raises(LiteContractError, match=r"messages\[1\]"):
        content_module.validate_message_roles([{"role": "user"}, {"role": role}])


def test_validate_message_content_parts_accepts_every_part_type_in_the_vocabulary() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image", "index": 0},
                {"type": "metadata", "data": {}},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "action_description", "text": "Click."},
                {"type": "inline_reasoning", "text": "Thinking."},
                {"type": "history_summary", "text": "Earlier steps."},
            ],
        },
    ]

    assert content_module.validate_message_content_parts(messages) is None


@pytest.mark.parametrize("part_type", ["video", "image_url", "", None])
def test_validate_message_content_parts_rejects_anything_outside_it(
    part_type: object,
) -> None:
    messages = [{"role": "user", "content": [{"type": part_type}]}]

    with pytest.raises(
        LiteContractError,
        match="outside the Lite content part vocabulary",
    ):
        content_module.validate_message_content_parts(messages)
    with pytest.raises(LiteContractError, match=r"messages\[0\]\.content\[0\]"):
        content_module.validate_message_content_parts(messages)


def test_validate_message_content_parts_rejects_non_dict_parts() -> None:
    messages = [{"role": "user", "content": ["not-a-content-part"]}]

    with pytest.raises(LiteContractError, match="content part must be a dict"):
        content_module.validate_message_content_parts(messages)


def test_validate_message_content_parts_rejects_provider_native_untyped_blocks() -> None:
    messages = [{"role": "user", "content": [{"video": "native-template-block"}]}]

    with pytest.raises(
        LiteContractError,
        match="outside the Lite content part vocabulary",
    ):
        content_module.validate_message_content_parts(messages)


def test_keep_model_visible_content_drops_provider_native_untyped_blocks() -> None:
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "goal"},
        {"image": "<provider-native>"},
        {"video": "<provider-native>"},
    ]}]

    assert content_module.keep_model_visible_content(messages)[0]["content"] == [
        {"type": "text", "text": "goal"}
    ]


@pytest.mark.parametrize("role", ["system", "user"])
def test_validate_message_content_parts_rejects_bad_system_user_content_container(
    role: str,
) -> None:
    messages = [{"role": role, "content": {"type": "text", "text": "hi"}}]

    with pytest.raises(
        LiteContractError,
        match=f"role:{role} content must be a string or list",
    ):
        content_module.validate_message_content_parts(messages)


@pytest.mark.parametrize(
    ("role", "part"),
    [
        ("system", {"type": "image", "index": 0}),
        ("user", {"type": "action_description", "text": "click"}),
        ("assistant", {"type": "image", "index": 0}),
        ("tool", {"type": "inline_reasoning", "text": "hidden"}),
    ],
)
def test_validate_message_content_parts_rejects_role_forbidden_parts(
    role: str,
    part: dict,
) -> None:
    messages = [{"role": role, "content": [part]}]

    with pytest.raises(LiteContractError, match=f"role:{role} messages cannot carry"):
        content_module.validate_message_content_parts(messages)


@pytest.mark.parametrize("role", ["assistant", "tool"])
def test_validate_message_content_parts_rejects_assistant_and_tool_string_content(
    role: str,
) -> None:
    messages = [{"role": role, "content": "bare string"}]

    with pytest.raises(LiteContractError, match=f"role:{role} content must be a list"):
        content_module.validate_message_content_parts(messages)


def test_validate_message_content_parts_rejects_missing_tool_content() -> None:
    messages = [{"role": "tool", "tool_call_id": "call_0"}]

    with pytest.raises(LiteContractError, match="role:tool content is required"):
        content_module.validate_message_content_parts(messages)


@pytest.mark.parametrize(
    "part",
    [
        {"type": "text"},
        {"type": "text", "text": 1},
        {"type": "action_description"},
        {"type": "action_description", "text": None},
        {"type": "inline_reasoning", "text": []},
        {"type": "history_summary", "text": {}},
    ],
)
def test_validate_message_content_parts_rejects_bad_text_payloads(part: dict) -> None:
    messages = [{"role": "assistant", "content": [part]}]

    with pytest.raises(
        LiteContractError,
        match=r"content\[0\]\.text must be a string",
    ):
        content_module.validate_message_content_parts(messages)


@pytest.mark.parametrize("data", [None, [], "metadata"])
def test_validate_message_content_parts_rejects_bad_metadata_payload(
    data: object,
) -> None:
    messages = [{"role": "user", "content": [{"type": "metadata", "data": data}]}]

    with pytest.raises(LiteContractError, match=r"content\[0\]\.data must be a dict"):
        content_module.validate_message_content_parts(messages)


def test_lite_sample_from_dict_rejects_a_role_outside_the_vocabulary() -> None:
    """``LiteSample.from_dict`` is the dict → trajectory boundary: its messages
    are data, so their roles are checked there rather than reinterpreted by every
    walk downstream."""
    meta = LiteCUAMetadata()
    good = LiteSample.from_dict(
        {"messages": [{"role": "user", "content": []}], "metadata": meta}
    )
    assert len(good.messages) == 1
    with pytest.raises(LiteContractError, match="outside the Lite role vocabulary"):
        LiteSample.from_dict(
            {"messages": [{"role": "human", "content": []}], "metadata": meta}
        )


def test_lite_sample_from_dict_rejects_malformed_content_parts() -> None:
    meta = LiteCUAMetadata()

    with pytest.raises(
        LiteContractError,
        match=r"content\[0\]\.text must be a string",
    ):
        LiteSample.from_dict(
            {
                "messages": [
                    {"role": "user", "content": [{"type": "text"}]},
                ],
                "metadata": meta,
            }
        )


def test_canonical_row_validation_rejects_a_role_outside_the_vocabulary() -> None:
    """The dataset-row contract is the other entry point: a bad role is rejected
    before the row can become a sample, be published, or be re-rendered."""
    row = {
        "messages": [{"role": "human", "content": [{"type": "text", "text": "hi"}]}],
        "images": [],
        "metadata": {},
    }
    with pytest.raises(LiteContractError, match="outside the Lite role vocabulary"):
        validate_canonical_rows([row], "unit")


@pytest.mark.parametrize("part_type", ["video", "image_url"])
def test_canonical_row_validation_rejects_a_content_part_type_outside_the_vocabulary(
    part_type: str,
) -> None:
    row = {
        "messages": [{"role": "user", "content": [{"type": part_type}]}],
        "images": [],
        "metadata": {},
    }

    with pytest.raises(
        LiteContractError,
        match="outside the Lite content part vocabulary",
    ):
        validate_canonical_rows([row], "unit")


def _use_row(message: dict) -> dict:
    return {
        "messages": [message],
        "images": [],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    }


def test_canonical_row_validation_allows_tool_call_assistant_without_content() -> None:
    row = _use_row({
        "role": "assistant",
        "tool_calls": [
            make_tool_call(
                "computer",
                {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                call_id="call_click",
            )
        ],
    })

    validate_canonical_rows([row], "unit/assistant-tool-call-content")


def test_canonical_row_validation_accepts_no_tool_assistant_content_list() -> None:
    row = _use_row({
        "role": "assistant",
        "content": [{"type": "text", "text": "Done."}],
    })

    validate_canonical_rows([row], "unit/assistant-content")


@pytest.mark.parametrize(
    ("message", "match"),
    [
        ({"role": "assistant", "content": "bare"}, "role:assistant content"),
        ({"role": "tool", "tool_call_id": "call_0", "content": "bare"}, "role:tool content"),
        (
            {"role": "assistant", "content": [{"type": "image", "index": 0}]},
            "role:assistant messages cannot carry",
        ),
        (
            {"role": "user", "content": [{"type": "metadata", "data": []}]},
            r"content\[0\]\.data must be a dict",
        ),
        (
            {"role": "assistant", "content": [{"type": "action_description"}]},
            r"content\[0\]\.text must be a string",
        ),
    ],
)
def test_canonical_row_validation_rejects_malformed_content_boundary_shapes(
    message: dict,
    match: str,
) -> None:
    row = _use_row(message)

    with pytest.raises(LiteContractError, match=match):
        validate_canonical_rows([row], "unit/message-content-boundary")


def test_canonical_row_validation_rejects_no_tool_assistant_none_content() -> None:
    row = _use_row({"role": "assistant", "content": None})

    with pytest.raises(LiteContractError, match="role:assistant content must be a list"):
        validate_canonical_rows([row], "unit/assistant-content")


def test_canonical_row_validation_rejects_no_tool_assistant_empty_content() -> None:
    row = _use_row({"role": "assistant", "content": []})

    with pytest.raises(ValueError, match="assistant|content-only final assistant turn"):
        validate_canonical_rows([row], "unit/assistant-content")
