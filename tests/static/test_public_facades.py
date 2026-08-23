"""Narrow public-facade guards for the canonical tool cleanup."""

from __future__ import annotations

import importlib

SCHEMA_OWNER_ONLY_NAMES = frozenset(
    {
        "_compile_tool_method_schemas",
        "is_optional_type",
        "python_type_to_json_schema",
        "tool_arguments_match_schema_route_keys",
        "tool_call_matches_schema_route_keys",
        "tool_name_and_arguments_match_schema_route_keys",
    }
)
ACTION_BATCH_EXPORTS = frozenset(
    {
        "LITE_ACTION_BATCH_TOOL_NAMES",
        "LITE_COMPUTER_ACTION_BATCH_TOOL_NAME",
        "LITE_MOBILE_ACTION_BATCH_TOOL_NAME",
        "lite_action_batch_tool_name_for_platform",
        "lite_action_names_by_action_batch_tool",
        "lite_action_names_for_action_batch_tool",
        "is_lite_action_name_or_action_batch_tool_name",
        "make_lite_action_batch_call",
        "make_lite_action_batch_schema",
        "unpack_action_batch_call",
        "validate_lite_action_batch_structure",
    }
)
ACTION_BATCH_OWNER_ONLY_NAMES = frozenset(
    {
        "LiteActionBatchMergeResult",
        "LiteActionBatchValidationError",
        "LiteActionBatchValidationKind",
        "filter_action_batch_schema",
        "merge_adjacent_lite_action_batches_with_provenance",
        "validate_lite_action_batch_child_arguments",
    }
)
CORE_ROOT_OWNER_ONLY_NAMES = frozenset({"RawAssistantTextSidecar", "parse_filter"})
CORE_ROOT_EXPORTS = frozenset(
    {
        "ActionDescriptionContent",
        "HistorySummaryContent",
        "ImageContent",
        "InlineReasoningContent",
        "LiteAssistantMessage",
        "LiteBaseMetadata",
        "LiteCUAMetadata",
        "LiteGenericMetadata",
        "LiteMessage",
        "LiteRLSample",
        "LiteRLStep",
        "LiteSample",
        "LiteSystemMessage",
        "LiteToolCall",
        "LiteToolMessage",
        "LiteToolResult",
        "LiteToolSchema",
        "LiteUserMessage",
        "MetadataContent",
        "STATUS_ABORTED",
        "STATUS_COMPLETED",
        "STATUS_FAILED",
        "STATUS_TRUNCATED",
        "STEP_STATUSES_BY_SEVERITY",
        "TextContent",
        "metadata_from_dict",
    }
)


def test_core_tools_facade_keeps_schema_owner_helpers_private() -> None:
    module = importlib.import_module("lite.core.tools")
    exported = set(module.__all__)

    assert not (SCHEMA_OWNER_ONLY_NAMES & exported)
    for name in SCHEMA_OWNER_ONLY_NAMES:
        assert not hasattr(module, name)


def test_action_batch_facade_exports_the_current_owner_names() -> None:
    module = importlib.import_module("lite.core.tools.action_space")
    exported = set(module.__all__)

    assert ACTION_BATCH_EXPORTS <= exported
    assert not (ACTION_BATCH_OWNER_ONLY_NAMES & exported)
    for name in ACTION_BATCH_EXPORTS:
        assert hasattr(module, name)
    for name in ACTION_BATCH_OWNER_ONLY_NAMES:
        assert not hasattr(module, name)


def test_core_root_facade_exposes_only_canonical_contracts() -> None:
    module = importlib.import_module("lite.core")
    exported = set(module.__all__)

    assert exported == CORE_ROOT_EXPORTS
    for name in CORE_ROOT_EXPORTS:
        assert hasattr(module, name)

    assert not (CORE_ROOT_OWNER_ONLY_NAMES & exported)
    for name in CORE_ROOT_OWNER_ONLY_NAMES:
        assert not hasattr(module, name)
