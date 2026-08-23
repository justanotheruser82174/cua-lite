from __future__ import annotations

from typing import Any

import pytest

import lite.core.tools as tool_facade
import lite.core.tools.schemas as schema_module
from lite.core.tools.calls import make_tool_call
from lite.core.tools.schemas import (
    BaseTools,
    install_tool_schemas,
    make_tool_schema,
    tool,
    tool_arguments_match_schema_route_keys,
    tool_call_matches_schema_route_keys,
    tool_call_satisfies_schema,
    tool_name_and_arguments_match_schema_route_keys,
    tool_schema_name,
    tool_schema_parameters,
    tool_schema_validation_errors,
    validate_extra_tool_schemas,
)

RETIRED_SCHEMA_HELPERS = frozenset(
    {
        "LITE_TOOL_SCHEMA_KEYS",
        "require_tool_schema",
        "tool_schema_function",
    }
)


def test_schema_module_exports_only_the_intended_public_surface() -> None:
    # Importable from this owner module (compiler path, route predicates), but
    # never through the ``lite.core.tools`` facade.
    owner_only_helpers = {
        "_compile_tool_method_schemas",
        "tool_arguments_match_schema_route_keys",
        "tool_call_matches_schema_route_keys",
        "tool_name_and_arguments_match_schema_route_keys",
    }
    assert schema_module.__all__ == [
        "BaseTools",
        "LiteToolSchema",
        "LiteToolSchemaFunction",
        "install_tool_schemas",
        "make_tool_schema",
        "schema_type_error_reason",
        "tool",
        "tool_call_satisfies_schema",
        "tool_schema_name",
        "tool_schema_parameters",
        "tool_schema_validation_errors",
        "validate_extra_tool_schemas",
    ]
    assert not (owner_only_helpers & set(schema_module.__all__))
    for name in owner_only_helpers:
        assert hasattr(schema_module, name)
        assert not hasattr(tool_facade, name)


def test_type_hint_compiler_helpers_are_module_private() -> None:
    # The ``@tool`` compiler owns these; nothing outside this module calls them.
    for private_name in ("_is_optional_type", "_python_type_to_json_schema"):
        public_name = private_name.removeprefix("_")
        assert hasattr(schema_module, private_name)
        assert not hasattr(schema_module, public_name)
        assert not hasattr(tool_facade, public_name)


def test_retired_schema_helper_names_stay_absent() -> None:
    for name in RETIRED_SCHEMA_HELPERS:
        assert name not in schema_module.__all__
        assert not hasattr(schema_module, name)
        assert not hasattr(tool_facade, name)


def test_make_tool_schema_uses_section_2_6_key_order() -> None:
    schema = make_tool_schema("goto", description="Go", parameters={})

    assert list(schema) == ["type", "function"]
    assert list(schema["function"]) == ["name", "description", "parameters"]
    assert schema == {
        "type": "function",
        "function": {"name": "goto", "description": "Go", "parameters": {}},
    }


def test_make_tool_schema_output_passes_its_validator() -> None:
    schema = make_tool_schema("goto", description="Go", parameters={})

    validate_extra_tool_schemas([schema])


def test_validate_extra_tool_schemas_requires_a_real_list() -> None:
    from lite.core.errors import LiteContractError

    schema = make_tool_schema("goto", description="Go", parameters={})

    with pytest.raises(
        LiteContractError,
        match="metadata\\.extra_tool_schemas must be a list",
    ):
        validate_extra_tool_schemas((schema,))


@pytest.mark.parametrize(
    ("function_patch", "match"),
    [
        ({"description": None}, "function\\.description must be a string"),
        ({"description": 7}, "function\\.description must be a string"),
        ({"strict": None}, "function\\.strict must be a bool"),
        ({"strict": "true"}, "function\\.strict must be a bool"),
    ],
)
def test_validate_extra_tool_schemas_types_optional_function_fields(
    function_patch,
    match,
) -> None:
    from lite.core.errors import LiteContractError

    schema = make_tool_schema("goto", parameters={})
    schema["function"].update(function_patch)

    with pytest.raises(LiteContractError, match=match):
        validate_extra_tool_schemas([schema])


def test_schema_accessors_read_the_nested_canonical_owner_shape() -> None:
    parameters = {"type": "object", "properties": {"url": {"type": "string"}}}
    schema = make_tool_schema("goto", description="Go", parameters=parameters)

    assert schema["function"]["name"] == "goto"
    assert tool_schema_name(schema) == "goto"
    assert tool_schema_parameters(schema) is parameters
    assert "name" not in schema
    assert "parameters" not in schema


def test_base_tools_exposes_only_the_emitted_tool_catalog() -> None:
    @install_tool_schemas
    class _Catalog(BaseTools):
        @staticmethod
        @tool(url="URL to open.")
        def goto(url: str) -> dict[str, Any]:
            """Go to a URL."""
            return make_tool_call("goto", {"url": url})

        @staticmethod
        @tool()
        def back() -> dict[str, Any]:
            """Go back."""
            return make_tool_call("back")

    emitted = [tool_schema_name(schema) for schema in _Catalog.get_tool_schemas()]

    assert emitted == ["goto", "back"]
    assert frozenset(emitted) == _Catalog.get_tool_names()
    # The tool-layer root carries no action vocabulary at all.
    assert not hasattr(_Catalog, "get_action_names")


def test_make_tool_schema_preserves_explicit_empty_parameters() -> None:
    schema = make_tool_schema("any_args", parameters={})

    assert schema["function"]["parameters"] == {}


def test_make_tool_schema_preserves_strict_on_function_declaration() -> None:
    schema = make_tool_schema("goto", parameters={}, strict=True)

    assert schema == {
        "type": "function",
        "function": {
            "name": "goto",
            "description": "",
            "parameters": {},
            "strict": True,
        },
    }
    validate_extra_tool_schemas([schema])


def test_top_level_strict_is_not_canonical_schema_shape() -> None:
    from lite.core.errors import LiteContractError

    schema = {
        "type": "function",
        "function": {"name": "goto", "description": "", "parameters": {}},
        "strict": True,
    }

    with pytest.raises(LiteContractError):
        validate_extra_tool_schemas([schema])


def test_top_level_schema_fields_are_not_canonical_schema_shape() -> None:
    from lite.core.errors import LiteContractError

    schema = {"type": "function", "name": "goto", "parameters": {}}

    with pytest.raises(
        LiteContractError,
        match="noncanonical outer keys",
    ):
        validate_extra_tool_schemas([schema])


def test_duplicate_schema_names_are_rejected_after_nested_unwrap() -> None:
    from lite.core.errors import LiteContractError

    schemas = [
        make_tool_schema("goto", parameters={}),
        make_tool_schema("goto", parameters={}),
    ]

    with pytest.raises(
        LiteContractError,
        match="duplicate metadata.extra_tool_schemas name 'goto'",
    ):
        validate_extra_tool_schemas(schemas)


def test_key_route_predicate_rejects_falsey_invalid_required() -> None:
    schema = make_tool_schema(
        "goto",
        parameters={
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": "",
        },
    )

    assert tool_arguments_match_schema_route_keys(schema, {"url": "https://example.test"}) is False
    assert tool_arguments_match_schema_route_keys(schema, "not a dict") is False


def test_key_route_predicate_requires_declared_required_keys() -> None:
    schema = make_tool_schema(
        "goto",
        parameters={
            "type": "object",
            "properties": {"url": {"type": "string"}, "wait": {"type": "integer"}},
            "required": ["url"],
        },
    )

    assert tool_arguments_match_schema_route_keys(schema, {"url": "https://x.test"}) is True
    assert tool_arguments_match_schema_route_keys(schema, {"wait": 3}) is False
    # A closed key set: an undeclared argument routes nowhere.
    assert tool_arguments_match_schema_route_keys(schema, {"url": "x", "other": 1}) is False


def test_key_route_predicates_are_not_full_schema_validation() -> None:
    schema = make_tool_schema(
        "annotate",
        parameters={
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
            "additionalProperties": {"type": "string"},
        },
    )

    arguments = {"title": "ok", "note": "accepted by JSON Schema"}

    assert tool_arguments_match_schema_route_keys(schema, arguments) is False
    assert (
        tool_name_and_arguments_match_schema_route_keys(
            name="annotate",
            arguments=arguments,
            schema=schema,
        )
        is False
    )
    assert (
        tool_call_matches_schema_route_keys(
            make_tool_call("annotate", arguments),
            schema,
        )
        is False
    )
    open_route_schema = make_tool_schema(
        "annotate",
        parameters={
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
            "additionalProperties": True,
        },
    )
    assert tool_arguments_match_schema_route_keys(open_route_schema, arguments) is True
    assert (
        tool_name_and_arguments_match_schema_route_keys(
            name="annotate",
            arguments=arguments,
            schema=open_route_schema,
        )
        is True
    )
    assert (
        tool_call_matches_schema_route_keys(
            make_tool_call("annotate", arguments),
            open_route_schema,
        )
        is True
    )
    assert (
        tool_name_and_arguments_match_schema_route_keys(
            name="other",
            arguments=arguments,
            schema=open_route_schema,
        )
        is False
    )
    assert tool_call_matches_schema_route_keys({"not": "canonical"}, open_route_schema) is False

    # Full satisfaction skips the closed-key route gate entirely and agrees with
    # the JSON Schema owner, including on schema-valued additionalProperties.
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(tool_schema_parameters(schema))
    for candidate in (arguments, {"title": "ok", "note": 123}, {"note": "no title"}):
        assert tool_call_satisfies_schema(
            make_tool_call("annotate", candidate), schema
        ) is validator.is_valid(candidate)
    assert tool_call_satisfies_schema(make_tool_call("annotate", arguments), schema) is True


def test_tool_call_satisfies_schema_rejects_wrong_name_and_bad_arguments() -> None:
    schema = make_tool_schema(
        "goto",
        parameters={
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    )

    assert tool_call_satisfies_schema(make_tool_call("back", {"url": "x"}), schema) is False
    assert tool_call_satisfies_schema(make_tool_call("goto", {}), schema) is False
    assert (
        tool_call_satisfies_schema(
            {"type": "function", "function": {"name": "goto", "arguments": "not a dict"}},
            schema,
        )
        is False
    )


def test_tool_schema_validation_errors_returns_sorted_errors() -> None:
    schema = make_tool_schema(
        "set_fields",
        parameters={
            "type": "object",
            "properties": {
                "zeta": {"type": "integer"},
                "alpha": {"type": "string"},
            },
            "required": ["zeta", "alpha"],
        },
    )

    errors = tool_schema_validation_errors(schema, {"zeta": "bad", "alpha": 123})

    assert [(list(error.path), error.validator) for error in errors] == [
        (["alpha"], "type"),
        (["zeta"], "type"),
    ]


def test_tool_schema_validation_errors_bad_schema_fails_loudly() -> None:
    from jsonschema.exceptions import SchemaError

    bad_schema = make_tool_schema(
        "goto",
        parameters={
            "type": "object",
            "properties": {"url": {"type": 123}},
            "required": ["url"],
        },
    )

    with pytest.raises(SchemaError):
        tool_schema_validation_errors(bad_schema, {"url": 123})
