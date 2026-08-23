"""Unit tests for ``lite.gym.utils.feedback.surface.resolve_extra_tools``.

Run:
    uv run pytest tests/gym/utils/feedback/test_extra_tools_surface.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lite.core import LiteCUAMetadata
from lite.core.errors import LiteContractError
from lite.core.tools.extra_tools import (
    BASH_TOOL_NAME,
    LiteBrowserNavToolSet,
    LiteFinishToolSet,
    LiteInfeasibilityToolSet,
    LiteShellToolSet,
    make_open_app_tool,
    make_report_infeasible_tool,
    open_app_names_from_metadata,
)
from lite.core.tools.schemas import (
    BaseTools,
    make_tool_schema,
    tool_schema_name,
    tool_schema_parameters,
)
from lite.gym.utils.feedback import surface as surface_module
from lite.gym.utils.feedback.ingress import is_active_extra_tool_call
from lite.gym.utils.feedback.surface import (
    merge_extra_tool_schemas,
    resolve_extra_tools,
)

_FOO = make_tool_schema("foo")
_BAR = make_tool_schema("bar")
_ANSWER = make_tool_schema("answer")


class _Tools(BaseTools):
    """A two-tool env set, declared the way every env declares one."""

    _SCHEMAS = {"foo": _FOO, "bar": _BAR}


class _NoTools(BaseTools):
    """An env that declares no extras at all — finish tools only."""


class _FinishOverrideTools(BaseTools):
    """Illegal: an env set that shadows a canonical finish tool."""

    _SCHEMAS = {
        "response": make_tool_schema("response"),
    }


#: The sandbox family's grant: everything declared, plus the transport's bash.
_SANDBOX_EXECUTABLE = _Tools.get_tool_names() | {BASH_TOOL_NAME}


def test_none_blocks_by_default():
    """``None`` (yaml omitted) → ``[]`` — opt-in default."""
    assert resolve_extra_tools(None, tools=_Tools) == []


def test_empty_list_blocks():
    """``[]`` — block everything (explicit, same effect as omission)."""
    assert resolve_extra_tools([], tools=_Tools) == []


def test_single_name_returns_schema():
    assert resolve_extra_tools(["foo"], tools=_Tools) == [_FOO]


def test_order_preserved():
    """Selection order is preserved in the output list."""
    assert resolve_extra_tools(["bar", "foo"], tools=_Tools) == [_BAR, _FOO]


def test_full_selection_equals_tool_set_values():
    assert resolve_extra_tools(["foo", "bar"], tools=_Tools) == [_FOO, _BAR]


def test_known_standalone_names_are_declared_tools_plus_finish():
    """The union is not optional: an env set structurally CANNOT declare a
    finish tool (see ``test_resolve_extra_tools_rejects_env_tool_set_finish_overrides``),
    so ``get_tool_names()`` alone is missing exactly ``response``/``terminate``."""
    assert _Tools.get_tool_names() | LiteFinishToolSet.get_tool_names() == (
        frozenset({"foo", "bar"}) | LiteFinishToolSet.get_tool_names()
    )


def test_feedback_ingress_routes_active_extra_by_key_shape_before_validation():
    click_extra = make_tool_schema(
        "click",
        parameters={
            "type": "object",
            "properties": {"index": {"type": "integer"}},
            "required": ["index"],
            "additionalProperties": False,
        },
    )

    # Bad type still routes to env feedback; full argument validity is not an
    # action-space-base admission decision.
    assert is_active_extra_tool_call(
        {"name": "click", "arguments": {"index": "not-an-int"}},
        [click_extra],
    )
    assert not is_active_extra_tool_call(
        {"name": "click", "arguments": {"coordinate": [1, 2]}},
        [click_extra],
    )


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown extra_tools"):
        resolve_extra_tools(["nope"], tools=_Tools)


def test_partial_unknown_raises():
    """Unknown names are reported even when some names are valid."""
    with pytest.raises(ValueError) as exc:
        resolve_extra_tools(["foo", "nope"], tools=_Tools)
    assert "nope" in str(exc.value)


def test_empty_tool_set_with_none():
    """Empty tool set + ``None`` selection → ``[]`` (no error)."""
    assert resolve_extra_tools(None, tools=_NoTools) == []


def test_empty_tool_set_with_unknown_raises():
    """Empty tool set + any selected name → ValueError."""
    with pytest.raises(ValueError):
        resolve_extra_tools(["foo"], tools=_NoTools)


def test_finish_tools_resolve_without_env_tool_set_entries():
    assert resolve_extra_tools(["terminate"], tools=_NoTools) == [
        LiteFinishToolSet.get_tool_schema("terminate")
    ]
    assert resolve_extra_tools(["response"], tools=_NoTools) == [
        LiteFinishToolSet.get_tool_schema("response")
    ]


def test_resolve_extra_tools_rejects_undeclared_dialect_spelling():
    """A family spelling outside this env's declaration is UNKNOWN.

    The gym resolver is env-relative and model-family-agnostic: every
    undeclared spelling follows the same unknown-name path, and the error still
    lists what the env DOES offer.
    """
    for name in ("answer", "INFO", "finished", "call_user"):
        with pytest.raises(ValueError, match=f"unknown extra_tools: \\['{name}'\\]"):
            resolve_extra_tools([name], tools=_Tools)


def test_finish_tools_preserve_yaml_order_with_env_tools():
    assert resolve_extra_tools(["foo", "terminate", "bar"], tools=_Tools) == [
        _FOO,
        LiteFinishToolSet.get_tool_schema("terminate"),
        _BAR,
    ]


def test_duplicate_names_fail_at_resolution_boundary():
    with pytest.raises(ValueError, match="duplicate extra_tools names: \\['foo'\\]"):
        resolve_extra_tools(["foo", "foo"], tools=_Tools)


def test_duplicate_finish_tools_fail_at_resolution_boundary():
    with pytest.raises(ValueError, match="duplicate extra_tools names: \\['terminate'\\]"):
        resolve_extra_tools(["terminate", "terminate"], tools=_Tools)


def test_duplicate_sandbox_extra_tools_fail_at_resolution_boundary():
    with pytest.raises(ValueError, match=f"duplicate extra_tools names: \\['{BASH_TOOL_NAME}'\\]"):
        resolve_extra_tools(
            [BASH_TOOL_NAME, BASH_TOOL_NAME],
            tools=_Tools,
            executable=_SANDBOX_EXECUTABLE,
        )


def test_merge_extra_tool_schemas_rejects_conflicting_duplicate_names():
    with pytest.raises(LiteContractError, match="duplicate extra_tool_schemas name 'foo'"):
        merge_extra_tool_schemas(
            [_FOO],
            [
                {
                    **make_tool_schema(
                        "foo",
                        description="different schema",
                    ),
                }
            ],
        )


def test_merge_extra_tool_schemas_allows_only_explicit_identical_deduplication():
    assert merge_extra_tool_schemas(
        [_FOO],
        [dict(_FOO)],
        allow_identical_duplicates=True,
    ) == [_FOO]

    with pytest.raises(LiteContractError, match="duplicate extra_tool_schemas name 'foo'"):
        merge_extra_tool_schemas([_FOO], [dict(_FOO)], allow_identical_duplicates=False)


def test_every_browser_nav_action_resolves_to_a_schema():
    """Every browser-nav catalog name resolves to one non-empty schema.

    A browser env may list any ``LiteBrowserNavToolSet`` name in ``extra_tools``; the
    selected schema must be present in model-visible metadata.
    """
    nav_tools = LiteBrowserNavToolSet.get_tool_names()
    assert nav_tools, "LiteBrowserNavToolSet.get_tool_names() unexpectedly empty"
    for name in sorted(nav_tools):
        schemas = LiteBrowserNavToolSet.get_tool_schemas(include=[name])
        assert len(schemas) == 1, (
            f"{name!r} in LiteBrowserNavToolSet.get_tool_names() resolved to {schemas!r}"
        )
        assert tool_schema_name(schemas[0]) == name


def test_open_app_schema_is_canonical_lite_schema():
    schema = make_open_app_tool(["Chrome", "Settings"])

    assert schema["type"] == "function"
    assert tool_schema_name(schema) == "open_app"
    assert tool_schema_parameters(schema)["properties"]["app_name"]["enum"] == [
        "Chrome",
        "Settings",
    ]


def test_open_app_names_use_schema_enum_as_single_catalog_owner():
    metadata = LiteCUAMetadata(
        extra_tool_schemas=[make_open_app_tool(["Chrome"])],
        others={"apps": ["Settings"]},
    )

    assert open_app_names_from_metadata(metadata) == ["Chrome"]


def test_open_app_names_do_not_fall_back_to_metadata_apps():
    metadata = LiteCUAMetadata(others={"apps": ["Settings"]})

    assert open_app_names_from_metadata(metadata) == []


def test_browser_nav_tool_schemas_are_canonical():

    for name in sorted(LiteBrowserNavToolSet.get_tool_names()):
        schemas = LiteBrowserNavToolSet.get_tool_schemas(include=[name])
        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["type"] == "function"
        assert tool_schema_name(schema) == name


def test_browser_nav_tool_schemas_emit_in_declaration_order():
    """Selection order is the catalog's, not the caller's.

    ``get_tool_schemas(include=)`` is the selector for nav tools: it filters
    the declared schema list, emits in declaration order, and collapses
    duplicate include names. Tool order is model-visible input, so this is a
    load-bearing catalog contract.
    """
    declared = [tool_schema_name(schema) for schema in LiteBrowserNavToolSet.get_tool_schemas()]

    schemas = LiteBrowserNavToolSet.get_tool_schemas(include=["close_tab", "goto", "back"])
    assert [tool_schema_name(schema) for schema in schemas] == [
        name for name in declared if name in {"close_tab", "goto", "back"}
    ]
    assert [tool_schema_name(schema) for schema in schemas] == ["goto", "back", "close_tab"]

    # Duplicates collapse under the catalog selector.
    assert [
        tool_schema_name(schema)
        for schema in LiteBrowserNavToolSet.get_tool_schemas(include=["goto", "goto", "back"])
    ] == ["goto", "back"]


def test_finish_and_nav_schema_resolvers_match_shared_catalog_authority():
    """Every path to a finish/nav schema lands on the SAME catalog body.

    The finish path compares the gym resolver's selected schemas with the core
    catalog accessor. The nav path compares the include selector with the
    singular catalog scan. Each pair exercises independent code paths that must
    share the same catalog authority.
    """
    finish_names = sorted(LiteFinishToolSet.get_tool_names())
    # Source 1: the gym resolver. Source 2: the core catalog accessor.
    assert resolve_extra_tools(
        finish_names,
        tools=_NoTools,
        env_name="finish-only",
        executable=frozenset(),
    ) == [LiteFinishToolSet.get_tool_schema(name) for name in finish_names]
    # ...and the include= selector path agrees with the singular scan.
    assert LiteFinishToolSet.get_tool_schemas(include=finish_names) == [
        LiteFinishToolSet.get_tool_schema(tool_schema_name(schema))
        for schema in LiteFinishToolSet.get_tool_schemas()
    ]

    nav_names = sorted(LiteBrowserNavToolSet.get_tool_names())
    assert LiteBrowserNavToolSet.get_tool_schemas(include=nav_names) == [
        LiteBrowserNavToolSet.get_tool_schema(tool_schema_name(schema))
        for schema in LiteBrowserNavToolSet.get_tool_schemas()
    ]

    class _NavTools(BaseTools):
        _SCHEMAS = {
            tool_schema_name(schema): schema
            for schema in LiteBrowserNavToolSet.get_tool_schemas()
        }

    mixed = resolve_extra_tools(
        ["goto", "response", "back", "terminate"],
        tools=_NavTools,
        env_name="browser",
    )
    assert mixed == [
        LiteBrowserNavToolSet.get_tool_schema("goto"),
        LiteFinishToolSet.get_tool_schema("response"),
        LiteBrowserNavToolSet.get_tool_schema("back"),
        LiteFinishToolSet.get_tool_schema("terminate"),
    ]


def test_bash_and_report_infeasible_schemas_come_from_their_core_catalogs():
    """Gym feedback selects ``bash`` / ``report_infeasible``; it declares neither.

    ``LiteShellToolSet`` and ``LiteInfeasibilityToolSet`` are the schema/name
    owners, so the gym resolver's output must be the catalog body verbatim.
    """
    assert BASH_TOOL_NAME == "bash"
    assert LiteShellToolSet.get_tool_names() == frozenset({BASH_TOOL_NAME})
    assert resolve_extra_tools(
        [BASH_TOOL_NAME],
        tools=_NoTools,
        env_name="sandbox",
        executable=frozenset({BASH_TOOL_NAME}),
    ) == [LiteShellToolSet.get_tool_schema(BASH_TOOL_NAME)]

    assert LiteInfeasibilityToolSet.get_tool_names() == frozenset({"report_infeasible"})
    assert make_report_infeasible_tool() == LiteInfeasibilityToolSet.get_tool_schema(
        "report_infeasible"
    )
    # ...and report_infeasible is NOT a finish tool: finish tools are the
    # model/task terminal channels, this one is env feedback.
    assert LiteInfeasibilityToolSet.get_tool_names().isdisjoint(
        LiteFinishToolSet.get_tool_names()
    )


def test_report_infeasible_env_wording_decorates_without_a_second_declaration():
    """Per-env wording is a stamped COPY; the core catalog stays untouched."""
    base = LiteInfeasibilityToolSet.get_tool_schema("report_infeasible")
    stamped = make_report_infeasible_tool(
        description="Report that the task cannot be completed in the current VM.",
        reason_description="Why the requested element is not present.",
    )

    assert stamped["function"]["description"] == (
        "Report that the task cannot be completed in the current VM."
    )
    assert tool_schema_parameters(stamped)["properties"]["reason"]["description"] == (
        "Why the requested element is not present."
    )
    assert tool_schema_parameters(stamped)["required"] == ["reason"]
    assert LiteInfeasibilityToolSet.get_tool_schema("report_infeasible") == base


def test_gym_feedback_surface_declares_no_shared_extra_tool_schema_bodies():
    """The surface module resolves and stamps; it never builds a schema body."""
    source = Path(surface_module.__file__).read_text(encoding="utf-8")

    assert "make_tool_schema" not in source
    assert '"required": ["command"]' not in source
    assert '"required": ["reason"]' not in source


def test_resolve_extra_tools_keeps_canonical_schemas_and_filters():
    """The resolver enforces executable wiring and canonical schemas."""
    schemas = resolve_extra_tools(
        ["response"],
        tools=_NoTools,
        env_name="browsergym",
        executable=frozenset(),
    )
    assert len(schemas) == 1
    assert tool_schema_name(schemas[0]) == "response"
    assert schemas[0]["type"] == "function"
    assert "function" in schemas[0]


def test_finish_tools_resolve_even_when_nothing_is_wired():
    """Finish tools bypass the ``executable`` gate — they are never env-executed."""
    schemas = resolve_extra_tools(
        ["terminate", "response"],
        tools=_NoTools,
        env_name="finish-only",
        executable=frozenset(),
    )
    assert [tool_schema_name(schema) for schema in schemas] == ["terminate", "response"]
    assert all(schema["type"] == "function" and "function" in schema for schema in schemas)


def test_resolution_does_not_second_guess_a_declared_env_tool_name():
    """An env-declared and wired ``answer`` is valid at env resolution.

    ``resolve_extra_tools`` is scoped to env declarations and cannot see model
    families. Dialect-only spelling rejection belongs to row validation, so this
    env-boundary test does not make ``answer`` a canonical finish tool.
    """
    class _AnswerTools(BaseTools):
        _SCHEMAS = {"answer": _ANSWER}

    assert resolve_extra_tools(
        ["answer"],
        tools=_AnswerTools,
        env_name="browsergym",
        executable=frozenset({"answer"}),
    ) == [_ANSWER]


def test_a_declared_dialect_finish_name_is_deferred_to_the_row_boundary():
    """Declared dialect finish names are env-valid but row-invalid.

    ``resolve_extra_tools`` admits a dialect finish spelling the env declares.
    ``lite/data/utils/rows.py::_reject_dialect_only_tool_name`` rejects those
    names from both row contracts by inspecting
    ``metadata.extra_tool_schemas[].name``.

    ``surface.py`` may name that row-boundary guard in prose, but the gym
    package must not import ``lite.data`` to enforce it during env
    configuration.
    """
    from lite.data.utils.rows import (
        _NATIVE_FINISH_TOOL_ALIASES,
        validate_canonical_rows,
        validate_raw_rollout_rows,
    )

    class _AnswerTools(BaseTools):
        _SCHEMAS = {"answer": _ANSWER}

    resolved = resolve_extra_tools(
        ["answer"],
        tools=_AnswerTools,
        env_name="browsergym",
        executable=frozenset({"answer"}),
    )
    row = {
        "images": [],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        ],
        "metadata": LiteCUAMetadata(
            dims=("browser", "use"),
            extra_tool_schemas=resolved,
        ).to_dict(),
    }
    for validate in (validate_canonical_rows, validate_raw_rollout_rows):
        with pytest.raises(ValueError, match=r"'answer' is dialect-only"):
            validate([row], "x13-probe")

    # The gym layer keeps this enforcement at the row boundary: the resolver may
    # name the guard in prose, but it must not import ``lite.data``.
    source = Path(surface_module.__file__).read_text()
    assert "_reject_dialect_only_tool_name" in source
    assert not re.search(r"^\s*(?:from|import)\s+lite\.data\b", source, re.MULTILINE)

    # Every spelling the census carries, not just the one this env declared.
    assert set(_NATIVE_FINISH_TOOL_ALIASES) >= {"answer", "INFO", "finished"}
    for spelling in sorted(_NATIVE_FINISH_TOOL_ALIASES):
        dialect_row = {
            **row,
            "metadata": {
                **row["metadata"],
                "extra_tool_schemas": [make_tool_schema(spelling)],
            },
        }
        with pytest.raises(ValueError, match="is dialect-only"):
            validate_canonical_rows([dialect_row], "x13-probe")


def test_executable_gated_resolution_rejects_duplicate_names():
    class _GotoTools(BaseTools):
        _SCHEMAS = {"goto": LiteBrowserNavToolSet.get_tool_schema("goto")}

    with pytest.raises(ValueError, match="duplicate extra_tools names: \\['goto'\\]"):
        resolve_extra_tools(
            ["goto", "goto"],
            tools=_GotoTools,
            env_name="browsergym",
            executable=frozenset({"goto"}),
        )


def test_resolve_extra_tools_rejects_env_tool_set_finish_overrides():
    """This is WHY every name predicate must union in ``LiteFinishToolSet.get_tool_names()``."""
    with pytest.raises(ValueError, match="must not define canonical finish tools"):
        resolve_extra_tools(["response"], tools=_FinishOverrideTools)


def test_resolve_extra_tools_rejects_noncanonical_schema_keys():
    class _InvalidSchemaTools(BaseTools):
        _SCHEMAS = {
            "foo": {
                **make_tool_schema("foo"),
                "name": "foo",
            },
        }

    with pytest.raises(LiteContractError, match="noncanonical outer keys"):
        resolve_extra_tools(
            ["foo"],
            tools=_InvalidSchemaTools,
            env_name="browsergym",
            executable=frozenset({"foo"}),
        )


def test_resolve_extra_tools_rejects_declared_but_unwired_names():
    """``executable`` is the SECOND, orthogonal gate: ``back`` is declared and
    still rejected because this backend is not wired to run it."""
    class _NavTools(BaseTools):
        _SCHEMAS = {
            tool_schema_name(schema): schema
            for schema in LiteBrowserNavToolSet.get_tool_schemas(include=["goto", "back"])
        }

    with pytest.raises(ValueError, match="cannot execute extra_tools"):
        resolve_extra_tools(
            ["back"],
            tools=_NavTools,
            env_name="webharbor.webvoyager",
            executable=frozenset({"goto"}),
        )
