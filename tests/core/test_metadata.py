"""Container-field contract for Lite task metadata.

The durable row shape is tagged by ``metadata_kind``. ``dims`` is the generic
routing coordinate tuple; CUA-only facts such as ``platform``, ``task_type``,
and ``valid_actions`` live only on :class:`LiteCUAMetadata`.

Run:
    uv run pytest tests/core/test_metadata.py -q
"""

from __future__ import annotations

import pytest

from lite.core.errors import LiteContractError
from lite.core.metadata import (
    LiteBaseMetadata,
    LiteCUAMetadata,
    LiteGenericMetadata,
    metadata_from_dict,
)
from lite.core.tools.schemas import make_tool_schema, tool_schema_name, tool_schema_parameters


def _cua_row(**over):
    row = {"metadata_kind": "cua", "dims": ["desktop", "use"]}
    row.update(over)
    return row


def _generic_row(**over):
    row = {"metadata_kind": "generic", "dims": []}
    row.update(over)
    return row


def test_cua_row_round_trips_through_dispatcher():
    md = metadata_from_dict(
        _cua_row(
            extra_tool_schemas=[make_tool_schema("response", parameters={})],
            valid_actions=[],
            others={"env_id": "webgym"},
        )
    )

    assert isinstance(md, LiteCUAMetadata)
    assert md.dims == ("desktop", "use")
    assert md.platform is LiteCUAMetadata.Platform.DESKTOP
    assert md.task_type is LiteCUAMetadata.TaskType.USE
    assert md.to_dict() == {
        "metadata_kind": "cua",
        "dims": ["desktop", "use"],
        "extra_tool_schemas": [make_tool_schema("response", parameters={})],
        "valid_actions": [],
        "others": {"env_id": "webgym"},
    }


def test_generic_row_with_empty_dims_round_trips():
    md = metadata_from_dict(
        _generic_row(
            extra_tool_schemas=[make_tool_schema("answer", parameters={})],
            others={"dataset": "geo3k"},
        )
    )

    assert isinstance(md, LiteGenericMetadata)
    assert md.dims == ()
    assert md.to_dict() == {
        "metadata_kind": "generic",
        "dims": [],
        "extra_tool_schemas": [make_tool_schema("answer", parameters={})],
        "others": {"dataset": "geo3k"},
    }


def test_missing_optional_containers_default_to_owned_mutable_values():
    cua = metadata_from_dict(_cua_row())
    generic = metadata_from_dict(_generic_row())

    assert cua.extra_tool_schemas == []
    assert cua.valid_actions is None
    assert cua.others == {}
    assert generic.extra_tool_schemas == []
    assert generic.others == {}


@pytest.mark.parametrize(
    ("row_factory", "field", "value", "match"),
    [
        (_cua_row, "others", None, "metadata\\.others must be a dict"),
        (_generic_row, "others", [], "metadata\\.others must be a dict"),
        (
            _cua_row,
            "extra_tool_schemas",
            None,
            "metadata\\.extra_tool_schemas must be a list",
        ),
        (
            _generic_row,
            "extra_tool_schemas",
            {},
            "metadata\\.extra_tool_schemas must be a list",
        ),
        (_cua_row, "valid_actions", "click", "metadata\\.valid_actions"),
        (_cua_row, "valid_actions", ["click", 7], "metadata\\.valid_actions"),
    ],
)
def test_explicit_bad_container_fields_are_rejected(row_factory, field, value, match):
    with pytest.raises(LiteContractError, match=match):
        metadata_from_dict(row_factory(**{field: value}))


def test_container_fields_are_copied_not_aliased():
    schema = make_tool_schema("open_app", parameters={})
    schemas = [schema]
    others = {"k": 1}
    md = LiteCUAMetadata(extra_tool_schemas=schemas, others=others)

    md.extra_tool_schemas.append(make_tool_schema("close_app", parameters={}))
    md.others["k"] = 2

    assert schemas == [schema]
    assert others == {"k": 1}


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param({"type": "function", "parameters": {}}, id="no-name"),
        pytest.param(
            {"type": "function", "function": {"name": "", "parameters": {}}},
            id="empty-name",
        ),
        pytest.param(
            {"type": "function", "function": {"name": None, "parameters": {}}},
            id="null-name",
        ),
        pytest.param(
            {"type": "function", "function": {"name": 7, "parameters": {}}},
            id="int-name",
        ),
        pytest.param({"type": "function", "function": {"name": "a"}}, id="no-parameters"),
        pytest.param(
            {"type": "function", "function": {"name": "a", "parameters": None}},
            id="null-parameters",
        ),
        pytest.param(
            {"type": "function", "function": {"name": "a", "parameters": []}},
            id="list-parameters",
        ),
    ],
)
def test_nameless_or_parameterless_extra_tool_schema_cannot_be_constructed(bad):
    with pytest.raises(LiteContractError):
        metadata_from_dict(_generic_row(extra_tool_schemas=[bad]))


@pytest.mark.parametrize(
    ("function_patch", "match"),
    [
        ({"description": None}, "function\\.description must be a string"),
        ({"description": 7}, "function\\.description must be a string"),
        ({"strict": None}, "function\\.strict must be a bool"),
        ({"strict": "true"}, "function\\.strict must be a bool"),
    ],
)
def test_optional_extra_tool_schema_fields_are_typed(function_patch, match):
    schema = make_tool_schema("open_app", parameters={})
    schema["function"].update(function_patch)

    with pytest.raises(LiteContractError, match=match):
        metadata_from_dict(_cua_row(extra_tool_schemas=[schema]))


def test_every_admitted_extra_tool_schema_has_a_str_name_and_dict_parameters():
    md = metadata_from_dict(
        _generic_row(
            extra_tool_schemas=[
                make_tool_schema("open_app", parameters={"type": "object"}),
                make_tool_schema("answer", parameters={}),
            ]
        )
    )
    for schema in md.extra_tool_schemas:
        assert isinstance(tool_schema_name(schema), str) and tool_schema_name(schema)
        assert isinstance(tool_schema_parameters(schema), dict)


def test_lite_metadata_accepts_canonical_extra_tool_schema():
    schema = make_tool_schema(
        "goto",
        description="Navigate.",
        parameters={
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    )

    meta = LiteCUAMetadata(extra_tool_schemas=[schema])

    assert meta.extra_tool_schemas == [schema]


@pytest.mark.parametrize(
    "schema,match",
    [
        ({"type": "function", "name": "goto", "parameters": {}}, "noncanonical"),
        (
            {"type": "function_call", "function": {"name": "goto", "parameters": {}}},
            "type",
        ),
        (
            {"type": "function", "function": {"name": "goto", "parameters": {}}, "id": "native"},
            "noncanonical",
        ),
        ({"type": "function", "function": {"name": "goto"}}, "parameters"),
    ],
)
def test_lite_metadata_rejects_noncanonical_extra_tool_schemas(schema, match):
    with pytest.raises(LiteContractError, match=match):
        LiteCUAMetadata(extra_tool_schemas=[schema])


def test_lite_metadata_rejects_duplicate_extra_tool_schema_names():
    schema = make_tool_schema("goto")

    with pytest.raises(LiteContractError, match="duplicate"):
        LiteCUAMetadata(extra_tool_schemas=[schema, dict(schema)])


@pytest.mark.parametrize("name", ["answer", "computer", "mobile", "point", "bbox"])
def test_lite_metadata_accepts_tool_names_shape_only(name):
    schema = make_tool_schema(name)

    meta = LiteCUAMetadata(extra_tool_schemas=[schema])

    assert meta.extra_tool_schemas == [schema]


@pytest.mark.parametrize("missing", ["metadata_kind", "dims"])
def test_required_tag_and_dims_are_required(missing):
    row = _cua_row()
    del row[missing]

    with pytest.raises(LiteContractError, match="missing required keys"):
        metadata_from_dict(row)


def test_unknown_metadata_kind_fails():
    with pytest.raises(LiteContractError, match="unknown metadata\\.metadata_kind"):
        metadata_from_dict({"metadata_kind": "math", "dims": []})


@pytest.mark.parametrize("field", ["platform", "task_type"])
def test_retired_top_level_cua_axis_fields_are_rejected(field):
    with pytest.raises(LiteContractError, match="unknown keys"):
        metadata_from_dict(_cua_row(**{field: "desktop"}))


def test_generic_row_with_valid_actions_is_rejected():
    with pytest.raises(LiteContractError, match="unknown keys"):
        metadata_from_dict(_generic_row(valid_actions=[]))


def test_generic_metadata_rejects_dynamic_valid_actions_assignment():
    md = LiteGenericMetadata()

    with pytest.raises(AttributeError):
        md.valid_actions = []


@pytest.mark.parametrize(
    ("row_factory", "unknown"),
    [
        (_cua_row, {"x": 1}),
        (_generic_row, {"x": 1}),
    ],
)
def test_unknown_keys_fail(row_factory, unknown):
    with pytest.raises(LiteContractError, match="unknown keys"):
        metadata_from_dict(row_factory(**unknown))


def test_to_dict_key_sets_are_exactly_canonical():
    assert list(LiteCUAMetadata().to_dict()) == [
        "metadata_kind",
        "dims",
        "extra_tool_schemas",
        "valid_actions",
        "others",
    ]
    assert list(LiteGenericMetadata().to_dict()) == [
        "metadata_kind",
        "dims",
        "extra_tool_schemas",
        "others",
    ]


def test_dispatcher_passes_through_existing_metadata_instances():
    for md in (LiteCUAMetadata(), LiteGenericMetadata()):
        assert metadata_from_dict(md) is md
        assert isinstance(md, LiteBaseMetadata)


def test_dispatcher_rejects_non_dict_non_metadata_inputs():
    for value in (None, [], "metadata"):
        with pytest.raises(LiteContractError, match="metadata must be a dict"):
            metadata_from_dict(value)


def test_direct_cua_construction_defaults_to_desktop_use_dims():
    md = LiteCUAMetadata()

    assert md.dims == ("desktop", "use")
    assert md.platform is LiteCUAMetadata.Platform.DESKTOP
    assert md.task_type is LiteCUAMetadata.TaskType.USE


@pytest.mark.parametrize(
    ("platform", "task_type"),
    [
        pytest.param(
            platform.value,
            task_type.value,
            id=f"{platform.value}-{task_type.value}",
        )
        for platform in LiteCUAMetadata.Platform
        for task_type in LiteCUAMetadata.TaskType
    ],
)
def test_cua_dims_canonical_matrix_round_trips(platform, task_type):
    md = metadata_from_dict(_cua_row(dims=[platform, task_type]))

    assert isinstance(md, LiteCUAMetadata)
    assert md.dims == (platform, task_type)
    assert md.platform.value == platform
    assert md.task_type.value == task_type
    row = md.to_dict()
    assert row["dims"] == [platform, task_type]
    assert "platform" not in row
    assert "task_type" not in row


@pytest.mark.parametrize(
    "dims",
    [
        pytest.param([], id="empty"),
        pytest.param(["geo3k"], id="single-axis"),
        pytest.param(["math", "answer.with.dot"], id="dotted-axis"),
    ],
)
def test_generic_dims_canonical_matrix_round_trips(dims):
    md = metadata_from_dict(_generic_row(dims=dims))

    assert isinstance(md, LiteGenericMetadata)
    assert md.dims == tuple(dims)
    row = md.to_dict()
    assert row["dims"] == dims
    assert "valid_actions" not in row


@pytest.mark.parametrize(
    ("dims", "match"),
    [
        ("desktop", "metadata\\.dims must be a list or tuple"),
        (["desktop", "use", "extra"], "CUA metadata must have two entries"),
        (["desktop"], "CUA metadata must have two entries"),
        ([7, "use"], "metadata\\.dims entries must be strings"),
        (["", "use"], "metadata\\.dims entries must be non-empty"),
        (["desk@top", "use"], "metadata\\.dims entries must not contain '@'"),
        (["tablet", "use"], "metadata\\.dims\\[0\\] must be one of"),
        (["desktop", "browse"], "metadata\\.dims\\[1\\] must be one of"),
    ],
)
def test_invalid_dims_fail(dims, match):
    with pytest.raises(LiteContractError, match=match):
        LiteCUAMetadata(dims=dims)


def test_valid_actions_none_is_preserved_not_normalised():
    assert LiteCUAMetadata.from_dict(_cua_row()).valid_actions is None
    assert LiteCUAMetadata.from_dict(_cua_row(valid_actions=None)).valid_actions is None
    assert LiteCUAMetadata.from_dict(_cua_row(valid_actions=[])).valid_actions == []


def test_valid_actions_none_and_empty_list_stay_distinct_through_to_dict():
    no_gate = LiteCUAMetadata(valid_actions=None).to_dict()
    gate_everything = LiteCUAMetadata(valid_actions=[]).to_dict()

    assert no_gate["valid_actions"] is None
    assert gate_everything["valid_actions"] == []
    assert no_gate != gate_everything


def test_single_turn_task_types_partition_the_metadata_enum():
    from lite.core.metadata import SINGLE_TURN_TASK_TYPES

    assert SINGLE_TURN_TASK_TYPES | {LiteCUAMetadata.TaskType.USE} == set(
        LiteCUAMetadata.TaskType
    )
