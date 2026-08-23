"""BrowserGym tool-schema compilation + ``extra_tools`` tri-state regressions.

Two behaviours pinned here, both of which used to be silently wrong:

  - ``_json_schema_for_annotation`` carries BrowserGym's ``Literal[...]`` /
    ``list[Literal[...]]`` / union annotations THROUGH to the emitted tool
    schema. The compiler previously flattened every non-trivial annotation to
    ``"string"``/``"array"``, so ``click.button``, ``click.modifiers`` and the
    four ``mouse_*`` buttons advertised no enum at all and
    ``select_option.options`` never advertised its multi-select list form.
    Reachability is asserted on the emitted schemas of NAMED action subsets and
    on a live ``BrowserGymEnv``'s ``metadata.extra_tool_schemas`` — not on the
    helper alone.
  - ``extra_tools`` is a TRI-state here, unlike the two-state shared
    ``resolve_extra_tools``: ``None`` (the shipped yaml default) exposes the
    whole ``action_subsets``-derived catalog, ``[]`` exposes nothing, and a
    list selects that subset. All three legs are pinned together — a revision
    that collapsed ``None`` onto ``[]`` silently removed ``response``
    (``send_msg_to_user``) from every WebArena / VisualWebArena episode, and
    nothing caught it because the tests pinned only the then-current
    behaviour.

Requires:
  - ``uv sync`` (browsergym is a project dependency).

Run:
    uv run pytest tests/gym/envs/browsergym/test_browsergym_tool_schema.py -v
"""

from __future__ import annotations

from typing import Any, Literal

import pytest

pytest.importorskip("browsergym.core", reason="browsergym not installed")

from lite.core.tools.schemas import tool_schema_name, tool_schema_parameters
from lite.gym.envs.browsergym.main import (
    CFG,
    BrowserGymConfig,
    BrowserGymEnv,
    _bgym_param_info,
    _extra_tool_schemas_for_subsets,
    _json_schema_for_annotation,
    _tool_schema_from_signature,
    _tools_for_subsets,
)

# The two enums BrowserGym 0.14.3 declares on its pointer actions
# (.venv/lib/python3.12/site-packages/browsergym/core/action/functions.py:148/149).
_BUTTON_ENUM = ["left", "middle", "right"]
_MODIFIER_ENUM = ["Alt", "Control", "ControlOrMeta", "Meta", "Shift"]
# ``str | list[str]`` (select_option.options :127, upload_file.file :616).
_STR_OR_LIST = {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]}


def _props(name: str, subsets: tuple[str, ...]) -> dict[str, Any]:
    """The emitted properties for ``name`` in the catalog ``subsets`` derives."""
    catalog = {tool_schema_name(schema): schema for schema in _tools_for_subsets(subsets)}
    assert name in catalog, f"{name} not in {sorted(catalog)} for subsets={subsets}"
    return tool_schema_parameters(catalog[name])["properties"]


# ---------------------------------------------------------------------------
# Annotation → JSON-schema fragment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "annotation,expected",
    [
        (str, {"type": "string"}),
        (int, {"type": "integer"}),
        (float, {"type": "number"}),
        (bool, {"type": "boolean"}),
        (list, {"type": "array", "items": {"type": "string"}}),
        (list[str], {"type": "array", "items": {"type": "string"}}),
        (Literal["left", "middle", "right"], {"type": "string", "enum": _BUTTON_ENUM}),
        (
            list[Literal["Alt", "Shift"]],
            {"type": "array", "items": {"type": "string", "enum": ["Alt", "Shift"]}},
        ),
        (str | list[str], _STR_OR_LIST),
    ],
)
def test_annotation_compiles_to_fragment(annotation: Any, expected: dict[str, Any]):
    assert _json_schema_for_annotation(annotation) == expected


def test_unknown_annotation_degrades_to_string():
    class _Weird:
        pass

    assert _json_schema_for_annotation(_Weird) == {"type": "string"}


def test_param_info_carries_the_fragment_not_a_type_name():
    button = {p.name: p for p in _bgym_param_info("click")}["button"]
    assert button.schema == {"type": "string", "enum": _BUTTON_ENUM}
    assert button.required is False


def test_emitted_property_dicts_do_not_alias_the_cache():
    """Each emitted schema owns its fragments; mutating one cannot poison the next."""
    first = tool_schema_parameters(_tool_schema_from_signature("click"))["properties"]
    first["button"]["enum"].append("poison")
    second = tool_schema_parameters(_tool_schema_from_signature("click"))["properties"]
    assert second["button"]["enum"] == _BUTTON_ENUM


# ---------------------------------------------------------------------------
# Reachability: the enum reaches the emitted schema of real action subsets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subsets", [("webarena",), ("visualwebarena",), ("bid",)])
def test_click_enums_reach_bid_style_subsets(subsets: tuple[str, ...]):
    props = _props("click", subsets)
    assert props["button"] == {"type": "string", "enum": _BUTTON_ENUM}
    assert props["modifiers"] == {
        "type": "array",
        "items": {"type": "string", "enum": _MODIFIER_ENUM},
    }


def test_dblclick_enums_reach_the_bid_subset():
    props = _props("dblclick", ("bid",))
    assert props["button"]["enum"] == _BUTTON_ENUM
    assert props["modifiers"]["items"]["enum"] == _MODIFIER_ENUM


@pytest.mark.parametrize("name", ["mouse_click", "mouse_dblclick", "mouse_down", "mouse_up"])
def test_mouse_button_enums_reach_the_miniwob_subset(name: str):
    props = _props(name, ("miniwob_all",))
    assert props["button"] == {"type": "string", "enum": _BUTTON_ENUM}
    # The coordinates keep their scalar types.
    assert props["x"] == {"type": "number"}


@pytest.mark.parametrize("subsets", [("webarena",), ("visualwebarena",), ("bid",)])
def test_select_option_advertises_the_multi_select_form(subsets: tuple[str, ...]):
    assert _props("select_option", subsets)["options"] == _STR_OR_LIST


def test_upload_file_union_survives_the_deleted_special_case():
    """``upload_file.file`` used to be hand-special-cased; the generic union rule
    must reproduce the exact same fragment."""
    assert _props("upload_file", ("visualwebarena",))["file"] == _STR_OR_LIST


def test_every_array_property_declares_items():
    """Strict consumers (OpenAI function-tool API) reject ``"array"`` without ``items``."""
    for subsets in [("webarena",), ("visualwebarena",), ("bid",), ("miniwob_all",)]:
        for schema in _tools_for_subsets(subsets):
            for pname, prop in tool_schema_parameters(schema)["properties"].items():
                if prop.get("type") == "array":
                    assert "items" in prop, f"{subsets} {tool_schema_name(schema)}.{pname}"


@pytest.mark.asyncio
async def test_click_enum_reaches_live_env_metadata():
    """End-to-end: a WebArena env that selects ``click`` advertises the enum."""
    config = BrowserGymConfig(
        bgym_task_id="miniwob.click-dialog",
        benchmark="miniwob",
        action_subsets=("webarena",),
    )
    env = BrowserGymEnv(config=config, use_fake=True, extra_tools=["click", "select_option"])
    try:
        await env.reset()
        schemas = {tool_schema_name(s): s for s in env.metadata.extra_tool_schemas}
        click_props = tool_schema_parameters(schemas["click"])["properties"]
        assert click_props["button"]["enum"] == _BUTTON_ENUM
        assert click_props["modifiers"]["items"]["enum"] == _MODIFIER_ENUM
        assert (
            tool_schema_parameters(schemas["select_option"])["properties"]["options"]
            == _STR_OR_LIST
        )
    finally:
        await env.close()


# ---------------------------------------------------------------------------
# extra_tools TRI-state: None -> whole catalog, [] -> none, [names] -> subset
#
# All three legs are asserted together, and against the SAME catalog, so no
# single leg can be "fixed" by collapsing two of them onto each other. That is
# exactly how the regression escaped: a test pinned only ``None == []``.
# ---------------------------------------------------------------------------

_TRI_STATE_SUBSETS = ("coord", "chat", "infeas", "nav", "tab")


def _catalog_names(subsets: tuple[str, ...]) -> list[str]:
    return [tool_schema_name(schema) for schema in _tools_for_subsets(subsets)]


def test_resolver_tri_state():
    catalog = _catalog_names(_TRI_STATE_SUBSETS)
    # The catalog must be non-empty or the three legs are indistinguishable.
    assert "response" in catalog and "goto" in catalog

    # None -> the WHOLE catalog, in catalog order.
    assert [
        tool_schema_name(s) for s in _extra_tool_schemas_for_subsets(_TRI_STATE_SUBSETS, None)
    ] == catalog
    # [] -> nothing.
    assert _extra_tool_schemas_for_subsets(_TRI_STATE_SUBSETS, []) == []
    # [names] -> exactly those, in the REQUESTED order (not catalog order).
    requested = ["goto", "response"]
    assert [
        tool_schema_name(s) for s in _extra_tool_schemas_for_subsets(_TRI_STATE_SUBSETS, requested)
    ] == requested
    # ...and the three legs are pairwise distinct.
    assert catalog != [] and catalog != requested


@pytest.mark.parametrize(
    "extra_tools,expected",
    [
        (None, "__catalog__"),
        ([], []),
        (["response"], ["response"]),
        (["goto", "response"], ["goto", "response"]),
    ],
)
@pytest.mark.asyncio
async def test_env_extra_tools_tri_state(extra_tools: list[str] | None, expected: Any):
    """The same tri-state through the live env's ``metadata.extra_tool_schemas``."""
    config = BrowserGymConfig(
        bgym_task_id="miniwob.click-dialog",
        benchmark="miniwob",
        action_subsets=_TRI_STATE_SUBSETS,
    )
    if expected == "__catalog__":
        expected = _catalog_names(_TRI_STATE_SUBSETS)
    env = BrowserGymEnv(config=config, use_fake=True, extra_tools=extra_tools)
    try:
        await env.reset()
        assert [tool_schema_name(s) for s in env.metadata.extra_tool_schemas] == expected
        # None is KEPT as None — collapsing it to [] is the regression.
        assert env._extra_tools == extra_tools
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_default_construction_answers_through_send_msg_to_user():
    """The SHIPPED default (``configs/*.yaml: extra_tools: null``) must leave
    ``response`` — the canonical name for BrowserGym's ``send_msg_to_user`` —
    reachable. WebArena / VisualWebArena information-seeking tasks answer
    through that call; without it they are structurally unanswerable (reward 0,
    no episode can terminate, every one burns its full step budget)."""
    assert CFG.env_kwargs["extra_tools"] is None
    config = BrowserGymConfig(
        bgym_task_id="webarena.0",
        benchmark="webarena",
        action_subsets=_TRI_STATE_SUBSETS,
    )
    # No ``extra_tools`` kwarg: the yaml default flows through the signature.
    env = BrowserGymEnv(config=config, use_fake=True)
    try:
        await env.reset()
        assert "response" in {tool_schema_name(s) for s in env.metadata.extra_tool_schemas}
    finally:
        await env.close()


def test_unknown_extra_tool_still_raises_at_construction():
    config = BrowserGymConfig(bgym_task_id="miniwob.click-dialog", benchmark="miniwob")
    with pytest.raises(ValueError, match="unknown extra_tools"):
        BrowserGymEnv(config=config, use_fake=True, extra_tools=["totally_fake"])
