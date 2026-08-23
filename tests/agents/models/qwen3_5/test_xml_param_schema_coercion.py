"""Qwen3.5 XML ``<parameter=...>`` coercion is driven by the DECLARED schema.

``_coerce_param_value`` must convert parameter bodies from the declared schema,
not from hardcoded key-name allowlists (``pixels`` int, ``time``/``duration``
float, ``coordinate``/``keys`` structured). A standalone extra tool's numeric
params match none of those names, so ``scroll(delta_x: number)`` can otherwise
arrive as a string and fail inside browsergym arithmetic.

Every schema asserted on here is DERIVED FROM PRODUCTION CODE — browsergym's
``_tool_schema_from_signature`` (the exact builder behind
``metadata.extra_tool_schemas``) and ``lite.core.tools.extra_tools.LiteBrowserNavToolSet``
— never hand-written next to the assertion, and the tests additionally assert
the declared type is visible in the rendered ``<tools>`` prompt block, so the
parse side is checked against the contract the model was actually shown.

Run:
    uv run pytest tests/agents/models/qwen3_5/test_xml_param_schema_coercion.py -v
"""

from __future__ import annotations

import json

import pytest

from lite.agents.models.qwen3_5.adapter import (
    Qwen3_5BaseAdapter,
    Qwen3_5DesktopUseAdapter,
    Qwen3_5MobileUseAdapter,
    _coerce_param_value,
    _param_types_from_tool_schemas,
)
from lite.core import LiteCUAMetadata
from lite.core.tools.extra_tools import LiteBrowserNavToolSet
from lite.core.tools.schemas import tool_schema_name, tool_schema_parameters

# =============================================================================
# Helpers — production schema sources + the real wire format
# =============================================================================

def _bgym_schemas(*names: str) -> list[dict]:
    """Schemas exactly as the browsergym env puts them in metadata.

    ``_tool_schema_from_signature`` introspects the live BrowserGym action
    function, so the JSON types come from BrowserGym's own signatures
    (``scroll(delta_x: float, delta_y: float)`` → ``number``), not from us.
    """
    pytest.importorskip(
        "browsergym.core.action.functions",
        reason="browsergym extra not installed; bid-mode schemas unavailable",
    )
    from lite.gym.envs.browsergym.main import _tool_schema_from_signature

    schemas = [_tool_schema_from_signature(n) for n in names]
    assert all(s is not None for s in schemas), f"unknown browsergym actions in {names}"
    return schemas


def _bid_mode_adapter(extra_tool_schemas: list[dict]) -> Qwen3_5BaseAdapter:
    """The adapter shape used on bid-mode rows.

    ``adapter_key: "qwen3_5.base@browser@use"`` + ``valid_actions: []``
    (provider-native wrapper dropped) + env ``extra_tool_schemas``.
    """
    return Qwen3_5BaseAdapter(
        metadata=LiteCUAMetadata(
            dims=(LiteCUAMetadata.Platform.BROWSER, LiteCUAMetadata.TaskType.USE),
            extra_tool_schemas=extra_tool_schemas,
            valid_actions=[],
        ),
    )


def _xml(name: str, **params: str) -> str:
    """The model's wire format: every value is TEXT, types come from schema."""
    body = "".join(
        f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items()
    )
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def _args(adapter, response: str) -> dict:
    parsed = adapter.parse_raw_assistant_response(response)
    [call] = parsed["tool_calls"]
    return call["arguments"]


def _declared_type(schemas: list[dict], tool: str, param: str) -> str:
    [schema] = [s for s in schemas if tool_schema_name(s) == tool]
    return tool_schema_parameters(schema)["properties"][param]["type"]


# =============================================================================
# The defect: numeric params of standalone extra tools
# =============================================================================

def test_number_param_of_standalone_extra_parses_as_float():
    """``scroll(delta_x, delta_y)`` — the 1360-turn campaign failure.

    Pre-fix: ``{'delta_x': '0', 'delta_y': '-300'}`` (str) → browsergym
    ``scroll('0', '-300')`` → ``TypeError: unsupported operand type(s) for /``.
    """
    schemas = _bgym_schemas("scroll")
    assert _declared_type(schemas, "scroll", "delta_x") == "number"
    adapter = _bid_mode_adapter(schemas)

    # The prompt the model sees declares the type we coerce by (no circularity:
    # the schema is browsergym's, and it reaches both prompt and parser).
    assert '"delta_x": {"type": "number"}' in adapter._build_tools_section()

    args = _args(adapter, _xml("scroll", delta_x="0", delta_y="-300"))
    assert args == {"delta_x": 0.0, "delta_y": -300.0}
    assert isinstance(args["delta_y"], float)


def test_integer_param_of_standalone_extra_parses_as_int():
    """``tab_focus(index: int)`` → ``integer``. Pre-fix: ``'2'`` (str)."""
    schemas = _bgym_schemas("tab_focus")
    assert _declared_type(schemas, "tab_focus", "index") == "integer"
    adapter = _bid_mode_adapter(schemas)

    args = _args(adapter, _xml("tab_focus", index="2"))
    assert args == {"index": 2}
    assert isinstance(args["index"], int) and not isinstance(args["index"], bool)


def test_integer_param_of_canonical_nav_extra_parses_as_int():
    """Same defect on the canonical (coord-mode) nav surface, which needs no
    browsergym install: ``switch_tab(index)`` is declared ``integer`` by
    ``lite.agents.core.action_space.base``. Pre-fix: ``'1'`` (str)."""
    schemas = LiteBrowserNavToolSet.get_tool_schemas(include=["switch_tab", "goto"])
    assert _declared_type(schemas, "switch_tab", "index") == "integer"
    adapter = _bid_mode_adapter(schemas)

    assert _args(adapter, _xml("switch_tab", index="1")) == {"index": 1}
    # ...and a declared ``string`` param on the same surface stays a string.
    assert _args(adapter, _xml("goto", url="http://localhost:8600/")) == {
        "url": "http://localhost:8600/",
    }


def test_boolean_param_of_standalone_extra_parses_as_bool():
    """``fill(..., enable_autocomplete_menu: bool)``. Pre-fix: ``'false'``
    (str) — truthy in Python, so the flag was stuck ON."""
    schemas = _bgym_schemas("fill")
    assert _declared_type(schemas, "fill", "enable_autocomplete_menu") == "boolean"
    adapter = _bid_mode_adapter(schemas)

    args = _args(adapter, _xml("fill", bid="132", value="x", enable_autocomplete_menu="false"))
    assert args["enable_autocomplete_menu"] is False


# =============================================================================
# What must NOT change: declared strings, tolerant fallback
# =============================================================================

def test_numeric_looking_string_param_stays_string():
    """A ``string``-declared param keeps its text even when it looks numeric.

    browsergym bids ARE numeric-looking strings ("12"); coercing them would
    emit ``click(12)`` and break every bid-mode click. Guards against a fix
    that sniffs the value instead of reading the schema.
    """
    schemas = _bgym_schemas("fill", "click")
    assert _declared_type(schemas, "fill", "bid") == "string"
    assert _declared_type(schemas, "fill", "value") == "string"
    adapter = _bid_mode_adapter(schemas)

    args = _args(adapter, _xml("fill", bid="132", value="42.5"))
    assert args == {"bid": "132", "value": "42.5"}
    assert isinstance(args["bid"], str) and isinstance(args["value"], str)
    assert _args(adapter, _xml("click", bid="12")) == {"bid": "12"}


def test_unparseable_value_falls_back_to_raw_string():
    """Tolerant fallback: a value that does not parse as its declared type is
    returned verbatim, never raised. Callers depend on always getting args."""
    schemas = _bgym_schemas("scroll", "tab_focus")
    adapter = _bid_mode_adapter(schemas)

    # integer-declared, unparseable
    assert _args(adapter, _xml("tab_focus", index="second")) == {"index": "second"}
    # number-declared, unparseable
    assert _args(adapter, _xml("scroll", delta_x="a bit", delta_y="-300")) == {
        "delta_x": "a bit", "delta_y": -300.0,
    }
    # array-declared, unparseable → raw string (browsergym ``click.modifiers``)
    [click] = _bgym_schemas("click")
    adapter2 = _bid_mode_adapter([click])
    assert _declared_type([click], "click", "modifiers") == "array"
    assert _args(adapter2, _xml("click", bid="a1", modifiers="Ctrl and Shift")) == {
        "bid": "a1", "modifiers": "Ctrl and Shift",
    }


@pytest.mark.parametrize("json_type", ["integer", "number"])
@pytest.mark.parametrize("raw", ["1e400", "-1e400", "inf", "-inf", "nan"])
def test_nonfinite_numeric_values_fall_back_to_raw_string(json_type: str, raw: str):
    """The XML wire is plain text, so non-finite numbers bypass JSON hooks.

    Declared numeric parameters keep the existing tolerant contract: malformed
    values come back raw rather than escaping as ``OverflowError`` or becoming
    ``float("inf")``.
    """
    assert _coerce_param_value("pixels", raw, json_type=json_type) == raw


# =============================================================================
# computer_use — the 0%-failure path, must be byte-for-byte unchanged
# =============================================================================

_COMPUTER_USE_CASES = [
    # (params emitted, expected arguments) — the values the pre-fix allowlists
    # produced, asserted verbatim so any drift on this path fails loudly.
    ({"action": "scroll", "pixels": "-300"}, {"action": "scroll", "pixels": -300}),
    ({"action": "scroll", "pixels": "-3.0"}, {"action": "scroll", "pixels": -3}),
    ({"action": "wait", "time": "2.5"}, {"action": "wait", "time": 2.5}),
    # ``duration`` is emitted by the model but declared by NO computer_use
    # schema — it must still ride the name fallback to float.
    ({"action": "wait", "duration": "3"}, {"action": "wait", "duration": 3.0}),
    (
        {"action": "left_click", "coordinate": "[491, 91]"},
        {"action": "left_click", "coordinate": [491, 91]},
    ),
    ({"action": "key", "keys": '["ctrl", "a"]'}, {"action": "key", "keys": ["ctrl", "a"]}),
    # double-wrapped keys still get the _clean_key_tokens repair
    ({"action": "key", "keys": '["[\'ctrl", "a\']"]'}, {"action": "key", "keys": ["ctrl", "a"]}),
    # numeric-looking ``text`` is a declared string and must stay one
    ({"action": "type", "text": "2024"}, {"action": "type", "text": "2024"}),
    # unparseable structured value falls back to the raw string
    ({"action": "left_click", "coordinate": "somewhere"},
     {"action": "left_click", "coordinate": "somewhere"}),
]


@pytest.mark.parametrize("params,expected", _COMPUTER_USE_CASES)
def test_computer_use_coercion_unchanged(params, expected):
    """The native GUI path is at 0% failure in the campaign — pin it."""
    adapter = Qwen3_5DesktopUseAdapter()
    args = _args(adapter, _xml("computer_use", **params))
    assert args == expected
    for key, want in expected.items():
        assert type(args[key]) is type(want), f"{key}: {type(args[key])} != {type(want)}"


def test_computer_use_schema_types_match_the_legacy_allowlists():
    """Why computer_use cannot drift: its declared types ARE the allowlists.

    ``pixels: integer``, ``time: number``, ``coordinate``/``keys: array`` — the
    schema-driven branch takes the identical path the name-driven one did.
    """
    adapter = Qwen3_5DesktopUseAdapter()
    types = adapter._xml_param_types()["computer_use"]
    assert types["pixels"] == "integer"
    assert types["time"] == "number"
    assert types["coordinate"] == "array"
    assert types["keys"] == "array"
    assert types["text"] == "string"
    assert "duration" not in types  # → name fallback, see the case above

    mobile = Qwen3_5MobileUseAdapter()._xml_param_types()["mobile_use"]
    assert mobile["coordinate"] == "array"
    assert mobile["coordinate2"] == "array"
    assert mobile["time"] == "number"
    assert mobile["text"] == "string"


def test_canonical_schema_param_types_do_not_fall_back_to_name_heuristics():
    """Canonical tool schemas must still drive XML param coercion.

    ``status`` is outside the legacy fallback allowlists, so a non-empty wrapper
    map proves the parser is reading the nested schema instead of silently
    relying on parameter names.
    """
    [schema] = Qwen3_5DesktopUseAdapter()._assemble_tool_schemas()
    assert tool_schema_name(schema) == "computer_use"

    types = _param_types_from_tool_schemas([schema])

    assert types
    assert types["computer_use"]["pixels"] == "integer"
    assert types["computer_use"]["time"] == "number"
    assert types["computer_use"]["coordinate"] == "array"
    assert types["computer_use"]["status"] == "string"


def test_extra_tool_types_do_not_leak_across_tools():
    """A param name declared differently by two tools resolves per tool.

    ``computer_use`` has no ``index``; ``switch_tab.index`` is an integer.
    Coercion is keyed by (tool, param), not by param name alone.
    """
    schemas = LiteBrowserNavToolSet.get_tool_schemas(include=["switch_tab"])
    adapter = Qwen3_5DesktopUseAdapter(
        metadata=LiteCUAMetadata(
            dims=(LiteCUAMetadata.Platform.BROWSER, LiteCUAMetadata.TaskType.USE),
            extra_tool_schemas=schemas,
        ),
    )
    types = adapter._xml_param_types()
    assert types["switch_tab"]["index"] == "integer"
    assert "index" not in types["computer_use"]

    assert _args(adapter, _xml("switch_tab", index="3")) == {"index": 3}
    # Same key on the wrapper tool is undeclared → left as text.
    assert _args(adapter, _xml("computer_use", action="left_click", index="3")) == {
        "action": "left_click", "index": "3",
    }


def test_declared_types_come_from_the_rendered_tools_block():
    """The parser's type map and the prompt's ``<tools>`` block are the same
    objects — every tool rendered into the prompt is coercible."""
    schemas = _bgym_schemas("scroll", "fill", "tab_focus")
    adapter = _bid_mode_adapter(schemas)
    rendered = adapter._build_tools_section()
    types = adapter._xml_param_types()
    for schema in schemas:
        assert json.dumps(schema) in rendered
        schema_name = tool_schema_name(schema)
        for param, prop in tool_schema_parameters(schema)["properties"].items():
            assert types[schema_name][param] == prop["type"]
