"""Oracle tests for the Qwen3-VL ``<tools>`` envelope.

Every Qwen family was trained on the nested function-tool envelope --
``{"type": "function", "function": {"name", "description", "parameters"}}`` --
so the Qwen adapters must serialize canonical Lite schemas in that shape.

These assertions deliberately do not depend on goldens. The goldens are
regenerated whenever rendering changes, so they cannot justify a render change.
The oracle is ``transformers.utils.get_json_schema`` and, when mounted, the
vendored Qwen reference implementations under ``$CUA_LITE_REFERENCES_ROOT``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

import lite.agents.models.qwen3_vl.adapter  # noqa: F401
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.core import LiteCUAMetadata
from lite.core.errors import LiteContractError
from lite.core.tools.extra_tools import LiteFinishToolSet
from lite.core.tools.schemas import make_tool_schema

QWEN3_VL_TOOLS_ADAPTER_KEYS = [
    "qwen3_vl@desktop@use",
    "qwen3_vl@mobile@use",
]


def _tools_section(adapter_key: str, *, with_extras: bool) -> str:
    platform = "mobile" if "@mobile" in adapter_key else "desktop"
    extras = (
        LiteFinishToolSet.get_tool_schemas(include=["response", "terminate"])
        if with_extras
        else []
    )
    metadata = LiteCUAMetadata(dims=(platform, "use"), extra_tool_schemas=extras)
    adapter = AgentAdapterRegistry.get(adapter_key, metadata=metadata)
    return adapter._build_tools_section()


def _tool_objects(section: str) -> list[dict]:
    """Every JSON object on its own line inside the ``<tools>`` block."""
    body = re.search(r"<tools>\n(.*?)\n</tools>", section, re.DOTALL)
    assert body is not None, f"no <tools> block in section:\n{section[:400]}"
    lines = [ln for ln in body.group(1).splitlines() if ln.strip()]
    assert lines, "empty <tools> block"
    return [json.loads(ln) for ln in lines]


def _oracle_envelope_keys() -> set[str]:
    """Top-level keys transformers puts on a tool schema, from a callable."""
    from transformers.utils import get_json_schema

    def computer_use(action: str) -> None:
        """Use a mouse and keyboard.

        Args:
            action: the action to perform
        """

    return set(get_json_schema(computer_use).keys())


# This shared Qwen-family oracle has no single family owner. Keep it in Qwen3-VL
# because that was the active mixed source before the family split.
def test_transformers_oracle_is_nested() -> None:
    """Guard the oracle itself before comparing Qwen rendered tools to it."""
    assert _oracle_envelope_keys() == {"type", "function"}


@pytest.mark.parametrize("adapter_key", QWEN3_VL_TOOLS_ADAPTER_KEYS)
@pytest.mark.parametrize("with_extras", [False, True], ids=["bare", "extras"])
def test_tools_envelope_matches_transformers_oracle(
    adapter_key: str, with_extras: bool
) -> None:
    """Every advertised tool uses transformers' nested envelope."""
    expected_keys = _oracle_envelope_keys()
    for tool in _tool_objects(_tools_section(adapter_key, with_extras=with_extras)):
        assert set(tool.keys()) == expected_keys, (
            f"{adapter_key}: tool envelope {sorted(tool)} != oracle "
            f"{sorted(expected_keys)} -- the tool schema was serialized wrongly"
        )
        assert tool["type"] == "function"
        inner = tool["function"]
        assert isinstance(inner, dict)
        assert "name" in inner and isinstance(inner["name"], str)
        assert "parameters" in inner
        assert "type" not in inner or inner["type"] != "function"


@pytest.mark.parametrize("adapter_key", QWEN3_VL_TOOLS_ADAPTER_KEYS)
def test_tools_block_has_no_flat_schema_marker(adapter_key: str) -> None:
    """The flat spelling must not appear anywhere in the rendered block."""
    section = _tools_section(adapter_key, with_extras=True)
    assert '{"type": "function", "name"' not in section, (
        f"{adapter_key}: rendered <tools> carries a legacy flat envelope"
    )
    assert '{"type": "function", "function":' in section


@pytest.mark.parametrize("adapter_key", QWEN3_VL_TOOLS_ADAPTER_KEYS)
def test_tools_block_uses_default_json_ascii_policy(adapter_key: str) -> None:
    """Qwen rendered tool schemas are one default-json object per line."""
    schema = make_tool_schema("unicode_tool", description="Ré­sumé — ✓")
    platform = "mobile" if "@mobile" in adapter_key else "desktop"
    adapter = AgentAdapterRegistry.get(
        adapter_key,
        metadata=LiteCUAMetadata(dims=(platform, "use")),
    )
    adapter.action_space.get_tool_schemas = lambda: [schema]  # type: ignore[method-assign]

    body = re.search(
        r"<tools>\n(.*?)\n</tools>",
        adapter._build_tools_section(),
        re.DOTALL,
    )
    assert body is not None
    [line] = body.group(1).splitlines()

    assert line == json.dumps(schema)
    assert "\\u" in line
    assert json.loads(line) == schema


def test_qwen_tools_section_rejects_flat_tool_schema() -> None:
    """Qwen adapters serialize canonical Lite tool schemas only."""
    flat_schema = {
        "type": "function",
        "name": "flat_tool",
        "description": "",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }

    adapter = AgentAdapterRegistry.get(
        "qwen3_vl@desktop@use",
        metadata=LiteCUAMetadata(dims=("desktop", "use")),
    )
    adapter.action_space.get_tool_schemas = lambda: [flat_schema]  # type: ignore[method-assign]
    with pytest.raises(LiteContractError, match="noncanonical outer keys"):
        adapter._build_tools_section()


_REFS = os.environ.get("CUA_LITE_REFERENCES_ROOT")
_NESTED_PREFIX_RE = re.compile(
    r'\{\s*\\*"type\\*":\s*\\*"function\\*",\s*\\*"function\\*"\s*:'
)


def _qwen3_vl_reference_files() -> list[Path]:
    if not _REFS:
        return []
    root = Path(_REFS)
    candidates = [
        root / "Qwen3-VL" / "cookbooks" / "computer_use.ipynb",
    ]
    return [p for p in candidates if p.is_file()]


if _qwen3_vl_reference_files():

    @pytest.mark.parametrize(
        "reference", _qwen3_vl_reference_files(), ids=lambda p: p.name
    )
    def test_reference_implementations_use_nested_envelope(reference: Path) -> None:
        """The upstream Qwen3-VL reference builds the nested envelope."""
        text = reference.read_text(encoding="utf-8", errors="replace")
        assert _NESTED_PREFIX_RE.search(text), (
            f"{reference} no longer shows the nested envelope; the premise of "
            "Qwen tool-schema rendering must be re-derived before trusting it"
        )
