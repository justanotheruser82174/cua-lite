"""Tests for Fara-owned tool prompt serialization."""

from __future__ import annotations

from lite.agents.models.fara.adapter import FaraDesktopUseAdapter
from lite.core import LiteCUAMetadata
from lite.core.tools.schemas import make_tool_schema


def test_fara_tool_json_preserves_non_ascii() -> None:
    schema = make_tool_schema("unicode_tool", description="Resume - check")
    schema["function"]["description"] = "Ré­sumé — ✓"
    adapter = FaraDesktopUseAdapter(metadata=LiteCUAMetadata(dims=("browser", "use")))
    adapter.action_space.get_tool_schemas = lambda: [schema]  # type: ignore[method-assign]

    fara_json = adapter._build_tools_section()
    assert "Ré­sumé — ✓" in fara_json
    assert "\\u2713" not in fara_json
