from __future__ import annotations

from lite.core.tools.results import (
    LiteToolResult,
    extract_projected_tool_result_error,
    project_tool_result_text,
    text_has_projected_tool_result_error,
)


def test_project_tool_result_text_exact_format() -> None:
    assert project_tool_result_text("obs\n", "bad") == (
        "obs\n\n\n## Error from previous action:\nbad"
    )
    assert project_tool_result_text("", "bad") == "\n\n## Error from previous action:\nbad"
    assert project_tool_result_text(None, "bad") == "## Error from previous action:\nbad"
    assert project_tool_result_text("", None) == ""


def test_tool_results_prefers_per_call_text() -> None:
    step_result = type(
        "StepResult",
        (),
        {"results": [LiteToolResult(tool_call_id="call_0000", text="per-call stdout")]},
    )()
    [result] = [
        result for result in step_result.results if result.tool_call_id == "call_0000"
    ]

    assert project_tool_result_text(result.text, result.error) == "per-call stdout"


def test_projected_tool_result_error_read_helpers() -> None:
    projected = (
        "## AXTree:\nbody\n\n"
        "## Error from previous action:\nunsupported action: bogus\n"
        "## HTML:\n<button>Search</button>"
    )

    assert text_has_projected_tool_result_error(projected) is True
    assert (
        extract_projected_tool_result_error(projected)
        == "unsupported action: bogus\n## HTML:\n<button>Search</button>"
    )
    assert extract_projected_tool_result_error(
        "## Error from previous action:\n## HTML:\n<button>Search</button>"
    ) == "## HTML:\n<button>Search</button>"
    single_newline = (
        "## AXTree:\nbody\n"
        "## Error from previous action:\nordinary page heading"
    )
    assert text_has_projected_tool_result_error(single_newline) is False
    assert extract_projected_tool_result_error(single_newline) is None
    assert text_has_projected_tool_result_error(
        "AXTree text mentions ## Error from previous action:\nnot a header line"
    ) is False
    assert extract_projected_tool_result_error(
        "AXTree text mentions ## Error from previous action:\nnot a header line"
    ) is None
    assert text_has_projected_tool_result_error(
        "## Error from previous action: inline detail"
    ) is False
    assert extract_projected_tool_result_error(
        "## Error from previous action: inline detail"
    ) is None
    assert extract_projected_tool_result_error("## AXTree:\nbody") is None


def test_projected_error_at_offset_zero_is_a_reachable_shape() -> None:
    """Pins the header-at-offset-0 branch of the inverse."""
    for error in ("element not visible", "multi\nline detail", ""):
        projected = project_tool_result_text(None, error)
        assert projected.startswith("## Error from previous action:\n")
        assert extract_projected_tool_result_error(projected) == error

    assert project_tool_result_text(None, "") == "## Error from previous action:\n"
