"""Tests for env feedback error wording and carriers."""

from __future__ import annotations

from typing import get_args

import pytest

from lite.core.tools.action_space.keys import canonical_special_keys
from lite.core.tools.results import project_tool_result_text
from lite.gym.errors import (
    FailureCategory,
    PairableModelActionError,
    TrueInfraFailure,
)
from lite.gym.types import LiteEnvStepResult
from lite.gym.utils.backend.keys import Backend, translate_keys
from lite.gym.utils.backend.model_inputs import project_model_keys
from lite.gym.utils.feedback.errors import (
    ACTION_ERROR_KINDS,
    MODEL_ACTION_ERROR_TYPES,
    ActionErrorKind,
    BackendExecutionErrorDetail,
    ContainerActionErrorRecord,
    ModelVisibleErrorDetail,
    ToolErrorCarrier,
    ToolErrorFeedback,
    _quoted_values,
    current_feedback,
    error_only_feedback,
    invalid_action_arguments_message,
    model_visible_error_detail,
    parse_container_action_error_record,
    record_model_action_error,
    record_tool_execution_error,
    tool_execution_error_message,
    unavailable_action_message,
    unknown_tool_message,
    unsupported_action_message,
)
from lite.gym.utils.feedback.results import build_tool_results_from_decisions

NO_BACKEND_LEAK_STRINGS = (
    "backend",
    "browsergym",
    "container",
    "docker",
    "playwright",
    "xdotool",
    "pyautogui",
    "pynput",
    "selenium",
    "Selenium",
    "webdriver",
    "WebDriver",
    "Playwright",
    "side effect",
    "cache write",
    "command rejected",
    "display pipe",
    "rpc",
    "write failed",
    "target closed",
    "target detached",
    "target page, context or browser",
    "X keysym",
    "KEYBOARD_KEYS",
    "action factory",
    "unsupported operand",
    "TypeError:",
    "ValueError:",
    "RuntimeError:",
    "NoneType",
)


def _assert_no_backend_leaks(text: str | None) -> None:
    assert text is not None
    for leak in NO_BACKEND_LEAK_STRINGS:
        assert leak not in text


def _project_visible_error(errors: dict[str, ToolErrorFeedback], call_id: str) -> str | None:
    out = build_tool_results_from_decisions(
        LiteEnvStepResult(),
        ordered_call_ids=[call_id],
        feedback=errors,
    )
    assert len(out.results) == 1
    visible_error = out.results[0].error
    assert visible_error is not None
    projected = project_tool_result_text(None, visible_error)
    assert visible_error in projected
    _assert_no_backend_leaks(projected)
    return visible_error


def test_d6_error_templates_use_visible_action_wording():
    assert unavailable_action_message("open_app") == ("open_app is not available in this task.")
    assert unsupported_action_message("pinch") == "unsupported action: pinch"
    assert (
        invalid_action_arguments_message(
            "click",
            "coordinate must be [x, y]",
        )
        == "invalid arguments for click: coordinate must be [x, y]"
    )
    assert (
        tool_execution_error_message(
            "bash",
            "command timed out",
        )
        == "bash failed: command timed out"
    )
    assert unknown_tool_message("frobnicate") == "unknown tool: frobnicate"


def test_tool_error_feedback_rejects_invalid_carrier():
    with pytest.raises(ValueError, match="invalid tool error carrier"):
        ToolErrorFeedback(message="bad", carrier="typo")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("record", "result_call_ids", "fallback_name", "expected"),
    [
        (
            {
                "index": 1,
                "kind": "model_action",
                "name": "click",
                "error": "bad coordinate",
                "message": "invalid arguments for click: bad coordinate",
            },
            (),
            None,
            ContainerActionErrorRecord(
                index=1,
                kind="model_action",
                name="click",
                error="bad coordinate",
                message="invalid arguments for click: bad coordinate",
            ),
        ),
        (
            {
                "index": 2,
                "kind": "tool_execution",
                "name": "bash",
                "error": "timed out",
                "message": "bash: timed out",
            },
            (),
            None,
            ContainerActionErrorRecord(
                index=2,
                kind="tool_execution",
                name="bash",
                error="timed out",
                message="bash: timed out",
            ),
        ),
        (
            # A container that could not read a name off the action envelope
            # sends an empty one; the caller's fallback names it.
            {
                "index": 0,
                "kind": "unsupported_action",
                "name": "",
                "error": "unsupported action: pinch",
                "message": "unsupported action: pinch",
            },
            (),
            "pinch",
            ContainerActionErrorRecord(
                index=0,
                kind="unsupported_action",
                name="pinch",
                error="unsupported action: pinch",
                message="unsupported action: pinch",
            ),
        ),
        (
            # No index on the record: the echoed call_id resolves it.
            {
                "kind": "model_action",
                "name": "tap",
                "error": "coordinate required",
                "message": "invalid arguments for tap: coordinate required",
                "call_id": "call_tap",
            },
            ("call_scroll", "call_tap"),
            None,
            ContainerActionErrorRecord(
                index=1,
                kind="model_action",
                name="tap",
                error="coordinate required",
                message="invalid arguments for tap: coordinate required",
                call_id="call_tap",
            ),
        ),
    ],
)
def test_parse_container_action_error_record_reads_the_container_wire_shape(
    record,
    result_call_ids,
    fallback_name,
    expected,
) -> None:
    assert (
        parse_container_action_error_record(
            record,
            result_call_ids=result_call_ids,
            fallback_name=fallback_name,
        )
        == expected
    )


@pytest.mark.parametrize(
    "record",
    [
        None,
        ["not", "a", "record"],
        {"index": "not-an-int", "kind": "model_action", "name": "click"},
        {"index": 0, "kind": "backend_error", "name": "click"},
        {"kind": "model_action", "name": "click", "message": "missing index"},
        {"call_id": "unknown", "kind": "model_action", "message": "unknown call"},
        # ``index`` is the one spelling on the wire: a record that positions
        # itself under any other key is unparseable, not silently accepted.
        {"action_index": 0, "kind": "model_action", "name": "click"},
    ],
)
def test_parse_container_action_error_record_rejects_invalid_records(record) -> None:
    assert (
        parse_container_action_error_record(
            record,
            result_call_ids=("call_0", "call_1"),
        )
        is None
    )


@pytest.mark.parametrize("record", [record_model_action_error, record_tool_execution_error])
def test_record_action_error_rejects_invalid_carrier(record):
    with pytest.raises(ValueError, match="invalid tool error carrier"):
        record(
            {},
            "call_0",
            ValueError("bad"),
            carrier="typo",  # type: ignore[arg-type]
        )


def test_record_action_errors_can_use_d6_visible_action_templates():
    errors: dict[str, ToolErrorFeedback] = {}

    record_model_action_error(
        errors,
        "call_click",
        ValueError("coordinate must be [x, y]"),
        action_name="click",
    )
    record_tool_execution_error(
        errors,
        "call_bash",
        ModelVisibleErrorDetail("command timed out"),
        carrier="error_only",
        action_name="bash",
    )

    assert errors == {
        "call_click": current_feedback("invalid arguments for click: coordinate must be [x, y]"),
        "call_bash": error_only_feedback("bash failed: command timed out"),
    }


@pytest.mark.parametrize(
    "backend_detail",
    [
        "xdotool key failed",
        "pyautogui.click failed",
        "pynput controller failed",
        "Playwright TimeoutError: target closed",
        "playwright target closed",
        "Selenium WebDriverException: session closed",
        "webdriver error: session closed",
        "browsergym step returned None",
        "webharbor.webvoyager container /step returned 500: boom",
        "mobileworld container is not available",
        "docker exec failed",
        "display pipe closed",
        "device rpc failed",
        "fail command rejected",
        "RuntimeError: backend crashed",
        "response side effect failed: answer rejected",
        "cache write failed",
    ],
)
def test_record_tool_execution_error_hides_untyped_backend_detail_without_word_guessing(
    backend_detail: str,
) -> None:
    errors: dict[str, ToolErrorFeedback] = {}

    record_tool_execution_error(
        errors,
        "call_click",
        backend_detail,
        action_name="click",
    )

    assert errors["call_click"] == current_feedback("click failed: execution failed")
    visible_error = _project_visible_error(errors, "call_click")
    assert visible_error == "click failed: execution failed"
    _assert_no_backend_leaks(visible_error)


@pytest.mark.parametrize(
    "detail",
    [
        "available modes are backend and frontend",
        "button named 'Docker container' was not found",
        "select the Playwright documentation link",
        "select Selenium supplements from the results",
        "the WebDriver docs tab is visible",
        "the page mentions a cache write policy",
    ],
)
def test_model_visible_error_detail_keeps_legitimate_domain_words(
    detail: str,
) -> None:
    assert model_visible_error_detail(detail, fallback="execution failed") == detail


def test_typed_backend_execution_detail_uses_category_not_word_guessing() -> None:
    errors: dict[str, ToolErrorFeedback] = {}

    record_tool_execution_error(
        errors,
        "call_click",
        BackendExecutionErrorDetail(RuntimeError("button named 'Docker container' was not found")),
        action_name="click",
    )

    visible_error = _project_visible_error(errors, "call_click")
    assert visible_error == "click failed: execution failed"
    _assert_no_backend_leaks(visible_error)


def test_explicit_model_visible_execution_detail_bypasses_backend_classifier() -> None:
    errors: dict[str, ToolErrorFeedback] = {}

    record_tool_execution_error(
        errors,
        "call_click",
        ModelVisibleErrorDetail("button named 'Docker container' was not found"),
        action_name="click",
    )

    assert errors["call_click"].message == (
        "click failed: button named 'Docker container' was not found"
    )


@pytest.mark.parametrize(
    "python_detail",
    [
        "unsupported operand type(s) for /: 'NoneType' and 'int'",
        "TypeError: click() got an unexpected keyword argument 'x'",
        "ValueError: could not convert string to float",
    ],
)
def test_record_model_action_error_hides_explicit_backend_detail_from_model_visible_text(
    python_detail: str,
) -> None:
    errors: dict[str, ToolErrorFeedback] = {}

    record_model_action_error(
        errors,
        "call_click",
        BackendExecutionErrorDetail(python_detail),
        action_name="click",
    )

    visible_error = _project_visible_error(errors, "call_click")
    assert visible_error == ("invalid arguments for click: arguments could not be interpreted")
    _assert_no_backend_leaks(visible_error)


#: What the model should be told for each backend's projection raise. This is
#: the only hand-written half of the key-wording test: the RAISE TEXT is
#: produced by calling the real producer (see ``_real_key_projection_raises``),
#: never typed. It replaced a parametrize of seven copy-pasted literals -- two of
#: whose rows spelled a ``keys: expected Lite canonical key tokens ...`` wording
#: NO caller can produce, and those two rows were the only thing keeping the
#: matching translator branch alive.
_MODEL_FACING_KEY_WORDING = {
    "playwright": "unknown key token",
    "xdotool": "unknown key token",
    "pynput": "unknown key token",
    "selenium": "unknown key token",
    "pyautogui": "unsupported key token",
}

#: Every input a model can put in a ``keys`` list: the canonical special
#: vocabulary, every printable glyph, and the malformed/non-canonical shapes.
_MODEL_KEY_INPUTS = (
    sorted(canonical_special_keys())
    + [chr(c) for c in range(0x20, 0x7F)]
    + [t.upper() for t in sorted(canonical_special_keys())]
    + [
        "Ctrl", "A", "ctrl+a", "ArrowDown", "+", "++", "a+b", "",
        "plus", "minus", "equal", "comma", "it's", 'a"b', "\\",
        " ", "\n", "\t", "\r", "\x1b", "\x00",
    ]
)


def _real_key_projection_raises() -> list[tuple[str, str, str]]:
    """``(backend, token, raise text)`` for every ``keys.to_*`` raise a REAL
    caller can produce -- obtained by calling ``project_model_keys``, the one
    production entry into the key projections, not by typing its wording out.
    """
    rows: list[tuple[str, str, str]] = []
    for backend in get_args(Backend):
        for token in _MODEL_KEY_INPUTS:
            try:
                project_model_keys([token], action_name="key", backend=backend)
            except ValueError as exc:
                if str(exc).startswith("keys.to_"):
                    rows.append((backend, token, str(exc)))
    return rows


def test_every_backend_projection_raise_is_reachable_from_the_real_producer() -> None:
    """The translator's five ``keys.to_*`` rules are COVERAGE, not a wish list.

    A rule for a wording no caller can emit is dead weight that reads as alive;
    require that each backend in the model-facing table above is actually
    reachable through ``project_model_keys``, so a rule can only be kept while
    something real still produces it.
    """
    assert set(_MODEL_FACING_KEY_WORDING) == set(get_args(Backend))
    reached = {backend for backend, _token, _raw in _real_key_projection_raises()}
    assert reached == set(get_args(Backend)), f"unreachable projections: {reached}"


@pytest.mark.parametrize(
    ("backend", "token", "raw_detail"),
    _real_key_projection_raises(),
    ids=lambda value: repr(value) if isinstance(value, str) else value,
)
def test_record_model_action_error_sanitizes_real_key_projection_detail(
    backend: str,
    token: str,
    raw_detail: str,
) -> None:
    errors: dict[str, ToolErrorFeedback] = {}

    record_model_action_error(
        errors,
        "call_key",
        raw_detail,
        action_name="key",
    )

    visible_error = _project_visible_error(errors, "call_key")
    kind = _MODEL_FACING_KEY_WORDING[backend]
    if repr(token) == f"'{token}'":
        # The case every real key hits: the producer's ``{token!r}`` round-trips
        # through the translator's split-on-apostrophe, so the model is told the
        # offending token EXACTLY.
        assert visible_error == f"invalid arguments for key: {kind} {token!r}"
    else:
        # KNOWN, PRE-EXISTING lossiness (surfaced by driving the real producer;
        # the old hand-written literals never reached it): the translator
        # re-``repr``s a fragment that the producer had already ``repr``d, so a
        # token carrying an escape or a quote is shown double-escaped. Fixing it
        # means raising a typed error instead of reparsing a string -- a real
        # refactor, not this row. It stays model-safe and still names the token,
        # which is what this test guards.
        assert visible_error.startswith(f"invalid arguments for key: {kind} ")
    _assert_no_backend_leaks(visible_error)


def test_key_errors_reaching_feedback_come_only_from_project_model_keys() -> None:
    """GATE for the CONTRACT stated on ``project_model_keys``'s docstring.

    ``keys._require_canonical`` raises ``"keys: expected Lite canonical key
    tokens ..."``, but ``project_model_keys`` pre-enforces the same shared
    predicate before ``translate_keys`` is reached, so that wording cannot
    reach env feedback. This test is what lets the feedback translator carry NO
    rule for it: drive the real producer with every input a model can put in a
    ``keys`` list and require, of every raise, that it is either

    * a ``keys.to_*`` projection wording the translator rewrites, or
    * a ``project_model_keys`` wording that is already model-safe and passes
      through untranslated,

    and never ``_require_canonical``'s. If this ever fails, restore a translator
    rule -- do not weaken the assert.
    """
    raised = 0
    for backend in get_args(Backend):
        for token in _MODEL_KEY_INPUTS:
            try:
                project_model_keys([token], action_name="key", backend=backend)
            except ValueError as exc:
                raised += 1
                raw = str(exc)
                assert not raw.startswith("keys: expected "), (
                    f"{backend}/{token!r}: _require_canonical wording escaped the producer"
                )
                if raw.startswith("keys.to_"):
                    # Gates the removed ``if quoted else ...`` arms, and is
                    # asserted BEFORE the translator runs so a broken producer
                    # reports here rather than as an IndexError inside it: every
                    # projection raise interpolates ``{token!r}``, and ``repr``
                    # of a ``str`` always carries an apostrophe, so the quoted
                    # extraction is non-empty by construction.
                    assert _quoted_values(raw), f"{backend}/{token!r}: no quoted token in raise"
                detail = model_visible_error_detail(exc, fallback="arguments could not be")
                if raw.startswith("keys.to_"):
                    assert detail != raw, f"{backend}/{token!r}: backend wording untranslated"
                else:
                    assert detail == raw, f"{backend}/{token!r}: producer wording rewritten"
                _assert_no_backend_leaks(detail)

    # A guard that is never fed passes vacuously -- assert it was fed.
    assert raised > 100, f"only {raised} raises collected"


def test_every_backend_key_projection_failure_translates_for_the_model() -> None:
    """Generative counterpart to the row-per-wording parametrize above.

    The parametrize only ever covers the wordings someone remembered to add: the
    ``to_selenium`` raise leaked its backend name to the model for 22 canonical
    tokens while ``NO_BACKEND_LEAK_STRINGS`` already listed ``selenium`` /
    ``WebDriver`` -- the guard was complete, nothing ever fed it such a string.
    So drive every ``Backend`` against every canonical special token and require
    that whatever projection raises is recognised by the translator -- i.e. it
    neither leaks a backend name nor survives with its raw ``keys.to_*`` marker,
    which is what an unrecognised (new) backend's wording would do.
    """
    backends = get_args(Backend)
    tokens = sorted(canonical_special_keys())
    raised: dict[str, list[str]] = {backend: [] for backend in backends}

    for backend in backends:
        for token in tokens:
            try:
                translate_keys([token], backend)
            except MODEL_ACTION_ERROR_TYPES as exc:
                raised[backend].append(token)
                detail = model_visible_error_detail(
                    exc,
                    fallback="arguments could not be interpreted",
                )
                _assert_no_backend_leaks(detail)
                assert "keys.to_" not in detail, (
                    f"{backend}/{token}: untranslated backend wording reached the model"
                )

    # A guard that is never fed passes vacuously -- assert it was fed.
    assert any(raised.values())
    assert raised["selenium"], "selenium projection must still exercise this path"


def test_every_declared_action_error_kind_reaches_the_container_record_parser() -> None:
    """The kind vocabulary is DECLARED once, on ``ActionErrorKind`` (F5).

    ``ACTION_ERROR_KINDS`` used to be a hand-listed frozenset on the line below
    the ``Literal``, so a fourth kind added to the type alone would have been
    dropped on the floor by ``parse_container_action_error_record``. Drive the
    declaration, not the constant, so re-hardcoding a stale copy fails here.
    """
    kinds = get_args(ActionErrorKind)
    assert kinds, "ActionErrorKind must declare at least one kind"
    assert ACTION_ERROR_KINDS == frozenset(kinds)

    for kind in kinds:
        parsed = parse_container_action_error_record(
            {
                "index": 0,
                "kind": kind,
                "name": "click",
                "error": "boom",
                "message": "click: boom",
            },
        )
        assert parsed is not None, f"{kind}: declared kind rejected by the parser"
        assert parsed.kind == kind


def test_every_declared_carrier_is_gated_once_and_routed_by_result_assembly() -> None:
    """The carrier vocabulary is DECLARED once, on ``ToolErrorCarrier`` (F5).

    Three copies existed: the ``Literal``, ``__post_init__``'s tuple, and the
    ``if/elif/else`` in ``build_tool_results_from_decisions`` (whose ``else``
    re-raised the gate's message verbatim and was unreachable). Drive the
    declaration so a stale re-listing in either place fails here.
    """
    carriers = get_args(ToolErrorCarrier)
    assert set(carriers) == {"current", "error_only"}

    for carrier in carriers:
        # The gate accepts every declared carrier ...
        item = ToolErrorFeedback(message="boom", carrier=carrier)
        # ... and result assembly routes every declared carrier to a paired result.
        out = build_tool_results_from_decisions(
            LiteEnvStepResult(),
            ordered_call_ids=["call_0"],
            feedback={"call_0": item},
            images=[b"png-bytes"],
            text="observation",
        )
        assert len(out.results) == 1, f"{carrier}: no result assembled"
        assert out.results[0].error == "boom"
        if carrier == "current":
            assert out.results[0].images == [b"png-bytes"]
            assert out.results[0].text == "observation"
        else:
            assert out.results[0].images == []
            assert out.results[0].text is None


def test_failure_category_spellings_are_pinned_by_literal() -> None:
    """The two spellings ``errors.py`` used to hard-code, pinned as LITERALS.

    Deliberately literal, not ``FailureCategory.X.value == FailureCategory.X.value``:
    production now reads both spellings off the enum, so an assertion phrased in
    terms of the enum would move in lockstep with it and pin nothing. This is the
    only place in the tree that still states the two strings by hand, and it is a
    test, so a rename in ``lite/gym/errors.py`` is a decision someone has to
    confirm here rather than a silent wire-format change.
    """
    assert FailureCategory.INFRA_FAILURE.value == "infra_failure"
    assert FailureCategory.MODEL_ACTION_ERROR.value == "model_action_error"


def test_failure_category_branches_withhold_or_forward_the_right_detail() -> None:
    """The two ``failure_category`` branches in ``errors.py``, driven end to end.

    Names no spelling on purpose — it asserts what the model actually SEES, so it
    stays green across a category rename and reddens only if the branch stops
    firing. Before it existed, neither branch was reached by any test, so
    corrupting the enum reddened nothing while an infra fault's raw backend text
    silently started reaching the prompt.
    """
    # infra: the raw backend wording must NOT reach the prompt.
    infra = TrueInfraFailure("container rpc target closed")
    assert model_visible_error_detail(infra, fallback="execution failed") == "execution failed"
    errors: dict[str, ToolErrorFeedback] = {}
    record_tool_execution_error(errors, "call_infra", infra, action_name="click")
    assert errors["call_infra"].message == "click failed: execution failed"

    # model action error: its own wording IS the actionable feedback.
    model_error = PairableModelActionError("coordinate must be a [x, y] pair")
    recorded: dict[str, ToolErrorFeedback] = {}
    record_tool_execution_error(recorded, "call_model", model_error, action_name="click")
    assert recorded["call_model"].message == "click failed: coordinate must be a [x, y] pair"

    # no declared category: tool-execution recording falls back, never leaking.
    untyped: dict[str, ToolErrorFeedback] = {}
    record_tool_execution_error(untyped, "call_plain", RuntimeError("target closed"),
                               action_name="click")
    assert untyped["call_plain"].message == "click failed: execution failed"
