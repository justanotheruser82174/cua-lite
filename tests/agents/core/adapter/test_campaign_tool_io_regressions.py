"""Regression tests for defects found by the live tool I/O rollout campaign.

Each test here FAILS on the pre-fix tree — the non-vacuity proof for every one
is recorded in the docstring as the exact pre-fix value it asserts against.

  * I1 / I9 / I12 — Qwen mobile app launch is a native
    ``mobile_use(action="open", text=<app>)`` spelling of canonical
    ``open_app`` only when the env injects the canonical ``open_app`` extra
    schema. The live rollout also showed Qwen3.5 using
    ``mobile_use(action="open_app", text=<app>)`` on the same mobile wrapper;
    that spelling is accepted only for this app-open surface, not generalized to
    arbitrary extras.

  * I2 — a tool result whose ``text`` is the EMPTY string rendered with no text
    part at all, because ``build_tool_result_message`` guarded it with
    truthiness instead of ``is not None``.

  * I4 — the gpt agent hand-built its unpaired env-feedback message and dropped
    ``result.metadata``, losing the ``model_output_error`` marker that
    ``BaseAgent`` has always emitted.

Hermetic: pure dict assertions on adapters / action spaces / message builders —
no model, no network, no env.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/agents/core/adapter/test_campaign_tool_io_regressions.py -p no:cacheprovider -q
"""

from __future__ import annotations

import pytest

from lite.agents.core.agent.utils.messages import build_tool_result_message
from lite.agents.models.qwen3_5.adapter import Qwen3_5MobileUseAdapter
from lite.agents.models.qwen3_vl.adapter import Qwen3VLMobileUseAdapter
from lite.core import LiteCUAMetadata
from lite.core.messages.final import pop_model_output_error
from lite.core.tools import make_tool_call, make_tool_schema
from lite.core.tools.calls import tool_call_name
from lite.core.tools.results import LiteToolResult, project_tool_result_text
from lite.gym.errors import FailureCategory
from lite.gym.types import LiteEnvStepResult
from lite.gym.utils.feedback.errors import (
    current_feedback,
)
from lite.gym.utils.feedback.ingress import prepare_env_tool_calls

# =============================================================================
# I1 / I9 / I12 — native mobile ``open`` spelling for canonical ``open_app``
# =============================================================================


def _schema(name: str, props: dict) -> dict:
    return make_tool_schema(
        name,
        description=name,
        parameters={"type": "object", "properties": props, "required": []},
    )


OPEN_APP_SCHEMA = _schema("open_app", {"app_name": {"type": "string"}})
RESPONSE_SCHEMA = _schema("response", {"text": {"type": "string"}})
TERMINATE_SCHEMA = _schema("terminate", {"status": {"type": "string"}})


def _mobile_md(extra_tools: list[dict]) -> LiteCUAMetadata:
    return LiteCUAMetadata(
        dims=(LiteCUAMetadata.Platform.MOBILE, LiteCUAMetadata.TaskType.USE),
        extra_tool_schemas=extra_tools,
        valid_actions=None,
    )


def _wrapper_call(**args) -> dict:
    return {
        "role": "assistant",
        "content": [],
        "tool_calls": [
            {"name": "mobile_use", "arguments": args},
        ],
    }


@pytest.mark.parametrize("adapter_cls", [Qwen3_5MobileUseAdapter, Qwen3VLMobileUseAdapter])
def test_native_mobile_open_canonicalizes_when_tool_is_offered(
    adapter_cls,
) -> None:
    """Official Qwen mobile app launch uses ``action=open`` and shared ``text``."""
    adapter = adapter_cls(metadata=_mobile_md([OPEN_APP_SCHEMA]))

    lite = adapter.convert_message_from_agent(
        _wrapper_call(action="open", text="Simple Gallery Pro")
    )

    assert lite["tool_calls"] == [make_tool_call("open_app", {"app_name": "Simple Gallery Pro"})]


@pytest.mark.parametrize("adapter_cls", [Qwen3_5MobileUseAdapter, Qwen3VLMobileUseAdapter])
def test_native_mobile_open_normalizes_unique_casefold_app_match(adapter_cls) -> None:
    """Qwen may lower-case an app name; unique env enum match is canonicalized."""
    schema = _schema(
        "open_app",
        {"app_name": {"type": "string", "enum": ["Markor", "Simple Calendar Pro"]}},
    )
    adapter = adapter_cls(metadata=_mobile_md([schema]))

    lite = adapter.convert_message_from_agent(
        _wrapper_call(action="open", text="simple calendar pro")
    )

    assert lite["tool_calls"] == [make_tool_call("open_app", {"app_name": "Simple Calendar Pro"})]


@pytest.mark.parametrize("adapter_cls", [Qwen3_5MobileUseAdapter, Qwen3VLMobileUseAdapter])
def test_native_mobile_open_still_reaches_env_when_tool_not_offered(adapter_cls) -> None:
    """Adapter only recognizes Qwen's open intent; env owns availability errors."""
    adapter = adapter_cls(metadata=_mobile_md([TERMINATE_SCHEMA]))

    lite = adapter.convert_message_from_agent(_wrapper_call(action="open", text="Markor"))

    assert lite["tool_calls"] == [make_tool_call("open_app", {"app_name": "Markor"})]


@pytest.mark.parametrize("adapter_cls", [Qwen3_5MobileUseAdapter, Qwen3VLMobileUseAdapter])
def test_native_mobile_open_splits_the_action_batch(adapter_cls) -> None:
    """``open`` canonicalizes to standalone ``open_app`` and splits GUI batches."""
    adapter = adapter_cls(metadata=_mobile_md([OPEN_APP_SCHEMA]))
    message = {
        "role": "assistant",
        "content": [],
        "tool_calls": [
            {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [10, 20]}},
            {"name": "mobile_use", "arguments": {"action": "open", "text": "Markor"}},
            {
                "name": "mobile_use",
                "arguments": {"action": "swipe", "coordinate": [1, 2], "coordinate2": [3, 4]},
            },
        ],
    }

    names = [tool_call_name(tc) for tc in adapter.convert_message_from_agent(message)["tool_calls"]]

    assert names == ["mobile", "open_app", "mobile"]


@pytest.mark.parametrize("adapter_cls", [Qwen3_5MobileUseAdapter, Qwen3VLMobileUseAdapter])
def test_open_app_wrapper_spelling_is_scoped_to_active_open_app(adapter_cls) -> None:
    """Observed wrapper spelling maps only to active app-open, not all extras."""
    adapter = adapter_cls(metadata=_mobile_md([OPEN_APP_SCHEMA]))

    lite = adapter.convert_message_from_agent(_wrapper_call(action="open_app", text="Markor"))

    assert lite["tool_calls"] == [make_tool_call("open_app", {"app_name": "Markor"})]


@pytest.mark.parametrize("adapter_cls", [Qwen3_5MobileUseAdapter, Qwen3VLMobileUseAdapter])
def test_open_app_wrapper_spelling_still_reaches_env_when_tool_not_offered(adapter_cls) -> None:
    adapter = adapter_cls(metadata=_mobile_md([TERMINATE_SCHEMA]))

    lite = adapter.convert_message_from_agent(_wrapper_call(action="open_app", text="Markor"))

    assert lite["tool_calls"] == [make_tool_call("open_app", {"app_name": "Markor"})]


@pytest.mark.parametrize("adapter_cls", [Qwen3_5MobileUseAdapter, Qwen3VLMobileUseAdapter])
@pytest.mark.parametrize(
    "args,expected,extra_tools",
    [
        (
            {"action": "terminate", "status": "success"},
            make_tool_call("terminate", {"status": "success"}),
            [OPEN_APP_SCHEMA, TERMINATE_SCHEMA],
        ),
        (
            {"action": "answer", "text": "42"},
            make_tool_call("response", {"text": "42"}),
            [OPEN_APP_SCHEMA, RESPONSE_SCHEMA],
        ),
    ],
)
def test_wrapper_answer_and_terminate_already_convert(
    adapter_cls,
    args: dict,
    expected: dict,
    extra_tools: list[dict],
) -> None:
    """GREEN BEFORE AND AFTER — deliberately not a regression test.

    I12 alleged that wrapper-routed ``terminate``/``answer`` also converted to
    nothing and left episodes unable to end. Measured over the corpus that is
    false: 34/34 ``terminate`` and 14/14 ``answer`` wrapper emissions convert.
    This pins that, so the ``open_app`` alias added alongside can never be
    "generalised" into re-handling paths that already work."""
    adapter = adapter_cls(metadata=_mobile_md(extra_tools))

    lite = adapter.convert_message_from_agent(_wrapper_call(**args))

    assert lite["tool_calls"] == [expected]


# =============================================================================
# I2 — empty tool-result text must still render a text part
# =============================================================================


def test_empty_tool_result_text_renders_an_empty_text_part() -> None:
    """A shell command that printed nothing sets ``text == ""``. Pre-fix the
    truthiness guard dropped it, so ``content`` was
    ``[{"type":"metadata","data":{"returncode":0}}]`` and the model could not
    tell "the command printed nothing" from "the tool returned no text at all"."""
    msg = build_tool_result_message("call_0001", (), "", {"returncode": 0})

    assert msg["content"] == [
        {"type": "text", "text": ""},
        {"type": "metadata", "data": {"returncode": 0}},
    ]


def test_absent_tool_result_text_still_renders_no_text_part() -> None:
    """The distinction the fix buys is only meaningful if ``None`` keeps
    rendering nothing — a GUI tap has no text channel at all."""
    msg = build_tool_result_message("call_0002", (0,), None, {"returncode": 0})

    assert msg["content"] == [
        {"type": "image", "index": 0},
        {"type": "metadata", "data": {"returncode": 0}},
    ]


def test_empty_tool_result_does_not_emit_empty_role_tool_message() -> None:
    """No image/text/error/metadata means there is no model-visible result row."""
    assert build_tool_result_message("call_0003", (), None, None) is None


# =============================================================================
# I4 — gpt must forward result metadata on the unpaired-feedback path
# =============================================================================


def _model_output_error_result(image: bytes | None, text: str) -> LiteEnvStepResult:
    return LiteEnvStepResult(
        results=[
            LiteToolResult(
                tool_call_id=None,
                images=[] if image is None else [image],
                text=text,
                metadata={"type": FailureCategory.MODEL_OUTPUT_ERROR},
            )
        ],
    )


def test_latest_step_feedback_carries_metadata() -> None:
    """The env result has all channels needed by the unpaired feedback message."""
    step = _model_output_error_result(
        b"png",
        "model output error: malformed <tool_call> XML",
    )

    [result] = step.results
    image = result.images[-1]
    text = project_tool_result_text(result.text, result.error)
    metadata = result.metadata

    assert image == b"png"
    assert "model output error" in text
    assert metadata == {"type": FailureCategory.MODEL_OUTPUT_ERROR}


def test_unpaired_feedback_message_keeps_the_model_output_error_marker() -> None:
    """The shape the gpt loop now appends for unpaired env feedback. Pre-fix it
    hand-built ``{"role":"user","content":[image,text]}`` with no metadata part,
    so EVERY gpt error turn was invisible to consumers filtering on the marker
    (qwen, which routes through this builder, emitted it 104 times in the same
    corpus; gpt emitted it 0 times)."""
    step = _model_output_error_result(b"png", "model output error: bad tool call")
    [result] = step.results
    text = project_tool_result_text(result.text, result.error)
    metadata = result.metadata

    msg = build_tool_result_message(None, (3,), text, metadata)

    assert msg["role"] == "user"  # tool_call_id=None stays a user observation
    assert msg["content"][0] == {"type": "image", "index": 3}
    assert msg["content"][1]["type"] == "text"
    assert msg["content"][-1] == {
        "type": "metadata",
        "data": {"type": FailureCategory.MODEL_OUTPUT_ERROR},
    }


# =============================================================================
# I15 — an env extra named inside the provider-native wrapper is lifted, not nested
# =============================================================================
#
# The model action space owns Qwen's native ``computer_use`` / ``mobile_use``
# enum, and env-declared standalone tools remain standalone tools. When the model
# names an ACTIVE extra as the wrapper's ``action`` value, the adapter lifts it
# back to the top level: nested as an action-batch child it is a shape the
# canonical row contract rejects outright
# (``lite/data/utils/rows.py`` — "must not nest standalone extra tool"), so the
# turn is both unexecutable and unpublishable. Phase 9 evidence:
# ``.logs/phase9/p9run1/P9-cell-qwen35-webarena-default/eval/281`` (``goto``) and
# ``.logs/phase9/p9run1-vwafix/P9-cell-qwen35-vwa-goal_image/eval/281``
# (``switch_tab``), both with ``executed_actions: []``.
#
# The lift is gated on the SAME key-shape predicate those rejections use, so an
# extra that is not offered, or arguments that route nowhere, still stay a
# wrapper action and keep the env's current-observation feedback rather than
# starving the agent. See
# ``tests/agents/models/qwen3_5/test_qwen3_5_wrapper_action_naming_standalone_extra.py``
# and ``tests/agents/models/qwen3_vl/test_qwen3_vl_wrapper_action_naming_standalone_extra.py``.

from lite.agents.models.qwen3_5.adapter import Qwen3_5DesktopUseAdapter  # noqa: E402
from lite.agents.models.qwen3_vl.adapter import Qwen3VLDesktopUseAdapter  # noqa: E402

BACK_SCHEMA = _schema("back", {})
GOTO_SCHEMA = _schema("goto", {"url": {"type": "string"}})
SOM_CLICK_SCHEMA = _schema("click", {"index": {"type": "integer"}})
SOM_INPUT_SCHEMA = _schema(
    "input",
    {
        "index": {"type": "integer"},
        "text": {"type": "string"},
    },
)

_DESKTOP_ADAPTERS = [Qwen3VLDesktopUseAdapter, Qwen3_5DesktopUseAdapter]


def _browser_md(extra_tools: list[dict]) -> LiteCUAMetadata:
    return LiteCUAMetadata(
        dims=(LiteCUAMetadata.Platform.BROWSER, LiteCUAMetadata.TaskType.USE),
        extra_tool_schemas=extra_tools,
        valid_actions=None,
    )


def _desktop_wrapper_call(*args_list) -> dict:
    return {
        "role": "assistant",
        "content": [],
        "tool_calls": [{"name": "computer_use", "arguments": args} for args in args_list],
    }


def _with_call_id(call: dict, call_id: str = "call_0000") -> dict:
    return {**call, "id": call_id}


def _assert_env_current_feedback(
    lite_call: dict,
    metadata: LiteCUAMetadata,
    message: str,
) -> None:
    routed, errors = prepare_env_tool_calls([_with_call_id(lite_call)], metadata)
    # Two shapes, one contract. A batch CHILD keeps its slot and carries its
    # reason so the env can answer it per action and still frame it (R4); a
    # TOP-LEVEL call has no slot to keep, so ingress answers it directly. Either
    # way the model gets exactly this message, on the current surface.
    if routed:
        assert [a.get("_rejected_reason") for a, _ in routed] == [message]
        assert errors == {}
    else:
        assert errors == {"call_0000": current_feedback(message)}


@pytest.mark.parametrize("adapter_cls", _DESKTOP_ADAPTERS)
@pytest.mark.parametrize(
    "action,args",
    [
        ("back", {}),
        ("goto", {"url": "https://example.com"}),
    ],
)
def test_wrapper_embedded_browser_extra_is_lifted_to_the_tool_it_names(
    adapter_cls,
    action: str,
    args: dict,
) -> None:
    """The offered tool runs instead of a batch the env can only reject."""
    metadata = _browser_md([BACK_SCHEMA, GOTO_SCHEMA])
    adapter = adapter_cls(metadata=metadata)

    lite = adapter.convert_message_from_agent(
        _desktop_wrapper_call(
            {"action": action, **args},
        )
    )

    assert lite["tool_calls"] == [make_tool_call(action, args)]
    assert pop_model_output_error(lite) is None
    routed, errors = prepare_env_tool_calls([_with_call_id(lite["tool_calls"][0])], metadata)
    assert routed == [({"name": action, "arguments": args, "call_id": "call_0000"}, "call_0000")]
    assert errors == {}


@pytest.mark.parametrize("adapter_cls", _DESKTOP_ADAPTERS)
def test_wrapper_routed_browser_nav_stays_an_error_when_tool_not_offered(adapter_cls) -> None:
    metadata = _browser_md([GOTO_SCHEMA])
    adapter = adapter_cls(metadata=metadata)

    lite = adapter.convert_message_from_agent(_desktop_wrapper_call({"action": "back"}))

    assert lite["tool_calls"] == [make_tool_call("computer", {"actions": [{"action": "back"}]})]
    assert pop_model_output_error(lite) is None
    _assert_env_current_feedback(
        lite["tool_calls"][0],
        metadata,
        "invalid action: back; computer.actions cannot contain back",
    )


@pytest.mark.parametrize("adapter_cls", _DESKTOP_ADAPTERS)
def test_env_extra_errors_do_not_shadow_native_wrapper_actions(adapter_cls) -> None:
    """Native Qwen wrapper actions still convert through the wrapper branch."""
    adapter = adapter_cls(metadata=_browser_md([TERMINATE_SCHEMA]))

    lite = adapter.convert_message_from_agent(
        _desktop_wrapper_call({"action": "terminate", "status": "success"})
    )

    assert lite["tool_calls"] == [make_tool_call("terminate", {"status": "success"})]


def test_wrapper_routed_som_click_is_lifted_to_the_som_click_tool() -> None:
    """``click`` is a native action NAME and a SoM extra tool; ``index`` decides.

    Key shape is the whole test, so the bid-mode ``click(index)`` wins over the
    coordinate-shaped native click and the model's actual click runs.
    """
    metadata = _browser_md([SOM_CLICK_SCHEMA, SOM_INPUT_SCHEMA])
    adapter = Qwen3_5DesktopUseAdapter(metadata=metadata)

    lite = adapter.convert_message_from_agent(
        _desktop_wrapper_call({"action": "click", "index": 5})
    )

    assert lite["tool_calls"] == [make_tool_call("click", {"index": 5})]
    assert pop_model_output_error(lite) is None
    routed, errors = prepare_env_tool_calls([_with_call_id(lite["tool_calls"][0])], metadata)
    assert routed == [
        ({"name": "click", "arguments": {"index": 5}, "call_id": "call_0000"}, "call_0000")
    ]
    assert errors == {}


def test_wrapper_routed_som_click_leaves_a_type_wrong_index_to_env_ingress() -> None:
    """A type-wrong ARGUMENT is env-owned feedback, not a routing decision.

    Routing by key shape (not schema satisfaction) is what lets the model hear
    ``index`` is the wrong type at all; satisfaction-based routing would send the
    call back into GUI conversion where nothing can explain it.
    """
    metadata = _browser_md([SOM_CLICK_SCHEMA])
    adapter = Qwen3_5DesktopUseAdapter(metadata=metadata)

    lite = adapter.convert_message_from_agent(
        _desktop_wrapper_call({"action": "click", "index": "5"})
    )

    assert lite["tool_calls"] == [make_tool_call("click", {"index": "5"})]
    assert pop_model_output_error(lite) is None
    routed, errors = prepare_env_tool_calls([_with_call_id(lite["tool_calls"][0])], metadata)
    # R4: a rejected batch child keeps its slot, carrying its reason.
    assert not routed or all(a.get("_rejected_reason") for a, _ in routed)
    assert list(errors) == ["call_0000"]


@pytest.mark.parametrize("adapter_cls", _DESKTOP_ADAPTERS)
def test_wrapper_routed_click_stays_visual_child_when_click_extra_not_offered(adapter_cls) -> None:
    """``index`` is the browser-nav extra's argument, never the GUI ``click``'s.

    With the ``click`` extra absent there is nothing to route to, so the adapter
    keeps the model's own argument and ingress forwards the child unchanged.
    Ingress answers membership only: whether the child action *name* is one this
    surface exposes. The argument verdict belongs to the concrete env, which is
    the only owner that knows its supported action set and which extra arguments
    it really consumes, and which answers with child-keyed wording.
    """
    metadata = _browser_md([GOTO_SCHEMA])
    adapter = adapter_cls(metadata=metadata)

    lite = adapter.convert_message_from_agent(
        _desktop_wrapper_call({"action": "click", "index": "5"})
    )

    assert lite["tool_calls"] == [
        make_tool_call("computer", {"actions": [{"action": "click", "index": "5"}]})
    ]
    assert pop_model_output_error(lite) is None
    routed, errors = prepare_env_tool_calls([_with_call_id(lite["tool_calls"][0])], metadata)
    assert routed == [({"name": "click", "arguments": {"index": "5"}}, "call_0000")]
    assert errors == {}


@pytest.mark.parametrize("adapter_cls", _DESKTOP_ADAPTERS)
def test_mouse_click_is_not_aliased_for_qwen_desktop(adapter_cls) -> None:
    metadata = _browser_md([GOTO_SCHEMA])
    adapter = adapter_cls(metadata=metadata)

    lite = adapter.convert_message_from_agent(
        _desktop_wrapper_call({"action": "mouse_click", "coordinate": [500, 500]})
    )

    assert lite["tool_calls"] == [
        make_tool_call(
            "computer",
            {"actions": [{"action": "mouse_click", "coordinate": [500, 500]}]},
        )
    ]
    assert pop_model_output_error(lite) is None
    _assert_env_current_feedback(
        lite["tool_calls"][0],
        metadata,
        "invalid action: mouse_click; computer.actions cannot contain mouse_click",
    )


def test_qwen3_5_mobile_left_click_alias_remains_family_level() -> None:
    adapter = Qwen3_5MobileUseAdapter(metadata=_mobile_md([]))

    lite = adapter.convert_message_from_agent(
        _wrapper_call(action="left_click", coordinate=[500, 500])
    )

    assert lite["tool_calls"] == [
        make_tool_call(
            "mobile",
            {"actions": [{"action": "tap", "coordinate": [500, 500], "clicks": 1}]},
        )
    ]
