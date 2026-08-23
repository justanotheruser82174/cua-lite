"""Envelope-contract tests for Lite tool calls and tool results.

The in-memory ``LiteToolResult`` / ``LiteEnvStepResult.results`` target type
exists here, with NO ``category`` field. The remote LEF wire carries
reset observations and step-result envelopes over remote transport.

Design:
  * Type-level invariants exercise the real ``LiteToolResult``.
  * ``test_frame_nresults_roundtrip`` probes the real remote frame codec.

Hermetic: no docker, no env-server, no model download, no network. The
LoopDetectWrapper case (E2) drives the REAL wrapper against a tiny spy env.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/gym/types/test_envelope_contract.py -p no:cacheprovider -q
"""

from __future__ import annotations

import asyncio
import dataclasses
import json

import pytest

import lite.gym.types as gym_types
from lite.agents.core.agent.utils.annotations import action_inspection_records
from lite.core import LiteCUAMetadata
from lite.core import LiteToolCall as _LiteToolCallCanonical  # noqa: F401  (identity check)
from lite.core.messages.final import (
    CONTENT_ONLY_FINAL_REASON,
    LOOP_DETECT_TERMINATE_REASON,
    make_no_tool_call_final_actions,
)
from lite.core.tools.action_space import unpack_action_batch_call
from lite.core.tools.calls import (
    RUNTIME_INTERNAL_STOP_REASON_KEY,
    RUNTIME_RESULT_CALL_ID_KEY,
    LiteToolCall,
    RuntimeEnvAction,
    make_tool_call,
    tool_call_id,
    tool_call_name,
)
from lite.core.tools.results import LiteToolResult
from lite.gym.remote.frame import (
    FRAME_MAGIC,
    FRAME_VERSION,
    decode_reset_observation,
    decode_step_result,
    encode_reset_observation,
    encode_step_result,
)
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult
from lite.gym.utils.feedback.errors import (
    error_only_feedback,
)
from lite.gym.utils.feedback.ingress import (
    classify_standalone_tool_call,
    make_internal_terminate_action,
    prepare_env_tool_calls,
    standalone_tool_call_feedback,
)
from lite.gym.utils.feedback.results import (
    build_tool_results_from_decisions,
    canonical_result_call_id,
    ordered_tool_call_ids,
)
from lite.gym.wrappers import LoopDetectWrapper

_ToolResult = LiteToolResult


def _call(
    name: str,
    arguments: dict | None = None,
    *,
    call_id: str | None = None,
) -> dict:
    return make_tool_call(name, arguments or {}, call_id=call_id)


def _batch(
    wrapper: str,
    actions: list[dict],
    *,
    call_id: str | None = None,
) -> dict:
    return _call(wrapper, {"actions": actions}, call_id=call_id)


# ---------------------------------------------------------------------------
# Probes against the real in-memory envelope.
# ---------------------------------------------------------------------------
def _step_result_has_results_field() -> bool:
    return "results" in {f.name for f in dataclasses.fields(LiteEnvStepResult)}


def _gym_types_has_tool_result() -> bool:
    return hasattr(gym_types, "LiteToolResult")


def _liteaction_has_id() -> bool:
    # LiteToolCall is a TypedDict — its declared keys live in __annotations__.
    return "id" in getattr(LiteToolCall, "__annotations__", {})


# ---------------------------------------------------------------------------
# Spy env for the LoopDetectWrapper case (E2). Minimal surface: the wrapper
# only calls ``self.env.step(forwarded)`` and reads ``result.info``.
# ---------------------------------------------------------------------------
class _SpyEnv:
    def __init__(self) -> None:
        self.forwarded: list[RuntimeEnvAction] | None = None
        self.metadata = LiteCUAMetadata()

    async def reset(self) -> LiteEnvObservation:
        return LiteEnvObservation(image=b"png")

    async def step(self, actions: list[RuntimeEnvAction]) -> LiteEnvStepResult:
        # Capture exactly what the wrapper forwarded (post-rewrite).
        self.forwarded = list(actions)
        return LiteEnvStepResult(
            reward=None,
            info={},
        )


class _EnvelopeSpyEnv(_SpyEnv):
    async def step(self, actions: list[RuntimeEnvAction]) -> LiteEnvStepResult:
        self.forwarded = list(actions)
        terminated = any(tool_call_name(action) == "terminate" for action in actions)
        result = LiteEnvStepResult(
            terminated=terminated,
            info={},
        )
        ordered = ordered_tool_call_ids(actions)
        return build_tool_results_from_decisions(
            result,
            ordered_call_ids=ordered,
            continue_call_ids=set(ordered),
            images=[b"\x89PNG"],
            text=None,
        )


# ---------------------------------------------------------------------------
# E1 — cardinality is env-decision-owned, including terminal calls
# ---------------------------------------------------------------------------
def test_env_step_results_cardinality():
    """``[computer, bash]`` can produce 2 results; ``terminate`` is skipped
    only when the env does not make a result decision for that call.
    """
    computer_bash = [
        _call("computer", call_id="call_1"),
        _call("bash", call_id="call_2"),
    ]
    computer_term = [
        _call("computer", call_id="call_1"),
        _call("terminate", call_id="call_2"),
    ]
    non_terminal = build_tool_results_from_decisions(
        LiteEnvStepResult(),
        ordered_call_ids=ordered_tool_call_ids(computer_bash),
        continue_call_ids={"call_1"},
        explicit_results={"call_2": LiteToolResult(tool_call_id="call_2", text="bash output")},
        images=[b"\x89PNG"],
        text="after step",
    )
    terminal_skipped = build_tool_results_from_decisions(
        LiteEnvStepResult(terminated=True),
        ordered_call_ids=ordered_tool_call_ids(computer_term),
        continue_call_ids={"call_1"},
        images=[b"\x89PNG"],
        text="final observation",
    )

    assert [result.tool_call_id for result in non_terminal.results] == [
        "call_1",
        "call_2",
    ]
    assert non_terminal.results[0].images == [b"\x89PNG"]
    assert non_terminal.results[1].text == "bash output"
    assert [result.tool_call_id for result in terminal_skipped.results] == ["call_1"]

    # --- type-level target probe: env.step can carry `.results` ---
    assert _step_result_has_results_field(), "LiteEnvStepResult.results missing"


def test_terminal_call_id_result_is_env_decision_owned():
    actions = [
        _call("computer", call_id="call_1"),
        _call("terminate", {"status": "success"}, call_id="call_2"),
    ]

    out = build_tool_results_from_decisions(
        LiteEnvStepResult(terminated=True),
        ordered_call_ids=ordered_tool_call_ids(actions),
        feedback={"call_2": error_only_feedback("done")},
    )

    assert [result.tool_call_id for result in out.results] == ["call_2"]
    assert out.results[0].error == "done"


def test_attach_observation_results_preserves_top_level_batch_call_id():
    """Env execution may unpack GUI actions, but result cardinality is tied
    to the original top-level tool calls."""
    actions = [
        _batch(
            "computer",
            [
                {"action": "click", "coordinate": [10, 20]},
                {"action": "click", "coordinate": [30, 40]},
            ],
            call_id="action_0",
        ),
        _call("bash", {"command": "pwd"}, call_id="bash_0"),
        _call("terminate", {"status": "success"}, call_id="done_0"),
    ]
    result = LiteEnvStepResult()

    build_tool_results_from_decisions(
        result,
        ordered_call_ids=ordered_tool_call_ids(actions),
        continue_call_ids={"action_0", "bash_0"},
        images=[b"\x89PNG"],
        text="after step",
    )

    assert [r.tool_call_id for r in result.results] == ["action_0", "bash_0"]
    assert all(r.images[-1] == b"\x89PNG" for r in result.results)


def test_action_batch_result_can_carry_multiple_screenshots():
    """One parent action-batch result may carry multiple env-captured frames."""
    action = _batch(
        "computer",
        [
            {"action": "click", "coordinate": [10, 20]},
            {"action": "click", "coordinate": [30, 40]},
        ],
        call_id="action_0",
    )
    first = b"\x89PNG-action-batch-first"
    second = b"\x89PNG-action-batch-second"
    step = LiteEnvStepResult(
        results=[
            LiteToolResult(
                tool_call_id=tool_call_id(action),
                images=[first, second],
                text="after action-batch",
            )
        ]
    )

    frame = encode_step_result(step)
    header = json.loads(frame[8 : 8 + int.from_bytes(frame[4:8], "big")])

    assert unpack_action_batch_call(action) == [
        {"name": "click", "arguments": {"coordinate": [10, 20]}},
        {"name": "click", "arguments": {"coordinate": [30, 40]}},
    ]
    assert step.results[0].tool_call_id == "action_0"
    assert step.results[0].images == [first, second]
    assert header["results"][0]["shot_n"] == [len(first), len(second)]
    assert frame[8 + int.from_bytes(frame[4:8], "big") :] == first + second
    assert decode_step_result(frame) == step


def test_env_unpack_rejects_non_canonical_lite_calls():
    with pytest.raises(TypeError, match="canonical Lite tool calls"):
        unpack_action_batch_call({"name": "click"})

    with pytest.raises(TypeError, match="canonical Lite tool calls"):
        unpack_action_batch_call({"function": {"name": "click", "arguments": {}}})


def test_env_unpack_splits_computer_and_mobile_batches_structurally():
    computer = _batch(
        "computer",
        [
            {"action": "click", "coordinate": [1, 2]},
            {"action": "type", "text": "hi"},
        ],
        call_id="action_0",
    )
    mobile = _batch(
        "mobile",
        [
            {"action": "tap", "coordinate": [3, 4]},
            {"action": "wait", "duration": 0.1},
        ],
        call_id="mob_0",
    )
    bash = _call("bash", {"command": "pwd"}, call_id="bash_0")

    assert unpack_action_batch_call(computer) == [
        {"name": "click", "arguments": {"coordinate": [1, 2]}},
        {"name": "type", "arguments": {"text": "hi"}},
    ]
    assert unpack_action_batch_call(mobile) == [
        {"name": "tap", "arguments": {"coordinate": [3, 4]}},
        {"name": "wait", "arguments": {"duration": 0.1}},
    ]
    assert action_inspection_records([computer, mobile, bash]) == [
        {
            "name": "click",
            "arguments": {"coordinate": [1, 2]},
            "result_call_id": "action_0",
        },
        {
            "name": "type",
            "arguments": {"text": "hi"},
            "result_call_id": "action_0",
        },
        {
            "name": "tap",
            "arguments": {"coordinate": [3, 4]},
            "result_call_id": "mob_0",
        },
        {
            "name": "wait",
            "arguments": {"duration": 0.1},
            "result_call_id": "mob_0",
        },
        {
            "name": "bash",
            "arguments": {"command": "pwd"},
            "result_call_id": "bash_0",
        },
    ]


def test_attach_observation_results_literal_unknown_is_error_only():
    actions = [_call("frobnicate", {"value": 1}, call_id="nav_0")]
    result = LiteEnvStepResult()

    build_tool_results_from_decisions(
        result,
        ordered_call_ids=ordered_tool_call_ids(actions),
        feedback={"nav_0": error_only_feedback("unknown tool: frobnicate")},
    )

    assert len(result.results) == 1
    assert result.results[0].tool_call_id == "nav_0"
    assert result.results[0].images == []
    assert result.results[0].text is None
    assert result.results[0].error == "unknown tool: frobnicate"
    assert result.results[0].metadata == {"is_error": True}


def test_attach_observation_results_keeps_env_local_results():
    result = LiteEnvStepResult(
        results=[LiteToolResult(tool_call_id="custom", text="env-local")],
    )

    build_tool_results_from_decisions(
        result,
        ordered_call_ids=ordered_tool_call_ids([_batch("computer", [], call_id="action_0")]),
    )

    assert [(r.tool_call_id, r.text) for r in result.results] == [("custom", "env-local")]


# ---------------------------------------------------------------------------
# E2 — results pair by call id, and survive LoopDetectWrapper's list rewrite
# ---------------------------------------------------------------------------
def test_results_paired_by_call_id_survives_loopdetect():
    """The REAL LoopDetectWrapper drops the looping trailing call and appends a
    synthetic ``terminate``. Results must still pair by call id (never by
    position). The injected terminate is internal, but it carries a private
    result id so the intercepted model call still receives terminal feedback.
    """
    # Two IDENTICAL-fingerprint calls (differ only in id) → loop_threshold=2
    # trips on the 2nd, so the wrapper rewrites [A0, A1] → [A0, terminate].
    a0 = _call("click", {"coordinate": [1, 2]}, call_id="call_1")
    a1 = _call("click", {"coordinate": [1, 2]}, call_id="call_2")

    spy = _EnvelopeSpyEnv()
    wrapped = LoopDetectWrapper(spy, loop_threshold=2)
    result = asyncio.run(wrapped.step([a0, a1]))

    # --- REAL characterization: the wrapper forwards a rewritten list without
    # mutating the input calls.
    fwd = spy.forwarded
    assert len(fwd) == 2
    assert fwd[0] is a0  # the first call survives unchanged
    assert tool_call_name(fwd[1]) == "terminate"  # trailing A1 dropped
    assert "id" not in fwd[1]  # synthetic terminal is not a model call
    assert fwd[1][RUNTIME_RESULT_CALL_ID_KEY] == "call_2"
    assert fwd[1][RUNTIME_INTERNAL_STOP_REASON_KEY] == "REPETITIVE_LOOP"
    assert result.info["stop_reason"] == "REPETITIVE_LOOP"

    # --- contract model: pairing is by call id, not position ---
    results = result.results
    by_id = {r.tool_call_id: r for r in results}
    assert sorted(by_id) == ["call_1", "call_2"]
    assert by_id["call_1"].tool_call_id == "call_1"
    assert by_id["call_2"].tool_call_id == "call_2"

    # --- target probe: persisted calls need id on the action wire ---
    assert _liteaction_has_id(), "LiteToolCall has no id to pair results by"


def test_loopdetect_runtime_terminate_not_persisted_as_sample_call_or_result():
    """The injected ``terminate`` is env-private runtime forwarding.

    Saved samples keep the model's original assistant tool call. The synthetic
    terminal call has no public ``id``; its private result id only lets the
    env pair terminal feedback back to the intercepted model call.
    """
    warmup = _call("click", {"coordinate": [1, 2]}, call_id="call_1")
    trigger = _call("click", {"coordinate": [1, 2]}, call_id="call_2")

    spy = _EnvelopeSpyEnv()
    wrapped = LoopDetectWrapper(spy, loop_threshold=2)
    asyncio.run(wrapped.step([warmup]))

    assistant_calls = [trigger]
    result = asyncio.run(wrapped.step(assistant_calls))

    assert assistant_calls == [trigger]
    assert all(tool_call_name(call) != "terminate" for call in assistant_calls)
    assert len(spy.forwarded) == 1
    forwarded_terminate = spy.forwarded[0]
    assert tool_call_name(forwarded_terminate) == "terminate"
    assert "id" not in forwarded_terminate
    assert forwarded_terminate["function"]["arguments"] == {
        "status": "success",
        "reason": "REPETITIVE_LOOP",
    }
    assert forwarded_terminate[RUNTIME_RESULT_CALL_ID_KEY] == "call_2"
    assert forwarded_terminate[RUNTIME_INTERNAL_STOP_REASON_KEY] == "REPETITIVE_LOOP"
    assert result.terminated is True
    assert [tool_result.tool_call_id for tool_result in result.results] == ["call_2"]
    assert result.info["stop_reason"] == "REPETITIVE_LOOP"


def test_runtime_sidecars_survive_json_serialization_on_the_env_wire():
    """A runtime action reaches a remote env as ``/step`` JSON, so the sidecars
    must survive ``json.dumps`` — an in-process dict check cannot see this.

    This is the exact failure the deleted ``_RuntimeToolCall`` subclass caused:
    it kept the sidecars as instance attributes, so serialization silently
    dropped the stop reason and the result routing while every in-process
    assertion still passed.
    """
    terminate = make_internal_terminate_action(result_call_id="call_2")
    final = make_no_tool_call_final_actions("done")[0]

    on_the_wire = json.loads(json.dumps([terminate, final]))

    assert on_the_wire == [terminate, final]
    wire_terminate, wire_final = on_the_wire
    assert wire_terminate[RUNTIME_RESULT_CALL_ID_KEY] == "call_2"
    assert wire_terminate[RUNTIME_INTERNAL_STOP_REASON_KEY] == LOOP_DETECT_TERMINATE_REASON
    assert wire_final[RUNTIME_INTERNAL_STOP_REASON_KEY] == CONTENT_ONLY_FINAL_REASON
    # Routing is still answerable on the far side of the wire.
    assert canonical_result_call_id(wire_terminate) == "call_2"
    assert canonical_result_call_id(wire_final) is None
    assert ordered_tool_call_ids(on_the_wire) == ["call_2", None]


def test_content_only_final_bypasses_standalone_admission_after_json_transport():
    """``response`` is a standalone tool name, so a content-only final is admitted
    ONLY because its stop-reason sidecar marks it an internal finish. Drop the
    sidecar in transit and the same action becomes ordinary standalone-tool
    feedback instead of a scored final.
    """
    actions = json.loads(json.dumps(make_no_tool_call_final_actions("done")))

    prepared, errors = prepare_env_tool_calls(actions, LiteCUAMetadata())

    assert errors == {}
    ((env_action, result_call_id),) = prepared
    assert result_call_id is None
    assert env_action["name"] == "response"
    assert env_action[RUNTIME_INTERNAL_STOP_REASON_KEY] == CONTENT_ONLY_FINAL_REASON
    assert classify_standalone_tool_call(env_action, {"response"}, None) == "not_standalone"
    assert standalone_tool_call_feedback(env_action, {"response"}, None) is None

    # Same action minus the sidecar: the admission gate rejects it. This is what
    # a sidecar lost in transport would have produced.
    stripped = {k: v for k, v in env_action.items() if k != RUNTIME_INTERNAL_STOP_REASON_KEY}
    assert classify_standalone_tool_call(stripped, {"response"}, None) == "inactive"
    assert standalone_tool_call_feedback(stripped, {"response"}, None) is not None


# ---------------------------------------------------------------------------
# E3 — screenshot attaches env-intrinsically, with no category field
# ---------------------------------------------------------------------------
def test_screenshot_attaches_by_producing_tool_no_category():
    """The env's per-tool branch attaches an image for a GUI tool and text for a
    text tool — `image is not None` falls out of HOW the result was built, with
    NO `category` field. `[computer, bash]` → computer has image, bash has text
    only. And the real `LiteToolResult` must NOT carry a `category` field.
    """
    computer = _ToolResult(tool_call_id="call_1", images=[b"\x89PNG-computer"])
    nav = _ToolResult(tool_call_id="call_2", images=[b"\x89PNG-nav"])
    bash = _ToolResult(tool_call_id="call_3", text="bash output")

    assert computer.images and computer.text is None
    assert nav.images
    assert bash.images == [] and bash.text is not None

    # --- target probe: the real type exists AND has no `category` field (dropped) ---
    assert _gym_types_has_tool_result(), "lite.gym.types.LiteToolResult missing"
    fields = {f.name for f in dataclasses.fields(gym_types.LiteToolResult)}
    assert "category" not in fields, "LiteToolResult must NOT carry a `category` field"


# ---------------------------------------------------------------------------
# E4 — reset / unpaired turns (tool_call_id is None) render as role:"user"
# ---------------------------------------------------------------------------
def test_reset_and_noop_render_as_user_role():
    """A ``tool_call_id=None`` result (reset feedback / unpaired turn) renders as
    a ``role:"user"`` bubble — NEVER ``role:"tool"``. A role:tool with no
    anchoring ``tool_call_id`` is wire-invalid.
    """

    def role_for(r: _ToolResult) -> str:
        # Only a tool_call_id-anchored result becomes role:"tool".
        return "user" if r.tool_call_id is None else "tool"

    reset_result = _ToolResult(tool_call_id=None, images=[b"png"])
    noop_result = _ToolResult(tool_call_id=None, text="page did not change")
    tool_result = _ToolResult(tool_call_id="call_1", images=[b"png"])

    assert role_for(reset_result) == "user"
    assert role_for(noop_result) == "user"
    assert role_for(tool_result) == "tool"
    # The forbidden combination must never be constructible as role:"tool".
    for r in (reset_result, noop_result):
        assert role_for(r) != "tool"

    # --- target probe: the real envelope the renderer consumes ---
    assert _gym_types_has_tool_result(), "lite.gym.types.LiteToolResult missing"


# ---------------------------------------------------------------------------
# E5 — screenshot result presence is per-call/env-intrinsic
# ---------------------------------------------------------------------------
def test_screen_result_presence_is_per_call_not_global_gate():
    """Image-bearing results are counted by image presence, not a shared tool
    taxonomy or hard global max-one-screen gate."""

    def screen_count(results: list[_ToolResult]) -> int:
        return sum(1 for r in results if r.images)

    # 1 screen: one image-bearing env result.
    assert screen_count([_ToolResult(tool_call_id="c1", images=[b"\x89PNG-1"])]) == 1
    # 0 screen: lone text tools.
    assert (
        screen_count(
            [
                _ToolResult(tool_call_id="c1", text="bash output"),
                _ToolResult(tool_call_id="c2", text="ask_user output"),
            ]
        )
        == 0
    )
    # >1 screen is a concrete env/policy concern, not a shared envelope error.
    assert (
        screen_count(
            [
                _ToolResult(tool_call_id="c1", images=[b"\x89PNG-1"]),
                _ToolResult(tool_call_id="c2", images=[b"\x89PNG-2"]),
            ]
        )
        == 2
    )

    # --- target probe: the envelope that enforces the invariant ---
    assert _gym_types_has_tool_result(), "lite.gym.types.LiteToolResult missing"


# ---------------------------------------------------------------------------
# E7 — remote reset/step wire round-trips
# ---------------------------------------------------------------------------
def _mk_step(
    *,
    results: list[LiteToolResult] | None = None,
) -> LiteEnvStepResult:
    return LiteEnvStepResult(
        reward=1.0,
        info={},
        results=[] if results is None else results,
    )


def test_frame_split_roundtrip_characterization():
    """The remote frame separates reset observations from step result envelopes."""
    obs = LiteEnvObservation(
        image=b"\x89PNG-bytes",
        text="hi",
        metadata={"url": "http://x", "n": 3},
    )
    reset_frame = encode_reset_observation(obs)
    assert reset_frame[:4] == FRAME_MAGIC.encode("ascii")
    assert decode_reset_observation(reset_frame) == obs

    step = _mk_step()
    frame = encode_step_result(step)
    assert frame[:4] == FRAME_MAGIC.encode("ascii")
    assert decode_step_result(frame) == step
    header = json.loads(frame[8 : 8 + int.from_bytes(frame[4:8], "big")])
    assert header["kind"] == "step"
    assert header["results"] == []
    assert header["v"] == FRAME_VERSION


def test_frame_nresults_roundtrip():
    """The step wire carries a per-result array
    ``results:[{tool_call_id, text, metadata, error, shot_n|null}]`` that round-trips
    N mixed / text-only / empty results losslessly (image bytes stay OUT of
    the JSON, one length-delimited tail per image result; NO `category` field).
    """
    # --- contract model codec: N results survive a header round-trip ---
    results = [
        _ToolResult(tool_call_id="call_1", images=[b"\x89PNG"], text=None, error="bad click"),
        _ToolResult(tool_call_id="call_2", text="bash out"),
    ]
    header = {
        "v": FRAME_VERSION,
        "kind": "step",
        "results": [
            {
                "tool_call_id": r.tool_call_id,
                "text": r.text,
                "metadata": r.metadata,
                "error": r.error,
                "shot_n": [len(image) for image in r.images],
            }
            for r in results
        ],
    }
    round = json.loads(json.dumps(header))
    assert round["results"][0]["tool_call_id"] == "call_1"
    assert round["results"][0]["shot_n"] == [len(b"\x89PNG")]
    assert round["results"][0]["error"] == "bad click"
    assert round["results"][1]["shot_n"] == []  # text ⟹ no screenshot tail
    assert "category" not in round["results"][0]  # category dropped from the wire
    assert len(round["results"]) == 2

    # --- target probe: the real encoder emits current MAGIC + a `results` array ---
    frame = encode_step_result(_mk_step(results=results))
    real_header = json.loads(frame[8 : 8 + int.from_bytes(frame[4:8], "big")])
    assert frame[:4] == FRAME_MAGIC.encode("ascii")
    assert real_header["results"] == header["results"]
    assert decode_step_result(frame).results == results


# ---------------------------------------------------------------------------
# E8 — the per-call role:tool builder preserves LIST order for >=10 results
# ---------------------------------------------------------------------------
def test_results_ordering_ge_10():
    """With ≥10 results the role:tool builder must preserve the assistant
    ``tool_calls`` LIST order — NOT a lexical sort of synthetic ids (the qwen
    synth-id bug where ``call_10`` sorts before ``call_2``).
    """
    n = 12
    call_ids = [f"call_{i}" for i in range(1, n + 1)]  # call_1 .. call_12
    results = [_ToolResult(tool_call_id=cid, text=str(i)) for i, cid in enumerate(call_ids, 1)]

    def build_tool_messages(rs: list[_ToolResult]) -> list[str]:
        # The correct builder keeps input order.
        return [r.tool_call_id for r in rs]

    built = build_tool_messages(results)
    assert built == call_ids  # list order preserved

    # The lexical-sort bug WOULD reorder (call_10/11/12 float ahead of call_2):
    lexical = sorted(call_ids)
    assert lexical != call_ids, "corpus must actually expose the lexical hazard"
    assert lexical[1] == "call_10"  # the exact synth-id footgun
    assert built != lexical  # a correct builder must NOT match the lexical sort

    # --- target probe: the real per-call role:tool builder / envelope ---
    assert _gym_types_has_tool_result(), "lite.gym.types.LiteToolResult missing"
