"""Focused wrapper tests for canonical GUI batching.

GUI actions may batch into one canonical ``computer`` / ``mobile`` tool call::

    {"id": "call_0",
     "type": "function",
     "function": {
         "name": "computer",
         "arguments": {"actions": [
             {"action": "click", "coordinate": [..]},
             {"action": "type",  "text": ".."}]}}}

These tests keep :class:`LoopDetectWrapper` strict against reward drift while
preserving the single logical GUI call: wrappers may inspect inner actions,
but forwarded pre-trigger GUI work stays a canonical truncated ``computer`` /
``mobile`` batch. Runtime-injected
``terminate`` is env-private forwarding and must not mutate the input assistant
calls.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/gym/wrappers/test_wrappers_batched.py -p no:cacheprovider -q
"""

from __future__ import annotations

import io
from typing import Any

import pytest
from PIL import Image

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.calls import (
    RUNTIME_RESULT_CALL_ID_KEY,
    EnvAction,
    LiteToolCall,
    make_tool_call,
    tool_call_id,
    tool_call_name,
)
from lite.core.tools.results import LiteToolResult
from lite.core.tools.schemas import make_tool_schema
from lite.gym.base import LiteBaseEnv
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult
from lite.gym.utils.feedback.ingress import (
    make_internal_terminate_action,
    prepare_env_tool_calls,
)
from lite.gym.utils.feedback.results import (
    build_tool_results_from_decisions,
    ordered_tool_call_ids,
)
from lite.gym.wrappers import (
    LoopDetectWrapper,
)

# ---------------------------------------------------------------------------
# Stub env — copied idiom from tests/gym/wrappers/test_wrappers.py::_StubEnv.
# Records the actions it RECEIVED (post-wrapper), returns vanilla obs.
# ---------------------------------------------------------------------------


class _StubEnv(LiteBaseEnv):
    def __init__(self):
        self.stepped: list[list[LiteToolCall]] = []
        self.reset_count = 0

    def _runtime_metadata(self) -> LiteCUAMetadata:
        return LiteCUAMetadata(dims=("desktop", "use"))

    async def reset(self) -> LiteEnvObservation:
        self.reset_count += 1
        return LiteEnvObservation(image=None, text="reset")

    async def step(self, actions: list[LiteToolCall]) -> LiteEnvStepResult:
        self.stepped.append(list(actions))
        return LiteEnvStepResult()

    async def close(self) -> None:
        pass


class _TerminateAwareEnv(_StubEnv):
    """Env that ends the episode when it receives a ``terminate`` call.

    Models the real env's terminate handler (which sets ``terminated=True``
    and runs ``_evaluate``) at the granularity these tests need. The wrapper's
    runtime ``terminate`` is matched here by name and is not a persisted
    assistant tool call.
    """

    async def step(self, actions: list[LiteToolCall]) -> LiteEnvStepResult:
        self.stepped.append(list(actions))
        terminated = any(tool_call_name(a) in {"terminate", "response"} for a in actions)
        return LiteEnvStepResult(
            terminated=terminated,
        )


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", (64, 64), (255, 255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


class _PairingTerminalEnv(_StubEnv):
    async def step(self, actions: list[LiteToolCall]) -> LiteEnvStepResult:
        self.stepped.append(list(actions))
        terminated = any(tool_call_name(a) == "terminate" for a in actions)
        result = LiteEnvStepResult(
            reward=0.61 if terminated else None,
            terminated=terminated,
            truncated=False,
            info={"env_terminal": terminated},
        )
        ordered = ordered_tool_call_ids(actions)
        continue_call_ids = set(ordered)
        return build_tool_results_from_decisions(
            result,
            ordered_call_ids=ordered,
            continue_call_ids=continue_call_ids,
            images=[_png()],
            text="env observation",
        )


class _MetadataEnv(_StubEnv):
    def __init__(
        self,
        valid_actions: list[str] | None,
        extra_tool_schemas: list[dict] | None = None,
        *,
        platform: str = "desktop",
        task_type: str = "use",
    ):
        super().__init__()
        self._valid_actions = valid_actions
        self._extra_tool_schemas = extra_tool_schemas or []
        self._platform = platform
        self._task_type = task_type

    def _runtime_metadata(self) -> LiteCUAMetadata:
        return LiteCUAMetadata(
            dims=(self._platform, self._task_type),
            valid_actions=self._valid_actions,
            extra_tool_schemas=self._extra_tool_schemas,
        )

    async def step(self, actions: list[LiteToolCall]) -> LiteEnvStepResult:
        self.stepped.append(list(actions))
        screenshot = _png()
        terminated = any(tool_call_name(a) in {"terminate", "response"} for a in actions)
        return LiteEnvStepResult(
            terminated=terminated,
            results=[
                LiteToolResult(tool_call_id=call_id, images=[screenshot])
                for a in actions
                if (call_id := tool_call_id(a))
            ],
        )


# ---------------------------------------------------------------------------
# Action builders.
# ---------------------------------------------------------------------------


def _bare(name: str, **arguments: Any) -> EnvAction:
    """Env-internal child action shape used for fingerprint expectations."""
    return {"name": name, "arguments": arguments}


def _call(name: str, arguments: dict | None = None, *, call_id: str | None = None) -> LiteToolCall:
    return make_tool_call(name, arguments or {}, call_id=call_id)


def _click(x: int, y: int, *, call_id: str | None = None) -> LiteToolCall:
    """Canonical click in the ``{x, y}`` schema used by loop tests."""
    return _call("click", {"x": x, "y": y}, call_id=call_id)


def _computer(inner_actions: list[dict], call_id: str = "call_0") -> LiteToolCall:
    """A single batched ``computer`` tool call wrapping k GUI actions.

    Inner actions use the canonical sub-action shape (``{"action": ...,
    "coordinate"/"text": ...}``), NOT the LiteToolCall ``function`` envelope.
    """
    return _call("computer", {"actions": inner_actions}, call_id=call_id)


def _mobile(inner_actions: list[dict], call_id: str = "call_0") -> LiteToolCall:
    """A single batched ``mobile`` tool call — mobile batches like ``computer``
    (owner decision), so the wrapper's descend-into-``.actions`` fix must handle
    it identically. mobilegym/androidworld use LoopDetect, so a computer-only
    fix would silently regress reward on mobile batched turns."""
    return _call("mobile", {"actions": inner_actions}, call_id=call_id)


def _terminate_success(result_call_id: str | None = None) -> LiteToolCall:
    # Mirror LoopDetectWrapper's default injected terminal status.
    return make_internal_terminate_action(result_call_id=result_call_id)


def _without_canonical_id(action: LiteToolCall) -> LiteToolCall:
    """Return a copy for runtime-injected forwarding with no assistant-call id."""
    return {key: value for key, value in action.items() if key != "id"}


_CLICK_INDEX_SCHEMA = make_tool_schema(
    "click",
    description="Click a DOM element by index.",
    parameters={
        "type": "object",
        "properties": {"index": {"type": "integer"}},
        "required": ["index"],
    },
)


# ===========================================================================
# W1 — LoopDetect fingerprint granularity (the most important: reward drift)
# ===========================================================================


@pytest.mark.asyncio
async def test_loopdetect_does_not_classify_active_extra_action_name_collision():
    """Name/argument collisions are env-owned. The wrapper fingerprints the
    canonical Lite call without reading extra_tool_schemas."""
    inner = _MetadataEnv(
        valid_actions=[],
        extra_tool_schemas=[_CLICK_INDEX_SCHEMA],
    )
    env = LoopDetectWrapper(inner, loop_threshold=2)
    await env.reset()
    first = _call("click", {"index": 7}, call_id="dom_1")
    second = _call("click", {"index": 7}, call_id="dom_2")

    r1 = await env.step([first])
    r2 = await env.step([second])

    assert "stop_reason" not in r1.info
    assert r2.info["stop_reason"] == "REPETITIVE_LOOP"
    assert inner.stepped == [[first], [_terminate_success("dom_2")]]


def test_prepare_env_tool_calls_rejects_only_the_bad_child():
    metadata = LiteCUAMetadata(dims=("desktop", "use"))
    malformed = _computer(
        [
            {"action": "open_app", "app_name": "Clock"},
            {"action": "click", "coordinate": [10, 20]},
        ],
        call_id="call_action",
    )

    routed, feedback = prepare_env_tool_calls([malformed], metadata)

    # R4: the bad child costs itself, not its sibling. Both slots reach the env
    # -- one carrying its reason so the env can answer and frame it, the other
    # untouched so it still runs.
    assert [a["name"] for a, _ in routed] == ["open_app", "click"]
    assert routed[0][0]["_rejected_reason"] == (
        "invalid action: open_app; computer.actions cannot contain open_app"
    )
    assert routed[1][0].get("_rejected_reason") is None
    assert feedback == {}


@pytest.mark.asyncio
async def test_loopdetect_batched_computer_fingerprint_granularity():
    """A ``computer`` call batching k=2 distinct actions must produce k
    fingerprints matching the per-action baseline.

    Assertion: after ONE batched step the wrapper's fingerprint buffer holds
    the 2-element per-action baseline sequence.
    """
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=3, max_period=2)
    await env.reset()

    batched = _computer(
        [
            {"action": "click", "coordinate": [1, 1]},
            {"action": "click", "coordinate": [2, 2]},
        ]
    )
    await env.step([batched])

    # Descend into computer.actions -> k=2 fingerprints, one per action,
    # byte-identical to the unbatched per-action baseline sequence.
    baseline = [
        LoopDetectWrapper._fingerprint(_bare("click", coordinate=[1, 1])),
        LoopDetectWrapper._fingerprint(_bare("click", coordinate=[2, 2])),
    ]
    assert list(env._recent_fps) == baseline


@pytest.mark.asyncio
async def test_loopdetect_batched_mobile_fingerprint_granularity():
    """Symmetric with the computer W1 case: a batched ``mobile`` call with k=2
    actions must produce k per-action fingerprints, not 1 top-level. The
    descend-into-`.actions` fix must cover `mobile`, not just `computer`."""
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=3, max_period=2)
    await env.reset()

    await env.step(
        [
            _mobile(
                [{"action": "tap", "coordinate": [1, 1]}, {"action": "swipe", "coordinate": [2, 2]}]
            )
        ]
    )

    baseline = [
        LoopDetectWrapper._fingerprint(_bare("tap", coordinate=[1, 1])),
        LoopDetectWrapper._fingerprint(_bare("swipe", coordinate=[2, 2])),
    ]
    assert list(env._recent_fps) == baseline


@pytest.mark.asyncio
async def test_loopdetect_single_action_computer_matches_unbatched_baseline():
    """The k=1 baseline stays byte-stable.

    A ``computer`` call wrapping a SINGLE action must drive loop detection
    identically to the equivalent unbatched per-action call: the loop closes
    at the same step and each turn contributes exactly one fingerprint.
    """
    threshold = 3

    # Baseline: unbatched per-action clicks.
    unbatched_inner = _StubEnv()
    unbatched_env = LoopDetectWrapper(unbatched_inner, loop_threshold=threshold)
    await unbatched_env.reset()
    unbatched_trigger_step = None
    for step in range(1, threshold + 1):
        r = await unbatched_env.step([_click(50, 50)])
        if r.info.get("stop_reason") == "REPETITIVE_LOOP":
            unbatched_trigger_step = step
            break

    # Batched single-action computer calls.
    batched_inner = _StubEnv()
    batched_env = LoopDetectWrapper(batched_inner, loop_threshold=threshold)
    await batched_env.reset()
    one = _computer([{"action": "click", "coordinate": [50, 50]}])
    batched_trigger_step = None
    for step in range(1, threshold + 1):
        r = await batched_env.step([one])
        if r.info.get("stop_reason") == "REPETITIVE_LOOP":
            batched_trigger_step = step
            break

    # Same trigger step, and one fingerprint per single-action turn.
    assert unbatched_trigger_step == threshold
    assert batched_trigger_step == unbatched_trigger_step
    assert len(batched_env._recent_fps) == len(unbatched_env._recent_fps)


# ===========================================================================
# W2 — Injected terminate actually ends the episode
# ===========================================================================


@pytest.mark.asyncio
async def test_loopdetect_injected_terminate_sets_terminated():
    """The wrapper's injected ``terminate`` must make the env end
    with ``terminated=True``.

    ``_make_terminate_action`` emits a runtime internal ``terminate`` call, and
    a faithful env stub terminates on that name.
    The input assistant calls are not mutated.
    """
    inner = _TerminateAwareEnv()
    env = LoopDetectWrapper(inner, loop_threshold=2)
    await env.reset()
    await env.step([_click(1, 1)])
    actions = [_click(1, 1)]
    r = await env.step(actions)  # 2nd identical -> loop -> inject terminate

    assert r.info["stop_reason"] == "REPETITIVE_LOOP"
    assert r.terminated is True
    assert actions == [_click(1, 1)]
    # The env saw exactly the injected terminate (the original click dropped).
    assert inner.stepped[-1] == [_terminate_success()]


@pytest.mark.parametrize(
    ("builder", "prefix_action", "repeat_name"),
    [
        (_computer, "mouse_move", "click"),
        (_mobile, "swipe", "tap"),
    ],
    ids=["computer", "mobile"],
)
@pytest.mark.asyncio
async def test_loopdetect_terminal_rewrite_preserves_surviving_call_id_results(
    builder,
    prefix_action: str,
    repeat_name: str,
):
    inner = _PairingTerminalEnv()
    env = LoopDetectWrapper(inner, loop_threshold=2, max_period=2)
    await env.reset()
    prefix = {"action": prefix_action, "coordinate": [1, 1]}
    trigger = {"action": repeat_name, "coordinate": [2, 2]}
    trailing = {"action": repeat_name, "coordinate": [3, 3]}
    await env.step(
        [
            _call(prefix_action, {"coordinate": [1, 1]}),
            _call(repeat_name, {"coordinate": [2, 2]}),
        ]
    )

    assistant_call = builder([prefix, trigger, trailing], call_id="call_action")
    result = await env.step([assistant_call])

    assert inner.stepped[-1] == [
        _without_canonical_id(builder([prefix], call_id="call_action")),
        _terminate_success("call_action"),
    ]
    assert "id" not in inner.stepped[-1][1]
    assert inner.stepped[-1][1][RUNTIME_RESULT_CALL_ID_KEY] == "call_action"
    assert [tool_result.tool_call_id for tool_result in result.results] == ["call_action"]
    assert result.results[0].images
    assert result.results[0].text == "env observation"
    assert result.results[0].error is None
    assert result.terminated is True
    assert result.truncated is False
    assert result.reward == 0.61
    assert result.info["env_terminal"] is True
    assert result.info["stop_reason"] == "REPETITIVE_LOOP"
    assert assistant_call == builder([prefix, trigger, trailing], call_id="call_action")


@pytest.mark.parametrize(
    ("builder", "prefix_action", "repeat_name"),
    [
        (_computer, "mouse_move", "click"),
        (_mobile, "swipe", "tap"),
    ],
    ids=["computer", "mobile"],
)
@pytest.mark.asyncio
async def test_loopdetect_batched_action_mid_batch_preserves_prefix_and_input(
    builder,
    prefix_action: str,
    repeat_name: str,
):
    """If an inner action closes the loop, earlier inner actions stay in
    one truncated canonical GUI batch before the runtime terminate call."""
    inner = _StubEnv()
    env = LoopDetectWrapper(inner, loop_threshold=2, max_period=2)
    await env.reset()
    prefix = {"action": prefix_action, "coordinate": [1, 1]}
    trigger = {"action": repeat_name, "coordinate": [2, 2]}
    trailing = {"action": repeat_name, "coordinate": [3, 3]}
    await env.step(
        [
            _call(prefix_action, {"coordinate": [1, 1]}),
            _call(repeat_name, {"coordinate": [2, 2]}),
        ]
    )

    batched = builder([prefix, trigger, trailing], call_id="call_keep")
    actions = [batched]

    r = await env.step(actions)

    assert r.info["stop_reason"] == "REPETITIVE_LOOP"
    assert actions == [builder([prefix, trigger, trailing], call_id="call_keep")]
    assert inner.stepped[-1] == [
        _without_canonical_id(builder([prefix], call_id="call_keep")),
        _terminate_success("call_keep"),
    ]
    assert inner.stepped[-1][0] is not batched
