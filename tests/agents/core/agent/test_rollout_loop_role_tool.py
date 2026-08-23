"""Agent rollout loop + agent↔env integration tests for role:"tool" feedback.

Contract under test:

  * ``env.step`` will return ``LiteEnvStepResult.results: list[LiteToolResult]``
    for the per-call feedback path, and the shared rollout loop
    (``lite/agents/core/agent/base.py``) builds **N per-call**
    ``role:"tool"`` messages before the next assistant turn — one per tool_call
    of the *previous* assistant turn, in tool_call order, each carrying its
    ``tool_call_id``.
  * The GPT / Claude bespoke loops, which write their OWN canonical
    ``trajectory.messages`` LiteSample, must likewise write per-call
    ``role:"tool"`` so a teacher trajectory is on-distribution for a qwen student.
  * reset (turn-0) and empty-action feedback have no tool call id → they STAY
    ``role:"user"``.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/agents/core/agent/test_rollout_loop_role_tool.py \
        -p no:cacheprovider -q
"""

from __future__ import annotations

import asyncio
import io
import logging
import threading
from typing import Any

import pandas as pd
import pytest
from PIL import Image

from lite.agents.core.action_space.utils.geometry import required_coord
from lite.agents.core.agent.base import (
    AdapterBasedAgent,
    align_tool_results_to_tool_calls,
)
from lite.agents.core.agent.hooks import SampleHook
from lite.agents.core.agent.logger import TrajectoryLogger
from lite.agents.core.agent.utils.loop import record_lite_env_result
from lite.agents.core.agent.utils.messages import build_tool_result_message
from lite.agents.models.qwen3_5.adapter import (
    Qwen3_5DesktopUseAdapter,
    Qwen3_5MobileUseAdapter,
)
from lite.core import (
    STATUS_FAILED,
    STATUS_TRUNCATED,
    LiteCUAMetadata,
    LiteRLStep,
    LiteSample,
    LiteToolCall,
)
from lite.core.messages.final import (
    CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY,
    CONTENT_ONLY_FINAL_REASON,
    CONTENT_ONLY_FINAL_TEXT,
    EMPTY_FINAL_REASON,
    MODEL_OUTPUT_ERROR_KEY,
    PARSE_FAILURE_FINAL_REASON,
    canonicalize_no_tool_call_final_message,
    mark_model_output_error,
)
from lite.core.messages.image_refs import referenced_image_indices_in_message_order
from lite.core.tools.calls import (
    RUNTIME_INTERNAL_STOP_REASON_KEY,
    make_tool_call,
    stamp_message_tool_call_ids,
    tool_call_arguments,
    tool_call_id,
    tool_call_name,
    with_runtime_internal_stop_reason,
)
from lite.core.tools.extra_tools import LiteFinishToolSet
from lite.core.tools.results import LiteToolResult
from lite.core.tools.results import LiteToolResult as CoreLiteToolResult
from lite.data.staging import coerce_messages
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult
from lite.gym.utils.backend.model_inputs import (
    coerce_model_duration,
)
from lite.gym.utils.feedback.errors import (
    record_model_action_error,
)
from lite.gym.utils.feedback.results import (
    build_tool_results_from_decisions,
    ordered_tool_call_ids,
)

# The GPT/Claude bespoke agents pull in litellm/openai at import; guard so a
# missing optional dep degrades to SKIP (never a collection ImportError that
# would take the whole module — and the hermetic shared-loop tests — down).
try:
    from lite.agents.models.claude.agent import ClaudeDesktopUseAgent
    from lite.agents.models.gpt.agent import GPTDesktopUseAgent

    _BESPOKE_IMPORT_OK = True
except Exception:  # pragma: no cover - only when optional deps absent
    _BESPOKE_IMPORT_OK = False


# ---------------------------------------------------------------------------
# Hermetic fakes for the SHARED rollout loop (AdapterBasedAgent.sample)
# ---------------------------------------------------------------------------
#
# No network, no model download, no real env. The loop's per-turn observation
# message is what we probe: it is built at the TOP of iteration N from the
# result of iteration N-1's env.step, so a two-iteration episode exposes exactly
# one built observation between the two assistant turns.


def _png_bytes(w: int = 4, h: int = 4) -> bytes:
    """A tiny but valid RGB PNG so ``decode_image`` round-trips."""
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (127, 127, 127)).save(buf, format="PNG")
    return buf.getvalue()


# Canned tool_calls. ``id`` is carried on each persisted action so the
# canonical role:tool messages can reference it through ``tool_call_id``.
def _computer_call(cid: str | None = "call_computer") -> LiteToolCall:
    return make_tool_call(
        "computer",
        {"actions": [{"action": "click", "coordinate": [10, 20]}]},
        call_id=cid,
    )


def _bash_call(cid: str | None = "call_bash") -> LiteToolCall:
    return make_tool_call("bash", {"command": "ls"}, call_id=cid)


def _wait_call() -> LiteToolCall:
    return make_tool_call("wait", call_id="call_wait")


def _bad_wait_duration_call() -> LiteToolCall:
    return make_tool_call("wait", {"duration": "forever"}, call_id="call_wait_bad")


def _terminate_call() -> LiteToolCall:
    return make_tool_call("terminate", {"status": "success"}, call_id="call_terminal")


def _call_id(call: dict[str, Any]) -> str | None:
    return tool_call_id(call)


def _call_name(call: dict[str, Any]) -> str:
    return tool_call_name(call)


def _call_args(call: dict[str, Any]) -> dict[str, Any]:
    return tool_call_arguments(call)


def _is_finish_call(call: dict[str, Any]) -> bool:
    return _call_name(call) in {"response", "terminate"}


def _internal_response_call(text: str) -> LiteToolCall:
    return with_runtime_internal_stop_reason(
        make_tool_call("response", {"text": text}),
        CONTENT_ONLY_FINAL_REASON,
    )


def test_agent_alignment_uses_core_tool_result_contract():
    result = CoreLiteToolResult(tool_call_id="call_0000", text="stdout")

    assert LiteToolResult is CoreLiteToolResult
    assert align_tool_results_to_tool_calls(
        [make_tool_call("bash", call_id="call_0000")],
        [result],
    ) == [result]


def test_shared_rollout_image_reference_helpers_stay_in_core_owner():
    from lite.agents.core.agent import base as agent_base
    from lite.core.messages import image_refs as core_image_refs

    assert callable(core_image_refs.referenced_image_indices_in_message_order)
    assert callable(core_image_refs.referenced_images_in_message_order)
    assert not hasattr(agent_base, "referenced_image_indices_in_message_order")
    assert not hasattr(agent_base, "referenced_images_in_message_order")


class _FakeAdapter:
    """Minimal adapter: scripts one assistant message (with tool_calls) per turn.

    ``render_step`` just echoes the accumulated messages (the loop sanitizes +
    orders images off it); ``process_image`` is identity; the parse/convert pair
    ignores the model text and pops the next scripted turn's tool_calls.
    """

    _registry_key = "fake@desktop@use"

    @classmethod
    def get_registry_key(cls) -> str:
        return "fake@desktop@use"
    metadata = LiteCUAMetadata()

    def __init__(self, turns: list[list[LiteToolCall] | str]):
        self._turns = turns
        self._i = 0

    def render_step(self, lite_sample, k: int, processed_images) -> list[dict[str, Any]]:
        return list(lite_sample.messages)

    def process_image(self, img):
        return img

    def select_action_batch_image_indices(
        self, *, tool_call, tool_result, result_image_indices
    ):
        # Mirrors BaseAgentAdapter's default (final frame only). Spelled out
        # here rather than inherited because this fake deliberately does not
        # subclass the real adapter -- if the base default changes, this stays
        # pinned to what these loop tests assert.
        del tool_call, tool_result
        return result_image_indices[-1:]

    def parse_raw_assistant_response(self, response: str) -> dict[str, Any]:
        return {"role": "assistant", "_raw": response}

    def convert_message_from_agent(self, agent_message) -> dict[str, Any]:
        turn = self._turns[self._i] if self._i < len(self._turns) else []
        self._i += 1
        if isinstance(turn, str):
            return {"role": "assistant", "content": [{"type": "text", "text": turn}]}
        return {"role": "assistant", "content": [], "tool_calls": list(turn)}


class _RawTextAdapter(_FakeAdapter):
    """Adapter that preserves the generated raw text as assistant content."""

    def __init__(self):
        self._i = 0

    def parse_raw_assistant_response(self, response: str) -> dict[str, Any]:
        self._i += 1
        return {"role": "assistant", "content": [{"type": "text", "text": response}]}

    def convert_message_from_agent(self, agent_message) -> dict[str, Any]:
        return dict(agent_message)


class _ParseErrorThenActionAdapter(_FakeAdapter):
    def __init__(self):
        self._i = 0

    def parse_raw_assistant_response(self, response: str) -> dict[str, Any]:
        if self._i == 0:
            message = {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            }
            mark_model_output_error(message, "malformed <tool_call> JSON")
            return message
        return {"role": "assistant", "content": [], "tool_calls": [_computer_call(None)]}

    def convert_message_from_agent(self, agent_message) -> dict[str, Any]:
        self._i += 1
        return dict(agent_message)


class _AgentOnlyParseErrorMarkerAdapter(_FakeAdapter):
    def __init__(self):
        self._i = 0

    def parse_raw_assistant_response(self, response: str) -> dict[str, Any]:
        self._i += 1
        message = {
            "role": "assistant",
            "content": [{"type": "text", "text": response}],
        }
        mark_model_output_error(message, "agent-side parse marker")
        return message

    def convert_message_from_agent(self, agent_message) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": list(agent_message.get("content") or []),
        }


class _MalformedToolCallsThenActionAdapter(_FakeAdapter):
    def __init__(self):
        self._i = 0

    def parse_raw_assistant_response(self, response: str) -> dict[str, Any]:
        if self._i == 0:
            return {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
                "tool_calls": {"name": "mobile", "arguments": {}},
            }
        return {"role": "assistant", "content": [], "tool_calls": [_computer_call(None)]}

    def convert_message_from_agent(self, agent_message) -> dict[str, Any]:
        self._i += 1
        return dict(agent_message)


class _DroppedParsedToolCallThenActionAdapter(_FakeAdapter):
    def __init__(self):
        self._i = 0

    def parse_raw_assistant_response(self, response: str) -> dict[str, Any]:
        if self._i == 0:
            return {
                "role": "assistant",
                "content": [{"type": "text", "text": "Action: open Settings"}],
                "tool_calls": [
                    {
                        "name": "mobile_use",
                        "arguments": {"action": "open_app", "app_name": "Settings"},
                    }
                ],
            }
        return {"role": "assistant", "content": [], "tool_calls": [_computer_call(None)]}

    def convert_message_from_agent(self, agent_message) -> dict[str, Any]:
        self._i += 1
        if self._i == 1:
            return {
                "role": "assistant",
                "content": [{"type": "action_description", "text": "open Settings"}],
                "tool_calls": [],
            }
        return dict(agent_message)


class _FakeProcessor:
    """``apply_chat_template`` stub — renders messages to a throwaway string."""

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False) -> str:
        return f"<prompt {len(messages)} msgs>"


def _qwen3_5_xml_tool_call(name: str, **params: str) -> str:
    body = "".join(
        f"<parameter={key}>\n{value}\n</parameter>\n"
        for key, value in params.items()
    )
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


class _FakeEnv:
    """Async env stub. reset()/step() return a screenshot each turn; terminates
    once ``step`` has been called ``terminate_after`` times so the loop exits."""

    def __init__(self, *, instruction: str = "do the task", terminate_after: int = 2):
        self.metadata = LiteCUAMetadata()
        self._instruction = instruction
        self._terminate_after = terminate_after
        self._steps = 0
        self.closed = False
        self.seen_actions: list[list[LiteToolCall]] = []

    async def reset(self) -> LiteEnvObservation:
        return LiteEnvObservation(image=_png_bytes(), text=self._instruction)

    async def step(self, actions: list[LiteToolCall]) -> LiteEnvStepResult:
        self.seen_actions.append(list(actions))
        self._steps += 1
        terminated = (
            not actions
            or any(_is_finish_call(a) for a in actions)
            or self._steps >= self._terminate_after
        )
        results = [
            LiteToolResult(
                tool_call_id=_call_id(action),
                images=[] if _call_name(action) == "bash" else [_png_bytes()],
                text="bash ok" if _call_name(action) == "bash" else None,
            )
            for action in actions
            if not _is_finish_call(action)
        ]
        terminal_action = any(_is_finish_call(a) for a in actions)
        return LiteEnvStepResult(
            reward=1.0 if terminal_action else 0.0,
            terminated=terminated,
            results=results,
        )

    async def close(self):
        self.closed = True


class _ContentOnlyFeedbackEnv(_FakeEnv):
    """Env that scores natural-language answers over several attempts."""

    def __init__(self, *, max_turns: int = 2):
        super().__init__(terminate_after=10_000)
        self.max_turns = max_turns

    async def step(self, actions: list[LiteToolCall]) -> LiteEnvStepResult:
        self.seen_actions.append(list(actions))
        self._steps += 1
        submitted = _call_args(actions[0]).get("text", "") if actions else ""
        terminated = self._steps >= self.max_turns
        return LiteEnvStepResult(
            reward=1.0 if terminated else 0.0,
            terminated=terminated,
            results=[
                LiteToolResult(
                    tool_call_id=None,
                    text=f"attempt {self._steps}: {submitted}",
                )
            ],
        )


class _EmptyContentOnlyFeedbackEnv(_FakeEnv):
    """Env returns a result row that cannot become model-visible feedback."""

    def __init__(self):
        super().__init__(terminate_after=10_000)

    async def step(self, actions: list[LiteToolCall]) -> LiteEnvStepResult:
        self.seen_actions.append(list(actions))
        self._steps += 1
        return LiteEnvStepResult(
            reward=0.0,
            terminated=False,
            results=[LiteToolResult(tool_call_id=None, text="")],
        )


class _PairedContentOnlyFeedbackEnv(_FakeEnv):
    """Env incorrectly pairs feedback to a runtime-only response action."""

    def __init__(self):
        super().__init__(terminate_after=10_000)

    async def step(self, actions: list[LiteToolCall]) -> LiteEnvStepResult:
        self.seen_actions.append(list(actions))
        self._steps += 1
        return LiteEnvStepResult(
            reward=0.0,
            terminated=False,
            results=[LiteToolResult(tool_call_id="call_orphan", text="revise")],
        )


class _CloseRaisesEnv(_FakeEnv):
    async def close(self):
        self.closed = True
        raise RuntimeError("shared close exploded")


class _CloseCancelledEnv(_FakeEnv):
    async def close(self):
        self.closed = True
        raise asyncio.CancelledError


class _MultiImageFirstStepEnv(_FakeEnv):
    async def step(self, actions: list[LiteToolCall]) -> LiteEnvStepResult:
        self.seen_actions.append(list(actions))
        self._steps += 1
        terminated = (
            not actions
            or any(_is_finish_call(a) for a in actions)
            or self._steps >= self._terminate_after
        )
        images = [_png_bytes(4, 4), _png_bytes(5, 5)] if self._steps == 1 else [_png_bytes(6, 6)]
        return LiteEnvStepResult(
            reward=0.0,
            terminated=terminated,
            results=[
                LiteToolResult(
                    tool_call_id=_call_id(action),
                    images=images,
                )
                for action in actions
                if not _is_finish_call(action)
            ],
        )


class _StepRaisesAndCloseRaisesEnv(_CloseRaisesEnv):
    async def step(self, actions: list[LiteToolCall]) -> LiteEnvStepResult:
        self.seen_actions.append(list(actions))
        raise RuntimeError("shared step exploded")


class _ScriptedResultsEnv(_FakeEnv):
    def __init__(self, result_call_ids: list[str], *, terminate_after: int = 2):
        super().__init__(terminate_after=terminate_after)
        self._result_call_ids = result_call_ids

    async def step(self, actions: list[LiteToolCall]) -> LiteEnvStepResult:
        self.seen_actions.append(list(actions))
        self._steps += 1
        result_call_ids = (
            self._result_call_ids
            if self._steps == 1
            else [
                _call_id(action)
                for action in actions
                if not _is_finish_call(action) and _call_id(action)
            ]
        )
        return LiteEnvStepResult(
            reward=0.0,
            terminated=self._steps >= self._terminate_after,
            results=[
                LiteToolResult(tool_call_id=call_id, text=f"result {call_id}")
                for call_id in result_call_ids
            ],
        )


class _NoResultsEnv(_FakeEnv):
    async def step(self, actions: list[LiteToolCall]) -> LiteEnvStepResult:
        self.seen_actions.append(list(actions))
        self._steps += 1
        return LiteEnvStepResult(
            reward=0.0,
            terminated=self._steps >= self._terminate_after,
            results=[],
        )


class _ErrorObservationEnv(_FakeEnv):
    async def step(self, actions: list[LiteToolCall]) -> LiteEnvStepResult:
        self.seen_actions.append(list(actions))
        self._steps += 1
        if self._steps == 1:
            return LiteEnvStepResult(
                reward=0.0,
                terminated=False,
                results=[
                    LiteToolResult(
                        tool_call_id=_call_id(actions[0]),
                        images=[_png_bytes()],
                        text="## AXTree:\nbody",
                        metadata={"is_error": True},
                        error="invalid action: mouse_move",
                    )
                ],
            )
        return LiteEnvStepResult(reward=1.0, terminated=True, results=[])


class _FreshRouteKnownErrorEnv(_FakeEnv):
    async def step(self, actions: list[LiteToolCall]) -> LiteEnvStepResult:
        self.seen_actions.append(list(actions))
        self._steps += 1
        if self._steps == 1:
            return LiteEnvStepResult(
                reward=0.0,
                terminated=False,
                results=[
                    LiteToolResult(
                        tool_call_id=_call_id(actions[0]),
                        images=[_png_bytes(6, 5)],
                        text="## AXTree:\nbutton Search",
                        metadata={"is_error": True},
                        error="invalid action: screenshot",
                    )
                ],
            )
        return LiteEnvStepResult(reward=1.0, terminated=True, results=[])


class _TextOnlyRouteKnownErrorEnv(_FakeEnv):
    async def step(self, actions: list[LiteToolCall]) -> LiteEnvStepResult:
        self.seen_actions.append(list(actions))
        self._steps += 1
        if self._steps == 1:
            return LiteEnvStepResult(
                reward=0.0,
                terminated=False,
                results=[
                    LiteToolResult(
                        tool_call_id=_call_id(actions[0]),
                        text="stdout before failure",
                        metadata={"is_error": True},
                        error="unsupported action: bogus",
                    )
                ],
            )
        return LiteEnvStepResult(reward=1.0, terminated=True, results=[])


class _PairableBadExecutableArgsEnv(_FakeEnv):
    async def step(self, actions: list[LiteToolCall]) -> LiteEnvStepResult:
        self.seen_actions.append(list(actions))
        self._steps += 1
        if any(_is_finish_call(action) for action in actions):
            return LiteEnvStepResult(reward=1.0, terminated=True, results=[])

        ordered = ordered_tool_call_ids(actions)
        feedback = {}
        action = actions[0]
        try:
            coerce_model_duration(
                _call_args(action).get("duration", 5.0),
                action_name=_call_name(action),
            )
        except ValueError as exc:
            record_model_action_error(
                feedback,
                _call_id(action),
                exc,
                action_name=_call_name(action),
            )

        return build_tool_results_from_decisions(
            LiteEnvStepResult(reward=0.0, terminated=False),
            ordered_call_ids=ordered,
            continue_call_ids=ordered,
            images=[_png_bytes(6, 6)],
            feedback=feedback,
        )


class _CaptureStepResults(SampleHook):
    def __init__(self):
        self.result_call_ids: list[list[str | None]] = []

    def on_step(self, data) -> None:
        self.result_call_ids.append([result.tool_call_id for result in data.step_result.results])


class _CaptureStepActions(SampleHook):
    def __init__(self):
        self.actions: list[list[dict[str, Any]]] = []
        self.infos: list[dict[str, Any]] = []

    def on_step(self, data) -> None:
        self.actions.append(list(data.actions))
        self.infos.append(dict(data.step_result.info or {}))


class _CaptureHookThreads(SampleHook):
    """Records the thread each blocking hook method ran on.

    Loggers encode images and write files, so every loop must run them off the
    event loop thread.
    """

    def __init__(self):
        self.step_threads: list[int] = []
        self.complete_threads: list[int] = []

    def on_step(self, data) -> None:
        self.step_threads.append(threading.get_ident())

    def on_complete(self, result) -> None:
        self.complete_threads.append(threading.get_ident())


async def _run_loop(turns: list[list[LiteToolCall] | str]) -> list[dict[str, Any]]:
    """Drive the shared loop for a 2-iteration episode; return canonical messages."""

    async def _gen(**kwargs):
        return {"response": "canned"}

    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_FakeAdapter(turns),
    )
    rl = await agent.sample(_FakeEnv(terminate_after=2), max_steps=10)
    return rl.lite_sample.messages


async def _run_loop_with_env(
    turns: list[list[LiteToolCall] | str],
    *,
    terminate_after: int = 2,
) -> tuple[list[dict[str, Any]], _FakeEnv]:
    async def _gen(**kwargs):
        return {"response": "canned"}

    env = _FakeEnv(terminate_after=terminate_after)
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_FakeAdapter(turns),
    )
    rl = await agent.sample(env, max_steps=10)
    return rl.lite_sample.messages, env


async def _run_loop_with_custom_env(
    turns: list[list[LiteToolCall] | str],
    env: _FakeEnv,
) -> list[dict[str, Any]]:
    async def _gen(**kwargs):
        return {"response": "canned"}

    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_FakeAdapter(turns),
    )
    rl = await agent.sample(env, max_steps=10)
    return rl.lite_sample.messages


async def _run_raw_text_loop_with_env(
    response: str,
    env: _FakeEnv,
):
    async def _gen(**kwargs):
        return {"response": response}

    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_RawTextAdapter(),
    )
    rl = await agent.sample(env, max_steps=10)
    return rl


def _obs_between_assistants(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The observation message(s) the loop built between the first and second
    assistant turns — i.e. the feedback for turn-0's tool_calls."""
    a = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    assert len(a) >= 2, f"expected ≥2 assistant turns, got messages={messages}"
    return messages[a[0] + 1 : a[1]]


def _assert_tool_results_pair_with_previous_assistant(messages: list[dict[str, Any]]) -> None:
    pending: list[str] = []
    seen_results: set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            pending = [
                _call_id(tc)
                for tc in msg.get("tool_calls", [])
                if _call_name(tc) not in {"terminate", "response"}
            ]
            seen_results = set()
            continue
        if msg.get("role") != "tool":
            continue
        cid = msg.get("tool_call_id")
        assert cid in pending, f"orphan role:tool result tool_call_id={cid!r}; pending={pending!r}"
        assert cid not in seen_results, f"duplicate role:tool result tool_call_id={cid!r}"
        seen_results.add(cid)


def _plain(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items() if v is not None}
    return value


def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    normalized = _plain(dict(message))
    if normalized.get("tool_calls") == []:
        normalized.pop("tool_calls")
    return normalized


# ---------------------------------------------------------------------------
# Shared loop tests
# ---------------------------------------------------------------------------


def test_stamp_message_tool_call_ids_restamps_canonical_calls():
    message = {
        "role": "assistant",
        "content": [],
        "tool_calls": [
            make_tool_call("computer", call_id="provider_1"),
            make_tool_call("bash", {"command": "pwd"}),
        ],
    }

    assert stamp_message_tool_call_ids(message, start=3, preserve=False) is message
    assert [_call_id(tc) for tc in message["tool_calls"]] == ["call_0003", "call_0004"]


@pytest.mark.parametrize("bad_call", ["not-a-call", None, ["name", "bash"]])
def test_stamp_message_tool_call_ids_rejects_non_dict_tool_call_entries(bad_call):
    message = {
        "role": "assistant",
        "content": [],
        "tool_calls": [
            make_tool_call("computer"),
            bad_call,
        ],
    }

    with pytest.raises(TypeError):
        stamp_message_tool_call_ids(message, preserve=False)


@pytest.mark.parametrize(
    "tool_calls",
    [
        None,
        (make_tool_call("bash", {"command": "pwd"}),),
        make_tool_call("bash", {"command": "pwd"}),
    ],
)
def test_stamp_message_tool_call_ids_rejects_non_list_tool_calls(tool_calls):
    message = {"role": "assistant", "content": [], "tool_calls": tool_calls}

    with pytest.raises(TypeError, match="list of dict calls"):
        stamp_message_tool_call_ids(message, preserve=False)


async def test_loop_builds_role_tool_per_call():
    # Turn 0 emits two tool_calls; the loop builds the feedback for them at the
    # top of turn 1: one role:tool message per call, in order, each carrying its
    # call_id.
    messages = await _run_loop([[_computer_call(), _bash_call()], [_wait_call()]])
    obs = _obs_between_assistants(messages)

    tool_msgs = [m for m in obs if m.get("role") == "tool"]
    assert len(tool_msgs) == 2, (
        "expected 2 per-call role:tool messages; loop still builds a single "
        f"role:user bubble today → obs={obs}"
    )
    # per-call ordering + tool_call_id plumbed through (computer first, bash second).
    assert tool_msgs[0].get("tool_call_id") == "call_0000"
    assert tool_msgs[1].get("tool_call_id") == "call_0001"


async def test_loop_text_only_tool_result_does_not_consume_image_index():
    async def _gen(**kwargs):
        return {"response": "canned"}

    env = _FakeEnv(terminate_after=2)
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_FakeAdapter([[_computer_call(), _bash_call()], [_wait_call()]]),
    )

    rl = await agent.sample(env, max_steps=10)
    obs = _obs_between_assistants(rl.lite_sample.messages)
    tool_msgs = [m for m in obs if m.get("role") == "tool"]

    assert len(rl.lite_sample.images) == 3
    assert tool_msgs[0]["tool_call_id"] == "call_0000"
    assert tool_msgs[0]["content"] == [{"type": "image", "index": 1}]
    assert tool_msgs[1]["tool_call_id"] == "call_0001"
    assert tool_msgs[1]["content"] == [{"type": "text", "text": "bash ok"}]
    terminal_tool_msg = rl.lite_sample.messages[-1]
    assert terminal_tool_msg["tool_call_id"] == "call_0002"
    assert terminal_tool_msg["content"] == [{"type": "image", "index": 2}]


async def test_loop_projects_tool_error_with_current_observation_into_next_turn():
    env = _ErrorObservationEnv(terminate_after=99)
    messages = await _run_loop_with_custom_env(
        [[_computer_call()], "finished"],
        env,
    )

    obs = _obs_between_assistants(messages)
    assert len(obs) == 1
    (tool_msg,) = obs
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_0000"
    assert tool_msg["content"][0] == {"type": "image", "index": 1}
    assert tool_msg["content"][1] == {
        "type": "text",
        "text": ("## AXTree:\nbody\n\n## Error from previous action:\ninvalid action: mouse_move"),
    }
    assert tool_msg["content"][2] == {"type": "metadata", "data": {"is_error": True}}


async def test_pairable_bad_executable_args_are_role_tool_feedback_not_empty():
    async def _gen(**kwargs):
        return {"response": "canned", "response_tokens": [101]}

    env = _PairableBadExecutableArgsEnv(terminate_after=99)
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_FakeAdapter([[_bad_wait_duration_call()], "finished"]),
    )

    rl = await agent.sample(env, max_steps=3)
    obs = _obs_between_assistants(rl.lite_sample.messages)

    assert [step.status for step in rl.steps] == ["completed", "completed"]
    assert rl.steps[0].response_tokens == [101]
    assert [m["role"] for m in obs] == ["tool"]
    tool_msg = obs[0]
    assert tool_msg["tool_call_id"] == "call_0000"
    assert tool_msg["content"][0] == {"type": "image", "index": 1}
    assert tool_msg["content"][1] == {
        "type": "text",
        "text": (
            "## Error from previous action:\n"
            "invalid arguments for wait: wait.duration must be a finite number"
        ),
    }
    assert tool_msg["content"][2] == {"type": "metadata", "data": {"is_error": True}}
    assert env.seen_actions[0] == [
        make_tool_call("wait", {"duration": "forever"}, call_id="call_0000")
    ]


async def test_loop_next_prompt_after_route_known_error_has_fresh_carrier_once():
    captured: list[dict[str, Any]] = []

    async def _gen(**kwargs):
        captured.append(
            {
                "messages": kwargs["messages"],
                "images": kwargs["images"],
            }
        )
        return {"response": "canned"}

    env = _FreshRouteKnownErrorEnv(terminate_after=99)
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_FakeAdapter([[_computer_call()], "finished"]),
    )

    await agent.sample(env, max_steps=3)

    assert len(captured) >= 2
    second = captured[1]
    tool_msgs = [message for message in second["messages"] if message.get("role") == "tool"]
    assert len(tool_msgs) == 1
    tool_msg = tool_msgs[0]
    assert tool_msg["tool_call_id"] == "call_0000"
    assert tool_msg["content"][0] == {"type": "image", "index": 1}
    assert [image.size for image in second["images"]] == [(4, 4), (6, 5)]

    texts = [
        part["text"]
        for message in second["messages"]
        for part in message.get("content", [])
        if part.get("type") == "text"
    ]
    joined = "\n".join(texts)
    assert joined.count("## Error from previous action:") == 1
    assert joined.count("invalid action: screenshot") == 1
    assert (
        "## AXTree:\nbutton Search\n\n## Error from previous action:\ninvalid action: screenshot"
    ) in joined


async def test_logger_persists_exact_online_tool_error_projection(tmp_path):
    async def _gen(**kwargs):
        return {"response": "canned"}

    env = _FreshRouteKnownErrorEnv(terminate_after=99)
    logger = TrajectoryLogger(tmp_path / "run")
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_FakeAdapter([[_computer_call()], "finished"]),
        preserve_raw_response=False,
    )

    rl = await agent.sample(env, max_steps=3, hooks=[logger])

    row = pd.read_parquet(tmp_path / "run" / "trajectory.parquet").iloc[0]
    persisted = [_normalize_message(message) for message in coerce_messages(row["messages"])]
    online = [_normalize_message(message) for message in rl.lite_sample.messages]

    assert persisted == online
    tool_message = next(message for message in persisted if message.get("role") == "tool")
    assert tool_message["content"] == [
        {"type": "image", "index": 1},
        {
            "type": "text",
            "text": (
                "## AXTree:\nbutton Search\n\n"
                "## Error from previous action:\ninvalid action: screenshot"
            ),
        },
        {"type": "metadata", "data": {"is_error": True}},
    ]


async def test_logger_persists_exact_online_text_only_tool_error_projection(tmp_path):
    async def _gen(**kwargs):
        return {"response": "canned"}

    env = _TextOnlyRouteKnownErrorEnv(terminate_after=99)
    logger = TrajectoryLogger(tmp_path / "run")
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_FakeAdapter([[_computer_call()], "finished"]),
        preserve_raw_response=False,
    )

    rl = await agent.sample(env, max_steps=3, hooks=[logger])

    row = pd.read_parquet(tmp_path / "run" / "trajectory.parquet").iloc[0]
    persisted = [_normalize_message(message) for message in coerce_messages(row["messages"])]
    online = [_normalize_message(message) for message in rl.lite_sample.messages]

    assert persisted == online
    tool_message = next(message for message in persisted if message.get("role") == "tool")
    assert tool_message["content"] == [
        {
            "type": "text",
            "text": (
                "stdout before failure\n\n## Error from previous action:\nunsupported action: bogus"
            ),
        },
        {"type": "metadata", "data": {"is_error": True}},
    ]


async def test_loop_role_tool_results_pair_to_previous_assistant_calls():
    messages = await _run_loop([[_computer_call(), _bash_call()], [_wait_call()]])
    _assert_tool_results_pair_with_previous_assistant(messages)
    assert any(m.get("role") == "tool" for m in messages)


async def test_loop_orders_env_results_by_previous_tool_calls():
    env = _ScriptedResultsEnv(["call_0001", "call_0000"], terminate_after=2)
    messages = await _run_loop_with_custom_env(
        [[_computer_call(), _bash_call()], [_wait_call()]],
        env,
    )

    tool_msgs = [m for m in _obs_between_assistants(messages) if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_0000", "call_0001"]
    assert [m["content"][0]["text"] for m in tool_msgs] == [
        "result call_0000",
        "result call_0001",
    ]


async def test_loop_hooks_receive_results_aligned_to_previous_tool_calls():
    async def _gen(**kwargs):
        return {"response": "canned"}

    env = _ScriptedResultsEnv(["call_0001", "call_0000"], terminate_after=2)
    hook = _CaptureStepResults()
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_FakeAdapter([[_computer_call(), _bash_call()], [_wait_call()]]),
    )

    await agent.sample(env, max_steps=10, hooks=[hook])

    assert hook.result_call_ids[0] == ["call_0000", "call_0001"]


@pytest.mark.parametrize(
    ("result_call_ids", "match"),
    [
        (["call_0000"], "missing"),
        (["call_0000", "call_0000"], "duplicate"),
        (["call_0000", "call_0001", "call_orphan"], "orphan"),
    ],
)
async def test_loop_rejects_mispaired_env_results(
    result_call_ids: list[str],
    match: str,
):
    env = _ScriptedResultsEnv(result_call_ids, terminate_after=2)
    with pytest.raises(RuntimeError, match=match):
        await _run_loop_with_custom_env(
            [[_computer_call(), _bash_call()], [_wait_call()]],
            env,
        )


async def test_loop_rejects_missing_env_results_for_nonterminal_calls():
    env = _NoResultsEnv(terminate_after=2)
    with pytest.raises(RuntimeError, match="missing"):
        await _run_loop_with_custom_env(
            [[_computer_call(), _bash_call()], [_wait_call()]],
            env,
        )


async def test_loop_allows_env_terminal_tool_call_without_fake_tool_result():
    env = _NoResultsEnv(terminate_after=1)

    messages = await _run_loop_with_custom_env(
        [
            [
                make_tool_call(
                    "custom_terminal",
                    {"value": "done"},
                    call_id="call_custom_terminal",
                )
            ]
        ],
        env,
    )

    assert _call_name(env.seen_actions[0][0]) == "custom_terminal"
    assert [message.get("role") for message in messages] == ["user", "assistant"]


async def test_loop_accepts_result_for_inactive_terminal_shaped_call():
    env = _ScriptedResultsEnv(["call_0000"], terminate_after=2)
    messages = await _run_loop_with_custom_env([[_terminate_call()], [_wait_call()]], env)

    tool_msg = next(m for m in messages if m.get("role") == "tool")
    assert tool_msg["tool_call_id"] == "call_0000"


async def test_loop_mixed_action_and_inactive_terminal_requires_both_results():
    env = _ScriptedResultsEnv(["call_0000", "call_0001"], terminate_after=2)
    messages = await _run_loop_with_custom_env(
        [[_computer_call(), _terminate_call()], [_wait_call()]],
        env,
    )

    assert [_call_name(a) for a in env.seen_actions[0]] == ["computer", "terminate"]

    obs = _obs_between_assistants(messages)
    assert [m["role"] for m in obs] == ["tool", "tool"]
    assert [m["tool_call_id"] for m in obs] == ["call_0000", "call_0001"]


async def test_loop_single_action_turn_characterization():
    # SACRED single-GUI baseline: a single-click turn builds exactly one
    # paired role:tool result carrying the screenshot.
    messages = await _run_loop([[_computer_call()], [_wait_call()]])
    obs = _obs_between_assistants(messages)

    assert len(obs) == 1, f"single-GUI turn must build exactly one observation, got {obs}"
    (msg,) = obs
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_0000"
    kinds = [p.get("type") for p in msg.get("content", [])]
    assert "image" in kinds, f"observation must carry the screenshot, got {msg}"


async def test_loop_action_batch_result_stores_all_images_and_next_turn_uses_final():
    calls: list[dict[str, Any]] = []

    async def _gen(**kwargs):
        calls.append(kwargs)
        return {"response": "canned"}

    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_FakeAdapter([[_computer_call(None)], [_wait_call()]]),
    )

    rl = await agent.sample(_MultiImageFirstStepEnv(terminate_after=2), max_steps=10)

    tool_msg = next(
        message
        for message in rl.lite_sample.messages
        if message.get("tool_call_id") == "call_0000"
    )
    assert tool_msg["content"] == [{"type": "image", "index": 2}]
    assert len(rl.processed_images) == len(rl.lite_sample.images)
    assert rl.processed_images[1] is None
    assert rl.processed_images[2] is not None
    assert len(calls) == 2
    second_turn_messages = calls[1]["messages"]
    assert referenced_image_indices_in_message_order(second_turn_messages) == (0, 2)
    assert calls[1]["images"] == [rl.processed_images[0], rl.processed_images[2]]


@pytest.mark.parametrize(
    ("adapter", "tool_name", "first_raw", "expected_call", "expected_actions"),
    [
        (
            Qwen3_5DesktopUseAdapter(),
            "computer_use",
            _qwen3_5_xml_tool_call(
                "computer_use",
                action="left_click",
                coordinate="[12, 34]",
            )
            + _qwen3_5_xml_tool_call(
                "computer_use",
                action="type",
                text="hello",
            ),
            "computer",
            ["click", "type"],
        ),
        (
            Qwen3_5MobileUseAdapter(),
            "mobile_use",
            _qwen3_5_xml_tool_call(
                "mobile_use",
                action="click",
                coordinate="[12, 34]",
            )
            + _qwen3_5_xml_tool_call(
                "mobile_use",
                action="type",
                text="hello",
            ),
            "mobile",
            ["tap", "type"],
        ),
    ],
)
async def test_qwen3_5_adjacent_xml_action_batch_stores_all_images_and_next_turn_uses_final(
    adapter,
    tool_name: str,
    first_raw: str,
    expected_call: str,
    expected_actions: list[str],
):
    calls: list[dict[str, Any]] = []

    async def _gen(**kwargs):
        calls.append(kwargs)
        return {"response": first_raw if len(calls) == 1 else "Done."}

    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=adapter,
    )

    rl = await agent.sample(_MultiImageFirstStepEnv(terminate_after=2), max_steps=10)

    assert len(calls) == 2
    first_actions = rl.lite_sample.messages[1]["tool_calls"]
    assert [tool_call_name(call) for call in first_actions] == [expected_call]
    assert [
        action["action"]
        for action in tool_call_arguments(first_actions[0])["actions"]
    ] == expected_actions
    assert tool_name in first_raw

    assert len(rl.lite_sample.images) == 3
    assert len(rl.processed_images) == 3
    assert rl.processed_images[1] is None
    assert rl.processed_images[2] is not None

    tool_msg = next(
        message
        for message in rl.lite_sample.messages
        if message.get("tool_call_id") == "call_0000"
    )
    assert tool_msg["content"] == [{"type": "image", "index": 2}]

    second_turn_messages = calls[1]["messages"]
    assert referenced_image_indices_in_message_order(second_turn_messages) == (0, 2)
    assert calls[1]["images"] == [rl.processed_images[0], rl.processed_images[2]]


async def test_loop_persists_terminal_step_tool_result_before_return():
    messages, env = await _run_loop_with_env([[_computer_call()]], terminate_after=1)

    assert len(env.seen_actions) == 1
    assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
    tool_msg = messages[-1]
    assert tool_msg["tool_call_id"] == "call_0000"
    assert tool_msg["content"] == [{"type": "image", "index": 1}]


async def test_loop_does_not_persist_env_terminal_empty_results():
    env = _NoResultsEnv(terminate_after=1)
    messages = await _run_loop_with_custom_env([[_computer_call()]], env)

    assert len(env.seen_actions) == 1
    assert [m["role"] for m in messages] == ["user", "assistant"]


async def test_loop_marks_max_steps_exhaustion_truncated_with_paired_tool_result():
    async def _gen(**kwargs):
        return {"response": "canned"}

    env = _FakeEnv(terminate_after=99)
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_FakeAdapter([[_computer_call()]]),
    )

    rl = await agent.sample(env, max_steps=1)
    messages = rl.lite_sample.messages

    assert rl.terminated is False
    assert rl.truncated is True
    assert rl.steps[-1].status == "truncated"
    assert len(env.seen_actions) == 1
    assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
    tool_msg = messages[-1]
    assert tool_msg["tool_call_id"] == "call_0000"
    assert tool_msg["content"] == [{"type": "image", "index": 1}]


async def test_empty_no_call_no_text_is_a_content_only_final_not_a_failure():
    """A zero-tool-call turn still reaches env response scoring.

    Empty visible text is a content-only response attempt; only malformed output
    is forced terminal/failure by the loop.
    """

    async def _gen(**kwargs):
        return {"response": "canned"}

    env = _FakeEnv(terminate_after=99)
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_FakeAdapter([[]]),
    )

    rl = await agent.sample(env, max_steps=10)
    messages = rl.lite_sample.messages

    assert messages[0]["role"] == "user"
    # The env IS stepped, with the runtime response(text="") action.
    assert len(env.seen_actions) == 1
    assert [_call_name(a) for a in env.seen_actions[0]] == ["response"]
    assert _call_args(env.seen_actions[0][0]) == {"text": ""}
    assert rl.terminated is True
    assert rl.truncated is False
    # The synthesized response action is runtime-only and must NOT be persisted:
    # the trajectory keeps the raw assistant turn, not a fabricated tool call.
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert not any(m.get("role") == "tool" for m in messages)
    assert not messages[-1].get("tool_calls")


async def test_parse_error_without_tool_calls_is_terminal_response_not_noop():
    async def _gen(**kwargs):
        return {"response": "malformed tool block"}

    env = _FakeEnv(terminate_after=1)
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_ParseErrorThenActionAdapter(),
    )

    rl = await agent.sample(env, max_steps=2)
    messages = rl.lite_sample.messages

    assert len(env.seen_actions) == 1
    finish = env.seen_actions[0][0]
    assert _call_name(finish) == "response"
    assert _call_args(finish) == {"text": "malformed tool block"}
    assert not any(_call_name(action) == "noop" for turn in env.seen_actions for action in turn)
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert rl.terminated is True
    assert rl.truncated is False


async def test_content_only_final_stops_without_persisted_tool_call():
    messages, env = await _run_loop_with_env(["Done."], terminate_after=99)

    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0]["content"] == [{"type": "text", "text": "Done."}]
    assert "tool_calls" not in assistant_msgs[0]

    assert len(env.seen_actions) == 1
    finish = env.seen_actions[0][0]
    # N3/N4: content-only final is always response(text=...).
    assert _call_name(finish) == "response"
    assert "id" not in finish
    assert finish[RUNTIME_INTERNAL_STOP_REASON_KEY] == CONTENT_ONLY_FINAL_REASON
    assert _call_args(finish) == {"text": "Done."}
    assert not any(m.get("role") == "tool" for m in messages)


async def test_content_only_final_follows_env_terminal_flags():
    async def _gen(**kwargs):
        return {"response": "canned"}

    env = _FakeEnv(terminate_after=99)
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_FakeAdapter(["First answer", "Second answer"]),
    )

    rl = await agent.sample(env, max_steps=10)
    messages = rl.lite_sample.messages

    assert rl.terminated is True
    assert rl.truncated is False
    assert len(env.seen_actions) == 1
    assert _call_name(env.seen_actions[0][0]) == "response"
    assert _call_args(env.seen_actions[0][0]) == {"text": "First answer"}
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert all("Second answer" not in repr(message) for message in messages)
    assert not any(m.get("role") == "tool" for m in messages)


async def test_content_only_response_attempt_can_continue_when_env_returns_feedback():
    async def _gen(**kwargs):
        return {"response": "canned"}

    env = _ContentOnlyFeedbackEnv(max_turns=2)
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_FakeAdapter(["First answer", "Second answer"]),
    )

    rl = await agent.sample(env, max_steps=10)
    messages = rl.lite_sample.messages

    assert rl.terminated is True
    assert rl.truncated is False
    assert len(env.seen_actions) == 2
    assert [_call_args(turn[0]) for turn in env.seen_actions] == [
        {"text": "First answer"},
        {"text": "Second answer"},
    ]
    assert [m["role"] for m in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert messages[2]["content"] == [
        {"type": "text", "text": "attempt 1: First answer"}
    ]
    assert not any(m.get("role") == "tool" for m in messages)


async def test_content_only_response_attempt_rejects_empty_unrenderable_feedback():
    async def _gen(**kwargs):
        return {"response": "canned"}

    env = _EmptyContentOnlyFeedbackEnv()
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_FakeAdapter(["First answer", "Second answer"]),
    )

    with pytest.raises(RuntimeError, match="no model-visible feedback"):
        await agent.sample(env, max_steps=10)

    assert len(env.seen_actions) == 1
    assert _call_args(env.seen_actions[0][0]) == {"text": "First answer"}


async def test_content_only_response_attempt_rejects_paired_feedback():
    async def _gen(**kwargs):
        return {"response": "canned"}

    env = _PairedContentOnlyFeedbackEnv()
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_FakeAdapter(["First answer", "Second answer"]),
    )

    with pytest.raises(RuntimeError, match="paired tool results"):
        await agent.sample(env, max_steps=10)

    assert len(env.seen_actions) == 1
    assert _call_args(env.seen_actions[0][0]) == {"text": "First answer"}


async def test_content_only_response_attempt_truncates_at_agent_max_steps():
    async def _gen(**kwargs):
        return {"response": "canned"}

    env = _ContentOnlyFeedbackEnv(max_turns=2)
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_FakeAdapter(["First answer", "Second answer"]),
    )

    rl = await agent.sample(env, max_steps=1)

    assert rl.terminated is False
    assert rl.truncated is True
    assert len(env.seen_actions) == 1
    assert [m["role"] for m in rl.lite_sample.messages] == ["user", "assistant"]
    assert all("attempt 1" not in repr(message) for message in rl.lite_sample.messages)
    assert rl.steps[-1].status == STATUS_TRUNCATED


async def test_content_only_response_attempt_can_terminate_on_first_env_attempt():
    async def _gen(**kwargs):
        return {"response": "canned"}

    env = _ContentOnlyFeedbackEnv(max_turns=1)
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_FakeAdapter(["Only answer", "Second answer"]),
    )

    rl = await agent.sample(env, max_steps=10)

    assert rl.terminated is True
    assert rl.truncated is False
    assert len(env.seen_actions) == 1
    assert [m["role"] for m in rl.lite_sample.messages] == ["user", "assistant"]
    assert all("Second answer" not in repr(message) for message in rl.lite_sample.messages)


async def test_parse_failure_stays_terminal_even_when_env_would_return_feedback():
    async def _gen(**kwargs):
        return {"response": "malformed tool block"}

    env = _ContentOnlyFeedbackEnv(max_turns=3)
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_ParseErrorThenActionAdapter(),
    )

    rl = await agent.sample(env, max_steps=10)

    assert rl.terminated is True
    assert rl.truncated is False
    assert rl.steps[-1].status == STATUS_FAILED
    assert len(env.seen_actions) == 1
    assert [m["role"] for m in rl.lite_sample.messages] == ["user", "assistant"]


async def test_provider_lifecycle_rejects_nonterminal_response_without_feedback():
    async def _unused_terminal_feedback(**kwargs):
        raise AssertionError("terminal feedback appender should not run")

    with pytest.raises(RuntimeError, match="no feedback after non-terminal response"):
        await record_lite_env_result(
            trajectory=LiteSample(metadata=LiteCUAMetadata()),
            persisted_actions=[],
            step_result=LiteEnvStepResult(
                results=[],
                reward=0.0,
                terminated=False,
                truncated=False,
            ),
            rl_step=LiteRLStep(prompt="", image_indices=(), response="answer"),
            step=0,
            max_steps=10,
            episode_return=0.0,
            append_terminal_tool_feedback=_unused_terminal_feedback,
        )


async def test_provider_lifecycle_rejects_nonterminal_response_with_unrenderable_feedback():
    async def _unused_terminal_feedback(**kwargs):
        raise AssertionError("terminal feedback appender should not run")

    with pytest.raises(RuntimeError, match="no model-visible feedback"):
        await record_lite_env_result(
            trajectory=LiteSample(metadata=LiteCUAMetadata()),
            persisted_actions=[],
            step_result=LiteEnvStepResult(
                results=[LiteToolResult(tool_call_id=None, text="")],
                reward=0.0,
                terminated=False,
                truncated=False,
            ),
            rl_step=LiteRLStep(prompt="", image_indices=(), response="answer"),
            step=0,
            max_steps=10,
            episode_return=0.0,
            append_terminal_tool_feedback=_unused_terminal_feedback,
        )


async def test_provider_lifecycle_rejects_nonterminal_response_with_paired_feedback():
    async def _unused_terminal_feedback(**kwargs):
        raise AssertionError("terminal feedback appender should not run")

    with pytest.raises(RuntimeError, match="paired tool results"):
        await record_lite_env_result(
            trajectory=LiteSample(metadata=LiteCUAMetadata()),
            persisted_actions=[],
            step_result=LiteEnvStepResult(
                results=[LiteToolResult(tool_call_id="call_orphan", text="revise")],
                reward=0.0,
                terminated=False,
                truncated=False,
            ),
            rl_step=LiteRLStep(prompt="", image_indices=(), response="answer"),
            step=0,
            max_steps=10,
            episode_return=0.0,
            append_terminal_tool_feedback=_unused_terminal_feedback,
        )


async def test_provider_lifecycle_allows_empty_feedback_at_agent_max_steps():
    async def _unused_terminal_feedback(**kwargs):
        raise AssertionError("terminal feedback appender should not run")

    rl_step = LiteRLStep(prompt="", image_indices=(), response="answer")
    step_result = LiteEnvStepResult(
        results=[],
        reward=0.0,
        terminated=False,
        truncated=False,
    )

    episode_return, terminated, truncated = await record_lite_env_result(
        trajectory=LiteSample(metadata=LiteCUAMetadata()),
        persisted_actions=[],
        step_result=step_result,
        rl_step=rl_step,
        step=0,
        max_steps=1,
        episode_return=0.0,
        append_terminal_tool_feedback=_unused_terminal_feedback,
    )

    assert episode_return == 0.0
    assert terminated is False
    assert truncated is True
    assert step_result.truncated is True
    assert rl_step.status == STATUS_TRUNCATED


async def test_provider_lifecycle_still_requires_tool_results_at_agent_max_steps():
    async def _terminal_feedback(**kwargs):
        raise AssertionError("terminal feedback appender should not run after alignment failure")

    with pytest.raises(RuntimeError, match="missing=\\['call_0000'\\]"):
        await record_lite_env_result(
            trajectory=LiteSample(metadata=LiteCUAMetadata()),
            persisted_actions=[make_tool_call("computer", call_id="call_0000")],
            step_result=LiteEnvStepResult(
                results=[],
                reward=0.0,
                terminated=False,
                truncated=False,
            ),
            rl_step=LiteRLStep(prompt="", image_indices=(), response="action"),
            step=0,
            max_steps=1,
            episode_return=0.0,
            append_terminal_tool_feedback=_terminal_feedback,
        )


async def test_content_only_final_persists_canonical_text_without_fake_tool_call():
    raw_text = "Done.\nFinal answer."
    env = _FakeEnv(terminate_after=99)
    rl = await _run_raw_text_loop_with_env(raw_text, env)
    messages = rl.lite_sample.messages

    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    assert len(assistant_msgs) == 1
    assistant = assistant_msgs[0]
    assert assistant["content"] == [{"type": "text", "text": raw_text}]
    assert "tool_calls" not in assistant
    assert "raw_response" not in assistant

    assert len(env.seen_actions) == 1
    finish = env.seen_actions[0][0]
    # N3/N4: content-only final is always response(text=...).
    assert _call_name(finish) == "response"
    assert "id" not in finish
    assert finish[RUNTIME_INTERNAL_STOP_REASON_KEY] == CONTENT_ONLY_FINAL_REASON
    assert _call_args(finish) == {"text": raw_text}
    assert rl.terminated is True
    assert rl.episode_return == 1.0
    assert not any(m.get("role") == "tool" for m in messages)


async def test_content_only_final_hooks_receive_executed_runtime_action():
    async def _gen(**kwargs):
        return {"response": "Final answer"}

    env = _FakeEnv(terminate_after=99)
    hook = _CaptureStepActions()
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_RawTextAdapter(),
    )

    rl = await agent.sample(env, max_steps=3, hooks=[hook])

    assert hook.actions == env.seen_actions
    assert len(hook.actions) == 1
    finish = hook.actions[0][0]
    assert _call_name(finish) == "response"
    assert _call_args(finish) == {"text": "Final answer"}
    assert hook.infos == [{
        "executed_actions": [{"call": "response", "args": {"text": "Final answer"}}],
        "stop_reason": CONTENT_ONLY_FINAL_REASON,
    }]
    assert rl.lite_sample.messages[-1].get("tool_calls") is None


async def test_content_only_final_uses_runtime_response_when_enabled():
    response_schema = LiteFinishToolSet.get_tool_schema("response")
    assert response_schema is not None
    env = _FakeEnv(terminate_after=99)
    env.metadata = LiteCUAMetadata(extra_tool_schemas=[response_schema])

    rl = await _run_raw_text_loop_with_env("Final answer", env)
    assistant = next(m for m in rl.lite_sample.messages if m.get("role") == "assistant")

    assert assistant["content"] == [{"type": "text", "text": "Final answer"}]
    assert "raw_response" not in assistant
    assert "tool_calls" not in assistant
    assert len(env.seen_actions) == 1
    finish = env.seen_actions[0][0]
    assert _call_name(finish) == "response"
    assert "id" not in finish
    assert finish[RUNTIME_INTERNAL_STOP_REASON_KEY] == CONTENT_ONLY_FINAL_REASON
    assert _call_args(finish) == {"text": "Final answer"}
    assert not any(m.get("role") == "tool" for m in rl.lite_sample.messages)
    assert rl.episode_return == 1.0


async def test_direct_predict_no_tool_final_does_not_attach_raw_response():
    async def _gen(**kwargs):
        return {"response": "Final answer"}

    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_RawTextAdapter(),
        preserve_raw_response=True,
    )
    sample = LiteSample(
        metadata=LiteCUAMetadata(),
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    )

    message = await agent.predict(sample)

    assert message["content"] == [{"type": "text", "text": "Final answer"}]
    assert not message.get("tool_calls")
    assert "raw_response" not in message


async def test_direct_predict_parse_failure_does_not_attach_raw_response():
    raw_response = "Action:\n<tool_call>{bad json}</tool_call>"

    async def _gen(**kwargs):
        return {"response": raw_response}

    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_ParseErrorThenActionAdapter(),
        preserve_raw_response=True,
    )
    sample = LiteSample(
        metadata=LiteCUAMetadata(),
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    )

    result = await agent._predict_with_details(sample, processed_images=[])

    assert result.model_output_error == "malformed <tool_call> JSON"
    assert result.lite_message["content"] == [{"type": "text", "text": raw_response}]
    assert not result.lite_message.get("tool_calls")
    assert "raw_response" not in result.lite_message
    assert "_lite_model_output_error" not in result.lite_message


async def test_agent_side_parse_error_marker_is_preserved_as_parse_failure():
    raw_response = "Action:\n<tool_call>{bad json}</tool_call>"

    async def _gen(**kwargs):
        return {"response": raw_response}

    env = _FakeEnv(terminate_after=1)
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_AgentOnlyParseErrorMarkerAdapter(),
    )

    rl = await agent.sample(env, max_steps=2)
    messages = rl.lite_sample.messages

    assert len(env.seen_actions) == 1
    finish = env.seen_actions[0][0]
    assert _call_name(finish) == "response"
    assert _call_args(finish) == {"text": raw_response}
    assert rl.terminated is True
    assert rl.truncated is False

    assistant = messages[1]
    assert assistant["content"] == [{"type": "text", "text": raw_response}]
    assert MODEL_OUTPUT_ERROR_KEY not in assistant
    assert assistant[CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY]["stop_reason"] == (
        PARSE_FAILURE_FINAL_REASON
    )
    assert assistant[CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY]["has_model_output_error"] is True


async def test_parse_failure_is_terminal_response_instead_of_noop():
    async def _gen(**kwargs):
        return {"response": "Action:\n<tool_call>{bad json}</tool_call>"}

    env = _FakeEnv(terminate_after=1)
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_ParseErrorThenActionAdapter(),
    )

    rl = await agent.sample(env, max_steps=3)
    messages = rl.lite_sample.messages

    assert len(env.seen_actions) == 1
    finish = env.seen_actions[0][0]
    assert _call_name(finish) == "response"
    assert _call_args(finish) == {"text": "Action:\n<tool_call>{bad json}</tool_call>"}
    assert rl.terminated is True
    assert rl.truncated is False

    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == [
        {"type": "text", "text": "Action:\n<tool_call>{bad json}</tool_call>"}
    ]
    assert "raw_response" not in messages[1]
    assert "_lite_model_output_error" not in messages[1]
    assert messages[1][CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY]["stop_reason"] == (
        PARSE_FAILURE_FINAL_REASON
    )
    assert messages[1][CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY]["has_model_output_error"] is True


@pytest.mark.parametrize(
    ("finish_reason", "status"),
    [("length", STATUS_TRUNCATED), ("stop", STATUS_FAILED)],
    ids=["cut off at the token budget", "malformed on a complete turn"],
)
async def test_parse_failure_does_not_clobber_a_truncated_step(finish_reason, status):
    """The truncation is WHY the tool call did not parse — keep the cause.

    The parse-failure relabel used to overwrite ``STATUS_TRUNCATED`` (derived
    from ``finish_reason`` in ``_predict_with_details``) with ``STATUS_FAILED``,
    so a step cut off at the token budget was reported as a model-output
    failure and the rollout's ``truncated`` metric never counted it.
    """
    async def _gen(**kwargs):
        return {
            "response": "Action:\n<tool_call>{bad json}</tool_call>",
            "finish_reason": finish_reason,
        }

    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_ParseErrorThenActionAdapter(),
    )

    rl = await agent.sample(_FakeEnv(terminate_after=1), max_steps=3)

    assert [s.status for s in rl.steps] == [status]


async def test_malformed_tool_calls_container_shape_stays_loud():
    async def _gen(**kwargs):
        return {"response": "malformed canonical tool_calls"}

    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_MalformedToolCallsThenActionAdapter(),
    )

    with pytest.raises(TypeError, match="list of dict calls"):
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)


async def test_stamp_validation_failure_is_model_output_error():
    async def _gen(**kwargs):
        return {"response": "canned"}

    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_FakeAdapter([[make_tool_call("computer", call_id="")]]),
    )
    sample = LiteSample(
        metadata=LiteCUAMetadata(),
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    )

    result = await agent._predict_with_details(sample, processed_images=[])

    assert result.model_output_error == (
        "tool_calls[0].id must be a non-empty string when present"
    )
    assert result.lite_message == {
        "role": "assistant",
        "content": [{"type": "text", "text": "canned"}],
    }
    assert "raw_response" not in result.lite_message


async def test_parsed_tool_calls_dropped_by_adapter_is_terminal_response():
    async def _gen(**kwargs):
        return {"response": "wrapped open_app"}

    env = _FakeEnv(terminate_after=1)
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_DroppedParsedToolCallThenActionAdapter(),
    )

    rl = await agent.sample(env, max_steps=2)
    messages = rl.lite_sample.messages

    assert len(env.seen_actions) == 1
    finish = env.seen_actions[0][0]
    assert _call_name(finish) == "response"
    assert _call_args(finish) == {"text": "wrapped open_app"}
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert rl.terminated is True
    assert rl.truncated is False


async def test_shared_loop_stamps_missing_tool_call_ids_before_persisting():
    messages, env = await _run_loop_with_env(
        [[_computer_call(None), _bash_call(None)]],
        terminate_after=1,
    )

    assistant = next(m for m in messages if m.get("role") == "assistant")
    assert [_call_id(tc) for tc in assistant["tool_calls"]] == ["call_0000", "call_0001"]
    assert [_call_id(tc) for tc in env.seen_actions[0]] == ["call_0000", "call_0001"]


async def test_shared_loop_invalid_present_call_id_is_terminal_response():
    messages, env = await _run_loop_with_env(
        [[make_tool_call("computer", call_id="")]],
        terminate_after=1,
    )

    assert len(env.seen_actions) == 1
    assert env.seen_actions[0] == [_internal_response_call("canned")]
    assistant = next(m for m in messages if m.get("role") == "assistant")
    assert not assistant.get("tool_calls")
    assert assistant["content"] == [{"type": "text", "text": "canned"}]
    assert assistant[CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY]["stop_reason"] == (
        PARSE_FAILURE_FINAL_REASON
    )
    assert assistant[CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY]["has_model_output_error"] is True
    assert all(m.get("role") != "tool" for m in messages)


@pytest.mark.parametrize(
    "tool_calls",
    [
        [
            {
                "id": "call_0000",
                "type": "function",
                "function": {"name": "computer", "arguments": {}},
                "_result_call_id": "reserved",
            }
        ],
        [
            {
                "id": "provider_1",
                "type": "function",
                "function": {"name": "computer", "arguments": "{}"},
            }
        ],
    ],
)
async def test_shared_loop_unpairable_tool_call_shapes_are_terminal_response(
    tool_calls: list[LiteToolCall],
):
    messages, env = await _run_loop_with_env([tool_calls], terminate_after=1)

    assert len(env.seen_actions) == 1
    assert env.seen_actions[0] == [_internal_response_call("canned")]
    assistant = next(m for m in messages if m.get("role") == "assistant")
    assert not assistant.get("tool_calls")
    assert assistant["content"] == [{"type": "text", "text": "canned"}]
    assert assistant[CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY]["stop_reason"] == (
        PARSE_FAILURE_FINAL_REASON
    )
    assert all(m.get("role") != "tool" for m in messages)


async def test_shared_loop_restamps_duplicate_model_emitted_call_ids():
    """Lite owns call ids: model-emitted duplicates are restamped, not rejected."""
    messages, env = await _run_loop_with_env(
        [
            [
                make_tool_call("computer", call_id="dup"),
                make_tool_call("bash", {"command": "pwd"}, call_id="dup"),
            ]
        ],
        terminate_after=1,
    )

    assistant = next(m for m in messages if m.get("role") == "assistant")
    assert [_call_id(tc) for tc in assistant["tool_calls"]] == ["call_0000", "call_0001"]
    assert [_call_id(a) for a in env.seen_actions[0]] == ["call_0000", "call_0001"]
    assert [m["tool_call_id"] for m in messages if m.get("role") == "tool"] == [
        "call_0000",
        "call_0001",
    ]


async def test_shared_loop_call_ids_accumulate_across_turns():
    messages, env = await _run_loop_with_env(
        [[_computer_call(None)], [_bash_call(None)]],
        terminate_after=2,
    )

    assistant_calls = [
        msg["tool_calls"]
        for msg in messages
        if msg.get("role") == "assistant" and msg.get("tool_calls")
    ]
    assert [_call_id(tc) for tc in assistant_calls[0]] == ["call_0000"]
    assert [_call_id(tc) for tc in assistant_calls[1]] == ["call_0001"]
    assert [_call_id(turn[0]) for turn in env.seen_actions] == ["call_0000", "call_0001"]


# ---------------------------------------------------------------------------
# Bespoke-loop (GPT / Claude) canonical-write tests
# ---------------------------------------------------------------------------
#
# The GPT/Claude bespoke sample() loops build the API-side conversation AND a
# canonical ``trajectory.messages`` LiteSample. The API side already writes
# per-call tool results (Claude) / *_call_output items (GPT); it is the CANONICAL
# ``trajectory.messages`` observation write must remain per-call role:tool.
#
async def test_gpt_bespoke_loop_writes_role_tool(monkeypatch):
    if not _BESPOKE_IMPORT_OK:
        pytest.skip("gpt/claude agents unavailable (optional deps missing)")
    from unittest.mock import AsyncMock

    _stub_gpt_echoed_dim_fetch(monkeypatch)
    provider_tool_turn = _gpt_response(
        [
            {
                "type": "computer_call",
                "call_id": "provider_computer_1",
                "actions": [{"type": "screenshot"}],
            }
        ]
    )
    monkeypatch.setattr(
        "litellm.aresponses",
        AsyncMock(side_effect=[provider_tool_turn, _gpt_response(_GPT_TEXT_FINAL_OUTPUT)]),
    )

    env = _desktop_env(terminate_after=99)
    rl = await GPTDesktopUseAgent(model_id="gpt-5.5").sample(env, max_steps=3)
    obs = _obs_between_assistants(rl.lite_sample.messages)

    assert [m["role"] for m in obs] == ["tool"]
    tool_msg = obs[0]
    assert tool_msg["tool_call_id"] == "call_0000"
    assert tool_msg["content"] == [{"type": "image", "index": 1}]
    assert env.seen_actions[0] == [
        make_tool_call(
            "computer",
            {"actions": [{"action": "screenshot"}]},
            call_id="call_0000",
        )
    ]


async def test_claude_bespoke_loop_writes_role_tool(monkeypatch):
    if not _BESPOKE_IMPORT_OK:
        pytest.skip("gpt/claude agents unavailable (optional deps missing)")
    from unittest.mock import AsyncMock

    provider_tool_turn = _claude_response(
        "",
        tool_calls=[
            _provider_tool_call(
                "computer",
                '{"action": "screenshot"}',
                id_="provider_computer_1",
            )
        ],
    )
    monkeypatch.setattr(
        "litellm.acompletion",
        AsyncMock(side_effect=[provider_tool_turn, _claude_response(_CLAUDE_TEXT_FINAL_CONTENT)]),
    )

    env = _desktop_env(terminate_after=99)
    rl = await ClaudeDesktopUseAgent().sample(env, max_steps=3)
    obs = _obs_between_assistants(rl.lite_sample.messages)

    assert [m["role"] for m in obs] == ["tool"]
    tool_msg = obs[0]
    assert tool_msg["tool_call_id"] == "call_0000"
    assert tool_msg["content"] == [{"type": "image", "index": 1}]
    assert env.seen_actions[0] == [
        make_tool_call(
            "computer",
            {"actions": [{"action": "screenshot"}]},
            call_id="call_0000",
        )
    ]


# ---------------------------------------------------------------------------
# info.tool_results per-call lookup
# ---------------------------------------------------------------------------


def test_info_tool_results_prefers_per_call_text():
    from lite.agents.core.agent.utils.tool_results import tool_result_text_for_call_id

    step_result = LiteEnvStepResult(
        results=[LiteToolResult(tool_call_id="call_0000", text="per-call stdout")],
    )
    assert (
        tool_result_text_for_call_id(
            step_result,
            call_id="call_0000",
            default="ok",
        )
        == "per-call stdout"
    )


def test_info_tool_results_projects_per_call_error_with_observation_text():
    from lite.agents.core.agent.utils.tool_results import tool_result_text_for_call_id

    step_result = LiteEnvStepResult(
        results=[
            LiteToolResult(
                tool_call_id="call_0000",
                text="## AXTree:\nbody",
                error="unsupported action: bogus",
            )
        ],
    )
    expected = "## AXTree:\nbody\n\n## Error from previous action:\nunsupported action: bogus"

    assert (
        tool_result_text_for_call_id(
            step_result,
            call_id="call_0000",
            default="ok",
        )
        == expected
    )


def test_latest_step_feedback_preserves_image_text_and_metadata():
    from lite.agents.core.agent.utils.tool_results import latest_step_feedback

    image = _png_bytes()
    metadata = {"is_error": True, "source": "env"}
    step_result = LiteEnvStepResult(
        results=[
            LiteToolResult(
                tool_call_id=None,
                images=[image],
                text="## AXTree:\nbody",
                error="invalid action: mouse_move",
                metadata=metadata,
            )
        ],
    )
    expected_text = "## AXTree:\nbody\n\n## Error from previous action:\ninvalid action: mouse_move"

    assert latest_step_feedback(step_result) == (image, expected_text, metadata)


def test_bespoke_role_tool_writer_projects_text_error_metadata_parity():
    image = _png_bytes()
    step_result = LiteEnvStepResult(
        results=[
            LiteToolResult(
                tool_call_id="call_0000",
                images=[image],
                text="## AXTree:\nbody",
                error="invalid action: mouse_move",
                metadata={"is_error": True},
            ),
        ],
    )
    expected = [
        {
            "role": "tool",
            "tool_call_id": "call_0000",
            "content": [
                {"type": "image", "index": 7},
                {
                    "type": "text",
                    "text": (
                        "## AXTree:\nbody\n\n"
                        "## Error from previous action:\ninvalid action: mouse_move"
                    ),
                },
                {"type": "metadata", "data": {"is_error": True}},
            ],
        }
    ]

    [result] = step_result.results
    assert [
        build_tool_result_message(
            result.tool_call_id,
            (7,),
            result.text,
            result.metadata,
            error=result.error,
        )
    ] == expected


def test_bespoke_role_tool_writer_uses_per_call_result_carrier():
    messages = [
        build_tool_result_message("call_0000", (7,), None, None),
        build_tool_result_message("call_0001", (), "stdout", None),
    ]

    assert [m["tool_call_id"] for m in messages] == ["call_0000", "call_0001"]
    assert messages[0]["content"] == [{"type": "image", "index": 7}]
    assert messages[1]["content"] == [{"type": "text", "text": "stdout"}]


async def test_loop_terminal_action_batch_result_stores_all_images_and_message_uses_final():
    async def _gen(**kwargs):
        return {"response": "canned"}

    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_FakeAdapter(
            [
                [
                    make_tool_call(
                        "computer",
                        {"actions": [{"action": "click", "coordinate": [10, 20]}]},
                        call_id=None,
                    )
                ]
            ]
        ),
    )

    rl = await agent.sample(
        _MultiImageFirstStepEnv(terminate_after=1),
        max_steps=10,
    )

    assert len(rl.lite_sample.images) == 3
    assert len(rl.processed_images) == 3
    assert rl.processed_images[1] is None
    assert rl.processed_images[2] is not None
    assert rl.lite_sample.messages == [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "do the task"},
            ],
        },
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [10, 20]}]},
                    call_id="call_0000",
                )
            ],
            "raw_response": {"adapter_key": "fake@desktop@use", "text": "canned"},
        },
        {
            "role": "tool",
            "tool_call_id": "call_0000",
            "content": [
                {"type": "image", "index": 2},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Text-oriented response attempts — ONE definition across ALL three loops
# ---------------------------------------------------------------------------
#
# Owner's contract:
#   * a turn that emits ONLY text (any text) and no tool call becomes an
#     internal response(text=...) attempt, for EVERY agent — the shared
#     ``AdapterBasedAgent`` loop, the GPT loop, and the Claude loop;
#   * envs decide whether that response attempt is terminal; non-terminal env
#     feedback is appended as role:user and the loop continues;
#   * pure text that is the residue of a failed tool-call parse is also terminal
#     under N3 when there is no executable tool call, but it is marked with a
#     parse-failure stop reason instead of a clean content-only reason.
#
# The classifier is :func:`no_tool_call_final_text` (``type == "text"`` parts
# only). ``action_description`` is excluded by construction: it is the
# narration-accompanying-an-action channel and is only ever produced for a turn
# that HAS tool calls.


class _RejectEmptyActionsEnv(_FakeEnv):
    """``env.step([])`` is never a legal way to score a content-only final."""

    async def step(self, actions: list[LiteToolCall]) -> LiteEnvStepResult:
        if not actions:
            raise AssertionError("content-only final must not call env.step([])")
        return await super().step(actions)


def _qwen_use_adapter():
    """A REAL open-source ``@use`` adapter (``qwen3_vl@desktop@use``)."""
    from lite.agents.models.qwen3_vl.adapter import Qwen3VLDesktopUseAdapter

    return Qwen3VLDesktopUseAdapter()


class _QwenFakeEnv(_RejectEmptyActionsEnv):
    """Screenshots big enough for the qwen adapter's ``smart_resize``."""

    async def reset(self) -> LiteEnvObservation:
        return LiteEnvObservation(image=_png_bytes(256, 256), text=self._instruction)

    async def step(self, actions: list[LiteToolCall]) -> LiteEnvStepResult:
        result = await super().step(actions)
        for tool_result in result.results:
            if tool_result.images:
                tool_result.images = [_png_bytes(256, 256)]
        return result


async def _run_open_source_use_loop(response: str, env: _FakeEnv, *, max_steps: int = 2):
    async def _gen(**kwargs):
        return {"response": response}

    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_qwen_use_adapter(),
    )
    return await agent.sample(env, max_steps=max_steps)


# --- classifier unit contract ---------------------------------------------


_TEXT_FINAL_CASES: list[tuple[str, dict[str, Any], str]] = [
    ("plain text", {"content": [{"type": "text", "text": "Done."}]}, "Done."),
    (
        "text is whitespace padded",
        {"content": [{"type": "text", "text": "  18 x 24  "}]},
        "18 x 24",
    ),
    (
        "several text parts join",
        {
            "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ]
        },
        "first\nsecond",
    ),
    ("empty output", {"content": []}, ""),
    (
        "reasoning-only output is not a final",
        {"content": [{"type": "inline_reasoning", "text": "hmm"}]},
        "",
    ),
    (
        "native reasoning-only output is not a final",
        {"content": [], "reasoning_content": "thinking..."},
        "",
    ),
    (
        "action narration is not a final",
        {"content": [{"type": "action_description", "text": "click Save"}]},
        "",
    ),
    (
        "history summary is not a final",
        {"content": [{"type": "history_summary", "text": "so far..."}]},
        "",
    ),
    (
        "narration alongside real text yields only the text",
        {
            "content": [
                {"type": "action_description", "text": "click Save"},
                {"type": "text", "text": "Saved."},
            ]
        },
        "Saved.",
    ),
]


@pytest.mark.parametrize(
    ("label", "content", "expected"),
    _TEXT_FINAL_CASES,
    ids=[case[0] for case in _TEXT_FINAL_CASES],
)
def test_all_three_loops_share_one_text_final_definition(label, content, expected):
    """The SAME assistant message must classify identically in every loop.

    Enforced structurally (the adapter loop, provider-native lifecycle helper,
    and GPT/Claude agent modules call the one classifier) and behaviourally
    (the classifier's verdict is pinned per message shape).
    """
    from lite.agents.core.agent import base as shared_loop
    from lite.agents.core.agent.utils import loop as provider_lifecycle_loop
    from lite.agents.core.agent.utils.final import begin_no_tool_call_final
    from lite.core.messages.final import no_tool_call_final_text

    message = {"role": "assistant", **content}
    assert no_tool_call_final_text(message) == expected

    modules = [shared_loop, provider_lifecycle_loop]
    if _BESPOKE_IMPORT_OK:
        from lite.agents.models.claude import agent as claude_agent
        from lite.agents.models.claude.utils import loop as claude_loop
        from lite.agents.models.gpt import agent as gpt_agent
        from lite.agents.models.gpt.utils import loop as gpt_loop

        assert gpt_agent.execute_lite_turn is provider_lifecycle_loop.execute_lite_turn
        assert claude_agent.execute_lite_turn is provider_lifecycle_loop.execute_lite_turn
        assert not hasattr(gpt_loop, "_execute_lite_turn")
        assert not hasattr(claude_loop, "_execute_lite_turn")
        for module in (gpt_loop, claude_loop):
            assert not hasattr(module, "begin_no_tool_call_final")
            assert not hasattr(module, "no_tool_call_final_text")

    for module in modules:
        # The lifecycle owners resolve the same classifier function object and
        # do not keep the raw text definition they used to re-derive the final
        # from.
        assert module.begin_no_tool_call_final is begin_no_tool_call_final
        assert not hasattr(module, "no_tool_call_final_text")

    # ...and the one classifier puts exactly that text on the wire.
    final = begin_no_tool_call_final(
        message,
        model_output_error=None,
        step=LiteRLStep(prompt="", image_indices=(), response=""),
    )
    assert _call_args(final.actions[0])["text"] == expected


def test_no_loop_keeps_a_private_text_final_definition():
    if not _BESPOKE_IMPORT_OK:
        pytest.skip("gpt/claude agents unavailable (optional deps missing)")
    from lite.agents.models.claude import agent as claude_agent
    from lite.agents.models.gpt import agent as gpt_agent

    for module in (claude_agent, gpt_agent):
        assert not hasattr(module, "_text_from_lite_message"), (
            f"{module.__name__} re-introduced a private text-final definition"
        )


def test_action_description_only_final_roundtrips_after_canonicalization():
    adapter = _qwen_use_adapter()
    source = {
        "role": "assistant",
        "content": [{"type": "action_description", "text": "click Save"}],
    }

    canonical = canonicalize_no_tool_call_final_message(source)
    diagnostic = canonical[CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY]
    assert diagnostic["stop_reason"] == EMPTY_FINAL_REASON
    assert diagnostic["action_description_part_count"] == 1

    rendered = adapter.convert_message_to_agent(canonical)
    roundtrip = adapter.convert_message_from_agent(rendered)

    assert rendered["content"] == [{"type": "text", "text": CONTENT_ONLY_FINAL_TEXT}]
    assert roundtrip["content"] == [{"type": "text", "text": CONTENT_ONLY_FINAL_TEXT}]
    assert not roundtrip.get("tool_calls")


# --- shared loop (AdapterBasedAgent) on a real open-source @use adapter -----


async def test_shared_loop_open_source_use_adapter_text_final_terminates():
    """REGRESSION (defect A): the ``@use`` adapter used to retag ALL assistant
    text as ``action_description``, so a genuine pure-text final read as empty
    model output and the episode was truncated instead of terminated."""
    env = _QwenFakeEnv(terminate_after=99)
    rl = await _run_open_source_use_loop("The answer is 42.", env)

    assistant_msgs = [m for m in rl.lite_sample.messages if m.get("role") == "assistant"]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0]["content"] == [{"type": "text", "text": "The answer is 42."}]
    assert "raw_response" not in assistant_msgs[0]
    assert not assistant_msgs[0].get("tool_calls")

    # Terminal evaluation still ran: exactly one env.step, with the runtime-only
    # response action carrying the original answer text (never persisted, never
    # ``env.step([])``).
    assert len(env.seen_actions) == 1
    (finish,) = env.seen_actions[0]
    # N3/N4: content-only final is always response(text=...).
    assert _call_name(finish) == "response"
    assert "id" not in finish
    assert finish[RUNTIME_INTERNAL_STOP_REASON_KEY] == CONTENT_ONLY_FINAL_REASON
    assert _call_args(finish) == {"text": "The answer is 42."}

    assert rl.terminated is True
    assert rl.truncated is False
    assert rl.episode_return == 1.0
    assert not any(m.get("role") == "tool" for m in rl.lite_sample.messages)


async def test_shared_loop_open_source_use_adapter_parse_failure_is_terminal():
    """A tool call that failed to parse has no executable call, so N3 treats the
    raw model output as the terminal response instead of retrying."""
    env = _QwenFakeEnv(terminate_after=99)
    rl = await _run_open_source_use_loop(
        "Action: click Save\n<tool_call>{not json}</tool_call>",
        env,
    )

    assert len(env.seen_actions) == 1
    (finish,) = env.seen_actions[0]
    assert _call_name(finish) == "response"
    assert "Action: click Save" in _call_args(finish)["text"]
    assert rl.terminated is True
    assert rl.truncated is False


async def test_shared_loop_open_source_use_adapter_action_turn_still_narrates():
    """The ``action_description`` channel is intact for turns that DO act — the
    defect-A fix must not silently change SFT/unroll rendering of ACTION turns."""
    adapter = _qwen_use_adapter()
    parsed = adapter.parse_raw_assistant_response(
        'Action: click Save\n<tool_call>{"name": "computer_use", "arguments": '
        '{"action": "left_click", "coordinate": [10, 20]}}</tool_call>'
    )
    lite = adapter.convert_message_from_agent(parsed)

    assert lite["content"] == [{"type": "action_description", "text": "click Save"}]
    assert lite["tool_calls"]
    # …and it round-trips back to the ``Action:`` wire line.
    rendered = adapter.convert_message_to_agent(lite)
    assert rendered["content"] == [{"type": "text", "text": "Action: click Save"}]


def test_canonical_content_only_final_ignores_stale_raw_response_on_replay():
    adapter = _qwen_use_adapter()
    lite = {
        "role": "assistant",
        "content": [{"type": "text", "text": "Done."}],
        "_lite_content_only_final": {
            "version": 1,
            "stop_reason": "text",
            "content_types": ["text"],
        },
        "raw_response": {
            "adapter_key": adapter.get_registry_key(),
            "text": "The stale raw answer should not replay.",
        },
    }

    rendered = adapter.convert_message_to_agent(lite)
    assert rendered["content"] == [{"type": "text", "text": "Done."}]

    roundtrip = adapter.convert_message_from_agent(rendered)
    assert roundtrip["content"] == [{"type": "text", "text": "Done."}]
    assert not roundtrip.get("tool_calls")
    assert "raw_response" not in roundtrip


# --- GPT / Claude bespoke loops --------------------------------------------
#
# Driven hermetically by patching the provider entry point
# (``litellm.aresponses`` / ``litellm.acompletion``) — no network.


def _stub_gpt_echoed_dim_fetch(
    monkeypatch,
    dims: tuple[int, int] = (800, 600),
) -> None:
    """Keep the GPT loops hermetic.

    ``_call_api_with_actual_dim`` (wired into both GPT loops) follows the
    provider call with a GET for the dims the API echoed back, so mocking only
    ``litellm.aresponses`` still leaves one outbound request.
    """
    from unittest.mock import AsyncMock

    monkeypatch.setattr(
        "lite.agents.models.gpt.utils.image_io._fetch_processed_image_dims",
        AsyncMock(return_value=[dims]),
    )


def _stub_bespoke_gpt_echoed_dim_fetch(monkeypatch, which: str) -> None:
    if which == "gpt_desktop":
        _stub_gpt_echoed_dim_fetch(monkeypatch, (800, 600))
    elif which == "gpt_mobile":
        _stub_gpt_echoed_dim_fetch(monkeypatch, (540, 1200))


def _gpt_response(
    output: list[dict[str, Any]], incomplete_reason: str | None = None
) -> dict[str, Any]:
    resp: dict[str, Any] = {"output": output, "id": "resp_test", "usage": {}}
    if incomplete_reason is not None:
        # The Responses API's spelling of a chat-completions ``finish_reason``:
        # ``status="incomplete"`` plus ``incomplete_details.reason``.
        resp["status"] = "incomplete"
        resp["incomplete_details"] = {"reason": incomplete_reason}
    return resp


_GPT_TEXT_FINAL_OUTPUT = [
    {"type": "message", "content": [{"type": "output_text", "text": "  The answer is 42.  "}]},
]
# Text AND a tool call that failed to parse, in one turn: N3's terminal
# response path must win because there is no executable tool call.
_GPT_PARSE_FAILURE_OUTPUT = [
    {"type": "message", "content": [{"type": "output_text", "text": "The answer is 42."}]},
    {"type": "function_call", "name": "bogus_tool", "arguments": "{}", "call_id": "fc_1"},
]


def _claude_response(
    content: Any,
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content, tool_calls=tool_calls or [], role="assistant"
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        model_dump=lambda: {"choices": []},
    )


def _provider_tool_call(name: str, arguments: str, *, id_: str) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        model_dump=lambda: {
            "id": id_,
            "function": {"name": name, "arguments": arguments},
        }
    )


# NB: unpadded — the Claude loop persists the provider text verbatim (only the
# runtime response action's ``text`` argument is stripped by the shared classifier.
_CLAUDE_TEXT_FINAL_CONTENT = "The answer is 42."
_CLAUDE_PARSE_FAILURE_CONTENT = [
    {"type": "text", "text": "The answer is 42."},
    {"type": "tool_use", "id": "tu_1", "name": "bogus_tool", "input": {}},
]


def _cut_off_provider_tool_call(name: str, arguments: str) -> Any:
    """A provider tool call whose argument JSON stops mid-object.

    The real truncation shape: the budget expired while the provider was
    streaming ``input_json_delta``, so ``arguments`` is a prefix that no parser
    can accept.
    """
    return _provider_tool_call(name, arguments, id_="tc_cut_off")


# O1: the provider reports a BUDGET stop *and* the partial tool-call JSON is cut
# mid-object, so the parse fails. The truncation is WHY it did not parse — the
# zero-call relabel must keep the cause, not overwrite it with the symptom.
_BESPOKE_TRUNCATED_PARSE_FAILURE = {
    "gpt_desktop": lambda: _gpt_response(
        [
            {"type": "message", "content": [{"type": "output_text", "text": "Clicking"}]},
            {
                "type": "function_call",
                "name": "click",
                "arguments": '{"coordinate": [12',
                "call_id": "fc_cut_off",
            },
        ],
        incomplete_reason="max_output_tokens",
    ),
    "gpt_mobile": lambda: _gpt_response(
        [
            {"type": "message", "content": [{"type": "output_text", "text": "Tapping"}]},
            {
                "type": "function_call",
                "name": "tap",
                "arguments": '{"coordinate": [12',
                "call_id": "fc_cut_off",
            },
        ],
        incomplete_reason="max_output_tokens",
    ),
    "claude_desktop": lambda: _claude_response(
        [{"type": "text", "text": "Clicking"}],
        tool_calls=[
            _cut_off_provider_tool_call(
                "computer", '{"action": "left_click", "coordinate": [12'
            )
        ],
        finish_reason="length",
    ),
    "claude_mobile": lambda: _claude_response(
        [{"type": "text", "text": "Tapping"}],
        tool_calls=[_cut_off_provider_tool_call("tap", '{"coordinate": [12')],
        finish_reason="length",
    ),
}


class _BespokeFakeEnv(_RejectEmptyActionsEnv):
    """Provider-loop env: needs a declared resolution + real-sized screenshots."""

    def __init__(self, platform: LiteCUAMetadata.Platform, size: tuple[int, int], **kwargs):
        super().__init__(**kwargs)
        self.metadata = LiteCUAMetadata(
            dims=(platform, LiteCUAMetadata.TaskType.USE),
            others={"resolution": list(size)},
        )
        self._size = size

    async def reset(self) -> LiteEnvObservation:
        return LiteEnvObservation(image=_png_bytes(*self._size), text=self._instruction)

    async def step(self, actions: list[LiteToolCall]) -> LiteEnvStepResult:
        result = await super().step(actions)
        for tool_result in result.results:
            if tool_result.images:
                tool_result.images = [_png_bytes(*self._size)]
        return result


def _desktop_env(**kwargs) -> _BespokeFakeEnv:
    return _BespokeFakeEnv(LiteCUAMetadata.Platform.DESKTOP, (800, 600), **kwargs)


def _mobile_env(**kwargs) -> _BespokeFakeEnv:
    return _BespokeFakeEnv(LiteCUAMetadata.Platform.MOBILE, (540, 1200), **kwargs)


def _make_bespoke_agent(which: str):
    from lite.agents.models.claude.agent import ClaudeMobileUseAgent
    from lite.agents.models.gpt.agent import GPTMobileUseAgent

    return {
        "gpt_desktop": lambda: GPTDesktopUseAgent(model_id="gpt-5.5"),
        "gpt_mobile": lambda: GPTMobileUseAgent(model_id="gpt-5.5"),
        "claude_desktop": lambda: ClaudeDesktopUseAgent(),
        "claude_mobile": lambda: ClaudeMobileUseAgent(),
    }[which]()


_BESPOKE_LOOPS = {
    # which          → (env factory, litellm entry point, text-final payload,
    #                   parse-failure payload)
    "gpt_desktop": (
        _desktop_env,
        "litellm.aresponses",
        lambda: _gpt_response(_GPT_TEXT_FINAL_OUTPUT),
        lambda: _gpt_response(_GPT_PARSE_FAILURE_OUTPUT),
    ),
    "gpt_mobile": (
        _mobile_env,
        "litellm.aresponses",
        lambda: _gpt_response(_GPT_TEXT_FINAL_OUTPUT),
        lambda: _gpt_response(_GPT_PARSE_FAILURE_OUTPUT),
    ),
    "claude_desktop": (
        _desktop_env,
        "litellm.acompletion",
        lambda: _claude_response(_CLAUDE_TEXT_FINAL_CONTENT),
        lambda: _claude_response(_CLAUDE_PARSE_FAILURE_CONTENT),
    ),
    "claude_mobile": (
        _mobile_env,
        "litellm.acompletion",
        lambda: _claude_response(_CLAUDE_TEXT_FINAL_CONTENT),
        lambda: _claude_response(_CLAUDE_PARSE_FAILURE_CONTENT),
    ),
}


@pytest.mark.parametrize("which", sorted(_BESPOKE_LOOPS))
async def test_bespoke_loop_text_final_terminates(which, monkeypatch):
    if not _BESPOKE_IMPORT_OK:
        pytest.skip("gpt/claude agents unavailable (optional deps missing)")
    from unittest.mock import AsyncMock

    env_factory, entry_point, text_final, _ = _BESPOKE_LOOPS[which]
    _stub_bespoke_gpt_echoed_dim_fetch(monkeypatch, which)
    monkeypatch.setattr(entry_point, AsyncMock(return_value=text_final()))

    env = env_factory(terminate_after=99)
    rl = await _make_bespoke_agent(which).sample(env, max_steps=2)

    assistant = next(m for m in rl.lite_sample.messages if m.get("role") == "assistant")
    expected_text = "  The answer is 42.  " if which.startswith("gpt_") else "The answer is 42."
    assert assistant["content"] == [{"type": "text", "text": expected_text}]
    assert "raw_response" not in assistant
    assert not assistant.get("tool_calls")

    assert len(env.seen_actions) == 1
    (finish,) = env.seen_actions[0]
    # N3/N4: content-only final is always response(text=...).
    assert _call_name(finish) == "response"
    assert "id" not in finish
    assert finish[RUNTIME_INTERNAL_STOP_REASON_KEY] == CONTENT_ONLY_FINAL_REASON
    assert _call_args(finish) == {"text": "The answer is 42."}

    assert rl.terminated is True
    assert rl.truncated is False
    assert rl.episode_return == 1.0


# --- hook_actions vs persisted_actions on the provider loops ----------------
#
# ``SampleStepData.actions`` means the actions actually passed to ``env.step()``
# (``hook_actions``). The Lite sample stores only the assistant's own tool calls
# (``persisted_actions``). The two are the SAME list for an ordinary tool turn
# and deliberately differ for a zero-tool-call final, where the env executes a
# synthetic ``response(text=...)`` that must never become an assistant call.


@pytest.mark.parametrize("which", sorted(_BESPOKE_LOOPS))
async def test_bespoke_no_tool_final_hook_actions_are_not_persisted(which, monkeypatch):
    """No-tool final: hooks see the executed action, the sample persists none."""
    if not _BESPOKE_IMPORT_OK:
        pytest.skip("gpt/claude agents unavailable (optional deps missing)")
    from unittest.mock import AsyncMock

    env_factory, entry_point, text_final, _ = _BESPOKE_LOOPS[which]
    _stub_bespoke_gpt_echoed_dim_fetch(monkeypatch, which)
    monkeypatch.setattr(entry_point, AsyncMock(return_value=text_final()))

    env = env_factory(terminate_after=99)
    hook = _CaptureStepActions()
    rl = await _make_bespoke_agent(which).sample(env, max_steps=2, hooks=[hook])

    # hook_actions == what the env executed, never an empty list.
    assert hook.actions == env.seen_actions
    assert len(hook.actions) == 1
    (finish,) = hook.actions[0]
    assert _call_name(finish) == "response"
    assert finish[RUNTIME_INTERNAL_STOP_REASON_KEY] == CONTENT_ONLY_FINAL_REASON

    # persisted_actions is empty, so no unpaired role:"tool" message is written.
    assistant = next(m for m in rl.lite_sample.messages if m.get("role") == "assistant")
    assert not assistant.get("tool_calls")
    assert not any(m.get("role") == "tool" for m in rl.lite_sample.messages)


# One real provider tool call on the terminal step: the complementary case where
# hook_actions, persisted_actions, and the assistant's tool calls are all one
# and the same list.
_BESPOKE_TERMINAL_TOOL_TURN = {
    "gpt_desktop": lambda: _gpt_response(
        [
            {
                "type": "computer_call",
                "call_id": "provider_computer_1",
                "actions": [{"type": "screenshot"}],
            }
        ]
    ),
    "gpt_mobile": lambda: _gpt_response(
        [
            {
                "type": "function_call",
                "name": "tap",
                "arguments": '{"x": 120, "y": 240}',
                "call_id": "fc_tap_1",
            }
        ]
    ),
    "claude_desktop": lambda: _claude_response(
        "",
        tool_calls=[
            _provider_tool_call("computer", '{"action": "screenshot"}', id_="provider_computer_1")
        ],
    ),
    "claude_mobile": lambda: _claude_response(
        "",
        tool_calls=[_provider_tool_call("tap", '{"coordinate": [120, 240]}', id_="tu_tap_1")],
    ),
}


@pytest.mark.parametrize("which", sorted(_BESPOKE_TERMINAL_TOOL_TURN))
async def test_bespoke_terminal_tool_call_hook_and_persisted_actions_match(which, monkeypatch):
    """Tool turn: hook_actions == persisted_actions == the assistant's calls."""
    if not _BESPOKE_IMPORT_OK:
        pytest.skip("gpt/claude agents unavailable (optional deps missing)")
    from unittest.mock import AsyncMock

    env_factory, entry_point, _, _ = _BESPOKE_LOOPS[which]
    _stub_bespoke_gpt_echoed_dim_fetch(monkeypatch, which)
    monkeypatch.setattr(entry_point, AsyncMock(return_value=_BESPOKE_TERMINAL_TOOL_TURN[which]()))

    # ``terminate_after=1`` makes the very first tool turn the terminal one.
    env = env_factory(terminate_after=1)
    hook = _CaptureStepActions()
    rl = await _make_bespoke_agent(which).sample(env, max_steps=3, hooks=[hook])

    assistant = next(m for m in rl.lite_sample.messages if m.get("role") == "assistant")
    assistant_tool_calls = assistant["tool_calls"]
    assert assistant_tool_calls
    assert hook.actions == env.seen_actions == [assistant_tool_calls]

    # Terminal feedback is still persisted per call, paired by tool_call_id.
    tool_msgs = [m for m in rl.lite_sample.messages if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == [
        tool_call_id(call) for call in assistant_tool_calls
    ]
    assert rl.terminated is True


class _EnvTruncatesWithoutResults(_BespokeFakeEnv):
    """Env that ends the episode by TRUNCATION on its first step.

    Env truncation is an external cut-off, not a natural end and not a model
    failure, so the env may stop before it can report per-call results.
    """

    async def step(self, actions: list[LiteToolCall]) -> LiteEnvStepResult:
        self.seen_actions.append(list(actions))
        self._steps += 1
        return LiteEnvStepResult(reward=0.0, terminated=False, truncated=True, results=[])


@pytest.mark.parametrize("which", sorted(_BESPOKE_TERMINAL_TOOL_TURN))
async def test_bespoke_env_truncation_ends_the_episode_without_tool_results(which, monkeypatch):
    """``step_result.truncated`` alone ends the episode and relaxes result pairing.

    ``max_steps`` is deliberately far from exhaustion, so the truncation can
    only come from the env. Because the step is terminal, the env is allowed to
    omit per-call results: a non-terminal step missing them would instead fail
    loudly in ``align_tool_results_to_tool_calls``.
    """
    if not _BESPOKE_IMPORT_OK:
        pytest.skip("gpt/claude agents unavailable (optional deps missing)")
    from unittest.mock import AsyncMock

    _, entry_point, _, _ = _BESPOKE_LOOPS[which]
    _stub_bespoke_gpt_echoed_dim_fetch(monkeypatch, which)
    monkeypatch.setattr(entry_point, AsyncMock(return_value=_BESPOKE_TERMINAL_TOOL_TURN[which]()))

    if which.endswith("_mobile"):
        env = _EnvTruncatesWithoutResults(LiteCUAMetadata.Platform.MOBILE, (540, 1200))
    else:
        env = _EnvTruncatesWithoutResults(LiteCUAMetadata.Platform.DESKTOP, (800, 600))

    rl = await _make_bespoke_agent(which).sample(env, max_steps=5)

    assert rl.truncated is True
    assert rl.terminated is False
    assert [step.status for step in rl.steps] == [STATUS_TRUNCATED]

    # The tool turn still ran, and the assistant's calls are what the env saw.
    assistant = next(m for m in rl.lite_sample.messages if m.get("role") == "assistant")
    assert assistant["tool_calls"]
    assert env.seen_actions == [assistant["tool_calls"]]

    # No result was reported, so the trajectory ends on the assistant tool call.
    assert [m for m in rl.lite_sample.messages if m.get("role") == "tool"] == []


@pytest.mark.parametrize("which", sorted(_BESPOKE_LOOPS))
async def test_bespoke_loop_runs_blocking_hooks_off_the_event_loop(which, monkeypatch):
    """Provider loops must offload hooks like the adapter-backed loop does."""
    if not _BESPOKE_IMPORT_OK:
        pytest.skip("gpt/claude agents unavailable (optional deps missing)")
    from unittest.mock import AsyncMock

    env_factory, entry_point, text_final, _ = _BESPOKE_LOOPS[which]
    _stub_bespoke_gpt_echoed_dim_fetch(monkeypatch, which)
    monkeypatch.setattr(entry_point, AsyncMock(return_value=text_final()))

    env = env_factory(terminate_after=99)
    hook = _CaptureHookThreads()
    await _make_bespoke_agent(which).sample(env, max_steps=2, hooks=[hook])

    loop_thread = threading.get_ident()
    assert hook.step_threads and all(t != loop_thread for t in hook.step_threads)
    assert hook.complete_threads and all(t != loop_thread for t in hook.complete_threads)


async def test_shared_loop_runs_blocking_hooks_off_the_event_loop():
    """The adapter-backed loop's behaviour the provider loops must match."""

    async def _gen(**kwargs):
        return {"response": "Final answer"}

    hook = _CaptureHookThreads()
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_RawTextAdapter(),
    )
    await agent.sample(_FakeEnv(terminate_after=99), max_steps=3, hooks=[hook])

    loop_thread = threading.get_ident()
    assert hook.step_threads and all(t != loop_thread for t in hook.step_threads)
    assert hook.complete_threads and all(t != loop_thread for t in hook.complete_threads)


def test_adapter_loop_calls_the_shared_lifecycle_owner():
    """``AdapterBasedAgent.sample()`` calls the owner instead of re-inlining it.

    The adapter loop lives in the same package as the lifecycle owner, so the
    two lifecycle steps it shares with the provider-native loops — the
    truncated-last-step stamp and hook-then-close finalization — must come from
    ``lite.agents.core.agent.utils.loop``, not from a second copy.
    """
    import inspect

    from lite.agents.core.agent import base as agent_base
    from lite.agents.core.agent.utils import loop as provider_lifecycle_loop

    assert agent_base.mark_steps_truncated is provider_lifecycle_loop.mark_steps_truncated
    assert agent_base.finalize_lite_sample is provider_lifecycle_loop.finalize_lite_sample

    source = inspect.getsource(agent_base.AdapterBasedAgent.sample)
    assert "mark_steps_truncated(" in source
    assert "finalize_lite_sample(" in source
    # The re-copied bodies: an inline env teardown and an inline status stamp.
    assert "env.close()" not in source
    assert "on_complete" not in source


def test_provider_lifecycle_owner_keeps_only_lifecycle_symbols():
    """The provider-independent loop owner exposes lifecycle steps, not shims."""
    import inspect

    from lite.agents.core.agent.utils import loop as provider_lifecycle_loop

    # Ordinary env-result persistence is a plain provider call, not a named
    # protocol, and the env step is part of ``execute_lite_turn`` itself.
    assert not hasattr(provider_lifecycle_loop, "TerminalToolFeedbackAppender")
    assert not hasattr(provider_lifecycle_loop, "step_env")
    # Provider usage/log summaries belong to the provider, not the finalizer.
    assert (
        "before_complete"
        not in inspect.signature(provider_lifecycle_loop.finalize_lite_sample).parameters
    )
    assert set(provider_lifecycle_loop.__all__) == {
        "ExecutedLiteTurn",
        "append_lite_rl_step",
        "build_lite_rl_sample",
        "dispatch_lite_step_hooks",
        "execute_lite_turn",
        "finalize_lite_sample",
        "mark_steps_truncated",
        "record_lite_env_result",
    }

    if not _BESPOKE_IMPORT_OK:
        pytest.skip("gpt/claude agents unavailable (optional deps missing)")
    from lite.agents.models.claude import agent as claude_agent
    from lite.agents.models.gpt import agent as gpt_agent

    for module in (gpt_agent, claude_agent):
        # Provider loops call the lifecycle owner directly; no per-family
        # ``_record_env_result`` forwarder in between.
        assert not hasattr(module, "_record_env_result")
        assert module.record_lite_env_result is provider_lifecycle_loop.record_lite_env_result


@pytest.mark.parametrize("which", sorted(_BESPOKE_LOOPS))
async def test_bespoke_loop_parse_failure_is_terminal_response(which, monkeypatch):
    """GPT/Claude bespoke loops must match the shared N3 zero-call path."""
    if not _BESPOKE_IMPORT_OK:
        pytest.skip("gpt/claude agents unavailable (optional deps missing)")
    from unittest.mock import AsyncMock

    env_factory, entry_point, _, parse_failure = _BESPOKE_LOOPS[which]
    _stub_bespoke_gpt_echoed_dim_fetch(monkeypatch, which)
    monkeypatch.setattr(entry_point, AsyncMock(return_value=parse_failure()))

    env = env_factory(terminate_after=99)
    rl = await _make_bespoke_agent(which).sample(env, max_steps=2)

    assert len(env.seen_actions) == 1
    (finish,) = env.seen_actions[0]
    assert _call_name(finish) == "response"
    assert _call_args(finish) == {"text": "The answer is 42."}
    assert "id" not in finish
    assert finish[RUNTIME_INTERNAL_STOP_REASON_KEY] == CONTENT_ONLY_FINAL_REASON
    assert rl.terminated is True
    assert rl.truncated is False

    assistant = next(m for m in rl.lite_sample.messages if m.get("role") == "assistant")
    assert "raw_response" not in assistant
    assert "_lite_model_output_error" not in assistant
    assert assistant[CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY]["stop_reason"] == (
        PARSE_FAILURE_FINAL_REASON
    )


@pytest.mark.parametrize("which", sorted(_BESPOKE_TRUNCATED_PARSE_FAILURE))
async def test_bespoke_loop_parse_failure_does_not_clobber_a_truncated_step(
    which, monkeypatch
):
    """O1: each bespoke loop must keep the cause, like the shared loop does.

    All four provider loops derive ``STATUS_TRUNCATED`` from the provider's own
    budget stop signal (``finish_reason="length"`` /
    ``incomplete_details.reason="max_output_tokens"``) and then used to overwrite
    it with ``STATUS_FAILED`` on the zero-call relabel, so a budget-truncated
    turn read downstream as a model parse failure — ``segmenter.py`` takes max
    severity (``failed`` beats ``truncated``) and ``engine.py`` drops the turn on
    ``status != FAILED``. Same assertion as
    ``test_parse_failure_does_not_clobber_a_truncated_step`` on the shared
    adapter loop.
    """
    if not _BESPOKE_IMPORT_OK:
        pytest.skip("gpt/claude agents unavailable (optional deps missing)")
    from unittest.mock import AsyncMock

    env_factory, entry_point, _, _ = _BESPOKE_LOOPS[which]
    _stub_bespoke_gpt_echoed_dim_fetch(monkeypatch, which)
    truncated_parse_failure = _BESPOKE_TRUNCATED_PARSE_FAILURE[which]
    monkeypatch.setattr(entry_point, AsyncMock(return_value=truncated_parse_failure()))

    env = env_factory(terminate_after=99)
    rl = await _make_bespoke_agent(which).sample(env, max_steps=2)

    # The turn IS still a terminal zero-call final — only the recorded cause
    # changes.
    assert len(env.seen_actions) == 1
    (finish,) = env.seen_actions[0]
    assert _call_name(finish) == "response"
    assert finish[RUNTIME_INTERNAL_STOP_REASON_KEY] == CONTENT_ONLY_FINAL_REASON
    assistant = next(m for m in rl.lite_sample.messages if m.get("role") == "assistant")
    assert assistant[CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY]["stop_reason"] == (
        PARSE_FAILURE_FINAL_REASON
    )

    assert [s.status for s in rl.steps] == [STATUS_TRUNCATED]


@pytest.mark.parametrize("which", sorted(_BESPOKE_TRUNCATED_PARSE_FAILURE))
async def test_bespoke_loop_parse_failure_on_a_complete_turn_is_still_failed(
    which, monkeypatch
):
    """The counterfactual: no budget stop, so the parse failure IS the cause."""
    if not _BESPOKE_IMPORT_OK:
        pytest.skip("gpt/claude agents unavailable (optional deps missing)")
    from unittest.mock import AsyncMock

    env_factory, entry_point, _, parse_failure = _BESPOKE_LOOPS[which]
    _stub_bespoke_gpt_echoed_dim_fetch(monkeypatch, which)
    monkeypatch.setattr(entry_point, AsyncMock(return_value=parse_failure()))

    env = env_factory(terminate_after=99)
    rl = await _make_bespoke_agent(which).sample(env, max_steps=2)

    assert [s.status for s in rl.steps] == [STATUS_FAILED]


# ---------------------------------------------------------------------------
# I15 — zero-call parse failures used to bypass ``env.step`` and run until the
# parse-error/safety cap. N3 collapses them into a single terminal response.
# ---------------------------------------------------------------------------


class _AlwaysParseErrorAdapter(_FakeAdapter):
    """Every turn fails to parse — the runaway shape, reproduced hermetically."""

    def __init__(self):
        self._i = 0

    def parse_raw_assistant_response(self, response: str) -> dict[str, Any]:
        self._i += 1
        message = {"role": "assistant", "content": [{"type": "text", "text": response}]}
        mark_model_output_error(message, 'Unknown action: computer_use(action="back")')
        return message

    def convert_message_from_agent(self, agent_message) -> dict[str, Any]:
        return dict(agent_message)


class _ProgrammerBugAdapter(_FakeAdapter):
    def parse_raw_assistant_response(self, response: str) -> dict[str, Any]:
        raise RuntimeError("programmer bug in parser")


class _ParserValueBugAdapter(_FakeAdapter):
    def parse_raw_assistant_response(self, response: str) -> dict[str, Any]:
        raise ValueError("programmer value bug in parser")


class _ParserKeyBugAdapter(_FakeAdapter):
    def parse_raw_assistant_response(self, response: str) -> dict[str, Any]:
        raise KeyError("programmer key bug in parser")


class _ConverterTypeBugAdapter(_FakeAdapter):
    def parse_raw_assistant_response(self, response: str) -> dict[str, Any]:
        return {"role": "assistant", "content": [{"type": "text", "text": response}]}

    def convert_message_from_agent(self, agent_message) -> dict[str, Any]:
        raise TypeError("programmer type bug in converter")


class _ConverterValueBugAdapter(_FakeAdapter):
    def parse_raw_assistant_response(self, response: str) -> dict[str, Any]:
        return {"role": "assistant", "content": [{"type": "text", "text": response}]}

    def convert_message_from_agent(self, agent_message) -> dict[str, Any]:
        raise ValueError("programmer value bug in converter")


class _ConverterIndexBugAdapter(_FakeAdapter):
    def parse_raw_assistant_response(self, response: str) -> dict[str, Any]:
        return {"role": "assistant", "content": [{"type": "text", "text": response}]}

    def convert_message_from_agent(self, agent_message) -> dict[str, Any]:
        raise IndexError("programmer index bug in converter")


class _RequiredCoordParseErrorAdapter(_FakeAdapter):
    def __init__(self):
        self._i = 0

    def parse_raw_assistant_response(self, response: str) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": response}],
            "tool_calls": [
                {
                    "name": "computer_use",
                    "arguments": {
                        "action": "left_click",
                        "coordinate": ["nan", 0],
                    },
                }
            ],
        }

    def convert_message_from_agent(self, agent_message) -> dict[str, Any]:
        self._i += 1
        required_coord(
            agent_message["tool_calls"][0]["arguments"].get("coordinate"),
            dimensions=2,
        )
        return {"role": "assistant", "content": [], "tool_calls": [_computer_call(None)]}


async def test_n3_parse_failure_zero_call_is_final_not_retry_budget():
    from lite.agents.core.agent.hooks import SampleHook

    class _StopReasons(SampleHook):
        def __init__(self):
            self.seen = []

        def on_step(self, data) -> None:
            self.seen.append(data.step_result.info.get("stop_reason"))

    raw_response = "Action: go back\n<tool_call>...</tool_call>"

    async def _gen(**kwargs):
        return {"response": raw_response}

    env = _FakeEnv(terminate_after=10_000)
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_AlwaysParseErrorAdapter(),
    )
    reasons = _StopReasons()

    rl = await agent.sample(env, max_steps=500, hooks=[reasons])
    messages = rl.lite_sample.messages

    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert len(env.seen_actions) == 1
    finish = env.seen_actions[0][0]
    assert _call_name(finish) == "response"
    assert _call_args(finish) == {"text": raw_response}
    assert reasons.seen == ["parse_failure"]
    assert rl.terminated is True
    assert rl.truncated is False


async def test_shared_loop_parser_programmer_exception_stays_loud():
    async def _gen(**kwargs):
        return {"response": "raw"}

    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_ProgrammerBugAdapter([]),
    )

    with pytest.raises(RuntimeError, match="programmer bug in parser"):
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=1)


@pytest.mark.parametrize(
    ("adapter_cls", "exc_type", "match"),
    [
        (_ParserValueBugAdapter, ValueError, "programmer value bug in parser"),
        (_ParserKeyBugAdapter, KeyError, "programmer key bug in parser"),
        (_ConverterTypeBugAdapter, TypeError, "programmer type bug in converter"),
        (_ConverterValueBugAdapter, ValueError, "programmer value bug in converter"),
        (_ConverterIndexBugAdapter, IndexError, "programmer index bug in converter"),
    ],
)
async def test_shared_loop_adapter_parse_convert_exceptions_stay_loud(
    adapter_cls,
    exc_type,
    match,
):
    async def _gen(**kwargs):
        return {"response": "raw"}

    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=adapter_cls([]),
    )

    with pytest.raises(exc_type, match=match):
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=1)


async def test_required_coord_parse_error_becomes_model_output_error():
    raw_response = "Action: left_click [nan, 0]"

    async def _gen(**kwargs):
        return {"response": raw_response}

    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_RequiredCoordParseErrorAdapter(),
        preserve_raw_response=True,
    )
    sample = LiteSample(
        metadata=LiteCUAMetadata(),
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    )

    result = await agent._predict_with_details(sample, processed_images=[])

    assert result.model_output_error == "coordinate must contain only finite numeric values"
    assert result.lite_message["content"] == [{"type": "text", "text": raw_response}]
    assert not result.lite_message.get("tool_calls")
    assert "raw_response" not in result.lite_message


async def test_shared_loop_env_close_failure_is_logged_and_swallowed(caplog):
    async def _gen(**kwargs):
        return {"response": "raw"}

    caplog.set_level(logging.WARNING, logger="lite.agents.core.agent.base")
    env = _CloseRaisesEnv(terminate_after=1)
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_FakeAdapter([[_computer_call()]]),
    )

    rl = await agent.sample(env, max_steps=1)

    assert rl.terminated is True
    assert env.closed is True
    assert "env.close() failed: shared close exploded" in caplog.text


async def test_shared_loop_env_close_runs_when_on_complete_raises():
    class _CompleteRaises(SampleHook):
        def on_complete(self, lite_rl_sample) -> None:
            raise RuntimeError("hook complete exploded")

    async def _gen(**kwargs):
        return {"response": "raw"}

    env = _FakeEnv(terminate_after=1)
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_FakeAdapter([[_computer_call()]]),
    )

    with pytest.raises(RuntimeError, match="hook complete exploded"):
        await agent.sample(env, max_steps=1, hooks=[_CompleteRaises()])

    assert env.closed is True


async def test_shared_loop_env_close_cancellation_propagates():
    async def _gen(**kwargs):
        return {"response": "raw"}

    env = _CloseCancelledEnv(terminate_after=1)
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_FakeAdapter([[_computer_call()]]),
    )

    with pytest.raises(asyncio.CancelledError):
        await agent.sample(env, max_steps=1)

    assert env.closed is True


async def test_shared_loop_action_error_is_not_masked_by_close_failure(caplog):
    async def _gen(**kwargs):
        return {"response": "raw"}

    caplog.set_level(logging.WARNING, logger="lite.agents.core.agent.base")
    env = _StepRaisesAndCloseRaisesEnv(terminate_after=1)
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_FakeAdapter([[_computer_call()]]),
    )

    with pytest.raises(RuntimeError, match="shared step exploded"):
        await agent.sample(env, max_steps=1)

    assert env.closed is True
    assert "env.close() failed: shared close exploded" in caplog.text


async def test_unparseable_model_finishes_once_not_the_safety_cap():
    """The agent-side safety cap must not be the bound for parse failures.

    A zero-call parse failure is a terminal N3 response. It runs one env step
    and never retries into either the old parse-error budget or ``max_steps``.
    """
    from lite.agents.core.agent.hooks import SampleHook

    class _StopReasons(SampleHook):
        def __init__(self):
            self.seen = []

        def on_step(self, data) -> None:
            self.seen.append(data.step_result.info.get("stop_reason"))

    async def _gen(**kwargs):
        return {"response": "Action: go back\n<tool_call>...</tool_call>"}

    env = _FakeEnv(terminate_after=10_000)
    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=_AlwaysParseErrorAdapter(),
    )
    reasons = _StopReasons()

    rl = await agent.sample(env, max_steps=500, hooks=[reasons])

    assistant_turns = [m for m in rl.lite_sample.messages if m.get("role") == "assistant"]
    assert len(assistant_turns) == 1
    assert len(env.seen_actions) == 1
    assert _call_name(env.seen_actions[0][0]) == "response"
    assert reasons.seen == ["parse_failure"]
    assert rl.terminated is True
    assert rl.truncated is False
