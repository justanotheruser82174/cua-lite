"""Gemini desktop agent rollout loop.

Hermetic: ``utils.transport.agenerate_content`` is stubbed, so no HTTP and no
env container. The stub records every payload, which is what lets the wire-shape
assertions below run against the real loop rather than a hand-built request.
"""

from __future__ import annotations

from typing import Any

import pytest
from agents.models._support.provider_fakes import FakeEnv

from lite.agents.models.gemini import agent as agent_module
from lite.agents.models.gemini.agent import GeminiDesktopUseAgent
from lite.core import LiteGenericMetadata
from lite.core.messages.final import CONTENT_ONLY_FINAL_REASON
from lite.core.tools.calls import (
    runtime_internal_stop_reason,
    tool_call_arguments,
    tool_call_id,
    tool_call_name,
)


def _response(parts: list[dict[str, Any]], finish_reason: str = "STOP") -> dict[str, Any]:
    return {
        "candidates": [
            {"finishReason": finish_reason, "content": {"role": "model", "parts": parts}}
        ]
    }


def _stub(monkeypatch, responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Serve ``responses`` in order; return the list of payloads seen."""
    seen: list[dict[str, Any]] = []
    queue = list(responses)

    async def _fake(*, model, api_base, api_key, payload, timeout):
        seen.append(payload)
        return queue.pop(0) if queue else _response([{"text": "done"}])

    monkeypatch.setattr(agent_module, "agenerate_content", _fake)
    return seen


def _make(**kwargs) -> GeminiDesktopUseAgent:
    env = FakeEnv()
    return GeminiDesktopUseAgent(
        api_base="https://proxy", api_key="k", metadata=env.metadata, **kwargs
    ), env


def test_provider_native_rejects_generic_metadata_for_tool_surface():
    agent = GeminiDesktopUseAgent(
        api_base="https://proxy",
        api_key="k",
        metadata=LiteGenericMetadata(dims=()),
    )

    with pytest.raises(TypeError, match="GeminiDesktopUseAgent requires LiteCUAMetadata"):
        agent._build_tools()


async def test_single_click_turn_terminates_and_records(monkeypatch):
    seen = _stub(
        monkeypatch,
        [
            _response(
                [
                    {
                        "functionCall": {
                            "id": "c1",
                            "name": "click",
                            "args": {"intent": "click ok", "x": 500, "y": 300},
                        }
                    }
                ]
            )
        ],
    )
    agent, env = _make()

    sample = await agent.sample(env, max_steps=3)

    assert sample.terminated is True
    assert len(sample.steps) == 1
    step = sample.steps[0]
    # The step's prompt is the redacted payload, and it binds exactly the frames
    # the request carried.
    assert "[base64 image omitted]" in step.prompt
    assert step.image_indices == (0,)

    payload = seen[0]
    assert payload["tools"] == [{"computerUse": {"environment": "ENVIRONMENT_DESKTOP"}}]
    # Step 0 seeds instruction + screenshot as a native user Content.
    assert payload["contents"][0]["role"] == "user"
    assert payload["contents"][0]["parts"][0]["text"] == "instr"
    assert "inlineData" in payload["contents"][0]["parts"][1]


async def test_tool_call_at_max_steps_persists_terminal_feedback(monkeypatch):
    seen = _stub(
        monkeypatch,
        [
            _response(
                [
                    {
                        "functionCall": {
                            "id": "c1",
                            "name": "click",
                            "args": {"x": 1, "y": 2},
                        }
                    }
                ]
            )
        ],
    )
    agent, env = _make()
    env._terminate_after = 99

    sample = await agent.sample(env, max_steps=1)

    assert sample.terminated is False
    assert sample.truncated is True
    assert sample.steps[-1].status == "truncated"
    assert len(seen) == 1
    assert [tuple(step.image_indices) for step in sample.steps] == [(0,)]
    assert [message["role"] for message in sample.lite_sample.messages] == [
        "user",
        "assistant",
        "tool",
    ]
    tool_message = sample.lite_sample.messages[-1]
    assert tool_message["tool_call_id"] == "call_0000"
    assert tool_message["content"] == [
        {"type": "image", "index": 1},
        {"type": "text", "text": "instr"},
    ]


async def test_markers_are_stripped_from_the_request(monkeypatch):
    """Native REST rejects a Part with an unknown field, so this is correctness.

    The episode must reach a SECOND request: result frames are marked inside
    ``functionResponse.parts``, so a turn-1-only run never produces a nested
    part and the nested half of the strip would be untested -- while dropping it
    would 500 every rollout from turn 2 on.
    """
    seen = _stub(
        monkeypatch,
        [
            _response([{"functionCall": {"id": "c1", "name": "click", "args": {"x": 1, "y": 2}}}]),
            _response([{"text": "done"}]),
        ],
    )
    agent, env = _make()
    env._terminate_after = 2

    await agent.sample(env, max_steps=3)

    # The nested assertions below are only meaningful if a result frame exists.
    nested_parts = [
        nested
        for payload in seen
        for content in payload["contents"]
        for part in content["parts"]
        for nested in (part.get("functionResponse") or {}).get("parts") or []
    ]
    assert nested_parts, "no nested functionResponse part — the nested strip is untested"

    for payload in seen:
        for content in payload["contents"]:
            for part in content["parts"]:
                assert "_cua_lite_image_index" not in part
                function_response = part.get("functionResponse")
                for nested in (function_response or {}).get("parts") or []:
                    assert "_cua_lite_image_index" not in nested


async def test_second_turn_replays_model_then_answers_in_one_user_content(monkeypatch):
    """FC/FR ordering: the model turn, then ONE user Content with every response."""
    seen = _stub(
        monkeypatch,
        [
            _response(
                [
                    {
                        "functionCall": {"id": "c1", "name": "click", "args": {"x": 1, "y": 2}},
                        "thoughtSignature": "SIG",
                    }
                ]
            ),
            _response([{"functionCall": {"id": "c2", "name": "take_screenshot", "args": {}}}]),
        ],
    )
    agent, env = _make()
    env._terminate_after = 2

    await agent.sample(env, max_steps=3)

    contents = seen[1]["contents"]
    roles = [content["role"] for content in contents]
    assert roles == ["user", "model", "user"]
    # The replayed model turn keeps the signature verbatim -- rebuilding the
    # part field-by-field would drop it and earn a 400.
    assert contents[1]["parts"][0]["thoughtSignature"] == "SIG"
    # Exactly one user Content answers it, and the screenshot is nested inside
    # the functionResponse (not a sibling part) so it binds to the call.
    answer_parts = contents[2]["parts"]
    assert len(answer_parts) == 1
    function_response = answer_parts[0]["functionResponse"]
    assert function_response["name"] == "click"
    assert function_response["response"]["output"] is not None
    assert "inlineData" in function_response["parts"][0]
    assert function_response["parts"][0]["inlineData"]["mimeType"] == "image/png"
    # Bare base64 -- no data: URL prefix, unlike the OpenAI/Anthropic wires.
    assert not function_response["parts"][0]["inlineData"]["data"].startswith("data:")


async def test_text_only_turn_is_a_deliberate_final(monkeypatch):
    seen = _stub(monkeypatch, [_response([{"text": "the answer is 42"}])])
    agent, env = _make()

    sample = await agent.sample(env, max_steps=3)

    assert sample.terminated is True
    assert sample.truncated is False
    assert sample.steps[0].status == "completed"
    assert [tuple(step.image_indices) for step in sample.steps] == [(0,)]
    assert [message["role"] for message in sample.lite_sample.messages] == [
        "user",
        "assistant",
    ]
    assert sample.lite_sample.messages[-1]["content"] == [
        {"type": "text", "text": "the answer is 42"}
    ]
    assert "tool_calls" not in sample.lite_sample.messages[-1]
    assert not any(message.get("role") == "tool" for message in sample.lite_sample.messages)
    assert len(seen) == 1
    assert [content["role"] for content in seen[0]["contents"]] == ["user"]
    assert len(env.actions_seen) == 1
    finish = env.actions_seen[0][0]
    assert tool_call_name(finish) == "response"
    assert tool_call_id(finish) is None
    assert runtime_internal_stop_reason(finish) == CONTENT_ONLY_FINAL_REASON
    assert tool_call_arguments(finish) == {"text": "the answer is 42"}


async def test_text_only_turn_can_continue_with_env_feedback(monkeypatch):
    seen = _stub(
        monkeypatch,
        [
            _response([{"text": "first answer"}]),
            _response([{"text": "second answer"}]),
        ],
    )
    agent, env = _make()
    env._terminate_after = 2

    sample = await agent.sample(env, max_steps=3)

    assert sample.terminated is True
    assert [tool_call_arguments(turn[0]) for turn in env.actions_seen] == [
        {"text": "first answer"},
        {"text": "second answer"},
    ]
    assert [tuple(step.image_indices) for step in sample.steps] == [(0,), (0, 1)]
    assert [message["role"] for message in sample.lite_sample.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert not any(message.get("role") == "tool" for message in sample.lite_sample.messages)

    contents = seen[1]["contents"]
    assert [content["role"] for content in contents] == ["user", "model", "user"]
    assert contents[1]["parts"] == [{"text": "first answer"}]
    feedback_parts = contents[2]["parts"]
    assert any(part.get("text") == "instr" for part in feedback_parts)
    assert any("inlineData" in part for part in feedback_parts)
    assert not any("functionResponse" in part for part in feedback_parts)


async def test_undeclared_call_becomes_model_visible_feedback(monkeypatch):
    seen = _stub(
        monkeypatch,
        [
            _response([{"functionCall": {"id": "c1", "name": "not_a_verb", "args": {}}}]),
            _response([{"text": "ok"}]),
        ],
    )
    agent, env = _make()

    sample = await agent.sample(env, max_steps=3)

    # No canonical call parsed -> the shared loop treats it as a parse-failure
    # final, so the episode ends on turn 1 and the step is marked failed.
    assert sample.steps[0].status == "failed"
    assert len(seen) == 1


async def test_max_tokens_marks_the_step_truncated(monkeypatch):
    """The truncation set must be lowercase: append_lite_rl_step lowercases."""
    _stub(monkeypatch, [_response([{"text": "cut off"}], finish_reason="MAX_TOKENS")])
    agent, env = _make()

    sample = await agent.sample(env, max_steps=3)

    assert sample.steps[0].status == "truncated"


async def test_missing_thought_signature_is_loud(monkeypatch):
    """A request-construction bug must not be recorded as model output error."""
    _stub(monkeypatch, [_response([{"text": "x"}], finish_reason="MISSING_THOUGHT_SIGNATURE")])
    agent, env = _make()

    with pytest.raises(RuntimeError, match="request-construction bug"):
        await agent.sample(env, max_steps=3)


async def test_valid_actions_empty_drops_the_tool(monkeypatch):
    seen = _stub(monkeypatch, [_response([{"text": "no tools"}])])
    env = FakeEnv()
    env.metadata.valid_actions = []
    agent = GeminiDesktopUseAgent(api_base="https://p", api_key="k", metadata=env.metadata)

    await agent.sample(env, max_steps=2)

    assert "tools" not in seen[0]


async def test_truncated_images_do_not_leak_into_image_indices(monkeypatch):
    """``image_indices`` binds RL steps to processor image slots, so it must
    count only frames the request actually carried.

    Truncation rewrites the dropped Part into a text placeholder. If it kept the
    provider-visible image-index marker, the harvest that runs right after would
    still read it -- and every later step would claim the seed frame forever,
    handing the trainer one more image than the prompt contains.
    """
    _stub(
        monkeypatch,
        [
            _response([{"functionCall": {"id": "c1", "name": "click", "args": {"x": 1, "y": 2}}}]),
            _response([{"functionCall": {"id": "c2", "name": "click", "args": {"x": 3, "y": 4}}}]),
            _response([{"text": "done"}]),
        ],
    )
    agent, env = _make(only_n_most_recent_images=1)
    env._terminate_after = 3  # three real turns, so truncation actually fires

    sample = await agent.sample(env, max_steps=4)

    for step in sample.steps:
        assert len(step.image_indices) <= 1, (
            f"step claims {step.image_indices} but the request carries one image"
        )


async def test_every_replayed_function_call_gets_a_response(monkeypatch):
    """FC/FR closure: the replay is verbatim over every part, so an id-less
    ``functionCall`` must still be answered -- by name, carrying its rejection.

    A replayed call with no matching ``functionResponse`` makes the NEXT request
    malformed, and dropping it would also hide the rejection from the model.
    """
    seen = _stub(
        monkeypatch,
        [
            _response(
                [
                    {"functionCall": {"name": "click", "args": {"x": 1, "y": 2}}},  # no id
                    {"functionCall": {"id": "c2", "name": "click", "args": {"x": 3, "y": 4}}},
                ]
            ),
            _response([{"text": "done"}]),
        ],
    )
    agent, env = _make()
    env._terminate_after = 2  # a second request is what carries the replay

    await agent.sample(env, max_steps=3)

    contents = seen[1]["contents"]
    replayed = [
        part["functionCall"]
        for content in contents
        if content["role"] == "model"
        for part in content["parts"]
        if "functionCall" in part
    ]
    answered = [
        part["functionResponse"]
        for content in contents
        if content["role"] == "user"
        for part in content["parts"]
        if "functionResponse" in part
    ]
    assert len(replayed) == len(answered) == 2
    # The id-less one is answered by name and carries the parser's rejection.
    by_name_only = [r for r in answered if "id" not in r]
    assert len(by_name_only) == 1
    assert by_name_only[0]["name"] == "click"
    assert "error" in by_name_only[0]["response"]


async def test_thinking_level_defaults_to_medium_and_is_upper_cased(monkeypatch):
    """``thinkingLevel`` is an UPPERCASE enum on the wire.

    The family default is spelled lowercase in ``GEMINI_API_KWARGS_DEFAULTS`` to
    match the yaml rows, so the projection is the only thing standing between
    that and a wire-level rejection -- a lowercase value is not a Python error,
    it is a bad request the loop would only discover live.
    """
    seen = _stub(monkeypatch, [_response([{"text": "done"}])])
    agent, env = _make()

    await agent.sample(env, max_steps=2)

    assert seen[0]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "MEDIUM"}


async def test_thinking_level_override_rides_through(monkeypatch):
    """``high`` is the ceiling Gemini offers; xhigh/max are rejected."""
    seen = _stub(monkeypatch, [_response([{"text": "done"}])])
    agent, env = _make(api_kwargs={"thinking_level": "high", "max_output_tokens": 8192})

    await agent.sample(env, max_steps=2)

    config = seen[0]["generationConfig"]
    assert config["thinkingConfig"] == {"thinkingLevel": "HIGH"}
    assert config["maxOutputTokens"] == 8192
