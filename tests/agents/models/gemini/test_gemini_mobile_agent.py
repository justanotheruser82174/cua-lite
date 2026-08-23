"""Gemini mobile agent rollout loop.

Hermetic: ``utils.transport.agenerate_content`` is stubbed. The mobile agent
reuses the desktop ``sample()`` verbatim, so these tests cover only what
actually differs -- the environment value, the exclusion set, the mobile
projection, and the pass-through policy.
"""

from __future__ import annotations

from typing import Any

from agents.models._support.provider_fakes import FakeMobileEnv

from lite.agents.models.gemini import agent as agent_module
from lite.agents.models.gemini.agent import GeminiMobileUseAgent


def _response(parts: list[dict[str, Any]], finish_reason: str = "STOP") -> dict[str, Any]:
    return {
        "candidates": [
            {"finishReason": finish_reason, "content": {"role": "model", "parts": parts}}
        ]
    }


def _stub(monkeypatch, responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []
    queue = list(responses)

    async def _fake(*, model, api_base, api_key, payload, timeout):
        seen.append(payload)
        return queue.pop(0) if queue else _response([{"text": "done"}])

    monkeypatch.setattr(agent_module, "agenerate_content", _fake)
    return seen


def _make(**kwargs):
    env = FakeMobileEnv()
    agent = GeminiMobileUseAgent(
        api_base="https://proxy", api_key="k", metadata=env.metadata, **kwargs
    )
    return agent, env


async def test_sends_mobile_environment_with_the_two_exclusions(monkeypatch):
    """open_app/list_apps are always excluded -- the env supplies a better open_app."""
    seen = _stub(
        monkeypatch,
        [
            _response(
                [{"functionCall": {"id": "c1", "name": "click", "args": {"x": 400, "y": 600}}}]
            )
        ],
    )
    agent, env = _make()

    await agent.sample(env, max_steps=2)

    assert seen[0]["tools"] == [
        {
            "computerUse": {
                "environment": "ENVIRONMENT_MOBILE",
                "excludedPredefinedFunctions": ["list_apps", "open_app"],
            }
        }
    ]


async def test_native_mobile_verbs_project_onto_canonical_mobile(monkeypatch):
    """click->tap, drag_and_drop->swipe, go_back->system_button("Back")."""
    _stub(
        monkeypatch,
        [
            _response(
                [
                    {"functionCall": {"id": "c1", "name": "click", "args": {"x": 1, "y": 2}}},
                    {
                        "functionCall": {
                            "id": "c2",
                            "name": "drag_and_drop",
                            "args": {"start_x": 1, "start_y": 9, "end_x": 1, "end_y": 3},
                        }
                    },
                    {"functionCall": {"id": "c3", "name": "go_back", "args": {}}},
                ]
            )
        ],
    )
    agent, env = _make()

    sample = await agent.sample(env, max_steps=2)

    actions = [
        action["action"]
        for message in sample.lite_sample.messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls", [])
        for action in call["function"]["arguments"]["actions"]
    ]
    assert actions == ["tap", "swipe", "system_button"]


async def test_type_with_press_enter_expands(monkeypatch):
    """Canonical mobile ``type`` has no press_enter, so it becomes two actions."""
    _stub(
        monkeypatch,
        [
            _response(
                [
                    {
                        "functionCall": {
                            "id": "c1",
                            "name": "type",
                            "args": {"text": "hi", "press_enter": True},
                        }
                    }
                ]
            )
        ],
    )
    agent, env = _make()

    sample = await agent.sample(env, max_steps=2)

    actions = [
        action
        for message in sample.lite_sample.messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls", [])
        for action in call["function"]["arguments"]["actions"]
    ]
    assert [a["action"] for a in actions] == ["type", "system_button"]
    assert actions[1]["button"] == "Enter"


async def test_env_extra_open_app_passes_through_by_name(monkeypatch):
    """The env's open_app (display-name enum) reaches the env, unconverted."""
    env = FakeMobileEnv()
    env.metadata.extra_tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": "open_app",
                "description": "Open an app.",
                "parameters": {
                    "type": "object",
                    "properties": {"app_name": {"type": "string", "enum": ["Chrome"]}},
                    "required": ["app_name"],
                },
            },
        }
    ]
    seen = _stub(
        monkeypatch,
        [
            _response(
                [{"functionCall": {"id": "c1", "name": "open_app", "args": {"app_name": "Chrome"}}}]
            )
        ],
    )
    agent = GeminiMobileUseAgent(api_base="https://p", api_key="k", metadata=env.metadata)

    sample = await agent.sample(env, max_steps=2)

    # Declared as a native FunctionDeclaration Tool alongside computerUse.
    assert {
        "functionDeclarations": [
            {
                "name": "open_app",
                "description": "Open an app.",
                "parameters": {
                    "type": "object",
                    "properties": {"app_name": {"type": "string", "enum": ["Chrome"]}},
                    "required": ["app_name"],
                },
            }
        ]
    } in seen[0]["tools"]
    # Routed to the env by name -- never through the GUI projection.
    calls = [
        call["function"]["name"]
        for message in sample.lite_sample.messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls", [])
    ]
    assert calls == ["open_app"]


def test_every_android_keyevent_maps_to_a_canonical_button():
    """The Gemini keyevent map duplicates canonical button vocabulary."""
    from lite.agents.models.gemini.action_space import (
        _GEMINI_KEY_TO_SYSTEM_BUTTON,
        GeminiMobileActionSpace,
    )

    space = GeminiMobileActionSpace()
    for wire_key, expected in sorted(_GEMINI_KEY_TO_SYSTEM_BUTTON.items()):
        lite = space.convert_tool_calls_from_agent(
            [{"name": "press_key", "arguments": {"key": wire_key}}]
        )
        actions = lite[0]["function"]["arguments"]["actions"]
        assert [a["action"] for a in actions] == ["system_button"], wire_key
        assert actions[0]["button"] == expected, wire_key


def test_valid_actions_gate_hides_verbs_the_env_cannot_run():
    """A verb stays visible only when EVERY canonical action it can produce is
    allowed -- this is what stops the model emitting an unrunnable action."""
    from lite.agents.models.gemini.action_space import GeminiMobileActionSpace

    space = GeminiMobileActionSpace()
    everything = space.visible_predefined_verbs(platform="mobile", valid_actions=None)
    assert {"type", "long_press", "drag_and_drop"} <= everything

    # Drop `type` from the env's repertoire: the verb producing it must vanish,
    # and only that one.
    without_type = space.visible_predefined_verbs(
        platform="mobile",
        valid_actions=sorted({"tap", "long_press", "swipe", "system_button", "screenshot", "wait"}),
    )
    assert "type" not in without_type
    assert {"long_press", "drag_and_drop"} <= without_type
