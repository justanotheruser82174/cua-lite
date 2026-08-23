"""Split GPT agent characterization tests.

Run:
    uv run pytest tests/agents/models/gpt/test_gpt_*.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from agents.models.gpt._support import _fake_response, _FakeEnv

from lite.agents.models.gpt.agent import GPTDesktopGroundingPointAgent, GPTDesktopUseAgent
from lite.core.tools.calls import tool_call_arguments, tool_call_name

# -----------------------------------------------------------------------------
# reasoning_effort (already exposed; document + assert custom value)
# -----------------------------------------------------------------------------


class TestReasoningEffort:
    """reasoning_effort kwarg controls reasoning.effort in request."""

    async def test_default_medium(self, monkeypatch):
        """The family-wide default, shared with the Claude and Gemini rows so the
        three API families sit at a comparable reasoning depth. It is also
        OpenAI's own documented default for gpt-5.5. The grounding subclass
        drops to ``none``; see ``GPT_GROUNDING_API_KWARGS_DEFAULTS``."""
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(model_id="gpt-5.5")
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        assert mock.call_args.kwargs["reasoning"]["effort"] == "medium"

    async def test_grounding_drops_to_none(self, monkeypatch):
        """Grounding must not inherit the family's ``medium``.

        ``GPT_GROUNDING_API_KWARGS_DEFAULTS`` spreads the family defaults and
        then overrides; reversing that order would silently move grounding to
        medium with nothing else failing.
        """
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopGroundingPointAgent(model_id="gpt-5.5")
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        assert mock.call_args.kwargs["reasoning"]["effort"] == "none"

    async def test_non_default_high(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)
        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            api_kwargs={"max_output_tokens": 4096, "reasoning_effort": "high"},
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)
        assert mock.call_args.kwargs["reasoning"]["effort"] == "high"

    async def test_non_default_low(self, monkeypatch):
        """An override BELOW the default. Pointed at ``low`` rather than
        ``medium``: medium is now the family default, so asserting it would pass
        even if ``reasoning_effort`` were ignored entirely."""
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            api_kwargs={"max_output_tokens": 4096, "reasoning_effort": "low"},
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        assert mock.call_args.kwargs["reasoning"]["effort"] == "low"


# -----------------------------------------------------------------------------
# parallel_tool_calls (already hardcoded False; now kwarg-gated)
# -----------------------------------------------------------------------------


class TestParallelToolCalls:
    """parallel_tool_calls kwarg exposes the already-correct default."""

    async def test_default_false(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(model_id="gpt-5.5")
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        assert mock.call_args.kwargs["parallel_tool_calls"] is False

    async def test_kwarg_true_allows_parallel(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            api_kwargs={"max_output_tokens": 4096, "parallel_tool_calls": True},
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        assert mock.call_args.kwargs["parallel_tool_calls"] is True


# -----------------------------------------------------------------------------
# system_prompt
# -----------------------------------------------------------------------------


class TestSystemPrompt:
    """``system_prompt`` flows through the top-level ``instructions`` field
    (single field — there is no separate suffix; a caller that wants a custom
    prompt sets ``system_prompt`` directly)."""

    async def test_system_prompt_sent_as_developer_message(self, monkeypatch):
        """``system_prompt_in_input`` (DEFAULT ON): the system prompt rides as a
        persistent ``role:"developer"`` message in ``input`` (so it survives
        ``previous_response_id`` chaining), NOT the one-shot top-level
        ``instructions`` field."""
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            api_kwargs={"max_output_tokens": 4096},
            system_prompt="CUSTOM_ANCHOR",
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        input_items = mock.call_args.kwargs["input"]
        dev = [it for it in input_items if it.get("role") == "developer"]
        assert dev and "CUSTOM_ANCHOR" in dev[0]["content"]
        assert not mock.call_args.kwargs.get("instructions")  # one-shot field unused

    async def test_no_system_prompt_means_no_developer_message(self, monkeypatch):
        """Desktop default ``system_prompt=None`` → native computer tool is enough,
        so no developer system-prompt message is injected."""
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            api_kwargs={"max_output_tokens": 4096},
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)
        input_items = mock.call_args.kwargs["input"]
        assert not any(it.get("role") == "developer" for it in input_items)
        assert not mock.call_args.kwargs.get("instructions")

    async def test_top_level_instructions_resend_when_unchained(self, monkeypatch):
        r1 = {
            "output": [
                {"type": "computer_call", "call_id": "c1", "actions": [{"type": "screenshot"}]}
            ],
            "id": "resp1",
            "usage": {},
        }
        mock = AsyncMock(side_effect=[r1, _fake_response()])
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            api_kwargs={"system_prompt_in_input": False, "chain_previous_response": False},
            system_prompt="CUSTOM_ANCHOR",
        )
        await agent.sample(_FakeEnv(terminate_after=2), max_steps=3)

        assert mock.call_count == 2
        for call in mock.call_args_list:
            assert call.kwargs.get("instructions") == "CUSTOM_ANCHOR"
            assert call.kwargs.get("previous_response_id") is None
            assert not [it for it in call.kwargs["input"] if it.get("role") == "developer"]


# ---------------------------------------------------------------------------
# Assistant text output is always parsed as a plain ``text`` part (the model's
# prose, e.g. a WebGym Memory/Progress block). Subclasses that distill a
# structured reasoning/action channel (e.g. the teacher extension) re-tag this
# text themselves; the base parsers never emit ``action_description``.
# ---------------------------------------------------------------------------


class TestTextParsedAsPlainText:
    _ITEMS = [
        {
            "type": "message",
            "content": [
                {"type": "output_text", "text": 'Memory: {"k": "v"}\nAction: Click login.'}
            ],
        }
    ]

    def test_desktop_parser_yields_plain_text(self):
        from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
        from lite.agents.models.gpt.utils.parse import parse_output_items_with_provenance

        parts = parse_output_items_with_provenance(
            self._ITEMS,
            GPTDesktopActionSpace(),
            (1920, 1080),
        ).message["content"]
        assert [p["type"] for p in parts] == ["text"]
        assert parts[0]["text"] == 'Memory: {"k": "v"}\nAction: Click login.'

    def test_mobile_parser_yields_plain_text(self):
        from lite.agents.models.gpt.action_space import GPTMobileActionSpace
        from lite.agents.models.gpt.utils.parse import (
            parse_gpt_mobile_output_items_with_provenance,
        )

        parts = parse_gpt_mobile_output_items_with_provenance(
            self._ITEMS,
            GPTMobileActionSpace(),
            (1080, 2400),
        ).message["content"]
        assert [p["type"] for p in parts] == ["text"]


# -----------------------------------------------------------------------------
# Responses request payload surface
# -----------------------------------------------------------------------------


class TestDesktopRequestPayloadFields:
    """The exact kwarg set ``litellm.aresponses`` receives.

    Individual knobs have their own tests above; this pins the whole payload so
    a silently added or dropped request field (a new default, an accidentally
    unconditional ``instructions``/``previous_response_id``) fails here.
    """

    # Fields sent on every desktop request, regardless of config.
    _ALWAYS = {
        "model",
        "input",
        "tools",
        "parallel_tool_calls",
        "stream",
        "reasoning",
        "truncation",
        "extra_body",
    }

    async def test_default_request_sends_only_the_unconditional_fields(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        await GPTDesktopUseAgent(model_id="gpt-5.5").sample(
            _FakeEnv(terminate_after=1), max_steps=2
        )

        kwargs = mock.call_args.kwargs
        assert set(kwargs) == self._ALWAYS
        assert kwargs["model"] == "gpt-5.5"
        assert kwargs["stream"] is False
        assert kwargs["parallel_tool_calls"] is False
        assert kwargs["reasoning"] == {"effort": "medium", "summary": "concise"}
        assert kwargs["truncation"] == "auto"
        assert kwargs["extra_body"]["prompt_cache_key"].startswith("cua-lite-")
        assert [tool.get("type") for tool in kwargs["tools"]] == ["computer"]
        assert isinstance(kwargs["input"], list)

    async def test_optional_fields_are_sent_only_when_configured(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            api_key="sk-test",
            api_base="https://example.invalid/v1",
            system_prompt="POLICY",
            api_kwargs={
                "system_prompt_in_input": False,
                "max_output_tokens": 4096,
                "temperature": 0.25,
                "tool_choice": "required",
            },
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        kwargs = mock.call_args.kwargs
        assert set(kwargs) == self._ALWAYS | {
            "max_output_tokens",
            "instructions",
            "temperature",
            "tool_choice",
            "api_key",
            "api_base",
        }
        assert kwargs["max_output_tokens"] == 4096
        assert kwargs["instructions"] == "POLICY"
        assert kwargs["temperature"] == 0.25
        assert kwargs["tool_choice"] == "required"
        assert kwargs["api_key"] == "sk-test"
        assert kwargs["api_base"] == "https://example.invalid/v1"
        # ``system_prompt_in_input: false`` means no developer item on this path.
        assert not [it for it in kwargs["input"] if it.get("role") == "developer"]

    async def test_first_turn_bundles_instruction_and_screenshot_in_one_user_item(
        self,
        monkeypatch,
    ):
        """Turn 0 sends ONE user item carrying instruction + screenshot, matching
        the single Lite decision turn recorded for the same step."""
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        result = await GPTDesktopUseAgent(model_id="gpt-5.5").sample(
            _FakeEnv(terminate_after=1), max_steps=2
        )

        first_input = mock.call_args_list[0].kwargs["input"]
        assert [it.get("role") for it in first_input] == ["user"]
        content = first_input[0]["content"]
        assert content[0] == {"type": "input_text", "text": "instr"}
        assert content[1]["type"] == "input_image"
        assert len(content) == 2

        # One Lite user message for the same turn, carrying the same frame.
        lite_user = [m for m in result.lite_sample.messages if m["role"] == "user"]
        assert len(lite_user) == 1
        assert lite_user[0]["content"] == [
            {"type": "image", "index": 0},
            {"type": "text", "text": "instr"},
        ]

    async def test_chained_turn_swaps_instructions_for_previous_response_id(self, monkeypatch):
        """With chaining ON and ``system_prompt_in_input: false``, the one-shot
        ``instructions`` field is sent on turn 0 only — the chained turn carries
        ``previous_response_id`` instead. This is the branch the unchained-resend
        test contrasts with."""
        r1 = {
            "output": [
                {"type": "computer_call", "call_id": "c1", "actions": [{"type": "screenshot"}]}
            ],
            "id": "resp1",
            "usage": {},
        }
        mock = AsyncMock(side_effect=[r1, _fake_response()])
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            api_kwargs={"system_prompt_in_input": False},
            system_prompt="CUSTOM_ANCHOR",
        )
        await agent.sample(_FakeEnv(terminate_after=2), max_steps=3)

        assert mock.call_count == 2
        first, second = mock.call_args_list
        assert first.kwargs["instructions"] == "CUSTOM_ANCHOR"
        assert "previous_response_id" not in first.kwargs
        assert "instructions" not in second.kwargs
        assert second.kwargs["previous_response_id"] == "resp1"

    async def test_system_prompt_keeps_literal_json_braces(self, monkeypatch):
        """The ``{w}``/``{h}`` substitution is literal-brace safe: a prompt that
        also contains JSON braces must survive verbatim. ``str.format`` would
        raise ``KeyError: 'ok'`` here."""
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            system_prompt='Return JSON like {"ok": true} on the {w}×{h} screen.',
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        dev = [it for it in mock.call_args.kwargs["input"] if it.get("role") == "developer"]
        assert dev
        assert dev[0]["content"] == 'Return JSON like {"ok": true} on the 800×600 screen.'


def test_content_only_final_actions_are_metadata_free_final_policy():
    """N4: no-tool final scoring is final-message policy, not extra-tool catalog."""
    from lite.core.messages.final import make_no_tool_call_final_actions

    acts = make_no_tool_call_final_actions("the answer is 42")
    assert [tool_call_name(a) for a in acts] == ["response"]
    assert tool_call_arguments(acts[0]) == {"text": "the answer is 42"}
    acts = make_no_tool_call_final_actions("")
    assert tool_call_name(acts[0]) == "response"
    assert tool_call_arguments(acts[0]) == {"text": ""}
