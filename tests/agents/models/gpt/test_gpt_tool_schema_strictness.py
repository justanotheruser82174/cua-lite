"""Split GPT agent characterization tests.

Run:
    uv run pytest tests/agents/models/gpt/test_gpt_*.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from agents._support.valid_actions_gating import (
    BASH_SCHEMA,
    RESPONSE_SCHEMA,
    openai_provider_tool_names,
    openai_tools_sent,
)
from agents.models._support.provider_fakes import FakeMobileEnv
from agents.models.gpt._support import _fake_response, _FakeEnv, _tools_sent

from lite.agents.models.gpt.action_space import GPTDesktopGroundingPointActionSpace
from lite.agents.models.gpt.agent import (
    GPTDesktopGroundingPointAgent,
    GPTDesktopUseAgent,
    GPTMobileUseAgent,
)
from lite.core import LiteCUAMetadata, LiteGenericMetadata
from lite.core.messages.final import pop_model_output_error
from lite.core.tools import make_tool_call, make_tool_schema
from lite.core.tools.calls import tool_call_name

# -----------------------------------------------------------------------------
# computer tool schema + truncation
# -----------------------------------------------------------------------------


class TestComputerToolSchema:
    """GA ``{"type":"computer"}`` is a bare type tag — no display_* / environment fields."""

    def test_provider_native_rejects_generic_metadata_for_tool_surface(self):
        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            metadata=LiteGenericMetadata(dims=()),
        )

        with pytest.raises(TypeError, match="GPTDesktopUseAgent requires LiteCUAMetadata"):
            agent._build_tools()

    async def test_ga_computer_is_bare_type_tag(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(model_id="gpt-5.5")
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        tools = mock.call_args.kwargs["tools"]
        computer_tool = next(t for t in tools if t.get("type") == "computer")
        assert computer_tool == {"type": "computer"}

    async def test_valid_actions_computer_verb_keeps_native_tool(self, monkeypatch):
        """Non-empty ``valid_actions`` cannot filter GPT's opaque computer tool.

        The native tool is kept whole. Finish tools (response/terminate) are not
        action-space members and surface only as schema-backed extras.
        ``[]`` drops the native tool; ``None`` keeps it.
        """
        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5", metadata=LiteCUAMetadata(valid_actions=["click"])
        )
        tools = await _tools_sent(agent, monkeypatch)
        assert any(t.get("type") == "computer" for t in tools)

    async def test_grounding_click_schema_is_action_space_owned(self, monkeypatch):
        """Grounding declares only the action-space-owned click function."""
        agent = GPTDesktopGroundingPointAgent(model_id="gpt-5.5")
        tools = await _tools_sent(agent, monkeypatch)
        owner_schema = type(agent.action_space).get_tool_schema("click")

        assert owner_schema is not None
        owner_function = owner_schema["function"]
        # The whole request tool list is the action space's click schema,
        # projected into the Responses function-tool shape.
        assert tools == [
            {
                "type": "function",
                "name": owner_function["name"],
                "description": owner_function["description"],
                "parameters": owner_function["parameters"],
            }
        ]
        assert GPTDesktopGroundingPointActionSpace.get_tool_names() == frozenset({"click"})
        assert GPTDesktopGroundingPointActionSpace.get_declared_action_schema_names() == frozenset(
            {"click"}
        )
        assert all(tool.get("type") != "computer" for tool in tools)
        assert tools[0]["parameters"]["additionalProperties"] is False

    async def test_grounding_strict_click_schema_keeps_required_closed_params(self, monkeypatch):
        agent = GPTDesktopGroundingPointAgent(
            model_id="gpt-5.5",
            function_tool_strict=True,
        )
        click_tool = (await _tools_sent(agent, monkeypatch))[0]

        assert click_tool["name"] == "click"
        assert click_tool["strict"] is True
        assert click_tool["parameters"]["required"] == ["x", "y"]
        assert click_tool["parameters"]["additionalProperties"] is False

    async def test_grounding_rejects_provider_downsampled_fixed_frame(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)
        monkeypatch.setattr(
            "lite.agents.models.gpt.utils.image_io._fetch_processed_image_dims",
            AsyncMock(return_value=[(640, 480)]),
        )

        agent = GPTDesktopGroundingPointAgent(model_id="gpt-5.5")
        with pytest.raises(RuntimeError, match="fixed-frame screenshot"):
            await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

    async def test_valid_actions_empty_drops_native_tool(self, monkeypatch):
        """``valid_actions=[]`` (used by browsergym text+bid configs)
        suppresses the native ``computer`` tool: only env-supplied function tools
        surface. Mirrors Qwen's ``valid_actions: []`` convention through the
        action space's public ``filter_tool_schemas_for_valid_actions`` hook."""
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(
            metadata=LiteCUAMetadata(valid_actions=[], extra_tool_schemas=[
                    make_tool_schema(
                        "click",
                        description="Click an element by bid.",
                        parameters={
                            "type": "object",
                            "properties": {"bid": {"type": "string"}},
                            "required": ["bid"],
                        },
                    ),
                ]),
            model_id="gpt-5.5",
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        tools = mock.call_args.kwargs["tools"]
        # No native computer tool present.
        assert all(t.get("type") != "computer" for t in tools)
        # Only the env-supplied function tool surfaces.
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["name"] == "click"

    async def test_valid_actions_none_withholds_finish_tools_osworld_parity(self, monkeypatch):
        """OSWORLD PARITY / DRIFT GUARD. For GPT the finish tools
        (``response``/``terminate``) are a STRICT env-gated opt-in layer — NOT part
        of the native-enum ``None``=expose-all contract (the opaque computer tool
        can't carry them in an enum). So a default ``valid_actions=None`` desktop env
        (osworld) gets ONLY the native ``computer`` tool — no ``response``/``terminate``.
        osworld terminates via an empty action, never a ``terminate`` tool-call;
        offering the tool would change behavior AND recorded trajectory format.
        Browser envs opt in by resolving ``response`` into ``extra_tool_schemas``."""
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(model_id="gpt-5.5")  # valid_actions defaults None
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        tools = mock.call_args.kwargs["tools"]
        fn_names = {t.get("name") for t in tools if t.get("type") == "function"}
        assert "response" not in fn_names and "terminate" not in fn_names, fn_names
        assert any(t.get("type") == "computer" for t in tools)

    async def test_valid_actions_does_not_opt_in_finish_tools(self, monkeypatch):
        """``valid_actions`` is GUI-only for GPT; finish tools are schema-backed
        extras and must not appear without extra_tool_schemas."""
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5", metadata=LiteCUAMetadata(valid_actions=["click"])
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        tools = mock.call_args.kwargs["tools"]
        fn_names = {t.get("name") for t in tools if t.get("type") == "function"}
        assert "response" not in fn_names and "terminate" not in fn_names
        assert any(t.get("type") == "computer" for t in tools)

    def test_parse_routes_extra_tool_passthrough_ignores_undeclared_finish(self):
        """Declared extra function_calls pass through verbatim.

        Undeclared finish tools are not native GPT desktop actions.
        """
        from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
        from lite.agents.models.gpt.utils.parse import parse_output_items_with_provenance

        items = [
            {
                "type": "function_call",
                "call_id": "call_infeasible",
                "name": "report_infeasible",
                "arguments": '{"reason": "x"}',
            },
            {
                "type": "function_call",
                "call_id": "call_terminate",
                "name": "terminate",
                "arguments": '{"status": "success"}',
            },
        ]
        msg = parse_output_items_with_provenance(
            items,
            GPTDesktopActionSpace(),
            (1024, 768),
            extra_tool_names=frozenset({"report_infeasible"}),
        ).message
        calls = msg["tool_calls"]
        # extra tool passes through verbatim (name preserved, not coord-converted)
        assert any(tool_call_name(c) == "report_infeasible" for c in calls), calls
        assert not any(tool_call_name(c) == "terminate" for c in calls), calls

        declared = parse_output_items_with_provenance(
            items,
            GPTDesktopActionSpace(),
            (1024, 768),
            extra_tool_names=frozenset({"report_infeasible", "terminate"}),
        ).message
        assert any(tool_call_name(c) == "terminate" for c in declared["tool_calls"])

    async def test_extra_schema_unwraps_for_provider_and_wraps_back_to_canonical_call(
        self, monkeypatch
    ):
        from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
        from lite.agents.models.gpt.utils.parse import parse_output_items_with_provenance

        schema = make_tool_schema(
            "goto",
            description="Navigate to a URL.",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
                "additionalProperties": False,
            },
        )
        agent = GPTDesktopUseAgent(metadata=LiteCUAMetadata(extra_tool_schemas=[schema]))

        sent_tools = await _tools_sent(agent, monkeypatch)
        provider_tool = next(tool for tool in sent_tools if tool.get("name") == "goto")
        assert provider_tool == {
            "type": "function",
            "name": "goto",
            "description": "Navigate to a URL.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
                "additionalProperties": False,
            },
        }
        assert "function" not in provider_tool

        msg = parse_output_items_with_provenance(
            [
                {
                    "type": "function_call",
                    "call_id": "call_goto",
                    "name": "goto",
                    "arguments": '{"url": "https://example.com"}',
                }
            ],
            GPTDesktopActionSpace(),
            (1024, 768),
            extra_tool_names=frozenset({"goto"}),
        ).message

        assert msg["tool_calls"] == [
            make_tool_call("goto", {"url": "https://example.com"}, call_id="call_0000")
        ]

    def test_parse_routes_invalid_active_extra_to_env_feedback(self):
        """A malformed active extra survives the parser and reaches env ingress.

        Mixed output — one valid GUI call plus one bad-argument extra — must not
        lose the extra: the parser routes an advertised env tool by name, and
        ``prepare_env_tool_calls`` is what names the bad argument back to the
        model. Dropping it here would delete that answer.
        """
        from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
        from lite.agents.models.gpt.utils.parse import parse_output_items_with_provenance
        from lite.gym.utils.feedback.ingress import prepare_env_tool_calls

        report_schema = make_tool_schema(
            "report_infeasible",
            description="Report infeasible.",
            parameters={
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
                "additionalProperties": False,
            },
        )

        msg = parse_output_items_with_provenance(
            [
                {
                    "type": "computer_call",
                    "call_id": "call_screenshot",
                    "actions": [{"type": "screenshot"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_infeasible",
                    "name": "report_infeasible",
                    "arguments": "{}",
                },
            ],
            GPTDesktopActionSpace(),
            (1024, 768),
            extra_tool_names=frozenset({"report_infeasible"}),
        ).message

        assert [tool_call_name(call) for call in msg["tool_calls"]] == [
            "computer",
            "report_infeasible",
        ]
        assert pop_model_output_error(msg) is None

        routed, errors = prepare_env_tool_calls(
            msg["tool_calls"],
            LiteCUAMetadata(extra_tool_schemas=[report_schema]),
        )
        assert all(call["name"] != "report_infeasible" for call, _ in routed)
        assert set(errors) == {"call_0001"}
        assert "report_infeasible" in errors["call_0001"].message

    def test_parse_drops_malformed_function_call_arguments(self, caplog):
        from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
        from lite.agents.models.gpt.utils.parse import parse_output_items_with_provenance

        msg = parse_output_items_with_provenance(
            [
                {
                    "type": "function_call",
                    "call_id": "call_bad_json",
                    "name": "report_infeasible",
                    "arguments": "{not json",
                },
                {
                    "type": "function_call",
                    "call_id": "call_none_args",
                    "name": "report_infeasible",
                    "arguments": None,
                },
            ],
            GPTDesktopActionSpace(),
            (1024, 768),
            extra_tool_names=frozenset({"report_infeasible"}),
        ).message

        assert msg["tool_calls"] == [
            make_tool_call("report_infeasible", call_id="call_0000"),
        ]
        assert "malformed arguments" in caplog.text

    async def test_webgym_scenario_surfaces_response_and_nav_extras(self, monkeypatch):
        """Behavioral-drift guard for the WebGym contract. WebGym passes
        ``valid_actions=["click","type","key","scroll","wait"]`` plus resolved
        env extras ``goto``/``back``/``response``. Expected GPT tool surface:
          - native ``computer`` tool kept WHOLE (its action enum is a black box);
          - ``response`` surfaced because it is present in extra_tool_schemas;
          - ``terminate`` NOT surfaced because it is absent from extra_tool_schemas;
          - ``goto``/``back`` extra tools pass through and are present.
        """
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)
        extras = [
            make_tool_schema(
                "goto",
                description="Navigate to a URL.",
                parameters={
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            ),
            make_tool_schema(
                "back",
                description="Go back one page.",
            ),
            make_tool_schema(
                "response",
                description="Submit the final answer.",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            ),
        ]
        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            metadata=LiteCUAMetadata(
                valid_actions=["click", "type", "key", "scroll", "wait"],
                extra_tool_schemas=extras,
            ),
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        tools = mock.call_args.kwargs["tools"]
        types = [t.get("type") for t in tools]
        fn_names = {t.get("name") for t in tools if t.get("type") == "function"}
        assert "computer" in types  # native tool kept whole
        assert "response" in fn_names  # env extra schema present
        assert "terminate" not in fn_names  # no env extra schema
        assert {"goto", "back"} <= fn_names  # env extras present (passthrough)


class TestTruncationKwarg:
    """truncation kwarg; default 'auto', can be None to omit."""

    async def test_default_sends_truncation_auto(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(model_id="gpt-5.5")
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        assert mock.call_args.kwargs.get("truncation") == "auto"

    async def test_none_omits_truncation(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            api_kwargs={"max_output_tokens": 4096, "truncation": None},
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        # None should not be sent as a kwarg
        assert "truncation" not in mock.call_args.kwargs


# -----------------------------------------------------------------------------
# function_tool_strict
# -----------------------------------------------------------------------------


class TestFunctionToolStrict:
    """strict field on extra_tools function schemas; default off for safety."""

    async def test_default_omits_strict(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(
            metadata=LiteCUAMetadata(extra_tool_schemas=[
                    make_tool_schema(
                        "goto",
                        description="navigate",
                        parameters={"type": "object", "properties": {"url": {"type": "string"}}},
                    )
                ]),
            model_id="gpt-5.5",
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        tools = mock.call_args.kwargs["tools"]
        fn_tool = next(t for t in tools if t.get("name") == "goto")
        assert "strict" not in fn_tool

    async def test_kwarg_true_adds_strict(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(
            metadata=LiteCUAMetadata(extra_tool_schemas=[
                    make_tool_schema(
                        "goto",
                        description="navigate",
                        parameters={
                            "type": "object",
                            "properties": {"url": {"type": "string"}},
                            "required": ["url"],
                            "additionalProperties": False,
                        },
                    )
                ]),
            model_id="gpt-5.5",
            api_kwargs={"max_output_tokens": 4096},
            function_tool_strict=True,
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        tools = mock.call_args.kwargs["tools"]
        fn_tool = next(t for t in tools if t.get("name") == "goto")
        assert fn_tool.get("strict") is True

    async def test_kwarg_true_omits_strict_when_schema_is_not_closed(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(
            metadata=LiteCUAMetadata(extra_tool_schemas=[
                    make_tool_schema(
                        "goto",
                        description="navigate",
                        parameters={
                            "type": "object",
                            "properties": {"url": {"type": "string"}},
                            "required": ["url"],
                        },
                    )
                ]),
            model_id="gpt-5.5",
            function_tool_strict=True,
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        tools = mock.call_args.kwargs["tools"]
        fn_tool = next(t for t in tools if t.get("name") == "goto")
        assert "strict" not in fn_tool


async def test_gpt_finish_exposure_uses_extra_tool_schemas_not_valid_actions(monkeypatch) -> None:
    valid_only = await openai_tools_sent(
        GPTDesktopUseAgent(
            metadata=LiteCUAMetadata(dims=("desktop", "use"), valid_actions=["click"])
        ),
        monkeypatch,
    )
    assert openai_provider_tool_names(valid_only).isdisjoint({"response", "terminate"})

    with_schema = await openai_tools_sent(
        GPTDesktopUseAgent(
            metadata=LiteCUAMetadata(dims=("desktop", "use"), extra_tool_schemas=[RESPONSE_SCHEMA])
        ),
        monkeypatch,
    )
    assert "response" in openai_provider_tool_names(with_schema)


async def test_gpt_mobile_finish_exposure_uses_extra_tool_schemas_not_valid_actions(
    monkeypatch,
) -> None:
    valid_only = await openai_tools_sent(
        GPTMobileUseAgent(metadata=LiteCUAMetadata(dims=("mobile", "use"), valid_actions=["tap"])),
        monkeypatch,
        FakeMobileEnv(terminate_after=1),
    )
    assert openai_provider_tool_names(valid_only).isdisjoint({"response", "terminate"})

    with_schema = await openai_tools_sent(
        GPTMobileUseAgent(
            metadata=LiteCUAMetadata(
                dims=("mobile", "use"),
                extra_tool_schemas=[RESPONSE_SCHEMA],
            )
        ),
        monkeypatch,
        FakeMobileEnv(terminate_after=1),
    )
    assert "response" in openai_provider_tool_names(with_schema)
    response_tool = next(t for t in with_schema if t["name"] == "response")
    assert "function" not in response_tool


async def test_gpt_grounding_point_valid_actions_point_controls_click_schema(
    monkeypatch,
) -> None:
    async def names(valid_actions, extra_tool_schemas=None):
        return openai_provider_tool_names(
            await openai_tools_sent(
                GPTDesktopGroundingPointAgent(
                    metadata=LiteCUAMetadata(
                        dims=("desktop", "grounding.point"),
                        valid_actions=valid_actions,
                        extra_tool_schemas=extra_tool_schemas or [],
                    )
                ),
                monkeypatch,
            )
        )

    assert "click" in await names(None)
    assert "click" in await names(["point"])
    assert "click" not in await names([])
    assert "click" not in await names(["click"])
    assert await names([], [BASH_SCHEMA]) == {"bash"}


# -----------------------------------------------------------------------------
# Safety-check auto-ack (kwarg removed — always auto-ack, let agent decide)
# -----------------------------------------------------------------------------


class TestSafetyCheckAutoAck:
    """Pending safety checks are echoed back unconditionally. We don't gate
    on an operator-approval flag — the agent should self-direct.
    """

    async def test_pending_checks_always_acked(self, monkeypatch):
        pending = [{"id": "sc_1", "code": "suspicious_url", "message": "check"}]
        first_resp = _fake_response(
            [
                {
                    "type": "computer_call",
                    "call_id": "call_1",
                    "actions": [{"type": "screenshot"}],
                    "pending_safety_checks": pending,
                }
            ]
        )
        second_resp = _fake_response()

        mock = AsyncMock(side_effect=[first_resp, second_resp])
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(model_id="gpt-5.5")
        await agent.sample(_FakeEnv(terminate_after=2), max_steps=5)

        second_input = mock.call_args_list[1].kwargs["input"]
        ack_items = [
            item
            for item in second_input
            if item.get("type") == "computer_call_output"
            and item.get("acknowledged_safety_checks") == pending
        ]
        assert ack_items, f"Expected unconditional ack echo, got: {second_input}"
