"""Tests for the ``raw_response`` byte-verbatim short-circuit in
``BaseAgentAdapter.convert_message_to_agent``.

Six groups:
  A. Short-circuit path correctness (matching adapter_key → text-only output).
  B. adapter_key fingerprint behavior (mismatch → canonical fallback + warn).
  C. End-to-end byte fidelity via real chat_template (Qwen3-VL-Thinking / Instruct / MAI-UI).
  F. Contract / invariants (stale raw, pop-to-canonical, all subclasses renamed).

Run: uv run pytest tests/agents/core/adapter/test_raw_response.py -v
"""

from __future__ import annotations

import copy
import logging

import pytest

from lite.agents.bootstrap import register_all
from lite.agents.core.adapter import AgentAdapterRegistry, AsIsAdapter
from lite.agents.core.agent.utils.messages import (
    build_tool_result_message,
)
from lite.agents.models.qwen3_vl.adapter import Qwen3VLDesktopUseAdapter
from lite.core.messages import no_tool_call_final_text
from lite.core.tools import make_tool_call
from lite.core.tools.results import TOOL_RESULT_ERROR_SECTION_HEADER

register_all()

# =============================================================================
# A. Short-circuit path correctness
# =============================================================================


def _make_qwen3_vl_adapter():
    return Qwen3VLDesktopUseAdapter()


def _computer_call(
    arguments: dict | None = None,
    *,
    action: str = "click",
    call_id: str | None = None,
) -> dict:
    return make_tool_call(
        "computer",
        {"actions": [{"action": action, **(arguments or {})}]},
        call_id=call_id,
    )


def _mobile_call(
    arguments: dict | None = None,
    *,
    action: str = "tap",
    call_id: str | None = None,
) -> dict:
    return make_tool_call(
        "mobile",
        {"actions": [{"action": action, **(arguments or {})}]},
        call_id=call_id,
    )


_RAW_QWEN3_VL_CLICK = (
    "Action: click\n"
    '<tool_call>{"name":"computer_use","arguments":'
    '{"action":"left_click","coordinate":[1,2]}}</tool_call>'
)


class TestShortCircuit:
    def test_assistant_matching_adapter_key_short_circuits(self):
        """Matching raw_response → minimal {role, content=[text]} with
        ``tool_calls`` and ``reasoning_content`` **dropped** (chat_template
        will not re-emit them; raw string embeds the `<tool_call>` / `<think>`
        tokens verbatim)."""
        adapter = _make_qwen3_vl_adapter()
        raw = '<think>planning</think>Action: click\n<tool_call>{"name":"x"}</tool_call>'
        out = adapter.convert_message_to_agent(
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": "click"}],
                "tool_calls": [_computer_call()],
                "reasoning_content": "planning",
                "raw_response": {"text": raw, "adapter_key": adapter._registry_key},
            }
        )
        assert out == {
            "role": "assistant",
            "content": [{"type": "text", "text": raw}],
        }
        assert "tool_calls" not in out
        assert "reasoning_content" not in out

    def test_registered_pattern_sidecar_matches_concrete_runtime_key(self):
        """Runtime replay uses the public registry matcher, including the
        registered-pattern → concrete-key direction used by export."""
        adapter = _make_qwen3_vl_adapter()
        adapter._registry_key = "qwen3_vl@desktop@use"
        raw = '<tool_call>{"name":"x"}</tool_call>'
        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "click"}],
            "tool_calls": [_computer_call()],
            "raw_response": {
                "text": raw,
                "adapter_key": r"qwen3_vl@(desktop|browser)@use",
            },
        }

        out = adapter.convert_message_to_agent(msg)

        assert out == {
            "role": "assistant",
            "content": [{"type": "text", "text": raw}],
        }

    def test_matching_raw_response_does_not_validate_tool_availability(self):
        """Raw replay is a render shortcut; env/data validation owns whether
        the Lite message's canonical tool was advertised."""
        adapter = _make_qwen3_vl_adapter()
        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "ask"}],
            "tool_calls": [
                make_tool_call("ask_user", {"question": "Continue?"}),
            ],
            "raw_response": {
                "text": "VERBATIM RAW SHOULD NOT MASK THE BAD TOOL SURFACE",
                "adapter_key": adapter._registry_key,
            },
        }

        assert adapter.convert_message_to_agent(msg) == {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "VERBATIM RAW SHOULD NOT MASK THE BAD TOOL SURFACE",
                }
            ],
        }

    def test_user_msg_does_not_short_circuit(self):
        """Role guard: raw_response keyed on a user message must not trigger
        short-circuit (only assistant turns have raw output)."""
        adapter = _make_qwen3_vl_adapter()
        user_msg = {
            "role": "user",
            "content": [{"type": "text", "text": "hi"}],
            "raw_response": {"text": "ignored", "adapter_key": adapter._registry_key},
        }
        out = adapter.convert_message_to_agent(user_msg)
        # falls through to _convert_message_to_agent which returns the user
        # message effectively unchanged (no short-circuit collapse)
        assert out["role"] == "user"
        # content is preserved as-is (not collapsed to "ignored")
        assert out.get("content") != [{"type": "text", "text": "ignored"}]

    def test_tool_result_msg_does_not_short_circuit_or_drop_projected_error(self):
        """Role guard for per-call tool feedback: even with a stray sidecar,
        role:tool error text remains the model-visible projected observation."""
        adapter = _make_qwen3_vl_adapter()
        tool_msg = build_tool_result_message(
            "call_0000",
            image_indices=(),
            text="before failure",
            error="invalid arguments for click",
        )
        tool_msg["raw_response"] = {
            "text": "ignored raw replay",
            "adapter_key": adapter._registry_key,
        }

        out = adapter.convert_message_to_agent(tool_msg)

        assert out["role"] == "tool"
        assert out["tool_call_id"] == "call_0000"
        assert out["content"] == [
            {
                "type": "text",
                "text": (
                    "before failure\n\n"
                    f"{TOOL_RESULT_ERROR_SECTION_HEADER}\n"
                    "invalid arguments for click"
                ),
            }
        ]

    def test_no_raw_response_falls_through(self):
        """No raw_response → canonical _convert_message_to_agent output."""
        adapter = _make_qwen3_vl_adapter()
        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "click"}],
            "tool_calls": [_computer_call({"coordinate": [10, 20]})],
        }
        public = adapter.convert_message_to_agent(copy.deepcopy(msg))
        private = adapter._convert_message_to_agent(copy.deepcopy(msg))
        assert public == private

    def test_asis_adapter_bypasses_short_circuit(self):
        """AsIsAdapter override: preserves tool_calls / reasoning_content
        even when raw_response is present (pass-through semantics)."""
        adapter = AsIsAdapter()
        raw_text = "SOMETHING ELSE"
        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "x"}],
            "tool_calls": [_computer_call()],
            "reasoning_content": "keep me",
            "raw_response": {"text": raw_text, "adapter_key": adapter._registry_key},
        }
        out = adapter.convert_message_to_agent(msg)
        assert out["tool_calls"] == msg["tool_calls"]
        assert out["reasoning_content"] == "keep me"
        # content is deep-copied from input, not collapsed to the raw string
        assert out["content"] == msg["content"]


# =============================================================================
# B. adapter_key fingerprint behavior
# =============================================================================


class TestRegistryMatcher:
    def test_exact_literal_dot_keys_do_not_act_like_regex(self):
        literal = "qwen3_vl@mobile@grounding.point"
        assert literal in AgentAdapterRegistry.list()

        assert AgentAdapterRegistry.raw_response_key_matches(literal, literal)
        assert not AgentAdapterRegistry.raw_response_key_matches(
            literal,
            "qwen3_vl@mobile@groundingXpoint",
        )
        assert not AgentAdapterRegistry.raw_response_key_matches(
            "qwen3_vl@mobile@groundingXpoint",
            literal,
        )

    def test_registered_pattern_matches_in_both_directions(self):
        pattern = r"qwen3_vl@(desktop|browser)@use"
        concrete = "qwen3_vl@desktop@use"
        assert pattern in AgentAdapterRegistry.list_patterns()

        assert AgentAdapterRegistry.raw_response_key_matches(pattern, concrete)
        assert AgentAdapterRegistry.raw_response_key_matches(concrete, pattern)

    def test_unregistered_regex_looking_keys_do_not_authorize_replay(self):
        unregistered_pattern = r"qwen3_vl@(desktop|browser)@use-unregistered"

        assert not AgentAdapterRegistry.raw_response_key_matches(
            unregistered_pattern,
            "qwen3_vl@desktop@use-unregistered",
        )
        assert not AgentAdapterRegistry.raw_response_key_matches(
            "qwen3_vl@desktop@use-unregistered",
            unregistered_pattern,
        )


class TestFingerprint:
    def test_mismatched_adapter_key_falls_back_to_canonical(self, caplog):
        """raw_response present but adapter_key wrong → canonical output (with
        tool_calls etc.) + one warning."""
        adapter = _make_qwen3_vl_adapter()
        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "click"}],
            "tool_calls": [_computer_call({"coordinate": [10, 20]})],
            "raw_response": {
                "text": "verbatim",
                "adapter_key": "test_raw_response:mismatched_adapter_key",
            },
        }
        with caplog.at_level(logging.WARNING, logger="lite.agents.core.adapter"):
            out = adapter.convert_message_to_agent(copy.deepcopy(msg))
        # (a) canonical output: tool_calls retained (qwen3_vl's canonical form
        # converts them via action_space; presence is what matters here).
        assert "tool_calls" in out
        # (b) exactly one mismatch warning recorded.
        mismatch_records = [r for r in caplog.records if "adapter_key mismatch" in r.getMessage()]
        assert len(mismatch_records) == 1

    def test_mismatch_warning_dedupes(self, caplog):
        """Same (expected, actual) pair: only the first call emits; later
        calls do not repeat the warning."""
        adapter = _make_qwen3_vl_adapter()
        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "c"}],
            "tool_calls": [_computer_call({"coordinate": [1, 2]})],
            "raw_response": {
                "text": "v",
                "adapter_key": "test_raw_response:dedupe",
            },
        }
        with caplog.at_level(logging.WARNING, logger="lite.agents.core.adapter"):
            for _ in range(5):
                adapter.convert_message_to_agent(copy.deepcopy(msg))
        mismatch_records = [r for r in caplog.records if "adapter_key mismatch" in r.getMessage()]
        assert len(mismatch_records) == 1

    def test_cross_platform_mismatches(self, caplog):
        """adapter_key is the FULL registry key. Rollout on the desktop+browser
        navigation key saves that full key; replay via
        ``qwen3_vl@mobile@use`` has a different registry key →
        mismatch → canonical fallback. Strict match by design: different
        platform may mean different chat_template / action_space."""
        from lite.agents.models.qwen3_vl.adapter import Qwen3VLMobileUseAdapter

        desktop_adapter = _make_qwen3_vl_adapter()
        mobile_adapter = Qwen3VLMobileUseAdapter()

        # desktop+browser share one collapsed navigation key (tools refactor); mobile
        # is its own. _registry_key returns the verbatim registered key.
        assert desktop_adapter._registry_key == r"qwen3_vl@(desktop|browser)@use"
        assert mobile_adapter._registry_key == "qwen3_vl@mobile@use"
        assert desktop_adapter._registry_key != mobile_adapter._registry_key

        raw = '<tool_call>{"name":"anything"}</tool_call>'
        # Mobile-shaped canonical message with a sidecar stamped by the desktop
        # adapter, forcing mismatch and canonical fallback.
        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "tap"}],
            "tool_calls": [_mobile_call({"coordinate": [10, 20]})],
            "raw_response": {"text": raw, "adapter_key": desktop_adapter._registry_key},
        }
        with caplog.at_level(logging.WARNING, logger="lite.agents.core.adapter"):
            out = mobile_adapter.convert_message_to_agent(copy.deepcopy(msg))
        # Canonical fallback — NOT the short-circuit shape.
        assert out != {
            "role": "assistant",
            "content": [{"type": "text", "text": raw}],
        }
        mismatch_records = [r for r in caplog.records if "adapter_key mismatch" in r.getMessage()]
        assert len(mismatch_records) == 1

    def test_cross_model_adapter_key_mismatch(self, caplog):
        """Different models have different registry keys → mismatch →
        canonical fallback (e.g. rollout by qwen3_vl, SFT with ui_tars)."""
        from lite.agents.models.ui_tars.adapter import UITarsDesktopUseAdapter

        ui_tars_adapter = UITarsDesktopUseAdapter()

        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "click"}],
            "tool_calls": [_computer_call({"coordinate": [1, 2]})],
            "raw_response": {
                "text": "qwen3 output",
                "adapter_key": r"qwen3_vl@(desktop|browser)@use",
            },
        }
        assert ui_tars_adapter._registry_key == r"ui_tars@(desktop|browser)@use"
        with caplog.at_level(logging.WARNING, logger="lite.agents.core.adapter"):
            out = ui_tars_adapter.convert_message_to_agent(copy.deepcopy(msg))
        # Canonical path (UI-TARS plain text wire format): the short-circuit
        # would have returned exactly {"role", "content": [{"type": "text",
        # "text": "qwen3 output"}]}; canonical output is different.
        assert out != {
            "role": "assistant",
            "content": [{"type": "text", "text": "qwen3 output"}],
        }
        mismatch_records = [r for r in caplog.records if "adapter_key mismatch" in r.getMessage()]
        assert len(mismatch_records) == 1

    def test_missing_adapter_key_treated_as_mismatch(self, caplog):
        """raw_response with only ``text`` (no ``adapter_key``, e.g. corrupted
        data) → treat as mismatch, fall back to canonical, warn."""
        adapter = _make_qwen3_vl_adapter()
        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "c"}],
            "tool_calls": [_computer_call({"coordinate": [1, 2]})],
            "raw_response": {"text": "v"},  # missing adapter_key
        }
        with caplog.at_level(logging.WARNING, logger="lite.agents.core.adapter"):
            out = adapter.convert_message_to_agent(copy.deepcopy(msg))
        assert "tool_calls" in out  # canonical output
        mismatch_records = [r for r in caplog.records if "adapter_key mismatch" in r.getMessage()]
        assert len(mismatch_records) == 1


# =============================================================================
# F. Contract / invariants
# =============================================================================


class TestContract:
    def test_mutating_tool_calls_without_pop_returns_stale_raw(self):
        """**Documented contract**: if you mutate ``tool_calls`` /
        ``content`` / ``reasoning_content`` without popping ``raw_response``,
        short-circuit still returns the ORIGINAL raw text — the structural
        edit is invisible. Future refactors that break this invariant see
        this named test fail and know to update the contract doc
        accordingly.
        """
        adapter = _make_qwen3_vl_adapter()
        original_raw = (
            "Action: original\n"
            '<tool_call>{"name":"computer_use","arguments":'
            '{"action":"left_click","coordinate":[1,1]}}</tool_call>'
        )
        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "original"}],
            "tool_calls": [_computer_call({"coordinate": [1, 1]})],
            "raw_response": {"text": original_raw, "adapter_key": adapter._registry_key},
        }
        # Caller mutates tool_calls but forgets to pop raw_response:
        msg["tool_calls"] = [_computer_call({"coordinate": [9, 9]})]
        out = adapter.convert_message_to_agent(msg)
        # Short-circuit fires → returns original raw text, ignoring mutation.
        assert out["content"][0]["text"] == original_raw
        assert "[9,9]" not in out["content"][0]["text"]
        assert "[9, 9]" not in out["content"][0]["text"]

    def test_pop_raw_response_reverts_to_canonical(self):
        """Conversely, popping raw_response before converting lets structural
        edits flow through the canonical path."""
        adapter = _make_qwen3_vl_adapter()
        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "original"}],
            "tool_calls": [_computer_call({"coordinate": [1, 1]})],
            "raw_response": {
                "text": "<tool_call>original</tool_call>",
                "adapter_key": adapter._registry_key,
            },
        }
        msg["tool_calls"] = [_computer_call({"coordinate": [5, 5]})]
        msg.pop("raw_response", None)  # explicit invalidation
        out = adapter.convert_message_to_agent(msg)
        # canonical path renders via action_space; tool_calls should reflect
        # the mutation (presence & structural form).
        assert "tool_calls" in out
        # The mutated arguments coordinate survives the canonical conversion.
        args = out["tool_calls"][0]["arguments"]
        # qwen3_vl canonical form wraps into computer_use(action=..., coordinate=...);
        # the coordinate [5, 5] round-trips through the action_space.
        assert args.get("coordinate") == [5, 5] or args == {"coordinate": [5, 5]}


class TestAgentFlag:
    """``AdapterBasedAgent.preserve_raw_response`` switch controls whether
    rollout attaches the sidecar. When False, assistant messages in the
    trajectory carry NO ``raw_response`` field, forcing all downstream
    conversion through the canonical path (pre-refactor behavior).
    """

    @pytest.mark.asyncio
    async def test_preserve_true_attaches_raw_response(self):
        from lite.agents.models import AgentRegistry

        async def _fake_generate(**kwargs):
            return {"response": _RAW_QWEN3_VL_CLICK}

        agent = AgentRegistry.get(
            "qwen3_vl@desktop@use",
            generate_fn=_fake_generate,
            processor=None,
            # preserve_raw_response defaults to True
        )

        # Hand-build a minimal sample so _predict_with_details goes through
        # parse → convert_from → attach.
        from lite.core import LiteCUAMetadata, LiteSample

        sample = LiteSample(
            metadata=LiteCUAMetadata(
                dims=(LiteCUAMetadata.Platform.DESKTOP, LiteCUAMetadata.TaskType.USE)
            ),
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        )

        # Stub build_generation_prompt to avoid needing a real processor.
        agent.build_generation_prompt = lambda messages: "stub"
        result = await agent._predict_with_details(sample, processed_images=[])

        assert "raw_response" in result.lite_message
        # adapter_key is the full registry key (collapsed desktop+browser after the
        # tools refactor).
        assert (
            result.lite_message["raw_response"]["adapter_key"] == r"qwen3_vl@(desktop|browser)@use"
        )
        assert result.lite_message["raw_response"]["text"].startswith("Action: click")

    @pytest.mark.asyncio
    async def test_preserve_false_skips_attach(self):
        """preserve_raw_response=False → no raw_response on produced
        lite_message, everything else identical. Canonical path guaranteed
        on downstream reads."""
        from lite.agents.models import AgentRegistry

        async def _fake_generate(**kwargs):
            return {"response": _RAW_QWEN3_VL_CLICK}

        agent = AgentRegistry.get(
            "qwen3_vl@desktop@use",
            generate_fn=_fake_generate,
            processor=None,
            preserve_raw_response=False,
        )

        from lite.core import LiteCUAMetadata, LiteSample

        sample = LiteSample(
            metadata=LiteCUAMetadata(
                dims=(LiteCUAMetadata.Platform.DESKTOP, LiteCUAMetadata.TaskType.USE)
            ),
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        )
        agent.build_generation_prompt = lambda messages: "stub"
        result = await agent._predict_with_details(sample, processed_images=[])

        assert "raw_response" not in result.lite_message
        # Structural fields still produced — canonical path unaffected.
        assert result.lite_message["role"] == "assistant"
        assert "tool_calls" in result.lite_message


class TestRoundTripFinalText:
    @pytest.mark.parametrize(
        "adapter_key",
        [
            "lite@desktop@use",
            "lite@mobile@use",
            "qwen2_5_vl@desktop@use",
            "qwen3_vl@desktop@use",
            "qwen3_5@desktop@use",
            "mai_ui@mobile@use",
        ],
    )
    def test_no_tool_call_final_text_round_trips_without_retag(self, adapter_key):
        """``from_agent(to_agent(x))`` must keep final prose in plain text.

        If this regresses to ``action_description`` / reasoning-only content,
        terminal prose becomes invisible to ``no_tool_call_final_text`` and the
        turn looks like an empty final.
        """
        adapter = AgentAdapterRegistry.get(adapter_key)
        source = {
            "role": "assistant",
            "content": [{"type": "text", "text": "Final answer from teacher."}],
        }

        rendered = adapter.convert_message_to_agent(copy.deepcopy(source))
        back = adapter.convert_message_from_agent(copy.deepcopy(rendered))

        assert not back.get("tool_calls")
        assert back.get("content") == source["content"]
        assert no_tool_call_final_text(back) == "Final answer from teacher."


class TestAllSubclassesRenamed:
    """CI-level invariant: all registered adapter subclasses implement the
    protected hook ``_convert_message_to_agent``. Public
    ``convert_message_to_agent`` must resolve to
    ``BaseAgentAdapter.convert_message_to_agent`` (the wrapper doing the
    fingerprint short-circuit), except for ``AsIsAdapter`` which deliberately
    overrides the wrapper for pass-through semantics.
    """

    def test_all_registered_adapters_have_private_hook(self):
        # Iterate exact-match items AND regex patterns so the invariant covers
        # EVERY registered adapter class. (Literal-dot keys like
        # ``grounding.point`` are EXACT keys — in ``list()`` — while genuine
        # regex keys like ``qwen3_vl@(desktop|browser)@grounding.action`` are in
        # ``list_patterns()``; union both to be exhaustive.)
        items = AgentAdapterRegistry.list()
        patterns = AgentAdapterRegistry.list_patterns()
        keys = list(items) + list(patterns)
        assert len(keys) > 10  # sanity check — we registered ~30+ adapters
        for key in keys:
            cls = AgentAdapterRegistry.get_class(key)
            # 1. Private hook is callable
            assert hasattr(cls, "_convert_message_to_agent"), (
                f"{cls.__name__} ({key}): missing _convert_message_to_agent hook"
            )
            # 2. Public wrapper comes from the base class, EXCEPT AsIsAdapter
            if cls is AsIsAdapter:
                continue
            qualname = cls.convert_message_to_agent.__qualname__
            assert qualname.startswith("BaseAgentAdapter."), (
                f"{cls.__name__} ({key}): convert_message_to_agent resolves "
                f"to {qualname!r}, expected BaseAgentAdapter.convert_message_to_agent. "
                f"Did you override the public method instead of the _convert_message_to_agent hook?"
            )

    def test_asis_adapter_overrides_public(self):
        """Sanity-check the one exception."""
        qualname = AsIsAdapter.convert_message_to_agent.__qualname__
        assert qualname == "AsIsAdapter.convert_message_to_agent"
