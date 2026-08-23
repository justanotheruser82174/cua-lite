"""BrowserGymGenericProtocol rendering and message-shaping tests."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("browsergym.core", reason="browsergym not installed")

import lite.core.tools.results as tool_results
from lite.agents.extensions.browsergym.protocol import (
    BrowserGymGenericProtocol,
    _extract_section_after,
    _render_history,
    _render_instructions,
    _render_observation,
    render_tool_call_json,
    render_tool_call_xml,
)
from lite.core.tools import make_tool_call
from lite.core.tools.results import TOOL_RESULT_ERROR_SECTION_HEADER

# ---------------------------------------------------------------------------
# BrowserGymGenericProtocol
# ---------------------------------------------------------------------------


class TestExtractSectionAfter:
    """Helper used by the protocol to pull AXTree/HTML/Focused-element blocks."""

    def test_simple(self):
        text = "goal\n## AXTree:\nbody\n## Other:\nrest"
        # Cut at the next "## " marker, then lstrip leading newlines.
        assert _extract_section_after(text, "## AXTree:") == "body"

    def test_no_following_section(self):
        text = "goal\n## AXTree:\nrest of text"
        assert _extract_section_after(text, "## AXTree:") == "rest of text"

    def test_missing_header(self):
        assert _extract_section_after("just text", "## AXTree:") is None

    def test_none_or_empty_input(self):
        assert _extract_section_after(None, "## X") is None
        assert _extract_section_after("", "## X") is None


class TestRenderInstructions:
    def test_includes_goal(self):
        out = _render_instructions("find the cheapest hat")
        assert "# Instructions" in out
        assert "## Goal:" in out
        assert "find the cheapest hat" in out


class TestRenderObservation:
    def test_axtree_only(self):
        out = _render_observation(axtree_txt="[a47] BUTTON 'OK'\n")
        assert "## AXTree:" in out
        assert "[bid]" in out  # bid info note
        assert "[a47] BUTTON 'OK'" in out
        assert "## HTML:" not in out

    def test_focused_bid(self):
        out = _render_observation(focused_bid="a47")
        assert "## Focused element:" in out
        assert "bid='a47'" in out

    def test_focused_none(self):
        out = _render_observation(focused_bid=None)
        assert "## Focused element:" in out
        assert "None" in out

    def test_action_error(self):
        out = _render_observation(last_action_error="Error: timeout")
        assert TOOL_RESULT_ERROR_SECTION_HEADER in out
        assert "Error: timeout" in out

    def test_no_action_error_no_section(self):
        out = _render_observation(last_action_error=None)
        assert TOOL_RESULT_ERROR_SECTION_HEADER not in out

    def test_disable_focused_section(self):
        out = _render_observation(focused_bid="a47", use_focused_element=False)
        assert "## Focused element:" not in out

    def test_tabs_section_before_axtree(self):
        # `## Currently open tabs:` must precede `## AXTree:` so the model
        # parses tab context before reading the page (matches AgentLab order).
        tabs = (
            "Tab 0 (active tab):\n    Title: Google\n    URL: https://google.com/\n"
            "Tab 1:\n    Title: WA Shopping\n    URL: http://localhost:7770/\n"
        )
        out = _render_observation(axtree_txt="[a47] BUTTON 'OK'\n", tabs_txt=tabs)
        assert "## Currently open tabs:" in out
        assert out.index("## Currently open tabs:") < out.index("## AXTree:")
        assert "Tab 0 (active tab):" in out
        assert "Tab 1:" in out

    def test_no_tabs_section_when_absent(self):
        out = _render_observation(axtree_txt="[a47] BUTTON")
        assert "## Currently open tabs:" not in out


class TestRenderToolCallJson:
    """Default ``render_tool_call`` — Qwen3-VL's JSON ``<tool_call>`` wire block."""

    def test_renders_json_object_in_tool_call(self):
        out = render_tool_call_json(make_tool_call("click", {"bid": "a47"}, call_id="call_0"))
        # Wire format byte-matches the system prompt's documented example.
        assert out == '<tool_call>\n{"name": "click", "arguments": {"bid": "a47"}}\n</tool_call>'

    def test_bare_agent_wire_is_not_silently_rendered_empty(self):
        # Pre-refactor this renderer dug through bare agent-wire fields and fell
        # back to ``{"name": "", "arguments": {}}`` when the shape didn't match
        # -- an agent-wire call therefore rendered as a valid-looking but EMPTY
        # tool_call. It now consumes nested Lite calls and fails loudly on the
        # missing canonical key instead.
        with pytest.raises(KeyError):
            render_tool_call_json({"name": "fill", "arguments": {"bid": "a2"}})

    def test_string_arguments_are_not_reparsed(self):
        # The old JSON-string repair path (``json.loads`` on non-dict
        # ``arguments``) is gone by design: parsing/repair belongs before this
        # raw adapter-wire renderer. Guard that no repair layer creeps back in.
        out = render_tool_call_json(
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "fill", "arguments": '{"bid": "a2"}'},
            }
        )
        assert '"arguments": {"bid": "a2"}' not in out
        assert '"arguments": "{\\"bid\\": \\"a2\\"}"' in out


class TestRenderToolCallXml:
    """``tool_call_format="xml"`` — Qwen3.5's XML ``<tool_call>`` wire block."""

    def test_renders_xml_function_block(self):
        out = render_tool_call_xml(
            make_tool_call("fill", {"bid": "a2", "value": "hi"}, call_id="call_0")
        )
        # Byte-matches Qwen3.5's ``_render_xml_tool_call`` format.
        assert out == (
            "<tool_call>\n<function=fill>\n"
            "<parameter=bid>\na2\n</parameter>\n"
            "<parameter=value>\nhi\n</parameter>\n"
            "</function>\n</tool_call>"
        )

    def test_bool_param_lowercased(self):
        out = render_tool_call_xml(
            make_tool_call("fill", {"bid": "a2", "enable": True}, call_id="call_0")
        )
        assert "<parameter=enable>\ntrue\n</parameter>" in out


class TestRenderHistory:
    def test_empty_history(self):
        assert _render_history([]) == ""

    def test_single_step_places_block_under_header(self):
        # ``_render_history`` receives the ALREADY wire-rendered ``<tool_call>``
        # block (the renderer adds the wrapper, not ``_render_history``) and
        # simply places it under a ``## step N`` header.
        block = render_tool_call_json(make_tool_call("click", {"bid": "a47"}, call_id="call_0"))
        out = _render_history([block])
        assert "## step 0" in out
        assert "<tool_call>" in out
        assert '{"name": "click", "arguments": {"bid": "a47"}}' in out
        assert "</tool_call>" in out
        # Paper's <action> wrapper must NOT appear (we use tool_call).
        assert "<action>" not in out

    def test_multiple_steps_indexed(self):
        blocks = [
            render_tool_call_json(make_tool_call("click", {"bid": "a1"}, call_id="call_0")),
            render_tool_call_json(
                make_tool_call("fill", {"bid": "a2", "value": "hi"}, call_id="call_1")
            ),
        ]
        out = _render_history(blocks)
        assert "## step 0" in out
        assert "## step 1" in out

    def test_render_history_does_not_add_wrapper(self):
        # ``_render_history`` no longer wraps — it expects a pre-rendered block.
        # Passing a single block yields exactly one tool_call pair (no double-wrap).
        block = render_tool_call_json(make_tool_call("click", {"bid": "a47"}, call_id="call_0"))
        out = _render_history([block])
        assert out.count("<tool_call>") == 1
        assert out.count("</tool_call>") == 1


class TestProtocol:
    """End-to-end ``process_messages`` shape & content."""

    def _proto(self, **kwargs: Any) -> BrowserGymGenericProtocol:
        return BrowserGymGenericProtocol(**kwargs)

    def test_empty_messages(self):
        assert self._proto().process_messages([]) == []

    def test_system_only(self):
        sys_msg = {"role": "system", "content": [{"type": "text", "text": "sys"}]}
        out = self._proto().process_messages([sys_msg])
        # System preserved + a (mostly empty) user message rebuilt.
        assert len(out) == 2
        assert out[0]["role"] == "system"
        assert out[1]["role"] == "user"

    def test_no_system_no_extra_msg(self):
        # No system → output is just [user_msg].
        user_msg = {"role": "user", "content": [{"type": "text", "text": "find X"}]}
        out = self._proto().process_messages([user_msg])
        assert len(out) == 1
        assert out[0]["role"] == "user"

    def test_goal_extracted_from_first_user(self):
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": "find the cheapest hat\n## AXTree:\nbody"}],
            },
        ]
        out = self._proto().process_messages(msgs)
        user_text = next(c["text"] for c in out[1]["content"] if c.get("type") == "text")
        # Goal section captures only up to the AXTree marker.
        assert "## Goal:\nfind the cheapest hat" in user_text
        # AXTree body still rendered in the Observation section.
        assert "body" in user_text

    def test_goal_extracted_from_string_user_content(self):
        msgs = [
            {"role": "user", "content": "find the cheapest hat\n## AXTree:\nbody"},
        ]
        out = self._proto().process_messages(msgs)
        user_text = next(c["text"] for c in out[0]["content"] if c.get("type") == "text")
        assert "## Goal:\nfind the cheapest hat" in user_text
        assert "body" in user_text

    def test_multiline_goal_preserved(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "find X:\nInput image 1/1 below\n## AXTree:\nax"}
                ],
            },
        ]
        out = self._proto().process_messages(msgs)
        user_text = next(c["text"] for c in out[0]["content"] if c.get("type") == "text")
        # Multi-line goal preserved (was previously truncated at the first \n).
        assert "find X:\nInput image 1/1 below" in user_text

    def test_history_renders_actions_from_tool_calls(self):
        # History renders each past action from the STRUCTURED ``tool_calls`` —
        # the source of truth the env actually executes (the agent loop reads
        # ``lite_message["tool_calls"]`` for ``env.step``) — in the adapter's
        # wire format (JSON ``<tool_call>`` by default), NOT the raw model text.
        # The JSON body must match what the system prompt + tools section demand
        # so the history examples reinforce (not contradict) the live format.
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "goal\n## AXTree:\nbody"}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ""}],
                "tool_calls": [make_tool_call("click", {"bid": "a47"}, call_id="call_0")],
            },
            {"role": "user", "content": [{"type": "text", "text": "## AXTree:\nbody2"}]},
        ]
        out = self._proto().process_messages(msgs)
        user_text = next(c["text"] for c in out[-1]["content"] if c.get("type") == "text")
        assert "# History of interaction with the task:" in user_text
        # Rendered as the JSON wire format (default), NOT bare ``click('a47')``.
        assert '{"name": "click", "arguments": {"bid": "a47"}}' in user_text
        assert "click('a47')" not in user_text
        # Latest obs (body2) should be in the Observation section.
        assert "body2" in user_text

    def test_tool_role_observation_updates_current_obs_and_images(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "goal\n## AXTree:\nold_body"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ""}],
                "tool_calls": [make_tool_call("click", {"bid": "a47"}, call_id="call_0")],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0",
                "content": [
                    {"type": "image", "index": 1},
                    {"type": "text", "text": "## AXTree:\nnew_body"},
                ],
            },
        ]
        out = self._proto().process_messages(msgs)
        content = out[-1]["content"]
        user_text = next(c["text"] for c in content if c.get("type") == "text")
        image_parts = [c for c in content if c.get("type") == "image"]

        assert image_parts == [{"type": "image", "index": 1}]
        assert "new_body" in user_text
        assert "old_body" not in user_text
        assert '{"name": "click", "arguments": {"bid": "a47"}}' in user_text

    def test_history_xml_format_via_config(self):
        # XML-format adapters (Qwen3.5) set ``tool_call_format: "xml"`` in their
        # browsergym config so history matches their
        # ``<tool_call><function=...>...</tool_call>`` wire format, NOT the JSON
        # default. Threaded via ``protocol_kwargs`` — no shared adapter code.
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "goal\n## AXTree:\nbody"}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ""}],
                "tool_calls": [make_tool_call("click", {"bid": "a47"}, call_id="call_0")],
            },
            {"role": "user", "content": [{"type": "text", "text": "## AXTree:\nbody2"}]},
        ]
        out = self._proto(tool_call_format="xml").process_messages(msgs)
        user_text = next(c["text"] for c in out[-1]["content"] if c.get("type") == "text")
        assert "<function=click>" in user_text  # XML format used
        assert "<parameter=bid>" in user_text
        assert '{"name": "click"' not in user_text  # NOT the JSON default

    def test_history_ignores_raw_response_reasoning_blob(self):
        # Bug B regression: a noop turn (the model emitted no parseable action →
        # empty tool_calls) whose ``raw_response.text`` is a giant reasoning blob
        # must NOT dump that blob into ``# History``. The action source is the
        # structured ``tool_calls``, never the raw text — so a turn with no
        # tool_calls contributes nothing.
        blob = "I should think about whether to click. " * 500  # ~20KB reasoning
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "goal\n## AXTree:\nbody"}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ""}],
                "raw_response": {"text": blob},  # noop turn: blob, no tool_calls
            },
            {"role": "user", "content": [{"type": "text", "text": "## AXTree:\nbody2"}]},
        ]
        out = self._proto().process_messages(msgs)
        user_text = next(c["text"] for c in out[-1]["content"] if c.get("type") == "text")
        assert "I should think about whether to click." not in user_text  # blob dropped
        assert "body2" in user_text  # current obs still rendered normally

    def test_history_synthesizes_from_tool_calls(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "goal"}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ""}],
                "tool_calls": [
                    make_tool_call("click", {"bid": "a47"}, call_id="call_0"),
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "obs"}]},
        ]
        out = self._proto().process_messages(msgs)
        user_text = next(c["text"] for c in out[-1]["content"] if c.get("type") == "text")
        assert '{"name": "click", "arguments": {"bid": "a47"}}' in user_text  # JSON wire form

    def test_focused_bid_reparsed_from_text(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "goal\n## AXTree:\n[a47] BUTTON\n\n## Focused element:\nbid='a47'",
                    }
                ],
            },
        ]
        out = self._proto().process_messages(msgs)
        user_text = next(c["text"] for c in out[0]["content"] if c.get("type") == "text")
        # The rendered Observation must include the bid (not None).
        assert "bid='a47'" in user_text

    def test_focused_bid_absent_suppresses_section(self):
        # When the env didn't emit a "## Focused element:" block (env-side
        # `use_focused_element=False`), the protocol must suppress the
        # section entirely instead of rendering `bid=None`. Mirrors the
        # env's flag through the input text alone — no extra plumbing.
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "goal\n## AXTree:\nax"}]},
        ]
        out = self._proto().process_messages(msgs)
        user_text = next(c["text"] for c in out[0]["content"] if c.get("type") == "text")
        assert "## Focused element:" not in user_text

    def test_action_error_reparsed_from_labeled_section(self):
        # `BrowserGymEnv.step` (main.py) emits the error as a labeled section
        # so the protocol can extract it via the same `_extract_section_after`
        # helper used for AXTree / tabs / focused-element. Earlier we used a
        # fragile heuristic ("Error" substring at head); the labeled format is
        # robust.
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{TOOL_RESULT_ERROR_SECTION_HEADER}\nelement not visible\n"
                        "## AXTree:\nax",
                    }
                ],
            },
        ]
        out = self._proto().process_messages(msgs)
        user_text = next(c["text"] for c in out[0]["content"] if c.get("type") == "text")
        assert TOOL_RESULT_ERROR_SECTION_HEADER in user_text
        assert "element not visible" in user_text

    def test_action_error_no_substring_heuristic(self):
        # Regression guard: the OLD heuristic required "Error" or "Traceback"
        # in the head — body like ``"timeout exceeded"`` (no "Error" substring,
        # no "Traceback" prefix) would silently drop. Verify the labeled-section
        # path catches arbitrary error bodies.
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{TOOL_RESULT_ERROR_SECTION_HEADER}\ntimeout exceeded\n"
                        "## AXTree:\nax",
                    }
                ],
            },
        ]
        out = self._proto().process_messages(msgs)
        user_text = next(c["text"] for c in out[0]["content"] if c.get("type") == "text")
        assert TOOL_RESULT_ERROR_SECTION_HEADER in user_text
        assert "timeout exceeded" in user_text

    def test_action_error_reparse_and_render_follow_owner_header(self, monkeypatch):
        sentinel_header = "## BrowserGym owner header sentinel:"
        monkeypatch.setattr(
            tool_results,
            "TOOL_RESULT_ERROR_SECTION_HEADER",
            sentinel_header,
        )
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{sentinel_header}\npatched timeout\n## AXTree:\nax"}
                ],
            },
        ]

        out = self._proto().process_messages(msgs)
        user_text = next(c["text"] for c in out[0]["content"] if c.get("type") == "text")

        assert sentinel_header in user_text
        assert "patched timeout" in user_text
        assert TOOL_RESULT_ERROR_SECTION_HEADER not in user_text

    def test_same_observation_group_keeps_multiple_action_errors(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "goal\n## AXTree:\nold"}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ""}],
                "tool_calls": [
                    make_tool_call("click", {"bid": "a1"}, call_id="call_a"),
                    make_tool_call("click", {"bid": "a2"}, call_id="call_b"),
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_a",
                "content": [
                    {
                        "type": "text",
                        "text": f"## AXTree:\nnew\n\n{TOOL_RESULT_ERROR_SECTION_HEADER}\n"
                        "first failed",
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_b",
                "content": [
                    {"type": "text", "text": f"{TOOL_RESULT_ERROR_SECTION_HEADER}\nsecond failed"}
                ],
            },
        ]

        out = self._proto().process_messages(msgs)
        user_text = next(c["text"] for c in out[0]["content"] if c.get("type") == "text")

        assert "new" in user_text
        assert "old" not in user_text
        assert "first failed" in user_text
        assert "second failed" in user_text
        assert user_text.count(TOOL_RESULT_ERROR_SECTION_HEADER) == 2

    def test_no_action_error_no_section(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "goal text\n## AXTree:\nax"}]},
        ]
        out = self._proto().process_messages(msgs)
        user_text = next(c["text"] for c in out[0]["content"] if c.get("type") == "text")
        assert TOOL_RESULT_ERROR_SECTION_HEADER not in user_text

    def test_image_carry_forward_from_latest_user(self):
        # Turn-0 VWA goal-image case: image part on the LATEST user message
        # must be carried into the rebuilt single user message.
        img_part = {"type": "image", "image": "data:image/png;base64,XXX"}
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "find this product\n## AXTree:\nax"},
                    img_part,
                ],
            },
        ]
        out = self._proto().process_messages(msgs)
        contents = out[0]["content"]
        # Image part should be first (before the text); see protocol.py's
        # ``image_parts + [{"type": "text", ...}]`` ordering.
        assert any(c.get("type") == "image" for c in contents)
        assert contents[0]["type"] == "image"

    def test_image_only_from_latest_user_not_old(self):
        old_img = {"type": "image", "image": "old"}
        new_img = {"type": "image", "image": "new"}
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "goal"},
                    old_img,
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ""}],
                "raw_response": {"text": "click('a1')"},
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "## AXTree:\nax2"},
                    new_img,
                ],
            },
        ]
        out = self._proto().process_messages(msgs)
        imgs = [c for c in out[-1]["content"] if c.get("type") == "image"]
        # Only the latest user msg's image is kept.
        assert len(imgs) == 1
        assert imgs[0]["image"] == "new"

    def test_no_image_no_image_part(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "goal"}]},
        ]
        out = self._proto().process_messages(msgs)
        assert all(c.get("type") == "text" for c in out[0]["content"])

    def test_use_thinking_renders_examples_with_think(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "goal"}]},
        ]
        out = self._proto(
            use_thinking=True, use_concrete_example=True, use_abstract_example=True
        ).process_messages(msgs)
        user_text = next(c["text"] for c in out[0]["content"] if c.get("type") == "text")
        assert "<think>" in user_text
        assert "# Concrete Example" in user_text
        assert "# Abstract Example" in user_text

    def test_use_thinking_false_omits_think_blocks(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "goal"}]},
        ]
        out = self._proto(use_thinking=False, use_concrete_example=True).process_messages(msgs)
        user_text = next(c["text"] for c in out[0]["content"] if c.get("type") == "text")
        assert "<think>" not in user_text

    def test_action_examples_have_no_action_tag(self):
        # Regression: paper's `<action>code</action>` example was removed because
        # it taught the model the wrong format. Verify it's gone.
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "goal"}]},
        ]
        out = self._proto(use_concrete_example=True, use_abstract_example=True).process_messages(
            msgs
        )
        user_text = next(c["text"] for c in out[0]["content"] if c.get("type") == "text")
        assert "<action>" not in user_text
        assert "tool_call" in user_text  # still tells the model to use tool_call

    def test_use_hints(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "goal"}]},
        ]
        out = self._proto(use_hints=True).process_messages(msgs)
        user_text = next(c["text"] for c in out[0]["content"] if c.get("type") == "text")
        assert "Note:" in user_text
        assert "auto completion" in user_text

    def test_action_describe_text_rendered(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "goal"}]},
        ]
        out = self._proto(action_describe_text="click(bid)\nfill(bid, value)").process_messages(
            msgs
        )
        user_text = next(c["text"] for c in out[0]["content"] if c.get("type") == "text")
        assert "# Action space:" in user_text
        assert "click(bid)" in user_text

    def test_action_describe_none_omits_section(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "goal"}]},
        ]
        out = self._proto(action_describe_text=None).process_messages(msgs)
        user_text = next(c["text"] for c in out[0]["content"] if c.get("type") == "text")
        assert "# Action space:" not in user_text

    def test_section_ordering(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "goal\n## AXTree:\nax"}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ""}],
                "tool_calls": [make_tool_call("click", {"bid": "a1"}, call_id="call_0")],
            },
            {"role": "user", "content": [{"type": "text", "text": "## AXTree:\nax2"}]},
        ]
        out = self._proto(
            use_thinking=True,
            use_hints=True,
            use_concrete_example=True,
            use_abstract_example=True,
            action_describe_text="click(bid)",
        ).process_messages(msgs)
        user_text = next(c["text"] for c in out[-1]["content"] if c.get("type") == "text")
        # AgentLab ordering: Instructions, Observation, History, Action space,
        # Hints, Abstract, then Concrete.
        positions = {
            "instructions": user_text.find("# Instructions"),
            "observation": user_text.find("# Observation of current step:"),
            "history": user_text.find("# History of interaction with the task:"),
            "action_space": user_text.find("# Action space:"),
            # Use a hints-specific phrase (the AXTree section also says "Note:").
            "hints": user_text.find("auto completion"),
            "abstract": user_text.find("# Abstract Example"),
            "concrete": user_text.find("# Concrete Example"),
        }
        # All present.
        for name, pos in positions.items():
            assert pos >= 0, f"{name} section missing"
        # Strictly increasing.
        ordered = [
            "instructions",
            "observation",
            "history",
            "action_space",
            "hints",
            "abstract",
            "concrete",
        ]
        for a, b in zip(ordered, ordered[1:]):
            assert positions[a] < positions[b], f"{a} should come before {b}"

    def test_assistant_with_no_action_skipped(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "goal"}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ""}],
            },  # no raw_response, no tool_calls
            {"role": "user", "content": [{"type": "text", "text": "obs"}]},
        ]
        out = self._proto().process_messages(msgs)
        user_text = next(c["text"] for c in out[-1]["content"] if c.get("type") == "text")
        # Empty action → no history section emitted.
        assert "# History of interaction with the task:" not in user_text
