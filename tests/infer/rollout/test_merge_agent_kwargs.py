"""Tests for ``lite.infer.rollout._merge_agent_kwargs`` (footgun #1 fix).

A CLI ``--api-kwargs`` must override only the keys it names, not replace the
whole yaml ``api_kwargs`` dict.

Run: uv run pytest tests/infer/rollout/test_merge_agent_kwargs.py -v
"""
from __future__ import annotations

import pytest

from lite.agents.core.adapter import AGENT_KWARGS_TOOL_SURFACE_KEYS
from lite.infer.rollout import _merge_agent_kwargs


def test_cli_api_kwargs_deep_merge_preserves_config_keys():
    file_cfg = {
        "agent_id": "gpt",
        "api_kwargs": {
            "max_output_tokens": 4096,
            "reasoning_effort": "low",
            "parallel_tool_calls": True,
        },
    }
    cli = {"api_kwargs": {"reasoning_summary": "concise"}}
    merged = _merge_agent_kwargs(file_cfg, cli)
    # CLI key added, config keys preserved (NOT replaced wholesale).
    assert merged["api_kwargs"] == {
        "max_output_tokens": 4096,
        "reasoning_effort": "low",
        "parallel_tool_calls": True,
        "reasoning_summary": "concise",
    }
    assert merged["agent_id"] == "gpt"


def test_cli_api_kwargs_overrides_single_key():
    file_cfg = {"api_kwargs": {"reasoning_effort": "low", "max_output_tokens": 4096}}
    cli = {"api_kwargs": {"reasoning_effort": "high"}}
    merged = _merge_agent_kwargs(file_cfg, cli)
    assert merged["api_kwargs"] == {"reasoning_effort": "high", "max_output_tokens": 4096}


def test_top_level_keys_cli_wins():
    merged = _merge_agent_kwargs({"resolution": [1280, 768]}, {"resolution": [1024, 768]})
    assert merged["resolution"] == [1024, 768]


def test_no_cli_api_kwargs_keeps_config():
    file_cfg = {"api_kwargs": {"reasoning_effort": "low"}}
    merged = _merge_agent_kwargs(file_cfg, {})
    assert merged["api_kwargs"] == {"reasoning_effort": "low"}


def test_only_cli_api_kwargs_no_config():
    merged = _merge_agent_kwargs({}, {"api_kwargs": {"temperature": 0.5}})
    assert merged["api_kwargs"] == {"temperature": 0.5}


def test_deep_merge_extends_to_protocol_and_sampling_kwargs():
    """The deep-merge is generic — protocol_kwargs / sampling_kwargs get the
    same per-key protection as api_kwargs (not just api_kwargs)."""
    file_cfg = {
        "protocol_kwargs": {"full_history_size": 4, "use_thinking": False},
        "sampling_kwargs": {"temperature": 0.0, "top_p": 1.0},
    }
    cli = {
        "protocol_kwargs": {"full_history_size": 1},   # override only this leaf
        "sampling_kwargs": {"temperature": 0.6},        # override only this leaf
    }
    merged = _merge_agent_kwargs(file_cfg, cli)
    assert merged["protocol_kwargs"] == {"full_history_size": 1, "use_thinking": False}
    assert merged["sampling_kwargs"] == {"temperature": 0.6, "top_p": 1.0}


def test_nested_dict_recurses_more_than_one_level():
    merged = _merge_agent_kwargs(
        {"a": {"b": {"c": 1, "d": 2}}}, {"a": {"b": {"c": 9}}},
    )
    assert merged["a"]["b"] == {"c": 9, "d": 2}


@pytest.mark.parametrize("field", sorted(AGENT_KWARGS_TOOL_SURFACE_KEYS))
def test_yaml_agent_kwargs_reject_tool_surface_fields(field):
    with pytest.raises(ValueError, match="yaml agent_kwargs contains tool-surface settings"):
        _merge_agent_kwargs({field: "bad"}, {})


@pytest.mark.parametrize("field", sorted(AGENT_KWARGS_TOOL_SURFACE_KEYS))
def test_cli_agent_kwargs_reject_tool_surface_fields(field):
    with pytest.raises(ValueError, match="CLI agent_kwargs contains tool-surface settings"):
        _merge_agent_kwargs({}, {field: "bad"})
