"""Executable specs for old-input-only migration repair.

These tests pin old published-row shapes that are easy to corrupt silently:
desktop+mobile action-batch migration, imported terminal-final handling, id-owned
``role:"tool"`` results, metadata/image preservation without reindexing, noop
``screenshot`` / ``wait`` stripping that preserves reference image parts, refusal of
already nested rows, and parquet JSON-string rows.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        devs/migration/tests/test_upgrade_cases.py -p no:cacheprovider -q
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from pprint import pformat
from typing import Any

import pytest
from PIL import Image

from lite.core import LiteCUAMetadata
from lite.core.tools.calls import (
    make_tool_call,
    tool_call_arguments,
    tool_call_id,
    tool_call_name,
)
from lite.core.tools.schemas import make_tool_schema, tool_schema_name

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _fn(name: str, **arguments: Any) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


def _fn_raw(name: str, arguments: Any) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


def _user(
    text: str,
    *,
    image_index: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if image_index is not None:
        content.append({"type": "image", "index": image_index})
    content.append({"type": "text", "text": text})
    if metadata is not None:
        content.append({"type": "metadata", "data": metadata})
    return {"role": "user", "content": content}


def _assistant(*tool_calls: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": [], "tool_calls": list(tool_calls)}


def _old_desktop_use_sample() -> dict[str, Any]:
    return {
        "images": ["img0.png", "img1.png"],
        "metadata": {
            "platform": "desktop",
            "task_type": "use",
            "valid_actions": ["click", "type", "terminate", "response"],
            "others": {"fixture": "old-desktop"},
        },
        "messages": [
            _user("Type hi, then finish.", image_index=0, metadata={"task_id": "d0"}),
            _assistant(
                _fn("click", coordinate=[10, 20]),
                _fn("type", text="hi"),
            ),
            _user("typed", image_index=1, metadata={"url": "app://editor"}),
            _assistant(
                _fn("click", coordinate=[30, 40]),
                _fn("terminate", status="success"),
            ),
            _user("finished", image_index=1, metadata={"url": "app://editor"}),
        ],
    }


def _img(k: int) -> Image.Image:
    """Deterministic tiny image for adapter render equivalence checks."""
    return Image.new("RGB", (32, 32), color=(k * 40 % 256, 0, 0))


def _old_single_action_desktop_use_sample() -> dict[str, Any]:
    """Old contract fixture for the single-action render oracle."""
    return {
        "images": [_img(0), _img(1)],
        "metadata": {
            "platform": "desktop",
            "task_type": "use",
            "valid_actions": ["click", "type", "terminate", "response"],
            "others": {"fixture": "old-single-action"},
        },
        "messages": [
            _user("Click the menu.", image_index=0, metadata={"task_id": "single"}),
            _assistant(_fn("click", coordinate=[10, 20])),
            _user("menu opened", image_index=1, metadata={"url": "app://menu"}),
        ],
    }


def _old_mobile_use_sample() -> dict[str, Any]:
    return {
        "images": ["m0.png", "m1.png"],
        "metadata": {"platform": "mobile", "task_type": "use", "others": {"fixture": "old-mobile"}},
        "messages": [
            _user("Tap search.", image_index=0),
            _assistant(_fn("tap", coordinate=[100, 200])),
            _user("keyboard opened", image_index=1, metadata={"activity": "SearchActivity"}),
        ],
    }


def _old_mobile_multi_screen_turn() -> dict[str, Any]:
    row = _old_mobile_use_sample()
    row["messages"][1] = _assistant(
        _fn("tap", coordinate=[100, 200]),
        _fn("swipe", start_coordinate=[100, 800], coordinate=[100, 200]),
    )
    return row


def _old_desktop_standalone_predicate_sample() -> dict[str, Any]:
    row = _old_desktop_use_sample()
    row["messages"][1] = _assistant(
        _fn("click", coordinate=[10, 20]),
        _fn("response", text="done"),
        _fn("goto", url="https://example.com"),
        _fn_raw("back", None),
    )
    row["messages"][2] = _user("done", image_index=1, metadata={"url": "https://example.com"})
    row["messages"] = row["messages"][:3]
    return row


def _old_mobile_open_app_predicate_sample() -> dict[str, Any]:
    row = _old_mobile_use_sample()
    row["messages"][1] = _assistant(
        _fn("tap", coordinate=[100, 200]),
        _fn("open_app", app_name="Settings"),
    )
    return row


def _grounding_sample() -> dict[str, Any]:
    return {
        "images": ["g0.png"],
        "metadata": {
            "platform": "desktop",
            "task_type": "grounding.action",
            "extra_tool_schemas": [],
            "valid_actions": None,
            "others": {},
        },
        "messages": [
            _user("Click the button.", image_index=0),
            _assistant(_fn("click", coordinate=[500, 500])),
        ],
    }


def _old_structural_terminal_only_sample() -> dict[str, Any]:
    return {
        "images": ["s0.png", "s1.png"],
        "metadata": {
            "platform": "desktop",
            "task_type": "use",
            "valid_actions": ["click", "terminate"],
            "extra_tool_schemas": [
                {
                    "type": "function",
                    "function": {
                        "name": "terminate",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                }
            ],
            "others": {"fixture": "structural-terminal-only"},
        },
        "messages": [
            _user("Click the button, then stop.", image_index=0),
            _assistant(_fn("click", coordinate=[10, 20])),
            _user("button clicked", image_index=1),
            _assistant(_fn("terminate", status="success")),
        ],
    }


def _nested_desktop_use_sample() -> dict[str, Any]:
    return {
        "images": ["img0.png", "img1.png"],
        # Canonical rows carry ``valid_actions: None`` and an explicit
        # ``extra_tool_schemas`` list -- every ``lite/data/preproc`` script emits
        # exactly these two keys, so a migrated row must too.
        "metadata": {
            "platform": "desktop",
            "task_type": "use",
            "extra_tool_schemas": [],
            "valid_actions": None,
            "others": {},
        },
        "messages": [
            _user("Type hi.", image_index=0),
            _assistant(
                make_tool_call(
                    "computer",
                    {
                    "actions": [
                        {"action": "click", "coordinate": [10, 20]},
                        {"action": "type", "text": "hi"},
                    ]
                    },
                    call_id="call_0000",
                )
            ),
            {
                "role": "tool",
                "tool_call_id": "call_0000",
                "content": [{"type": "image", "index": 1}, {"type": "text", "text": "typed"}],
            },
        ],
    }


def _load_upgrade_module():
    path = _PROJECT_ROOT / "devs" / "migration" / "upgrade.py"
    assert path.exists(), "devs/migration/upgrade.py must own old-input repair"
    spec = importlib.util.spec_from_file_location("cua_lite_tool_io_upgrade", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_verify_module():
    path = _PROJECT_ROOT / "devs" / "migration" / "verify.py"
    spec = importlib.util.spec_from_file_location("cua_lite_tool_io_verify", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _upgrade_lite_sample(sample: dict[str, Any]) -> dict[str, Any]:
    module = _load_upgrade_module()
    upgrade = getattr(module, "upgrade_lite_sample", None)
    assert callable(upgrade), "expected devs.migration.upgrade_lite_sample(sample)"
    return upgrade(copy.deepcopy(sample))


def _upgrade_parquet_row(row: dict[str, Any]) -> dict[str, Any]:
    module = _load_upgrade_module()
    upgrade = getattr(module, "upgrade_parquet_row", None)
    assert callable(upgrade), "expected devs.migration.upgrade_parquet_row(row)"
    return upgrade(copy.deepcopy(row))


def _upgrade_lite_sample_expect_value_error(sample: dict[str, Any], match: str) -> None:
    module = _load_upgrade_module()
    upgrade = getattr(module, "upgrade_lite_sample", None)
    assert callable(upgrade), "expected devs.migration.upgrade_lite_sample(sample)"
    with pytest.raises(ValueError, match=match):
        upgrade(copy.deepcopy(sample))


def _verify_lite_sample(sample: dict[str, Any]) -> dict[str, Any]:
    verify = getattr(_load_verify_module(), "verify_lite_sample")
    return verify(copy.deepcopy(sample))


def _messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row["messages"]
    return json.loads(messages) if isinstance(messages, str) else messages


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row["metadata"]
    return json.loads(metadata) if isinstance(metadata, str) else metadata


def _tool_names(msg: dict[str, Any]) -> list[str]:
    return [_call_name(tc) for tc in msg.get("tool_calls", [])]


def _call_name(call: dict[str, Any]) -> str:
    return tool_call_name(call)


def _call_args(call: dict[str, Any]) -> dict[str, Any]:
    return tool_call_arguments(call)


def _call_id(call: dict[str, Any]) -> str:
    call_id = tool_call_id(call)
    assert isinstance(call_id, str) and call_id
    return call_id


def _assert_canonical_tool_call(call: dict[str, Any]) -> None:
    assert set(call) == {"id", "type", "function"}
    assert isinstance(tool_call_id(call), str) and tool_call_id(call)
    assert isinstance(tool_call_name(call), str) and tool_call_name(call)
    assert isinstance(tool_call_arguments(call), dict)


def _metadata_items(msg: dict[str, Any]) -> list[dict[str, Any]]:
    return [p for p in msg.get("content", []) if p.get("type") == "metadata"]


def _schema_names(metadata: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for schema in metadata.get("extra_tool_schemas") or []:
        names.add(tool_schema_name(schema))
    names.discard(None)
    return names


def _canonical_action_trace(messages: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Normalize migrated canonical batches to the action effect trace."""
    trace: list[tuple[str, dict[str, Any]]] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls", []):
            name = _call_name(call)
            args = copy.deepcopy(_call_args(call))
            if name in {"computer", "mobile"}:
                for action in args.get("actions", []):
                    action = copy.deepcopy(action)
                    action_name = action.pop("action")
                    trace.append((action_name, action))
            else:
                trace.append((name, args))
    return trace


def _legacy_source_action_trace(messages: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Normalize old migration-source envelopes to the same action effect trace."""
    trace: list[tuple[str, dict[str, Any]]] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls", []):
            # This helper is intentionally scoped to old migration-source rows.
            # Current canonical output is read by ``_canonical_action_trace`` via
            # tool-call accessors; do not make this a test-side dual-format
            # reader.
            function = call["function"]
            name = function["name"]
            args = copy.deepcopy(function.get("arguments") or {})
            trace.append((name, args))
    return trace


def _render_steps(
    adapter_key: str,
    sample: dict[str, Any],
    *,
    view: Any = None,
) -> str:
    """Render through a real adapter, but keep the oracle hermetic.

    ``view`` overrides the per-step normalization applied before pformat; it
    defaults to the Qwen3.5 chat-template surrogate for qwen3.5 keys.
    """
    from lite.agents.bootstrap import register_all

    register_all()
    from lite.agents.core.adapter import AgentAdapterRegistry
    from lite.core import LiteSample

    lite_sample = LiteSample.from_dict(sample)
    steps = AgentAdapterRegistry.get(adapter_key).unroll(lite_sample).steps
    if view is not None:
        steps = [view(step) for step in steps]
    elif adapter_key.startswith("qwen3_5@"):
        steps = [_qwen35_equivalence_view(step) for step in steps]
    return pformat(steps, sort_dicts=False, width=100)


def _qwen35_equivalence_view(step: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hermetic surrogate for Qwen3.5's chat-template tool-result wrapping.

    The new adapter intentionally leaves canonical ``role:"tool"`` messages
    unwrapped so the HF chat template can group/wrap them. This migration
    oracle compares the model-facing prompt shape without loading a tokenizer,
    so represent each tool result as the equivalent user ``<tool_response>``
    block before pformat comparison. Runtime code must not do this projection.
    """
    out: list[dict[str, Any]] = []
    for message in step:
        if message.get("role") != "tool":
            out.append(message)
            continue
        out.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "<tool_response>\n"},
                *(message.get("content") or []),
                {"type": "text", "text": "\n</tool_response>"},
            ],
        })
    return out


def _observation_content_view(step: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize the OBSERVATION CONTAINER away, keeping its payload.

    Legacy-source rows carried the post-action observation as a bare ``role:"user"``
    bubble; migrated rows carry it as ``role:"tool"``, which Qwen3.5's chat template
    then wraps in ``<tool_response>`` delimiters. That delimiter change is the
    deliberate point of the result-row representation: templates group
    consecutive ``role:"tool"`` into N ``<tool_response>`` blocks, and each
    result is owned by its assistant call id. Everything else about the prompt
    must be untouched by migration, so compare with the container normalized and
    assert the delimiter delta separately.
    """
    out: list[dict[str, Any]] = []
    for message in step:
        if message.get("role") != "tool":
            out.append(message)
            continue
        out.append({"role": "user", "content": message.get("content") or []})
    return out


def _legacy_single_action_render_oracle(sample: dict[str, Any]) -> dict[str, Any]:
    """Migration-test-only old-render oracle.

    Runtime adapters must not accept old source call envelopes anymore, so the
    old side of the single-action equivalence check uses the minimum
    old-adapter-equivalent shape that today's adapter can render: a canonical
    length-1 action-batch call with the legacy user observation. The migrated side
    still exercises ``tool_call_id``-owned ``role:"tool"`` result routing.
    """
    out = copy.deepcopy(sample)
    old_meta = out["metadata"]
    out["metadata"] = LiteCUAMetadata(
        dims=(old_meta["platform"], old_meta["task_type"]),
        extra_tool_schemas=old_meta.get("extra_tool_schemas", []),
        valid_actions=old_meta.get("valid_actions"),
        others=old_meta.get("others", {}),
    ).to_dict()
    next_call_id = 0
    for msg in out.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        lite_calls = []
        for call in msg.get("tool_calls", []):
            fn = call["function"]
            name = fn["name"]
            arguments = fn.get("arguments") or {}
            if name == "click":
                name = "computer"
                arguments = {"actions": [{"action": "click", **arguments}]}
            lite_calls.append(
                make_tool_call(
                    name,
                    arguments,
                    call_id=f"call_{next_call_id:04d}",
                )
            )
            next_call_id += 1
        msg["tool_calls"] = lite_calls
    return out


def test_pre_migration_desktop_use_batches_action_and_moves_observation_to_tool_result() -> None:
    """Old desktop/use rows group GUI actions into action-batch calls,
    keep terminal calls separate,
    and convert post-assistant user observations into id-owned tool results."""
    old = _old_desktop_use_sample()
    out = _upgrade_lite_sample(old)
    msgs = _messages(out)
    meta = _metadata(out)

    assert out["images"] == old["images"]
    lite_meta = LiteCUAMetadata.from_dict(meta)
    assert lite_meta.platform.value == old["metadata"]["platform"]
    assert lite_meta.task_type.value == old["metadata"]["task_type"]
    assert "platform" not in meta
    assert "task_type" not in meta
    assert meta["others"] == old["metadata"]["others"]

    assert msgs[0]["role"] == "user"
    assert _metadata_items(msgs[0]) == [{"type": "metadata", "data": {"task_id": "d0"}}]

    first_calls = msgs[1]["tool_calls"]
    assert _tool_names(msgs[1]) == ["computer"]
    computer = first_calls[0]
    _assert_canonical_tool_call(computer)
    computer_id = _call_id(computer)
    actions = _call_args(computer)["actions"]
    assert [a["action"] for a in actions] == ["click", "type"]
    assert actions[0]["coordinate"] == [10, 20]
    assert actions[1]["text"] == "hi"

    obs = msgs[2]
    assert obs["role"] == "tool"
    assert obs["tool_call_id"] == computer_id
    assert {"type": "image", "index": 1} in obs["content"]
    assert {"type": "text", "text": "typed"} in obs["content"]
    assert _metadata_items(obs) == [{"type": "metadata", "data": {"url": "app://editor"}}]

    terminal_calls = msgs[3]["tool_calls"]
    for call in terminal_calls:
        _assert_canonical_tool_call(call)
    assert [_call_name(tc) for tc in terminal_calls] == ["computer", "terminate"]
    assert _call_args(terminal_calls[0]) == {
        "actions": [{"action": "click", "coordinate": [30, 40]}]
    }
    terminal_obs = msgs[4]
    assert terminal_obs["role"] == "tool"
    assert terminal_obs["tool_call_id"] == _call_id(terminal_calls[0])


def test_pre_migration_strips_noop_actions_and_drops_empty_turn() -> None:
    """Noop-only old turns are representation noise; the row comes out COMPACTED.

    Dropping the noop turn drops the goal turn's own observation screenshot
    (``img0``), which no surviving message then references. The row must not
    publish that orphan, so ``compact_row_images`` removes it and renumbers the
    survivors -- and each surviving reference must still resolve to the SAME
    picture it did before.
    """
    old = _old_desktop_use_sample()
    old["images"] = ["img0.png", "img1.png", "img2.png"]
    old["messages"] = [
        _user("Do the task.", image_index=0),
        _assistant(_fn("screenshot"), _fn("wait")),
        _user("same screen", image_index=1),
        _assistant(_fn("click", coordinate=[10, 20]), _fn("screenshot")),
        _user("clicked", image_index=2),
    ]

    out = _upgrade_lite_sample(old)
    msgs = _messages(out)

    assert out["images"] == ["img1.png", "img2.png"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool"]
    assert [part for part in msgs[0]["content"] if part.get("type") == "image"] == [
        {"type": "image", "index": 0},
    ]
    assert {"type": "text", "text": "Do the task."} in msgs[0]["content"]
    actions = _call_args(msgs[1]["tool_calls"][0])["actions"]
    assert actions == [{"action": "click", "coordinate": [10, 20]}]
    assert msgs[2]["tool_call_id"] == _call_id(msgs[1]["tool_calls"][0])
    assert {"type": "image", "index": 1} in msgs[2]["content"]
    # By CONTENT, not by index: the renumbered references still address the very
    # pictures they addressed before compaction.
    assert out["images"][0] == old["images"][1]
    assert out["images"][1] == old["images"][2]
    _verify_lite_sample(out)


def test_pre_migration_folded_action_terminate_needs_no_trailing_observation() -> None:
    """An old row's folded terminate leaves the final action turn unpaired.

    Migration must not reject that at the raw boundary -- there is no screenshot
    in the input to pair it with. This legacy published-row migration path still
    re-applies its terminal policy and appends the structural ``Done.`` marker;
    raw-source preprocessors use the newer EOF-action policy instead.
    """
    old = _old_desktop_use_sample()
    old["messages"] = old["messages"][:4]

    out = _upgrade_lite_sample(old)
    msgs = _messages(out)

    assert [_call_name(tc) for tc in msgs[3]["tool_calls"]] == ["computer"]
    assert msgs[-1] == {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}
    _verify_lite_sample(out)


@pytest.mark.parametrize("adapter_key", ["qwen3_5@desktop@use"])
def test_pre_migration_single_action_render_equivalence_old_vs_migrated(adapter_key: str) -> None:
    """Single action is the strictest equivalence oracle:

    old contract -> old data -> old adapter render
    must equal
    old data -> migration -> nested contract -> current adapter render,
    **modulo the one delimiter change the redesign deliberately makes**.

    For a one-action GUI turn, id restamping and GUI wrapping
    are pure representation changes: the system prompt, tools block, task turn
    and assistant ``<tool_call>`` XML must render identically. The single
    intended delta is the result CONTAINER: old rows rendered a bare
    ``role:"user"`` bubble, migrated rows render ``role:"tool"``, which the Qwen3.5
    chat template wraps in ``<tool_response>``. That delta is required, not
    incidental — ``role:"tool"`` is kept precisely because
    templates group it into N ``<tool_response>`` blocks while ``role:"user"``
    is not grouped, which is what makes per-call owned results expressible.
    """
    old = _old_single_action_desktop_use_sample()
    migrated = _upgrade_lite_sample(old)
    legacy = _legacy_single_action_render_oracle(old)

    # 1. Everything except the observation container renders identically.
    assert _render_steps(adapter_key, migrated, view=_observation_content_view) == \
        _render_steps(adapter_key, legacy, view=_observation_content_view)

    # 2. The ONLY delta is the chat-template tool_response wrapper. Pin it so a
    #    future drift cannot hide behind the normalization above.
    wrapped = _render_steps(adapter_key, migrated)
    bare = _render_steps(adapter_key, legacy)
    assert wrapped != bare
    assert "<tool_response>" not in bare
    assert "</tool_response>" not in bare
    assert wrapped.count("<tool_response>") == 1
    assert wrapped.count("</tool_response>") == 1
    # No observation payload was added or lost besides the delimiter text parts.
    assert wrapped.count("'type': 'image'") == bare.count("'type': 'image'")
    assert wrapped.count("'text': 'menu opened'") == bare.count("'text': 'menu opened'")
    assert wrapped.count("'url': 'app://menu'") == bare.count("'url': 'app://menu'")


def test_pre_migration_qwen3_vl_render_keeps_canonical_role_tool_result() -> None:
    """Qwen3-VL receives canonical ``role:"tool"`` observations directly.

    Unlike the legacy old-data oracle, the migrated output must match the new
    canonical migrated data shape: nested call ids plus owned ``role:"tool"`` results,
    not a post-action role:user observation.
    """
    migrated = _upgrade_lite_sample(_old_single_action_desktop_use_sample())
    rendered = _render_steps("qwen3_vl@desktop@use", migrated)

    assert _canonical_action_trace(_messages(migrated)) == [("click", {"coordinate": [10, 20]})]
    assert "'role': 'tool'" in rendered
    assert "'tool_call_id': 'call_0000'" in rendered


@pytest.mark.parametrize(
    "sample_factory",
    [_old_desktop_use_sample, _old_mobile_multi_screen_turn],
)
def test_pre_migration_multi_action_trace_equivalence(sample_factory) -> None:
    """Multi-action migration may change representation, not behavior.

    Legacy-source bare per-action calls and migrated action-batch calls must
    unpack to the same ordered action trace, including terminal calls staying
    standalone rather than being swallowed into the action-batch call.
    """
    old = sample_factory()
    migrated = _upgrade_lite_sample(old)

    assert _canonical_action_trace(_messages(migrated)) == _legacy_source_action_trace(
        _messages(old)
    )


def test_pre_migration_batch_predicate_keeps_finish_nav_and_null_args_standalone() -> None:
    """Batch only GUI actions.

    ``response``/``goto``/``back`` stay standalone and a null-arguments nav call
    must not crash or get swallowed into ``computer.actions``.
    """
    out = _upgrade_lite_sample(_old_desktop_standalone_predicate_sample())
    calls = _messages(out)[1]["tool_calls"]

    for call in calls:
        _assert_canonical_tool_call(call)
    assert [_call_name(tc) for tc in calls] == ["computer", "response", "goto", "back"]
    assert _call_args(calls[0]) == {
        "actions": [{"action": "click", "coordinate": [10, 20]}]
    }
    assert _call_args(calls[-1]) == {}


def test_pre_migration_batch_predicate_keeps_open_app_standalone() -> None:
    """``open_app`` is a mobile/app extra tool, not a mobile GUI action."""
    out = _upgrade_lite_sample(_old_mobile_open_app_predicate_sample())
    calls = _messages(out)[1]["tool_calls"]

    for call in calls:
        _assert_canonical_tool_call(call)
    assert [_call_name(tc) for tc in calls] == ["mobile", "open_app"]
    assert _call_args(calls[0]) == {
        "actions": [{"action": "tap", "coordinate": [100, 200]}]
    }


def test_pre_migration_bash_ask_user_do_not_inherit_action_observation() -> None:
    """Text-result-only standalone calls must not inherit the GUI screenshot."""
    old = _old_desktop_use_sample()
    old["messages"][1] = _assistant(
        _fn("click", coordinate=[10, 20]),
        _fn("bash", command="pwd"),
        _fn("ask_user", text="Continue?"),
    )
    old["messages"][2] = _user("clicked", image_index=1, metadata={"window": "editor"})
    old["messages"] = old["messages"][:3]

    out = _upgrade_lite_sample(old)
    msgs = _messages(out)
    calls = msgs[1]["tool_calls"]

    for call in calls:
        _assert_canonical_tool_call(call)
    assert [_call_name(tc) for tc in calls] == ["computer", "bash", "ask_user"]
    assert _call_args(calls[0]) == {
        "actions": [{"action": "click", "coordinate": [10, 20]}]
    }
    assert msgs[2]["role"] == "tool"
    assert msgs[2]["tool_call_id"] == _call_id(calls[0])
    assert msgs[2]["tool_call_id"] not in {_call_id(calls[1]), _call_id(calls[2])}
    assert {"bash", "ask_user"} <= _schema_names(_metadata(out))
    from lite.core.tools.extra_tools import BASH_TOOL_NAME, LiteShellToolSet

    schemas = {
        tool_schema_name(schema): schema
        for schema in _metadata(out)["extra_tool_schemas"]
    }
    assert schemas["bash"] == LiteShellToolSet.get_tool_schema(BASH_TOOL_NAME)


def test_pre_migration_preserves_assistant_sidecars() -> None:
    """Migration rewrites tool_calls, not assistant reasoning/render sidecars."""
    old = _old_desktop_use_sample()
    old["messages"][1]["content"] = [{"type": "action_description", "text": "Click then type."}]
    old["messages"][1]["reasoning_content"] = "Need to focus the editor first."
    old["messages"][1]["raw_response"] = {
        "text": "raw teacher response",
        "adapter_key": "gpt@desktop@use",
    }

    out = _upgrade_lite_sample(old)
    asst = _messages(out)[1]

    assert asst["content"] == old["messages"][1]["content"]
    assert asst["reasoning_content"] == old["messages"][1]["reasoning_content"]
    assert asst["raw_response"] == old["messages"][1]["raw_response"]


def test_pre_migration_single_mobile_action_uses_length1_wrapper() -> None:
    """A one-action mobile turn uses a canonical ``mobile.actions`` wrapper."""
    out = _upgrade_lite_sample(_old_mobile_use_sample())
    msgs = _messages(out)

    calls = msgs[1]["tool_calls"]
    assert _tool_names(msgs[1]) == ["mobile"]
    _assert_canonical_tool_call(calls[0])
    assert _call_args(calls[0]) == {
        "actions": [{"action": "tap", "coordinate": [100, 200]}]
    }
    mobile_id = _call_id(calls[0])
    assert mobile_id, "mobile wrapper call needs id"

    obs = msgs[2]
    assert obs["role"] == "tool"
    assert obs["tool_call_id"] == mobile_id
    assert {"type": "image", "index": 1} in obs["content"]
    assert _metadata_items(obs) == [{"type": "metadata", "data": {"activity": "SearchActivity"}}]


def test_pre_migration_mobile_multi_action_turn_batches() -> None:
    """A mobile turn with two actions (tap + swipe) migrates to ONE batched
    ``mobile{actions:[tap, swipe]}`` — symmetric with desktop's computer batch,
    one screenshot for the turn. (No longer rejected: mobile batches like desktop.)"""
    out = _upgrade_lite_sample(_old_mobile_multi_screen_turn())
    calls = _messages(out)[1]["tool_calls"]
    assert _tool_names(_messages(out)[1]) == ["mobile"]
    _assert_canonical_tool_call(calls[0])
    actions = _call_args(calls[0])["actions"]
    assert [a["action"] for a in actions] == ["tap", "swipe"]


def test_pre_migration_grounding_nests_envelopes_without_batch_or_tool_results() -> None:
    """Grounding labels are not GUI rollout turns.

    Migration must not batch them or convert them into tool results; provider
    envelopes are still legacy input and must become nested Lite calls.
    """
    sample = _grounding_sample()
    out = _upgrade_lite_sample(sample)
    msgs = _messages(out)

    assert out["images"] == sample["images"]
    assert out["metadata"] == LiteCUAMetadata(
        dims=("desktop", "grounding.action"),
        extra_tool_schemas=[],
        valid_actions=None,
        others={},
    ).to_dict()
    assert msgs[0] == sample["messages"][0]
    assert msgs[1]["role"] == "assistant"
    assert _tool_names(msgs[1]) == ["click"]
    assert _call_args(msgs[1]["tool_calls"][0]) == {"coordinate": [500, 500]}
    assert _call_name(msgs[1]["tool_calls"][0]) != "computer"
    _assert_canonical_tool_call(msgs[1]["tool_calls"][0])


def test_pre_migration_terminal_only_success_defaults_to_text_final_without_schema() -> None:
    """A lone final ``terminate(success)`` is the old structural stop marker.

    The shared preproc terminal rule represents the stop as content-only
    ``Done.`` and must not invent a terminate tool surface in the persisted row.
    """
    out = _upgrade_lite_sample(_old_structural_terminal_only_sample())
    msgs = _messages(out)
    meta = _metadata(out)

    assert _tool_names(msgs[1]) == ["computer"]
    assert _call_args(msgs[1]["tool_calls"][0]) == {
        "actions": [{"action": "click", "coordinate": [10, 20]}]
    }
    assert msgs[2]["role"] == "tool"
    assert msgs[-1]["role"] == "assistant"
    assert msgs[-1]["content"] == [{"type": "text", "text": "Done."}]
    assert not msgs[-1].get("tool_calls")
    assert meta["valid_actions"] is None
    assert meta["extra_tool_schemas"] == []
    assert "terminate" not in _schema_names(meta)


def test_pre_migration_terminal_only_success_overwrites_legacy_content_with_done() -> None:
    old = _old_structural_terminal_only_sample()
    old["messages"][-1]["content"] = [{"type": "text", "text": "legacy final"}]

    out = _upgrade_lite_sample(old)
    final = _messages(out)[-1]

    assert final["role"] == "assistant"
    assert final["content"] == [{"type": "text", "text": "Done."}]
    assert not final.get("tool_calls")


def test_pre_migration_no_tool_call_final_action_description_becomes_done() -> None:
    old = _old_desktop_use_sample()
    old["messages"] = old["messages"][:-1]
    old["messages"].append({
        "role": "assistant",
        "content": [{"type": "action_description", "text": "Verified the setting."}],
    })

    out = _upgrade_lite_sample(old)
    final = _messages(out)[-1]

    assert final == {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}


def test_pre_migration_rejects_already_nested_lite_sample() -> None:
    """The forward migrator is a one-time legacy-source repair path."""
    _upgrade_lite_sample_expect_value_error(
        _nested_desktop_use_sample(),
        "legacy-source rows only",
    )


def test_pre_migration_rejects_message_image_index_without_matching_image() -> None:
    """Broken image references are rejected; migration never reindexes to repair them."""
    sample = _old_desktop_use_sample()
    sample["messages"][2]["content"][0]["index"] = len(sample["images"])

    _upgrade_lite_sample_expect_value_error(sample, "out of range")


@pytest.mark.parametrize(
    ("platform", "action", "arguments"),
    [
        ("desktop", "click", {"coordinate": [10, 20]}),
        ("mobile", "tap", {"coordinate": [100, 200]}),
        # ``web`` batches into ``computer`` like ``desktop``
        # (lite.core.tools.action_space.lite_action_set_tool_names_for_metadata),
        # so a bare top-level action is just as non-canonical there. Regression: the wrapper
        # lookup used to return None for ``web``, which disabled this check
        # entirely and let unbatched web GUI actions through the migrator.
        ("web", "click", {"coordinate": [10, 20]}),
        ("web", "type", {"text": "hi"}),
    ],
)
def test_pre_migration_rejects_nested_lite_input_before_tool_repair(
    platform: str,
    action: str,
    arguments: dict[str, Any],
) -> None:
    sample = _nested_desktop_use_sample()
    sample["metadata"]["platform"] = platform
    sample["metadata"]["valid_actions"] = [action]
    sample["messages"][1]["tool_calls"] = [
        make_tool_call(action, arguments, call_id="call_0000")
    ]
    sample["messages"][2]["tool_call_id"] = "call_0000"

    _upgrade_lite_sample_expect_value_error(sample, "legacy-source rows only")


@pytest.mark.parametrize(
    ("sample_factory", "platform", "action", "arguments", "wrapper"),
    [
        (
            _old_single_action_desktop_use_sample,
            "desktop",
            "click",
            {"coordinate": [10, 20]},
            "computer",
        ),
        (
            _old_single_action_desktop_use_sample,
            "web",
            "click",
            {"coordinate": [10, 20]},
            "computer",
        ),
        (_old_mobile_use_sample, "mobile", "tap", {"coordinate": [100, 200]}, "mobile"),
    ],
)
def test_pre_migration_legacy_source_bare_use_actions_still_upgrade_to_wrapper(
    sample_factory,
    platform: str,
    action: str,
    arguments: dict[str, Any],
    wrapper: str,
) -> None:
    sample = sample_factory()
    sample["metadata"]["platform"] = platform
    sample["metadata"]["valid_actions"] = [action]
    sample["messages"][1] = _assistant({
        "call_id": "legacy_source_0",
        "name": action,
        "arguments": arguments,
    })

    out = _upgrade_lite_sample(sample)
    msgs = _messages(out)
    call = msgs[1]["tool_calls"][0]

    assert _call_name(call) == wrapper
    assert _call_args(call) == {"actions": [{"action": action, **arguments}]}
    assert msgs[2]["role"] == "tool"
    assert msgs[2]["tool_call_id"] == _call_id(call)
    _verify_lite_sample(out)


@pytest.mark.parametrize(
    ("action_name", "arguments", "expected_arguments"),
    [
        ("key", {"keys": ["ctrl", "plus"]}, {"keys": ["ctrl", "+"]}),
        ("key", {"keys": ["ctrl", "+"]}, {"keys": ["ctrl", "+"]}),
        ("key_down", {"keys": "ctrl++"}, {"keys": ["ctrl", "+"]}),
        ("key_up", {"keys": "ctrl+-"}, {"keys": ["ctrl", "-"]}),
        (
            "hold_key",
            {"keys": ["Ctrl", "equal"], "duration": 0.25},
            {"keys": ["ctrl", "="], "duration": 0.25},
        ),
        (
            "hold_key",
            {"keys": "ctrl+=", "duration": 0.25},
            {"keys": ["ctrl", "="], "duration": 0.25},
        ),
    ],
)
def test_pre_migration_normalizes_legacy_key_action_tokens(
    action_name: str,
    arguments: dict[str, Any],
    expected_arguments: dict[str, Any],
) -> None:
    """Old rollout rows bypassed the action factory that normalizes key tokens."""
    for tool_call in [
        _fn(action_name, **arguments),
        {
            "call_id": f"legacy_{action_name}",
            "name": action_name,
            "arguments": arguments,
        },
    ]:
        sample = _old_single_action_desktop_use_sample()
        sample["metadata"]["valid_actions"] = [action_name]
        sample["messages"][1] = _assistant(tool_call)

        out = _upgrade_lite_sample(sample)
        call = _messages(out)[1]["tool_calls"][0]

        assert _call_name(call) == "computer"
        assert _call_args(call) == {
            "actions": [{"action": action_name, **expected_arguments}]
        }
        _verify_lite_sample(out)


def test_pre_migration_normalizes_grounding_action_key_tokens_without_batching() -> None:
    sample = _grounding_sample()
    sample["messages"][1] = _assistant(_fn("key", keys=["ctrl", "plus"]))

    out = _upgrade_lite_sample(sample)
    call = _messages(out)[1]["tool_calls"][0]

    assert _call_name(call) == "key"
    assert _call_args(call) == {"keys": ["ctrl", "+"]}
    _verify_lite_sample(out)


@pytest.mark.parametrize(
    "tool_call",
    [
        _fn("computer", actions=[{"action": "key", "keys": ["ctrl", "plus"]}]),
        {
            "call_id": "legacy_batch_0",
            "name": "computer",
            "arguments": {"actions": [{"action": "key", "keys": ["ctrl", "plus"]}]},
        },
    ],
)
def test_pre_migration_normalizes_legacy_action_batch_child_key_tokens(
    tool_call: dict[str, Any],
) -> None:
    sample = _old_single_action_desktop_use_sample()
    sample["metadata"]["valid_actions"] = ["key"]
    sample["messages"][1] = _assistant(tool_call)

    out = _upgrade_lite_sample(sample)
    call = _messages(out)[1]["tool_calls"][0]

    assert _call_name(call) == "computer"
    assert _call_args(call) == {
        "actions": [{"action": "key", "keys": ["ctrl", "+"]}]
    }
    _verify_lite_sample(out)


def test_pre_migration_noop_strip_preserves_reference_images_not_stale_screenshot() -> None:
    """Dropping noop turns keeps authored reference images, not stale screenshots.

    The authored reference image survives as a canonical message image part
    while the goal turn's own screenshot (``img0``) does not. Compaction then
    renumbers only message content image parts so the reference still points at
    ``img1.png``.
    """
    sample = {
        "images": ["img0.png", "img1.png", "img2.png", "img3.png"],
        "metadata": {
            "platform": "desktop",
            "task_type": "use",
            "valid_actions": ["click", "screenshot"],
            "others": {"fixture": "reference-image-noop"},
        },
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 1},
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "Find this item."},
                    {"type": "metadata", "data": {"source": "reference-image"}},
                ],
            },
            _assistant(_fn("screenshot")),
            _user("after noop", image_index=2),
            _assistant(_fn("click", coordinate=[10, 20])),
            _user("clicked", image_index=3),
        ],
    }

    out = _upgrade_lite_sample(sample)
    msgs = _messages(out)

    assert out["images"] == ["img1.png", "img2.png", "img3.png"]
    assert msgs[0]["content"] == [
        {"type": "image", "index": 0},
        {"type": "text", "text": "Find this item."},
        {"type": "metadata", "data": {"source": "reference-image"}},
        {"type": "image", "index": 1},
        {"type": "text", "text": "after noop"},
    ]
    assert [msg["role"] for msg in msgs] == ["user", "assistant", "tool"]
    assert _call_name(msgs[1]["tool_calls"][0]) == "computer"
    assert msgs[2]["content"] == [
        {"type": "image", "index": 2},
        {"type": "text", "text": "clicked"},
    ]
    # Same pictures, new numbers: the authored reference image is still img1.png
    # and the post-click observation is still img3.png.
    assert out["images"][msgs[0]["content"][0]["index"]] == "img1.png"
    assert out["images"][msgs[2]["content"][0]["index"]] == "img3.png"
    _verify_lite_sample(out)


def test_pre_migration_noop_strip_carries_mid_episode_failure_text() -> None:
    """A MID-EPISODE noop turn must not swallow the failed step's result text.

    Dropping a noop-only turn also drops the observation it answered, and that
    observation can be the only record that the producing action failed. Its
    authored text/metadata is carried onto the next observation -- which, once
    the noop turn is gone, is owned by the very call that failed -- while only
    the superseded screenshot is left behind. The rescue used to be gated on
    the goal turn, so a mid-episode failure round-tripped looking successful.
    """
    sample = {
        "images": ["img0.png", "img1.png", "img2.png", "img3.png"],
        "metadata": {
            "platform": "desktop",
            "task_type": "use",
            "valid_actions": ["click", "screenshot"],
            "others": {"fixture": "mid-episode-failure-noop"},
        },
        "messages": [
            _user("Open the settings dialog.", image_index=0),
            _assistant(_fn("click", coordinate=[10, 20])),
            _user(
                "Error: click at (10, 20) hit no element.",
                image_index=1,
                metadata={"url": "app://home"},
            ),
            _assistant(_fn("screenshot")),
            _user("same screen", image_index=2),
            _assistant(_fn("click", coordinate=[30, 40])),
            _user("settings open", image_index=3),
        ],
    }

    out = _upgrade_lite_sample(sample)
    msgs = _messages(out)

    assert [msg["role"] for msg in msgs] == ["user", "assistant", "tool", "assistant", "tool"]
    # The goal turn is untouched: the carry belongs to the mid-episode result.
    assert msgs[0]["content"] == [
        {"type": "image", "index": 0},
        {"type": "text", "text": "Open the settings dialog."},
    ]
    # The failed click owns the result that reports its failure.
    assert msgs[2]["tool_call_id"] == _call_id(msgs[1]["tool_calls"][0])
    assert _call_args(msgs[1]["tool_calls"][0]) == {
        "actions": [{"action": "click", "coordinate": [10, 20]}]
    }
    assert msgs[2]["content"] == [
        {"type": "text", "text": "Error: click at (10, 20) hit no element."},
        {"type": "metadata", "data": {"url": "app://home"}},
        {"type": "image", "index": 1},
        {"type": "text", "text": "same screen"},
    ]
    # The superseded screenshot (img1) is not promoted, so nothing references it
    # any more and the row is compacted: img1 is dropped, 2 -> 1 and 3 -> 2. The
    # orphan is MID-SEQUENCE, so every later reference shifts.
    assert out["images"] == ["img0.png", "img2.png", "img3.png"]
    assert out["images"][msgs[2]["content"][2]["index"]] == "img2.png"
    assert msgs[4]["tool_call_id"] == _call_id(msgs[3]["tool_calls"][0])
    assert msgs[4]["content"] == [
        {"type": "image", "index": 2},
        {"type": "text", "text": "settings open"},
    ]
    _verify_lite_sample(out)


@pytest.mark.parametrize(
    ("name", "arguments", "schema"),
    [
        (
            "response",
            {"text": "done"},
            make_tool_schema(
                "response",
                description="Submit an answer.",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            ),
        ),
        (
            "terminate",
            {"status": "failure"},
            make_tool_schema(
                "terminate",
                description="Finish the task.",
                parameters={
                    "type": "object",
                    "properties": {"status": {"type": "string"}},
                    "required": [],
                },
            ),
        ),
    ],
)
def test_pre_migration_rejects_nested_finish_without_tool_result(
    name: str,
    arguments: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    """Nested finish calls are already post-migration input, not repair input."""
    sample = {
        "images": ["finish0.png"],
        "metadata": {
            "platform": "desktop",
            "task_type": "use",
            "extra_tool_schemas": [schema],
            "valid_actions": None,
            "others": {"fixture": f"finish-{name}"},
        },
        "messages": [
            _user("Finish.", image_index=0),
            _assistant(make_tool_call(name, arguments, call_id=f"{name}_42")),
        ],
    }

    _upgrade_lite_sample_expect_value_error(sample, "legacy-source rows only")


def test_pre_migration_rejects_malformed_finish_only_legacy_source_tool_call() -> None:
    sample = {
        "images": ["finish0.png"],
        "metadata": {
            "platform": "desktop",
            "task_type": "use",
            "extra_tool_schemas": [
                make_tool_schema(
                    "terminate",
                    parameters={"type": "object", "properties": {}, "required": []},
                )
            ],
            "others": {"fixture": "malformed-terminate"},
        },
        "messages": [
            _user("Finish.", image_index=0),
            _assistant({
                "call_id": "terminate_42",
                "name": "terminate",
                "arguments": {"status": "failure"},
                "type": "function",
            }),
        ],
    }

    _upgrade_lite_sample_expect_value_error(sample, "noncanonical keys")


def test_pre_migration_rejects_nested_rows_instead_of_canonicalizing_metadata() -> None:
    """Metadata repair is for legacy-source rows, not already nested rows."""
    sample = _nested_desktop_use_sample()
    sample["metadata"]["extra_tool_schemas"] = [
        {
            "type": "function",
            "function": {
                "name": "response",
                "description": "Submit an answer.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        }
    ]

    _upgrade_lite_sample_expect_value_error(sample, "legacy-source rows only")


@pytest.mark.parametrize(
    "case",
    ["missing_id", "duplicate_id", "extra_result_id", "duplicate_tool_result_id"],
)
def test_pre_migration_rejects_malformed_nested_call_ids(case: str) -> None:
    """Nested rows are data-corruption inputs here, not migration opportunities."""
    sample = _nested_desktop_use_sample()
    msgs = sample["messages"]

    if case == "missing_id":
        del msgs[1]["tool_calls"][0]["id"]
    elif case == "duplicate_id":
        msgs.extend([
            _assistant(make_tool_call(
                "computer",
                {"actions": [{"action": "click", "coordinate": [30, 40]}]},
                call_id="call_0000",
            )),
            {
                "role": "tool",
                "tool_call_id": "call_0000",
                "content": [{"type": "text", "text": "again"}],
            },
        ])
    elif case == "extra_result_id":
        msgs[2]["tool_call_id"] = "call_9999"
    elif case == "duplicate_tool_result_id":
        msgs.append(copy.deepcopy(msgs[2]))
    else:  # pragma: no cover - parametrization guard
        raise AssertionError(case)

    _upgrade_lite_sample_expect_value_error(sample, "legacy-source rows only")


def test_pre_migration_rejects_orphan_tool_result_at_input_boundary() -> None:
    """``role:"tool"`` is already the nested result shape and is refused as input."""
    sample = _nested_desktop_use_sample()
    sample["messages"].append({
        "role": "tool",
        "tool_call_id": "call_9999",
        "content": [{"type": "text", "text": "orphan"}],
    })

    _upgrade_lite_sample_expect_value_error(sample, "legacy-source rows only")


def test_pre_migration_rejects_malformed_nested_tool_call() -> None:
    """Nested Lite rows do not get a repair attempt."""
    sample = _nested_desktop_use_sample()
    sample["messages"][1]["tool_calls"][0]["name"] = "computer"

    _upgrade_lite_sample_expect_value_error(sample, "legacy-source rows only")


@pytest.mark.parametrize(
    "schema",
    [
        {
            "type": "function_call",
            "name": "response",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "type": "function",
            "name": "response",
            "parameters": {"type": "object", "properties": {}, "required": []},
            "id": "provider_native",
        },
    ],
)
def test_pre_migration_rejects_noncanonical_schemas(schema: dict[str, Any]) -> None:
    sample = _old_single_action_desktop_use_sample()
    sample["metadata"]["extra_tool_schemas"] = [schema]

    _upgrade_lite_sample_expect_value_error(sample, "extra_tool_schemas")


def test_pre_migration_rejects_duplicate_extra_tool_schema_names() -> None:
    sample = _old_single_action_desktop_use_sample()
    sample["metadata"]["extra_tool_schemas"] = [
        make_tool_schema(
            "response",
            parameters={"type": "object", "properties": {}, "required": []},
        ),
        make_tool_schema(
            "response",
            parameters={"type": "object", "properties": {}, "required": []},
        ),
    ]

    _upgrade_lite_sample_expect_value_error(sample, "duplicate extra_tool_schemas")


def test_pre_migration_rejects_standalone_extra_call_without_schema() -> None:
    sample = _old_single_action_desktop_use_sample()
    sample["images"] = ["img0.png"]
    sample["messages"] = [
        _user("Report whether this is feasible.", image_index=0),
        _assistant({
            "call_id": "legacy_source_0",
            "name": "report_infeasible",
            "arguments": {},
        }),
    ]
    sample["metadata"]["extra_tool_schemas"] = []

    _upgrade_lite_sample_expect_value_error(
        sample,
        "missing from metadata.extra_tool_schemas",
    )


def test_pre_migration_allows_standalone_extra_call_with_schema() -> None:
    sample = _old_single_action_desktop_use_sample()
    # The replacement messages show only the first image; drop the second so the
    # row is dense and this stays a pure legacy-source schema assertion.
    # Compaction of a row that IS sparse is covered by the noop-strip tests above.
    sample["images"] = ["img0.png"]
    sample["messages"] = [
        _user("Report whether this is feasible.", image_index=0),
        _assistant({
            "call_id": "legacy_source_0",
            "name": "report_infeasible",
            "arguments": {"reason": "not visible"},
        }),
    ]
    sample["metadata"]["extra_tool_schemas"] = [
        make_tool_schema(
            "report_infeasible",
            description="Report that the task is infeasible.",
            parameters={
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        )
    ]
    sample["metadata"]["valid_actions"] = None

    out = _upgrade_lite_sample(sample)
    assert _call_name(_messages(out)[1]["tool_calls"][0]) == "report_infeasible"
    assert _schema_names(_metadata(out)) == {"report_infeasible"}


def test_pre_migration_normalizes_legacy_null_schema_properties() -> None:
    sample = _old_single_action_desktop_use_sample()
    sample["images"] = ["img0.png"]
    sample["messages"] = [
        _user("Report whether this is feasible.", image_index=0),
        _assistant({
            "call_id": "legacy_source_0",
            "name": "report_infeasible",
            "arguments": {},
        }),
    ]
    sample["metadata"]["extra_tool_schemas"] = [
        {
            "name": "report_infeasible",
            "description": "Report that the task is infeasible.",
            "parameters": {
                "type": "object",
                "properties": None,
                "required": [],
            },
        }
    ]
    sample["metadata"]["valid_actions"] = None

    out = _upgrade_lite_sample(sample)
    schema = _metadata(out)["extra_tool_schemas"][0]
    assert schema == make_tool_schema(
        "report_infeasible",
        description="Report that the task is infeasible.",
        parameters={"type": "object", "properties": {}, "required": []},
    )


def test_pre_migration_rejects_standalone_tool_nested_in_action_batch() -> None:
    sample = _old_single_action_desktop_use_sample()
    sample["messages"][1] = _assistant(_fn(
        "computer",
        actions=[
            {"action": "click", "coordinate": [10, 20]},
            {"action": "terminate", "status": "success"},
        ],
    ))
    sample["metadata"]["extra_tool_schemas"] = [
        make_tool_schema(
            "terminate",
            parameters={"type": "object", "properties": {}, "required": []},
        )
    ]

    _upgrade_lite_sample_expect_value_error(sample, "not valid for computer")


@pytest.mark.parametrize(
    ("platform", "wrapper_name", "wrong_action"),
    [
        ("desktop", "computer", "tap"),
        ("mobile", "mobile", "click"),
    ],
)
def test_pre_migration_rejects_canonical_wrapper_specific_wrong_child_names(
    platform: str,
    wrapper_name: str,
    wrong_action: str,
) -> None:
    sample = _old_single_action_desktop_use_sample()
    sample["metadata"]["platform"] = platform
    sample["metadata"]["valid_actions"] = ["tap"] if platform == "mobile" else ["click"]
    sample["messages"][1] = _assistant(_fn(
        wrapper_name,
        actions=[{"action": wrong_action, "coordinate": [10, 20]}],
    ))

    _upgrade_lite_sample_expect_value_error(sample, "not valid for")


def test_pre_migration_parquet_row_json_strings_are_preserved() -> None:
    """Parquet rows store messages/metadata as JSON strings. The row-level API
    must parse, upgrade, and re-emit JSON strings so pyarrow never sees a nested
    struct schema."""
    row = _old_desktop_use_sample()
    row["messages"] = json.dumps(row["messages"])
    row["metadata"] = json.dumps(row["metadata"])

    out = _upgrade_parquet_row(row)
    assert isinstance(out["messages"], str)
    assert isinstance(out["metadata"], str)
    msgs = _messages(out)
    _assert_canonical_tool_call(msgs[1]["tool_calls"][0])
    assert _call_name(msgs[1]["tool_calls"][0]) == "computer"
    assert msgs[2]["role"] == "tool"


def test_pre_migration_hf_lite_osworld_metadata_durable_keys_stay_in_others() -> None:
    """Published rollout facts follow the LiteCUAMetadata.others owner."""
    row = _old_desktop_use_sample()
    row["metadata"] = {
        "platform": "desktop",
        "task_type": "use",
        "extra_tool_schemas": [],
        "valid_actions": None,
        "others": {
            "domain": "chrome",
            "source": "perturb:osworld_chrome_030eeff7",
            "env_id": "lite.osworld",
            "task_id": "perturb_osworld_chrome_030eeff7_00aca7a8",
            "episode_return": 1.0,
            "terminated": True,
            "truncated": False,
        },
    }
    row["messages"] = json.dumps(row["messages"])
    row["metadata"] = json.dumps(row["metadata"])

    out = _upgrade_parquet_row(row)
    assert isinstance(out["metadata"], str)
    meta = _metadata(out)

    durable = {"env_id", "task_id", "episode_return", "terminated", "truncated"}
    assert not durable & set(meta)
    assert durable <= set(meta["others"])
    assert meta["others"] == {
        "domain": "chrome",
        "source": "perturb:osworld_chrome_030eeff7",
        "env_id": "lite.osworld",
        "task_id": "perturb_osworld_chrome_030eeff7_00aca7a8",
        "episode_return": 1.0,
        "terminated": True,
        "truncated": False,
    }


def test_pre_migration_top_level_durable_key_does_not_overwrite_others_owner() -> None:
    row = _old_desktop_use_sample()
    row["metadata"]["env_id"] = "top-level-env"
    row["metadata"]["others"]["env_id"] = "legacy-other-env"

    meta = _metadata(_upgrade_lite_sample(row))

    assert "env_id" not in meta
    assert meta["others"]["env_id"] == "legacy-other-env"


def test_pre_migration_rejects_direct_nested_json_strings() -> None:
    """Direct sample API parses JSON strings before enforcing the legacy-source gate."""
    row = _nested_desktop_use_sample()
    row["messages"] = json.dumps(row["messages"])
    row["metadata"] = json.dumps(row["metadata"])

    _upgrade_lite_sample_expect_value_error(row, "legacy-source rows only")


def test_pre_migration_drops_valid_actions_and_keeps_finish_as_schemas() -> None:
    """``valid_actions`` is not part of the canonical contract any more.

    Every ``lite/data/preproc`` script hardcodes ``valid_actions: None`` -- the
    action surface is carried by ``extra_tool_schemas``, so an old row's name
    filter must be resolved into schemas and then dropped, not narrowed.
    """
    out = _upgrade_lite_sample(_old_desktop_use_sample())
    meta = _metadata(out)
    assert meta["valid_actions"] is None
    assert {"response", "terminate"} <= _schema_names(meta)


def test_pre_migration_strips_text_only_standalone_tools_from_valid_actions_and_verifies() -> None:
    old = _old_desktop_use_sample()
    old["metadata"]["valid_actions"] = [
        "click",
        "bash",
        "ask_user",
        "type",
        "terminate",
        "response",
        "goto",
    ]

    out = _upgrade_lite_sample(old)
    meta = _metadata(out)

    assert meta["valid_actions"] is None
    assert {"bash", "ask_user", "terminate", "response", "goto"} <= _schema_names(meta)
    _verify_lite_sample(out)


def test_pre_migration_renames_legacy_extra_tools_metadata_key() -> None:
    """Published Lite.ScaleCUA migration rows spell the field ``extra_tools``.

    ``LiteCUAMetadata.from_dict`` only reads ``extra_tool_schemas``, so leaving the
    old key would silently drop the schemas AND leave an unknown key behind.
    """
    old = _old_desktop_use_sample()
    del old["metadata"]["valid_actions"]
    old["metadata"]["extra_tools"] = [{
        "type": "function",
        "function": {
            "name": "response",
            "description": "Submit the final answer.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }]

    meta = _metadata(_upgrade_lite_sample(old))

    assert "extra_tools" not in meta
    schemas = {tool_schema_name(schema): schema for schema in meta["extra_tool_schemas"]}
    assert schemas["response"]["function"]["description"] == "Submit the final answer."


def test_pre_migration_rejects_legacy_navigation_task_type_by_default() -> None:
    old = _old_desktop_use_sample()
    old["metadata"]["task_type"] = "navigation"

    _upgrade_lite_sample_expect_value_error(old, "task_type 'navigation'")


def test_pre_migration_strips_parquet_null_padding_from_messages() -> None:
    """Parquet unifies ``messages`` into one Arrow struct, padding with nulls.

    A published Lite.OSWorld terminate reads back as
    ``{"coordinate":null,"keys":null,"text":null,"status":"success"}``.
    ``lite/data/staging.coerce_messages`` strips those on the write path, so a
    canonical row never has them -- and until they are stripped the terminal
    predicate cannot see the lone ``status`` either.
    """
    old = _old_structural_terminal_only_sample()
    old["messages"][1]["tool_calls"] = [
        _fn("click", coordinate=[10, 20], text=None, keys=None, status=None),
    ]
    old["messages"][-1]["tool_calls"] = [
        _fn("terminate", coordinate=None, keys=None, text=None, status="success"),
    ]
    old["messages"][2]["content"] = [{"type": "image", "index": 1, "text": None}]

    out = _upgrade_lite_sample(old)
    msgs = _messages(out)

    assert _call_args(msgs[1]["tool_calls"][0]) == {
        "actions": [{"action": "click", "coordinate": [10, 20]}]
    }
    assert msgs[2]["content"] == [{"type": "image", "index": 1}]
    # The lone terminate is recognized as structural despite the padding.
    assert msgs[-1] == {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}
    assert _metadata(out)["extra_tool_schemas"] == []


def test_pre_migration_lite_osworld_integral_float_image_refs_are_lossless() -> None:
    old = _old_structural_terminal_only_sample()
    old["metadata"]["others"]["source"] = "cua-lite/Lite.OSWorld"
    old["messages"][0]["content"][0]["index"] = 0.0
    old["messages"][2]["content"][0]["index"] = 1.0
    old["messages"][1]["tool_calls"] = [
        _fn("click", coordinate=[10, 20], text=None, keys=None),
    ]
    old["messages"][-1]["tool_calls"] = [
        _fn("terminate", coordinate=None, keys=None, text=None, status="success"),
    ]

    out = _upgrade_lite_sample(old)
    msgs = _messages(out)

    assert msgs[0]["content"][0]["index"] == 0
    assert type(msgs[0]["content"][0]["index"]) is int
    assert msgs[2]["content"][0]["index"] == 1
    assert type(msgs[2]["content"][0]["index"]) is int
    assert msgs[-1] == {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}


@pytest.mark.parametrize("bad_index", [0.5, None])
def test_pre_migration_keeps_invalid_image_refs_invalid(bad_index: Any) -> None:
    old = _old_structural_terminal_only_sample()
    old["messages"][0]["content"][0]["index"] = bad_index

    with pytest.raises(ValueError, match="non-negative integer"):
        _upgrade_lite_sample(old)
