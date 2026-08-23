"""Old-input repair oracle for devs/migration.

This intentionally stays test-local and hermetic: the old side is decoded by
this file as published data, not by runtime compatibility helpers. The migrated
side must be the current canonical migration output: nested calls, owned
``role:"tool"`` results, and preserved image indices. It deliberately does not
assert byte identity with fresh raw-source preprocessing, because some sources
now have richer or different source-policy output than their already-published
legacy rows can recover.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        devs/migration/tests/test_old_new_equivalence_oracle.py \
        devs/migration/tests/test_upgrade_cases.py -p no:cacheprovider -q
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from lite.core import LiteCUAMetadata
from lite.core.tools.action_space import LITE_DESKTOP_KEY_ACTION_NAMES
from lite.core.tools.action_space.keys import normalize_keys
from lite.core.tools.calls import (
    tool_call_arguments,
    tool_call_id,
    tool_call_name,
    validate_lite_tool_call,
)
from lite.core.tools.schemas import (
    tool_schema_name,
    tool_schema_parameters,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_UPGRADE_PATH = _PROJECT_ROOT / "devs" / "migration" / "upgrade.py"
_DONE_CONTENT = [{"type": "text", "text": "Done."}]

#: Deliberately a hermetic local copy, not an import from ``lite.agents.core.action_space.base``:
#: an oracle that shares its vocabulary with the code under test cannot catch a
#: vocabulary bug. It covers every Lite desktop action old rows could hold.
_DESKTOP_ACTIONS = {
    "click",
    "drag",
    "mouse_move",
    "mouse_down",
    "mouse_up",
    "type",
    "key",
    "key_down",
    "key_up",
    "hold_key",
    "scroll",
    "wait",
    "screenshot",
    "cursor_position",
}
_SCREEN_PRODUCING_STANDALONE = {"goto", "back", "forward", "refresh"}


def _load_upgrade_module():
    assert _UPGRADE_PATH.exists(), "devs/migration/upgrade.py must own old-input repair"
    spec = importlib.util.spec_from_file_location(
        "cua_lite_old_new_equivalence_upgrade",
        _UPGRADE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _upgrade_lite_sample(sample: dict[str, Any]) -> dict[str, Any]:
    upgrade = getattr(_load_upgrade_module(), "upgrade_lite_sample", None)
    assert callable(upgrade), "expected devs.migration.upgrade_lite_sample(sample)"
    return upgrade(copy.deepcopy(sample))


def _fn(name: str, **arguments: Any) -> dict[str, Any]:
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


def _old_representative_desktop_use_row() -> dict[str, Any]:
    """One row covering the migration cases most likely to regress together."""
    return {
        "images": ["screen0.png", "screen1.png", "screen2.png"],
        "metadata": {
            "platform": "desktop",
            "task_type": "use",
            "valid_actions": ["click", "type", "key", "terminate", "response", "goto"],
            "extra_tool_schemas": [
                {
                    "type": "function",
                    "function": {
                        "name": "response",
                        "description": "Submit the final answer.",
                        "parameters": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    },
                    "strict": True,
                },
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    },
                },
            ],
            "others": {"fixture": "old-new-equivalence-oracle"},
        },
        "messages": [
            _user(
                "Open the launcher and start the report.",
                image_index=0,
                metadata={"task_id": "eq-0"},
            ),
            _assistant(_fn("click", coordinate=[12, 34])),
            _user("launcher opened", image_index=1, metadata={"window": "launcher"}),
            _assistant(
                _fn("click", coordinate=[88, 100]),
                _fn("type", text="report"),
                _fn("key", keys=["ENTER"]),
            ),
            _user("report opened", image_index=2, metadata={"window": "report"}),
            _assistant(_fn("terminate", status="success")),
        ],
    }


def _old_semantic_trace(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    pending_result_turn: int | None = None

    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            effects, screen_producing = _old_assistant_effects(msg)
            trace.append({"effects": effects})
            pending_result_turn = len(trace) - 1 if screen_producing else None
            continue
        if role == "user" and pending_result_turn is not None:
            trace[pending_result_turn]["observation"] = copy.deepcopy(msg["content"])
            pending_result_turn = None

    return trace


def _new_semantic_trace(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    call_id_to_turn: dict[str, int] = {}

    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            effects, screen_call_ids = _new_assistant_effects(msg)
            trace.append({"effects": effects})
            for call_id in screen_call_ids:
                call_id_to_turn[call_id] = len(trace) - 1
            continue
        if role == "tool":
            turn_index = call_id_to_turn[msg["tool_call_id"]]
            trace[turn_index]["observation"] = copy.deepcopy(msg["content"])

    return trace


def _old_assistant_effects(msg: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    calls = msg.get("tool_calls") or []
    if _is_old_terminal_success(calls):
        return [{"kind": "final", "content": _DONE_CONTENT}], False

    effects: list[dict[str, Any]] = []
    screen_producing = False
    for call in calls:
        fn = call["function"]
        name = fn["name"]
        args = copy.deepcopy(fn.get("arguments") or {})
        if name in _DESKTOP_ACTIONS:
            if name in LITE_DESKTOP_KEY_ACTION_NAMES and args.get("keys") is not None:
                args["keys"] = normalize_keys(args["keys"])
            effects.append({"kind": "action", "action": name, "arguments": args})
            screen_producing = True
        else:
            effects.append({"kind": "tool", "name": name, "arguments": args})
            screen_producing = screen_producing or name in _SCREEN_PRODUCING_STANDALONE
    return effects, screen_producing


def _new_assistant_effects(msg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    calls = msg.get("tool_calls") or []
    if not calls:
        return ([{"kind": "final", "content": copy.deepcopy(msg.get("content") or [])}], [])

    effects: list[dict[str, Any]] = []
    screen_call_ids: list[str] = []
    for call in calls:
        name = tool_call_name(call)
        args = copy.deepcopy(tool_call_arguments(call))
        if name == "computer":
            screen_call_ids.append(tool_call_id(call))
            for action in args["actions"]:
                action = copy.deepcopy(action)
                effects.append({
                    "kind": "action",
                    "action": action.pop("action"),
                    "arguments": action,
                })
            continue
        if name in _DESKTOP_ACTIONS:
            raise AssertionError(
                f"migrated USE GUI action {name!r} must be nested in computer.actions"
            )
        effects.append({"kind": "tool", "name": name, "arguments": args})
        if name in _SCREEN_PRODUCING_STANDALONE:
            screen_call_ids.append(tool_call_id(call))
    return effects, screen_call_ids


def _is_old_terminal_success(calls: list[dict[str, Any]]) -> bool:
    if len(calls) != 1:
        return False
    fn = calls[0]["function"]
    if fn.get("name") != "terminate":
        return False
    args = fn.get("arguments") or {}
    return set(args) <= {"status"} and args.get("status", "success") in {"success", "completed"}


def _assert_nested_contract(messages: list[dict[str, Any]]) -> None:
    seen_call_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for i, call in enumerate(msg.get("tool_calls") or []):
            assert validate_lite_tool_call(call, f"tool_calls[{i}]", require_id=True) is None
            call_id = tool_call_id(call)
            assert isinstance(call_id, str) and call_id
            assert call_id not in seen_call_ids
            seen_call_ids.add(call_id)


def test_old_desktop_use_row_upgrades_to_equivalent_nested_contract() -> None:
    """Representative old-format row preserves behavior under the new contract.

    Coverage in this one oracle:
    - single GUI action
    - multi-action desktop batch
    - synthetic final terminate(success) -> content-only Done.
    - old user observation -> role:tool result owned by the producing call id
    - old extra_tool_schemas canonicalized to nested schemas
    - GUI-only valid_actions
    """
    old = _old_representative_desktop_use_row()

    migrated = _upgrade_lite_sample(old)
    messages = migrated["messages"]
    metadata = migrated["metadata"]

    assert migrated["images"] == old["images"]
    assert _new_semantic_trace(messages) == _old_semantic_trace(old["messages"])
    _assert_nested_contract(messages)

    first_call = messages[1]["tool_calls"][0]
    assert tool_call_name(first_call) == "computer"
    assert tool_call_arguments(first_call) == {
        "actions": [{"action": "click", "coordinate": [12, 34]}]
    }
    assert messages[2]["role"] == "tool"
    assert messages[2]["tool_call_id"] == tool_call_id(first_call)
    assert messages[2]["content"] == old["messages"][2]["content"]

    second_call = messages[3]["tool_calls"][0]
    assert [tool_call_name(call) for call in messages[3]["tool_calls"]] == ["computer"]
    assert tool_call_arguments(second_call)["actions"] == [
        {"action": "click", "coordinate": [88, 100]},
        {"action": "type", "text": "report"},
        {"action": "key", "keys": ["enter"]},
    ]
    assert messages[4]["role"] == "tool"
    assert messages[4]["tool_call_id"] == tool_call_id(second_call)
    assert messages[4]["content"] == old["messages"][4]["content"]

    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == _DONE_CONTENT
    assert not messages[-1].get("tool_calls")

    lite_meta = LiteCUAMetadata.from_dict(metadata)
    assert lite_meta.platform.value == "desktop"
    assert lite_meta.task_type.value == "use"
    assert "platform" not in metadata
    assert "task_type" not in metadata
    assert metadata["others"] == old["metadata"]["others"]
    # ``valid_actions`` is dropped, not narrowed: every ``lite/data/preproc``
    # script hardcodes ``None`` and expresses the tool surface as schemas.
    assert metadata["valid_actions"] is None

    schemas = metadata["extra_tool_schemas"]
    assert all("function" in schema for schema in schemas)
    schemas_by_name = {tool_schema_name(schema): schema for schema in schemas}
    assert set(schemas_by_name) == {"response", "bash", "goto"}
    assert "strict" not in schemas_by_name["response"]
    assert schemas_by_name["response"]["function"]["description"] == "Submit the final answer."
    assert tool_schema_parameters(schemas_by_name["response"])["required"] == ["text"]
    assert tool_schema_parameters(schemas_by_name["bash"])["required"] == ["command"]
    assert "terminate" not in schemas_by_name


# ===========================================================================
# Canonical migration output oracle
# ===========================================================================
#
# The tests above compare a lossy semantic *projection* (call ids deliberately
# discarded). This section pins exact migrated rows for the terminal-policy
# cases that used to be compared against fresh preproc output. The source record
# is still the legacy published shape: provider ``{"type":"function",...}``
# envelopes, bare top-level GUI actions, ``role:"user"`` observations, an
# always-live trailing ``terminate``, image references in the old row's own
# index space, and no ``extra_tool_schemas`` / ``valid_actions`` keys.
#
# Coverage is one fixture per migration-owned terminal shape:
#   folded structural terminate -> stripped, ``Done.`` appended
#   folded semantic terminate   -> stripped, outcome moved to metadata
#   terminal failure reason     -> outcome fields appended to metadata.others

_VERIFY_PATH = _PROJECT_ROOT / "devs" / "migration" / "verify.py"


def _load_verify_module():
    spec = importlib.util.spec_from_file_location(
        "cua_lite_old_new_equivalence_verify",
        _VERIFY_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _old_fn_call(name: str, **arguments: Any) -> dict[str, Any]:
    """Old provider-envelope call, as ``LiteDesktopActionSpace`` used to emit."""
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


def _old_row(
    *,
    images: list[str],
    messages: list[dict[str, Any]],
    platform: str,
    others: dict[str, Any],
) -> dict[str, Any]:
    """Published pre-migration row shape.

    ``extra_tool_schemas`` / ``valid_actions`` are absent on purpose: several
    old ``use.py`` metadata literals never wrote them, which is exactly the
    "key presence" divergence the migration has to close.
    """
    return {
        "images": images,
        "messages": messages,
        "metadata": {"platform": platform, "task_type": "use", "others": others},
    }


def _assert_migrates_to_expected(
    old_row: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    verify = _load_verify_module()
    migrated = _upgrade_lite_sample(old_row)
    verify.verify_lite_sample(migrated)
    assert json.dumps(migrated, separators=(",", ":")) == json.dumps(
        expected,
        separators=(",", ":"),
    )
    return migrated


def _assistant_content(reasoning: str, description: str) -> list[dict[str, Any]]:
    return [
        {"type": "inline_reasoning", "text": reasoning},
        {"type": "action_description", "text": description},
    ]


def _legacy_mobile_terminal_row(
    *,
    status: str,
    reason: str | None = None,
    others: dict[str, Any] | None = None,
    assistant_content: bool = False,
) -> dict[str, Any]:
    """Legacy mobile row with a final real action plus folded ``terminate``.

    Old mobile sources folded ``terminate`` onto the last real action turn,
    emitted bare ``tap`` actions, and paired screenshots as ``role:"user"``
    messages. The final screenshot is the result for the preceding real action,
    not for the dropped terminator.
    """
    terminal_args = {"status": status}
    if reason is not None:
        terminal_args["reason"] = reason

    first_assistant = {"role": "assistant", "tool_calls": [
        _old_fn_call("tap", coordinate=[250, 500], clicks=1),
    ]}
    final_assistant = {"role": "assistant", "tool_calls": [
        _old_fn_call("tap", coordinate=[750, 500], clicks=1),
        _old_fn_call("terminate", **terminal_args),
    ]}
    if assistant_content:
        first_assistant["content"] = _assistant_content("because", "tap a")
        final_assistant["content"] = _assistant_content("then", "tap b")

    return _old_row(
        images=["0.jpeg", "1.jpeg", "2.jpeg"],
        platform="mobile",
        others={"source": "legacy-mobile", **(others or {})},
        messages=[
            {"role": "user", "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "finish"},
            ]},
            first_assistant,
            {"role": "user", "content": [{"type": "image", "index": 1}]},
            final_assistant,
            {"role": "user", "content": [{"type": "image", "index": 2}]},
        ],
    )


def _mobile_call(call_id: str, x: int) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "mobile",
            "arguments": {
                "actions": [
                    {"action": "tap", "coordinate": [x, 500], "clicks": 1},
                ],
            },
        },
    }


def _expected_mobile_terminal_row(
    *,
    others: dict[str, Any],
    assistant_content: bool = False,
) -> dict[str, Any]:
    first_assistant = {"role": "assistant", "tool_calls": [
        _mobile_call("call_0000", 250),
    ]}
    final_assistant = {"role": "assistant", "tool_calls": [
        _mobile_call("call_0001", 750),
    ]}
    if assistant_content:
        first_assistant = {
            "role": "assistant",
            "content": _assistant_content("because", "tap a"),
            "tool_calls": [_mobile_call("call_0000", 250)],
        }
        final_assistant = {
            "role": "assistant",
            "content": _assistant_content("then", "tap b"),
            "tool_calls": [_mobile_call("call_0001", 750)],
        }

    return {
        "images": ["0.jpeg", "1.jpeg", "2.jpeg"],
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "finish"},
            ]},
            first_assistant,
            {
                "role": "tool",
                "tool_call_id": "call_0000",
                "content": [{"type": "image", "index": 1}],
            },
            final_assistant,
            {
                "role": "tool",
                "tool_call_id": "call_0001",
                "content": [{"type": "image", "index": 2}],
            },
            {"role": "assistant", "content": _DONE_CONTENT},
        ],
        "metadata": LiteCUAMetadata(
            dims=("mobile", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others=others,
        ).to_dict(),
    }


def test_canonical_migration_drops_folded_structural_terminate() -> None:
    """``terminate(success)`` asserts only that the row ended."""
    old = _legacy_mobile_terminal_row(status="success")

    migrated = _assert_migrates_to_expected(
        old,
        _expected_mobile_terminal_row(others={"source": "legacy-mobile"}),
    )

    assert all(
        tool_call_name(call) != "terminate"
        for message in migrated["messages"]
        for call in (message.get("tool_calls") or [])
    )


@pytest.mark.parametrize("old_status", ["fail", "failure"])
def test_canonical_migration_records_legacy_failure_alias(old_status: str) -> None:
    """Old published rows spell the failure ``fail`` as well as ``failure``.

    Migration records the canonical spelling in ``metadata.others`` while still
    preserving the legacy row's own observations.
    """
    old = _legacy_mobile_terminal_row(status=old_status)

    migrated = _assert_migrates_to_expected(
        old,
        _expected_mobile_terminal_row(
            others={"source": "legacy-mobile", "terminate_status": "failure"},
        ),
    )

    assert list(migrated["metadata"]["others"])[-1] == "terminate_status"


def test_canonical_migration_never_overwrites_a_published_outcome() -> None:
    """A value the source already published wins over the dropped call's.

    A published row may already carry source-owned outcome fields that disagree
    with the dropped call, so migration must never overwrite an existing key.
    """
    old = _legacy_mobile_terminal_row(
        status="failure",
        others={"terminate_status": "already_published"},
    )

    migrated = _assert_migrates_to_expected(
        old,
        _expected_mobile_terminal_row(
            others={
                "source": "legacy-mobile",
                "terminate_status": "already_published",
            },
        ),
    )

    assert migrated["metadata"]["others"]["terminate_status"] == "already_published"


def test_canonical_migration_failure_reason_moves_to_others() -> None:
    """The authored ``reason`` is the payload the uniform rewrite could destroy.

    The old row's ``arguments`` are null-padded exactly as parquet reads them
    back (a unified Arrow struct pads every call with the union of all sibling
    keys), which is the old-input handling ``clean_nones`` exists for -- without
    it the padded ``status`` / ``reason`` would be read off the *tap* calls too.
    """
    old = _legacy_mobile_terminal_row(
        status="failure",
        reason="app crashed",
        assistant_content=True,
    )
    old["messages"][1]["tool_calls"][0]["function"]["arguments"].update(
        {"status": None, "reason": None}
    )
    old["messages"][3]["tool_calls"][0]["function"]["arguments"].update(
        {"status": None, "reason": None}
    )

    migrated = _assert_migrates_to_expected(
        old,
        _expected_mobile_terminal_row(
            others={
                "source": "legacy-mobile",
                "terminate_status": "failure",
                "terminate_reason": "app crashed",
            },
            assistant_content=True,
        ),
    )

    assert list(migrated["metadata"]["others"])[-2:] == [
        "terminate_status",
        "terminate_reason",
    ]


def test_canonical_migration_rejects_second_pass() -> None:
    """The migration route is one-time legacy-source repair."""
    once = _upgrade_lite_sample(_legacy_mobile_terminal_row(status="failure"))

    with pytest.raises(ValueError, match="legacy-source rows only"):
        _upgrade_lite_sample(once)
