"""Canonical row validation specs for action-batch computer/mobile calls.

Canonical nested contract. GUI use rows carry desktop/mobile actions inside
``computer`` / ``mobile`` action-batch calls, whose payload lives at
``function.arguments.actions[i]``. Point/bbox grounding targets remain
task-local standalone Lite tool calls, not action-batch children.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run --extra data pytest \
        tests/data/utils/test_rows_action_batch_validation.py -p no:cacheprovider -q
"""


from __future__ import annotations

import pytest

from lite.core import LiteCUAMetadata
from lite.core.tools import make_tool_call, make_tool_schema
from lite.core.tools.calls import tool_call_id
from lite.data.utils.rows import validate_canonical_rows


# --- message / tool-call builders -------------------------------------------
def _user(text: str = "obs", img: bool = True) -> dict:
    content = [{"type": "text", "text": text}]
    if img:
        content.append({"type": "image", "index": 0})
    return {"role": "user", "content": content, "tool_calls": []}


def _asst(*tool_calls: dict) -> dict:
    return {"role": "assistant", "content": [], "tool_calls": list(tool_calls)}


def _tool(call_id: str, text: str = "result") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": [{"type": "text", "text": text}]}


def _fn(name: str, **arguments) -> dict:
    return make_tool_call(name, arguments, call_id=f"call_{name}")


def _paired(*tool_calls: dict) -> list[dict]:
    return [
        _user(),
        _asst(*tool_calls),
        *[
            _tool(call_id)
            for tc in tool_calls
            if isinstance(call_id := tool_call_id(tc), str)
        ],
        {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
    ]


def _grounding_label(*tool_calls: dict) -> list[dict]:
    return [_user(), _asst(*tool_calls)]


def _row(messages: list[dict], metadata: dict | None = None) -> dict:
    row: dict = {"messages": messages, "images": ["fixture.png"]}
    if metadata is not None:
        row["metadata"] = metadata
    return row


def _use_metadata(platform: str = "desktop") -> dict:
    return LiteCUAMetadata(
        dims=(platform, "use"),
        extra_tool_schemas=[],
        valid_actions=None,
        others={},
    ).to_dict()


def _batched_computer(*actions: dict) -> dict:
    """ONE canonical action-batch ``computer`` tool_call.

    ``actions`` are inline action dicts ``{"action": <verb>, <arg>: …}`` — the
    coordinate / verb-name the filters care about now sits at
    ``function.arguments.actions[i]``.
    """
    return make_tool_call("computer", {"actions": list(actions)}, call_id="call_computer")


def _batched_mobile(*actions: dict) -> dict:
    return make_tool_call("mobile", {"actions": list(actions)}, call_id="call_mobile")


# Canonical row validation must descend into action-batch coordinates.
def test_stage_validator_flags_oob_standalone_gui_call():
    rows = [_row([_user(), _asst(_fn("click", coordinate=[1500, 5]))], _use_metadata())]
    with pytest.raises(ValueError, match="out-of-range coordinates"):
        validate_canonical_rows(rows, "unit/standalone-action")


def test_stage_validator_descends_into_computer_actions():
    rows = [_row([
        _user(),
        _asst(_batched_computer({"action": "click", "coordinate": [1500, 5]})),
    ], _use_metadata())]
    with pytest.raises(ValueError, match="out-of-range coordinates"):
        validate_canonical_rows(rows, "unit/batched")


def test_stage_validator_accepts_canonical_key_glyphs_inside_action_batch():
    rows = [_row([
        _user(),
        _asst(_batched_computer({"action": "key", "keys": ["ctrl", "+", "-", "="]})),
        _tool("call_computer"),
        {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
    ], _use_metadata())]

    validate_canonical_rows(rows, "unit/batched-key-glyphs")


@pytest.mark.parametrize(
    "key",
    ["plus", "minus", "equal", "comma", " ", "\n", "\t", "\r", "\x1b", "\x00"],
)
def test_stage_validator_rejects_noncanonical_keys_inside_action_batch(key: str):
    rows = [_row([
        _user(),
        _asst(_batched_computer({"action": "key", "keys": ["ctrl", key]})),
    ], _use_metadata())]

    with pytest.raises(ValueError, match="noncanonical or unsupported key"):
        validate_canonical_rows(rows, "unit/batched-bad-key")


@pytest.mark.parametrize("action_name", ["key", "key_down", "key_up", "hold_key"])
def test_stage_validator_rejects_empty_keys_inside_action_batch(action_name: str):
    action = {"action": action_name, "keys": []}
    if action_name == "hold_key":
        action["duration"] = 0.25
    rows = [_row([
        _user(),
        _asst(_batched_computer(action)),
    ], _use_metadata())]

    with pytest.raises(ValueError, match=f"{action_name}.keys must not be empty"):
        validate_canonical_rows(rows, "unit/batched-empty-keys")


def test_stage_validator_descends_into_mobile_start_coordinate():
    rows = [_row([
        _user(),
        _asst(_batched_mobile({
            "action": "swipe",
            "start_coordinate": [1, 1001],
            "coordinate": [10, 10],
        })),
    ], _use_metadata("mobile"))]
    with pytest.raises(ValueError, match="out-of-range coordinates"):
        validate_canonical_rows(rows, "unit/mobile-batched")


@pytest.mark.parametrize("action_name", ["terminate", "goto"])
def test_stage_validator_rejects_standalone_extra_nested_inside_action_batch(action_name: str):
    rows = [_row(
        [_user(), _asst(_batched_computer({"action": action_name}))],
        _use_metadata(),
    )]
    with pytest.raises(ValueError, match=f"computer\\.actions cannot contain {action_name}"):
        validate_canonical_rows(rows, "unit/nested-extra")


@pytest.mark.parametrize("action_name", ["computer", "mobile"])
def test_stage_validator_rejects_action_wrapper_nested_inside_action_batch(action_name: str):
    rows = [_row(
        [_user(), _asst(_batched_computer({"action": action_name}))],
        _use_metadata(),
    )]
    with pytest.raises(ValueError, match=f"computer\\.actions cannot contain {action_name}"):
        validate_canonical_rows(rows, "unit/nested-action-wrapper")


@pytest.mark.parametrize(
    "actions,match",
    [
        ("[]", "actions must be a list"),
        ([], "actions must be a non-empty list"),
        (["click"], "actions\\[0\\] must be a dict"),
        ([{}], "action must be a non-empty string"),
    ],
)
def test_stage_validator_rejects_malformed_action_batch_actions(actions, match):
    rows = [
        _row([
            _user(),
            _asst(make_tool_call(
                "computer",
                {"actions": actions},
                call_id="call_computer",
            )),
        ], _use_metadata())
    ]
    with pytest.raises(ValueError, match=match):
        validate_canonical_rows(rows, "unit/malformed-batch")


@pytest.mark.parametrize(
    "tool_call,match",
    [
        (
            {
                "call_id": "call_click",
                "name": "click",
                "arguments": {"coordinate": [1, 2]},
                "type": "function_call",
            },
            "noncanonical",
        ),
        ({"call_id": "call_click", "name": "click", "arguments": "null"}, "arguments"),
        ({"function": {"name": "click", "arguments": {"coordinate": [1, 2]}}}, "noncanonical"),
    ],
)
def test_stage_validator_rejects_noncanonical_tool_calls_at_stage(tool_call: dict, match: str):
    rows = [_row([_user(), _asst(tool_call)])]
    with pytest.raises(ValueError, match=match):
        validate_canonical_rows(rows, "unit/noncanonical")


def test_stage_validator_rejects_noncanonical_extra_tool_schemas_at_stage():
    rows = [_row(
        [_user(), _asst(_fn("click", coordinate=[1, 2]))],
        LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[{
                "type": "function",
                "function": {"name": "goto", "parameters": {}},
            }],
            valid_actions=None,
            others={},
        ).to_dict(),
    )]
    with pytest.raises(ValueError, match="extra_tool_schemas"):
        validate_canonical_rows(rows, "unit/noncanonical-schema")


def test_stage_validator_rejects_metadata_split_at_stage():
    rows = [_row(
        [_user(), _asst(_fn("click", coordinate=[1, 2]))],
        {**LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(), 'split': "train"},
    )]
    with pytest.raises(ValueError, match="metadata\\.split"):
        validate_canonical_rows(rows, "unit/split-leak")


def test_stage_validator_rejects_missing_metadata_at_stage():
    rows = [_row([_user(), _asst(_fn("click", coordinate=[1, 2]))])]

    with pytest.raises(ValueError, match="metadata is required"):
        validate_canonical_rows(rows, "unit/missing-metadata")


@pytest.mark.parametrize(
    "tool_call,match",
    [
        (_fn("click", coordinate=[1, 2]), "missing from metadata\\.extra_tool_schemas"),
        (_fn("tap", coordinate=[1, 2]), "missing from metadata\\.extra_tool_schemas"),
        (
            _fn("point", coordinate=[1, 2]),
            r"not valid for metadata_kind=cua dims=\('desktop', 'use'\)",
        ),
        (
            _fn("bbox", coordinate=[1, 2, 3, 4]),
            r"not valid for metadata_kind=cua dims=\('desktop', 'use'\)",
        ),
        (_fn("terminate", status="success"), "missing from metadata\\.extra_tool_schemas"),
        (_fn("goto", url="https://example.com"), "missing from metadata\\.extra_tool_schemas"),
    ],
)
def test_stage_validator_rejects_standalone_tool_call_missing_extra_schema(
    tool_call: dict,
    match: str,
):
    rows = [_row(
        _paired(tool_call),
        LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    )]

    with pytest.raises(ValueError, match=match):
        validate_canonical_rows(rows, "unit/missing-schema")


def test_stage_validator_accepts_canonical_stage_row():
    rows = [_row(
        _paired(_fn("goto", url="https://example.com")),
        LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[make_tool_schema(
                "goto",
                parameters={
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            )],
            valid_actions=None,
            others={},
        ).to_dict(),
    )]
    validate_canonical_rows(rows, "unit/canonical")


@pytest.mark.parametrize(
    "platform,tool_call",
    [
        ("desktop", _batched_computer({"action": "click", "coordinate": [1, 2]})),
        ("browser", _batched_computer({"action": "click", "coordinate": [1, 2]})),
        ("mobile", _batched_mobile({"action": "tap", "coordinate": [3, 4]})),
    ],
)
def test_stage_validator_accepts_canonical_action_wrappers_without_extra_schema(
    platform, tool_call
):
    rows = [_row(
        _paired(tool_call),
        LiteCUAMetadata(
            dims=(platform, "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    )]

    validate_canonical_rows(rows, "unit/canonical-action")


def test_stage_validator_rejects_wrong_platform_action_wrapper():
    rows = [_row(
        _paired(_batched_mobile({"action": "tap", "coordinate": [3, 4]})),
        LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    )]

    with pytest.raises(
        ValueError,
        match=r"not valid for metadata_kind=cua dims=\('desktop', 'use'\)",
    ):
        validate_canonical_rows(rows, "unit/wrong-platform-action")


def test_stage_validator_rejects_schema_free_tool_declared_as_extra_schema():
    rows = [_row(
        [_user(), _asst(_batched_computer({"action": "click", "coordinate": [1, 2]}))],
        LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[make_tool_schema(
                "computer",
                parameters={
                    "type": "object",
                    "properties": {"actions": {"type": "array"}},
                    "required": ["actions"],
                },
            )],
            valid_actions=None,
            others={},
        ).to_dict(),
    )]

    with pytest.raises(ValueError, match="must not redeclare canonical top-level GUI"):
        validate_canonical_rows(rows, "unit/schema-free-extra")


def test_stage_validator_allows_schema_shaped_action_named_extra_tool():
    rows = [_row(
        _paired(_fn("click", index=7)),
        LiteCUAMetadata(
            dims=("browser", "use"),
            extra_tool_schemas=[make_tool_schema(
                "click",
                parameters={
                    "type": "object",
                    "properties": {"index": {"type": "integer"}},
                    "required": ["index"],
                },
            )],
            valid_actions=[],
            others={},
        ).to_dict(),
    )]

    validate_canonical_rows(rows, "unit/action-named-extra")


def test_stage_validator_rejects_nested_schema_shaped_action_named_extra_tool():
    rows = [_row(
        _paired(_batched_computer({"action": "click", "index": 7})),
        LiteCUAMetadata(
            dims=("browser", "use"),
            extra_tool_schemas=[make_tool_schema(
                "click",
                parameters={
                    "type": "object",
                    "properties": {"index": {"type": "integer"}},
                    "required": ["index"],
                },
            )],
            valid_actions=None,
            others={},
        ).to_dict(),
    )]

    with pytest.raises(ValueError, match="must not nest standalone extra tool 'click'"):
        validate_canonical_rows(rows, "unit/nested-action-named-extra")


@pytest.mark.parametrize(
    "task_type,tool_call",
    [
        ("grounding.point", _fn("point", coordinate=[1, 2])),
        ("grounding.bbox", _fn("bbox", coordinate=[1, 2, 3, 4])),
    ],
)
def test_stage_validator_accepts_task_local_grounding_calls_without_extra_schema(
    task_type: str,
    tool_call: dict,
):
    rows = [_row(
        _grounding_label(tool_call),
        LiteCUAMetadata(
            dims=("desktop", task_type),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    )]

    validate_canonical_rows(rows, f"unit/{task_type}")


@pytest.mark.parametrize(
    "platform,tool_call",
    [
        ("desktop", _batched_computer({"action": "click", "coordinate": [1, 2]})),
        ("browser", _batched_computer({"action": "type", "text": "hello"})),
        ("mobile", _batched_mobile({"action": "tap", "coordinate": [3, 4]})),
    ],
)
def test_stage_validator_accepts_grounding_action_calls_without_extra_schema(
    platform: str,
    tool_call: dict,
):
    rows = [_row(
        _grounding_label(tool_call),
        LiteCUAMetadata(
            dims=(platform, "grounding.action"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    )]

    validate_canonical_rows(rows, f"unit/grounding-action-{platform}")


@pytest.mark.parametrize(
    "platform,tool_call",
    [
        ("desktop", _fn("click", coordinate=[1, 2])),
        ("browser", _fn("type", text="hello")),
        ("mobile", _fn("tap", coordinate=[3, 4])),
    ],
)
def test_stage_validator_rejects_legacy_bare_grounding_action_calls(
    platform: str,
    tool_call: dict,
):
    rows = [_row(
        _grounding_label(tool_call),
        LiteCUAMetadata(
            dims=(platform, "grounding.action"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    )]

    with pytest.raises(ValueError, match="standalone but missing|not valid"):
        validate_canonical_rows(rows, f"unit/grounding-action-{platform}-legacy")


def test_stage_validator_rejects_fake_grounding_tool_result():
    rows = [_row(
        _paired(_batched_computer({"action": "click", "coordinate": [1, 2]})),
        LiteCUAMetadata(
            dims=("desktop", "grounding.action"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    )]

    with pytest.raises(ValueError, match="orphan role:tool result"):
        validate_canonical_rows(rows, "unit/grounding-action-fake-result")
