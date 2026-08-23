"""Regression tests for the SFT-export *same-source* contract.

The bug these guard against: ``export_sft._convert_sample`` used to cherry-pick
``extra_tools`` / ``valid_actions`` from the saved metadata and pass them as
loose adapter **kwargs**. Those are ``LiteCUAMetadata`` fields, not adapter fields,
so ``BaseAgentAdapter._apply_kwargs`` silently dropped them — the SFT-rendered
system prompt then (a) lost the env's nav ``extra_tool_schemas`` and (b) never
trimmed the action enum. The fix: forward the WHOLE ``metadata`` object (exactly
as rollout's ``make`` does), so the rendered SFT prompt matches the surface
the data was COLLECTED under.

Two consequences are pinned here for the gpt→qwen3_vl distill case:
  1. nav ``extra_tool_schemas`` (goto/back/...) survive into the SFT system prompt;
  2. a grounded ``valid_actions`` keeps the interaction enum (click/type/key/
     scroll) — it is NOT collapsed to finish-only — AND export renders the exact
     same tools section the rollout adapter would (same-source parity).

Run: uv run pytest tests/train/export/test_export_sft_metadata.py -v
"""
from __future__ import annotations

import json
import sys

import pytest

pytest.importorskip("transformers", reason="transformers not installed")

from PIL import Image

import lite.train.export.export_sft as export_sft
from lite.agents.core.adapter import (
    AGENT_KWARGS_TOOL_SURFACE_KEYS,
    AgentAdapterRegistry,
)
from lite.agents.core.adapter import (
    TOOL_SURFACE_KWARGS as ADAPTER_TOOL_SURFACE_KWARGS,
)
from lite.core import LiteBaseMetadata, LiteCUAMetadata, LiteGenericMetadata
from lite.core.errors import LiteContractError
from lite.core.tools import make_tool_call, make_tool_schema
from lite.core.tools.calls import tool_call_name
from lite.core.tools.extra_tools import LiteBrowserNavToolSet, LiteFinishToolSet
from lite.core.tools.schemas import tool_schema_name
from lite.train.export.export_sft import _convert_sample
from lite.utils.parquet import write_records_to_parquet

# Grounded interaction verbs; standalone response is selected via extra_tool_schemas.
_GROUNDED_VALID = ["click", "type", "key", "scroll", "wait"]

# Small Qwen3-VL processor matching the qwen3_vl adapter under test — tokenizes
# the export step so `_convert_sample` emits serialized LiteRLStep structs. The
# assertions read the chat-template-rendered `prompt` (system + first user turn).
_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


def _cua_metadata_dict(
    *,
    platform: str = "desktop",
    task_type: str = "use",
    extra_tool_schemas: list[dict] | None = None,
    valid_actions: list[str] | None = None,
    others: dict | None = None,
) -> dict:
    return LiteCUAMetadata(
        dims=(platform, task_type),
        extra_tool_schemas=extra_tool_schemas or [],
        valid_actions=valid_actions,
        others=others or {},
    ).to_dict()


def _rendered_tool_call_name(tool_call: dict) -> str:
    """Name from either canonical Lite calls or adapter-rendered model calls."""
    if isinstance(tool_call.get("function"), dict):
        return tool_call_name(tool_call)
    return tool_call["name"]


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [1] * len(text.split())


class _FakeProcessor:
    tokenizer = _FakeTokenizer()
    image_token = "<|image_pad|>"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    ):
        chunks: list[str] = []
        for message in messages:
            chunks.append(f"<{message.get('role')}>")
            for part in message.get("content") or []:
                if part.get("type") == "image":
                    chunks.append("<|image_pad|>")
                elif part.get("type") == "text" and part.get("text"):
                    chunks.append(part["text"])
            for tool_call in message.get("tool_calls") or []:
                chunks.append(f"<tool_call>{_rendered_tool_call_name(tool_call)}</tool_call>")
        if add_generation_prompt:
            chunks.append("<assistant>")
        return "\n".join(chunks)


def _system_text(steps: list) -> str:
    """The first step's rendered prompt string (system + first user turn). Steps
    are serialized LiteRLStep structs now, so the system section lives inside the
    chat-template-rendered ``prompt``."""
    return steps[0]["prompt"]


def _webgym_row(tmp_path, *, extra_tool_schemas, valid_actions) -> dict:
    """A GPT-teacher webgym-style trajectory row (parquet schema) with one
    on-disk image, ready for ``_convert_sample``."""
    img_path = tmp_path / "frame0.png"
    Image.new("RGB", (1280, 768), (200, 200, 200)).save(img_path)
    messages = [
        {"role": "user", "content": [
            {"type": "image", "index": 0},
            {"type": "text", "text": "Find the answer."},
        ]},
        {"role": "assistant",
         "content": [{"type": "action_description", "text": "click the result link"}],
         "tool_calls": [make_tool_call(
             "computer",
             {
                 "actions": [{"action": "click", "coordinate": [500, 300]}],
             },
             call_id="call_0000",
         )]},
        {
            "role": "tool",
            "tool_call_id": "call_0000",
            "content": [{"type": "text", "text": "ok"}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
    ]
    metadata = _cua_metadata_dict(
        platform="browser",
        extra_tool_schemas=extra_tool_schemas,
        valid_actions=valid_actions,
        others={"env_id": "webgym"},
    )
    return {"images": [str(img_path)], "messages": messages, "metadata": metadata}


def _content_final_row(tmp_path) -> dict:
    img_path = tmp_path / "frame0.png"
    Image.new("RGB", (1280, 768), (200, 200, 200)).save(img_path)
    return {
        "images": [str(img_path)],
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "Finish the desktop task."},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        ],
        "metadata": _cua_metadata_dict(others={"env_id": "lite.osworld"}),
    }


def _explicit_eof_finish_row(tmp_path, *, name: str, arguments: dict) -> dict:
    img_path = tmp_path / f"frame0_{name}.png"
    Image.new("RGB", (1280, 768), (200, 200, 200)).save(img_path)
    return {
        "images": [str(img_path)],
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "Finish the desktop task."},
            ]},
            {
                "role": "assistant",
                "tool_calls": [
                    make_tool_call(name, arguments, call_id=f"call_{name}_eof")
                ],
            },
        ],
        "metadata": _cua_metadata_dict(
            extra_tool_schemas=[LiteFinishToolSet.get_tool_schema(name)],
            others={"env_id": "lite.osworld"},
        ),
    }


_AGENT_SURFACE_REJECT_VALUES = {
    "extra_tools": ["terminate"],
    "extra_tool_schemas": LiteBrowserNavToolSet.get_tool_schemas(include=["goto"]),
    "metadata": _cua_metadata_dict(platform="browser"),
    "valid_actions": ["click"],
    "others": {"surface": "from-agent-kwargs"},
}


def test_export_uses_agent_kwargs_surface_rejection_keys():
    """Adapter surface kwargs are metadata fields; agent_kwargs rejection also
    rejects ``metadata`` itself because export supplies the resolved metadata.

    The census is written out as a LITERAL and cross-checked against the names
    ``test_export_rejects_env_surface_or_metadata_in_agent_kwargs`` actually
    feeds through ``_convert_sample``. It used to read
    ``AGENT_KWARGS_TOOL_SURFACE_KEYS == ADAPTER_TOOL_SURFACE_KWARGS |
    {"metadata"}``, which is the DEFINITION at ``adapter/base.py:129`` restated
    one import away (``ADAPTER_TOOL_SURFACE_KWARGS`` is an alias of the very
    ``TOOL_SURFACE_KWARGS`` the union is built from) -- both sides recomputed
    from the same object at assert time, so no edit to the definition could
    redden it. Dropping a name now reddens here AND drops a rejection case.
    """
    assert "metadata" not in ADAPTER_TOOL_SURFACE_KWARGS
    assert AGENT_KWARGS_TOOL_SURFACE_KEYS == frozenset({
        "extra_tools", "extra_tool_schemas", "metadata", "others", "valid_actions",
    })
    assert frozenset(_AGENT_SURFACE_REJECT_VALUES) == AGENT_KWARGS_TOOL_SURFACE_KEYS
    assert not hasattr(export_sft, "TOOL_SURFACE_KWARGS")


def test_export_preserves_nav_extra_tools_and_interaction_enum(tmp_path):
    """gpt→qwen3_vl webgym export: nav extra_tool_schemas survive AND the grounded
    valid_actions keeps the interaction enum (not collapsed to answer-only)."""
    nav = LiteBrowserNavToolSet.get_tool_schemas(include=["goto", "back"])
    row = _webgym_row(tmp_path, extra_tool_schemas=nav, valid_actions=_GROUNDED_VALID)
    out = _convert_sample(
        row, agent_id="qwen3_vl", agent_kwargs={}, model_id=_MODEL_ID,
        image_root=None, strict=True,
    )
    assert out["_error"] == ""
    # SFT call-site guard (issue #42): this webgym row carries ``{type:metadata}``
    # parts, so reaching here under strict=True proves ``agent_step_to_rl_step``
    # sanitized them before ``apply_chat_template`` (else it raises). And the SFT
    # target must SURVIVE that sanitize — a silent target-drop wouldn't trip
    # ``_error``, so assert the response is non-empty.
    assert out["steps"][0]["response"].strip(), (
        "SFT target empty — metadata sanitize must not drop the assistant target"
    )
    sys_text = _system_text(out["steps"])

    # (1) nav extra_tools survived the export → rendered into the <tools> block.
    assert "goto" in sys_text, "goto extra_tool dropped from SFT system prompt"
    assert "back" in sys_text, "back extra_tool dropped from SFT system prompt"
    # (2) interaction enum NOT collapsed — the grounded verbs are present.
    for verb in ("click", "type", "scroll"):
        assert verb in sys_text, f"interaction verb {verb!r} missing — enum collapsed"


def test_export_projects_action_batch_computer_to_qwen_wrapper(tmp_path):
    row = _webgym_row(
        tmp_path,
        extra_tool_schemas=LiteBrowserNavToolSet.get_tool_schemas(include=["goto"]),
        valid_actions=["click", "type"],
    )
    row["messages"][1]["tool_calls"] = [make_tool_call(
        "computer",
        {
            "actions": [
                {"action": "click", "coordinate": [500, 300]},
            ]
        },
        call_id="call_0000",
    )]

    out = _convert_sample(
        row, agent_id="qwen3_vl", agent_kwargs={}, model_id=_MODEL_ID,
        image_root=None, strict=True,
    )
    assert out["_error"] == ""
    response = out["steps"][0]["response"]
    assert "computer_use" in response
    assert "left_click" in response
    assert '"name": "computer"' not in response


def test_export_preserves_canonical_json_metadata_storage(tmp_path):
    row = _webgym_row(
        tmp_path,
        extra_tool_schemas=LiteBrowserNavToolSet.get_tool_schemas(include=["goto"]),
        valid_actions=_GROUNDED_VALID,
    )
    assert "extra_tools" not in row["metadata"]
    stored_metadata = json.dumps(row["metadata"])
    row["metadata"] = stored_metadata

    out = _convert_sample(
        row, agent_id="qwen3_vl", agent_kwargs={}, model_id=_MODEL_ID,
        image_root=None, strict=True,
    )

    assert out["metadata"] == stored_metadata
    parsed = json.loads(out["metadata"])
    assert "extra_tool_schemas" in parsed
    assert [tool_schema_name(s) for s in parsed["extra_tool_schemas"]] == ["goto"]
    assert "extra_tools" not in parsed


def test_export_generic_empty_dims_uses_bare_base_adapter(monkeypatch):
    """Generic rows route by empty dims and replay only their stored tool surface."""
    monkeypatch.setattr(
        "lite.train.export.export_sft._get_processor",
        lambda model_id: _FakeProcessor(),
    )
    row = {
        "images": [],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Compute the diagonal."}],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    make_tool_call(
                        "response",
                        {"text": "sqrt(2)"},
                        call_id="call_response",
                    )
                ],
            },
        ],
        "metadata": LiteGenericMetadata(
            dims=(),
            extra_tool_schemas=[LiteFinishToolSet.get_tool_schema("response")],
            others={"env_id": "geo3k"},
        ).to_dict(),
    }

    out = _convert_sample(
        row,
        agent_id="qwen3_5.base",
        agent_kwargs={},
        model_id=_MODEL_ID,
        image_root=None,
        strict=True,
    )

    assert out["_error"] == ""
    assert out["processed_images"] == []
    assert len(out["steps"]) == 1
    assert "sqrt(2)" in out["steps"][0]["response"]
    sys_text = _system_text(out["steps"])
    assert "response" in sys_text
    assert "computer_use" not in sys_text
    assert "left_click" not in sys_text
    parsed = json.loads(out["metadata"])
    assert parsed["metadata_kind"] == "generic"
    assert parsed["dims"] == []
    assert [tool_schema_name(s) for s in parsed["extra_tool_schemas"]] == ["response"]
    assert "valid_actions" not in parsed


def test_export_preserves_content_only_final_text_without_finish_schema(tmp_path):
    """Default lite/data final policy: a no-tool final ``Done.`` remains text.

    Export must not synthesize a terminate tool call or a terminate schema just
    because the target student is qwen. Offline render gets tool availability
    only from the parquet row's stored metadata.
    """
    row = _content_final_row(tmp_path)
    out = _convert_sample(
        row, agent_id="qwen3_vl", agent_kwargs={}, model_id=_MODEL_ID,
        image_root=None, strict=True,
    )

    assert out["_error"] == ""
    assert "Done." in out["steps"][0]["response"]
    assert "Action: Done." not in out["steps"][0]["response"]
    assert "<tool_call>" not in out["steps"][0]["response"]
    assert "terminate(" not in out["steps"][0]["response"].lower()
    parsed = json.loads(out["metadata"]) if isinstance(out["metadata"], str) else out["metadata"]
    assert parsed["extra_tool_schemas"] == []


@pytest.mark.parametrize(
    ("name", "arguments", "forbidden_feedback"),
    [
        ("response", {"text": "Done."}, "Final answer submitted: Done."),
        ("terminate", {"status": "success"}, "Task terminated: success"),
    ],
)
def test_export_accepts_explicit_eof_finish_without_fake_tool_response(
    tmp_path,
    monkeypatch,
    name,
    arguments,
    forbidden_feedback,
):
    monkeypatch.setattr(
        "lite.train.export.export_sft._get_processor",
        lambda model_id: _FakeProcessor(),
    )
    row = _explicit_eof_finish_row(tmp_path, name=name, arguments=arguments)
    assert row["messages"][-1]["role"] == "assistant"
    assert tool_call_name(row["messages"][-1]["tool_calls"][0]) == name
    assert all(message.get("role") != "tool" for message in row["messages"])

    out = _convert_sample(
        row,
        agent_id="qwen3_vl",
        agent_kwargs={},
        model_id=_MODEL_ID,
        image_root=None,
        strict=True,
    )

    assert out["_error"] == ""
    assert len(out["steps"]) == 1
    rendered = f"{out['steps'][0]['prompt']}\n{out['steps'][0]['response']}"
    assert "<tool_call>" in out["steps"][0]["response"]
    assert "<tool>" not in rendered
    assert forbidden_feedback not in rendered
    assert "Final answer submitted:" not in rendered
    assert "Task terminated:" not in rendered


def test_export_does_not_gate_action_description_only_final(tmp_path):
    row = _content_final_row(tmp_path)
    row["messages"][-1]["content"] = [{"type": "action_description", "text": "Done."}]

    out = _convert_sample(
        row, agent_id="qwen3_vl", agent_kwargs={}, model_id=_MODEL_ID,
        image_root=None, strict=True,
    )

    assert out["_error"] == ""
    assert "Action: Done." in out["steps"][0]["response"]


def test_export_replays_matching_raw_response_for_noncanonical_final(tmp_path):
    row = _content_final_row(tmp_path)
    row["messages"][-1]["content"] = [{"type": "action_description", "text": "Done."}]
    row["messages"][-1]["raw_response"] = {
        "adapter_key": "qwen3_vl@desktop@use",
        "text": "Done.",
    }

    out = _convert_sample(
        row, agent_id="qwen3_vl", agent_kwargs={}, model_id=_MODEL_ID,
        image_root=None, strict=True,
    )

    assert out["_error"] == ""
    assert "Done." in out["steps"][0]["response"]


def test_export_ignores_foreign_raw_response_without_content_gate(tmp_path):
    row = _content_final_row(tmp_path)
    row["messages"].insert(1, {
        "role": "assistant",
        "content": [{"type": "action_description", "text": "click"}],
        "tool_calls": [make_tool_call(
            "computer",
            {"actions": [{"action": "click", "coordinate": [100, 200]}]},
            call_id="call_0000",
        )],
        "raw_response": {
            "adapter_key": "qwen3_vl@desktop@use",
            "text": "matching prior raw response",
        },
    })
    row["messages"].insert(2, {
        "role": "tool",
        "tool_call_id": "call_0000",
        "content": [{"type": "text", "text": "ok"}],
    })
    row["messages"][-1]["content"] = [{"type": "action_description", "text": "Done."}]
    row["messages"][-1]["raw_response"] = {
        "adapter_key": "gpt@desktop@use",
        "text": "FOREIGN RAW FINAL SHOULD NOT TRAIN",
    }

    out = _convert_sample(
        row, agent_id="qwen3_vl", agent_kwargs={}, model_id=_MODEL_ID,
        image_root=None, strict=True,
    )

    assert out["_error"] == ""
    assert "Action: Done." in out["steps"][-1]["response"]
    assert "FOREIGN RAW FINAL SHOULD NOT TRAIN" not in out["steps"][-1]["response"]


def test_export_strips_matching_raw_response_from_content_only_final(tmp_path):
    row = _content_final_row(tmp_path)
    row["messages"][-1]["raw_response"] = {
        "adapter_key": "qwen3_vl@desktop@use",
        "text": "RAW FINAL SHOULD NOT TRAIN",
    }

    out = _convert_sample(
        row, agent_id="qwen3_vl", agent_kwargs={}, model_id=_MODEL_ID,
        image_root=None, strict=True,
    )

    assert "Done." in out["steps"][0]["response"]
    assert "RAW FINAL SHOULD NOT TRAIN" not in out["steps"][0]["response"]


def _click_index_schema() -> dict:
    return make_tool_schema(
        "click",
        description="Click an indexed element.",
        parameters={
            "type": "object",
            "properties": {"index": {"type": "integer"}},
            "required": ["index"],
        },
    )


def test_raw_export_replays_matching_raw_response_after_schema_validation(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "lite.train.export.export_sft._get_processor",
        lambda model_id: _FakeProcessor(),
    )
    row = _webgym_row(tmp_path, extra_tool_schemas=[_click_index_schema()], valid_actions=[])
    row["messages"][1]["tool_calls"] = [
        make_tool_call("click", {"index": 7}, call_id="call_0000")
    ]
    row["messages"][1]["raw_response"] = {
        "adapter_key": "qwen3_vl@browser@use",
        "text": "same-family raw click",
    }

    out = _convert_sample(
        row,
        agent_id="qwen3_vl",
        agent_kwargs={},
        model_id=_MODEL_ID,
        image_root=None,
        strict=True,
    )

    assert out["_error"] == ""
    assert "same-family raw click" in out["steps"][0]["response"]


def test_raw_export_does_not_repair_flat_tool_call_or_replay_raw_response(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "lite.train.export.export_sft._get_processor",
        lambda model_id: _FakeProcessor(),
    )
    row = _webgym_row(tmp_path, extra_tool_schemas=[_click_index_schema()], valid_actions=[])
    row["messages"][1]["tool_calls"] = [
        {
            "call_id": "call_0000",
            "name": "click",
            "arguments": {"index": 7},
        }
    ]
    row["messages"][1]["raw_response"] = {
        "adapter_key": "qwen3_vl@browser@use",
        "text": "same-family raw click",
    }

    out = _convert_sample(
        row,
        agent_id="qwen3_vl",
        agent_kwargs={},
        model_id=_MODEL_ID,
        image_root=None,
        strict=False,
    )

    assert out["_error"]
    assert "devs/migration" not in out["_error"]
    assert out["steps"] == []


def test_export_strict_false_metadata_parse_failure_uses_generic_sentinel(tmp_path):
    row = _content_final_row(tmp_path)
    row["metadata"] = {
        "dims": ["desktop", "use"],
        "extra_tool_schemas": [],
        "valid_actions": None,
        "others": {"env_id": "lite.osworld"},
    }

    out = _convert_sample(
        row,
        agent_id="qwen3_vl",
        agent_kwargs={},
        model_id=_MODEL_ID,
        image_root=None,
        strict=False,
    )

    assert out["_error"]
    assert out["processed_images"] == []
    assert out["steps"] == []
    assert json.loads(out["metadata"]) == LiteGenericMetadata(
        dims=("export_error",),
        others={"reason": "metadata_parse_failed"},
    ).to_dict()


def test_export_filter_receives_lite_base_metadata(tmp_path, monkeypatch):
    rows = [
        {
            "images": [],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "skip"}]}],
            "metadata": LiteCUAMetadata(
                dims=("desktop", "use"),
                others={"keep": True},
            ).to_dict(),
        },
        {
            "images": [],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "keep"}]}],
            "metadata": LiteGenericMetadata(
                dims=("filter",),
                others={"keep": True},
            ).to_dict(),
        },
    ]
    input_path = tmp_path / "input.parquet"
    write_records_to_parquet(rows, input_path, json_fields=("messages", "metadata"))

    def recording_filter(metadata: LiteBaseMetadata) -> bool:
        assert isinstance(metadata, LiteBaseMetadata)
        return (
            metadata.metadata_kind == "generic"
            and metadata.dims == ("filter",)
            and metadata.others.get("keep")
        )

    def fake_convert(raw: dict, **_kwargs) -> dict:
        metadata = raw["metadata"]
        if not isinstance(metadata, str):
            metadata = json.dumps(metadata)
        return {
            "_error": "",
            "processed_images": [],
            "steps": [],
            "metadata": metadata,
        }

    written_rows: list[dict] = []

    def fake_write(dataset, _output, **_kwargs) -> None:
        written_rows.extend(dataset.to_list())

    monkeypatch.setattr(export_sft, "parse_filter", lambda _expr: recording_filter)
    monkeypatch.setattr(export_sft, "_convert_sample", fake_convert)
    monkeypatch.setattr(export_sft, "_write_model_ready_parquet", fake_write)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_sft",
            "--agent-id",
            "stub",
            "--model-id",
            "stub",
            "--data-paths",
            str(tmp_path),
            "--filter",
            "lambda m: True",
            "--num-proc",
            "1",
            "-o",
            str(tmp_path / "out.parquet"),
        ],
    )

    export_sft.main()

    assert [json.loads(row["metadata"])["metadata_kind"] for row in written_rows] == ["generic"]


def test_export_filter_generic_platform_access_fails_at_filter_boundary(
    tmp_path,
    monkeypatch,
):
    rows = [
        {
            "images": [],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "generic"}]}],
            "metadata": LiteGenericMetadata(dims=("filter",)).to_dict(),
        }
    ]
    write_records_to_parquet(
        rows,
        tmp_path / "input.parquet",
        json_fields=("messages", "metadata"),
    )

    def cua_only_filter(metadata: LiteBaseMetadata) -> bool:
        return bool(metadata.platform)

    monkeypatch.setattr(export_sft, "parse_filter", lambda _expr: cua_only_filter)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_sft",
            "--agent-id",
            "stub",
            "--model-id",
            "stub",
            "--data-paths",
            str(tmp_path),
            "--filter",
            "lambda m: True",
            "--num-proc",
            "1",
            "-o",
            str(tmp_path / "out.parquet"),
        ],
    )

    with pytest.raises(AttributeError, match="platform"):
        export_sft.main()


def test_export_rejects_out_of_range_image_index(tmp_path):
    row = _content_final_row(tmp_path)
    row["messages"][0]["content"][0]["index"] = 1

    with pytest.raises(LiteContractError, match="out of range for images length 1"):
        _convert_sample(
            row, agent_id="qwen3_vl", agent_kwargs={}, model_id=_MODEL_ID,
            image_root=None, strict=True,
        )


def test_export_lite_osworld_accepts_integral_float_image_index(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "lite.train.export.export_sft._get_processor",
        lambda model_id: _FakeProcessor(),
    )
    row = _content_final_row(tmp_path)
    row["messages"][0]["content"][0]["index"] = 0.0

    out = _convert_sample(
        row,
        agent_id="qwen3_vl",
        agent_kwargs={},
        model_id=_MODEL_ID,
        image_root=None,
        strict=True,
    )

    assert out["_error"] == ""
    assert "<|image_pad|>" in out["steps"][0]["prompt"]
    assert "Done." in out["steps"][0]["response"]


def test_convert_sample_preserves_sparse_processed_images(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "lite.train.export.export_sft._get_processor",
        lambda model_id: _FakeProcessor(),
    )
    row = _content_final_row(tmp_path)
    for i in (1, 2):
        img_path = tmp_path / f"frame{i}.png"
        Image.new("RGB", (1280, 768), (200, 200, 200)).save(img_path)
        row["images"].append(str(img_path))
    row["messages"][0]["content"].insert(1, {"type": "image", "index": 2})

    out = _convert_sample(
        row,
        agent_id="qwen3_vl",
        agent_kwargs={},
        model_id=_MODEL_ID,
        image_root=None,
        strict=True,
    )

    assert out["_error"] == ""
    assert out["steps"][0]["image_indices"] == [0, 2]
    assert len(out["processed_images"]) == 3
    assert isinstance(out["processed_images"][0], bytes)
    assert out["processed_images"][1] is None
    assert isinstance(out["processed_images"][2], bytes)


def test_export_keeps_nonintegral_float_image_refs_invalid(tmp_path):
    row = _content_final_row(tmp_path)
    row["messages"][0]["content"][0]["index"] = 0.5

    with pytest.raises(LiteContractError, match="non-negative integer"):
        _convert_sample(
            row,
            agent_id="qwen3_vl",
            agent_kwargs={},
            model_id=_MODEL_ID,
            image_root=None,
            strict=True,
        )


def test_export_does_not_run_persisted_row_id_gate_on_provider_envelope(tmp_path):
    """Export is conversion smoke, so stage owns persisted call-id rejection."""
    row = _webgym_row(tmp_path, extra_tool_schemas=[], valid_actions=None)
    row["messages"][1]["tool_calls"] = [{
        "type": "function",
        "function": {
            "name": "click",
            "arguments": {"coordinate": [500, 300]},
        },
    }]

    out = _convert_sample(
        row,
        agent_id="qwen3_vl",
        agent_kwargs={},
        model_id=_MODEL_ID,
        image_root=None,
        strict=True,
    )

    assert out["_error"] == ""
    assert "computer_use" in out["steps"][0]["response"]


#: Parametrized off the LITERAL census, not off ``AGENT_KWARGS_TOOL_SURFACE_KEYS``:
#: a name dropped from the definition must still be fed to ``_convert_sample``
#: and still be rejected. Driving the parametrize from the constant would make a
#: dropped name SHRINK the parametrize -- the case vanishes instead of failing.
@pytest.mark.parametrize(
    ("field", "value"),
    [
        (field, _AGENT_SURFACE_REJECT_VALUES[field])
        for field in sorted(_AGENT_SURFACE_REJECT_VALUES)
    ],
)
def test_export_rejects_env_surface_or_metadata_in_agent_kwargs(tmp_path, field, value):
    """Offline export config may tune adapter rendering via agent_kwargs, but the
    env/tool surface must come only from the parquet row's metadata."""
    row = _webgym_row(tmp_path, extra_tool_schemas=[], valid_actions=None)

    with pytest.raises(TypeError, match="tool-surface settings|not adapter kwargs"):
        _convert_sample(
            row,
            agent_id="qwen3_vl",
            agent_kwargs={field: value},
            model_id=_MODEL_ID,
            image_root=None,
            strict=True,
        )


def test_export_render_matches_rollout_render_same_source(tmp_path):
    """Same-source parity: the tools section export renders for a trajectory's
    saved metadata is byte-identical to what the rollout adapter renders for the
    same metadata (export forwards the whole metadata object, like make)."""
    nav = LiteBrowserNavToolSet.get_tool_schemas(include=["goto", "back", "forward"])
    row = _webgym_row(tmp_path, extra_tool_schemas=nav, valid_actions=_GROUNDED_VALID)

    # Export path.
    out = _convert_sample(
        row, agent_id="qwen3_vl", agent_kwargs={}, model_id=_MODEL_ID,
        image_root=None, strict=True,
    )
    export_sys = _system_text(out["steps"])

    # Rollout path: same adapter key + same metadata object.
    adapter = AgentAdapterRegistry.get(
        "qwen3_vl@browser@use",
        metadata=LiteCUAMetadata.from_dict(row["metadata"]),
    )
    rollout_tools = adapter._build_tools_section()

    # The rollout tools section is embedded verbatim in the export system prompt.
    assert rollout_tools in export_sys, (
        "export-rendered tools section diverged from rollout — same-source broken"
    )


def test_export_response_surface_comes_from_extra_tool_schema(tmp_path):
    """Response availability is stored as a schema, not in valid_actions."""
    extras = [
        *LiteBrowserNavToolSet.get_tool_schemas(include=["goto"]),
        LiteFinishToolSet.get_tool_schema("response"),
    ]
    row = _webgym_row(tmp_path, extra_tool_schemas=extras, valid_actions=_GROUNDED_VALID)
    assert "response" not in row["metadata"]["valid_actions"]
    out = _convert_sample(
        row, agent_id="qwen3_vl", agent_kwargs={}, model_id=_MODEL_ID,
        image_root=None, strict=True,
    )
    sys_text = _system_text(out["steps"])
    assert "click" in sys_text
    assert "goto" in sys_text
    parsed = json.loads(out["metadata"]) if isinstance(out["metadata"], str) else out["metadata"]
    assert [tool_schema_name(s) for s in parsed["extra_tool_schemas"]] == [
        "goto", "response",
    ]


def test_export_preserves_browsergym_goal_images_after_history_windowing(
    tmp_path,
    monkeypatch,
):
    """Offline export must apply the same goal-image splice as live rollout."""
    monkeypatch.setattr(
        "lite.train.export.export_sft._get_processor",
        lambda model_id: _FakeProcessor(),
    )

    image_paths = []
    for i in range(7):
        img_path = tmp_path / f"frame{i}.png"
        Image.new("RGB", (1280, 768), (20 + i, 20, 20)).save(img_path)
        image_paths.append(str(img_path))

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "index": 1},
            {"type": "image", "index": 2},
            {"type": "image", "index": 0},
            {"type": "text", "text": "Find this product."},
            {"type": "metadata", "data": {"goal_image_indices": [1, 2]}},
        ],
    }]
    for turn, image_index in enumerate((3, 4, 5, 6)):
        call_id = f"call_{turn:04d}"
        messages.append({
            "role": "assistant",
            "content": [{"type": "action_description", "text": f"click {turn}"}],
            "tool_calls": [make_tool_call(
                "computer",
                {
                    "actions": [{"action": "click", "coordinate": [100, 200]}],
                },
                call_id=call_id,
            )],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": [{"type": "image", "index": image_index}],
        })
    messages.append({"role": "assistant", "content": [{"type": "text", "text": "Done."}]})

    row = {
        "images": image_paths,
        "messages": messages,
        "metadata": _cua_metadata_dict(
            platform="browser",
            valid_actions=["click"],
            others={"env_id": "browsergym.visualwebarena"},
        ),
    }

    out = _convert_sample(
        row,
        agent_id="qwen3_vl",
        agent_kwargs={
            "protocol_key": "browsergym.goal_image.qwen3_vl.history",
            "protocol_kwargs": {"full_history_size": 1},
        },
        model_id=_MODEL_ID,
        image_root=None,
        strict=True,
    )

    final_step = out["steps"][-1]
    assert final_step["image_indices"] == [1, 2, 6]
    assert final_step["prompt"].count("<|image_pad|>") == 3
    assert final_step["prompt"].split("<|image_pad|>", 1)[0].rstrip().endswith(
        "Task reference image(s) for the instruction below:"
    )

    out_multi = _convert_sample(
        row,
        agent_id="qwen3_vl",
        agent_kwargs={
            "protocol_key": "browsergym.goal_image.qwen3_vl.history",
            "protocol_kwargs": {"full_history_size": 2},
        },
        model_id=_MODEL_ID,
        image_root=None,
        strict=True,
    )
    final_multi = out_multi["steps"][-1]
    assert final_multi["image_indices"] == [1, 2, 5, 6]
    segments = final_multi["prompt"].split("<|image_pad|>")
    assert len(segments) == 5
    assert segments[0].rstrip().endswith(
        "Task reference image(s) for the instruction below:"
    )
    assert "Current screenshot:" not in segments[2]
    assert "Current screenshot:" in segments[3]
    assert "Instruction: Find this product." in final_multi["prompt"]
