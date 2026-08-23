from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset, Value
from PIL import Image

import lite.train.export.export_sft as export_sft
from lite.agents.core.adapter import AsIsAdapter
from lite.core import LiteCUAMetadata
from lite.train.export.export_sft import (
    _adapter_kwargs_for_export,
    _convert_sample,
    _output_features,
    _write_model_ready_parquet,
)
from lite.utils.config import load_config
from lite.utils.image import decode_hf_images


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    img = Image.new("RGB", (2, 2), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _cua_metadata_dict(*, others: dict | None = None) -> dict:
    return LiteCUAMetadata(
        dims=("desktop", "use"),
        extra_tool_schemas=[],
        valid_actions=None,
        others=others or {},
    ).to_dict()


def _cua_metadata_json(*, others: dict | None = None) -> str:
    return json.dumps(_cua_metadata_dict(others=others))


def _model_ready_dataset(rows: list[dict]) -> Dataset:
    """map ``rows`` through the explicit output features, mirroring main()."""
    source = Dataset.from_list(rows)
    return source.map(
        lambda row: {
            "_error": row["error"],
            "processed_images": [b"image"] if not row["error"] else [],
            "steps": [{
                "prompt": "prompt",
                "image_indices": [0],
                "response": "response",
                "response_tokens": [1],
                "reward": 0.0,
                "status": "completed",
                "prompt_tokens": None,
            }] if not row["error"] else [],
            "metadata": row["metadata"],
        },
        remove_columns=source.column_names,
        features=_output_features(source.features["metadata"]),
        writer_batch_size=1,
    )


class _TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [1] * len(text.split())


class _TinyProcessor:
    image_token = "<|image_pad|>"
    tokenizer = _TinyTokenizer()

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    ):
        del tokenize, enable_thinking
        chunks: list[str] = []
        for message in messages:
            chunks.append(f"<{message.get('role')}>")
            for part in message.get("content") or []:
                if part.get("type") == "image":
                    chunks.append("<|image_pad|>")
                elif part.get("type") == "text" and part.get("text"):
                    chunks.append(part["text"])
        if add_generation_prompt:
            chunks.append("<assistant>")
        return "\n".join(chunks)


def test_export_adapter_kwargs_drop_only_rollout_sampling_kwargs() -> None:
    source = {
        "sampling_kwargs": {"temperature": 0.0},
        "resolution": [1280, 720],
        "unknown_adapter_kwarg": True,
    }

    assert _adapter_kwargs_for_export(source) == {
        "resolution": [1280, 720],
        "unknown_adapter_kwarg": True,
    }
    assert "sampling_kwargs" in source


def test_current_qwen_rollout_configs_do_not_forward_sampling_kwargs() -> None:
    for config_path in (
        "scripts/configs/qwen3_vl/default/osworld.yaml",
        "scripts/configs/qwen3_5/default/osworld.yaml",
    ):
        agent_kwargs = load_config(config_path)["agent_kwargs"]

        assert "sampling_kwargs" in agent_kwargs
        assert "sampling_kwargs" not in _adapter_kwargs_for_export(agent_kwargs)


def test_convert_sample_does_not_forward_sampling_kwargs_to_adapter(
    tmp_path,
    monkeypatch,
) -> None:
    img_path = tmp_path / "frame0.png"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(img_path)
    row = {
        "images": [str(img_path)],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "Finish the task."},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        ],
        "metadata": _cua_metadata_dict(others={"env_id": "osworld"}),
    }
    captured: dict = {}

    def fake_get(adapter_key, **kwargs):
        captured["adapter_key"] = adapter_key
        captured["kwargs"] = kwargs

        def unroll(_sample):
            return SimpleNamespace(processed_images=[], steps=[[
                {"role": "user", "content": [{"type": "text", "text": "obs"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "act"}]},
            ]])

        return SimpleNamespace(enable_thinking=False, unroll=unroll)

    monkeypatch.setattr(export_sft, "register_all", lambda: None)
    monkeypatch.setattr(export_sft.AgentAdapterRegistry, "get", staticmethod(fake_get))
    monkeypatch.setattr(export_sft, "_get_processor", lambda _model_id: _TinyProcessor())

    out = _convert_sample(
        row,
        agent_id="qwen3_vl",
        agent_kwargs={
            "sampling_kwargs": {"temperature": 0.0},
            "resolution": [1280, 720],
        },
        model_id="local-model",
        image_root=None,
        strict=True,
    )

    assert out["_error"] == ""
    assert captured["adapter_key"] == "qwen3_vl@desktop@use"
    assert "sampling_kwargs" not in captured["kwargs"]
    assert captured["kwargs"]["resolution"] == [1280, 720]
    assert captured["kwargs"]["metadata"].others["env_id"] == "osworld"


def test_convert_sample_loads_local_path_carrier_before_adapter_unroll(
    tmp_path,
    monkeypatch,
) -> None:
    img_path = tmp_path / "frame0.png"
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(img_path)
    row = {
        "images": [str(img_path)],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "Finish the task."},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        ],
        "metadata": _cua_metadata_dict(others={"env_id": "osworld"}),
    }
    adapter = AsIsAdapter(resolution=(4, 3))
    original_unroll = adapter.unroll
    captured: dict = {}

    def capturing_unroll(sample):
        captured["sample_images"] = list(sample.images)
        return original_unroll(sample)

    def fake_get(adapter_key, **kwargs):
        captured["adapter_key"] = adapter_key
        captured["kwargs"] = kwargs
        adapter.metadata = kwargs["metadata"]
        return adapter

    adapter.unroll = capturing_unroll
    monkeypatch.setattr(export_sft, "register_all", lambda: None)
    monkeypatch.setattr(export_sft.AgentAdapterRegistry, "get", staticmethod(fake_get))
    monkeypatch.setattr(export_sft, "_get_processor", lambda _model_id: _TinyProcessor())

    out = _convert_sample(
        row,
        agent_id="as_is",
        agent_kwargs={},
        model_id="local-model",
        image_root=None,
        strict=True,
    )

    assert out["_error"] == ""
    assert captured["adapter_key"] == "as_is@desktop@use"
    assert isinstance(captured["sample_images"][0], Image.Image)
    assert captured["sample_images"][0].size == (8, 6)
    with Image.open(io.BytesIO(out["processed_images"][0])) as processed:
        assert processed.size == (4, 3)
        assert processed.getpixel((0, 0)) == (10, 20, 30)
    assert out["steps"][0]["image_indices"] == [0]


def _convert_two_step_row(tmp_path, monkeypatch, others: dict) -> list[dict]:
    """Run ``_convert_sample`` over a fake two-step unroll; return its steps."""
    img_path = tmp_path / "frame0.png"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(img_path)
    row = {
        "images": [str(img_path)],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "Finish the task."},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        ],
        "metadata": _cua_metadata_dict(others=others),
    }
    unrolled = [
        [
            {"role": "user", "content": [{"type": "text", "text": "obs 1"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "act 1"}]},
        ],
        [
            {"role": "user", "content": [{"type": "text", "text": "obs 2"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "act 2"}]},
        ],
    ]

    def fake_get(_adapter_key, **_kwargs):
        return SimpleNamespace(
            enable_thinking=False,
            unroll=lambda _sample: SimpleNamespace(
                processed_images=[], steps=unrolled),
        )

    monkeypatch.setattr(export_sft, "register_all", lambda: None)
    monkeypatch.setattr(export_sft.AgentAdapterRegistry, "get", staticmethod(fake_get))
    monkeypatch.setattr(export_sft, "_get_processor", lambda _model_id: _TinyProcessor())

    out = _convert_sample(
        row,
        agent_id="qwen3_vl",
        agent_kwargs={},
        model_id="local-model",
        image_root=None,
        strict=True,
    )
    assert out["_error"] == ""
    return out["steps"]


def test_convert_sample_truncates_last_step_of_a_truncated_episode(
    tmp_path,
    monkeypatch,
) -> None:
    """The row's own env feedback decides step status. Exporting a truncated
    episode as all-``completed`` would train it as complete and make the
    TRUNCATED branch in grpo/reinforce/dagger unreachable for SFT-sourced rows.
    """
    steps = _convert_two_step_row(
        tmp_path, monkeypatch, {"env_id": "osworld", "truncated": True})

    assert [step["status"] for step in steps] == ["completed", "truncated"]


@pytest.mark.parametrize("others", [
    {"env_id": "osworld", "truncated": False},
    {"env_id": "osworld"},  # offline demonstration row: no env feedback at all
])
def test_convert_sample_keeps_completed_without_recorded_truncation(
    tmp_path,
    monkeypatch,
    others,
) -> None:
    steps = _convert_two_step_row(tmp_path, monkeypatch, others)

    assert [step["status"] for step in steps] == ["completed", "completed"]


def test_export_shares_the_truncated_step_projection_owner() -> None:
    """Offline export applies the loops' projection instead of re-stamping it.

    ``metadata.others.truncated`` is offline export's own trigger, but the
    "truncated episode -> trailing step" stamp itself has one owner in
    ``lite.agents.core.agent.utils.loop``.
    """
    import inspect

    from lite.agents.core.agent.utils.loop import mark_steps_truncated

    assert export_sft.mark_steps_truncated is mark_steps_truncated
    assert "STATUS_TRUNCATED" not in inspect.getsource(export_sft._convert_sample)


def test_convert_sample_fails_row_that_has_no_supervisable_step(
    tmp_path,
    monkeypatch,
) -> None:
    """A row whose ONLY step is a partial turn has no assistant target anywhere.

    Skipping it would leave ``steps == []`` — an image-carrying parquet row whose
    only downstream effect is the zero-gradient dummy branch in
    ``lite.train.rollout.sft``. It goes through export's row-failure contract
    instead: raise under ``--strict``, ``_error`` + drop under ``--no-strict``.
    """
    img_path = tmp_path / "frame0.png"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(img_path)
    row = {
        "images": [str(img_path)],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "Finish the task."},
            ],
        }],
        "metadata": _cua_metadata_dict(others={"env_id": "osworld"}),
    }

    def fake_get(_adapter_key, **_kwargs):
        return SimpleNamespace(
            enable_thinking=False,
            unroll=lambda _sample: SimpleNamespace(
                processed_images=[],
                steps=[[{"role": "user", "content": [{"type": "text", "text": "prompt"}]}]],
            ),
        )

    monkeypatch.setattr(export_sft, "register_all", lambda: None)
    monkeypatch.setattr(export_sft.AgentAdapterRegistry, "get", staticmethod(fake_get))
    monkeypatch.setattr(export_sft, "_get_processor", lambda _model_id: object())

    with pytest.raises(ValueError, match="no supervisable SFT step"):
        _convert_sample(
            row,
            agent_id="qwen3_vl",
            agent_kwargs={},
            model_id="local-model",
            image_root=None,
            strict=True,
        )

    out = _convert_sample(
        row,
        agent_id="qwen3_vl",
        agent_kwargs={},
        model_id="local-model",
        image_root=None,
        strict=False,
    )
    assert "no supervisable SFT step" in out["_error"]
    assert out["steps"] == []


def _convert_row_with_partial_turn_at(
    tmp_path,
    monkeypatch,
    *,
    partial_index: int,
    strict: bool = True,
) -> dict:
    """Convert a two-step row whose step ``partial_index`` has no assistant target.

    ``agent_step_to_rl_step`` is faked so the test pins export's step-position
    policy, not the tokenizer boundary (covered in ``test_sft_tokenize``).
    """
    img_path = tmp_path / "frame0.png"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(img_path)
    row = {
        "images": [str(img_path)],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "Finish the task."},
            ],
        }],
        "metadata": _cua_metadata_dict(others={"env_id": "osworld"}),
    }
    complete_turn = [
        {"role": "user", "content": [{"type": "text", "text": "prompt"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "target"}]},
    ]
    partial_turn = [
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": [{"type": "text", "text": "obs"}],
        }
    ]
    steps = [complete_turn, complete_turn]
    steps[partial_index] = partial_turn

    def fake_get(_adapter_key, **_kwargs):
        return SimpleNamespace(
            enable_thinking=False,
            unroll=lambda _sample: SimpleNamespace(
                processed_images=[], steps=steps),
        )

    def fake_agent_step_to_rl_step(step, _processor, _enable_thinking):
        if step is partial_turn:
            return None
        return SimpleNamespace(
            prompt="prompt",
            image_indices=[],
            response="target",
            response_tokens=[1],
            reward=0.0,
            status="completed",
            prompt_tokens=[2],
        )

    monkeypatch.setattr(export_sft, "register_all", lambda: None)
    monkeypatch.setattr(export_sft.AgentAdapterRegistry, "get", staticmethod(fake_get))
    monkeypatch.setattr(export_sft, "_get_processor", lambda _model_id: object())
    monkeypatch.setattr(export_sft, "agent_step_to_rl_step", fake_agent_step_to_rl_step)

    return _convert_sample(
        row,
        agent_id="qwen3_vl",
        agent_kwargs={},
        model_id="local-model",
        image_root=None,
        strict=strict,
    )


def test_convert_sample_keeps_row_whose_last_step_is_a_terminal_observation(
    tmp_path,
    monkeypatch,
) -> None:
    """The normal rollout tail must not cost the row its supervisable steps.

    A terminal turn persists its tool feedback AFTER the assistant action
    (``lite/agents/core/agent/base.py``), so a saved trajectory commonly ends on
    a ``role:"tool"`` observation. ``unroll`` emits that as a trailing partial
    turn with no assistant target — nothing to supervise, not corruption. Failing
    the row here would discard every complete turn before it.
    """
    out = _convert_row_with_partial_turn_at(
        tmp_path, monkeypatch, partial_index=1)

    assert out["_error"] == ""
    assert [step["response"] for step in out["steps"]] == ["target"]


def test_convert_sample_skips_a_mid_trajectory_partial_turn_too(
    tmp_path,
    monkeypatch,
) -> None:
    """Position does not change the verdict: a partial turn is dropped, not fatal.

    ``turn_spans`` can also emit a non-trailing ``assistant=None`` span (a system
    message landing mid-trajectory, ``lite/core/messages/turns.py``), which is
    genuinely anomalous — but that shape appears in 0 of 26,711 real trajectory
    rows on this host, so export carries no unproven guard for it. What stays
    loud is the row-level failure when NOTHING is left to supervise.
    """
    out = _convert_row_with_partial_turn_at(
        tmp_path, monkeypatch, partial_index=0)

    assert out["_error"] == ""
    assert [step["response"] for step in out["steps"]] == ["target"]


def test_output_features_use_large_offsets_for_images() -> None:
    schema = _output_features(Value("string")).arrow_schema
    images = schema.field("processed_images").type

    assert pa.types.is_large_list(images)
    assert pa.types.is_large_binary(images.value_type)


def test_map_to_parquet_preserves_large_image_offsets(tmp_path) -> None:
    metadata = _cua_metadata_json(others={"task_id": "example"})
    source = Dataset.from_list([{
        "source": 1,
        "metadata": metadata,
    }])
    dataset = source.map(
        lambda row: {
            "_error": "",
            "processed_images": [b"image"],
            "steps": [{
                "prompt": "prompt",
                "image_indices": [0],
                "response": "response",
                "response_tokens": [1],
                "reward": 0.0,
                "status": "completed",
                "prompt_tokens": None,
            }],
            "metadata": row["metadata"],
        },
        remove_columns=source.column_names,
        features=_output_features(source.features["metadata"]),
        writer_batch_size=1,
    )

    path = tmp_path / "sft.parquet"
    pq.write_table(dataset.data.table, path)
    images = pq.ParquetFile(path).schema_arrow.field("processed_images").type

    assert pa.types.is_large_list(images)
    assert pa.types.is_large_binary(images.value_type)
    assert dataset[0]["metadata"] == metadata


def test_sparse_processed_images_preserve_null_slots(tmp_path) -> None:
    """SFT rows keep trajectory image indices even when some slots are unprocessed."""
    metadata = _cua_metadata_json(others={"task_id": "example"})
    source = Dataset.from_list([{
        "source": 1,
        "metadata": metadata,
    }])
    dataset = source.map(
        lambda row: {
            "_error": "",
            "processed_images": [b"first", None, b"third"],
            "steps": [{
                "prompt": "prompt",
                "image_indices": [2],
                "response": "response",
                "response_tokens": [1],
                "reward": 0.0,
                "status": "completed",
                "prompt_tokens": None,
            }],
            "metadata": row["metadata"],
        },
        remove_columns=source.column_names,
        features=_output_features(source.features["metadata"]),
        writer_batch_size=1,
    )

    path = tmp_path / "sft.parquet"
    _write_model_ready_parquet(
        dataset.remove_columns("_error"),
        path,
        row_group_size=None,
        writer_batch_size=1,
    )

    table = pq.read_table(path)
    images = table.schema.field("processed_images").type
    assert pa.types.is_large_list(images)
    assert pa.types.is_large_binary(images.value_type)
    assert table["processed_images"][0].as_py() == [b"first", None, b"third"]
    assert table["steps"][0].as_py()[0]["image_indices"] == [2]


def test_decode_hf_images_preserves_sparse_null_slots() -> None:
    images = decode_hf_images([
        _png_bytes((10, 20, 30)),
        None,
        {"bytes": _png_bytes((40, 50, 60))},
    ])

    assert images[1] is None
    assert images[0].getpixel((0, 0)) == (10, 20, 30)
    assert images[2].getpixel((0, 0)) == (40, 50, 60)


def test_write_respects_lazy_filter_selection(tmp_path) -> None:
    """--no-strict drops error rows via a LAZY ``filter`` (an indices mapping);
    the writer must not emit the dropped rows from the unfiltered table."""
    metadata = _cua_metadata_json()
    dataset = _model_ready_dataset([
        {"error": "", "metadata": metadata},
        {"error": "boom", "metadata": metadata},
        {"error": "", "metadata": metadata},
    ])
    filtered = dataset.filter(
        lambda row: not row["_error"], writer_batch_size=1,
    ).remove_columns("_error")
    # Premise of the regression: the selection is lazy — the underlying table
    # still holds every row. If a datasets upgrade makes filter eager, this
    # assert (not the writer) is what should fail, loudly.
    assert filtered.data.table.num_rows == 3

    path = tmp_path / "sft.parquet"
    _write_model_ready_parquet(
        filtered, path, row_group_size=None, writer_batch_size=1)

    pf = pq.ParquetFile(path)
    assert pf.metadata.num_rows == 2 == len(filtered)
    images = pf.schema_arrow.field("processed_images").type
    assert pa.types.is_large_list(images)
    assert pa.types.is_large_binary(images.value_type)


def test_write_without_selection_is_passthrough(tmp_path) -> None:
    metadata = _cua_metadata_json()
    dataset = _model_ready_dataset(
        [{"error": "", "metadata": metadata}]
    ).remove_columns("_error")

    path = tmp_path / "sft.parquet"
    _write_model_ready_parquet(
        dataset, path, row_group_size=None, writer_batch_size=1)

    pf = pq.ParquetFile(path)
    assert pf.metadata.num_rows == 1
    assert pa.types.is_large_list(pf.schema_arrow.field("processed_images").type)
