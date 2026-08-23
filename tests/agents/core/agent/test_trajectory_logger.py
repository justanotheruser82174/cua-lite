"""Unit tests for TrajectoryLogger's persisted metadata contract.

Pins the identity + provenance semantics from the metadata redesign (metadata contract):
  * identity (``env_id``/``task_id``) lives under ``metadata.others``. The
    rollout spec wins when truthy; a falsy spec falls through to live metadata;
  * run provenance (``model_id``/``agent_id``/``config_path``/``commit``/
    ``command``) is spread FLAT into others;
  * outcome keys (``episode_return``/``terminated``/``truncated``) also live
    under ``metadata.others``;
  * ``provenance=None`` adds no extra keys.

Run: uv run pytest tests/agents/core/agent/test_trajectory_logger.py -v
"""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

import lite.agents.core.agent.logger as logger_module
from lite.agents.core.agent import AdapterBasedAgent
from lite.agents.core.agent.hooks import SampleStepData
from lite.agents.core.agent.logger import TrajectoryLogger
from lite.agents.types import PredictResult
from lite.core import LiteBaseMetadata, LiteCUAMetadata, LiteGenericMetadata, LiteSample
from lite.core.errors import LiteContractError
from lite.core.samples import LiteRLSample, LiteRLStep
from lite.core.tools.calls import make_tool_call, tool_call_id, tool_call_name
from lite.core.tools.extra_tools import LiteFinishToolSet
from lite.core.tools.results import LiteToolResult
from lite.core.tools.schemas import make_tool_schema, tool_schema_name
from lite.data.staging import coerce_messages, coerce_meta
from lite.gym.types import LiteEnvStepResult

PROVENANCE = {
    "model_id": "gpt-5.5",
    "agent_id": "gpt.teacher",
    "config_path": "scripts/configs/x.yaml",
    "commit": "abc1234",
    "command": "uv run python scripts/rollout.py --model-id gpt-5.5",
}


class _PassthroughImageAdapter:
    def process_image(self, image: Image.Image) -> Image.Image:
        return image

    def select_action_batch_image_indices(
        self, *, tool_call, tool_result, result_image_indices
    ):
        # Mirrors BaseAgentAdapter's default (final frame only). Spelled out
        # because this fake deliberately does not subclass the real adapter, so
        # the default stays pinned here even if the base one changes.
        del tool_call, tool_result
        return result_image_indices[-1:]


async def _unused_generate_fn(**_kwargs):
    raise AssertionError("generate_fn is not used by these append-only tests")


def _empty_lite_sample() -> LiteSample:
    return LiteSample(
        metadata=LiteCUAMetadata(dims=("desktop", "use"))
    )


def _sample(
    others: dict,
    messages: list[dict] | None = None,
    images: list[Image.Image] | None = None,
    extra_tool_schemas: list[dict] | None = None,
    metadata_fields: dict | None = None,
    metadata: LiteBaseMetadata | None = None,
) -> LiteRLSample:
    if metadata is None:
        metadata = LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=extra_tool_schemas or [],
            others={**others, **(metadata_fields or {})},
        )
    return LiteRLSample(
        lite_sample=LiteSample(
            metadata=metadata,
            images=images or [],
            messages=messages or [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        ),
        processed_images=list(images or []),
        steps=[],
        episode_return=1.0,
        terminated=True,
        truncated=False,
    )


def _persisted_metadata(log_dir) -> dict:
    row = pd.read_parquet(log_dir / "trajectory.parquet").iloc[0]
    return coerce_meta(row["metadata"])


def _persisted_others(log_dir) -> dict:
    others = dict(_persisted_metadata(log_dir)["others"])
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in others.items()}


def _persisted_messages(log_dir) -> list[dict]:
    row = pd.read_parquet(log_dir / "trajectory.parquet").iloc[0]
    return coerce_messages(row["messages"])


def _persisted_images(log_dir) -> list[str]:
    row = pd.read_parquet(log_dir / "trajectory.parquet").iloc[0]
    return list(row["images"])


def _plain(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items() if v is not None}
    return value


def _png_bytes(color: str = "green") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (100, 80), color).save(buf, format="PNG")
    return buf.getvalue()


def _gif_frame_pixels(
    path: Path,
    xy: tuple[int, int] = (1, 1),
) -> list[tuple[int, int, int]]:
    gif = Image.open(path)
    pixels: list[tuple[int, int, int]] = []
    try:
        while True:
            pixels.append(gif.convert("RGB").getpixel(xy))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return pixels


def _single_image_messages(instruction: str = "hi") -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": instruction},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
    ]


def _normalize_persisted_message(msg: dict) -> dict:
    out = {}
    for key, value in dict(msg).items():
        if value is None:
            continue
        value = _plain(value)
        if key == "tool_calls" and value == []:
            continue
        out[key] = value
    return out


def _run(tmp_path, others: dict, **logger_kwargs) -> dict:
    metadata_fields = logger_kwargs.pop("metadata_fields", None)
    traj_logger = TrajectoryLogger(tmp_path / "run", **logger_kwargs)
    traj_logger.on_complete(_sample(others, metadata_fields=metadata_fields))
    return _persisted_metadata(tmp_path / "run")


_TERMINAL_NAMES = LiteFinishToolSet.get_tool_names() | {"report_infeasible"}


def _call_id(call: dict) -> str | None:
    return tool_call_id(call)


def _call_name(call: dict) -> str:
    return tool_call_name(call)


def test_trajectory_logger_imports_core_action_space_for_pure_inspection():
    source = Path(logger_module.__file__).read_text(encoding="utf-8")

    assert "lite.gym.utils.actions" not in source
    assert "lite.gym.utils.backend.coordinate" not in source
    assert "action_inspection_records" in source
    assert "from lite.core.tools.action_space.geometry import" in source
    assert "strict_norm_to_pixel" in source
    # The annotation owner decides which argument carries the point and whether
    # it is drawable; the logger only projects the resolved coordinate.
    assert "annotation_coordinate" in source
    assert "COORDINATE_ARGUMENT_NAMES" not in source


def _step_data(
    actions: list[dict],
    *,
    step_idx: int = 0,
    image_indices: tuple[int, ...] = (0,),
    current_image_index: int | None = None,
    screenshot: Image.Image | None = None,
    lite_message: dict | None = None,
    results: list[LiteToolResult] | None = None,
    info: dict | None = None,
) -> SampleStepData:
    if lite_message is None:
        lite_message = {"role": "assistant", "content": []}
        if actions:
            lite_message["tool_calls"] = actions
    if results is None:
        results = [
            LiteToolResult(tool_call_id=_call_id(action), text=f"result {_call_id(action)}")
            for action in actions
            if _call_name(action) not in _TERMINAL_NAMES and _call_id(action)
        ]
    return SampleStepData(
        step_idx=step_idx,
        image=screenshot or Image.new("RGB", (100, 80), "white"),
        predict_result=PredictResult(
            lite_message=lite_message,
            agent_message={"role": "assistant", "content": "agent"},
            step=LiteRLStep(
                prompt="prompt text",
                image_indices=image_indices,
                response="raw response",
            ),
        ),
        step_result=LiteEnvStepResult(
            reward=0.25,
            info=info or {
                "executed_actions": [
                    {"name": "click", "arguments": {"coordinate": [50, 20]}}
                ]
            },
            results=results,
        ),
        actions=actions,
        current_image_index=current_image_index,
    )


def test_p2a_truthy_spec_writes_identity_provenance_outcome(tmp_path):
    env_others = {"domain": "calc", "env_id": "lite.osworld", "task_id": "t1"}
    md = _run(
        tmp_path, env_others,
        env_id="lite.osworld", task_id="t1", provenance=PROVENANCE,
    )
    others = md["others"]
    assert others["domain"] == "calc"                      # env task attrs pass through
    assert others["env_id"] == "lite.osworld"
    assert others["task_id"] == "t1"
    assert "env_id" not in md and "task_id" not in md
    for key, value in PROVENANCE.items():                  # provenance keys FLAT
        assert others[key] == value
    assert others["episode_return"] == 1.0
    assert others["terminated"] is True
    assert others["truncated"] is False
    assert not {"episode_return", "terminated", "truncated"} & set(md)


def test_p2a_truthy_spec_overrides_env_authored_identity(tmp_path):
    """The spec's key-derived value wins over an env-authored old ``others`` form."""
    md = _run(
        tmp_path, {"task_id": "env-authored-form"},
        env_id="e1", task_id="key-form",
    )
    assert md["others"]["task_id"] == "key-form"
    assert md["others"]["env_id"] == "e1"
    assert "task_id" not in md


def test_provenance_identity_keys_do_not_override_spec_or_live_identity(tmp_path):
    provenance = {
        **PROVENANCE,
        "env_id": "provenance-env",
        "task_id": "provenance-task",
    }

    spec_md = _run(
        tmp_path / "spec",
        {"env_id": "live-env", "task_id": "live-task"},
        env_id="spec-env",
        task_id="spec-task",
        provenance=provenance,
    )
    assert spec_md["others"]["env_id"] == "spec-env"
    assert spec_md["others"]["task_id"] == "spec-task"
    assert spec_md["others"]["model_id"] == PROVENANCE["model_id"]

    live_md = _run(
        tmp_path / "live",
        {},
        env_id="",
        task_id="",
        provenance=provenance,
        metadata_fields={"env_id": "live-env", "task_id": "live-task"},
    )
    assert live_md["others"]["env_id"] == "live-env"
    assert live_md["others"]["task_id"] == "live-task"
    assert live_md["others"]["agent_id"] == PROVENANCE["agent_id"]


def test_p2b_falsy_spec_falls_through_to_live_identity(tmp_path):
    """env_id=""/task_id="" skip the spec write (logger guard) — the row keeps
    whatever identity the live metadata carried."""
    md = _run(
        tmp_path,
        {},
        env_id="", task_id="",
        metadata_fields={"env_id": "live-env", "task_id": "live-task"},
    )
    assert md["others"]["env_id"] == "live-env"
    assert md["others"]["task_id"] == "live-task"


def test_p2c_window_dependency_spec_is_sole_identity_source(tmp_path):
    """A spec-carrying logger writes identity even when live metadata has none."""
    md = _run(
        tmp_path, {"domain": "search"},                    # no identity on the env side
        env_id="webgym", task_id="wg_001",
    )
    assert md["others"]["env_id"] == "webgym"
    assert md["others"]["task_id"] == "wg_001"


def test_no_provenance_adds_no_extra_keys(tmp_path):
    env_others = {"domain": "calc", "env_id": "e", "task_id": "t"}
    md = _run(tmp_path, env_others, env_id="e", task_id="t")
    assert md["others"] == {
        "domain": "calc",
        "env_id": "e",
        "task_id": "t",
        "episode_return": 1.0,
        "terminated": True,
        "truncated": False,
    }


def test_render_instruction_banner_default_burns_instruction_into_frame(tmp_path):
    """Default (True): the top-left corner of the frame is overpainted with the
    banner's semi-transparent black background, so it no longer matches the
    plain white screenshot."""
    traj_logger = TrajectoryLogger(tmp_path / "run", save_gif=True)
    traj_logger.on_complete(_sample(
        {},
        messages=_single_image_messages("hi"),
        images=[Image.new("RGB", (200, 100), "white")],
    ))
    frame = Image.open(tmp_path / "run" / "trajectory.gif").convert("RGB")
    assert frame.getpixel((2, 2)) != (255, 255, 255)


def test_render_instruction_banner_false_leaves_frame_untouched(tmp_path):
    """False: no banner is drawn, so the gif frame is pixel-identical to the
    plain screenshot (modulo gif quantization, which is lossless for a flat
    solid-colour frame)."""
    traj_logger = TrajectoryLogger(
        tmp_path / "run", save_gif=True, render_instruction_banner=False,
    )
    traj_logger.on_complete(_sample(
        {},
        messages=_single_image_messages("hi"),
        images=[Image.new("RGB", (200, 100), "white")],
    ))
    frame = Image.open(tmp_path / "run" / "trajectory.gif").convert("RGB")
    assert frame.getpixel((2, 2)) == (255, 255, 255)


def test_gif_has_no_sample_step_image_fallback(tmp_path):
    final_message = {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        final_message,
    ]
    traj_logger = TrajectoryLogger(
        tmp_path / "run",
        save_gif=True,
        render_instruction_banner=False,
    )

    traj_logger.on_step(_step_data(
        [],
        screenshot=Image.new("RGB", (20, 20), "blue"),
        lite_message=final_message,
        results=[],
    ))
    traj_logger.on_complete(_sample({}, messages=messages))

    assert not (tmp_path / "run" / "trajectory.gif").exists()


def test_trajectory_parquet_preserves_role_tool_call_id_messages(tmp_path):
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [
                make_tool_call("computer", {"actions": []}, call_id="call_0000")
            ],
        },
        {"role": "tool", "tool_call_id": "call_0000", "content": [{"type": "text", "text": "ok"}]},
    ]
    traj_logger = TrajectoryLogger(tmp_path / "run")
    traj_logger.on_complete(_sample({}, messages=messages))

    persisted = [
        _normalize_persisted_message(msg)
        for msg in _persisted_messages(tmp_path / "run")
    ]
    assert persisted == messages
    assert persisted[2]["role"] == "tool"
    assert persisted[2]["tool_call_id"] == tool_call_id(persisted[1]["tool_calls"][0])


def test_trajectory_parquet_writes_metadata_as_opaque_json(tmp_path):
    click_schema = make_tool_schema(
        "click",
        parameters={
            "type": "object",
            "properties": {"index": {"type": "integer"}},
            "required": ["index"],
        },
    )
    scroll_schema = make_tool_schema(
        "scroll",
        parameters={
            "type": "object",
            "properties": {
                "down": {"type": "boolean"},
                "pages": {"type": "number"},
            },
            "required": ["down", "pages"],
        },
    )

    traj_logger = TrajectoryLogger(tmp_path / "run")
    traj_logger.on_complete(_sample(
        {},
        extra_tool_schemas=[click_schema, scroll_schema],
    ))

    row = pd.read_parquet(tmp_path / "run" / "trajectory.parquet").iloc[0]
    assert isinstance(row["messages"], str)
    assert isinstance(row["metadata"], str)
    assert coerce_meta(row["metadata"])["extra_tool_schemas"] == [
        click_schema,
        scroll_schema,
    ]


def test_trajectory_parquet_writes_tagged_generic_metadata(tmp_path):
    response_schema = LiteFinishToolSet.get_tool_schema("response")
    metadata = LiteGenericMetadata(
        dims=(),
        extra_tool_schemas=[response_schema],
        others={"env_id": "geo3k", "task_id": "unit_square_diagonal"},
    )

    traj_logger = TrajectoryLogger(
        tmp_path / "run",
        provenance={"agent_id": "qwen3_5.base"},
    )
    traj_logger.on_complete(_sample({}, metadata=metadata))

    persisted = _persisted_metadata(tmp_path / "run")
    assert persisted["metadata_kind"] == "generic"
    assert persisted["dims"] == []
    assert "valid_actions" not in persisted
    assert [tool_schema_name(s) for s in persisted["extra_tool_schemas"]] == [
        "response"
    ]
    assert persisted["others"]["env_id"] == "geo3k"
    assert persisted["others"]["task_id"] == "unit_square_diagonal"
    assert persisted["others"]["agent_id"] == "qwen3_5.base"
    assert persisted["others"]["episode_return"] == 1.0
    assert persisted["others"]["terminated"] is True
    assert persisted["others"]["truncated"] is False


def test_content_only_final_logs_reward_without_synthetic_tool_call(tmp_path):
    assistant = {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        assistant,
    ]
    response_schema = LiteFinishToolSet.get_tool_schema("response")
    assert response_schema is not None

    traj_logger = TrajectoryLogger(tmp_path / "run")
    traj_logger.on_step(_step_data(
        [],
        lite_message=assistant,
        results=[],
        info={"stop_reason": "content_only_final"},
    ))
    traj_logger.on_complete(_sample(
        {},
        messages=messages,
        extra_tool_schemas=[response_schema],
    ))

    actions_json = json.loads((tmp_path / "run" / "turn_0000" / "03_actions.json").read_text())
    results_json = json.loads((tmp_path / "run" / "turn_0000" / "04_results.json").read_text())
    assert actions_json["lite_message"] == assistant
    assert "tool_calls" not in actions_json["lite_message"]
    assert results_json["reward"] == 0.25
    assert results_json["info"]["stop_reason"] == "content_only_final"
    summary_json = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert "stop_reason" not in summary_json
    assert "stop_reason" not in _persisted_others(tmp_path / "run")

    persisted = [
        _normalize_persisted_message(msg)
        for msg in _persisted_messages(tmp_path / "run")
    ]
    assert persisted == messages


def test_trajectory_parquet_persists_only_allowlisted_final_stop_reason_without_info_leak(tmp_path):
    assistant = {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        assistant,
    ]

    traj_logger = TrajectoryLogger(
        tmp_path / "run",
        provenance={"stop_reason": "SECRET provenance"},
    )
    traj_logger.on_step(_step_data(
        [],
        lite_message=assistant,
        results=[],
        info={
            "stop_reason": "parse_failure",
            "executed_actions": [{"call": "response", "args": {"text": "SECRET answer"}}],
            "parser_error": "SECRET parser prose",
        },
    ))
    traj_logger.on_complete(_sample(
        {"stop_reason": "SECRET env metadata"},
        messages=messages,
    ))

    summary_json = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary_json["stop_reason"] == "parse_failure"
    others = _persisted_others(tmp_path / "run")
    assert others["stop_reason"] == "parse_failure"
    assert "executed_actions" not in others
    assert "parser_error" not in others
    assert "SECRET" not in json.dumps(others, sort_keys=True)


@pytest.mark.parametrize("stop_reason", ["content_only_final", "empty", "SECRET arbitrary prose"])
def test_trajectory_parquet_drops_non_allowlisted_stop_reason(tmp_path, stop_reason):
    traj_logger = TrajectoryLogger(tmp_path / "run")
    traj_logger.on_step(_step_data(
        [],
        results=[],
        info={"stop_reason": stop_reason},
    ))
    traj_logger.on_complete(_sample({"stop_reason": "SECRET env metadata"}))

    summary_json = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert "stop_reason" not in summary_json
    others = _persisted_others(tmp_path / "run")
    assert "stop_reason" not in others
    assert "SECRET" not in json.dumps(others, sort_keys=True)


def test_trajectory_parquet_writes_stable_litesample_image_paths(tmp_path):
    messages = [
        {"role": "user", "content": [{"type": "image", "index": 0}]},
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "screenshot"}]},
                    call_id="call_0000",
                )
            ],
        },
        {"role": "tool", "tool_call_id": "call_0000", "content": [{"type": "image", "index": 1}]},
    ]
    images = [
        Image.new("RGB", (3, 3), "red"),
        Image.new("RGB", (3, 3), "green"),
    ]

    traj_logger = TrajectoryLogger(tmp_path / "run")
    traj_logger.on_complete(_sample({}, messages=messages, images=images))

    paths = _persisted_images(tmp_path / "run")
    assert paths == [
        str((tmp_path / "run" / "images" / "000000.png").resolve()),
        str((tmp_path / "run" / "images" / "000001.png").resolve()),
    ]
    assert all(Path(path).is_file() for path in paths)


def test_trajectory_parquet_text_only_tool_results_do_not_shift_image_indices(tmp_path):
    messages = [
        {"role": "user", "content": [{"type": "image", "index": 0}]},
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [
                make_tool_call("bash", {"command": "pwd"}, call_id="call_0000")
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0000",
            "content": [{"type": "text", "text": "stdout"}],
        },
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [10, 20]}]},
                    call_id="call_0001",
                )
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0001",
            "content": [
                {"type": "text", "text": "invalid action arguments: bad coordinate"},
                {"type": "metadata", "data": {"is_error": True}},
            ],
        },
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [
                make_tool_call(
                    "mobile",
                    {"actions": [{"action": "tap", "coordinate": [500, 500]}]},
                    call_id="call_0002",
                )
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0002",
            "content": [
                {"type": "image", "index": 1},
                {"type": "text", "text": "invalid action: type; valid_actions=['tap']"},
                {"type": "metadata", "data": {"is_error": True}},
            ],
        },
    ]
    images = [
        Image.new("RGB", (3, 3), "red"),
        Image.new("RGB", (3, 3), "green"),
    ]

    traj_logger = TrajectoryLogger(tmp_path / "run")
    traj_logger.on_complete(_sample({}, messages=messages, images=images))

    paths = _persisted_images(tmp_path / "run")
    assert paths == [
        str((tmp_path / "run" / "images" / "000000.png").resolve()),
        str((tmp_path / "run" / "images" / "000001.png").resolve()),
    ]
    persisted = [
        _normalize_persisted_message(msg)
        for msg in _persisted_messages(tmp_path / "run")
    ]
    assert persisted == messages


def test_trajectory_parquet_keeps_action_batch_internal_images(tmp_path):
    actions = [
        make_tool_call(
            "computer",
            {"actions": [{"action": "click"}, {"action": "type", "text": "x"}]},
            call_id="call_batch",
        )
    ]
    messages = [
        {"role": "user", "content": [{"type": "image", "index": 0}]},
        {"role": "assistant", "content": [], "tool_calls": actions},
        {"role": "tool", "tool_call_id": "call_batch", "content": [{"type": "image", "index": 2}]},
    ]
    images = [
        Image.new("RGB", (3, 3), "red"),
        Image.new("RGB", (3, 3), "green"),
        Image.new("RGB", (3, 3), "blue"),
    ]
    sample = LiteRLSample(
        lite_sample=LiteSample(
            metadata=LiteCUAMetadata(dims=("desktop", "use")),
            images=images,
            messages=messages,
        ),
        processed_images=[images[0], None, images[2]],
        steps=[
            LiteRLStep(
                prompt="",
                image_indices=(0, 2),
                response="",
            )
        ],
    )

    TrajectoryLogger(tmp_path / "run").on_complete(sample)

    assert len(_persisted_images(tmp_path / "run")) == 3
    assert [
        _normalize_persisted_message(msg)
        for msg in _persisted_messages(tmp_path / "run")
    ] == messages
    assert sample.steps[0].image_indices == (0, 2)


@pytest.mark.asyncio
async def test_adapter_action_batch_result_exposes_only_last_image_to_model():
    agent = AdapterBasedAgent(
        generate_fn=_unused_generate_fn,
        adapter=_PassthroughImageAdapter(),
    )
    lite_sample = _empty_lite_sample()
    lite_rl_sample = LiteRLSample(lite_sample=lite_sample)
    action_batch = make_tool_call(
        "computer",
        {"actions": [{"action": "click"}, {"action": "type", "text": "x"}]},
        call_id="call_batch",
    )

    hook_image, hook_image_index = await agent._append_tool_result_messages(
        lite_sample,
        lite_rl_sample,
        [
            LiteToolResult(
                tool_call_id="call_batch",
                images=[_png_bytes("red"), _png_bytes("blue"), _png_bytes("green")],
                text="ok",
            )
        ],
        tool_calls=[action_batch],
    )

    assert len(lite_sample.images) == 3
    assert len(lite_rl_sample.processed_images) == 3
    assert lite_rl_sample.processed_images[0] is None
    assert lite_rl_sample.processed_images[1] is None
    assert lite_rl_sample.processed_images[2] is not None
    assert hook_image_index == 2
    assert hook_image is not None
    assert hook_image.getpixel((1, 1)) == (0, 128, 0)
    assert lite_sample.messages == [
        {
            "role": "tool",
            "tool_call_id": "call_batch",
            "content": [
                {"type": "image", "index": 2},
                {"type": "text", "text": "ok"},
            ],
        }
    ]


@pytest.mark.asyncio
async def test_adapter_non_action_batch_result_keeps_all_model_visible_images():
    agent = AdapterBasedAgent(
        generate_fn=_unused_generate_fn,
        adapter=_PassthroughImageAdapter(),
    )
    lite_sample = _empty_lite_sample()
    lite_rl_sample = LiteRLSample(lite_sample=lite_sample)
    tool_call = make_tool_call("inspect", call_id="call_tool")

    hook_image, hook_image_index = await agent._append_tool_result_messages(
        lite_sample,
        lite_rl_sample,
        [
            LiteToolResult(
                tool_call_id="call_tool",
                images=[_png_bytes("red"), _png_bytes("blue")],
            )
        ],
        tool_calls=[tool_call],
    )

    assert len(lite_sample.images) == 2
    assert len(lite_rl_sample.processed_images) == 2
    assert all(image is not None for image in lite_rl_sample.processed_images)
    assert hook_image_index == 0
    assert hook_image is not None
    assert hook_image.getpixel((1, 1)) == (255, 0, 0)
    assert lite_sample.messages == [
        {
            "role": "tool",
            "tool_call_id": "call_tool",
            "content": [
                {"type": "image", "index": 0},
                {"type": "image", "index": 1},
            ],
        }
    ]


@pytest.mark.asyncio
async def test_adapter_single_image_non_action_batch_result_keeps_model_visible_image():
    agent = AdapterBasedAgent(
        generate_fn=_unused_generate_fn,
        adapter=_PassthroughImageAdapter(),
    )
    lite_sample = _empty_lite_sample()
    lite_rl_sample = LiteRLSample(lite_sample=lite_sample)
    tool_call = make_tool_call("inspect", call_id="call_tool")

    hook_image, hook_image_index = await agent._append_tool_result_messages(
        lite_sample,
        lite_rl_sample,
        [
            LiteToolResult(
                tool_call_id="call_tool",
                images=[_png_bytes("yellow")],
            )
        ],
        tool_calls=[tool_call],
    )

    assert len(lite_sample.images) == 1
    assert len(lite_rl_sample.processed_images) == 1
    assert lite_rl_sample.processed_images[0] is not None
    assert hook_image_index == 0
    assert hook_image is not None
    assert hook_image.getpixel((1, 1)) == (255, 255, 0)
    assert lite_sample.messages == [
        {
            "role": "tool",
            "tool_call_id": "call_tool",
            "content": [{"type": "image", "index": 0}],
        }
    ]


def test_trajectory_parquet_rejects_out_of_range_image_refs(tmp_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 1},
                {"type": "text", "text": "missing image"},
            ],
        }
    ]
    images = [Image.new("RGB", (3, 3), "red")]

    traj_logger = TrajectoryLogger(tmp_path / "run")
    with pytest.raises(LiteContractError, match="index 1 out of range for images length 1"):
        traj_logger.on_complete(_sample({}, messages=messages, images=images))

    summary_json = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary_json["n_turns"] == 0
    assert not (tmp_path / "run" / "trajectory.parquet").exists()


def test_on_step_uses_canonical_tool_calls_for_logs_and_artifacts(tmp_path, caplog):
    actions = [
        make_tool_call(
            "click",
            {"coordinate": [500, 250], "button": "left"},
            call_id="call_0000",
        )
    ]
    traj_logger = TrajectoryLogger(
        tmp_path / "run",
        save_video=True,
        debug_artifacts=True,
    )

    with caplog.at_level(logging.INFO, logger="lite.agents.core.agent.logger"):
        traj_logger.on_step(_step_data(actions))

    assert "step 0: actions=['click'] reward=0.25" in caplog.text

    actions_json = json.loads((tmp_path / "run" / "turn_0000" / "03_actions.json").read_text())
    assert actions_json["lite_message"]["tool_calls"] == actions
    assert actions_json["executed_actions"] == [
        {"name": "click", "arguments": {"coordinate": [50, 20]}}
    ]
    results_json = json.loads((tmp_path / "run" / "turn_0000" / "04_results.json").read_text())
    assert results_json["results"] == [
        {
            "tool_call_id": "call_0000",
            "images": [],
            "text": "result call_0000",
            "metadata": None,
            "error": None,
        }
    ]
    image_path = tmp_path / "run" / "turn_0000" / "prompt_images" / "0000_reset.png"
    annotated_path = (
        tmp_path / "run" / "turn_0000" / "prompt_images_annotated" / "0000_reset.png"
    )
    assert image_path.is_file()
    assert annotated_path.stat().st_size > 0


def test_on_step_omits_prompt_image_debug_cache_by_default(tmp_path):
    actions = [
        make_tool_call(
            "click",
            {"coordinate": [500, 250], "button": "left"},
            call_id="call_0000",
        )
    ]
    traj_logger = TrajectoryLogger(tmp_path / "run")

    traj_logger.on_step(_step_data(actions))

    turn_dir = tmp_path / "run" / "turn_0000"
    assert (turn_dir / "01_prompt.txt").is_file()
    assert (turn_dir / "04_results.json").is_file()
    assert not (turn_dir / "prompt_images").exists()
    assert not (turn_dir / "prompt_images_annotated").exists()
    assert not (turn_dir / "images").exists()
    assert not (turn_dir / "annotated").exists()


def test_on_step_omits_prompt_image_debug_cache_when_save_data_is_disabled(tmp_path):
    actions = [
        make_tool_call(
            "click",
            {"coordinate": [500, 250], "button": "left"},
            call_id="call_0000",
        )
    ]
    traj_logger = TrajectoryLogger(
        tmp_path / "run",
        save_data=False,
        debug_artifacts=True,
    )

    traj_logger.on_step(_step_data(actions))

    assert not (tmp_path / "run" / "turn_0000").exists()


def test_on_complete_none_writes_no_resume_gate_summary(tmp_path):
    traj_logger = TrajectoryLogger(tmp_path / "run")

    traj_logger.on_complete(None)

    assert not (tmp_path / "run" / "summary.json").exists()
    assert not (tmp_path / "run" / "trajectory.parquet").exists()


def test_on_complete_save_data_false_writes_summary_only_resume_gate(tmp_path):
    traj_logger = TrajectoryLogger(tmp_path / "run", save_data=False)

    traj_logger.on_complete(_sample({}))

    summary_json = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary_json["n_turns"] == 0
    assert summary_json["episode_return"] == 1.0
    assert summary_json["terminated"] is True
    assert not (tmp_path / "run" / "trajectory.parquet").exists()
    assert not (tmp_path / "run" / "images").exists()


def test_on_complete_media_failure_leaves_resume_gate_summary(tmp_path, monkeypatch):
    def _raise_gif_error(self, frames):
        del self, frames
        raise RuntimeError("gif writer exploded")

    monkeypatch.setattr(TrajectoryLogger, "_save_gif", _raise_gif_error)
    traj_logger = TrajectoryLogger(tmp_path / "run", save_gif=True)

    with pytest.raises(RuntimeError, match="gif writer exploded"):
        traj_logger.on_complete(_sample(
            {},
            messages=_single_image_messages("hi"),
            images=[Image.new("RGB", (20, 20), "white")],
        ))

    summary_json = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary_json["n_turns"] == 0
    assert (tmp_path / "run" / "trajectory.parquet").exists()


def test_on_complete_parquet_failure_leaves_resume_gate_summary(tmp_path, monkeypatch):
    import lite.utils.parquet as parquet_module

    def _raise_parquet_error(*_args, **_kwargs):
        raise RuntimeError("parquet writer exploded")

    monkeypatch.setattr(parquet_module, "write_records_to_parquet", _raise_parquet_error)
    traj_logger = TrajectoryLogger(tmp_path / "run")

    with pytest.raises(RuntimeError, match="parquet writer exploded"):
        traj_logger.on_complete(_sample({}))

    summary_json = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary_json["n_turns"] == 0
    assert summary_json["episode_return"] == 1.0
    assert summary_json["terminated"] is True
    assert not (tmp_path / "run" / "trajectory.parquet").exists()


def test_on_step_sanitizes_debug_json_large_values(tmp_path):
    traj_logger = TrajectoryLogger(tmp_path / "run")
    long_text = "x" * 600

    traj_logger.on_step(_step_data(
        [],
        info={
            "executed_actions": [],
            "image_url": "data:image/png;base64," + ("a" * 600),
            "note": long_text,
            "blob": b"abc",
        },
    ))

    results_json = json.loads((tmp_path / "run" / "turn_0000" / "04_results.json").read_text())
    assert results_json["info"]["image_url"] == "[omitted]"
    assert results_json["info"]["note"] == "x" * 200 + "... [600 chars total]"
    assert results_json["info"]["blob"] == "[omitted 3 bytes]"


def test_on_step_uses_matching_from_call_image_and_annotation_names(tmp_path):
    traj_logger = TrajectoryLogger(tmp_path / "run", debug_artifacts=True)
    traj_logger.on_step(_step_data(
        [make_tool_call("screenshot", call_id="call_0000")],
        results=[
            LiteToolResult(tool_call_id="call_0000", images=[_png_bytes()], text="ok")
        ],
    ))

    actions = [
        make_tool_call(
            "click",
            {"coordinate": [500, 250], "button": "left"},
            call_id="call_0001",
        )
    ]
    traj_logger.on_step(_step_data(actions, step_idx=1, image_indices=(0, 1)))

    image_path = tmp_path / "run" / "turn_0001" / "prompt_images" / "0001_from_call_0000.png"
    annotated_path = (
        tmp_path
        / "run"
        / "turn_0001"
        / "prompt_images_annotated"
        / "0001_from_call_0000.png"
    )
    assert image_path.is_file()
    assert annotated_path.is_file()


def test_on_step_prefers_current_observation_index_over_prompt_order(tmp_path):
    actions = [
        make_tool_call(
            "click",
            {"coordinate": [500, 250], "button": "left"},
            call_id="call_0001",
        )
    ]
    traj_logger = TrajectoryLogger(tmp_path / "run", debug_artifacts=True)

    traj_logger.on_step(
        _step_data(
            actions,
            image_indices=(0, 2, 1),
            current_image_index=2,
        )
    )

    image_path = tmp_path / "run" / "turn_0000" / "prompt_images" / "0002_reset.png"
    annotated_path = (
        tmp_path / "run" / "turn_0000" / "prompt_images_annotated" / "0002_reset.png"
    )
    assert image_path.is_file()
    assert annotated_path.is_file()


def test_on_step_uses_canonical_batch_children_for_logs_and_crosshairs(tmp_path, caplog):
    actions = [
        make_tool_call(
            "computer",
            {
                "actions": [
                    {"action": "click", "coordinate": [500, 250], "button": "left"},
                ],
            },
            call_id="call_0002",
        )
    ]
    traj_logger = TrajectoryLogger(
        tmp_path / "run",
        save_video=True,
        debug_artifacts=True,
    )

    with caplog.at_level(logging.INFO, logger="lite.agents.core.agent.logger"):
        traj_logger.on_step(_step_data(actions))

    assert "step 0: actions=['click'] reward=0.25" in caplog.text

    actions_json = json.loads((tmp_path / "run" / "turn_0000" / "03_actions.json").read_text())
    assert actions_json["lite_message"]["tool_calls"] == actions
    results_json = json.loads((tmp_path / "run" / "turn_0000" / "04_results.json").read_text())
    assert results_json["results"] == [
        {
            "tool_call_id": "call_0002",
            "images": [],
            "text": "result call_0002",
            "metadata": None,
            "error": None,
        }
    ]
    annotated = Image.open(
        tmp_path / "run" / "turn_0000" / "prompt_images_annotated" / "0000_reset.png"
    ).convert("RGB")
    assert annotated.getpixel((50, 20)) == (255, 0, 0)


def test_on_step_annotation_uses_clamped_coordinate_conversion(tmp_path):
    actions = [
        make_tool_call(
            "click",
            {"coordinate": [1000, 500], "button": "left"},
            call_id="call_0003",
        )
    ]
    traj_logger = TrajectoryLogger(tmp_path / "run", debug_artifacts=True)

    traj_logger.on_step(_step_data(actions))

    annotated = Image.open(
        tmp_path / "run" / "turn_0000" / "prompt_images_annotated" / "0000_reset.png"
    ).convert("RGB")
    assert annotated.getpixel((99, 40)) == (255, 0, 0)


def test_on_step_prompt_image_debug_overlay_failures_are_non_fatal(
    tmp_path,
    monkeypatch,
    caplog,
):
    import lite.agents.core.agent.logger as logger_module

    actions = [
        make_tool_call(
            "click",
            {"coordinate": [500, 250], "button": "left"},
            call_id="call_0003",
        )
    ]
    monkeypatch.setattr(
        logger_module,
        "coordinate_annotation_records",
        lambda _actions: [{
            "name": "click",
            "arguments": {"coordinate": ["bad-x", "bad-y"]},
            "result_call_id": "call_0003",
        }],
    )
    traj_logger = TrajectoryLogger(tmp_path / "run", debug_artifacts=True)

    with caplog.at_level(logging.DEBUG, logger="lite.agents.core.agent.logger"):
        traj_logger.on_step(_step_data(actions))

    turn_dir = tmp_path / "run" / "turn_0000"
    assert (turn_dir / "01_prompt.txt").is_file()
    assert (turn_dir / "04_results.json").is_file()
    assert (turn_dir / "prompt_images" / "0000_reset.png").is_file()
    assert not (turn_dir / "prompt_images_annotated" / "0000_reset.png").exists()
    assert "Failed to write prompt image debug overlay" in caplog.text


def test_on_step_logs_canonical_noop_payload(tmp_path):
    actions = [make_tool_call("wait", {"duration": 1}, call_id="call_0001")]
    traj_logger = TrajectoryLogger(tmp_path / "run", save_gif=True)

    traj_logger.on_step(_step_data(actions))

    actions_json = json.loads((tmp_path / "run" / "turn_0000" / "03_actions.json").read_text())
    assert actions_json["lite_message"]["tool_calls"] == actions
    results_json = json.loads((tmp_path / "run" / "turn_0000" / "04_results.json").read_text())
    assert [result["tool_call_id"] for result in results_json["results"]] == ["call_0001"]


def test_on_step_logs_canonical_batch_noop_payload(tmp_path):
    actions = [
        make_tool_call(
            "mobile",
            {
                "actions": [
                    {"action": "screenshot"},
                    {"action": "wait", "duration": 1},
                ],
            },
            call_id="call_0004",
        )
    ]
    traj_logger = TrajectoryLogger(tmp_path / "run", save_gif=True)

    traj_logger.on_step(_step_data(actions))

    actions_json = json.loads((tmp_path / "run" / "turn_0000" / "03_actions.json").read_text())
    assert actions_json["lite_message"]["tool_calls"] == actions
    results_json = json.loads((tmp_path / "run" / "turn_0000" / "04_results.json").read_text())
    assert [result["tool_call_id"] for result in results_json["results"]] == ["call_0004"]


def test_gif_keeps_wait_prompt_image_from_canonical_messages(tmp_path):
    messages = [
        {"role": "user", "content": [{"type": "image", "index": 0}]},
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [
                make_tool_call("wait", {"duration": 1}, call_id="call_0000")
            ],
        },
    ]
    traj_logger = TrajectoryLogger(
        tmp_path / "run",
        save_gif=True,
        render_instruction_banner=False,
    )

    traj_logger.on_complete(_sample(
        {},
        messages=messages,
        images=[Image.new("RGB", (20, 20), "white")],
    ))

    assert _gif_frame_pixels(tmp_path / "run" / "trajectory.gif") == [(255, 255, 255)]


def test_content_only_terminal_logs_without_fake_finish_call(tmp_path):
    final_message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "Done."}],
    }
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        final_message,
    ]
    traj_logger = TrajectoryLogger(tmp_path / "run")

    traj_logger.on_step(_step_data([], lite_message=final_message, results=[]))
    traj_logger.on_complete(_sample({}, messages=messages))

    actions_json = json.loads((tmp_path / "run" / "turn_0000" / "03_actions.json").read_text())
    assert "tool_calls" not in actions_json["lite_message"]
    results_json = json.loads((tmp_path / "run" / "turn_0000" / "04_results.json").read_text())
    assert results_json["results"] == []
    persisted = [
        _normalize_persisted_message(msg)
        for msg in _persisted_messages(tmp_path / "run")
    ]
    assert persisted == messages


def test_on_step_results_log_preserves_native_error_field(tmp_path):
    actions = [make_tool_call("computer", {"actions": []}, call_id="call_0000")]
    traj_logger = TrajectoryLogger(tmp_path / "run")

    traj_logger.on_step(_step_data(
        actions,
        results=[
            LiteToolResult(
                tool_call_id="call_0000",
                text="## AXTree:\nbody",
                error="unsupported action: bogus",
                metadata={"is_error": True},
            )
        ],
    ))

    results_json = json.loads((tmp_path / "run" / "turn_0000" / "04_results.json").read_text())
    assert results_json["results"] == [
        {
            "tool_call_id": "call_0000",
            "images": [],
            "text": "## AXTree:\nbody",
            "metadata": {"is_error": True},
            "error": "unsupported action: bogus",
        }
    ]


def test_trajectory_parquet_does_not_reproject_logged_tool_result_error_text(tmp_path):
    actions = [make_tool_call("computer", {"actions": []}, call_id="call_0000")]
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        {"role": "assistant", "content": [], "tool_calls": actions},
        {
            "role": "tool",
            "tool_call_id": "call_0000",
            "content": [{"type": "text", "text": "## AXTree:\nbody"}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
    ]
    traj_logger = TrajectoryLogger(tmp_path / "run")

    traj_logger.on_step(_step_data(
        actions,
        results=[
            LiteToolResult(
                tool_call_id="call_0000",
                text="## AXTree:\nbody",
                error="unsupported action: bogus",
                metadata={"is_error": True},
            )
        ],
    ))
    traj_logger.on_complete(_sample({}, messages=messages))

    persisted = [
        _normalize_persisted_message(msg)
        for msg in _persisted_messages(tmp_path / "run")
    ]
    assert persisted[2]["content"] == [{
        "type": "text",
        "text": "## AXTree:\nbody",
    }]


def test_on_step_logs_env_result_images_separately_from_canonical_sample_images(tmp_path):
    actions = [make_tool_call("computer", {"actions": []}, call_id="call_0000")]
    traj_logger = TrajectoryLogger(tmp_path / "run")
    images = [
        Image.new("RGB", (3, 3), "white"),
        Image.new("RGB", (3, 3), "red"),
    ]
    messages = [
        {"role": "user", "content": [{"type": "image", "index": 0}]},
        {"role": "assistant", "content": [], "tool_calls": actions},
        {"role": "tool", "tool_call_id": "call_0000", "content": [{"type": "image", "index": 1}]},
    ]

    traj_logger.on_step(_step_data(
        actions,
        results=[
            LiteToolResult(
                tool_call_id="call_0000",
                images=[_png_bytes("red"), _png_bytes("blue")],
            )
        ],
    ))
    before_complete = json.loads(
        (tmp_path / "run" / "turn_0000" / "04_results.json").read_text()
    )
    before_images = before_complete["results"][0]["images"]
    assert len(before_images) == 2
    assert [image["source"] for image in before_images] == [
        "env_result_images",
        "env_result_images",
    ]
    assert [image["path"] for image in before_images] == [
        "env_result_images/0000_0000_from_call_0000.png",
        "env_result_images/0000_0001_from_call_0000.png",
    ]
    assert all(
        (tmp_path / "run" / "turn_0000" / image["path"]).is_file()
        for image in before_images
    )

    traj_logger.on_complete(_sample({}, messages=messages, images=images))

    results_json = json.loads((tmp_path / "run" / "turn_0000" / "04_results.json").read_text())
    assert results_json["results"][0]["images"] == before_images
    assert (tmp_path / "run" / "images" / "000001.png").is_file()
    assert not (tmp_path / "run" / "turn_0000" / "result_images").exists()


def test_gif_uses_full_env_timeline_but_parquet_keeps_canonical_images(tmp_path):
    actions = [
        make_tool_call(
            "computer",
            {"actions": [{"action": "click"}, {"action": "type", "text": "x"}]},
            call_id="call_batch",
        )
    ]
    messages = [
        {"role": "user", "content": [{"type": "image", "index": 0}]},
        {"role": "assistant", "content": [], "tool_calls": actions},
        {"role": "tool", "tool_call_id": "call_batch", "content": [{"type": "image", "index": 1}]},
    ]
    images = [
        Image.new("RGB", (20, 20), "white"),
        Image.new("RGB", (20, 20), "green"),
    ]
    traj_logger = TrajectoryLogger(
        tmp_path / "run",
        save_gif=True,
        render_instruction_banner=False,
    )

    traj_logger.on_step(_step_data(
        actions,
        screenshot=Image.new("RGB", (100, 80), "white"),
        results=[
            LiteToolResult(
                tool_call_id="call_batch",
                images=[_png_bytes("red"), _png_bytes("blue")],
            )
        ],
    ))
    traj_logger.on_complete(_sample({}, messages=messages, images=images))

    assert _gif_frame_pixels(tmp_path / "run" / "trajectory.gif") == [
        (255, 255, 255),
        (255, 0, 0),
        (0, 0, 255),
    ]
    assert len(_persisted_images(tmp_path / "run")) == 2


def test_gif_corrupt_env_result_image_falls_back_to_stored_images(
    tmp_path,
    caplog,
):
    actions = [make_tool_call("computer", {"actions": []}, call_id="call_0000")]
    images = [
        Image.new("RGB", (20, 20), "white"),
        Image.new("RGB", (20, 20), "green"),
    ]
    messages = [
        {"role": "user", "content": [{"type": "image", "index": 0}]},
        {"role": "assistant", "content": [], "tool_calls": actions},
        {"role": "tool", "tool_call_id": "call_0000", "content": [{"type": "image", "index": 1}]},
    ]
    traj_logger = TrajectoryLogger(
        tmp_path / "run",
        save_gif=True,
        render_instruction_banner=False,
    )

    with caplog.at_level(logging.WARNING, logger="lite.agents.core.agent.logger"):
        traj_logger.on_step(_step_data(
            actions,
            screenshot=Image.new("RGB", (100, 80), "white"),
            results=[
                LiteToolResult(
                    tool_call_id="call_0000",
                    images=[_png_bytes("red"), b"not a png"],
                )
            ],
        ))
        traj_logger.on_complete(_sample({}, messages=messages, images=images))

    assert "falling back to stored trajectory images" in caplog.text
    assert _gif_frame_pixels(tmp_path / "run" / "trajectory.gif") == [
        (255, 255, 255),
        (0, 128, 0),
    ]


def test_gif_fallback_uses_stored_images_and_keeps_terminal_result(tmp_path):
    actions = [make_tool_call("computer", {"actions": []}, call_id="call_0000")]
    images = [
        Image.new("RGB", (20, 20), "white"),
        Image.new("RGB", (20, 20), "red"),
    ]
    messages = [
        {"role": "user", "content": [{"type": "image", "index": 0}]},
        {"role": "assistant", "content": [], "tool_calls": actions},
        {"role": "tool", "tool_call_id": "call_0000", "content": [{"type": "image", "index": 1}]},
    ]
    traj_logger = TrajectoryLogger(
        tmp_path / "run",
        save_gif=True,
        render_instruction_banner=False,
    )

    traj_logger.on_complete(_sample({}, messages=messages, images=images))

    assert _gif_frame_pixels(tmp_path / "run" / "trajectory.gif") == [
        (255, 255, 255),
        (255, 0, 0),
    ]


def test_gif_fallback_uses_all_stored_images_even_when_messages_skip_indices(tmp_path):
    images = [
        Image.new("RGB", (20, 20), "red"),
        Image.new("RGB", (20, 20), "green"),
        Image.new("RGB", (20, 20), "blue"),
    ]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 2},
                {"type": "image", "index": 0},
            ],
        },
    ]
    traj_logger = TrajectoryLogger(
        tmp_path / "run",
        save_gif=True,
        render_instruction_banner=False,
    )

    traj_logger.on_complete(_sample({}, messages=messages, images=images))

    assert _gif_frame_pixels(tmp_path / "run" / "trajectory.gif") == [
        (255, 0, 0),
        (0, 128, 0),
        (0, 0, 255),
    ]


def test_gif_dedupes_adjacent_duplicate_hashes_after_wait_and_keeps_screenshot_result(tmp_path):
    wait_actions = [make_tool_call("wait", {"duration": 1}, call_id="call_0000")]
    screenshot_actions = [make_tool_call("screenshot", call_id="call_0001")]
    images = [
        Image.new("RGB", (20, 20), "red"),
        Image.new("RGB", (20, 20), "red"),
        Image.new("RGB", (20, 20), "blue"),
    ]
    messages = [
        {"role": "user", "content": [{"type": "image", "index": 0}]},
        {"role": "assistant", "content": [], "tool_calls": wait_actions},
        {"role": "tool", "tool_call_id": "call_0000", "content": [{"type": "image", "index": 1}]},
        {"role": "assistant", "content": [], "tool_calls": screenshot_actions},
        {"role": "tool", "tool_call_id": "call_0001", "content": [{"type": "image", "index": 2}]},
    ]
    traj_logger = TrajectoryLogger(
        tmp_path / "run",
        save_gif=True,
        render_instruction_banner=False,
    )

    traj_logger.on_complete(_sample({}, messages=messages, images=images))

    assert _gif_frame_pixels(tmp_path / "run" / "trajectory.gif") == [
        (255, 0, 0),
        (0, 0, 255),
    ]


import copy  # noqa: E402

from lite.agents.core.agent.utils.annotations import action_inspection_records  # noqa: E402


def test_action_inspection_records_expand_to_uniform_trace_records() -> None:
    """Trace expansion emits ONE record shape, never mixed canonical/child dicts.

    Action-batch children and standalone calls are equally
    ``{name, arguments, result_call_id}``, and a child carries its PARENT's id
    because the env pairs every child result against the batch call.
    """
    batch = {
        "id": "call_0000",
        "type": "function",
        "function": {
            "name": "computer",
            "arguments": {
                "actions": [
                    {"action": "click", "coordinate": [1, 2]},
                    {"action": "type", "text": "hi"},
                ]
            },
        },
    }
    standalone = {
        "id": "call_0001",
        "type": "function",
        "function": {"name": "bash", "arguments": {"command": "pwd"}},
    }

    assert action_inspection_records([batch, standalone]) == [
        {"name": "click", "arguments": {"coordinate": [1, 2]}, "result_call_id": "call_0000"},
        {"name": "type", "arguments": {"text": "hi"}, "result_call_id": "call_0000"},
        {"name": "bash", "arguments": {"command": "pwd"}, "result_call_id": "call_0001"},
    ]

    # A child naming an action the batch tool does not carry is no longer an
    # envelope error -- it keeps its slot so the env can answer it per action.
    # The trace should SHOW it: the model emitted it, and a record the model
    # made is exactly what inspection exists to surface.
    unknown_child = copy.deepcopy(batch)
    unknown_child["function"]["arguments"] = {
        "actions": [{"action": "definitely_not_an_action"}]
    }
    assert action_inspection_records([unknown_child]) == [
        {
            "name": "definitely_not_an_action",
            "arguments": {},
            "result_call_id": "call_0000",
        },
    ]

    # Best-effort for trace/log inspection: a genuinely malformed payload (no
    # expandable children at all) still contributes no records instead of
    # raising.
    malformed = copy.deepcopy(batch)
    malformed["function"]["arguments"] = {"actions": "not-a-list"}
    assert action_inspection_records([malformed]) == []
    assert action_inspection_records([{"name": "click", "arguments": {}}]) == []
