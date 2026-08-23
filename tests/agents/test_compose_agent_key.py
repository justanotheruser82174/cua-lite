"""Golden-lock the env-derived agent-key composition rule.

These tests pin the public key grammar and the observable boundaries that consume
it:

  1. base format ``{id}@<metadata dims>``;
  2. a ``.base``-modifier id rides ahead of the ``@`` suffix;
  3. the goal_image guard: an ``adapter_key`` already containing ``@`` is used
     verbatim (never re-composed).

No source-code string guards: the tests observe emitted registry keys and
metadata forwarding at the factory / goal-image boundaries. No network / GPU /
model.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/agents/test_compose_agent_key.py -p no:cacheprovider -q
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lite.agents.core.adapter import AgentAdapterRegistry, AsIsAdapter
from lite.agents.core.agent import AgentRegistry
from lite.agents.extensions.browsergym.goal_image import VisualWebArenaGoalImageAgent
from lite.agents.factory import make
from lite.core import LiteCUAMetadata, LiteGenericMetadata
from lite.utils.registry import compose_key

Platform = LiteCUAMetadata.Platform
TaskType = LiteCUAMetadata.TaskType


def _goal_image_compose(adapter_key: str, meta: LiteCUAMetadata | None) -> str:
    """Mirror of the goal_image guard form: only suffix when no ``@`` present.
    The suffix composition itself delegates to the real ``compose_key``."""
    if "@" not in adapter_key:
        if meta is not None:
            adapter_key = compose_key(adapter_key, *meta.dims)
    return adapter_key


class _OnlineSample(SimpleNamespace):
    class Status:
        FAILED = "failed"


class _OnlineGenerateState:
    def __init__(self, args: Any = None) -> None:
        self.args = args
        self.tokenizer = SimpleNamespace(eos_token_id=0, eos_token="")
        self.processor = object()
        self.aborted = False


def _stub_module(name: str, *, package: bool = False, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    if package:
        module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


async def _unexpected_post(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("online routing test should not call SGLang")

def _install_online_rollout_stubs(monkeypatch) -> None:
    for name, module in {
        "slime": _stub_module("slime", package=True),
        "slime.rollout": _stub_module("slime.rollout", package=True),
        "slime.rollout.base_types": _stub_module(
            "slime.rollout.base_types",
            RolloutFnEvalOutput=object,
            RolloutFnTrainOutput=object,
        ),
        "slime.rollout.filter_hub": _stub_module(
            "slime.rollout.filter_hub", package=True
        ),
        "slime.rollout.filter_hub.base_types": _stub_module(
            "slime.rollout.filter_hub.base_types",
            MetricGatherer=object,
            call_dynamic_filter=lambda *args, **kwargs: None,
        ),
        "slime.rollout.sglang_rollout": _stub_module(
            "slime.rollout.sglang_rollout",
            GenerateState=_OnlineGenerateState,
            generate_and_rm=lambda *args, **kwargs: None,
        ),
        "slime.utils": _stub_module("slime.utils", package=True),
        "slime.utils.async_utils": _stub_module(
            "slime.utils.async_utils",
            run=lambda coro: asyncio.run(coro),
        ),
        "slime.utils.http_utils": _stub_module(
            "slime.utils.http_utils",
            get=lambda *args, **kwargs: None,
            post=_unexpected_post,
        ),
        "slime.utils.misc": _stub_module(
            "slime.utils.misc",
            load_function=lambda path: None,
        ),
        "slime.utils.processing_utils": _stub_module(
            "slime.utils.processing_utils",
            encode_image_for_rollout_engine=lambda image: "encoded",
        ),
        "slime.utils.types": _stub_module("slime.utils.types", Sample=_OnlineSample),
        "lite.train.rollout.core.segmenter": _stub_module(
            "lite.train.rollout.core.segmenter",
            build_segment_samples=(
                lambda rl_sample, sample, processor, tokenizer, state: [sample]
            ),
        ),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def _online_rollout_engine(monkeypatch):
    _install_online_rollout_stubs(monkeypatch)
    module_path = Path(__file__).parents[2] / "lite/train/rollout/core/engine.py"
    spec = importlib.util.spec_from_file_location("_cua_lite_agent_key_engine", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_cua_lite_agent_key_engine", module)
    spec.loader.exec_module(module)
    return module


def _online_rollout_agent_key(
    monkeypatch,
    *,
    metadata,
    agent_id: str,
    prompt_agent_key: str | None = None,
) -> tuple[str, object]:
    engine = _online_rollout_engine(monkeypatch)

    class _Env:
        def __init__(self) -> None:
            self.metadata = metadata
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    class _Agent:
        async def sample(self, env):
            return object()

    env = _Env()
    captured: dict[str, object] = {}

    def _get_agent(key, **kwargs):
        captured["agent_key"] = key
        captured["metadata"] = kwargs["metadata"]
        return _Agent()

    engine.gym = SimpleNamespace(
        make=lambda env_key, **kwargs: env,
        registry=SimpleNamespace(env_supports_kwarg=lambda *args, **kwargs: False),
    )
    engine.register_all = lambda: None
    engine.AgentRegistry = SimpleNamespace(get=_get_agent)
    engine.GenerateState = _OnlineGenerateState

    sample_metadata = {"env_key": "lite.demo@task-a", "split": "train"}
    if prompt_agent_key is not None:
        sample_metadata["agent_key"] = prompt_agent_key
    sample = _OnlineSample(
        group_index=1,
        index=2,
        group_id=2,
        prompt="source prompt",
        metadata=sample_metadata,
    )
    args = SimpleNamespace(
        partial_rollout=False,
        group_shared_seed=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        agent_id=agent_id,
        agent_kwargs={},
        env_kwargs={},
    )

    result = asyncio.run(engine.generate(args, sample, {"sampling_seed": 7}))

    assert result == [sample]
    assert env.closed is True
    return captured["agent_key"], captured["metadata"]


# ===========================================================================
# 1. Base format
# ===========================================================================
def test_base_format_browser_navigation():
    meta = LiteCUAMetadata(dims=(Platform.BROWSER, TaskType.USE))
    assert compose_key("qwen3_vl", *meta.dims) == "qwen3_vl@browser@use"


def test_base_format_desktop_navigation():
    meta = LiteCUAMetadata(dims=(Platform.DESKTOP, TaskType.USE))
    assert compose_key("claude", *meta.dims) == "claude@desktop@use"


def test_enum_str_mapping_is_the_load_bearing_piece():
    """The key uses ``str(enum)`` (lowercase dim names), not ``.name`` /
    ``repr``. Pin it so the dedup keeps using ``str()``."""
    meta = LiteCUAMetadata(dims=(Platform.BROWSER, TaskType.USE))
    assert str(meta.platform) == "browser"
    assert str(meta.task_type) == "use"


# ===========================================================================
# 2. ``.base`` modifier id
# ===========================================================================
def test_base_modifier_id_rides_ahead_of_suffix():
    meta = LiteCUAMetadata(dims=(Platform.BROWSER, TaskType.USE))
    assert compose_key("qwen3_vl.base", *meta.dims) == "qwen3_vl.base@browser@use"


# ===========================================================================
# 3. Attribute and dict-subscript call sites agree on equivalent input
# ===========================================================================
def test_metadata_dims_and_dict_dims_forms_byte_identical():
    """Runtime metadata objects and serialized row metadata share ``dims``."""
    meta = LiteCUAMetadata(dims=(Platform.BROWSER, TaskType.USE))
    metadata_dict = meta.to_dict()
    attr_key = compose_key("gpt", *meta.dims)
    dict_key = compose_key("gpt", *metadata_dict["dims"])
    assert attr_key == dict_key == "gpt@browser@use"


# ===========================================================================
# 4. goal_image "@" guard
# ===========================================================================
def test_goal_image_bare_key_gets_suffix():
    meta = LiteCUAMetadata(dims=(Platform.BROWSER, TaskType.USE))
    assert _goal_image_compose("qwen3_vl", meta) == "qwen3_vl@browser@use"


def test_goal_image_already_qualified_key_used_verbatim():
    """A key already containing ``@`` is NOT re-composed (the extra guard that
    distinguishes this site from the other three)."""
    meta = LiteCUAMetadata(dims=(Platform.DESKTOP, TaskType.USE))
    already = "qwen3_vl@browser@use"
    assert _goal_image_compose(already, meta) == already  # not desktop, verbatim


def test_goal_image_base_modifier_bare_key_gets_suffix():
    meta = LiteCUAMetadata(dims=(Platform.BROWSER, TaskType.USE))
    assert (
        _goal_image_compose("qwen3_vl.base", meta)
        == "qwen3_vl.base@browser@use"
    )


# ===========================================================================
# Boundary behavior — registry keys and metadata forwarding
# ===========================================================================
@pytest.mark.parametrize(
    ("model_id", "agent_id", "meta", "expected_key", "expected_kwargs"),
    [
        (
            "gpt-5.5",
            None,
            LiteCUAMetadata(dims=(Platform.MOBILE, TaskType.USE)),
            "gpt@mobile@use",
            {"model_id": "gpt-5.5"},
        ),
        (
            "Qwen/Qwen3-VL-8B-Instruct",
            "qwen3_vl.base",
            LiteCUAMetadata(dims=(Platform.BROWSER, TaskType.USE)),
            "qwen3_vl.base@browser@use",
            {},
        ),
    ],
)
def test_factory_make_routing_matrix_composes_from_metadata_dims(
    monkeypatch,
    model_id,
    agent_id,
    meta,
    expected_key,
    expected_kwargs,
):
    env = SimpleNamespace(metadata=meta)
    calls = []
    sentinel = object()

    def fake_get(cls, name, **kwargs):
        calls.append((name, kwargs))
        return sentinel

    monkeypatch.setattr(AgentRegistry, "get", classmethod(fake_get))

    assert make(model_id, env=env, agent_id=agent_id) is sentinel

    assert calls == [(expected_key, {**expected_kwargs, "metadata": meta})]


def test_factory_make_resolves_agent_key_from_env_metadata(monkeypatch):
    meta = LiteCUAMetadata(dims=(Platform.BROWSER, TaskType.USE))
    env = SimpleNamespace(metadata=meta)
    calls = []
    sentinel = object()

    def fake_get(cls, name, **kwargs):
        calls.append((name, kwargs))
        return sentinel

    monkeypatch.setattr(AgentRegistry, "get", classmethod(fake_get))

    assert make("gpt-5.5", env=env) is sentinel

    assert calls == [
        (
            "gpt@browser@use",
            {"model_id": "gpt-5.5", "metadata": meta},
        )
    ]


def test_factory_make_respects_agent_id_modifier_override(monkeypatch):
    meta = LiteCUAMetadata(dims=(Platform.BROWSER, TaskType.USE))
    env = SimpleNamespace(metadata=meta)
    calls = []
    sentinel = object()

    def fake_get(cls, name, **kwargs):
        calls.append((name, kwargs))
        return sentinel

    monkeypatch.setattr(AgentRegistry, "get", classmethod(fake_get))

    assert (
        make("Qwen/Qwen3-VL-8B-Instruct", env=env, agent_id="qwen3_vl.base")
        is sentinel
    )

    name, kwargs = calls[0]
    assert name == "qwen3_vl.base@browser@use"
    assert kwargs == {"metadata": meta}


def test_goal_image_agent_resolves_bare_adapter_key_from_metadata(monkeypatch):
    meta = LiteCUAMetadata(dims=(Platform.BROWSER, TaskType.USE))
    calls = []
    adapter = SimpleNamespace(
        protocol=SimpleNamespace(process_messages=lambda messages, **kwargs: messages)
    )

    def fake_get(cls, name, **kwargs):
        calls.append((name, kwargs))
        return adapter

    monkeypatch.setattr(AgentAdapterRegistry, "get", classmethod(fake_get))

    agent = VisualWebArenaGoalImageAgent(
        generate_fn=lambda *args, **kwargs: None,
        kwargs={"adapter_key": "qwen3_vl.base", "metadata": meta},
    )

    assert agent.adapter is adapter
    assert calls == [("qwen3_vl.base@browser@use", {"metadata": meta})]


@pytest.mark.parametrize(
    ("adapter_key", "meta", "expected_key"),
    [
        (
            "qwen3_vl",
            LiteCUAMetadata(dims=(Platform.BROWSER, TaskType.USE)),
            "qwen3_vl@browser@use",
        ),
        (
            "qwen3_vl.base",
            LiteCUAMetadata(dims=(Platform.DESKTOP, TaskType.USE)),
            "qwen3_vl.base@desktop@use",
        ),
    ],
)
def test_goal_image_agent_routing_matrix_composes_from_metadata_dims(
    monkeypatch,
    adapter_key,
    meta,
    expected_key,
):
    calls = []
    adapter = SimpleNamespace(
        protocol=SimpleNamespace(process_messages=lambda messages, **kwargs: messages)
    )

    def fake_get(cls, name, **kwargs):
        calls.append((name, kwargs))
        return adapter

    monkeypatch.setattr(AgentAdapterRegistry, "get", classmethod(fake_get))

    VisualWebArenaGoalImageAgent(
        generate_fn=lambda *args, **kwargs: None,
        kwargs={"adapter_key": adapter_key, "metadata": meta},
    )

    assert calls == [(expected_key, {"metadata": meta})]


def test_factory_make_generic_empty_dims_keeps_bare_agent_key(monkeypatch):
    meta = LiteGenericMetadata(dims=())
    env = SimpleNamespace(metadata=meta)
    calls = []
    sentinel = object()

    def fake_get(cls, name, **kwargs):
        calls.append((name, kwargs))
        return sentinel

    monkeypatch.setattr(AgentRegistry, "get", classmethod(fake_get))

    assert make("gpt-5.5", env=env, agent_id="generic_agent") is sentinel

    assert calls == [
        (
            "generic_agent",
            {"model_id": "gpt-5.5", "metadata": meta},
        )
    ]


def test_goal_image_agent_generic_empty_dims_keeps_bare_adapter_key(monkeypatch):
    meta = LiteGenericMetadata(dims=())
    calls = []
    adapter = SimpleNamespace(
        protocol=SimpleNamespace(process_messages=lambda messages, **kwargs: messages)
    )

    def fake_get(cls, name, **kwargs):
        calls.append((name, kwargs))
        return adapter

    monkeypatch.setattr(AgentAdapterRegistry, "get", classmethod(fake_get))

    VisualWebArenaGoalImageAgent(
        generate_fn=lambda *args, **kwargs: None,
        kwargs={"adapter_key": "generic_adapter", "metadata": meta},
    )

    assert calls == [("generic_adapter", {"metadata": meta})]


def test_online_rollout_fallback_composes_agent_key_from_metadata_dims(monkeypatch):
    meta = LiteCUAMetadata(dims=(Platform.BROWSER, TaskType.USE))

    agent_key, forwarded_metadata = _online_rollout_agent_key(
        monkeypatch,
        metadata=meta,
        agent_id="qwen3_vl.base",
    )

    assert agent_key == "qwen3_vl.base@browser@use"
    assert forwarded_metadata is meta


def test_online_rollout_generic_empty_dims_keeps_bare_agent_key(monkeypatch):
    meta = LiteGenericMetadata(dims=())

    agent_key, forwarded_metadata = _online_rollout_agent_key(
        monkeypatch,
        metadata=meta,
        agent_id="generic_agent",
    )

    assert agent_key == "generic_agent"
    assert forwarded_metadata is meta


def test_online_rollout_prompt_data_agent_key_override_has_highest_priority(monkeypatch):
    meta = LiteCUAMetadata(dims=(Platform.DESKTOP, TaskType.USE))

    agent_key, forwarded_metadata = _online_rollout_agent_key(
        monkeypatch,
        metadata=meta,
        agent_id="qwen3_vl",
        prompt_agent_key="explicit@prompt@key",
    )

    assert agent_key == "explicit@prompt@key"
    assert forwarded_metadata is meta


def test_cua_adapter_surface_rejects_generic_metadata():
    adapter = AsIsAdapter(metadata=LiteGenericMetadata(dims=()))

    with pytest.raises(TypeError, match="requires LiteCUAMetadata"):
        adapter._assemble_tool_schemas()


def test_goal_image_agent_uses_qualified_adapter_key_verbatim(monkeypatch):
    meta = LiteCUAMetadata(dims=(Platform.BROWSER, TaskType.USE))
    calls = []
    adapter = SimpleNamespace(
        protocol=SimpleNamespace(process_messages=lambda messages, **kwargs: messages)
    )

    def fake_get(cls, name, **kwargs):
        calls.append((name, kwargs))
        return adapter

    monkeypatch.setattr(AgentAdapterRegistry, "get", classmethod(fake_get))

    agent = VisualWebArenaGoalImageAgent(
        generate_fn=lambda *args, **kwargs: None,
        kwargs={"adapter_key": "qwen3_vl@desktop@use", "metadata": meta},
    )

    assert agent.adapter is adapter
    assert calls == [("qwen3_vl@desktop@use", {"metadata": meta})]
