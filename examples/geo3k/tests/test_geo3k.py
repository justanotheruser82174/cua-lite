from __future__ import annotations

import base64
import importlib
import json
import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pytest
import yaml
from PIL import Image

import lite.gym as gym
from examples.geo3k import Geo3KEnv, register_geo3k
from examples.geo3k.env import ENV_ID, SMOKE_TASKS, TASK_ID, geo3k_source_fingerprint
from lite.agents.core.agent.base import AdapterBasedAgent
from lite.agents.core.agent.logger import TrajectoryLogger
from lite.core import LiteGenericMetadata, LiteSample, metadata_from_dict
from lite.core.messages.final import make_no_tool_call_final_actions
from lite.core.tools import make_tool_call
from lite.core.tools.calls import RUNTIME_INTERNAL_STOP_REASON_KEY
from lite.core.tools.schemas import tool_schema_name
from lite.data.load import load_file_as_dataset
from lite.data.staging import coerce_messages
from lite.gym import registry
from lite.gym.types import EXECUTED_ACTIONS_INFO_KEY
from lite.utils.image import encode_png
from lite.utils.parquet import write_records_to_parquet
from lite.utils.registry import compose_key

_ROOT = Path(__file__).resolve().parents[3]
_REGISTRY_MODULE = importlib.import_module("lite.gym.registry")


class _Geo3KFakeProcessor:
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False) -> str:
        del add_generation_prompt, tokenize
        return f"<prompt {len(messages)} messages>"


class _Geo3KScriptedTextAdapter:
    _registry_key = "geo3k.fake"
    metadata = LiteGenericMetadata()

    def __init__(self, turns: list[str | list[dict[str, object]]]) -> None:
        self._turns = turns
        self._i = 0

    @classmethod
    def get_registry_key(cls) -> str:
        return cls._registry_key

    def render_step(self, lite_sample, k: int, processed_images) -> list[dict[str, object]]:
        del k, processed_images
        return list(lite_sample.messages)

    def process_image(self, img):
        return img

    def parse_raw_assistant_response(self, response: str) -> dict[str, object]:
        return {"role": "assistant", "content": [{"type": "text", "text": response}]}

    def convert_message_from_agent(self, agent_message) -> dict[str, object]:
        del agent_message
        turn = self._turns[self._i] if self._i < len(self._turns) else ""
        self._i += 1
        if isinstance(turn, list):
            return {"role": "assistant", "content": [], "tool_calls": turn}
        text = turn
        return {"role": "assistant", "content": [{"type": "text", "text": text}]}


@pytest.fixture(autouse=True)
def _local_registry_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_URL", raising=False)
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_TOKEN", raising=False)


@pytest.fixture(autouse=True)
def _clear_geo3k_registry() -> None:
    _REGISTRY_MODULE._clear_env_registration(ENV_ID)
    yield
    _REGISTRY_MODULE._clear_env_registration(ENV_ID)


def _geo3k_source(tmp_path: Path) -> Path:
    image = Image.new("RGB", (16, 16), "white")
    png = encode_png(image)
    data_url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    path = tmp_path / "geo3k_source.parquet"
    write_records_to_parquet(
        [
            {
                "problem": "A 3-4-5 right triangle has hypotenuse 5. What is its hypotenuse?",
                "answer": "5",
                "images": [data_url],
                "preprocessed_images": [{"bytes": png, "path": None}],
            },
            {
                "problem": "What is the area of a rectangle with sides 2 and 7?",
                "answer": "14",
                "images": [],
                "preprocessed_images": [],
            },
        ],
        path,
    )
    return path


def _single_row_geo3k_source(tmp_path: Path, *, answer: str) -> Path:
    path = tmp_path / f"geo3k_source_{answer}.parquet"
    write_records_to_parquet(
        [{
            "problem": f"What number is {answer}?",
            "answer": answer,
            "images": [],
            "preprocessed_images": [],
        }],
        path,
    )
    return path


def _write(path: Path, text: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if executable:
        path.chmod(0o755)


def _write_fake_models_helper(root: Path) -> None:
    _write(
        root / "scripts" / "train" / "utils" / "models.sh",
        'case "$MODEL_ID" in\n'
        '  Qwen/Qwen3.5*) MODEL_FAMILY=qwen3_5 ;;\n'
        '  *) MODEL_FAMILY=qwen3_vl ;;\n'
        "esac\n",
    )


def _stage_geo3k_grpo_wrapper(root: Path) -> Path:
    wrapper = root / "examples" / "geo3k" / "scripts" / "run_grpo.sh"
    _write(
        wrapper,
        (_ROOT / "examples" / "geo3k" / "scripts" / "run_grpo.sh").read_text(),
        executable=True,
    )
    return wrapper


def test_geo3k_metadata_is_generic_and_routes_to_bare_agent_key() -> None:
    env = Geo3KEnv()

    metadata = env.metadata

    assert isinstance(metadata, LiteGenericMetadata)
    assert metadata.dims == ()
    assert not hasattr(metadata, "platform")
    assert not hasattr(metadata, "task_type")
    assert not hasattr(metadata, "valid_actions")
    assert metadata.extra_tool_schemas == []
    assert compose_key("qwen3_5.base", *metadata.dims) == "qwen3_5.base"


def test_geo3k_registry_preserves_generic_task_metadata() -> None:
    register_geo3k()

    task_ids = registry.task_ids(ENV_ID, split="train")

    assert set(task_ids) == set(SMOKE_TASKS)
    metadata = registry.task_metadata(ENV_ID, TASK_ID)

    assert isinstance(metadata, LiteGenericMetadata)
    assert metadata.dims == ()
    assert metadata.others["env_id"] == ENV_ID
    assert metadata.others["task_id"] == TASK_ID
    assert metadata_from_dict(metadata.to_dict()) == metadata


def test_geo3k_registry_declares_runtime_env_kwargs(tmp_path) -> None:
    register_geo3k()

    assert set(registry.env_supported_kwargs(ENV_ID)) >= {"extra_tools", "max_turns"}
    assert registry.env_supports_kwarg(ENV_ID, "max_turns")

    source = _geo3k_source(tmp_path)
    register_geo3k(source_path=source)

    assert set(registry.env_supported_kwargs(ENV_ID)) >= {"extra_tools", "max_turns"}
    assert registry.env_supports_kwarg(ENV_ID, "max_turns")


def test_geo3k_registers_one_task_per_source_row(tmp_path) -> None:
    source = _geo3k_source(tmp_path)

    register_geo3k(source_path=source)

    assert registry.task_ids(ENV_ID, split="train") == ["row_000000", "row_000001"]
    metadata = registry.task_metadata(ENV_ID, "row_000000")
    assert isinstance(metadata, LiteGenericMetadata)
    assert metadata.dims == ()
    assert metadata.extra_tool_schemas == []


def test_geo3k_registration_replaces_prior_catalog(tmp_path) -> None:
    source = _geo3k_source(tmp_path)
    register_geo3k()

    register_geo3k(source_path=source)

    assert registry.task_ids(ENV_ID, split="train") == ["row_000000", "row_000001"]


def test_geo3k_registration_keeps_catalog_when_source_is_invalid(tmp_path) -> None:
    register_geo3k()

    with pytest.raises(FileNotFoundError):
        register_geo3k(source_path=tmp_path / "missing.parquet")

    assert set(registry.task_ids(ENV_ID, split="train")) == set(SMOKE_TASKS)


def test_geo3k_direct_preflight_loads_registration_module() -> None:
    env = os.environ.copy()
    env.update({
        "ENV_ID": ENV_ID,
        "CUA_LITE_REGISTRATION_MODULES": "examples.geo3k.registration",
    })
    env.pop("CUA_LITE_ENV_SERVER_URL", None)
    env.pop("CUA_LITE_ENV_SERVER_TOKEN", None)

    result = subprocess.run(
        ["bash", "-lc", "source scripts/train/utils/preflight.sh"],
        cwd=_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert f"DIRECT MODE: {ENV_ID} has no external backend." in result.stdout


def test_geo3k_make_loads_registration_module_without_import_all() -> None:
    env = os.environ.copy()
    env["CUA_LITE_REGISTRATION_MODULES"] = "examples.geo3k.registration"
    env.pop("CUA_LITE_ENV_SERVER_URL", None)
    env.pop("CUA_LITE_ENV_SERVER_TOKEN", None)

    code = f"""
import importlib
reg = importlib.import_module("lite.gym.registry")
def boom():
    raise AssertionError("_import_all should not run for registered external envs")
reg._import_all = boom
import lite.gym as gym
env = gym.make("{ENV_ID}@{TASK_ID}")
print(type(env).__name__)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert "StepTimeoutWrapper" in result.stdout


def test_geo3k_make_replays_cached_registration_module_after_registry_clear() -> None:
    env = os.environ.copy()
    env["CUA_LITE_REGISTRATION_MODULES"] = "examples.geo3k.registration"
    env.pop("CUA_LITE_ENV_SERVER_URL", None)
    env.pop("CUA_LITE_ENV_SERVER_TOKEN", None)

    code = f"""
import importlib
import examples.geo3k.registration
reg = importlib.import_module("lite.gym.registry")
reg._clear_env_registration("{ENV_ID}")
def boom():
    raise AssertionError("_import_all should not run for registered external envs")
reg._import_all = boom
import lite.gym as gym
env = gym.make("{ENV_ID}@{TASK_ID}")
print(type(env).__name__)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert "StepTimeoutWrapper" in result.stdout


def test_geo3k_export_tasks_wrapper_writes_control_metadata(tmp_path) -> None:
    path = tmp_path / "train.parquet"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.geo3k.export_tasks",
            "--env-id",
            ENV_ID,
            "--split",
            "train",
            "-o",
            str(path),
        ],
        cwd=_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    rows = load_file_as_dataset(path)
    assert {
        row["metadata"]["env_key"]
        for row in rows
    } == {f"{ENV_ID}@{task_id}" for task_id in SMOKE_TASKS}
    assert {row["metadata"]["split"] for row in rows} == {"train"}


def test_geo3k_export_tasks_wrapper_uses_source_split_and_fingerprint(tmp_path) -> None:
    source = _geo3k_source(tmp_path)
    output = tmp_path / "test.parquet"
    env = os.environ.copy()
    env["GEO3K_SOURCE"] = str(source)
    env["CUA_LITE_ENV_SERVER_URL"] = "http://127.0.0.1:39999"
    env["CUA_LITE_ENV_SERVER_TOKEN"] = "stale-token"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.geo3k.export_tasks",
            "--env-id",
            ENV_ID,
            "--split",
            "test",
            "-o",
            str(output),
        ],
        cwd=_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    rows = load_file_as_dataset(output)
    assert len(rows) == 2
    assert {row["metadata"]["split"] for row in rows} == {"test"}
    assert {
        row["metadata"]["env_kwargs"]["source_fingerprint"]
        for row in rows
    } == {geo3k_source_fingerprint(source)}


def test_geo3k_export_tasks_help_does_not_touch_source_env(tmp_path) -> None:
    env = os.environ.copy()
    env["GEO3K_SOURCE"] = str(tmp_path / "missing.parquet")

    result = subprocess.run(
        [sys.executable, "-m", "examples.geo3k.export_tasks", "--help"],
        cwd=_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert "Create a parquet task list" in result.stdout


def test_geo3k_export_tasks_import_does_not_mutate_env() -> None:
    code = """
import os
os.environ["CUA_LITE_ENV_SERVER_URL"] = "http://127.0.0.1:39999"
os.environ["CUA_LITE_ENV_SERVER_TOKEN"] = "token"
import examples.geo3k.export_tasks
print(os.environ["CUA_LITE_ENV_SERVER_URL"])
print(os.environ["CUA_LITE_ENV_SERVER_TOKEN"])
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["http://127.0.0.1:39999", "token"]


def test_geo3k_real_configs_make_single_turn_natural_language_explicit() -> None:
    for family, agent_id in [
        ("qwen3_vl", "qwen3_vl.base"),
        ("qwen3_5", "qwen3_5.base"),
    ]:
        config = yaml.safe_load(
            (_ROOT / "examples" / "geo3k" / "configs" / family / "geo3k.yaml").read_text()
        )
        assert config["env_id"] == ENV_ID
        assert config["agent_id"] == agent_id
        assert config["agent_kwargs"]["render_tools_section"] is False
        assert config["env_kwargs"] == {"max_turns": 1}


def test_geo3k_multiturn_configs_use_natural_language_feedback_loop() -> None:
    for family, agent_id in [
        ("qwen3_vl", "qwen3_vl.base"),
        ("qwen3_5", "qwen3_5.base"),
    ]:
        config = yaml.safe_load(
            (
                _ROOT
                / "examples"
                / "geo3k"
                / "configs"
                / family
                / "geo3k.mt.yaml"
            ).read_text()
        )
        assert config["env_id"] == ENV_ID
        assert config["agent_id"] == agent_id
        assert config["agent_kwargs"]["render_tools_section"] is False
        assert config["env_kwargs"] == {"max_turns": 3}


def test_geo3k_grpo_wrapper_execs_canonical_with_example_contract(tmp_path) -> None:
    root = tmp_path / "cua-lite"
    wrapper = _stage_geo3k_grpo_wrapper(root)
    canonical = root / "scripts" / "train" / "run_grpo.sh"
    _write(
        canonical,
        "#!/bin/bash\n"
        "printf '%s\\n' \"$MODEL_ID\" \"$ROLLOUT_MODULE\" \"$CONFIG_PATH\" \"$PROMPT_DATA\" "
        "\"$CUA_LITE_REGISTRATION_MODULES\" \"$NUM_TRAIN_GPUS\" "
        "\"$ROLLOUT_BATCH_SIZE\" \"$N_SAMPLES_PER_PROMPT\" "
        "\"$NUM_STEPS_PER_ROLLOUT\" \"$ENV_CONCURRENCY\" "
        "\"$NUM_ROLLOUT_GPUS\" \"${DUMP:-unset}\" "
        "\"${CUA_LITE_ENV_SERVER_URL:-unset}\" "
        "\"${CUA_LITE_ENV_SERVER_TOKEN:-unset}\" "
        "\"${CUA_LITE_RAY_ENV_VARS:-unset}\" \"${GEO3K_SOURCE:-unset}\" "
        "> \"$FAKE_CAPTURE\"\n",
        executable=True,
    )
    _write_fake_models_helper(root)
    config = root / "examples" / "geo3k" / "configs" / "qwen3_vl" / "geo3k.yaml"
    _write(config, "agent_id: qwen3_vl.base\nenv_id: geo3k\n")
    prompt = root / "prompt.parquet"
    prompt.write_text("fake")
    source = root / "geo3k_source.parquet"
    source.write_text("fake")

    env = os.environ.copy()
    env.update({
        "CUA_LITE_ROOT": str(tmp_path / "stale-root"),
        "CUA_LITE_REGISTRATION_MODULES": "extra.registration",
        "PROMPT_DATA": str(prompt),
        "FAKE_CAPTURE": str(root / "capture.txt"),
        "NUM_TRAIN_GPUS": "2",
        "NUM_ROLLOUT_GPUS": "2",
        "CUA_LITE_ENV_SERVER_URL": "http://127.0.0.1:39999",
        "CUA_LITE_ENV_SERVER_TOKEN": "stale-token",
        "GEO3K_SOURCE": str(source),
    })

    result = subprocess.run(
        ["bash", str(wrapper)],
        cwd=_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    captured = (root / "capture.txt").read_text().splitlines()
    assert captured == [
        "Qwen/Qwen3-VL-2B-Instruct",
        "lite.train.rollout.grpo",
        str(config),
        str(prompt),
        "extra.registration,examples.geo3k.registration",
        "2",
        "32",
        "8",
        "1",
        "256",
        "2",
        "unset",
        "unset",
        "unset",
        "GEO3K_SOURCE",
        str(root / "geo3k_source.parquet"),
    ]


def test_geo3k_grpo_wrapper_selects_qwen35_config_from_model_id(tmp_path) -> None:
    root = tmp_path / "cua-lite"
    wrapper = _stage_geo3k_grpo_wrapper(root)
    canonical = root / "scripts" / "train" / "run_grpo.sh"
    _write(
        canonical,
        "#!/bin/bash\nprintf '%s\\n' \"$CONFIG_PATH\" > \"$FAKE_CAPTURE\"\n",
        executable=True,
    )
    _write_fake_models_helper(root)
    config = root / "examples" / "geo3k" / "configs" / "qwen3_5" / "geo3k.yaml"
    _write(config, "agent_id: qwen3_5.base\nenv_id: geo3k\n")
    prompt = root / "prompt.parquet"
    prompt.write_text("fake")
    source = root / "geo3k_source.parquet"
    source.write_text("fake")

    env = os.environ.copy()
    env.update({
        "CUA_LITE_ROOT": str(tmp_path / "stale-root"),
        "MODEL_ID": "Qwen/Qwen3.5-2B",
        "PROMPT_DATA": str(prompt),
        "GEO3K_SOURCE": str(source),
        "FAKE_CAPTURE": str(root / "capture.txt"),
        "NUM_TRAIN_GPUS": "2",
        "NUM_ROLLOUT_GPUS": "2",
    })

    result = subprocess.run(
        ["bash", str(wrapper)],
        cwd=_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert (root / "capture.txt").read_text().strip() == str(config)


def test_geo3k_grpo_wrapper_requires_source_for_existing_prompt(tmp_path) -> None:
    root = tmp_path / "cua-lite"
    wrapper = _stage_geo3k_grpo_wrapper(root)
    canonical = root / "scripts" / "train" / "run_grpo.sh"
    _write(canonical, "#!/bin/bash\nexit 99\n", executable=True)
    _write_fake_models_helper(root)
    config = root / "examples" / "geo3k" / "configs" / "qwen3_vl" / "geo3k.yaml"
    _write(config, "agent_id: qwen3_vl.base\nenv_id: geo3k\n")
    prompt = root / "prompt.parquet"
    prompt.write_text("fake")

    env = os.environ.copy()
    env.update({
        "CUA_LITE_ROOT": str(tmp_path / "stale-root"),
        "PROMPT_DATA": str(prompt),
        "NUM_TRAIN_GPUS": "2",
        "NUM_ROLLOUT_GPUS": "2",
    })
    env.pop("GEO3K_SOURCE", None)

    result = subprocess.run(
        ["bash", str(wrapper)],
        cwd=_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 1
    assert "GEO3K_SOURCE is required for Geo3K GRPO" in result.stderr


def test_geo3k_grpo_wrapper_requires_existing_source(tmp_path) -> None:
    root = tmp_path / "cua-lite"
    wrapper = _stage_geo3k_grpo_wrapper(root)
    canonical = root / "scripts" / "train" / "run_grpo.sh"
    _write(canonical, "#!/bin/bash\nexit 99\n", executable=True)
    _write_fake_models_helper(root)
    config = root / "examples" / "geo3k" / "configs" / "qwen3_vl" / "geo3k.yaml"
    _write(config, "agent_id: qwen3_vl.base\nenv_id: geo3k\n")
    prompt = root / "prompt.parquet"
    prompt.write_text("fake")
    source = root / "missing-source.parquet"

    env = os.environ.copy()
    env.update({
        "CUA_LITE_ROOT": str(tmp_path / "stale-root"),
        "PROMPT_DATA": str(prompt),
        "GEO3K_SOURCE": str(source),
        "NUM_TRAIN_GPUS": "2",
        "NUM_ROLLOUT_GPUS": "2",
    })

    result = subprocess.run(
        ["bash", str(wrapper)],
        cwd=_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 1
    assert f"GEO3K_SOURCE not found: {source}" in result.stderr


def test_geo3k_grpo_wrapper_missing_prompt_points_to_example_exporter(
    tmp_path,
) -> None:
    root = tmp_path / "cua-lite"
    wrapper = _stage_geo3k_grpo_wrapper(root)
    canonical = root / "scripts" / "train" / "run_grpo.sh"
    _write(canonical, "#!/bin/bash\nexit 99\n", executable=True)
    _write_fake_models_helper(root)
    config = root / "examples" / "geo3k" / "configs" / "qwen3_vl" / "geo3k.yaml"
    _write(config, "agent_id: qwen3_vl.base\nenv_id: geo3k\n")

    env = os.environ.copy()
    env.update({
        "CUA_LITE_ROOT": str(tmp_path / "stale-root"),
        "PROMPT_DATA": str(root / "missing.parquet"),
        "NUM_TRAIN_GPUS": "2",
        "NUM_ROLLOUT_GPUS": "2",
    })

    result = subprocess.run(
        ["bash", str(wrapper)],
        cwd=_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 1
    assert "bash examples/geo3k/scripts/install.sh" in result.stderr
    assert "uv run bash examples/geo3k/scripts/install.sh" not in result.stderr
    assert "python -m examples.geo3k.export_tasks" in result.stderr
    assert "python -m lite.train.export.export_tasks" not in result.stderr


def test_geo3k_grpo_wrapper_requires_explicit_gpu_counts(tmp_path) -> None:
    root = tmp_path / "cua-lite"
    wrapper = _stage_geo3k_grpo_wrapper(root)
    canonical = root / "scripts" / "train" / "run_grpo.sh"
    _write(canonical, "#!/bin/bash\nexit 99\n", executable=True)
    _write_fake_models_helper(root)
    config = root / "examples" / "geo3k" / "configs" / "qwen3_vl" / "geo3k.yaml"
    _write(config, "agent_id: qwen3_vl.base\nenv_id: geo3k\n")
    prompt = root / "prompt.parquet"
    prompt.write_text("fake")
    source = root / "geo3k_source.parquet"
    source.write_text("fake")

    env = os.environ.copy()
    env.update({
        "CUA_LITE_ROOT": str(tmp_path / "stale-root"),
        "PROMPT_DATA": str(prompt),
        "GEO3K_SOURCE": str(source),
    })
    env.pop("NUM_TRAIN_GPUS", None)
    env.pop("NUM_ROLLOUT_GPUS", None)

    result = subprocess.run(
        ["bash", str(wrapper)],
        cwd=_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 1
    assert "Geo3K GRPO requires explicit GPU counts" in result.stderr


@pytest.mark.asyncio
async def test_geo3k_step_scores_natural_language_boxed_answer() -> None:
    env = Geo3KEnv()

    observation = await env.reset()
    result = await env.step(
        make_no_tool_call_final_actions(
            "The diagonal is found by Pythagoras.\nAnswer: \\boxed{\\sqrt{2}}"
        )
    )

    assert observation.image is None
    assert observation.text
    assert result.terminated is True
    assert result.truncated is False
    assert result.reward == 1.0
    assert result.results[0].tool_call_id is None
    assert "expected=sqrt(2)" in (result.results[0].text or "")
    assert result.info[EXECUTED_ACTIONS_INFO_KEY] == [
        {
            "call": "response",
            "args": {
                "text": "The diagonal is found by Pythagoras.\nAnswer: \\boxed{\\sqrt{2}}"
            },
        }
    ]
    assert result.info["attempt"] == 1
    assert result.info["max_turns"] == 1
    assert result.info["correct"] is True


@pytest.mark.parametrize("max_turns", [True, 1.5, "3"])
def test_geo3k_rejects_non_integer_max_turns(max_turns) -> None:
    with pytest.raises(TypeError, match="max_turns must be an integer"):
        Geo3KEnv(max_turns=max_turns)


def test_geo3k_rejects_nonpositive_max_turns() -> None:
    with pytest.raises(ValueError, match="max_turns must be >= 1"):
        Geo3KEnv(max_turns=0)


@pytest.mark.asyncio
async def test_geo3k_single_turn_wrong_answer_terminates_without_retry() -> None:
    env = Geo3KEnv(max_turns=1)

    await env.reset()
    result = await env.step(make_no_tool_call_final_actions("Answer: \\boxed{3}"))

    assert result.reward == 0.0
    assert result.terminated is True
    assert result.truncated is False
    assert "no attempts remain" in (result.results[0].text or "")
    assert "revise your reasoning" not in (result.results[0].text or "")
    assert "expected=sqrt(2)" in (result.results[0].text or "")
    assert result.info["attempt"] == 1
    assert result.info["max_turns"] == 1
    assert result.info["correct"] is False


@pytest.mark.asyncio
async def test_geo3k_rejects_explicit_response_tool_when_not_advertised() -> None:
    env = Geo3KEnv(max_turns=1)

    await env.reset()
    result = await env.step([
        make_tool_call("response", {"text": "Answer: \\boxed{\\sqrt{2}}"}, call_id="call_0")
    ])

    assert result.reward == 0.0
    assert result.terminated is True
    assert result.results[0].tool_call_id == "call_0"
    assert result.results[0].error == "response is not available in this task."
    assert result.info["attempt"] == 1
    assert result.info["correct"] is False


@pytest.mark.asyncio
async def test_geo3k_rejected_response_submission_consumes_attempt_budget() -> None:
    env = Geo3KEnv(max_turns=1)

    await env.reset()
    result = await env.step([
        make_tool_call("response", {"text": "Answer: \\boxed{\\sqrt{2}}"}, call_id="call_0")
    ])
    after_done = await env.step(make_no_tool_call_final_actions("Answer: \\boxed{\\sqrt{2}}"))

    assert result.terminated is True
    assert result.info["attempt"] == 1
    assert result.info["max_turns"] == 1
    assert result.results[0].error == "response is not available in this task."
    assert after_done.terminated is True
    assert after_done.info["attempt"] == 1
    assert after_done.results[0].error == "task already finished"


@pytest.mark.asyncio
async def test_geo3k_rejects_public_response_tool_with_internal_sidecar() -> None:
    env = Geo3KEnv(max_turns=1)
    action = make_tool_call(
        "response",
        {"text": "Answer: \\boxed{\\sqrt{2}}"},
        call_id="call_0",
    )
    action[RUNTIME_INTERNAL_STOP_REASON_KEY] = "content_only_final"

    await env.reset()
    with pytest.raises(TypeError, match="_internal_stop_reason is reserved"):
        await env.step([action])


def test_geo3k_extra_tools_only_exposes_response() -> None:
    with pytest.raises(ValueError, match="cannot execute extra_tools"):
        Geo3KEnv(extra_tools=["terminate"])


def test_geo3k_multiturn_metadata_exposes_response_tool() -> None:
    env = Geo3KEnv(max_turns=3, extra_tools=["response"])

    metadata = env.metadata

    assert isinstance(metadata, LiteGenericMetadata)
    assert metadata.dims == ()
    assert [tool_schema_name(schema) for schema in metadata.extra_tool_schemas] == [
        "response"
    ]


@pytest.mark.asyncio
async def test_geo3k_multiturn_wrong_answer_returns_feedback_before_retry() -> None:
    env = Geo3KEnv(max_turns=2, extra_tools=["response"])

    await env.reset()
    first = await env.step([
        make_tool_call("response", {"text": "Answer: \\boxed{3}"}, call_id="call_0")
    ])
    second = await env.step([
        make_tool_call(
            "response",
            {"text": "Revising the diagonal.\nAnswer: \\boxed{\\sqrt{2}}"},
            call_id="call_1",
        )
    ])

    assert first.reward == 0.0
    assert first.terminated is False
    assert first.results[0].tool_call_id == "call_0"
    assert "revise your reasoning" in (first.results[0].text or "")
    assert "expected=" not in (first.results[0].text or "")
    assert first.info["attempt"] == 1
    assert first.info["max_turns"] == 2
    assert first.info["correct"] is False

    assert second.reward == 1.0
    assert second.terminated is True
    assert second.results[0].tool_call_id == "call_1"
    assert "; correct" in (second.results[0].text or "")
    assert "incorrect" not in (second.results[0].text or "")
    assert second.info["attempt"] == 2
    assert second.info["correct"] is True


@pytest.mark.asyncio
async def test_geo3k_accepts_one_response_per_attempt() -> None:
    env = Geo3KEnv(max_turns=2, extra_tools=["response"])

    await env.reset()
    result = await env.step([
        make_tool_call("response", {"text": "Answer: \\boxed{3}"}, call_id="call_0"),
        make_tool_call(
            "response",
            {"text": "Answer: \\boxed{\\sqrt{2}}"},
            call_id="call_1",
        ),
    ])

    assert result.reward == 0.0
    assert result.terminated is False
    assert result.info["attempt"] == 1
    assert result.info["correct"] is False
    assert "revise your reasoning" in (result.results[0].text or "")
    assert result.results[1].tool_call_id == "call_1"
    assert result.results[1].error == "only one response is accepted per attempt"


@pytest.mark.asyncio
async def test_geo3k_multiturn_terminates_when_attempts_are_exhausted() -> None:
    env = Geo3KEnv(max_turns=2, extra_tools=["response"])

    await env.reset()
    first = await env.step([
        make_tool_call("response", {"text": "Answer: \\boxed{3}"}, call_id="call_0")
    ])
    second = await env.step([
        make_tool_call("response", {"text": "Answer: \\boxed{4}"}, call_id="call_1")
    ])
    after_done = await env.step([
        make_tool_call("response", {"text": "Answer: \\boxed{\\sqrt{2}}"}, call_id="call_2")
    ])

    assert first.terminated is False
    assert second.reward == 0.0
    assert second.terminated is True
    assert "no attempts remain" in (second.results[0].text or "")
    assert after_done.terminated is True
    assert after_done.results[0].error == "task already finished"


@pytest.mark.asyncio
async def test_geo3k_agent_loop_continues_natural_language_answers() -> None:
    async def _generate(**kwargs):
        return {"response": "ignored by scripted adapter"}

    env = Geo3KEnv(max_turns=2)
    agent = AdapterBasedAgent(
        generate_fn=_generate,
        processor=_Geo3KFakeProcessor(),
        adapter=_Geo3KScriptedTextAdapter([
            "Answer: \\boxed{3}",
            "Answer: \\boxed{\\sqrt{2}}",
        ]),
    )

    sample = await agent.sample(env, max_steps=10)
    messages = sample.lite_sample.messages

    assert sample.terminated is True
    assert sample.truncated is False
    assert sample.episode_return == 1.0
    assert len(sample.steps) == 2
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert "revise your reasoning" in messages[2]["content"][0]["text"]
    assert "expected=" not in messages[2]["content"][0]["text"]
    assert not any(message.get("role") == "tool" for message in messages)


@pytest.mark.asyncio
async def test_geo3k_agent_loop_exhausts_natural_language_attempts() -> None:
    async def _generate(**kwargs):
        return {"response": "ignored by scripted adapter"}

    env = Geo3KEnv(max_turns=2)
    adapter = _Geo3KScriptedTextAdapter([
        "Answer: \\boxed{3}",
        "Answer: \\boxed{4}",
        "Answer: \\boxed{\\sqrt{2}}",
    ])
    agent = AdapterBasedAgent(
        generate_fn=_generate,
        processor=_Geo3KFakeProcessor(),
        adapter=adapter,
    )

    sample = await agent.sample(env, max_steps=10)

    assert sample.terminated is True
    assert sample.truncated is False
    assert sample.episode_return == 0.0
    assert len(sample.steps) == 2
    assert adapter._i == 2
    assert [message["role"] for message in sample.lite_sample.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert not any(message.get("role") == "tool" for message in sample.lite_sample.messages)


@pytest.mark.asyncio
async def test_geo3k_logger_records_runtime_response_action_without_tool_message(
    tmp_path,
) -> None:
    async def _generate(**kwargs):
        return {"response": "ignored by scripted adapter"}

    env = Geo3KEnv(max_turns=2)
    logger = TrajectoryLogger(tmp_path / "run")
    agent = AdapterBasedAgent(
        generate_fn=_generate,
        processor=_Geo3KFakeProcessor(),
        adapter=_Geo3KScriptedTextAdapter([
            "Answer: \\boxed{3}",
            "Answer: \\boxed{\\sqrt{2}}",
        ]),
    )

    sample = await agent.sample(env, max_steps=10, hooks=[logger])

    assert sample.terminated is True
    assert (tmp_path / "run" / "turn_0000").is_dir()
    assert (tmp_path / "run" / "turn_0001").is_dir()
    first_actions = json.loads(
        (tmp_path / "run" / "turn_0000" / "03_actions.json").read_text()
    )
    second_actions = json.loads(
        (tmp_path / "run" / "turn_0001" / "03_actions.json").read_text()
    )
    assert first_actions[EXECUTED_ACTIONS_INFO_KEY] == [
        {"call": "response", "args": {"text": "Answer: \\boxed{3}"}}
    ]
    assert second_actions[EXECUTED_ACTIONS_INFO_KEY] == [
        {"call": "response", "args": {"text": "Answer: \\boxed{\\sqrt{2}}"}}
    ]

    row = load_file_as_dataset(tmp_path / "run" / "trajectory.parquet")[0]
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    assert metadata["metadata_kind"] == "generic"
    assert metadata["dims"] == []
    assert "platform" not in metadata
    assert "task_type" not in metadata
    assert "valid_actions" not in metadata
    messages = coerce_messages(row["messages"])
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert not any(message.get("role") == "tool" for message in messages)


@pytest.mark.asyncio
async def test_geo3k_agent_loop_handles_explicit_response_tool() -> None:
    async def _generate(**kwargs):
        return {"response": "ignored by scripted adapter"}

    env = Geo3KEnv(max_turns=2, extra_tools=["response"])
    agent = AdapterBasedAgent(
        generate_fn=_generate,
        processor=_Geo3KFakeProcessor(),
        adapter=_Geo3KScriptedTextAdapter([
            [make_tool_call("response", {"text": "Answer: \\boxed{3}"})],
            [
                make_tool_call(
                    "response",
                    {"text": "Answer: \\boxed{\\sqrt{2}}"},
                )
            ],
        ]),
    )

    sample = await agent.sample(env, max_steps=10)

    assert sample.terminated is True
    assert sample.truncated is False
    assert sample.episode_return == 1.0
    assert len(sample.steps) == 2
    assert [message["role"] for message in sample.lite_sample.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert "expected=" not in sample.lite_sample.messages[2]["content"][0]["text"]
    assert "; correct" in sample.lite_sample.messages[-1]["content"][0]["text"]


@pytest.mark.asyncio
async def test_geo3k_agent_loop_stops_natural_language_answer_at_max_turns_one() -> None:
    async def _generate(**kwargs):
        return {"response": "ignored by scripted adapter"}

    env = Geo3KEnv(max_turns=1)
    agent = AdapterBasedAgent(
        generate_fn=_generate,
        processor=_Geo3KFakeProcessor(),
        adapter=_Geo3KScriptedTextAdapter([
            "Answer: \\boxed{3}",
            "Answer: \\boxed{\\sqrt{2}}",
        ]),
    )

    sample = await agent.sample(env, max_steps=10)
    messages = sample.lite_sample.messages

    assert sample.terminated is True
    assert sample.truncated is False
    assert sample.episode_return == 0.0
    assert len(sample.steps) == 1
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert "revise your reasoning" not in json.dumps(messages)
    assert not any(message.get("role") == "tool" for message in messages)


@pytest.mark.asyncio
async def test_geo3k_step_scores_final_answer_not_reasoning_numbers() -> None:
    env = Geo3KEnv(
        task_id="wrong_final",
        question="What number?",
        answer="3",
        extra_tools=["response"],
    )

    await env.reset()
    result = await env.step([
        make_tool_call(
            "response",
            {"text": "I considered 3, but the final answer is \\\\boxed{4}."},
            call_id="call_response",
        )
    ])

    assert result.reward == 0.0


@pytest.mark.asyncio
async def test_geo3k_step_scores_boxed_answer_with_nested_latex() -> None:
    env = Geo3KEnv(
        task_id="nested_latex",
        question="What is twice the square root of 221?",
        answer="2 \\sqrt { 221 }",
        extra_tools=["response"],
    )

    await env.reset()
    result = await env.step([
        make_tool_call(
            "response",
            {"text": "Answer: \\boxed{2 \\sqrt { 221 }}"},
            call_id="call_response",
        )
    ])

    assert result.reward == 1.0


@pytest.mark.asyncio
async def test_geo3k_source_task_reset_returns_dataset_image(tmp_path) -> None:
    source = _geo3k_source(tmp_path)
    register_geo3k(source_path=source)
    env = gym.make(f"{ENV_ID}@row_000000")

    observation = await env.reset()
    result = await env.step(make_no_tool_call_final_actions("Answer: \\boxed{5}"))

    assert observation.text == "A 3-4-5 right triangle has hypotenuse 5. What is its hypotenuse?"
    assert observation.image is not None
    with Image.open(BytesIO(observation.image)) as image:
        assert image.size == (16, 16)
    assert result.reward == 1.0


def test_geo3k_source_fingerprint_binds_prompt_rows_to_source(tmp_path) -> None:
    source_a = _single_row_geo3k_source(tmp_path / "a", answer="3")
    source_b = _single_row_geo3k_source(tmp_path / "b", answer="4")

    with pytest.raises(ValueError, match="Geo3K source fingerprint mismatch"):
        Geo3KEnv(
            task_id="row_000000",
            source_path=str(source_b),
            row_index=0,
            source_fingerprint=geo3k_source_fingerprint(source_a),
        )


def test_geo3k_generic_row_survives_parquet_roundtrip(tmp_path) -> None:
    env = Geo3KEnv()
    row = {
        "images": [],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": SMOKE_TASKS[TASK_ID].question}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Answer: \\boxed{\\sqrt{2}}"}],
            },
        ],
        "metadata": env.metadata.to_dict(),
    }
    path = tmp_path / "geo3k.parquet"

    write_records_to_parquet([row], path, json_fields=("messages", "metadata"))
    loaded = load_file_as_dataset(path)[0]
    sample = LiteSample.from_dict({
        "images": [],
        "messages": json.loads(loaded["messages"]),
        "metadata": json.loads(loaded["metadata"]),
    })

    assert isinstance(sample.metadata, LiteGenericMetadata)
    assert sample.metadata.dims == ()
    assert sample.metadata.extra_tool_schemas == []
