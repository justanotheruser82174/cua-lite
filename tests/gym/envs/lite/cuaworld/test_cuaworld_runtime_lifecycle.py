"""CUAWorld tests split from _cuaworld_support.py: runtime lifecycle."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from lite.core.tools import make_tool_call, make_tool_schema
from lite.gym.envs.lite.cuaworld.src import software
from lite.gym.envs.lite.cuaworld.src.adapter import run_cuaworld_post_start
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult
from tests.gym.envs.lite._cuaworld_support import (
    _cuaworld_root,
    _png_bytes,
    _write_local_materials_tree,
)


@pytest.mark.asyncio
async def test_close_finalizes_an_unfinished_episode(tmp_path, monkeypatch):
    from lite.gym.sandbox.base import SandboxBaseEnv

    calls = []

    async def evaluate(task, computer, actions, debug):
        calls.append(("evaluate", task, computer, actions, debug))
        return 0.0

    async def close_base(env):
        calls.append(("close", env))

    monkeypatch.setattr(SandboxBaseEnv, "close", close_base)
    env_cls = software._make_env_class(
        "lite.cuaworld.test",
        {"image": "test", "display": "1920x1080"},
    )
    env = object.__new__(env_cls)
    episode_dir = tmp_path / "episode"
    episode_dir.mkdir()
    task = object()
    computer = object()
    env._cuaworld_episode_dir = episode_dir
    env._cuaworld_finalized = False
    env._cuaworld_episode_task = task
    env._cuaworld_episode_evaluate_final_fn = evaluate
    env._task = task
    env._computer = computer
    env._evaluate_final_fn = evaluate
    env._debug = False

    await env.close()

    assert calls[0] == ("evaluate", task, computer, [], False)
    assert calls[1] == ("close", env)
    assert env._cuaworld_finalized is True
    assert env._cuaworld_episode_dir is None
    assert not episode_dir.exists()


@pytest.mark.asyncio
async def test_runtime_post_start_runs_once_per_boot(monkeypatch):
    from lite.gym.sandbox.base import SandboxBaseEnv

    calls = []
    first = SimpleNamespace()
    second = SimpleNamespace()
    computers = iter((first, second))

    async def boot_base(env):
        if env._computer is None:
            env._computer = next(computers)

    async def post_start(value, *, timeout):
        calls.append((value, timeout))

    monkeypatch.setattr(SandboxBaseEnv, "boot", boot_base)
    monkeypatch.setattr(software, "run_cuaworld_post_start", post_start)
    env_cls = software._make_env_class(
        "lite.cuaworld.test",
        {"image": "test", "display": "1920x1080"},
    )
    env = env_cls()

    await env.boot()
    await env.boot()
    env._computer = None
    await env.boot()

    assert calls == [
        (first, software.CFG.make_kwargs["reset_timeout"]),
        (second, software.CFG.make_kwargs["reset_timeout"]),
    ]
    build_hook = (_cuaworld_root() / "docker" / "run_hooks.sh").read_text()
    assert "read_hook post_start" not in build_hook
    assert (
        "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        in build_hook
    )


@pytest.mark.asyncio
async def test_repeated_reset_finalizes_previous_episode(tmp_path, monkeypatch):
    from lite.gym.sandbox import SandboxTaskConfig
    from lite.gym.sandbox.base import SandboxBaseEnv

    calls = []

    async def evaluate(task, computer, actions, debug):
        calls.append((task.task_id, computer, actions, debug))
        return 0.0

    old_computer = SimpleNamespace()
    new_computer = SimpleNamespace()
    closed = []
    lifecycle = []

    async def close_base(env):
        closed.append(env._computer)
        env._computer = None
        env._container_name = None

    async def reset_base(env):
        lifecycle.append("reset")
        if env._computer is None:
            env._computer = new_computer
        return LiteEnvObservation(image=_png_bytes())

    async def boot_env(env):
        lifecycle.append("boot")
        if env._computer is None:
            env._computer = new_computer

    def monotonic():
        lifecycle.append("timer")
        return 10.0

    monkeypatch.setattr(SandboxBaseEnv, "close", close_base)
    monkeypatch.setattr(SandboxBaseEnv, "reset", reset_base)
    monkeypatch.setattr(software.time, "monotonic", monotonic)
    env_cls = software._make_env_class(
        "lite.cuaworld.test",
        {"image": "test", "display": "1920x1080"},
    )
    task = SandboxTaskConfig(
        "second",
        "second",
        {},
        metadata={
            "others": {
                "upstream_env_id": "test_env",
                "upstream_task_id": "second",
            }
        },
        evaluate_final_fn=evaluate,
    )
    old_task = SandboxTaskConfig(
        "first",
        "first",
        {},
        evaluate_final_fn=evaluate,
    )
    env = env_cls()
    env.boot = lambda: boot_env(env)
    env.bind(task)
    env._computer = old_computer
    env._container_name = "test-container"
    old_episode = tmp_path / "old-episode"
    old_episode.mkdir()
    env._cuaworld_episode_dir = old_episode
    env._cuaworld_finalized = False
    env._cuaworld_episode_task = old_task
    env._cuaworld_episode_evaluate_final_fn = evaluate

    obs = await env.reset()

    assert obs.image
    assert calls == [("first", old_computer, [], False)]
    assert closed == [old_computer]
    assert env._computer is new_computer
    assert not old_episode.exists()
    assert env._cuaworld_episode_dir is not None
    assert env._cuaworld_episode_dir.exists()
    assert env._cuaworld_episode_task is task
    assert lifecycle == ["boot", "reset", "timer"]
    env._clear_episode()


@pytest.mark.asyncio
async def test_close_cleans_container_when_finalization_is_cancelled(
    tmp_path, monkeypatch
):
    from lite.gym.sandbox.base import SandboxBaseEnv

    closed = []

    async def cancel(*_args):
        raise asyncio.CancelledError

    async def close_base(env):
        closed.append(env)

    monkeypatch.setattr(SandboxBaseEnv, "close", close_base)
    env_cls = software._make_env_class(
        "lite.cuaworld.test",
        {"image": "test", "display": "1920x1080"},
    )
    env = object.__new__(env_cls)
    episode_dir = tmp_path / "cancelled"
    episode_dir.mkdir()
    env._cuaworld_episode_dir = episode_dir
    env._cuaworld_finalized = False
    env._cuaworld_episode_task = object()
    env._cuaworld_episode_evaluate_final_fn = cancel
    env._computer = object()
    env._debug = False

    with pytest.raises(asyncio.CancelledError):
        await env.close()

    assert closed == [env]
    assert env._cuaworld_episode_dir is None
    assert not episode_dir.exists()


@pytest.mark.asyncio
async def test_cancelled_terminal_step_does_not_repeat_final_verifier_on_close(
    tmp_path, monkeypatch
):
    from lite.gym.sandbox.base import SandboxBaseEnv
    from lite.gym.sandbox.types import SandboxTaskConfig

    evaluations = []
    closed = []

    async def evaluate(*_args):
        evaluations.append("evaluate")
        raise asyncio.CancelledError

    async def step_base(env, _actions):
        await env._evaluate_final_fn(
            env._task,
            env._computer,
            [],
            env._debug,
        )

    async def close_base(env):
        closed.append(env)

    monkeypatch.setattr(SandboxBaseEnv, "step", step_base)
    monkeypatch.setattr(SandboxBaseEnv, "close", close_base)
    env_cls = software._make_env_class(
        "lite.cuaworld.test",
        {"image": "test", "display": "1920x1080"},
    )
    env = object.__new__(env_cls)
    episode_dir = tmp_path / "terminal-cancelled"
    episode_dir.mkdir()
    env._cuaworld_episode_dir = episode_dir
    env._cuaworld_finalized = False
    env._cuaworld_episode_task = object()
    env._cuaworld_episode_evaluate_final_fn = evaluate
    env._evaluate_final_fn = evaluate
    env._computer = SimpleNamespace()
    env._task = SandboxTaskConfig("test", "test", {})
    env._debug = False
    env._step_count = 0

    with pytest.raises(asyncio.CancelledError):
        await env.step([make_tool_call("terminate", {})])
    await env.close()

    assert evaluations == ["evaluate"]
    assert closed == [env]
    assert env._cuaworld_episode_dir is None
    assert not episode_dir.exists()


@pytest.mark.asyncio
async def test_terminal_step_calls_final_evaluator_through_base_signature(tmp_path):
    from lite.gym.sandbox import SandboxTaskConfig

    calls = []

    async def evaluate(task, computer, actions, debug):
        calls.append((task.task_id, actions, debug))
        return 0.5, {"source": "final"}

    class Interface:
        async def screenshot(self):
            return _png_bytes()

    env_cls = software._make_env_class(
        "lite.cuaworld.test",
        {"image": "test", "display": "1920x1080"},
    )
    task = SandboxTaskConfig(
        "terminal",
        "terminal",
        {},
        extra_tool_schemas=[make_tool_schema("terminate", parameters={})],
        evaluate_final_fn=evaluate,
    )
    env = object.__new__(env_cls)
    env._cuaworld_episode_dir = tmp_path
    env._cuaworld_finalized = False
    env._cuaworld_episode_task = task
    env._cuaworld_episode_evaluate_final_fn = evaluate
    env._cuaworld_episode_started_at = 0.0
    env._cuaworld_timeout_sec = 600.0
    env._evaluate_final_fn = evaluate
    env._evaluate_step_fn = None
    # Normally set in SandboxBaseEnv.__init__; step()'s screen-size fallback reads it.
    env._display_resolution = (1920, 1080)
    env._computer = SimpleNamespace(
        interface=Interface(),
        _lite_screen_size=(1920, 1080),
    )
    env._task = task
    env._debug = False
    env._step_count = 0
    env._max_steps = 80
    env._post_action_delay = 0.0

    result = await env.step([make_tool_call("terminate", {})])

    assert result.terminated is True
    assert result.reward == 0.5
    assert result.info["eval"] == {"source": "final"}
    assert calls == [
        (
            "terminal",
            [{"name": "terminate", "arguments": {}}],
            False,
        )
    ]


@pytest.mark.asyncio
async def test_episode_timeout_is_local_to_cuaworld(tmp_path, monkeypatch):
    from lite.gym.sandbox import SandboxTaskConfig
    from lite.gym.sandbox.base import SandboxBaseEnv

    calls = []

    async def evaluate(task, computer, actions, debug):
        calls.append((task.task_id, actions, debug))
        return 0.75, {"source": "timeout"}

    async def step_base(_env, _actions):
        return LiteEnvStepResult(
            reward=0.0,
            terminated=False,
            truncated=False,
            info={},
        )

    monkeypatch.setattr(SandboxBaseEnv, "step", step_base)
    monkeypatch.setattr(
        software.time,
        "monotonic",
        lambda: 12.0,
    )
    env_cls = software._make_env_class(
        "lite.cuaworld.test",
        {"image": "test", "display": "1920x1080"},
    )
    task = SandboxTaskConfig(
        "timed",
        "timed",
        {},
        metadata={"timeout_sec": 1},
        evaluate_final_fn=evaluate,
    )
    env = env_cls(task=task)
    env._computer = SimpleNamespace()
    env._step_count = 0
    env._debug = False
    env._cuaworld_episode_dir = tmp_path / "timed-episode"
    env._cuaworld_episode_dir.mkdir()
    env._cuaworld_episode_started_at = 10.0

    result = await env.step([
        make_tool_call(
            "computer",
            {"actions": [{"action": "click", "coordinate": [500, 500]}]},
            call_id="call_0",
        )
    ])

    assert result.truncated is True
    assert result.reward == 0.75
    assert result.info == {
        "termination_reason": "timeout",
        "eval": {"source": "timeout"},
    }
    assert calls == [
        ("timed", [{"name": "click", "arguments": {"coordinate": [500, 500]}}], False)
    ]
    env._clear_episode()


@pytest.mark.asyncio
async def test_post_start_bridge_preserves_upstream_warning_behavior():
    class FailingInterface:
        async def run_command(self, command, timeout=None):
            assert "/workspace/env.json" in command
            assert "/usr/bin/python3" in command
            assert (
                "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                in command
            )
            assert '[ ! -f "$hook" ] || chmod 755 "$hook"' in command
            assert "bash /tmp/cuaworld_post_start.sh" in command
            assert timeout == 1800.0
            return SimpleNamespace(
                returncode=17,
                stdout="",
                stderr="upstream setup failed",
            )

    await run_cuaworld_post_start(
        SimpleNamespace(interface=FailingInterface())
    )


def test_bind_refreshes_task_callbacks():
    from lite.gym.sandbox import SandboxTaskConfig

    async def setup_a(*_args):
        return None

    async def setup_b(*_args):
        return None

    async def eval_a(*_args):
        return 1.0

    async def eval_b(*_args):
        return 2.0

    env_cls = software._make_env_class(
        "lite.cuaworld.test",
        {"image": "test", "display": "1920x1080"},
    )
    task_a = SandboxTaskConfig(
        "a", "a", {}, setup_fn=setup_a, evaluate_final_fn=eval_a
    )
    task_b = SandboxTaskConfig(
        "b", "b", {}, setup_fn=setup_b, evaluate_final_fn=eval_b
    )
    env = env_cls(task=task_a)
    assert env._setup_fn is setup_a
    assert env._evaluate_final_fn is eval_a
    env.bind(task_b)
    assert env._setup_fn is setup_b
    assert env._evaluate_final_fn is eval_b


def test_services_ensure_uses_image_freshness(monkeypatch):
    """CUAWorld env-server ensure must reject stale images, not just missing ones."""
    from lite.gym.envs.lite.cuaworld.src import image_spec
    from lite.gym.utils.backend.freshness import image_for

    seen = {}

    def fake_require_image_present(image):
        seen["tag"] = image.tag
        seen["install"] = image.install
        seen["extra_hash_inputs"] = image.extra_hash_inputs

    monkeypatch.setattr("lite.gym.utils.backend.docker.require_image_present",
                        fake_require_image_present)
    monkeypatch.setitem(
        image_spec._REGISTERED_UPSTREAM_ENVS, "swalpha", "swalpha_env"
    )

    svc = software.CUAWorldServices(
        "cua-lite/lite.cuaworld.swalpha:latest",
        "uv run --no-sync bash lite/gym/envs/lite/cuaworld/scripts/install.sh build swalpha",
        lambda: None,
    )
    svc.ensure("lite.cuaworld.swalpha")

    assert seen["tag"] == "cua-lite/lite.cuaworld.swalpha:latest"
    assert seen["install"].endswith("install.sh build swalpha")
    assert seen["extra_hash_inputs"] == (
        "software=swalpha",
        "upstream_env=swalpha_env",
    )
    assert (
        "lite/gym/envs/lite/cuaworld/data/assets.lock.yaml"
        in image_for("lite.cuaworld.swalpha").sources
    )


def test_image_freshness_extracts_software_after_lite_namespace(monkeypatch):
    from lite.gym.envs.lite.cuaworld.src import image_spec
    from lite.gym.utils.backend.freshness import image_for

    monkeypatch.setitem(
        image_spec._REGISTERED_UPSTREAM_ENVS, "swalpha", "swalpha_env"
    )
    image = image_for("lite.cuaworld.swalpha")
    assert image.tag == "cua-lite/lite.cuaworld.swalpha:latest"
    assert image.extra_hash_inputs == (
        "software=swalpha",
        "upstream_env=swalpha_env",
    )
    assert "lite/gym/envs/lite/cuaworld/data/assets.lock.yaml" in image.sources
    assert image.install.endswith("install.sh build swalpha")


def test_image_freshness_does_not_accept_upstream_material_override():
    from lite.gym.utils.backend.freshness import image_for

    image = image_for("lite.cuaworld.freecad")
    assert image.install.endswith("install.sh build freecad")
    with pytest.raises(TypeError, match="upstream_env"):
        image_for("lite.cuaworld.freecad", upstream_env="freecad_env")


def test_local_materials_mode_hashes_checkout_contents(tmp_path, monkeypatch):
    from lite.gym.utils.backend.freshness import image_for

    remote = image_for("lite.cuaworld.pymol").src_hash()
    materials = tmp_path / "pymol_env"
    _write_local_materials_tree(materials, "one")
    monkeypatch.setenv("LITE_CUAWORLD_MATERIALS_REPO", str(tmp_path))
    first = image_for("lite.cuaworld.pymol").src_hash()
    payload = materials / "env.json"
    payload.write_text("two")
    second = image_for("lite.cuaworld.pymol").src_hash()
    assert first != second
    assert first != remote


def test_production_upstream_aliases_drive_image_freshness():
    from lite.gym.envs.lite.cuaworld.src import image_spec
    from lite.gym.utils.backend.freshness import image_for

    assert image_spec.upstream_env_for_software("freecad") == "freecad_envb"
    assert (
        image_spec.upstream_env_for_software("knime")
        == "knime_analytics_platform_env"
    )
    assert "upstream_env=freecad_envb" in image_for(
        "lite.cuaworld.freecad"
    ).extra_hash_inputs
    assert "upstream_env=knime_analytics_platform_env" in image_for(
        "lite.cuaworld.knime"
    ).extra_hash_inputs


@pytest.mark.parametrize(
    ("software", "upstream_env", "wrong_env"),
    [
        ("freecad", "freecad_envb", "freecad_env"),
        ("knime", "knime_analytics_platform_env", "knime_env"),
    ],
)
def test_local_materials_mode_uses_production_upstream_aliases(
    tmp_path, monkeypatch, software, upstream_env, wrong_env
):
    from lite.gym.envs.lite.cuaworld.src import image_spec
    from lite.gym.utils.backend.freshness import image_for

    _write_local_materials_tree(tmp_path / wrong_env, "wrong")
    monkeypatch.setenv("LITE_CUAWORLD_MATERIALS_REPO", str(tmp_path))
    with pytest.raises(FileNotFoundError, match=upstream_env):
        image_for(f"lite.cuaworld.{software}")

    _write_local_materials_tree(tmp_path / upstream_env, "right")
    first = image_for(f"lite.cuaworld.{software}").src_hash()
    first_materials = image_spec.local_materials_identity(upstream_env)
    (tmp_path / upstream_env / "scripts" / ".gitignore").write_text("*.tmp\n")
    second = image_for(f"lite.cuaworld.{software}").src_hash()
    (tmp_path / upstream_env / "tasks" / "verifier.py").write_text("changed\n")
    third = image_for(f"lite.cuaworld.{software}").src_hash()
    third_materials = image_spec.local_materials_identity(upstream_env)

    assert first != second
    assert second == third
    assert first_materials != third_materials
