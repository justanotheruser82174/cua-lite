"""Coverage for the ``register_tasks(env_class=SandboxBaseEnv)`` direct-
mode path that the oracle validate script
(`devs/envs/lite.osworld/validate/oracle/validate.py`) uses.

Why this test exists
====================

This path differs from ``gym.make("lite.osworld@...")`` in one
load-bearing way: ``LiteOsworldEnv.__init__`` defaults
``computer_config=_COMPUTER_CONFIG`` before binding, so ``_computer_config`` is
always non-None by the time ``bind()`` runs.
The bare ``SandboxBaseEnv`` factory ``register_tasks`` builds when no
``env_class`` is passed has NO such default — it expects ``bind()`` to
seed ``_computer_config`` from ``task.computer`` on first call.

Before the bind split, that seeding was a one-line "if X is None" branch in
:meth:`SandboxBaseEnv.bind`. A refactor initially deleted that branch as a
"first bind vs re-bind" smell. The lite.osworld unit suite (100+ tests) didn't
catch the regression because every test goes through ``LiteOsworldEnv`` (which
DOES pre-stamp); only the validate path uses bare ``SandboxBaseEnv``. The
regression surfaced in a manual ``train.synth`` / ``train.perturb`` oracle
validate run after the refactor landed.

This test exercises the bare-Sandbox register_tasks path with NO
docker boot, asserting that the bind body seeds ``_computer_config``
from the task's ``computer`` field. Future refactors that re-delete
the cold-fallback now fail CI here instead of silently breaking
oracle validate.

Run:
    uv run pytest tests/gym/sandbox/test_register_tasks_path.py -v
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

import lite.gym as gym
from lite.gym.errors import EnvDepsMissingError
from lite.gym.registry import _clear_env_registration, _splits
from lite.gym.sandbox import SandboxBaseEnv, SandboxTaskConfig, register_tasks

_BARE_TASK = SandboxTaskConfig(
    task_id="cb_register_tasks_smoke",
    instruction="smoke",
    platform="desktop",
    computer={
        # The bare-Sandbox path treats this dict as authoritative
        # baked container shape. ``image`` is what oracle validate sets
        # to ``cua-lite/lite.osworld:latest`` (validate.py:_IMAGE) — the
        # principle is the same: validate-time tasks ship their own
        # container shape and the env class doesn't override it.
        "image": "lite.placeholder:latest",
        "display": "1920x1080",
        "memory": "4GB",
        "cpu": "1",
        "timeout": 120,
    },
    max_steps=15,
    metadata={"others": {"domain": "smoke"}},
)


@pytest.fixture(autouse=True)
def _clear_test_envs():
    yield
    for env_id in list(_splits):
        if env_id.startswith(("_test_cb_", "_test_mk_")):
            _clear_env_registration(env_id)


def test_bare_sandbox_seeds_computer_config_from_task() -> None:
    """register_tasks default (env_class=SandboxBaseEnv) + a task with a
    populated ``computer`` field: bind() must seed _computer_config
    from task.computer so boot() has something to spawn.

    This is the exact code path oracle validate uses (see
    devs/envs/lite.osworld/validate/oracle/validate.py). Without the
    seeding the validate script's first reset() raises
    ``RuntimeError: boot() called before bind() or computer_config``.
    """
    env_id = "_test_cb_bare_seeds"
    register_tasks(env_id, {"train": [_BARE_TASK]})
    env = gym.make(f"{env_id}@{_BARE_TASK.task_id}")
    inner = env.unwrapped

    # Bare Sandbox construction leaves _computer_config None — the
    # constant-default a subclass like LiteOsworldEnv would inject is absent
    # here. But construction always calls bind() once with the task kwarg, so
    # by the time we observe `env`, _computer_config should already be seeded.
    assert inner._computer_config is not None, (
        "bare Sandbox bind() failed to seed _computer_config from "
        "task.computer — oracle validate path will crash on boot()"
    )
    assert inner._computer_config["image"] == "lite.placeholder:latest"
    # Display gets stamped from env's _display_resolution (default
    # 1920x1080) — same string, but the assertion documents the path.
    assert inner._computer_config["display"] == "1920x1080"
    assert inner._task is not None
    assert inner._task.task_id == _BARE_TASK.task_id


@pytest.mark.asyncio
async def test_bare_sandbox_image_kwarg_overrides_boot_config(monkeypatch) -> None:
    """``--env-kwargs {"image": ...}`` reaches the actual spawn config.

    The replay script registers a temporary bare Sandbox env, then forwards
    env_kwargs to ``gym.make``. The task still carries its original computer
    image, but ``SandboxBaseEnv.boot`` must use the top-level baked ``image``
    kwarg when constructing the Docker provisioner.
    """
    from lite.gym.sandbox import base as sandbox_base

    captured: dict[str, object] = {}

    def live_snapshot() -> set[str]:
        with sandbox_base._LIVE_LOCK:
            return set(sandbox_base._LIVE_CONTAINERS)

    class FakeProvisioner:
        def __init__(self, name, cfg, **kwargs):
            captured["name"] = name
            captured["cfg"] = dict(cfg)
            captured["kwargs"] = kwargs

        async def run(self):
            captured["ran"] = True

    class FakeComputer:
        async def stop(self):
            captured["stopped"] = True

    @asynccontextmanager
    async def fake_slot():
        captured["slot"] = captured.get("slot", 0) + 1
        yield

    async def fake_attach(name, **kwargs):
        captured["attached"] = (name, kwargs)
        return FakeComputer()

    async def fake_rm(name, *, timeout, label):
        captured["removed"] = (name, timeout, label)

    monkeypatch.setattr(sandbox_base, "DockerProvisioner", FakeProvisioner)
    monkeypatch.setattr(sandbox_base, "attach", fake_attach)
    monkeypatch.setattr(sandbox_base, "docker_create_slot_async", fake_slot)
    monkeypatch.setattr(sandbox_base, "docker_rm_f_async", fake_rm)

    before_live = live_snapshot()
    env_id = "_test_cb_bare_image_override"
    register_tasks(env_id, {"train": [_BARE_TASK]})
    env = gym.make(
        f"{env_id}@{_BARE_TASK.task_id}",
        image="lite.private:mine",
    )
    inner = env.unwrapped

    assert inner._computer_config["image"] == "lite.placeholder:latest"

    await inner.boot()
    try:
        assert captured["slot"] == 1
        assert captured["ran"] is True
        assert captured["cfg"]["image"] == "lite.private:mine"
        assert captured["cfg"]["display"] == "1920x1080"
        assert captured["attached"][1]["exec_user"] == "user"
        assert inner._computer_config["image"] == "lite.placeholder:latest"
    finally:
        await inner.close()
        assert live_snapshot() == before_live


@pytest.mark.asyncio
async def test_bare_sandbox_unknown_image_provider_does_not_block_generic_image(
    monkeypatch,
) -> None:
    """Generic bare-sandbox tasks can use private/non-CUA-Lite images."""
    from lite.gym.sandbox import base as sandbox_base
    from lite.gym.utils.backend import freshness

    captured: dict[str, object] = {}

    class FakeProvisioner:
        def __init__(self, name, cfg, **_kwargs):
            captured["cfg"] = dict(cfg)

        async def run(self):
            captured["ran"] = True

    class FakeComputer:
        async def stop(self):
            pass

    @asynccontextmanager
    async def fake_slot():
        yield

    async def fake_attach(_name, **_kwargs):
        return FakeComputer()

    def unknown_provider(env_id, *, tag=None):
        captured["provider_call"] = (env_id, tag)
        raise freshness.UnknownImageFreshnessProvider(env_id)

    monkeypatch.setattr(sandbox_base, "DockerProvisioner", FakeProvisioner)
    monkeypatch.setattr(sandbox_base, "attach", fake_attach)
    monkeypatch.setattr(sandbox_base, "docker_create_slot_async", fake_slot)
    monkeypatch.setattr(freshness, "image_for", unknown_provider)

    env_id = "_test_cb_unknown_image_provider"
    register_tasks(env_id, {"train": [_BARE_TASK]})
    env = gym.make(f"{env_id}@{_BARE_TASK.task_id}", image="private/image:tag")
    inner = env.unwrapped

    await inner.boot()
    try:
        assert captured["provider_call"] == (env_id, "private/image:tag")
        assert captured["ran"] is True
        assert captured["cfg"]["image"] == "private/image:tag"
    finally:
        await inner.close()


@pytest.mark.asyncio
async def test_direct_bare_sandbox_private_image_skips_env_freshness_provider(
    monkeypatch,
) -> None:
    """Direct SandboxBaseEnv(task=...) has no env_id, so private images are generic."""
    from lite.gym.sandbox import base as sandbox_base
    from lite.gym.utils.backend import freshness

    captured: dict[str, object] = {}

    class FakeProvisioner:
        def __init__(self, name, cfg, **_kwargs):
            captured["cfg"] = dict(cfg)

        async def run(self):
            captured["ran"] = True

    class FakeComputer:
        async def stop(self):
            captured["stopped"] = True

    async def fake_attach(_name, **_kwargs):
        return FakeComputer()

    async def fake_rm(name, *, timeout, label):
        captured["removed"] = (name, timeout, label)

    def provider_must_not_be_called(*_args, **_kwargs):
        raise AssertionError("direct generic sandbox should not query env freshness")

    monkeypatch.setattr(sandbox_base, "DockerProvisioner", FakeProvisioner)
    monkeypatch.setattr(sandbox_base, "attach", fake_attach)
    monkeypatch.setattr(sandbox_base, "docker_rm_f_async", fake_rm)
    monkeypatch.setattr(freshness, "image_for", provider_must_not_be_called)

    env = SandboxBaseEnv(task=_BARE_TASK, image="private/image:tag")

    await env.boot()
    try:
        assert captured["ran"] is True
        assert captured["cfg"]["image"] == "private/image:tag"
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_bare_sandbox_provider_errors_fail_closed(monkeypatch) -> None:
    """Malformed env-local providers must not be swallowed as generic images."""
    from lite.gym.sandbox import base as sandbox_base
    from lite.gym.utils.backend import freshness

    captured: dict[str, object] = {}

    async def fake_rm(name, *, timeout, label):
        captured["removed"] = (name, timeout, label)

    def broken_provider(_env_id, *, tag=None):
        captured["tag"] = tag
        raise ValueError("bad provider")

    monkeypatch.setattr(sandbox_base, "docker_rm_f_async", fake_rm)
    monkeypatch.setattr(freshness, "image_for", broken_provider)

    env_id = "_test_cb_bad_image_provider"
    register_tasks(env_id, {"train": [_BARE_TASK]})
    env = gym.make(f"{env_id}@{_BARE_TASK.task_id}", image="private/image:tag")
    inner = env.unwrapped

    with pytest.raises(EnvDepsMissingError, match="image freshness inputs"):
        await inner.boot()
    assert captured["tag"] == "private/image:tag"
    assert captured["removed"][2] == "sandbox"


def test_bare_sandbox_rebind_does_not_clobber_computer_config() -> None:
    """Second bind() must NOT overwrite
    _computer_config — the container's display+image are baked at
    ``docker run`` time and rewriting the dict would let a stale
    image/display value silently drift into log lines and the
    legacy baked-shape metadata.
    """
    env_id = "_test_cb_bare_no_clobber"
    register_tasks(env_id, {"train": [_BARE_TASK]})
    env = gym.make(f"{env_id}@{_BARE_TASK.task_id}")
    inner = env.unwrapped

    original_image = inner._computer_config["image"]
    original_display = inner._computer_config["display"]

    # Bind to a task with a DIFFERENT computer shape. A backend may already
    # be booted with the original shape, so the env must keep
    # _computer_config at the original — the new task's ``computer`` field
    # is only consulted for stamping the display onto the returned `_task`,
    # not for overwriting the env's _computer_config.
    different_task = SandboxTaskConfig(
        task_id="cb_register_tasks_smoke_v2",
        instruction="smoke v2",
        platform="desktop",
        computer={
            "image": "lite.totally-different:latest",
            "display": "640x480",
            "memory": "1GB",
            "cpu": "1",
            "timeout": 60,
        },
        max_steps=15,
        metadata={"others": {"domain": "smoke"}},
    )
    inner.bind(different_task)

    assert inner._computer_config["image"] == original_image, (
        "rebind clobbered _computer_config[image] — but the running "
        "container's image is immutable; the dict must keep the "
        "original or downstream log lines / matcher state will drift"
    )
    assert inner._computer_config["display"] == original_display


# ---------------------------------------------------------------------------
# make_kwargs layering: env-wide default.yaml ``make_kwargs`` (layer 2) vs the
# per-task register kwarg (layer 3) vs gym.make() (layer 4), resolved through
# the SAME sandbox register path real envs use (register_tasks -> _register_one).
# ---------------------------------------------------------------------------


def _resolved_step_timeout(env):
    """Walk the wrapper chain and return the StepTimeoutWrapper's step_timeout.

    Returns ``None`` if no StepTimeoutWrapper was applied — which is exactly the
    failure signature of the None-clobber bug (make() skips the wrapper when the
    resolved step_timeout is None), so the regression test can assert on it.
    """
    from lite.gym.wrappers import StepTimeoutWrapper

    cur = env
    for _ in range(12):
        if isinstance(cur, StepTimeoutWrapper):
            return cur.step_timeout
        cur = getattr(cur, "env", None)
        if cur is None:
            break
    return None


def _make_kwargs_task(task_id: str) -> SandboxTaskConfig:
    return SandboxTaskConfig(
        task_id=task_id,
        instruction="smoke",
        platform="desktop",
        computer={
            "image": "lite.placeholder:latest",
            "display": "1920x1080",
            "memory": "4GB",
            "cpu": "1",
            "timeout": 120,
        },
        max_steps=15,
        metadata={"others": {"domain": "smoke"}},
    )


def test_env_make_kwargs_applies_and_none_does_not_clobber() -> None:
    """Regression for the lite.osworld None-clobber bug.

    ``register_tasks`` routes through ``_register_one``, whose ``step_timeout``
    param defaults to None. That None must NOT be written into ``spec.kwargs`` —
    otherwise it overrides the env-wide ``make_kwargs`` default at make()-merge
    time, leaving step_timeout=None and DISABLING the StepTimeoutWrapper entirely.
    With the env-wide default declared via ``set_env_make_kwargs`` (the uniform
    default.yaml path), gym.make must resolve it (not None, not the 120 fallback).
    """
    from lite.gym.registry import _env_make_kwargs, _specs, registry

    env_id = "_test_mk_none_clobber"
    task = _make_kwargs_task("cb_mk_none_clobber")
    key = f"{env_id}@{task.task_id}"
    try:
        register_tasks(env_id, {"train": [task]})  # _register_one(step_timeout=None)
        # No None should have leaked into spec.kwargs.
        assert "step_timeout" not in _specs[key].kwargs, (
            "sandbox _register_one wrote a None step_timeout into spec.kwargs — "
            "it would clobber the env-wide make_kwargs default at make()-merge"
        )
        registry.set_env_make_kwargs(env_id, {"step_timeout": 123.0})  # layer 2
        env = gym.make(key)
        assert _resolved_step_timeout(env) == 123.0, (
            "env-wide make_kwargs step_timeout was clobbered (or the wrapper was "
            "skipped) — this is the lite.osworld bug"
        )
    finally:
        _specs.pop(key, None)
        _env_make_kwargs.pop(env_id, None)


def test_make_kwargs_precedence_env_default_register_caller() -> None:
    """Layered precedence: env_make_kwargs(2) < register spec.kwargs(3) < gym.make(4).

    Also asserts the layer-1 fallback (make()'s own 120.0 default) applies when no
    layer sets step_timeout, and that set_env_make_kwargs filters to CARRIED keys.
    """
    from lite.gym.registry import _env_make_kwargs, _specs, registry

    # --- layer 1: nothing set anywhere -> make() default 120.0 ---------------
    env_a = "_test_mk_layer1"
    task_a = _make_kwargs_task("cb_mk_layer1")
    key_a = f"{env_a}@{task_a.task_id}"
    # --- layer 2 only: env-wide default applies ------------------------------
    env_b = "_test_mk_layer2"
    task_b = _make_kwargs_task("cb_mk_layer2")
    key_b = f"{env_b}@{task_b.task_id}"
    # --- layer 3: per-task register kwarg beats env-wide default --------------
    env_c = "_test_mk_layer3"
    task_c = _make_kwargs_task("cb_mk_layer3")
    key_c = f"{env_c}@{task_c.task_id}"
    try:
        register_tasks(env_a, {"train": [task_a]})
        assert _resolved_step_timeout(gym.make(key_a)) == 120.0

        register_tasks(env_b, {"train": [task_b]})
        registry.set_env_make_kwargs(env_b, {"step_timeout": 200.0})
        assert _resolved_step_timeout(gym.make(key_b)) == 200.0
        # layer 4 (caller) beats layer 2
        assert _resolved_step_timeout(gym.make(key_b, step_timeout=55.0)) == 55.0

        register_tasks(env_c, {"train": [task_c]}, step_timeout=90.0)  # layer 3
        registry.set_env_make_kwargs(env_c, {"step_timeout": 200.0})  # layer 2
        assert _resolved_step_timeout(gym.make(key_c)) == 90.0
        # layer 4 still beats layer 3
        assert _resolved_step_timeout(gym.make(key_c, step_timeout=55.0)) == 55.0

        # set_env_make_kwargs filters to CARRIED_SPEC_KWARGS (drops unknown keys).
        registry.set_env_make_kwargs(env_b, {"step_timeout": 200.0, "bogus": 1})
        assert registry.env_make_kwargs(env_b) == {"step_timeout": 200.0}
    finally:
        for k in (key_a, key_b, key_c):
            _specs.pop(k, None)
        for e in (env_a, env_b, env_c):
            _env_make_kwargs.pop(e, None)


def test_bare_sandbox_bind_accepts_universal_seed_kwarg() -> None:
    """SandboxBaseEnv.bind MUST accept the universal harness-injected ``seed`` soft kwarg
    (group_shared_seed / --env-kwargs seed=). lite.demo uses this base bind directly; before
    the fix a passed seed raised ``bind() got an unexpected keyword argument 'seed'`` →
    EnvServerPoolable.__init__ forwarded it → TypeError → env unconstructable. Deterministic
    sandbox tasks ignore the value, but bind must tolerate it (stashed on _seed)."""
    env_id = "_test_cb_bare_seed_kwarg"
    register_tasks(env_id, {"train": [_BARE_TASK]})
    # Passing seed through gym.make → constructor → bind(seed=...) must NOT raise.
    env = gym.make(f"{env_id}@{_BARE_TASK.task_id}", seed=42)
    assert env.unwrapped._seed == 42
