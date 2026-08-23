"""Cross-task backend reuse contract tests.

Exercises EnvServerPoolable.reset_with_recycle's four invariants (recycle
cap, first-reset gate, tear-down-BEFORE-pristine, cold-spawn-skip) plus the
declare-rollback ⇒ explicit-yaml-cap finalizer, with stub envs (no docker).

Run: uv run pytest tests/gym/services/test_services_reuse.py
"""

from __future__ import annotations

import asyncio

import pytest

from lite.gym.services import EnvServerPoolable


class _Recycler(EnvServerPoolable):
    """Stub reuse env: records the hook call order."""

    def __init__(self, *, cap: int = 0):
        self.events: list[str] = []
        self._max_resets_per_container = cap
        self._pristine_ok = True
        super().__init__()

    async def boot(self):
        self.events.append("boot")
        self._current_container = object()

    async def destroy_backend(self):
        self.events.append("destroy")
        self._current_container = None

    async def tear_down_task(self):
        self.events.append("tear_down")

    async def reset_to_pristine(self) -> bool:
        self.events.append("pristine")
        return self._pristine_ok

    async def init_task(self):
        self.events.append("init")

    async def close(self):  # pragma: no cover - unused
        pass

    async def reset(self):  # pragma: no cover - unused
        raise NotImplementedError

    async def step(self, actions):  # pragma: no cover - unused
        raise NotImplementedError

    def _runtime_metadata(self):  # pragma: no cover - unused
        raise NotImplementedError


def _run(coro):
    return asyncio.run(coro)


def test_cold_spawn_skips_teardown_and_pristine():
    env = _Recycler(cap=3)
    cold = _run(env.reset_with_recycle())
    assert cold is True
    assert env.events == ["boot", "init"], "cold spawn must skip tear_down + pristine"
    assert env._recycle_count == 0, "cold spawn is not a reuse — budget starts at 0"
    assert env._recycle_first_done is True


def test_in_place_reuse_order_teardown_before_pristine():
    env = _Recycler(cap=3)
    _run(env.reset_with_recycle())  # cold
    env.events.clear()
    cold = _run(env.reset_with_recycle())  # in-place reuse
    assert cold is False
    assert env.events == ["tear_down", "pristine", "init"], (
        "tear-down must run BEFORE pristine (androidworld's documented "
        "reward=0 regression when reversed)"
    )
    assert env._recycle_count == 1, "one successful in-place reuse"


def test_cap_reached_recycles_and_restarts_budget():
    env = _Recycler(cap=1)
    _run(env.reset_with_recycle())  # cold (0 reuses)
    _run(env.reset_with_recycle())  # reuse #1 (== cap)
    env.events.clear()
    cold = _run(env.reset_with_recycle())  # cap reached → recycle
    assert cold is True
    assert env.events == ["destroy", "boot", "init"], "cap → destroy + fresh boot"
    assert env._recycle_count == 0, "fresh backend's reuse budget restarts at 0"


def test_cap_recycle_logs_the_event(caplog):
    """Audit item: the cap-triggered destroy must stay observable — the
    hand-rolled copies each logged it; the framework owns the log now."""
    import logging

    env = _Recycler(cap=1)
    _run(env.reset_with_recycle())  # cold
    _run(env.reset_with_recycle())  # reuse #1 (== cap)
    with caplog.at_level(logging.INFO, logger="lite.gym.services"):
        _run(env.reset_with_recycle())  # cap reached → recycle
    assert any("recycle cap reached" in r.message for r in caplog.records), (
        "cap-triggered destroy must emit the recycle log"
    )


def test_first_reset_gate_protects_prebooted_backend():
    """A prebooted backend with first_done=False must NOT be
    destroyed by the first reset even at cap 0."""
    env = _Recycler(cap=0)
    _run(env.boot())
    env.events.clear()
    cold = _run(env.reset_with_recycle())
    assert cold is False
    assert "destroy" not in env.events, "first reset on a prebooted backend never recycles"
    assert env.events == ["tear_down", "pristine", "init"]
    assert env._recycle_count == 1


def test_false_pristine_destroys_and_reboots():
    env = _Recycler(cap=5)
    _run(env.reset_with_recycle())  # cold
    env._pristine_ok = False
    env.events.clear()
    cold = _run(env.reset_with_recycle())
    assert cold is True
    assert env.events == ["tear_down", "pristine", "destroy", "boot", "init"], (
        "a failed rollback must never leave a dirty backend in play"
    )
    assert env._recycle_count == 0, "failed rollback → fresh backend, budget 0"


def test_unwired_cap_fails_loud_at_first_recycle(monkeypatch):
    """The finalizer proves the YAML key exists, but only __init__ can
    wire the instance attr — an unwired cap must raise, never silently
    degrade to cap 0 (the cold-boot-per-episode cliff)."""
    from types import SimpleNamespace

    from lite.gym.utils import config as env_config

    monkeypatch.setattr(
        env_config,
        "load",
        lambda env_dir: SimpleNamespace(server_kwargs={"max_resets_per_container": 5}),
    )

    class _ForgotWiring(_Recycler):
        def __init__(self, *, cap: int = 0):
            super().__init__(cap=cap)
            del self._max_resets_per_container  # simulate the wiring omission

        async def reset_to_pristine(self) -> bool:
            return True

    env = _ForgotWiring(cap=5)
    with pytest.raises(RuntimeError, match="_max_resets_per_container"):
        _run(env.reset_with_recycle())


def test_finalizer_requires_yaml_cap_when_rollback_declared(monkeypatch):
    """Overriding reset_to_pristine without an explicit yaml
    max_resets_per_container fails AT CLASS CREATION."""
    from types import SimpleNamespace

    from lite.gym.utils import config as env_config

    monkeypatch.setattr(
        env_config,
        "load",
        lambda env_dir: SimpleNamespace(server_kwargs={}, env_kwargs={}),
    )
    with pytest.raises(RuntimeError, match="max_resets_per_container"):

        class _ForgotCap(EnvServerPoolable):
            async def reset_to_pristine(self) -> bool:
                return True

    # with the key present, the same class definition is accepted
    monkeypatch.setattr(
        env_config,
        "load",
        lambda env_dir: SimpleNamespace(server_kwargs={"max_resets_per_container": 30}),
    )

    class _HasCap(EnvServerPoolable):
        async def reset_to_pristine(self) -> bool:
            return True

    # and a subclass NOT declaring rollback needs nothing
    monkeypatch.setattr(
        env_config,
        "load",
        lambda env_dir: (_ for _ in ()).throw(AssertionError("must not be consulted")),
    )

    class _NoRollback(EnvServerPoolable):
        pass


@pytest.mark.parametrize(
    "mod_name,cls_name",
    [
        ("lite.gym.envs.androidworld.main", "AndroidWorldEnv"),
        ("lite.gym.envs.androidlab.main", "AndroidLabEnv"),
        ("lite.gym.envs.mobileworld.main", "MobileWorldEnv"),
        ("lite.gym.envs.lite.cuagym.main", "_CuaGymEnv"),
    ],
)
def test_env_reset_delegates_to_reset_with_recycle(mod_name, cls_name):
    """Wiring pin: the four container envs' ``reset()`` must route through
    the framework's ``reset_with_recycle`` — a hand-rolled recycle sequence
    re-introduced in an env's reset would silently drop the four invariants
    the stubs above prove (recycle cap, first-reset gate,
    tear-down-BEFORE-pristine, cold-spawn-skip)."""
    import importlib
    import inspect

    cls = getattr(importlib.import_module(mod_name), cls_name)
    assert "reset_with_recycle" in inspect.getsource(cls.reset), (
        f"{cls_name}.reset() no longer delegates to reset_with_recycle"
    )


def test_pristine_yaml_cap_key_in_every_config_variant():
    """U18's whole class, made impossible: LITE_*_CONFIG REPLACES the yaml
    (no merge with default), and the finalizer demands the cap key at
    import for pristine-overriding envs — so EVERY sibling config variant of
    an env whose default.yaml declares the key must also carry it.
    Auto-enumerated over lite/gym/envs/**/configs/ (no hand list)."""
    from pathlib import Path

    import yaml

    envs_dir = (
        Path(__file__).resolve().parents[3] / "lite" / "gym" / "envs"
    )  # tests/gym/services/<f> -> repo root
    checked = 0
    offenders: list[str] = []
    for default in sorted(envs_dir.rglob("configs/default.yaml")):
        cfg = yaml.safe_load(default.read_text()) or {}
        if "max_resets_per_container" not in (cfg.get("server_kwargs") or {}):
            continue
        for variant in sorted(default.parent.glob("*.yaml")):
            checked += 1
            vcfg = yaml.safe_load(variant.read_text()) or {}
            if "max_resets_per_container" not in (vcfg.get("server_kwargs") or {}):
                offenders.append(str(variant))
    assert checked >= 5, f"sweep went vacuous ({checked} configs checked)"
    assert not offenders, (
        "config variants missing server_kwargs.max_resets_per_container "
        f"(selecting them would kill the env import — the U18 class): {offenders}"
    )


def test_cuaworld_reset_delegates_to_reset_with_recycle():
    """Wiring pin for the function-local cuaworld env class (defined inside
    ``_make_env_class``, not importable by name). getsource on the whole factory
    is comment-satisfiable — so build the class and AST-match a real
    ``self.reset_with_recycle()`` Call inside its ``reset()`` body: a dropped
    delegation fails here even with the explanatory comments left intact."""
    import ast
    import inspect
    import textwrap

    from lite.gym.envs.lite.cuaworld.src import software

    cls = software._make_env_class(
        "lite.cuaworld.delegation_probe", {"image": "x", "display": "1x1"}
    )
    reset_tree = ast.parse(textwrap.dedent(inspect.getsource(cls.reset)))
    delegates = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "reset_with_recycle"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        for node in ast.walk(reset_tree)
    )
    assert delegates, "cuaworld reset() must call self.reset_with_recycle()"


def test_gym_anything_envs_pin_cap_zero():
    """gym-anything's fresh-runner-per-episode contract = Layer-B cap 0,
    declared in each env's default.yaml (the reset_to_pristine finalizer
    contract) and read into the module constant. The instance-side wiring is
    pinned BEHAVIORALLY by test_cuagym_constructor_wires_cap (cuagym) and the
    cuaworld constructor tests; pristine-override semantics by
    test_gym_anything_never_ran_pristine_semantics."""
    from lite.gym.envs.lite.cuagym import main as cuagym_main
    from lite.gym.envs.lite.cuagym.main import CFG as CUAGYM_CFG
    from lite.gym.envs.lite.cuaworld.src import software

    assert CUAGYM_CFG.server_kwargs["max_resets_per_container"] == 0
    assert software.CFG.server_kwargs["max_resets_per_container"] == 0
    # module constants defined (reading fails loud if a definition regresses —
    # the R3 NameError-at-instantiation near-miss)
    assert cuagym_main._MAX_RESETS_PER_CONTAINER == 0
    assert software._MAX_RESETS_PER_CONTAINER == 0


def test_gym_anything_never_ran_pristine_semantics():
    """The env-side half of the prepaid-boot invariant (#118 first-reset gate
    PLUS never-ran pristine): a backend that never served an episode is
    pristine (True — a fresh backend survives its first reset); one that
    served is not (False — destroy + fresh boot). Kills override deletion,
    unconditional-True, and unconditional-False in both envs."""
    from lite.gym.envs.lite.cuagym import main as cuagym_main
    from lite.gym.envs.lite.cuaworld.src import software

    env = object.__new__(cuagym_main._CuaGymEnv)
    env._cuagym_episode_started = False
    assert _run(env.reset_to_pristine()) is True
    env._cuagym_episode_started = True
    assert _run(env.reset_to_pristine()) is False

    cw_cls = software._make_env_class(
        "lite.cuaworld.pristine_probe", {"image": "x", "display": "1x1"}
    )
    cw = object.__new__(cw_cls)
    cw._cuaworld_episode_dir = None
    assert _run(cw.reset_to_pristine()) is True
    cw._cuaworld_episode_dir = object()
    assert _run(cw.reset_to_pristine()) is False
