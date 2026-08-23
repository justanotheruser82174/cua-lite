"""Metadata same-source invariant mechanics and per-env parity.

Invariant: for a registered task, ``registry.task_metadata(env_id, task_id)``
equals ``gym.make(key).metadata`` field for field when make passes no
env_kwargs; every live difference is attributable to an explicit env_kwargs
override. Identity (``env_id``/``task_id``) is framework-injected
(``registry.register`` for specs; ``gym.make`` + the ``LiteBaseEnv.metadata``
property for live envs).

Layout:
  * Structural guard coverage — every importable LiteBaseEnv subclass inherits
    the framework ``metadata`` property (only EnvWrapper / LiteEnvClient own it).
  * Registered==constructed parity + registration isolation,
    auto-parametrized/enumerated over registry.env_ids() plus captcha make-path
    composition, coverage floor, and lookup_task ownership. No hand list: a new
    env is covered the day its directory exists.
  * Bare-sandbox and osworld mechanics for identity injection, rebind,
    mutation isolation, wrapper passthrough, env_kwargs attribution, and
    registration idempotence.
  * Structural gates for builder purity, repo-root parents policy, config
    freeze, enumeration pins, overridability, and env_kwargs override chains.
  * Drift-guards — the None-vs-[] class (dep-free, data-free pins).

Run: uv run pytest tests/gym/metadata/test_metadata_invariant.py -v
"""
from __future__ import annotations

import copy
import importlib
from pathlib import Path

import pytest

import lite.gym as gym
from lite.core.metadata import (
    LiteBaseMetadata,
    LiteCUAMetadata,
    LiteGenericMetadata,
)
from lite.core.tools import make_tool_schema
from lite.core.tools.extra_tools import LiteFinishToolSet
from lite.core.tools.schemas import tool_schema_name
from lite.gym.base import LiteBaseEnv
from lite.gym.errors import EnvDepsMissingError
from lite.gym.registry import _clear_env_registration, _specs, _splits, registry
from lite.gym.sandbox import SandboxBaseEnv, SandboxTaskConfig, register_tasks
from lite.utils.path import project_root

# ---------------------------------------------------------------------------
# Direct mode is a precondition for every registry/make probe here (a set
# CUA_LITE_ENV_SERVER_URL reroutes task_ids to a remote server).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _direct_mode(monkeypatch):
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_URL", raising=False)
    yield
    for env_id in list(_splits):
        if env_id.startswith("_inv_"):
            _clear_env_registration(env_id)


# ---------------------------------------------------------------------------
# Structural guard coverage (the __init_subclass__ guard enforces the contract
# at import; this section imports the env surface and re-checks it)
# ---------------------------------------------------------------------------

#: Permanent framework exemptions (the ONLY classes allowed to own `metadata`
#: at the G end state). CuaSandboxEnv turned out to migrate trivially
#: (its metadata was already a pure ctor-arg function), so only the two
#: genuinely un-migratable classes remain.
FRAMEWORK_EXEMPT = {"EnvWrapper", "LiteEnvClient"}


def _structural_imports() -> list[str]:
    """Guard-bypass sweep roster: the 3 framework/SDK modules (explicit) plus
    EVERY env main, derived from registry.env_ids() through the conftest
    module-path resolver — no hand list to rot (issue 120's lesson, applied to
    the very test that once carried a 19-entry hand list)."""
    from gym.conftest import _env_main_module

    from lite.gym import registry

    mods = {"lite.gym.wrappers", "lite.gym.remote.client",
            "lite.gym.envs.cua.sandbox.env"}
    for env_id in registry.env_ids():
        mods.add(_env_main_module(env_id))
    return sorted(mods)


_STRUCTURAL_IMPORTS = _structural_imports()


def _walk_subclasses(cls) -> set[type]:
    out: set[type] = set()
    stack = list(cls.__subclasses__())
    while stack:
        c = stack.pop()
        if c in out:
            continue
        out.add(c)
        stack.extend(c.__subclasses__())
    return out


def test_lite_metadata_constructed_only_in_builders():
    """Task metadata constructors live only at env metadata builders.

    The old untagged metadata constructor is retired everywhere in the scanned
    env surface. New tagged metadata constructors are allowed only inside
    ``_task_metadata``/``_runtime_metadata`` bodies. Exemption: cua/sandbox/env.py
    (never registered; its metadata IS a pure ctor function).
    """
    import ast


    root = project_root()
    retired_name = "Lite" + "Metadata"
    offenders: list[str] = []
    for path in _builder_module_files(root):
        if path.match("*/cua/sandbox/env.py"):
            continue
        tree = ast.parse(path.read_text())
        builder_spans: list[tuple[int, int]] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                    node.name in ("_task_metadata", "_runtime_metadata"):
                builder_spans.append((node.lineno, node.end_lineno))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
                if name == retired_name:
                    offenders.append(f"{path}:{node.lineno}: retired metadata constructor")
                elif name in {"LiteCUAMetadata", "LiteGenericMetadata"} and not any(
                    a <= node.lineno <= b for a, b in builder_spans
                ):
                    offenders.append(f"{path}:{node.lineno}: {name} outside builder")
    assert not offenders, (
        "Tagged metadata constructed outside a builder or through a retired "
        f"constructor: {offenders}"
    )


def _assert_same_metadata_contract(
    key: str,
    live: LiteBaseMetadata,
    reg: LiteBaseMetadata,
) -> None:
    """Assert registered/live parity without forcing generic metadata into CUA fields."""
    assert type(live) is type(reg), key
    assert live.dims == reg.dims, key
    assert live.extra_tool_schemas == reg.extra_tool_schemas, key

    if isinstance(live, LiteCUAMetadata):
        assert isinstance(reg, LiteCUAMetadata), key
        assert live.platform == reg.platform, key
        assert live.task_type == reg.task_type, key
        assert live.valid_actions == reg.valid_actions, key
    elif isinstance(live, LiteGenericMetadata):
        assert isinstance(reg, LiteGenericMetadata), key
    else:
        raise AssertionError(f"{key}: unexpected metadata class {type(live).__name__}")


def test_structural_only_framework_exemptions_own_metadata():
    """PERMANENT (G end state): every importable LiteBaseEnv subclass either
    inherits the framework ``metadata`` property or is one of the two
    framework exemptions. The __init_subclass__ guard enforces this at class
    creation; this test additionally imports the env surface so a guard
    bypass (e.g. a future guard edit) still fails here."""
    unimportable: list[str] = []
    for mod in _STRUCTURAL_IMPORTS:
        try:
            importlib.import_module(mod)
        except Exception as e:  # dep-gated env module (e.g. webgym pkg)
            unimportable.append(f"{mod}: {type(e).__name__}")

    offenders = [
        f"{cls.__module__}.{cls.__name__}"
        for cls in _walk_subclasses(LiteBaseEnv)
        if "metadata" in cls.__dict__ and cls.__name__ not in FRAMEWORK_EXEMPT
    ]
    assert not offenders, (
        f"classes defining their own `metadata` beyond the 2 framework "
        f"exemptions: {offenders} — implement _task_metadata + "
        "_runtime_metadata instead"
    )
    # The two gym-anything mains are dep-light BY DESIGN (zstandard import is
    # function-local; dataset/materials checks deferred) — they must import on
    # every host, else U17's guard coverage silently zeroes out.
    for must in ("lite.gym.envs.lite.cuagym.main", "lite.gym.envs.lite.cuaworld.main"):
        assert not any(u.startswith(must) for u in unimportable), (
            f"{must} failed to import — structural guard coverage lost: {unimportable}"
        )
    if unimportable:
        print("structural check skipped (unimportable):", unimportable)


def test_metadata_override_guard_fires_exemption_works_and_does_not_inherit():
    from lite.gym.wrappers import EnvWrapper

    with pytest.raises(TypeError, match="overrides `metadata`"):
        class Bad(LiteBaseEnv):                       # noqa: F811
            metadata = property(lambda self: None)

    class Exempt(LiteBaseEnv, allow_metadata_override=True):
        metadata = property(lambda self: None)        # allowed

    # The keyword does NOT inherit: a subclass of an exempted class
    # (incl. the real EnvWrapper) re-defining metadata still trips.
    with pytest.raises(TypeError, match="overrides `metadata`"):
        class BadChild(Exempt):
            metadata = property(lambda self: None)

    with pytest.raises(TypeError, match="overrides `metadata`"):
        class BadWrapper(EnvWrapper):
            metadata = property(lambda self: None)

    # Defining the hooks never trips the guard.
    class Fine(LiteBaseEnv):
        def _runtime_metadata(self):
            raise NotImplementedError


def test_missing_runtime_metadata_fails_at_instantiation():
    """G end state: _runtime_metadata is @abstractmethod — an env that forgets
    it is un-instantiable (fail-early), not a first-read crash."""
    class Forgetful(LiteBaseEnv):
        async def reset(self): ...
        async def step(self, actions): ...
        async def close(self): ...

    with pytest.raises(TypeError, match="abstract"):
        Forgetful()


# ---------------------------------------------------------------------------
# Parity + registration isolation for every scanned env.
# ---------------------------------------------------------------------------


def _spec_keys(env_id: str) -> list[str]:
    """Spec keys owned by a scanned env — prefix-scoped so nested env_ids the
    directory scan can't name (cua.bench.webtop.demo@..., cuaworld.<sw>@...)
    are captured, while sibling scan ids never collide (osworld vs osworld_2:
    neither "@" nor "." follows the shared prefix)."""
    return [k for k in _specs
            if k.startswith(env_id + "@") or k.startswith(env_id + ".")]


def _trigger_env_import(env_id: str) -> Exception | None:
    """Trigger the env's lazy import via task_ids, tolerating the failure:
    namespace ids (browsergym, cua.bench) raise KeyError here AFTER their
    import registered the nested specs, and dep/data-gated envs raise before
    registering anything — either way the caller decides via _spec_keys.
    Returns the exception (for skip messages) or None."""
    try:
        registry.task_ids(env_id)
    except Exception as e:  # noqa: BLE001 — see docstring
        return e
    return None


# Collection-time parametrization over the CHEAP directory scan (env_ids()
# walks lite/gym/envs/ only — zero imports, so collection stays free and each
# test item imports just its own env, deterministically, inside the body).
@pytest.mark.parametrize("env_id", sorted(registry.env_ids()))
def test_registered_metadata_matches_constructed_env(env_id):
    """Registered == no-override live for EVERY scanned env (auto-discovered;
    no hand list to rot — a new env is covered the day its directory exists).
    Live side = spec.entry_point(**spec.kwargs): probe-free even where
    gym.make boots a backend or probes deps. Identity (env_id/task_id) is
    stripped from the registered copy (no gym.make → no live injection)."""
    err = _trigger_env_import(env_id)
    keys = _spec_keys(env_id)
    if not keys:
        detail = f" ({type(err).__name__}: {err})" if err else ""
        pytest.skip(f"{env_id}: nothing registered on this host{detail}")
    for key in keys[:20]:
        spec = _specs[key]
        actual_env_id, tid = key.split("@", 1)
        try:
            env = spec.entry_point(**spec.kwargs)
        except EnvDepsMissingError as e:  # ctor dep-gated on this host
            pytest.skip(f"{env_id}: entry_point construction failed ({type(e).__name__}: {e})")
        live = env._runtime_metadata()
        reg = registry.task_metadata(actual_env_id, tid)
        reg_others = {k2: v for k2, v in reg.others.items()
                      if k2 not in ("env_id", "task_id")}
        # Type contract: list, never None (the None-vs-[] class — see the
        # drift-guard banner below).
        assert isinstance(reg.extra_tool_schemas, list), key
        _assert_same_metadata_contract(key, live, reg)
        assert dict(live.others) == reg_others, key


@pytest.mark.parametrize(
    "env_id",
    [
        "webgym",
        "online_mind2web",
        "webharbor.webvoyager",
        "waa",
        "captcha",
    ],
)
def test_resolved_valid_actions_excludes_finish_nav_and_app_tools(env_id):
    err = _trigger_env_import(env_id)
    keys = _spec_keys(env_id)
    if not keys:
        detail = f" ({type(err).__name__}: {err})" if err else ""
        pytest.skip(f"{env_id}: nothing registered on this host{detail}")

    forbidden = {
        "response", "terminate", "answer", "done", "finish", "submit",
        "goto", "back", "forward", "new_tab", "switch_tab", "close_tab",
        "open_app", "ask_user", "web_search", "pause_and_memorize_fact",
    }
    for key in keys[:20]:
        spec = _specs[key]
        actual_env_id, tid = key.split("@", 1)
        try:
            env = spec.entry_point(**spec.kwargs)
        except EnvDepsMissingError as e:  # ctor dep-gated on this host
            pytest.skip(f"{env_id}: entry_point construction failed ({type(e).__name__}: {e})")
        reg = registry.task_metadata(actual_env_id, tid)
        live = env._runtime_metadata()
        assert isinstance(live, LiteCUAMetadata), key
        assert isinstance(reg, LiteCUAMetadata), key
        assert not (set(live.valid_actions or []) & forbidden), key
        assert not (set(reg.valid_actions or []) & forbidden), key


@pytest.mark.parametrize(
    ("env_id", "extra_tools"),
    [
        ("waa", ["response", "terminate", "report_infeasible"]),
        ("captcha", ["response", "terminate"]),
    ],
)
def test_finish_tools_resolve_only_from_env_extra_tools(env_id, extra_tools):
    err = _trigger_env_import(env_id)
    keys = _spec_keys(env_id)
    if not keys:
        detail = f" ({type(err).__name__}: {err})" if err else ""
        pytest.skip(f"{env_id}: nothing registered on this host{detail}")

    key = keys[0]
    spec = _specs[key]
    try:
        bare = spec.entry_point(**spec.kwargs)._runtime_metadata()
        tooled = spec.entry_point(**{**spec.kwargs, "extra_tools": extra_tools})._runtime_metadata()
    except EnvDepsMissingError as e:  # ctor dep-gated on this host
        pytest.skip(f"{env_id}: entry_point construction failed ({type(e).__name__}: {e})")
    assert isinstance(bare, LiteCUAMetadata), key
    assert isinstance(tooled, LiteCUAMetadata), key

    finish_names = LiteFinishToolSet.get_tool_names()
    assert not (set(bare.valid_actions or []) & finish_names), key
    assert not (
        {tool_schema_name(schema) for schema in bare.extra_tool_schemas} & finish_names
    ), key
    assert [tool_schema_name(schema) for schema in tooled.extra_tool_schemas] == extra_tools
    assert tooled.valid_actions == bare.valid_actions


@pytest.mark.parametrize(
    ("env_id", "extra_tools"),
    [
        ("webgym", ["goto", "back", "response", "terminate"]),
        ("online_mind2web", ["goto", "back", "forward", "response"]),
        ("webharbor.webvoyager", ["back", "response"]),
    ],
)
def test_web_extra_tools_resolve_from_env_kwargs_without_polluting_valid_actions(
    env_id,
    extra_tools,
):
    err = _trigger_env_import(env_id)
    keys = _spec_keys(env_id)
    if not keys:
        detail = f" ({type(err).__name__}: {err})" if err else ""
        pytest.skip(f"{env_id}: nothing registered on this host{detail}")

    key = keys[0]
    spec = _specs[key]
    try:
        bare = spec.entry_point(**spec.kwargs)._runtime_metadata()
        tooled = spec.entry_point(**{**spec.kwargs, "extra_tools": extra_tools})._runtime_metadata()
    except EnvDepsMissingError as e:  # ctor dep-gated on this host
        pytest.skip(f"{env_id}: entry_point construction failed ({type(e).__name__}: {e})")
    assert isinstance(bare, LiteCUAMetadata), key
    assert isinstance(tooled, LiteCUAMetadata), key

    selected = set(extra_tools)
    assert not (
        {tool_schema_name(schema) for schema in bare.extra_tool_schemas} & selected
    ), key
    assert [tool_schema_name(schema) for schema in tooled.extra_tool_schemas] == extra_tools
    assert not (set(tooled.valid_actions or []) & selected), key
    assert tooled.valid_actions == bare.valid_actions


def test_registered_constructed_parity_coverage_floor():
    """Anti-vacuity: at least 5 scanned envs must be constructible+compared
    on ANY host (this host: 13). FAILS (not skips) below the floor — silent
    zero-coverage is the failure mode this file exists to kill; the roster
    itself is visible as the per-env test items above."""
    constructible = 0
    for env_id in registry.env_ids():
        try:
            _trigger_env_import(env_id)
            key = _spec_keys(env_id)[0]
            _specs[key].entry_point(**_specs[key].kwargs)
            constructible += 1
        except Exception:  # noqa: BLE001 — dep/data-gated env
            continue
    assert constructible >= 5, f"only {constructible} envs constructible (vacuous parity)"


def test_registration_metadata_isolation_all_envs():
    """Builder output never aliases registration state.

    For every constructible env: mutate a constructed env's
    ``_runtime_metadata().others`` top-level AND inside every nested
    list/dict, then assert the registered copy is unchanged and a FRESH
    instance sees pristine values.
    """
    checked = 0
    for env_id in registry.env_ids():
        try:
            _trigger_env_import(env_id)
            key = _spec_keys(env_id)[0]
            spec = _specs[key]
            env = spec.entry_point(**spec.kwargs)
        except Exception:  # noqa: BLE001 — dep/data-gated env
            continue
        actual_env_id, tid = key.split("@", 1)
        before = copy.deepcopy(registry.task_metadata(actual_env_id, tid).to_dict())
        md = env._runtime_metadata()
        md.others["__t9__"] = True
        for v in md.others.values():
            if isinstance(v, list):
                v.append("__t9__")
            elif isinstance(v, dict):
                v["__t9__"] = True
        # Top-level list fields alias registration state the same way
        # (osworld_g's valid_actions did, uncaught until this sweep grew).
        if isinstance(md, LiteCUAMetadata) and isinstance(md.valid_actions, list):
            md.valid_actions.append("__t9__")
        md.extra_tool_schemas.append(make_tool_schema("__t9__"))
        assert registry.task_metadata(actual_env_id, tid).to_dict() == before, key
        fresh = spec.entry_point(**spec.kwargs)._runtime_metadata()
        assert "__t9__" not in fresh.others, key
        for k2, v in fresh.others.items():
            if isinstance(v, list):
                assert "__t9__" not in v, (key, k2)
        if isinstance(fresh, LiteCUAMetadata):
            assert "__t9__" not in (fresh.valid_actions or []), key
        assert all(
            tool_schema_name(schema) != "__t9__"
            for schema in fresh.extra_tool_schemas
        ), key
        checked += 1
    assert checked >= 5, f"isolation sweep covered only {checked} envs (vacuous)"


def test_lookup_task_returns_owned_copy():
    """``lookup_task`` is a bind-time exit from registration state and must
    deep-copy — the ``_TASKS`` entry IS
    the factory-closure task, so a leaked reference would let an instance
    mutation poison every future cold deepcopy too."""
    from lite.gym.sandbox import lookup_task
    env_id = "_inv_t9_lookup"
    task = _bare_task("t9_lookup", {"flavor": "x"})
    _register_bare(env_id, task)
    try:
        got = lookup_task(env_id, task.task_id)
        assert got is not task, "lookup_task leaked the registered object"
        got.metadata["others"]["__t9__"] = True
        assert "__t9__" not in lookup_task(env_id, task.task_id).metadata["others"], \
            "mutation reached the next lookup"
        assert "__t9__" not in registry.task_metadata(env_id, task.task_id).others, \
            "mutation reached the registered metadata"
    finally:
        _specs.pop(f"{env_id}@{task.task_id}", None)


def test_make_path_metadata_matches_registered():
    """Full gym.make composition (factory branch + wrappers + identity
    injection) == registered — captcha, whose make is probe-free
    (CaptchaServices has no ensure; LocalCaptchaEnv.__init__ is pure
    attribute assignment). The sweep above bypasses make's probes; this test
    pins the composed path. Enumeration is
    skip-guarded (captcha's one-time asset fetch is a host seed); once assets
    are present, make cannot dep-fail — a failure here is a real composition
    break, never host capability."""
    try:
        tid = registry.task_ids("captcha")["eval"][0]
    except Exception as e:  # noqa: BLE001 — asset seed absent on this host
        pytest.skip(f"captcha enumeration unavailable ({type(e).__name__}: {e})")
    assert gym.make(f"captcha@{tid}").metadata == registry.task_metadata("captcha", tid)


# ---------------------------------------------------------------------------
# Mechanics on the bare-sandbox harness (no docker, no deps)
# ---------------------------------------------------------------------------

def _bare_task(task_id: str, others: dict | None = None) -> SandboxTaskConfig:
    return SandboxTaskConfig(
        task_id=task_id,
        instruction="smoke",
        platform="desktop",
        computer={"image": "lite.placeholder:latest", "display": "1920x1080",
                  "memory": "4GB", "cpu": "1", "timeout": 120},
        max_steps=15,
        metadata={"others": {"domain": "smoke", **(others or {})}},
    )


def _register_bare(env_id: str, *tasks: SandboxTaskConfig):
    register_tasks(env_id, {"train": list(tasks)})


def test_bare_sandbox_registered_and_live_metadata_match():
    """Dep-free REAL parity: the bare-sandbox harness goes through the real
    register → gym.make → base-property chain with no docker/dep probes, so
    this runs on every host (the per-real-env rows above are data-gated)."""
    env_id = "_inv_t1_bare"
    task = _bare_task("t1_task", {"difficulty": 3})
    _register_bare(env_id, task)
    try:
        spec_md = registry.task_metadata(env_id, task.task_id)
        live_md = gym.make(f"{env_id}@{task.task_id}").metadata
        assert live_md == spec_md          # field-for-field, incl. identity
        assert live_md.others["env_id"] == env_id
        assert live_md.others["task_id"] == task.task_id
        assert live_md.others["difficulty"] == 3
    finally:
        _specs.pop(f"{env_id}@{task.task_id}", None)


def test_identity_present_on_made_env_absent_on_direct():
    env_id = "_inv_t5"
    task = _bare_task("t5_task")
    _register_bare(env_id, task)
    try:
        env = gym.make(f"{env_id}@{task.task_id}")
        md = env.metadata
        assert md.others["env_id"] == env_id
        assert md.others["task_id"] == task.task_id
        assert md.others["domain"] == "smoke"
        # Direct construction (no key): identity keys absent by domain scoping.
        direct = SandboxBaseEnv(task=task)
        assert "env_id" not in direct.metadata.others
        assert "task_id" not in direct.metadata.others
    finally:
        _specs.pop(f"{env_id}@{task.task_id}", None)


def test_rebind_refreshes_metadata_through_wrapper_stack():
    """Rebind shape: bind(new_task) + identity attr refresh → metadata read
    through the outer wrapper shows the NEW task's fields AND task_id."""
    env_id = "_inv_t6"
    t1 = _bare_task("t6_a", {"flavor": "one"})
    t2 = _bare_task("t6_b", {"flavor": "two"})
    _register_bare(env_id, t1, t2)
    try:
        env = gym.make(f"{env_id}@{t1.task_id}")   # wrapped stack (StepTimeout…)
        assert env.metadata.others["task_id"] == "t6_a"
        assert env.metadata.others["flavor"] == "one"
        # Rebind path: refresh the identity attr on UNWRAPPED after bind.
        env.unwrapped.bind(t2)
        env.unwrapped.task_id = t2.task_id
        md = env.metadata                            # read through the wrapper
        assert md.others["task_id"] == "t6_b"
        assert md.others["flavor"] == "two"
        assert md.others["env_id"] == env_id         # env_id never changes
    finally:
        _specs.pop(f"{env_id}@{t1.task_id}", None)
        _specs.pop(f"{env_id}@{t2.task_id}", None)


def test_metadata_read_mutation_isolation():
    env_id = "_inv_t9"
    task = _bare_task("t9_task")
    _register_bare(env_id, task)
    try:
        env = gym.make(f"{env_id}@{task.task_id}")
        env.metadata.others["INJECTED"] = True       # mutate a returned dict
        assert "INJECTED" not in env.metadata.others  # fresh dict per read
        assert "INJECTED" not in task.metadata["others"]  # task record untouched
        assert "INJECTED" not in registry.task_metadata(env_id, task.task_id).others
    finally:
        _specs.pop(f"{env_id}@{task.task_id}", None)


def test_task_metadata_returns_owned_copy():
    env_id = "_inv_t9_task_metadata_copy"
    task = _bare_task("t9_task_metadata_copy", {
        "nested": {"value": 1},
        "tags": ["a"],
    })
    _register_bare(env_id, task)
    try:
        task.metadata["others"]["nested"]["value"] = 9
        md = registry.task_metadata(env_id, task.task_id)
        assert md.others["nested"] == {"value": 1}

        md.others["INJECTED"] = True
        md.others["nested"]["value"] = 2
        md.others["tags"].append("b")
        md.extra_tool_schemas.append(make_tool_schema("__t9__"))

        fresh = registry.task_metadata(env_id, task.task_id)
        assert "INJECTED" not in fresh.others
        assert fresh.others["nested"] == {"value": 1}
        assert fresh.others["tags"] == ["a"]
        assert all(
            tool_schema_name(schema) != "__t9__"
            for schema in fresh.extra_tool_schemas
        )
    finally:
        _specs.pop(f"{env_id}@{task.task_id}", None)
        _splits.pop(env_id, None)


def test_wrapper_metadata_passthrough_matches_unwrapped():
    env_id = "_inv_t10"
    task = _bare_task("t10_task")
    _register_bare(env_id, task)
    try:
        env = gym.make(f"{env_id}@{task.task_id}")
        assert env.metadata == env.unwrapped.metadata
    finally:
        _specs.pop(f"{env_id}@{task.task_id}", None)


def test_reregistration_is_idempotent():
    env_id = "_inv_t11"
    task = _bare_task("t11_task")
    _register_bare(env_id, task)
    _register_bare(env_id, task)                     # second registration
    try:
        ids = registry.task_ids(env_id)["train"]
        assert ids.count(task.task_id) == 1, "duplicate _splits append on re-register"
        md = registry.task_metadata(env_id, task.task_id)
        assert md.others["task_id"] == task.task_id  # identity injection value-idempotent
        assert md.others["env_id"] == env_id
    finally:
        _specs.pop(f"{env_id}@{task.task_id}", None)


# ---------------------------------------------------------------------------
# Env-kwargs attribution + no-override defaults (osworld: dep-free make).
# ---------------------------------------------------------------------------

def test_osworld_env_kwargs_override_only_explicit_metadata_field():
    try:
        splits = registry.task_ids("osworld")
    except Exception as e:  # noqa: BLE001 — dep/data-gated enumeration
        pytest.skip(f"osworld: enumeration unavailable ({type(e).__name__}: {e})")
    keys = [t for tids in splits.values() for t in tids][:1]
    if not keys:
        pytest.skip("osworld: no registered tasks (data absent)")
    tid = keys[0]
    spec_md = registry.task_metadata("osworld", tid)
    # No-override defaults equal the registered values.
    try:
        bare = gym.make(f"osworld@{tid}").metadata
    except EnvDepsMissingError as e:
        pytest.skip(f"osworld: make dep-gated ({e})")
    assert isinstance(bare, LiteCUAMetadata)
    assert isinstance(spec_md, LiteCUAMetadata)
    assert bare.extra_tool_schemas == spec_md.extra_tool_schemas == []
    # An explicit env_kwargs override touches only its field.
    tooled = gym.make(f"osworld@{tid}", extra_tools=["report_infeasible"]).metadata
    assert isinstance(tooled, LiteCUAMetadata)
    assert [tool_schema_name(t) for t in tooled.extra_tool_schemas] == [
        "report_infeasible"
    ]
    assert dict(tooled.others) == dict(bare.others)
    assert tooled.valid_actions == bare.valid_actions
    assert tooled.platform == bare.platform and tooled.task_type == bare.task_type


# ---------------------------------------------------------------------------
# Durable structural tests.
# ---------------------------------------------------------------------------

def _builder_module_files(root) -> list:
    """Modules under the purity gate — DIRECTORY-SCANNED (no hand list to
    rot: a new env's main.py/env.py/software.py enters the gate the day it
    exists), plus sandbox/base.py (the family builder). software.py: lite
    .cuaworld's env class body lives there, not in a main.py — without the
    glob a future builder override in it would silently escape the gate.
    Scanned files without builders are umbrella/helper modules and pass
    through; the count floor below keeps the gate non-vacuous."""
    envs_dir = root / "lite" / "gym" / "envs"
    files = [p for p in
             sorted(envs_dir.rglob("main.py")) + sorted(envs_dir.rglob("env.py"))
             + sorted(envs_dir.rglob("software.py"))
             # install.sh clones UPSTREAM repos into env .cache/ dirs
             # (gitignored) — their main.py must not enter the gate.
             if ".cache" not in p.parts]
    return [root / "lite" / "gym" / "sandbox" / "base.py"] + files


def _builder_bodies(source: str) -> str:
    """Concatenate the source of every _task_metadata/_runtime_metadata def."""
    import ast
    tree = ast.parse(source)
    chunks: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                node.name in ("_task_metadata", "_runtime_metadata"):
            chunks.append(ast.get_source_segment(source, node) or "")
    return "\n".join(chunks)


def test_builder_purity_no_identity_no_environ():
    """Builders never hand-write identity keys and never read env/live state."""
    root = project_root()
    with_builders = 0
    for path in _builder_module_files(root):
        body = _builder_bodies(path.read_text())
        if not body:
            continue  # umbrella/helper module — owns no builder
        with_builders += 1
        rel = path.relative_to(root)
        assert '"task_id"' not in body and '"env_id"' not in body, (
            f"{rel}: builder hand-writes identity — framework-owned")
        assert "os.environ" not in body and "os.getenv" not in body, (
            f"{rel}: builder reads the environment — breaks registered/live parity")
    # Non-vacuity floor: all 18 migrated modules own builders today; mass
    # disappearance (or a scan bug) must be loud, not a silent pass.
    assert with_builders >= 18, f"purity gate saw only {with_builders} builder modules"


def test_repo_root_parents_policy():
    """Zero repo-root parents[N] derivations outside audited exemptions."""
    import re

    root = project_root()
    exempt = {
        "lite/gym/envs/waa/scripts/utils/sync_tasks.py",            # sibling checkout
        "lite/gym/envs/lite/osworld/src/gen/train/__main__.py",     # env-dir asset root
        "lite/gym/envs/lite/osworld/src/gen/eval/__main__.py",
        "lite/gym/envs/lite/osworld/src/gen/train/synth/__init__.py",  # env-dir data path
    }
    offenders = []
    for py in (root / "lite").rglob("*.py"):
        rel = str(py.relative_to(root))
        if rel in exempt or "/.cache/" in rel:
            continue
        for i, line in enumerate(py.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]              # comments are not code
            if re.search(r"parents\[[3-9]\]", code):
                offenders.append(f"{rel}:{i}")
    assert not offenders, (
        f"unaudited deep parents[N] derivations (repo-root risk): {offenders} — "
        "use lite.utils.path.project_root() or add an audited exemption")


def test_config_env_var_change_after_import_does_not_move_metadata(monkeypatch):
    """A <PREFIX>_CONFIG change AFTER env-module import is a no-op
    on BOTH sides — the env module froze its CFG-derived constants at import,
    so neither the registered copy nor a fresh instance re-reads the yaml.
    (Deliberately NO assertion on env_config.load's lru_cache identity:
    sibling tests legitimately cache_clear() it, which made an identity
    assert order-flaky across xdist workers.)"""
    from lite.gym.envs.webharbor.webvoyager import main as wv
    from lite.gym.utils.feedback.surface import copy_valid_actions

    monkeypatch.setenv("WEBHARBOR_WEBVOYAGER_CONFIG", "/nonexistent/other.yaml")
    ids = registry.task_ids("webharbor.webvoyager")["eval"]
    expected_valid_actions = copy_valid_actions(wv._VALID_ACTIONS)
    spec_md = registry.task_metadata("webharbor.webvoyager", ids[0])
    assert isinstance(spec_md, LiteCUAMetadata)
    spec_va = spec_md.valid_actions
    assert spec_va == expected_valid_actions
    # Live side: construction under the mutated env var still serves the
    # import-time constants.
    spec = _specs[f"webharbor.webvoyager@{ids[0]}"]
    env = spec.entry_point(**spec.kwargs)
    live_md = env._runtime_metadata()
    assert isinstance(live_md, LiteCUAMetadata)
    assert live_md.valid_actions == expected_valid_actions


def test_web_env_registration_count_pins():
    """Registration-count pins for web envs with dep-free enumeration."""
    assert len(registry.task_ids("webharbor.webvoyager")["eval"]) == 643
    assert len(registry.task_ids("online_mind2web")["eval"]) == 300


_ENV_KWARGS_OVERRIDE_CASES = [
    ("webharbor.webvoyager", "max_steps", 7,
     lambda md: md.others["max_steps"]),
    ("webharbor.webvoyager", "viewport", (640, 480),
     lambda md: tuple(md.others["viewport"])),
    ("online_mind2web", "max_steps", 9,
     lambda md: md.others["max_steps"]),
    ("online_mind2web", "trajectory_dir", "/tmp/env-kwargs",
     lambda md: md.others.get("trajectory_dir")),
]


@pytest.mark.parametrize(
    ("env_id", "field", "probe", "read"),
    _ENV_KWARGS_OVERRIDE_CASES,
    ids=[
        f"{env_id}-{field}"
        for env_id, field, _, _ in _ENV_KWARGS_OVERRIDE_CASES
    ],
)
def test_env_kwargs_overridability_matrix(env_id, field, probe, read):
    """Env-kwargs overrides reflect on live metadata,
    the registered value never moves (dep-free envs only).

    ONE PYTEST ROW PER CASE, not one loop over all four. The cases span two envs
    whose images go STALE independently (the freshness hash is over raw source
    bytes, so a comment-only edit to a baked file is enough), and
    ``EnvDepsMissingError`` is a SKIP here because that is host state. As a single
    loop the first unavailable env skipped the whole matrix and took the other
    env's cases with it — one ``s`` in the report standing for 0 of 4 assertions
    executed. Per-row, an available env keeps asserting and the skip count says
    how many cases went unchecked.
    """
    try:
        tid = registry.task_ids(env_id)["eval"][0]
        spec_md = registry.task_metadata(env_id, tid)
        live = gym.make(f"{env_id}@{tid}", **{field: probe}).metadata
    except EnvDepsMissingError as e:
        pytest.skip(f"{env_id} unavailable (image/deps; run its install.sh): {e}")
    got = read(live)
    assert got == probe or str(got) == str(probe), (env_id, field, got)
    # registered never moves
    assert registry.task_metadata(env_id, tid) == spec_md


def test_env_kwargs_override_chain_three_levels():
    """default.yaml → recipe-yaml env_kwargs → CLI env_kwargs, merged
    the way rollout.py does (top-level per-key, args win); registered never
    moves; each live difference is attributable to its layer."""
    env_id = "online_mind2web"
    try:
        tid = registry.task_ids(env_id)["eval"][0]
        key = f"{env_id}@{tid}"
        spec_md = registry.task_metadata(env_id, tid)

        # (i) bare: live == registered
        assert gym.make(key).metadata == spec_md
        # (ii) recipe yaml layer
        recipe = {"max_steps": 21}
        merged = {**recipe}
        assert gym.make(key, **merged).metadata.others["max_steps"] == 21
        # (iii) CLI layer wins per-key (rollout.py's merged_env_kwargs shape)
        cli = {"max_steps": 33}
        merged = {**recipe, **cli}
        live = gym.make(key, **merged).metadata
        assert live.others["max_steps"] == 33
        # registered untouched throughout
        assert registry.task_metadata(env_id, tid) == spec_md
    except EnvDepsMissingError as e:
        pytest.skip(f"{env_id} unavailable (image/deps; run its install.sh): {e}")


# ---------------------------------------------------------------------------
# Drift-guards — the None-vs-[] class (audit round 3)
# ---------------------------------------------------------------------------
# Lesson: lite.osworld registered extra_tool_schemas=None while the no-override
# live instance served [] — invisible for the whole migration because that
# env's parity row skips on hosts without eval.jsonl / desktop_env. These
# two guards express the same invariants WITHOUT task data or env deps, so
# they run (and would have failed) on every host. (Their enumerating siblings
# were folded into the registered-vs-constructed parity sweep.)


def test_sandbox_builder_normalizes_none_extra_tool_schemas():
    """``SandboxTaskConfig.extra_tool_schemas`` defaults to None but
    ``LiteCUAMetadata.extra_tool_schemas`` is ``list``-typed: the shared builder
    must normalize. A registered None also silently breaks registered==live
    for subclasses whose no-override amendment resolves to []."""
    md = SandboxBaseEnv._task_metadata(_bare_task("drift_norm"))
    assert md.extra_tool_schemas == []


def test_lite_osworld_amendment_default_equals_registered():
    """lite.osworld's only amendment (extra_tool_schemas) must serve the
    registered value when no env_kwargs override it. Expressed dep-free — the
    ``bind()`` signature default (``_EXTRA_TOOLS``, from default.yaml),
    resolved exactly as bind() resolves it, against the builder output; no
    instantiation (the ctor is desktop_env-gated), no task data.

    The builder override mirrors bind()'s resolver, so the two sides move
    TOGETHER when default.yaml changes; this pins that symmetry (and today's
    empty default)."""
    lo = importlib.import_module("lite.gym.envs.lite.osworld.main")
    registered = lo.LiteOsworldEnv._task_metadata(_bare_task("drift_lo"))
    live_default = lo.LiteOsworldEnv.extra_tool_schemas(lo._EXTRA_TOOLS)
    assert registered.extra_tool_schemas == live_default
    assert live_default == []


@pytest.mark.asyncio
async def test_osworld_g_report_infeasible_schema_gate_dep_free():
    """OSWorld-G data may be absent, but refusal extra-tool schema resolution is pure."""
    from lite.core.tools import make_tool_call
    from lite.gym.envs.osworld_g.main import OSWorldGEnv

    annotation = {
        "box_type": "refusal",
        "image_size": [100, 100],
        "image_path": "unused.png",
        "instruction": "Click the missing control.",
    }
    inactive = OSWorldGEnv(
        annotation_original=annotation,
        annotation_refined=annotation,
        images_dir=Path("."),
        extra_tools=[],
    )
    inactive._screenshot = b"shot"

    inactive_result = await inactive.step([
        make_tool_call(
            "report_infeasible",
            {"reason": "missing"},
            call_id="call_report",
        )
    ])

    assert inactive_result.reward == 0.0
    assert inactive_result.results[0].tool_call_id == "call_report"
    assert inactive_result.results[0].text is None
    assert inactive_result.results[0].error == (
        "report_infeasible is not available in this task."
    )
    assert inactive_result.results[0].metadata == {"is_error": True}
    assert inactive_result.info["executed_actions"][0] == {
        "call": "noop",
        "args": {"name": "report_infeasible", "reason": "inactive extra tool"},
    }

    active = OSWorldGEnv(
        annotation_original=annotation,
        annotation_refined=annotation,
        images_dir=Path("."),
        extra_tools=["report_infeasible"],
    )
    active._screenshot = b"shot"

    active_result = await active.step([
        make_tool_call(
            "report_infeasible",
            {"reason": "missing"},
            call_id="call_report",
        )
    ])

    assert active_result.reward == 1.0
    assert active_result.results == []
