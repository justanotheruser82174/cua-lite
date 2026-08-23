"""``env_kwargs.valid_actions`` behaves IDENTICALLY across every env (C3).

Before this suite, one yaml key meant four different things depending on which
env you pointed it at: a bare ``TypeError: bind() got an unexpected keyword
argument`` (sandbox / mobile family), silent ignore (osworld / osworld_2 /
captcha swallowed it in ``**kwargs``), silent FILTERING (webgym family +
browsergym, via the old desktop-only filter), or verbatim and
unvalidated (osworld_g / screenspot_pro). The filtering spelling was the sharp
one: ``valid_actions: ["terminate"]`` — a name that is legal in
``extra_tools`` — silently became ``[]``, which does NOT mean "no filtering"
but "delete the action-batch tool entirely". The rollout kept running with the
agent's whole GUI capability gone, and the failure read as a bad model.

The config-gate contract, now owned by
``lite.gym.utils.feedback.surface.resolve_valid_actions``:

  * ``None``  → ``None`` — no YAML/config filtering
  * ``[]``    → ``[]``   — deliberately none; strips the action-batch tool
                           (browsergym / webharbor.webvoyager bid-only rows)
  * subset    → verbatim, order preserved
  * anything else (typo, finish/nav tool, wrong-PLATFORM action) → ValueError

Agent-facing ``metadata.valid_actions`` may still be a concrete supported
schema surface when an env physically cannot execute every canonical action
(for example mobile envs do not advertise ``pinch``). That is not a config
filter; step-time rejection still uses the env's runtime gate and returns
env-owned unsupported-action feedback.

Table-driven over ``registry.registered_env_ids()`` so a new env is covered the
day its directory exists — not the day someone remembers to extend a hand list.

Run: uv run pytest tests/gym/utils/feedback/test_valid_actions_contract.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

# Wrapper kwargs ``gym.make`` consumes client-side; a raw entry_point rejects them.
from lite.gym.registry import CARRIED_SPEC_KWARGS, _import_env, _specs, registry
from lite.gym.utils.feedback.surface import (
    copy_valid_actions,
    resolve_valid_actions,
    valid_action_names,
    valid_action_order,
)

os.environ.pop("CUA_LITE_ENV_SERVER_URL", None)
os.environ.pop("CUA_LITE_ENV_SERVER_TOKEN", None)

#: Envs that cannot be constructed at all on a CI/dev host (missing assets,
#: un-vendored third-party package). Their per-env items skip; the coverage
#: floor below still FAILS if the constructible set collapses wholesale.
_MIN_COVERED_ENVS = 12


@pytest.fixture(autouse=True)
def _direct_mode(monkeypatch):
    """A set CUA_LITE_ENV_SERVER_URL would reroute registry lookups to a server."""
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_URL", raising=False)
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_TOKEN", raising=False)


# ---------------------------------------------------------------------------
# The resolver itself — the one implementation every env routes through
# ---------------------------------------------------------------------------

def test_none_and_empty_are_different():
    """The two meanings of "empty" must never collapse into each other."""
    assert resolve_valid_actions(None, env_name="x", platform="desktop") is None
    assert resolve_valid_actions([], env_name="x", platform="desktop") == []


def test_subset_is_verbatim_and_order_preserving():
    got = resolve_valid_actions(
        ["type", "click", "scroll"], env_name="x", platform="desktop",
    )
    assert got == ["type", "click", "scroll"]


def test_returns_a_fresh_list_never_an_alias():
    source = ["click"]
    got = resolve_valid_actions(source, env_name="x", platform="desktop")
    assert got is not source


def test_terminate_raises_instead_of_collapsing_to_empty():
    """THE regression. ``["terminate"]`` used to filter down to ``[]``, which
    deletes the action-batch tool — a one-token yaml typo silently stripping the
    agent's entire GUI capability while the rollout kept running."""
    with pytest.raises(ValueError) as excinfo:
        resolve_valid_actions(["terminate"], env_name="x", platform="desktop")
    message = str(excinfo.value)
    assert "terminate" in message
    # The message must point at the RIGHT knob, not just complain.
    assert "extra_tools" in message


def test_typo_raises_and_names_the_offender():
    with pytest.raises(ValueError, match=r"clik"):
        resolve_valid_actions(["clik", "type"], env_name="x", platform="desktop")


@pytest.mark.parametrize("platform,ok,rejected", [
    ("desktop", ["click", "mouse_move"], ["tap", "swipe"]),
    ("browser", ["click", "scroll"], ["tap", "long_press"]),
    ("mobile", ["tap", "swipe", "long_press"], ["click", "mouse_move"]),
])
def test_platform_axis(platform, ok, rejected):
    """A mobile row naming ``tap``/``swipe`` must be ACCEPTED, not erased.

    The pre-fix helper checked ``LiteDesktopActionSet.get_action_names()`` only, so every
    mobile action would have been silently dropped — the same bug as the
    ``terminate`` collapse but on rows that are entirely correct.
    """
    assert resolve_valid_actions(ok, env_name="x", platform=platform) == ok
    with pytest.raises(ValueError):
        resolve_valid_actions(rejected, env_name="x", platform=platform)


def test_grounding_surface_vocabulary():
    """Grounding envs' whole GUI vocabulary is ``point``/``bbox`` — a desktop-only
    check would have erased ``valid_actions: ["point"]`` into ``[]``."""
    assert valid_action_names("desktop", "grounding.point") == frozenset({"point"})
    assert valid_action_names("desktop", "grounding.bbox") == frozenset({"bbox"})
    assert resolve_valid_actions(
        ["point"], env_name="x", platform="desktop", task_type="grounding.point",
    ) == ["point"]
    with pytest.raises(ValueError):
        resolve_valid_actions(
            ["click"], env_name="x", platform="desktop", task_type="grounding.point",
        )


def test_copy_valid_actions_preserves_none_vs_empty():
    assert copy_valid_actions(None) is None
    assert copy_valid_actions([]) == []
    source = ["click"]
    assert copy_valid_actions(source) == source
    assert copy_valid_actions(source) is not source


# ---------------------------------------------------------------------------
# Per-env sweep — the same five probe inputs against every registered env
# ---------------------------------------------------------------------------

#: Envs knowingly NOT yet on the shared contract. Keep this empty unless a new
#: env lands with an explicit owner/date and a removal condition; do NOT widen
#: it to silence a real regression.
_PENDING_ENVS: set[str] = set()


def _browsergym_leaf_floor() -> list[str]:
    """Stable BrowserGym programmatic leaves for collection-order safety.

    BrowserGym leaves have no per-leaf directory, so they enter the registry via
    the umbrella's lazy catalog loader. When another test module imports the
    BrowserGym extension rows before this file is collected, pytest can observe
    a partially warm registry and silently drop ``browsergym.visualwebarena``
    from this module's parametrization. Keep the parameter table stable from
    the optional packages on disk; each item still validates through the real
    registry/build path below.
    """
    if "browsergym" not in registry.env_ids():
        return []
    return [
        "browsergym.miniwob",
        "browsergym.visualwebarena",
        "browsergym.webarena",
    ]


def _screenspot_annotation() -> dict:
    return {
        "img_filename": "unused.png",
        "instruction": "click the target",
        "bbox": [10, 10, 20, 20],
        "img_size": [100, 100],
    }


@pytest.mark.parametrize(
    "valid_actions, expected",
    [
        (None, None),
        ([], []),
    ],
)
def test_screenspot_pro_preserves_none_vs_empty_without_dataset(valid_actions, expected):
    """Direct constructor coverage: the registry sweep can skip without HF data."""
    from pathlib import Path

    from lite.gym.envs.screenspot_pro.main import ScreenSpotProEnv

    env = ScreenSpotProEnv(
        annotation=_screenspot_annotation(),
        images_dir=Path("/tmp"),
        valid_actions=valid_actions,
    )

    assert env._runtime_metadata().valid_actions == expected


@pytest.mark.parametrize("valid_actions, offender", [
    (["clik", "point"], "clik"),
    (["terminate"], "terminate"),
])
def test_screenspot_pro_rejects_unknown_valid_actions_without_dataset(
    valid_actions,
    offender,
):
    from pathlib import Path

    from lite.gym.envs.screenspot_pro.main import ScreenSpotProEnv

    with pytest.raises(ValueError, match=offender):
        ScreenSpotProEnv(
            annotation=_screenspot_annotation(),
            images_dir=Path("/tmp"),
            valid_actions=valid_actions,
        )


@pytest.mark.parametrize(
    "module_default, expected",
    [
        (None, None),
        ([], []),
        (["point"], ["point"]),
    ],
)
def test_screenspot_pro_task_metadata_copies_valid_actions_without_dataset(
    monkeypatch,
    module_default,
    expected,
):
    from lite.gym.envs.screenspot_pro import main as screenspot_pro

    monkeypatch.setattr(screenspot_pro, "_VALID_ACTIONS", module_default)

    md = screenspot_pro.ScreenSpotProEnv._task_metadata(_screenspot_annotation())

    assert md.valid_actions == expected
    if module_default is not None:
        assert md.valid_actions is not module_default
        md.valid_actions.append("__mutated__")
        assert module_default == expected


@pytest.mark.parametrize(
    "case",
    [
        {
            "env_var": "MOBILEGYM_CONFIG",
            "default_config": "lite/gym/envs/mobilegym/configs/default.yaml",
            "module": "lite.gym.envs.mobilegym.main",
            "class": "RemoteMobileGymEnv",
            "factory": """
                env = cls(
                    "cfg-default-probe",
                    difficulty="L2",
                    scope="S1",
                    objective="operate",
                    composition="atomic",
                    capabilities=["operate"],
                    task_apps=["clock"],
                    answer_fields=False,
                    base_max_steps=30,
                )
            """,
        },
        {
            "env_var": "MOBILE_WORLD_CONFIG",
            "default_config": "lite/gym/envs/mobileworld/configs/default.yaml",
            "module": "lite.gym.envs.mobileworld.main",
            "class": "MobileWorldEnv",
            "factory": 'env = cls(task_id="AcceptMeetingTask")',
        },
        {
            "env_var": "ANDROID_WORLD_CONFIG",
            "default_config": "lite/gym/envs/androidworld/configs/default.yaml",
            "module": "lite.gym.envs.androidworld.main",
            "class": "AndroidWorldEnv",
            "factory": """
                env = cls(
                    config=module.AndroidWorldTaskConfig(
                        use_fake=True,
                        instruction_template="test",
                        metadata={"tags": []},
                    )
                )
            """,
        },
        {
            "env_var": "ANDROID_LAB_CONFIG",
            "default_config": "lite/gym/envs/androidlab/configs/default.yaml",
            "module": "lite.gym.envs.androidlab.main",
            "class": "AndroidLabEnv",
            "factory": """
                env = cls(config=module.AndroidLabTaskConfig(task_id="probe"))
            """,
        },
        {
            "env_var": "WAA_CONFIG",
            "default_config": "lite/gym/envs/waa/configs/default.yaml",
            "module": "lite.gym.envs.waa.main",
            "class": "WindowsAgentArenaEnv",
            "factory": "env = cls()",
        },
    ],
)
def test_valid_actions_config_default_reaches_runtime_metadata(
    tmp_path: Path,
    case: dict[str, str],
) -> None:
    """A non-null config default must not be silently replaced by ``None``.

    Run in a fresh interpreter because env config is consumed at module import,
    and function signature defaults are bound at class definition time.
    """
    config_path = tmp_path / f"{case['env_var'].lower()}.yaml"
    raw = yaml.safe_load(Path(case["default_config"]).read_text(encoding="utf-8"))
    raw["env_kwargs"]["valid_actions"] = []
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    code = textwrap.dedent(
        f"""
        import inspect, json
        from importlib import import_module

        module = import_module({case['module']!r})
        cls = getattr(module, {case['class']!r})
        {textwrap.indent(textwrap.dedent(case['factory']), '        ').strip()}
        md = env._runtime_metadata()

        out = {{
            "ctor_default": inspect.signature(cls.__init__).parameters["valid_actions"].default,
            "runtime_valid_actions": md.valid_actions,
            "schema_default": getattr(
                module,
                "_SCHEMA_VALID_ACTIONS",
                getattr(module, "_VALID_ACTIONS", None),
            ),
        }}
        if hasattr(cls, "bind"):
            out["bind_default"] = inspect.signature(cls.bind).parameters["valid_actions"].default
        print(json.dumps(out))
        """
    )
    env = {
        **os.environ,
        case["env_var"]: str(config_path),
        "CUA_LITE_ENV_SERVER_URL": "",
    }
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[4],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)
    assert got["ctor_default"] == []
    assert got.get("bind_default", []) == []
    assert got["runtime_valid_actions"] == []
    assert got["schema_default"] == []


def _all_env_ids() -> list[str]:
    """Every makeable env id (namespaces expanded to their leaves).

    ``registered_env_ids`` already absorbs a dep/data-absent env itself (typed
    conditions in ``_import_all``, ``EnvDepsMissingError`` on a lazy leaf), so
    anything it raises is a real registration bug and must propagate. It used to
    be wrapped in ``except Exception -> registry.env_ids()``, which silently
    degraded to the cheap directory scan: a no-op today (both return the same 20
    ids) and floored by ``_MIN_COVERED_ENVS = 12``, so the fallback bought
    nothing while hiding the bug it claimed to survive.

    The dir-scan is kept as a FLOOR instead: every scanned env must be covered,
    either directly or by its leaves (a pure umbrella such as ``browsergym`` is
    scanned but replaced by ``browsergym.*`` here). So an env dropping out of
    registration reddens at collection instead of shrinking the parametrize list
    below.
    """
    registered = sorted(set(registry.registered_env_ids()) | set(_browsergym_leaf_floor()))
    covered = set(registered)
    missing = [
        env_id
        for env_id in registry.env_ids()
        if env_id not in covered
        and not any(r.startswith(env_id + ".") for r in registered)
    ]
    assert not missing, (
        f"directory-scanned envs absent from registered_env_ids(): {sorted(missing)}"
    )
    return [
        env_id
        for env_id in registered
        if not any(other.startswith(env_id + ".") for other in registered)
    ]


_ENV_IDS = _all_env_ids()

#: One parametrize argvalues list, reused by all five probe tests so the
#: exemption is declared exactly once.
_ENV_PARAMS = [
    pytest.param(
        env_id,
        marks=pytest.mark.xfail(
            reason=f"{env_id}: valid_actions not yet routed through "
                   "resolve_valid_actions (file owned by a concurrent change)",
        ),
    ) if env_id in _PENDING_ENVS else pytest.param(env_id)
    for env_id in _ENV_IDS
]


def _first_key(env_id: str) -> str | None:
    from lite.gym.errors import EnvDepsMissingError

    try:
        splits = registry.task_ids(env_id)
    except (EnvDepsMissingError, KeyError):
        return None
    for _split, ids in sorted(splits.items()):
        if ids:
            return f"{env_id}@{sorted(ids)[0]}"
    return None


def _build(env_id: str, valid_actions):
    return _build_env(env_id, valid_actions).metadata


def _build_env(env_id: str, valid_actions):
    """Construct one env with ``valid_actions`` and return the env object.

    Uses ``spec.entry_point`` rather than ``gym.make`` on purpose: construction
    is the config boundary this contract lives at, and it costs no container /
    emulator / VM (no ``close()`` needed — nothing is allocated until
    ``boot``/``reset``). ``gym.make`` would additionally run ``ensure_services``.

    Envs whose task materials are absent on this host raise
    :class:`EnvDepsMissingError` from the entry point; those items SKIP rather
    than pass, so the report never claims coverage it does not have.
    """
    from lite.gym.errors import EnvDepsMissingError

    key = _first_key(env_id)
    if key is None:
        pytest.skip(f"{env_id}: no registered tasks on this host")
    spec = _specs[key]
    kwargs = {k: v for k, v in spec.kwargs.items() if k not in CARRIED_SPEC_KWARGS}
    try:
        env = spec.entry_point(**kwargs, valid_actions=valid_actions)
    except EnvDepsMissingError as e:
        pytest.skip(f"{env_id}: env deps missing ({e})")
    return env


def _try_import(env_id: str) -> str | None:
    """Import an env's registration module; return a skip reason on failure.

    Uses the REGISTRY's own resolver rather than ``gym.conftest``'s
    id→module-path helper: several env families (``browsergym.*``,
    ``lite.cuaworld.*``) register many leaf ids from ONE ``main.py``, and the
    helper only special-cases the cuaworld one — the browsergym leaves would
    silently skip, which is exactly the coverage this table exists to have.
    """
    from lite.gym.errors import EnvDepsMissingError

    try:
        _import_env(env_id)
    except (EnvDepsMissingError, ModuleNotFoundError, ImportError) as e:
        return f"{env_id}: env deps missing ({type(e).__name__}: {e})"
    return None


def _skip_unless_importable(env_id: str) -> None:
    reason = _try_import(env_id)
    if reason:
        pytest.skip(reason)


@pytest.mark.parametrize("env_id", _ENV_PARAMS)
def test_none_means_no_filtering(env_id):
    """``valid_actions`` omitted keeps the runtime config gate unfiltered.

    Some envs publish a concrete agent-facing supported schema surface even
    when the YAML gate is omitted; that is not a runtime ``valid_actions``
    filter.
    """
    _skip_unless_importable(env_id)
    env = _build_env(env_id, None)
    runtime_gate = getattr(env, "_valid_actions", None)
    assert runtime_gate is None
    schema_surface = getattr(env, "_schema_valid_actions", None)
    if schema_surface is None:
        assert env.metadata.valid_actions is None
    else:
        assert env.metadata.valid_actions == list(schema_surface)


@pytest.mark.parametrize("env_id", _ENV_PARAMS)
def test_empty_list_is_honored_verbatim(env_id):
    """A DELIBERATE ``valid_actions: []`` still empties the GUI enum (which is
    what strips the wrapper tool) — every env, same answer."""
    _skip_unless_importable(env_id)
    assert _build(env_id, []).valid_actions == []


@pytest.mark.parametrize("env_id", _ENV_PARAMS)
def test_native_subset_is_accepted_verbatim(env_id):
    """A subset of the env's OWN native GUI vocabulary round-trips unchanged."""
    _skip_unless_importable(env_id)
    md = _build(env_id, None)
    native = valid_action_order(str(md.platform), str(md.task_type))
    if not native:
        pytest.skip(f"{env_id}: no GUI action surface ({md.platform}/{md.task_type})")
    subset = native[:2]
    assert _build(env_id, subset).valid_actions == subset


@pytest.mark.parametrize("env_id", _ENV_PARAMS)
def test_typo_raises_everywhere(env_id):
    """A misspelled action is a config error, not a smaller action set."""
    _skip_unless_importable(env_id)
    with pytest.raises(ValueError, match=r"clik"):
        _build(env_id, ["clik", "type"])


@pytest.mark.parametrize("env_id", _ENV_PARAMS)
def test_standalone_tool_name_raises_everywhere(env_id):
    """``["terminate"]`` is a finish tool: it belongs to ``extra_tools`` and must
    never reach the GUI enum — neither verbatim (the old osworld_g / screenspot_pro
    behaviour, which the orthogonality rule forbids) nor by silently filtering to
    ``[]`` (the old webgym-family behaviour, which deletes the action-batch tool)."""
    _skip_unless_importable(env_id)
    with pytest.raises(ValueError, match=r"terminate"):
        _build(env_id, ["terminate"])


def test_sweep_covers_enough_envs():
    """FAIL loud if the table above quietly degenerated into all-skips."""
    covered = [
        e for e in _ENV_IDS
        if _try_import(e) is None and _first_key(e) is not None
    ]
    assert len(covered) >= _MIN_COVERED_ENVS, (
        f"valid_actions sweep only covers {len(covered)} envs "
        f"({covered}); expected >= {_MIN_COVERED_ENVS}"
    )


@pytest.mark.parametrize(
    "env_id",
    ["androidlab", "androidworld", "mobilegym", "mobileworld", "waa"],
)
def test_config_advertises_valid_actions_for_envs_that_accept_it(env_id):
    _skip_unless_importable(env_id)
    assert registry.env_supports_kwarg(env_id, "valid_actions")
