"""Unit tests for lite.gym.utils.config (per-env default.yaml + override).

Run: uv run python -m pytest -n0 tests/gym/utils/config/test_config.py
"""

from __future__ import annotations

import ast
import importlib.util
import textwrap
from pathlib import Path

import pytest

from lite.gym.utils import config as env_config
from lite.gym.utils.config import defaults as config_defaults


def _write(env_dir: Path, text: str, name: str = "configs/default.yaml") -> None:
    p = env_dir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text))


@pytest.fixture(autouse=True)
def _clear_cache():
    env_config.load.cache_clear()
    yield
    env_config.load.cache_clear()


def test_loads_groups(tmp_path):
    _write(tmp_path, """
        env_var_prefix: MYENV
        env_kwargs: {max_steps: 15, post_action_delay: 2.0}
        server_kwargs: {memory_limit: "4GB"}
        make_kwargs: {cursor: true, step_timeout: 99.0}
    """)
    cfg = env_config.load(str(tmp_path))
    assert cfg.env_kwargs == {"max_steps": 15, "post_action_delay": 2.0}
    assert cfg.server_kwargs == {"memory_limit": "4GB"}
    assert cfg.make_kwargs == {"cursor": True, "step_timeout": 99.0}
    assert cfg.provenance() == {
        "env_var_prefix": "MYENV",
        "config_env_var": "MYENV_CONFIG",
        "config_env_var_set": False,
        "config_source": "default",
        "config_path": str((tmp_path / "configs/default.yaml").resolve()),
    }


def test_config_package_is_facade_not_implementation() -> None:
    repo = Path(__file__).resolve().parents[4]
    assert not (repo / "lite" / "gym" / "utils" / "config.py").exists()

    facade_spec = importlib.util.find_spec("lite.gym.utils.config")
    assert facade_spec is not None
    assert facade_spec.submodule_search_locations is not None
    assert Path(facade_spec.origin).resolve() == (
        repo / "lite" / "gym" / "utils" / "config" / "__init__.py"
    )

    defaults_spec = importlib.util.find_spec("lite.gym.utils.config.defaults")
    assert defaults_spec is not None
    assert Path(defaults_spec.origin).resolve() == (
        repo / "lite" / "gym" / "utils" / "config" / "defaults.py"
    )

    assert env_config.load is config_defaults.load
    assert env_config.EnvConfig is config_defaults.EnvConfig
    assert env_config.finalize_env_kwargs is config_defaults.finalize_env_kwargs

    tree = ast.parse(
        (repo / "lite" / "gym" / "utils" / "config" / "__init__.py").read_text()
    )
    public_local_defs = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and not node.name.startswith("_")
    }
    assert not public_local_defs


def test_missing_groups_default_to_empty(tmp_path):
    _write(tmp_path, "env_var_prefix: MYENV\n")
    cfg = env_config.load(str(tmp_path))
    assert cfg.env_kwargs == {}
    assert cfg.server_kwargs == {}
    assert cfg.make_kwargs == {}


def test_missing_prefix_asserts(tmp_path):
    _write(tmp_path, "env_kwargs: {max_steps: 15}\n")
    with pytest.raises(AssertionError, match="env_var_prefix"):
        env_config.load(str(tmp_path))


def test_override_replaces_whole(tmp_path, monkeypatch):
    _write(tmp_path, """
        env_var_prefix: MYENV
        env_kwargs: {max_steps: 15, only_in_default: 1}
        server_kwargs: {workers: 256}
        make_kwargs: {cursor: true, step_timeout: 99.0}
    """)
    override = tmp_path / "advanced.yaml"
    override.write_text(textwrap.dedent("""
        env_kwargs: {max_steps: 99}
        server_kwargs: {workers: 8}
        make_kwargs: {cursor: false}
    """))
    monkeypatch.setenv("MYENV_CONFIG", str(override))
    cfg = env_config.load(str(tmp_path))
    # FULL replacement — the override file IS the entire config; default keys gone.
    assert cfg.env_kwargs == {"max_steps": 99}          # only_in_default NOT carried over
    assert cfg.server_kwargs == {"workers": 8}
    assert cfg.make_kwargs == {"cursor": False}
    assert cfg.provenance() == {
        "env_var_prefix": "MYENV",
        "config_env_var": "MYENV_CONFIG",
        "config_env_var_set": True,
        "config_source": "override",
        "config_path": str(override.resolve()),
    }


def test_no_override_uses_default(tmp_path, monkeypatch):
    monkeypatch.delenv("MYENV_CONFIG", raising=False)
    _write(tmp_path, "env_var_prefix: MYENV\nenv_kwargs: {max_steps: 15}\n")
    assert env_config.load(str(tmp_path)).env_kwargs == {"max_steps": 15}


def test_bare_name_resolves_to_bundled_variant(tmp_path, monkeypatch):
    # mirrors browsergym: default.yaml + a COMPLETE named configs/isolation.yaml.
    _write(tmp_path, """
        env_var_prefix: BROWSERGYM
        env_kwargs: {}
        server_kwargs: {backend_isolation: "off", restore_poll_s: 20.0, restore_timeout_s: 900.0}
    """)
    _write(tmp_path, """
        env_var_prefix: BROWSERGYM
        env_kwargs: {}
        server_kwargs: {backend_isolation: "strict", restore_poll_s: 20.0, restore_timeout_s: 900.0}
    """, name="configs/isolation.yaml")
    # Bare name -> bundled variant, used as a whole replacement.
    monkeypatch.setenv("BROWSERGYM_CONFIG", "isolation")
    cfg = env_config.load(str(tmp_path))
    assert cfg.server_kwargs["backend_isolation"] == "strict"
    assert cfg.server_kwargs["restore_poll_s"] == 20.0


def test_config_path_modes(tmp_path, monkeypatch):
    # <PREFIX>_CONFIG accepts a relative name (→ configs/<name>.yaml) OR a full path.
    _write(tmp_path, "env_var_prefix: MYENV\nenv_kwargs: {a: 1}\n")
    _write(tmp_path, "env_kwargs: {a: 2}\n", name="configs/variant.yaml")
    abs = tmp_path / "elsewhere" / "abs.yaml"
    abs.parent.mkdir()
    abs.write_text("env_kwargs: {a: 3}\n")

    # Relative value with suffix -> configs/variant.yaml.
    monkeypatch.setenv("MYENV_CONFIG", "variant.yaml")
    env_config.load.cache_clear()
    assert env_config.load(str(tmp_path)).env_kwargs == {"a": 2}

    monkeypatch.setenv("MYENV_CONFIG", "variant")          # bare name → configs/variant.yaml
    env_config.load.cache_clear()
    assert env_config.load(str(tmp_path)).env_kwargs == {"a": 2}

    monkeypatch.setenv("MYENV_CONFIG", str(abs))           # absolute path → that file
    env_config.load.cache_clear()
    assert env_config.load(str(tmp_path)).env_kwargs == {"a": 3}


def test_for_override_deep_merges_keeps_siblings():
    """overrides[name] deep-merges over the base per-leaf: a nested override
    (computer.image) wins at the leaf and keeps its siblings (memory). Would
    fail under the old shallow ``{**base, **ov}`` (which replaced computer wholesale)."""
    cfg = env_config.EnvConfig(
        env_kwargs={"computer": {"image": "a", "memory": "2g"}, "max_steps": 10},
        server_kwargs={},
        make_kwargs={"cursor": True, "timeouts": {"step": 30.0, "reset": 10.0}},
        overrides={
            "x": {
                "env_kwargs": {"computer": {"image": "b"}},
                "make_kwargs": {"timeouts": {"step": 7.0}},
            }
        },
    )
    out = cfg.for_override("x")
    assert out.env_kwargs == {
        "computer": {"image": "b", "memory": "2g"},
        "max_steps": 10,
    }
    assert out.make_kwargs == {
        "cursor": True,
        "timeouts": {"step": 7.0, "reset": 10.0},
    }


def test_for_override_preserves_config_provenance():
    cfg = env_config.EnvConfig(
        env_kwargs={"a": 1},
        server_kwargs={"b": 2},
        overrides={"x": {"server_kwargs": {"c": 3}}},
        env_var_prefix="MYENV",
        config_env_var="MYENV_CONFIG",
        config_path="/tmp/myenv.yaml",
        config_source="override",
        config_env_var_set=True,
    )
    out = cfg.for_override("x")
    assert out.server_kwargs == {"b": 2, "c": 3}
    assert out.provenance() == cfg.provenance()


def test_finalize_env_kwargs_strips_top_level_none_and_coerces_resolution():
    out = env_config.finalize_env_kwargs(
        {
            "resolution": [800, 600],
            "seed": None,
            "nested": {"keep_none": None},
            "max_steps": 12,
        }
    )
    assert out == {
        "resolution": (800, 600),
        "nested": {"keep_none": None},
        "max_steps": 12,
    }
