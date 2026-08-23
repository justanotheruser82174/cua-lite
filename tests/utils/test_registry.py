"""
Tests for lite.utils.registry: BaseRegistry register, get, list, patterns,
contains, unregister, clear, caching — plus the key grammar
(``split_key`` / ``split_dims`` / ``compose_key`` / ``rebase_key``) this module
owns for every registry and for the durable ``env_key`` parquet column.

Run: uv run pytest tests/utils/test_registry.py -v
"""

from __future__ import annotations

import pytest

from lite.utils.registry import (
    BaseRegistry,
    compose_key,
    rebase_key,
    split_dims,
    split_key,
)

# -----------------------------------------------------------------------------
# Dummy class for testing
# -----------------------------------------------------------------------------

class DemoItem:
    def __init__(self, value: str = "default"):
        self.value = value

# -----------------------------------------------------------------------------
# Register / get / list
# -----------------------------------------------------------------------------

def test_register_and_get_exact():
    """Register exact name and get instance."""
    class R(BaseRegistry[DemoItem]):
        pass
    R.register("item-a", DemoItem)
    inst = R.get("item-a")
    assert isinstance(inst, DemoItem)
    assert inst.value == "default"

def test_register_and_get_with_kwargs():
    """get() passes kwargs to constructor."""
    class R(BaseRegistry[DemoItem]):
        pass
    R.register("item-b", DemoItem)
    inst = R.get("item-b", value="custom")
    assert inst.value == "custom"

def test_get_pattern_match():
    """Register regex pattern and get by matching key."""
    class R(BaseRegistry[DemoItem]):
        pass
    R.register(r"item@.*@type", DemoItem)
    inst = R.get("item@foo@type")
    assert isinstance(inst, DemoItem)
    inst2 = R.get("item@bar@type")
    assert isinstance(inst2, DemoItem)

def test_get_unknown_raises():
    """get() with unregistered key raises KeyError."""
    class R(BaseRegistry[DemoItem]):
        pass
    R.register("only", DemoItem)
    with pytest.raises(KeyError, match="not found"):
        R.get("unknown")

def test_list_returns_exact_keys():
    """list() returns only exact (non-pattern) keys."""
    class R(BaseRegistry[DemoItem]):
        pass
    R.register("a", DemoItem)
    R.register(r"p:.*", DemoItem)
    keys = R.list()
    assert "a" in keys
    assert "p:.*" not in keys

def test_list_patterns_returns_patterns():
    """list_patterns() returns pattern strings."""
    class R(BaseRegistry[DemoItem]):
        pass
    R.register("exact", DemoItem)
    R.register(r"pat:.*", DemoItem)
    patterns = R.list_patterns()
    assert r"pat:.*" in patterns
    assert "exact" not in patterns

# -----------------------------------------------------------------------------
# contains / get_class
# -----------------------------------------------------------------------------

def test_contains_exact_and_pattern():
    """contains() returns True for exact and pattern match."""
    class R(BaseRegistry[DemoItem]):
        pass
    R.register("x", DemoItem)
    R.register(r"n:.*", DemoItem)
    assert R.contains("x") is True
    assert R.contains("n:y") is True
    assert R.contains("other") is False

def test_get_class_returns_class():
    """get_class() returns registered class without instantiating."""
    class R(BaseRegistry[DemoItem]):
        pass
    R.register("c", DemoItem)
    cls = R.get_class("c")
    assert cls is DemoItem

# -----------------------------------------------------------------------------
# Caching
# -----------------------------------------------------------------------------

def test_caching_same_instance():
    """By default get() returns cached instance when called again with no args."""
    class R(BaseRegistry[DemoItem]):
        pass
    R.register("cache-me", DemoItem)
    a = R.get("cache-me")
    b = R.get("cache-me")
    assert a is b

def test_no_cache_when_kwargs():
    """get() with kwargs does not use cache (new instance)."""
    class R(BaseRegistry[DemoItem], cache_by_default=True):
        pass
    R.register("k", DemoItem)
    a = R.get("k", value="v1")
    b = R.get("k", value="v2")
    assert a is not b
    assert a.value == "v1"
    assert b.value == "v2"

# -----------------------------------------------------------------------------
# unregister / clear
# -----------------------------------------------------------------------------

def test_unregister_exact():
    """unregister() removes exact key; get() then raises."""
    class R(BaseRegistry[DemoItem]):
        pass
    R.register("remove", DemoItem)
    assert R.unregister("remove") is True
    with pytest.raises(KeyError):
        R.get("remove")
    assert R.unregister("remove") is False

def test_clear_removes_all():
    """clear() removes all registrations and cache."""
    class R(BaseRegistry[DemoItem]):
        pass
    R.register("a", DemoItem)
    R.register(r"p:.*", DemoItem)
    R.get("a")
    R.clear()
    assert len(R.list()) == 0
    assert len(R.list_patterns()) == 0
    with pytest.raises(KeyError):
        R.get("a")

def test_overwrite_true_replaces():
    """register(..., overwrite=True) replaces existing."""
    class R(BaseRegistry[DemoItem]):
        pass
    R.register("same", DemoItem)
    R.register("same", DemoItem, overwrite=True)
    inst = R.get("same")
    assert isinstance(inst, DemoItem)

def test_overwrite_false_raises_if_different():
    """register(..., overwrite=False) raises when key exists and item differs."""
    class Other:
        pass
    class R(BaseRegistry[DemoItem]):
        pass
    R.register("k", DemoItem)
    with pytest.raises(ValueError, match="already registered"):
        R.register("k", Other, overwrite=False)

# -----------------------------------------------------------------------------
# Key grammar — ``<name>[.<modifier>...]@<dim>[@<dim>...]``
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("name", "dims", "key", "new_name", "rebased"),
    [
        (
            "qwen3_vl",
            ("desktop", "use"),
            "qwen3_vl@desktop@use",
            "gpt",
            "gpt@desktop@use",
        ),
        (
            "qwen3_vl.base",
            ("browser", "grounding.point"),
            "qwen3_vl.base@browser@grounding.point",
            "gpt.history",
            "gpt.history@browser@grounding.point",
        ),
        (
            "lite.osworld",
            ("osworld_chrome_030eeff7",),
            "lite.osworld@osworld_chrome_030eeff7",
            "lite.demo",
            "lite.demo@osworld_chrome_030eeff7",
        ),
        ("lite.demo", (), "lite.demo", "gpt", "gpt"),
    ],
)
def test_compose_split_and_rebase_share_key_grammar(
    name,
    dims,
    key,
    new_name,
    rebased,
):
    assert compose_key(name, *dims) == key
    assert split_key(key) == (name, "@".join(dims))
    assert split_dims(key) == list(dims)
    assert rebase_key(key, new_name) == rebased


@pytest.mark.parametrize(
    "key, dims",
    [
        # agent / adapter grain: two coordinate axes
        ("qwen3_vl@desktop@use", ["desktop", "use"]),
        # a dot INSIDE an axis value is part of that value, not a separator
        ("qwen3_vl@desktop@grounding.point", ["desktop", "grounding.point"]),
        # a dot modifier binds to the NAME and never reaches the dims
        ("qwen3_vl.base@browser@use", ["browser", "use"]),
        # action-space grain: one axis
        ("qwen3_vl@point", ["point"]),
        # env grain: the task_id is the single dim, dots and all
        ("lite.osworld@osworld_chrome_030eeff7", ["osworld_chrome_030eeff7"]),
        # dim-less keys are legal (see split_key's docstring) → no dims
        ("lite.demo", []),
    ],
)
def test_split_dims(key, dims):
    """split_dims() returns the coordinate suffix one axis per element."""
    assert split_dims(key) == dims

def test_split_dims_is_the_inverse_of_compose_key():
    """compose_key(name, *split_dims(k)) reconstructs k exactly.

    The property that lets a caller scan the coordinate axes without knowing
    that the name sits at index 0 of a raw ``key.split("@")``.
    """
    for key in (
        "qwen3_vl@desktop@use",
        "qwen3_vl.base@browser@grounding.bbox",
        "lite.osworld@osworld_chrome_030eeff7",
        "lite.demo",
    ):
        name, _ = split_key(key)
        assert compose_key(name, *split_dims(key)) == key

def test_split_dims_rejects_what_split_key_rejects():
    """The dims accessor rejects empty names and empty coordinate segments."""
    for bad in ("", "@desktop@use"):
        with pytest.raises(ValueError, match="empty name portion"):
            split_key(bad)
        with pytest.raises(ValueError, match="empty name portion"):
            split_dims(bad)
    for bad in ("qwen3_vl@", "qwen3_vl@@use", "qwen3_vl@desktop@"):
        with pytest.raises(ValueError, match="empty dim entry"):
            split_dims(bad)

def test_compose_key_rejects_invalid_name_or_dims():
    """compose_key owns construction-time key grammar for registry callers."""
    for bad_name in ("", "@agent", "agent@desktop"):
        with pytest.raises(ValueError, match="key name"):
            compose_key(bad_name)

    for bad_dim in ("", "@desktop", "desktop@use"):
        with pytest.raises(ValueError, match="key dim"):
            compose_key("agent", bad_dim)

    assert compose_key("agent") == "agent"
    assert compose_key("agent", "desktop", "use") == "agent@desktop@use"

def test_rebase_key_keeps_dims():
    """rebase_key swaps the name; a dim-less key rebases to the bare name."""
    assert rebase_key("qwen3_vl@desktop@use", "gpt") == "gpt@desktop@use"
    assert rebase_key("qwen3_vl@desktop@grounding.point", "gpt") == (
        "gpt@desktop@grounding.point"
    )
    assert rebase_key("lite.demo", "gpt") == "gpt"


@pytest.mark.parametrize("bad_name", ["", "@agent", "agent@desktop"])
def test_rebase_key_rejects_invalid_new_name(bad_name):
    with pytest.raises(ValueError, match="key name"):
        rebase_key("qwen3_vl@desktop@use", bad_name)
