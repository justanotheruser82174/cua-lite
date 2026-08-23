"""
Tests for task filtering via registry metadata and parse_filter.

Covers:
  1. EnvSpec metadata field (LiteCUAMetadata)
  2. register() with metadata
  3. Registry.task_metadata()
  4. parse_filter() with lambda expressions on metadata.others
  5. collect_tasks() with filter_fn
  6. Filter ordering: filter before head/sample

Run:
    uv run pytest tests/gym/registry/test_task_filter.py -v
"""

from __future__ import annotations

import pytest

from lite.core import LiteCUAMetadata
from lite.core.errors import LiteContractError
from lite.core.utils.filters import parse_filter
from lite.gym.registry import EnvSpec, _specs, _splits, register, registry
from lite.infer.rollout import collect_tasks

# =============================================================================
# Helpers
# =============================================================================


def _cleanup_test_env(env_id: str = "_test"):
    """Remove all test registrations."""
    to_remove = [k for k in _specs if k.startswith(f"{env_id}@")]
    for k in to_remove:
        del _specs[k]
    _splits.pop(env_id, None)


def _register_test_tasks():
    """Register fake tasks with metadata for testing."""
    _cleanup_test_env()
    tasks = [
        ("t1", "train", {"difficulty": 1, "domain": "search"}),
        ("t2", "train", {"difficulty": 2, "domain": "search"}),
        ("t3", "train", {"difficulty": 3, "domain": "news"}),
        ("t4", "train", {"difficulty": 5, "domain": "shopping"}),
        ("t5", "train", {"difficulty": 7, "domain": "travel"}),
        ("t6", "train", {"difficulty": 10, "domain": "travel"}),
        ("t7", "eval", {"difficulty": 1, "domain": "search"}),
        ("t8", "eval", {"difficulty": 5, "domain": "news"}),
    ]
    for task_id, split, others in tasks:
        register(
            key=f"_test@{task_id}",
            entry_point=lambda **kw: None,  # type: ignore
            split=split,
            metadata=LiteCUAMetadata(
                dims=("desktop", "use"),
                others=others,
            ),
        )


@pytest.fixture(autouse=True)
def setup_and_teardown(monkeypatch):
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_URL", raising=False)
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_TOKEN", raising=False)
    _register_test_tasks()
    yield
    _cleanup_test_env()


# =============================================================================
# EnvSpec metadata
# =============================================================================


class TestEnvSpecMetadata:
    def test_default_none(self):
        spec = EnvSpec(key="x@y", entry_point=lambda **kw: None)  # type: ignore
        assert spec.metadata is None

    def test_with_metadata(self):
        meta = LiteCUAMetadata(dims=("desktop", "use"), others={"difficulty": 3})
        spec = EnvSpec(
            key="x@y",
            entry_point=lambda **kw: None,  # type: ignore
            metadata=meta,
        )
        assert spec.metadata is meta
        assert spec.metadata.others["difficulty"] == 3


# =============================================================================
# Registry.task_metadata()
# =============================================================================


class TestTaskMetadata:
    def test_returns_metadata(self):
        meta = registry.task_metadata("_test", "t1")
        assert isinstance(meta, LiteCUAMetadata)
        assert meta.others["env_id"] == "_test"
        assert meta.others["task_id"] == "t1"
        assert meta.others["difficulty"] == 1
        assert meta.others["domain"] == "search"
        assert meta.platform == "desktop"

    def test_returns_none_for_unknown(self):
        meta = registry.task_metadata("_test", "nonexistent")
        assert meta is None

    def test_returns_none_for_unknown_env(self):
        meta = registry.task_metadata("nosuchenv", "t1")
        assert meta is None


# =============================================================================
# parse_filter (operates on metadata.others dict)
# =============================================================================


class TestParseFilter:
    def _meta(self, **others):
        """Helper to create LiteCUAMetadata with given others dict."""
        from lite.core import LiteCUAMetadata

        return LiteCUAMetadata(dims=("desktop", "use"), others=others)

    def test_difficulty_filter(self):
        fn = parse_filter("lambda m: m.others.get('difficulty', 0) <= 3")
        assert fn(self._meta(difficulty=1)) is True
        assert fn(self._meta(difficulty=3)) is True
        assert fn(self._meta(difficulty=4)) is False

    def test_domain_filter(self):
        fn = parse_filter("lambda m: m.others.get('domain') in ('search', 'news')")
        assert fn(self._meta(domain="search")) is True
        assert fn(self._meta(domain="news")) is True
        assert fn(self._meta(domain="travel")) is False

    def test_domain_exclude(self):
        fn = parse_filter("lambda m: m.others.get('domain') != 'travel'")
        assert fn(self._meta(domain="search")) is True
        assert fn(self._meta(domain="travel")) is False

    def test_combined_filter(self):
        fn = parse_filter(
            "lambda m: m.others.get('difficulty', 0) <= 6 and m.others.get('domain') != 'travel'"
        )
        assert fn(self._meta(difficulty=3, domain="search")) is True
        assert fn(self._meta(difficulty=5, domain="shopping")) is True
        assert fn(self._meta(difficulty=5, domain="travel")) is False
        assert fn(self._meta(difficulty=7, domain="search")) is False

    def test_empty_metadata_graceful(self):
        fn = parse_filter("lambda m: m.others.get('difficulty', 0) <= 3")
        assert fn(self._meta()) is True  # default 0 <= 3

    def test_len_of_others_get_fallback(self):
        """Documented form: len(m.others.get("sites", []))."""
        fn = parse_filter("lambda m: len(m.others.get('sites', [])) > 1")
        assert fn(self._meta(sites=["a", "b"])) is True
        assert fn(self._meta(sites=["a"])) is False
        assert fn(self._meta()) is False

    def test_str_strip_on_others_get(self):
        """Documented form: m.others.get("domain", "").strip()."""
        fn = parse_filter("lambda m: m.others.get('domain', '').strip() == 'chrome'")
        assert fn(self._meta(domain=" chrome ")) is True
        assert fn(self._meta(domain="vs_code")) is False

    def test_str_removeprefix_chained_split(self):
        """Documented webgym site-start filter (devs/data/webgym/AGENTS.md)."""
        fn = parse_filter(
            "lambda m: m.others.get('website', '')"
            ".split('//')[-1].split('/')[0].removeprefix('www.') "
            "not in ('google.com', 'bing.com', 'duckduckgo.com')"
        )
        assert fn(self._meta(website="https://shop.example/cart")) is True
        assert fn(self._meta(website="https://www.google.com/search")) is False

    def test_or_fallback_receiver_for_value_method(self):
        """Documented migration filter (devs/migration/AGENTS.md): a value
        method chained off an ``others.get(...) or default`` fallback."""
        fn = parse_filter(
            "lambda m: 'incomplete' not in (m.others.get('exclude_reason') or '').split(',')"
        )
        assert fn(self._meta(exclude_reason="incomplete,blocked")) is False
        assert fn(self._meta()) is True

    @pytest.mark.parametrize(
        ("expr", "match"),
        [
            ("42", "Use the documented lambda form"),
            ("str", "Use the documented lambda form"),
            ("lambda: True", "Use exactly one metadata argument"),
            ("lambda m, n: True", "Use exactly one metadata argument"),
            ("lambda *ms: True", "Use exactly one metadata argument"),
        ],
    )
    def test_invalid_expressions_raise_contract_error(self, expr, match):
        with pytest.raises(LiteContractError, match=match):
            parse_filter(expr)


# =============================================================================
# collect_tasks with filter_fn (filter operates on metadata.others)
# =============================================================================


class TestCollectTasksFilter:
    def test_no_filter(self):
        _, tasks = collect_tasks("_test", splits=["train"])
        assert len(tasks) == 6

    def test_filter_easy_only(self):
        fn = parse_filter("lambda m: m.others.get('difficulty', 0) <= 3")
        _, tasks = collect_tasks("_test", splits=["train"], filter_fn=fn)
        assert len(tasks) == 3
        assert set(tasks) == {"t1", "t2", "t3"}

    def test_filter_by_domain(self):
        fn = parse_filter("lambda m: m.others.get('domain') == 'search'")
        _, tasks = collect_tasks("_test", splits=["train"], filter_fn=fn)
        assert set(tasks) == {"t1", "t2"}

    def test_filter_then_head(self):
        """head applies after filter — so head=2 on 3 easy tasks gives 2."""
        fn = parse_filter("lambda m: m.others.get('difficulty', 0) <= 3")
        _, tasks = collect_tasks("_test", splits=["train"], filter_fn=fn, head=2)
        assert len(tasks) == 2

    def test_filter_then_sample(self):
        """sample applies after filter."""
        fn = parse_filter("lambda m: m.others.get('difficulty', 0) <= 3")
        _, tasks = collect_tasks("_test", splits=["train"], filter_fn=fn, sample=2)
        assert len(tasks) == 2

    def test_sample_equal_to_len_shuffles_whole_set(self):
        """sample == len(tasks) returns every task in seed-deterministic random
        order (the domain-interleaving shuffle documented in the AGENTS.md
        collection workflows) instead of no-opping."""
        import random

        _, plain = collect_tasks("_test", splits=["train"])
        _, shuffled = collect_tasks(
            "_test", splits=["train"], sample=len(plain), rng=random.Random(42)
        )
        _, shuffled2 = collect_tasks(
            "_test", splits=["train"], sample=len(plain), rng=random.Random(42)
        )
        assert sorted(shuffled) == sorted(plain)
        assert shuffled != plain
        assert shuffled == shuffled2

    def test_filter_eval_split(self):
        fn = parse_filter("lambda m: m.others.get('difficulty', 0) <= 3")
        _, tasks = collect_tasks("_test", splits=["eval"], filter_fn=fn)
        assert tasks == ["t7"]

    def test_filter_removes_all_returns_empty(self):
        """If filter removes all tasks, collect_tasks returns empty list."""
        fn = parse_filter("lambda m: m.others.get('difficulty', 0) > 100")
        _, tasks = collect_tasks("_test", splits=["train"], filter_fn=fn)
        assert tasks == []
