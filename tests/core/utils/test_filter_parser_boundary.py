"""Filter parser boundary: one metadata lambda shape, not arbitrary callables.

The boundary owns the predicate *shape*. The body is ordinary trusted Python,
so a bad name or operand must surface loudly on the first row rather than being
rejected by a grammar whitelist that documented flows would trip over.
"""

from __future__ import annotations

import pytest

from lite.core.errors import LiteContractError
from lite.core.metadata import LiteCUAMetadata
from lite.core.utils.filters import parse_filter


def _metadata(**others) -> LiteCUAMetadata:
    return LiteCUAMetadata(dims=("desktop", "use"), others=others)


def test_parse_filter_accepts_documented_metadata_lambda() -> None:
    keep = parse_filter("lambda m: not m.others.get('exclude_reason')")

    assert keep(_metadata()) is True
    assert keep(_metadata(exclude_reason="blocked")) is False


def test_parse_filter_accepts_metadata_predicate_operations() -> None:
    keep = parse_filter(
        "lambda m: "
        "m.others['category'] == 'rotation' "
        "and m.platform == 'desktop' "
        "and m.others.get('mode') is None "
        "and 'map' not in m.others.get('sites', [])"
    )

    assert keep(_metadata(category="rotation", sites=["shop"])) is True
    assert keep(_metadata(category="rotation", sites=["map"])) is False
    assert keep(_metadata(category="crop", sites=[])) is False


def test_parse_filter_accepts_documented_webgym_site_start_filter() -> None:
    keep = parse_filter(
        "lambda m: "
        "m.others.get('difficulty',0)==3 "
        "and m.others.get('website','').split('//')[-1]"
        ".split('/')[0].removeprefix('www.') "
        "not in ('google.com','bing.com','duckduckgo.com')"
    )

    assert keep(_metadata(difficulty=3, website="https://shop.example/cart")) is True
    assert (
        keep(_metadata(difficulty=3, website="https://accounts.google.com/login"))
        is True
    )
    assert keep(_metadata(difficulty=3, website="https://www.google.com/search")) is False
    assert keep(_metadata(difficulty=3, website="https://duckduckgo.com/")) is False
    assert keep(_metadata(difficulty=4, website="https://shop.example/cart")) is False


def test_parse_filter_accepts_others_get_strip_fallback() -> None:
    """CD-CORE-FILTER-PARSER-BOUNDARY names this exact fallback-trim form as
    one the boundary must accept."""
    keep = parse_filter("lambda m: m.others.get('website', '').strip() == 'ok'")

    assert keep(_metadata(website=" ok ")) is True
    assert keep(_metadata(website="no")) is False


@pytest.mark.parametrize(
    "clause",
    [
        # devs/exps/eval/browsergym.{webarena,visualwebarena}/run.sh splice an
        # operator-supplied EVAL_READ_FILTER / EVAL_WRITE_FILTER clause into the
        # documented lambda. That clause is arbitrary user Python, so builtins,
        # comprehensions, and ordinary value methods must all keep working.
        "True",
        "len(m.others.get('sites', [])) > 1",
        "any(s in m.others.get('sites', []) for s in ('reddit', 'gitlab'))",
        "int(m.others.get('difficulty', 0)) <= 3",
        "abs(m.others.get('episode_return', 0)) > 0.5",
        "sorted(m.others.get('sites', []))[:1] != ['map']",
        "set(m.others.get('sites', [])) & {'reddit'}",
        "str(m.platform) == 'desktop'",
        "m.platform.value == 'desktop'",
        "m.others.get('task_id', '').startswith('webarena.')",
        "m.others.get('name', '').lower().endswith('.py')",
        "[s for s in m.others.get('sites', []) if s != 'map']",
    ],
)
def test_parse_filter_accepts_operator_supplied_clause(clause: str) -> None:
    keep = parse_filter(f"lambda m: (not m.others.get('mutating')) and ({clause})")

    assert keep(
        _metadata(
            sites=["reddit", "gitlab"],
            difficulty=2,
            episode_return=1.0,
            task_id="webarena.4",
            name="x.py",
        )
    )


@pytest.mark.parametrize(
    "expr",
    [
        "str",
        "print",
        "pytest.fail",
        "42",
        "m.others.get('x')",
        "lambda: True",
        "lambda m, extra: True",
        "lambda *items: True",
        "lambda **items: True",
        "lambda m=None: True",
        "lambda m, *, extra=1: True",
        "lambda m: (",
        str,
        print,
        None,
    ],
)
def test_parse_filter_rejects_non_metadata_lambda_shapes(expr: object) -> None:
    with pytest.raises(LiteContractError, match="one-argument Python lambda"):
        parse_filter(expr)


def test_parse_filter_body_errors_surface_on_the_row() -> None:
    """The boundary gates shape, not the body. An unresolvable name or a bad
    operand raises loudly on the first metadata row, naming the symbol."""
    keep = parse_filter("lambda m: imported_filter(m)")
    with pytest.raises(NameError, match="imported_filter"):
        keep(_metadata())

    keep = parse_filter("lambda m: m.others.get('difficulty') < 3")
    with pytest.raises(TypeError):
        keep(_metadata())


def test_parse_filter_body_does_not_see_owner_module_imports() -> None:
    """The predicate runs against builtins plus its own argument, so the
    filters module's own imports are not an accidental namespace."""
    keep = parse_filter("lambda m: ast")
    with pytest.raises(NameError, match="ast"):
        keep(_metadata())
