"""Tree-wide publish-gate regressions for dataset preprocessors.

Every preproc script must emit rows that pass
:func:`lite.data.utils.rows.validate_canonical_rows`. These common gates stay
here because they inspect every preproc script by inventory rather than one
dataset producer.

Run:
    uv run --extra data --extra dev pytest -q \
        tests/data/preproc/common/test_common_publish_gate.py
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_PREPROC = _ROOT / "lite" / "data" / "preproc"


def _load_preproc_script(path: Path):
    """Import a hyphenated preproc script (not importable as a module name)."""
    name = f"cua_lite_pg_{path.parent.name}_{path.stem.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _preproc_scripts() -> list[Path]:
    """Every dataset entry-point script, discovered from the tree."""
    return sorted(
        p
        for p in _PREPROC.glob("*/*.py")
        if p.stem == "use" or p.stem == "understanding" or p.stem.startswith("grounding")
    )


@pytest.mark.parametrize(
    "script",
    [pytest.param(p, id=f"{p.parent.name}/{p.name}") for p in _preproc_scripts()],
)
def test_script_emitting_tool_calls_uses_shared_stamper(script: Path) -> None:
    """A script that builds ``tool_calls`` must route them through the stamper.

    Canonical ``id`` is required on every tool call by the publish gate. Hand-rolling
    a per-dataset stamping loop is exactly the drift this pins against: the one
    shared implementation lives in ``lite.core.tools.calls``.
    """
    source = script.read_text()
    if '"tool_calls"' not in source:
        pytest.skip("script emits no tool_calls (understanding cohort)")
    assert (
        "stamp_messages_tool_call_ids" in source
        or "finalize_use_messages" in source
    ), (
        f"{script.parent.name}/{script.name} builds tool_calls but never calls the "
        "shared call-id stamper (lite.core.tools.calls explicit stamp helpers / "
        "message helpers); its rows will fail the publish gate."
    )


@pytest.mark.parametrize(
    "script",
    [pytest.param(p, id=f"{p.parent.name}/{p.name}") for p in _preproc_scripts()],
)
def test_script_isolates_corrupt_source_images(script: Path) -> None:
    """One unreadable source image must not abort a full dataset run."""
    source = script.read_text()
    assert "CorruptImageError" in source or "prestage_images" in source


class _ParserBuilt(Exception):
    """Raised in place of ``parse_args`` so ``main()`` stops once its parser exists."""

    def __init__(self, parser: argparse.ArgumentParser) -> None:
        self.parser = parser


def _script_parser(script: Path, monkeypatch) -> argparse.ArgumentParser:
    """The script's REAL ``argparse`` parser, captured out of its ``main()``.

    Every preproc script builds its parser inline at the top of ``main()``, so
    there is no ``build_parser`` to call. Intercepting ``parse_args`` gets the
    genuine parser without running any preprocessing: ``main()`` never reaches
    the line after it.
    """
    module = _load_preproc_script(script)

    def _capture(self, *args, **kwargs):
        raise _ParserBuilt(self)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", _capture)
    try:
        module.main()
    except _ParserBuilt as built:
        parser = built.parser
    else:  # pragma: no cover - a script that never parses argv
        pytest.fail(f"{script.parent.name}/{script.name}: main() built no parser")
    monkeypatch.undo()
    return parser


@pytest.mark.parametrize(
    "script",
    [pytest.param(p, id=f"{p.parent.name}/{p.name}") for p in _preproc_scripts()],
)
def test_script_accepts_a_head_smoke_bound(script: Path, monkeypatch) -> None:
    """Every script must ACCEPT ``--head N`` as an int bound.

    Without a bound these scripts are untestable: ``scalecua/use.py`` and
    ``opencua/use.py`` shipped without one, which is why a from-scratch run
    walked tens of GB and they had never been smoke-tested.

    Asks the real parser rather than grepping the source. The predecessor of
    this test (``assert '"--head"' in source`` over a hand-typed list of 8 of
    the 19 scripts) passed when the flag was deleted but the literal survived in
    a comment, and failed when the flag was kept and respelled.
    """
    parser = _script_parser(script, monkeypatch)
    args = parser.parse_args(["--head", "3"])
    assert args.head == 3, (
        f"{script.parent.name}/{script.name}: --head did not parse to int 3 "
        f"(got {args.head!r})"
    )
