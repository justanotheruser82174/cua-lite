"""Every published rollout route owns one `repo.json`, and both runbooks pass it.

The uploaded `cua-lite/Lite.ScaleCUA` card shipped with an empty `## Origin` and no
citation because the migration runbook's `hf.stage` call carried no card metadata at
all. Nothing failed — the card just rendered an empty section — so this is the gate.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DEVS_DATA = REPO_ROOT / "devs" / "data"
ROUTE_TABLE = DEVS_DATA / "AGENTS.md"
MIGRATION_DOC = REPO_ROOT / "devs" / "migration" / "AGENTS.md"
ROLLOUT_EXAMPLE = REPO_ROOT / "docs" / "examples" / "rollout_to_hf.md"


def _routes() -> list[str]:
    """Route dirs named by the route table in devs/data/AGENTS.md."""
    row = r"^\| `[\w.]+` \| \[devs/data/([\w.]+)/AGENTS\.md\]"
    rows = re.findall(row, ROUTE_TABLE.read_text(), re.M)
    assert rows, "route table in devs/data/AGENTS.md parsed empty"
    return rows


def _stage_commands(doc: Path) -> list[str]:
    """Every `lite.data.hf.stage` invocation in a runbook, backslash-continuations joined."""
    text = doc.read_text().replace("\\\n", " ")
    return [line for line in text.splitlines() if "lite.data.hf.stage" in line]


def test_route_table_matches_the_repo_json_dirs_on_disk() -> None:
    """A reformatted table row must not silently drop a route from the guard below."""
    on_disk = {p.parent.name for p in DEVS_DATA.glob("*/repo.json")}
    assert set(_routes()) == on_disk, (
        "devs/data/AGENTS.md route table and devs/data/*/repo.json disagree; "
        f"table-only={sorted(set(_routes()) - on_disk)} "
        f"disk-only={sorted(on_disk - set(_routes()))}"
    )


@pytest.mark.parametrize("route", _routes())
def test_route_repo_json_carries_upstream_attribution(route: str) -> None:
    meta = json.loads((DEVS_DATA / route / "repo.json").read_text())
    assert meta["original_urls"], f"{route}/repo.json has no upstream links"
    assert meta["citation"].strip(), f"{route}/repo.json has no citation"
    assert meta["description"].strip(), f"{route}/repo.json has no description"
    for url in meta["original_urls"]:
        assert url.startswith("https://"), f"{route}/repo.json: {url!r} is not an https URL"


@pytest.mark.parametrize("route", _routes())
def test_route_runbook_stages_with_its_repo_dir(route: str) -> None:
    expected = f"--repo-dir devs/data/{route}"
    published = [
        cmd for cmd in _stage_commands(DEVS_DATA / route / "AGENTS.md")
        # `.wip`-style scratch repos are deliberately throwaway and carry their own warning
        if ".wip" not in cmd
    ]
    assert published, f"{route}/AGENTS.md has no hf.stage command"
    for cmd in published:
        assert expected in cmd, f"{route}/AGENTS.md stages without {expected}: {cmd.strip()[:160]}"


def test_migration_runbook_stages_with_a_route_repo_dir() -> None:
    commands = _stage_commands(MIGRATION_DOC)
    assert commands, "devs/migration/AGENTS.md has no hf.stage command"
    for cmd in commands:
        assert re.search(r"--repo-dir devs/data/[\w.]+", cmd), (
            f"migration runbook stages without a route --repo-dir: {cmd.strip()[:160]}"
        )


def test_rollout_example_stages_with_attribution() -> None:
    """The worked example re-stages the same repo twice; both must carry attribution.

    Its "extend a published run" block re-uploads to the repo staged earlier in the
    same file, so a stage without attribution there republishes the card with an
    empty `## Origin`, undoing the first upload.
    """
    commands = _stage_commands(ROLLOUT_EXAMPLE)
    assert commands, "rollout_to_hf.md has no hf.stage command"
    for cmd in commands:
        assert "--original-urls" in cmd or "--repo-dir" in cmd, (
            f"docs example stages without upstream attribution: {cmd.strip()[:160]}"
        )
