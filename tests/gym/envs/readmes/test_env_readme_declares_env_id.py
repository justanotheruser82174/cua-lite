"""Every env README must state the ``--env-id`` its page is about.

The root README's benchmark and environment lists speak in product names
(``WindowsAgentArena``, ``WebArena``), because that is what a reader scanning
for "is my benchmark here" searches for. The id they then have to type lives at
the link destination instead, so the two surfaces do not carry duplicate copies
of the same fact.

That only works if the destination actually states it. Before this, ids appeared
in env READMEs only incidentally, buried inside a shell command 50 lines down --
and ``lite/osworld`` never stated its own at all. The mapping is nowhere near
mechanical for the cases that matter: ``WindowsAgentArena`` is ``waa``, and
``WebArena`` / ``VisualWebArena`` / ``MiniWoB`` are three ``browsergym.*`` ids
behind one shared page.
"""

from __future__ import annotations

import collections
import pathlib

import lite.gym as gym
from lite.utils.path import project_root

_ENVS = project_root() / "lite" / "gym" / "envs"
_SKIP = {".cache", "_vendor", "node_modules"}
_BADGE = "`--env-id`"


def _env_readmes() -> list[pathlib.Path]:
    return [p for p in _ENVS.rglob("README.md") if not _SKIP & set(p.parts)]


def _owner_of(env_id: str, readmes: list[pathlib.Path]) -> pathlib.Path | None:
    """The README whose directory is the longest dotted prefix of ``env_id``."""
    best: tuple[str, pathlib.Path] | None = None
    for readme in readmes:
        rel = readme.parent.relative_to(_ENVS).as_posix().replace("/", ".")
        if rel and (env_id == rel or env_id.startswith(rel + ".")):
            if best is None or len(rel) > len(best[0]):
                best = (rel, readme)
    return best[1] if best else None


def test_every_env_id_has_an_owning_readme() -> None:
    readmes = _env_readmes()
    orphans = sorted(e for e in gym.registry.registered_env_ids() if _owner_of(e, readmes) is None)
    assert not orphans, "registered env ids with no README under lite/gym/envs:\n  " + "\n  ".join(
        orphans
    )


def test_owning_readme_states_the_env_id_up_top() -> None:
    """The badge sits on the third line, right under the ``#`` heading.

    Not in the heading itself: that is the page title on GitHub and in the file
    tree, a heading cannot carry the three ``browsergym.*`` ids or the forty
    ``lite.cuaworld.*`` ones, and for the dozen envs whose id is just the
    lowercased name it would be pure redundancy.
    """
    readmes = _env_readmes()
    owned: dict[pathlib.Path, list[str]] = collections.defaultdict(list)
    for env_id in sorted(gym.registry.registered_env_ids()):
        owner = _owner_of(env_id, readmes)
        if owner is not None:
            owned[owner].append(env_id)

    missing_badge, unstated = [], []
    for readme, env_ids in owned.items():
        rel = readme.relative_to(project_root())
        head = readme.read_text().split("\n")[:3]
        if len(head) < 3 or not head[2].startswith(_BADGE):
            missing_badge.append(str(rel))
            continue
        # Families that expand (cuaworld) name a few leaves plus a count, so
        # require the shared prefix rather than every leaf.
        prefix = readme.parent.relative_to(_ENVS).as_posix().replace("/", ".")
        wanted = env_ids if len(env_ids) <= 4 else [prefix]
        for env_id in wanted:
            if f"`{env_id}`" not in head[2] and env_id not in head[2]:
                unstated.append(f"{rel}: {env_id}")

    assert not missing_badge, (
        "env READMEs with no `--env-id` line under the heading:\n  " + "\n  ".join(missing_badge)
    )
    assert not unstated, (
        "env READMEs whose `--env-id` line omits an id they own:\n  " + "\n  ".join(unstated)
    )
