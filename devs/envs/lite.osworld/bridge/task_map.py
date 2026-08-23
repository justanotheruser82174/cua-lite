"""lite.osworld ↔ official osworld task-id map (the cross-replay join).

Both envs run the SAME 369 upstream OSWorld tasks, but their registry keys differ,
so the canonical join is the full OSWorld UUID:

    osworld       : osworld@<uuid>
    lite.osworld  : lite.osworld@osworld_<domain>_<uuid[:8]>
                    (the lite eval-row also carries metadata.osworld_id == <uuid>)

Built from the committed ``lite/gym/envs/osworld/data/tasks.json`` alone — no
eval.jsonl (gitignored + needs OSWORLD_EXAMPLES to regen), no env import, no docker.
The lite key is the eval generator's formula (src/gen/eval/__main__.py:198); rows
carrying an ``exclude_reason`` (blocked / infeasible / google_auth / anti-bot) are
dropped so the cross-replay scope excludes all unwinnable tasks.

Run:
    uv run python devs/envs/lite.osworld/bridge/task_map.py
    uv run python devs/envs/lite.osworld/bridge/task_map.py --check-registry
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root (pyproject.toml) not found")


_REPO = _find_repo_root()
_TASKS_JSON = _REPO / "lite" / "gym" / "envs" / "osworld" / "data" / "tasks.json"


@dataclass(frozen=True)
class Pair:
    """One task, addressable on both substrates. ``uuid`` is the join key."""

    uuid: str
    domain: str
    lite_key: str
    osworld_key: str
    exclude_reason: str | None


def load_pairs(*, include_excluded: bool = False) -> list[Pair]:
    """All (lite_key, osworld_key) pairs from tasks.json. Excluded rows are
    dropped unless ``include_excluded`` (then ``Pair.exclude_reason`` is set)."""
    tasks = json.loads(_TASKS_JSON.read_text())["tasks"]
    pairs: list[Pair] = []
    for t in tasks:
        uuid, domain = t["task_id"], t["domain"]
        reason = t.get("exclude_reason")
        if reason and not include_excluded:
            continue
        pairs.append(
            Pair(
                uuid=uuid,
                domain=domain,
                lite_key=f"lite.osworld@osworld_{domain}_{uuid[:8]}",
                osworld_key=f"osworld@{uuid}",
                exclude_reason=reason,
            )
        )
    return pairs


def assert_no_prefix_collision(pairs: list[Pair]) -> None:
    """The lite key derives from ``uuid[:8]``; the full UUID is authoritative, so
    guard that the truncated form stays unique (keyed by domain, which the lite
    key embeds). Raises on any collision."""
    seen: dict[str, str] = {}
    for p in pairs:
        k = f"{p.domain}_{p.uuid[:8]}"
        if k in seen and seen[k] != p.uuid:
            raise AssertionError(
                f"uuid[:8] collision under {p.lite_key!r}: {p.uuid} vs {seen[k]}"
            )
        seen[k] = p.uuid


def check_registry(pairs: list[Pair]) -> tuple[int, list[str]]:
    """Assert each derived lite key is actually registered in lite.osworld (no
    dead pairs). Returns (n_registered, missing_keys). Imports lite.gym (heavier),
    so it's opt-in."""
    import lite.gym as gym

    registered = {f"lite.osworld@{tid}" for tid in gym.registry.task_ids("lite.osworld")}
    missing = [p.lite_key for p in pairs if p.lite_key not in registered]
    return len(pairs) - len(missing), missing


def _main() -> None:
    import sys

    all_pairs = load_pairs(include_excluded=True)
    live = [p for p in all_pairs if p.exclude_reason is None]
    excluded = [p for p in all_pairs if p.exclude_reason]
    assert_no_prefix_collision(all_pairs)

    reasons = sorted({p.exclude_reason for p in excluded})
    breakdown = {r: sum(1 for p in excluded if p.exclude_reason == r) for r in reasons}
    print(f"osworld tasks.json           : {len(all_pairs)} tasks")
    print(f"  in-scope (no exclude_reason): {len(live)}")
    print(f"  excluded                    : {len(excluded)}  {breakdown}")
    print("  uuid[:8] collisions         : none")
    print(f"  sample pair                 : {live[0].lite_key}  <->  {live[0].osworld_key}")

    if "--check-registry" in sys.argv:
        n_ok, missing = check_registry(live)
        tail = "  (all present)" if not missing else f"  MISSING[{len(missing)}]: {missing[:5]}"
        print(f"  registry coverage           : {n_ok}/{len(live)} lite keys registered{tail}")


if __name__ == "__main__":
    _main()
