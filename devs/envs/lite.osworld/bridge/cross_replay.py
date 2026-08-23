"""Bidirectional cross-substrate replay: bridge lite.osworld ↔ official osworld.

Records come from a normal ``scripts/rollout.py`` run (per-turn
``turn_NNNN/03_actions.json`` with ``lite_message.tool_calls`` plus debug
prompt-image artifacts read through ``lite.infer.debug.log_layout``). This driver replays ONE
substrate's recorded actions open-loop on the OTHER substrate, saves per-turn
screenshots, and diffs them — so a divergence is the substrate, not the model. See
../AGENTS.md §3–§4 for the plan and the 2×2 output layout.

Turn-set alignment (critical): a recorded turn with no ``tool_calls`` (text-only answer,
parse miss) is a screen no-op; ``_load_kept_turns`` keeps only ACTING turns **with their
original turn_name**, and both the replay and the original-screenshot flatten key on that
same set — otherwise the diff would compare mismatched states from the first skipped turn on.

We deliberately do NOT reuse ``replay_trajectory._load_actions`` (it discards turn identity);
we read the same ``lite_message.tool_calls`` field but keep names. ``visual_diff`` does the
diff; ``task_map`` is the UUID join.

Running a real replay needs a KVM host with BOTH env images fresh (osworld = VM-in-Docker).
Without that, ``--dry-run`` exercises the whole wiring (map + turn load + plan) with no docker.

Run:
    # dry-run (no docker): print the plan + acting-turn counts
    uv run python devs/envs/lite.osworld/bridge/cross_replay.py \\
        osworld_chrome_1704f00f --lite-rollout <sample_dir> --osworld-rollout <sample_dir> --dry-run

    # real (KVM host, both images built). For a DIVERGENT lite image, point the lite side at a
    # §1 private tag via LITE_OSWORLD_CONFIG (NOT an image= kwarg — it can't pass the gate, §1):
    LITE_OSWORLD_CONFIG=/tmp/mine.yaml \\
    uv run python devs/envs/lite.osworld/bridge/cross_replay.py osworld_chrome_1704f00f \\
        --lite-rollout .logs/rollout/gpt-5.5/lite.osworld/<ts>/eval/osworld_chrome_1704f00f/sample_00 \\
        --osworld-rollout .logs/rollout/gpt-5.5/osworld/<ts>/eval/<uuid>/sample_00
    # → outputs under .logs/bridge/<uuid>/ (orig_lite, orig_osworld, lite_to_osworld,
    #   osworld_to_lite, diff_lite_to_osworld, diff_osworld_to_lite)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import task_map  # noqa: E402  (sibling)
import visual_diff  # noqa: E402  (sibling)

from lite.core.tools.calls import tool_call_name, validate_lite_tool_call  # noqa: E402
from lite.gym.types import LiteEnvStepResult  # noqa: E402
from lite.infer.debug.log_layout import (  # noqa: E402
    ACTIONS_JSON,
    last_turn_image,
    turn_dirs,
    turn_timing_path,
)


def _load_kept_turns(sample_dir: pathlib.Path) -> list[tuple[str, list[dict], float]]:
    """``[(turn_name, tool_calls, think_s)]`` for turns that actually ACTED. Reads
    ``lite_message.tool_calls`` (the canonical portable ``LiteToolCall`` list, NOT the
    env-lowered ``executed_actions``) from each ``turn_NNNN/03_actions.json``, keeping the
    original ``turn_name``. Turns with no tool_calls (text-only answers / parse misses —
    screen no-ops) are skipped so replay and the original flatten key on the SAME set.

    ``think_s`` = the agent's recorded ``predict`` time for this turn (from
    the shared debug-layout timing helper) — how long the agent OBSERVED-then-thought before firing this
    action. Replaying that pace (not a fixed guess) is what lets a UI opened by the
    PREVIOUS action settle before this one fires; a too-short sleep is why open-loop
    replay drifts on focus/WM-sensitive actions (e.g. a ``ctrl+alt+t`` terminal). Falls
    back to 0 when unrecorded."""
    kept: list[tuple[str, list[dict], float]] = []
    for turn_dir in turn_dirs(sample_dir):
        af = turn_dir / ACTIONS_JSON
        if not af.exists():
            continue
        tool_calls = (json.loads(af.read_text()).get("lite_message") or {}).get("tool_calls") or []
        if not tool_calls:
            continue
        for index, tool_call in enumerate(tool_calls):
            reason = validate_lite_tool_call(
                tool_call,
                f"{turn_dir.name}/{ACTIONS_JSON}.lite_message.tool_calls[{index}]",
                require_id=False,
            )
            if reason is not None:
                raise ValueError(reason)
        tf = turn_timing_path(turn_dir)
        think = float(json.loads(tf.read_text()).get("predict", 0.0)) if tf is not None else 0.0
        kept.append((turn_dir.name, tool_calls, think))
    return kept


def _flatten_original(sample_dir: pathlib.Path, dst: pathlib.Path,
                      turn_names: list[str]) -> pathlib.Path:
    """Copy the ORIGINAL pre-action image (current ``turn_NNNN/prompt_images/*.png`` or legacy equivalents) into
    ``dst/turn_NNNN.png`` for ONLY the kept ``turn_names`` — aligning 1:1 with what
    ``replay_on`` saves (the pre-action obs of each kept turn).
    Copying every turn (incl. skipped no-op turns) would misalign the diff."""
    dst.mkdir(parents=True, exist_ok=True)
    for name in turn_names:
        shot = last_turn_image(sample_dir / name)
        if shot is not None:
            (dst / f"{name}.png").write_bytes(shot.read_bytes())
    return dst


def pair_for(lite_task_id: str) -> task_map.Pair:
    """The (lite_key, osworld_key) pair for a lite task_id (``osworld_<domain>_<hash8>``)."""
    for p in task_map.load_pairs(include_excluded=True):
        if p.lite_key == f"lite.osworld@{lite_task_id}":
            return p
    raise SystemExit(f"{lite_task_id!r} not in the task map (see task_map.py)")


async def replay_on(env_key: str, turns: list[tuple[str, list[dict], float]], out_dir: pathlib.Path,
                    *, pace_scale: float = 1.0, min_pace: float = 1.0, **env_kwargs) -> dict:
    """Open-loop replay of ``turns`` (``[(turn_name, LiteToolCalls, think_s)]``) on ``env_key`` — any
    gym-wrapped desktop env, all sharing ``LiteDesktopActionSpace``, so the list is portable.
    Saves each kept turn's **pre-action** observation under its ORIGINAL ``turn_name`` (aligns 1:1
    with :func:`_flatten_original`), then — critically — **waits ``think_s`` (the agent's recorded
    predict time) BEFORE firing that turn's action**, reproducing the closed-loop pace so a UI the
    PREVIOUS action opened has time to settle. A fixed short sleep instead is why open-loop replay
    drifts on focus/WM-sensitive actions (a ``ctrl+alt+t`` terminal that hadn't appeared yet). Env's
    own ``post_action_delay`` still applies inside ``step`` (after the action, before the screenshot);
    this pace is the additional observe→think gap on top of it. ``pace_scale`` dilates it (e.g. 1.5×
    for a slower substrate); ``min_pace`` floors unrecorded/tiny values.

    Returns ``{n_turns, out_dir}``. Reward is intentionally NOT returned — a trustworthy score needs
    the task evaluator, not the per-step value; the parity signal is the screenshot diff (§3). For a
    DIVERGENT lite image, set ``LITE_OSWORLD_CONFIG`` before invoking (NOT an ``image=`` kwarg — §1)."""
    import lite.gym as gym

    out_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(env_key, max_steps=len(turns) + 1, **env_kwargs)

    def _save(png: bytes | None, name: str) -> None:
        if png:
            (out_dir / f"{name}.png").write_bytes(png)

    def _first_result_image(result: LiteEnvStepResult) -> bytes | None:
        for tool_result in result.results:
            if tool_result.images:
                return tool_result.images[-1]
        return None

    obs = await env.reset()
    current_image = obs.image
    try:
        for turn_name, actions, think in turns:
            _save(current_image, turn_name)                   # PRE-action obs (what the agent saw)
            await asyncio.sleep(max(think * pace_scale, min_pace))  # mimic the agent's observe→think gap
            step_result = await env.step(actions)             # env then waits post_action_delay + screenshots
            current_image = _first_result_image(step_result)
            if step_result.terminated or step_result.truncated:
                break
    finally:
        await env.close()
    return {"n_turns": len(turns), "out_dir": str(out_dir)}


async def bridge(lite_task_id: str, lite_rollout: pathlib.Path, osworld_rollout: pathlib.Path,
                 out: pathlib.Path, *, pace_scale: float,
                 lite_env_kwargs: dict, osworld_env_kwargs: dict) -> None:
    """Both directions for one task: replay each substrate's recorded acting-turns on the
    OTHER substrate (at the agent's recorded pace), then diff each replay against the
    ORIGINAL substrate's pre-action screenshots (same kept-turn set → aligned)."""
    p = pair_for(lite_task_id)

    # lite→osworld: replay lite's actions on osworld, diff vs the lite original.
    lite_turns = _load_kept_turns(lite_rollout)
    await replay_on(p.osworld_key, lite_turns, out / "lite_to_osworld",
                    pace_scale=pace_scale, **osworld_env_kwargs)
    d1 = visual_diff.diff_dirs(
        _flatten_original(lite_rollout, out / "orig_lite", [t[0] for t in lite_turns]),
        out / "lite_to_osworld", out / "diff_lite_to_osworld")

    # osworld→lite: replay osworld's actions on lite, diff vs the osworld original.
    osw_turns = _load_kept_turns(osworld_rollout)
    await replay_on(p.lite_key, osw_turns, out / "osworld_to_lite",
                    pace_scale=pace_scale, **lite_env_kwargs)
    d2 = visual_diff.diff_dirs(
        _flatten_original(osworld_rollout, out / "orig_osworld", [t[0] for t in osw_turns]),
        out / "osworld_to_lite", out / "diff_osworld_to_lite")

    for label, d in (("lite→osworld", d1), ("osworld→lite", d2)):
        worst_ssim = min(d, key=lambda x: x["ssim"]) if d else {"ssim": 1.0, "turn": "-"}
        worst_de = max(d, key=lambda x: x["de_p95"]) if d else {"de_p95": 0.0, "turn": "-"}
        print(f"{label}: turns={len(d)} worst-SSIM={worst_ssim['ssim']:.4f}@{worst_ssim['turn']} "
              f"worst-ΔE-p95={worst_de['de_p95']:.2f}@{worst_de['turn']} "
              f"flagged={sum(1 for x in d if x['flagged'])}  (out {out})")


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("lite_task_id", help="e.g. osworld_chrome_1704f00f")
    ap.add_argument("--lite-rollout", type=pathlib.Path, help="sample dir of a lite.osworld recording")
    ap.add_argument("--osworld-rollout", type=pathlib.Path, help="sample dir of an osworld recording")
    ap.add_argument("--pace-scale", type=float, default=1.0,
                    help="dilate the recorded per-turn think-time (1.5 for a slower substrate)")  # divergent lite image → export LITE_OSWORLD_CONFIG (§1)
    ap.add_argument("-o", "--out", type=pathlib.Path, default=None,
                    help="output dir (default .logs/bridge/<uuid>)")
    ap.add_argument("--dry-run", action="store_true", help="no docker: print the plan + acting-turn counts")
    args = ap.parse_args()

    p = pair_for(args.lite_task_id)
    out = args.out or pathlib.Path(".logs/bridge") / p.uuid
    print(f"pair: {p.lite_key}  <->  {p.osworld_key}  (uuid {p.uuid}, domain {p.domain}"
          + (f", EXCLUDED={p.exclude_reason}" if p.exclude_reason else "") + f")  out={out}")
    if args.dry_run:
        for tag, d in (("lite", args.lite_rollout), ("osworld", args.osworld_rollout)):
            if d and d.exists():
                kept = _load_kept_turns(d)
                print(f"  {tag} recording {d}: {len(kept)} acting turns; "
                      f"first={[tool_call_name(tc) for tc in kept[0][1]] if kept else '-'}")
            else:
                print(f"  {tag} recording: {d} (absent)")
        print("dry-run OK (no docker touched)")
        return
    if not (args.lite_rollout and args.osworld_rollout):
        raise SystemExit("need --lite-rollout and --osworld-rollout for a real run (or --dry-run)")
    asyncio.run(bridge(args.lite_task_id, args.lite_rollout, args.osworld_rollout, out,
                       pace_scale=args.pace_scale,
                       lite_env_kwargs={}, osworld_env_kwargs={}))


if __name__ == "__main__":
    _main()
