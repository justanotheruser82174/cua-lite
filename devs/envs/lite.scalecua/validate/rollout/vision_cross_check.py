"""Produce ``metadata.others.vision_done`` so filter.py's dormant ``reward_vision_disagree``
soft tag fires — the missing producer for the landed adjudication rule (filter c58b556d).

For every ``sample_*`` under ``--log-root`` it samples multi-frame trajectory frames + the
instruction, asks a VLM whether the task was visually completed, and applies the ONE shared
adjudication rule (``filter._reward_vision_disagree``). Domain-agnostic: works for any env
with trajectory.parquet + frames + episode_return.

Run (writeback -> then filter lights the tag up):
    uv run python devs/envs/lite.scalecua/validate/rollout/vision_cross_check.py \
        --log-root .data/rollout/lite.osworld/gpt/d83c3e4c/train.synth \
        --out      .data/rollout/lite.osworld/gpt/d83c3e4c/train.synth_vision \
        --mode writeback --workers 8
    uv run python devs/data/lite.osworld/filter.py \
        --log-root .data/rollout/lite.osworld/gpt/d83c3e4c/train.synth_vision --out <annotated>

Run (census sidecar):
    ... --mode sidecar --sidecar-out .exps/vision/<run>/sidecars \
        --trace .exps/vision/<run>/uncertain.jsonl

Needs: a reachable VLM endpoint and a collected rollout root. The judge is
``lite.gym.envs.lite.cuaworld.src.vlm``, so a MODEL ID is the only required
configuration and it already defaults to ``gpt-4o``: litellm infers the provider
from the id and reads that provider's credentials/base URL from the environment
(``OPENAI_BASE_URL``/``OPENAI_API_KEY`` for OpenAI ids, ``ANTHROPIC_API_KEY``
for ``anthropic/…``, and so on). Override per run with ``VLM_MODEL`` /
``VLM_BASE_URL`` / ``VLM_API_KEY`` or the matching ``--vlm-*`` flags, e.g. a
local server: ``--vlm-model openai/Qwen/Qwen3-VL-8B-Instruct --vlm-base-url
http://localhost:8080/v1``.

Read-only w.r.t. --log-root; only --out / --sidecar-out / --trace are written.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[5]  # rollout/validate/lite.scalecua/envs/devs/<repo>


def _load(path: Path, name: str):
    """Import a module by file path (dotted dirs like ``lite.osworld`` are NOT packages)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


vision_gate = _load(_HERE.parent / "vision_gate.py", "vision_gate")
flt = _load(_REPO / "devs" / "data" / "lite.osworld" / "filter.py", "osworld_filter")

from lite.gym.envs.lite.cuaworld.src.vlm import query_vlm, sample_trajectory_frames  # noqa: E402

_PROMPT = (
    "You are auditing a computer-use agent trajectory. The user's task was:\n\n"
    "  {instruction}\n\n"
    "The images are screenshots in temporal order (first -> last) of the agent's session. "
    "Judge ONLY from the final visual state (earlier frames are context) whether the task "
    "was actually completed. Reply with strict JSON and nothing else:\n"
    '  {{"done": true|false, "confidence": "high"|"medium"|"low", "reasoning": "<one line>"}}'
)


def _coerce_meta(md: Any) -> dict[str, Any]:
    md = flt.to_plain(md)
    if isinstance(md, str):
        md = json.loads(md)
    return md or {}


def _first_user_text(messages: Any) -> str:
    """First user-turn text. ``flt.to_plain`` flattens the parquet ndarray columns (messages
    AND each message's ``content``) to plain lists so no numpy-truthiness guard can crash."""
    if isinstance(messages, str):
        messages = json.loads(messages)
    for m in flt.to_plain(messages) or []:
        if isinstance(m, dict) and m.get("role") == "user":
            return "\n".join(
                str(p.get("text") or "")
                for p in (m.get("content") or [])
                if isinstance(p, dict) and p.get("type") in {"text", "inline_reasoning"}
            ).strip()
    return ""


def _traj_details(trajectory_path: str, frame_paths: dict[str, str]) -> tuple[str, list[str]]:
    """Domain-agnostic (instruction, frames) from the trajectory parquet."""
    p = Path(trajectory_path)
    if not p.exists():
        return "", [v for v in (frame_paths or {}).values() if v]
    row = pd.read_parquet(p).iloc[0]
    others = _coerce_meta(row.get("metadata")).get("others") or {}
    instruction = str(
        others.get("instruction") or others.get("task") or others.get("goal")
        or _first_user_text(row.get("messages")) or ""
    )
    imgs = row.get("images")
    frames = (
        [str(x) for x in imgs]
        if imgs is not None and len(imgs)
        else [v for v in (frame_paths or {}).values() if v]
    )
    return instruction, [f for f in frames if f and Path(f).exists()]


def _judge(instruction: str, frames: list[str], config: dict | None) -> tuple[bool | None, str]:
    """Return (vision_done, note). None => uncertain => route to trace (#155 multi-frame)."""
    if not frames:
        return None, "no frames"
    sampled = sample_trajectory_frames({"frames": frames}, num_samples=min(4, len(frames)))
    res = query_vlm(
        _PROMPT.format(instruction=instruction or "(instruction unavailable)"),
        images=sampled, config=config,
    )
    if not res["success"]:
        return None, f"vlm error: {res['error']}"
    parsed = res["parsed"] or {}
    done = parsed.get("done")
    if not isinstance(done, bool):
        done = parsed.get("answer")
    if not isinstance(done, bool) or parsed.get("confidence") == "low":
        return None, f"uncertain: {res['response'][:160]}"
    return done, str(parsed.get("reasoning") or res["response"][:160])


def _set_vision_done(md_raw: Any, vision_done: bool) -> Any:
    """Encoding-preserving write of others.vision_done; never touches episode_return."""
    was_str = isinstance(md_raw, str)
    md = dict(json.loads(md_raw) if was_str else flt.to_plain(md_raw) or {})
    others = dict(md.get("others") or {})
    others["vision_done"] = bool(vision_done)
    md["others"] = others
    return json.dumps(md) if was_str else md


def _writeback(
    trajectory_path: str,
    log_root: Path,
    out_root: Path,
    vision_done: bool | None,
) -> None:
    # `--out` is a FULL MIRROR, not a delta-overlay: an uncertain
    # (`not_visually_decidable`) row is copied through UNMODIFIED so that a
    # downstream `filter.py --log-root <out>` sees the whole corpus and never
    # silently drops the ~7% uncertain trajectories.
    src = Path(trajectory_path)
    df = pd.read_parquet(src)
    if vision_done is not None:
        df["metadata"] = [_set_vision_done(m, vision_done) for m in df["metadata"]]
    dst = out_root / src.resolve().relative_to(log_root.resolve())
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)


def _sidecar(
    row: dict,
    queue_path: Path,
    vision_done: bool | None,
    note: str,
    disagree: bool,
) -> dict:
    if vision_done is None:
        label, vsucc = "not_visually_decidable", False
    else:
        label = "true_success" if vision_done else "true_failure"
        vsucc = bool(vision_done)
    sampled_ok = [v for v in (row.get("frame_paths") or {}).values() if v and Path(v).exists()]
    return {
        "schema_version": vision_gate.SCHEMA_VERSION, "run_id": row.get("run_id"),
        "source_queue_path": str(queue_path), "queue_id": row["queue_id"],
        "sample_id": row["sample_id"], "task_id": row["task_id"], "split": row["split"],
        "domain": row["domain"], "log_dir": row["log_dir"],
        "reported_reward": row.get("reported_reward"),
        "reported_terminated": row.get("reported_terminated"),
        "reported_truncated": row.get("reported_truncated"),
        "setup_ok": True, "action_ok": True, "eval_ok": True,
        "visual_label": label, "visual_success": vsucc, "notes": note,
        "reviewer": "vision_cross_check", "checked_at": dt.datetime.now(dt.UTC).isoformat(),
        "latest_turn": row.get("latest_turn"), "result_json_path": row.get("result_json_path", ""),
        "actions_path": row.get("actions_path", ""), "response_path": row.get("response_path", ""),
        "frame_paths": {f"frame_{i}": p for i, p in enumerate(sampled_ok)},
        "artifact_paths": {}, "probe_paths": {}, "vision_done": vision_done,
        "reward_vision_disagree": disagree,
        "probe_status": "pending" if vision_done is None else "none",
    }


def _process(row: dict, config: dict | None) -> dict:
    instruction, frames = _traj_details(row["trajectory_path"], row.get("frame_paths") or {})
    vision_done, note = _judge(instruction, frames, config)
    metadata = {
        "episode_return": row.get("reported_reward"),
        "others": {"vision_done": vision_done},
    }
    disagree = flt._reward_vision_disagree(metadata)  # the ONE shared rule (bool only)
    return {"row": row, "vision_done": vision_done, "note": note, "disagree": disagree}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--log-root", required=True, type=Path)
    ap.add_argument("--mode", choices=("sidecar", "writeback"), default="writeback")
    ap.add_argument("--out", type=Path, help="writeback: mirrored root for annotated parquets")
    ap.add_argument("--sidecar-out", type=Path, help="sidecar: dir for per-queue_id JSON")
    ap.add_argument("--trace", type=Path, help="JSONL of uncertain rows routed to evaluator probe")
    ap.add_argument("--workers", type=int, default=8)
    # Only what `vlm.get_vlm_config` actually consumes: the model id (litellm
    # resolves the provider from it) plus two optional overrides. There is no
    # backend switch to expose.
    ap.add_argument("--vlm-model")
    ap.add_argument("--vlm-base-url")
    ap.add_argument("--vlm-api-key")
    args = ap.parse_args()
    if args.mode == "writeback" and not args.out:
        ap.error("--out is required for --mode writeback")
    if args.mode == "sidecar" and not args.sidecar_out:
        ap.error("--sidecar-out is required for --mode sidecar")
    config = {
        k: v
        for k, v in {
            "model": args.vlm_model,
            "base_url": args.vlm_base_url,
            "api_key": args.vlm_api_key,
        }.items()
        if v
    } or None

    rows = vision_gate.build_queue(
        args.log_root,
        catalog={},  # domain-agnostic (no scalecua catalog)
    )
    queue_path = args.log_root  # provenance only; sidecars are self-describing
    n_dis = n_unc = 0
    trace: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_process, r, config): r for r in rows}
        for fut in as_completed(futs):
            out = fut.result()
            row, vd, note, dis = out["row"], out["vision_done"], out["note"], out["disagree"]
            n_dis += int(dis)
            n_unc += int(vd is None)
            if vd is None:
                trace.append({"queue_id": row["queue_id"], "log_dir": row["log_dir"], "note": note})
            if args.mode == "writeback":
                # Full mirror: certain rows get vision_done stamped, uncertain rows
                # are copied through unmodified (see _writeback) so `--out` is a
                # complete corpus for downstream filter.py, never a lossy delta.
                _writeback(row["trajectory_path"], args.log_root, args.out, vd)
            else:
                sc = _sidecar(row, queue_path, vd, note, dis)
                dst = args.sidecar_out / f"{row['queue_id'].replace('/', '__')}.json"
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(json.dumps(sc, indent=2, sort_keys=True) + "\n")
    if args.trace and trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        args.trace.write_text("\n".join(json.dumps(t) for t in trace) + "\n")
    print(f"{len(rows)} trajectories | disagreements={n_dis} | uncertain(routed to trace)={n_unc}")
    if args.mode == "writeback":
        print(f"vision_done written under {args.out} -- now run filter.py --log-root {args.out}")


if __name__ == "__main__":
    main()
