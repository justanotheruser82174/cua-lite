"""Validate lite_osworld task rows by launching real Docker containers.

For each task:
  1. setup
  2. eval WITHOUT oracle → must score 0.0 (trivial-pass check)
  3. oracle replay
  4. eval WITH oracle → must score 1.0

    Usage:
    uv run python devs/envs/lite.osworld/validate/oracle/validate.py \\
        --fixtures lite/gym/envs/lite/osworld/data/train.synth.jsonl \\
        --concurrency 4 --retries 3 \\
        --report /tmp/validate_train_synth.report.jsonl

The shared CLI flags (--fixtures/--filter/--limit/--concurrency/--retries/
--oracle-timeout/--report/--resume-from/--session-id) are kept identical to the
lite.scalecua oracle validator so one usage model works for both.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SESSION_ID_PREFIX = "lite_osworld_oracle_validate"
_DEFAULT_CONFIG = _REPO_ROOT / "lite/gym/envs/lite/osworld/configs/default.yaml"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _configure_image_override() -> None:
    """Point the env's baked container shape at ``CUA_LITE_VALIDATE_IMAGE``.

    Instead of mutating module globals at runtime, use the env's own config
    mechanism (``lite/gym/utils/config/defaults.py``): write a temp yaml that copies the
    env's ``configs/default.yaml`` with ``env_kwargs.computer.image`` swapped to
    ``$CUA_LITE_VALIDATE_IMAGE``, then point ``LITE_OSWORLD_CONFIG`` at it.
    ``config.load`` reads ``<PREFIX>_CONFIG`` once at first call (env-module
    import) and is ``lru_cache``'d, so the module-level
    ``_IMAGE = CFG.env_kwargs["computer"]["image"]`` and the derived
    ``_COMPUTER_CONFIG`` both resolve to the override with NO module patching.
    MUST be called before the first import of the env module. No-op unless the
    env var is set (the config default already carries the shipping tag).
    """
    image = os.environ.get("CUA_LITE_VALIDATE_IMAGE")
    if not image:
        return
    cfg = yaml.safe_load(_DEFAULT_CONFIG.read_text())
    cfg["env_kwargs"]["computer"]["image"] = image
    fd, path = tempfile.mkstemp(prefix="lite_osworld_validate_", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(cfg, f)
    os.environ["LITE_OSWORLD_CONFIG"] = path
    logger.info("Validating against image: %s (via LITE_OSWORLD_CONFIG=%s)", image, path)


def _safe_session_fragment(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text)
    return safe.strip("_") or "run"


def _default_session_id(args: argparse.Namespace) -> str:
    source = args.report or args.fixtures
    return f"{_SESSION_ID_PREFIX}_{_safe_session_fragment(Path(source).stem)}"


def _validate_container_prefix(session_id: str) -> str:
    return f"lite-env-{session_id}-_validate"


def _sweep_validate_containers(session_id: str) -> int | None:
    """Remove containers belonging to this validator session only.

    Returns the number **actually removed** — the ids ``docker rm`` echoed back
    — or ``None`` when the count is unknowable: docker could not be asked at all,
    or the removal itself blew its budget. ``None`` is NOT ``0``: a wedged daemon
    that times out ``docker ps`` otherwise reads as "this session had no
    containers", so the caller logs nothing and every container is left behind.
    Never raises, so the two callers that run from a ``finally:`` / a signal
    handler always reach their next statement.

    **The removed count comes from rm's stdout, not from what ``ps`` found.**
    Those two differ in practice: a container already being torn down by its own
    env fails with ``removal of container <id> is already in progress``, and this
    function used to return ``len(found)`` regardless — reporting "reaped 4"
    while four ``Exited (137)`` containers stayed on the host. Neither the exit
    code nor ``check=True`` can catch it: ``docker rm -f`` exits **0** on a failed
    removal (measured — ``-f`` swallows "No such container" into success), and
    prints nothing on stdout while the reason goes to stderr. So stdout is the
    only receipt, which is the same signal
    ``lite/gym/utils/backend/docker.py::docker_rm_f`` reads for its own 1/0.
    """
    prefix = _validate_container_prefix(session_id)
    try:
        found = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"name={prefix}"],
            capture_output=True, text=True, timeout=15, check=False,
        ).stdout.split()
        if not found:
            return 0
        # ``-v``, per lite/gym/utils/backend/docker.py::_rm_argv: it is part of
        # the framework's definition of removal, not an option — a force-rm
        # without it orphans osworld's anonymous ``/storage`` volume forever.
        rm = subprocess.run(["docker", "rm", "-f", "-v", *found],
                            capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    removed = set(rm.stdout.split()) & set(found)
    if len(removed) != len(found):
        # LOUD, and with the daemon's own words: a sweep that cannot remove
        # is the one event that strands containers on the host, and it was
        # previously routed to DEVNULL.
        logger.warning(
            "sweep: removed %d of %d container(s) matching %s — %d SURVIVED: %s; "
            "docker rm stderr: %s",
            len(removed), len(found), prefix, len(found) - len(removed),
            " ".join(sorted(set(found) - removed)),
            (rm.stderr or "").strip()[:1000] or "(none)",
        )
    return len(removed)


def _install_signal_cleanup(session_id: str) -> None:
    """On SIGTERM/SIGINT, reap this run's containers then exit. Covers the
    `timeout`-wrapped invocations the §12 fix-loop uses (SIGTERM) and Ctrl-C
    (SIGINT). ``os._exit`` runs from a ``finally:``, so no sweep outcome can keep
    this process alive — a sweep that raised used to unwind past the exit call
    and leave the process running on SIGTERM.

    Residues that survive this handler by design: an uncatchable SIGKILL, and a
    container the daemon materialises from a ``docker run`` it had already
    accepted. So ``docker ps -a --filter name=<prefix>`` AFTER this process exits
    is the authority, never this handler's count.
    """
    def _handler(signum, _frame):
        try:
            n = _sweep_validate_containers(session_id)
            logger.warning("signal %d: reaped %s validate container(s)", signum,
                           "UNKNOWN (docker unreachable)" if n is None else n)
        finally:
            os._exit(130)
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=str, required=True,
                        help="JSONL of task rows / oracle fixtures to validate — each row carries "
                             "oracle_actions (eval.jsonl / train.synth.jsonl / train.perturb.jsonl)")
    parser.add_argument("--concurrency", type=_positive_int, default=4)
    parser.add_argument(
        "--retries", type=_positive_int, default=1,
        help="Max attempts per task. A task is FAIL only if all attempts fail. "
             "Lets transient noise (concurrent VLC/Chrome contention, brief network "
             "blips, accessibility-tree races) self-heal. (default: 1)",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Comma-separated task_id list — validate only these. Supports "
             "exact ids ('synth_calc_sort_0001') or substrings ('calc_sort'). "
             "Used by the §12 step 7 iterative-fix loop to re-validate only "
             "the rows just patched.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Validate only the first N rows after --filter (quick smoke slice).",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Path to a previous --report JSONL. Tasks that already passed "
             "(passed=true) are skipped; their results are merged into the "
             "new report. Use this to restart a validate after a crash or kill.",
    )
    parser.add_argument(
        "--oracle-timeout", type=float, default=180.0,
        help="Per-oracle-action timeout in seconds. Actions that exceed this "
             "(e.g. hung soffice --headless) are killed and the task is marked FAIL. "
             "(default: 180)",
    )
    parser.add_argument(
        "--report", type=str, default=None,
        help="Write per-task pass/fail to this JSONL (one record per task: "
             "{task_id, passed, message}). Used by the iterative-fix loop "
             "to triage failures programmatically.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help=(
            "Direct-mode container session id. The validator only sweeps "
            "containers whose names match this session."
        ),
    )
    return parser.parse_args()


async def validate_task_spec(spec: dict, oracle_timeout: float = 180.0) -> tuple[bool, str]:
    """Validate one task in a real Docker container. Returns (passed, message)."""
    import lite.gym as gym
    from lite.gym.envs.lite.osworld.main import LiteOsworldEnv
    from lite.gym.envs.lite.osworld.src.utils.setup import setup_fn
    from lite.gym.envs.lite.osworld.src.utils.verify import evaluate_final_fn
    from lite.gym.sandbox import SandboxTaskConfig, register_tasks

    task_id = spec["task_id"]
    meta = spec["metadata"]
    others = meta.get("others") or {}
    task = SandboxTaskConfig(
        task_id=task_id,
        instruction=spec["instruction"],
        platform="desktop",
        # The container shape (image, memory, cpu, display, timeout) is fixed
        # by LiteOsworldEnv.__init__ from the config-derived _COMPUTER_CONFIG;
        # a per-task computer["image"] never reaches boot(), so leave it empty.
        computer={},
        max_steps=spec.get("max_steps", 15),
        metadata=meta,
        setup_fn=setup_fn,
        evaluate_final_fn=evaluate_final_fn,
    )

    temp_env_id = f"_validate_{task_id}"
    # env_class=LiteOsworldEnv is REQUIRED: the default SandboxBaseEnv.bind() has
    # no `expose_oracle` kwarg, so gym.make(..., expose_oracle=True) below raises
    # `TypeError: bind() got an unexpected keyword argument 'expose_oracle'` on
    # every run. LiteOsworldEnv.bind() accepts it (main.py:bind). The booted
    # image comes from the config-derived _COMPUTER_CONFIG, which main() points
    # at CUA_LITE_VALIDATE_IMAGE via LITE_OSWORLD_CONFIG before any env import.
    register_tasks(temp_env_id, {"train": [task]}, env_class=LiteOsworldEnv)

    # Match the framework/production reset budget (registry.py default = 600s),
    # NOT a tighter 300s: the heaviest multi_apps configs (3-app cold-start under
    # CPU-rendered Xvnc) legitimately need >300s but <600s, so a 300s validator
    # budget produces false-negative reset timeouts on tasks that are fine in
    # training. Keep the validator consistent with how the env actually runs.
    # expose_oracle=True: oracle-only metadata keys (oracle_actions /
    # oracle_trajectory / oracle_after_postconfig) are stripped from the exposed
    # metadata by default (see LiteOsworldEnv._task_metadata) — the validator
    # reads them from env.metadata.others, so re-expose them here.
    env = gym.make(f"{temp_env_id}@{task_id}", reset_timeout=600.0, expose_oracle=True)
    try:
        logger.info("[%s] Launching...", task_id)
        await env.reset()

        # --- Trivial-pass check: eval must fail before oracle ---
        # Deepcopy the evaluator so we don't mutate meta (postconfig flag, etc.)
        # and skip postconfig to avoid modifying state before oracle runs.
        # Exception: oracle_after_postconfig tasks need postconfig to run
        # before the evaluator can be meaningful (e.g. compare_table needs the
        # CSV that postconfig generates). For these tasks the trivial-pass check
        # runs postconfig first, mirroring the oracle flow.
        from lite.gym.envs.lite.osworld.src.eval.runner import evaluate_osworld_task
        _pre_eval = copy.deepcopy(meta.get("evaluator", {}))
        if others.get("oracle_after_postconfig"):
            from lite.gym.envs.lite.osworld.src.utils.dispatch import dispatch_actions
            try:
                await asyncio.wait_for(
                    dispatch_actions(env._computer, _pre_eval.get("postconfig", [])),
                    timeout=oracle_timeout,
                )
            except asyncio.TimeoutError:
                return False, f"Pre-eval postconfig timed out after {oracle_timeout:.0f}s"
        _pre_eval["_postconfig_done"] = True  # skip postconfig for raw check
        pre_reward = await evaluate_osworld_task(env._computer, _pre_eval)
        if pre_reward != 0.0:
            return False, f"trivial_pass: eval returns {pre_reward} before oracle (no agent action needed)"
        logger.info("[%s] Pre-oracle eval: 0.0 ✓ (non-trivial)", task_id)

        # Oracle source: oracle_actions in dispatch format (absolute coords /
        # shell commands).
        oracle_actions = others.get("oracle_actions", [])

        if oracle_actions:
            # Some tasks need oracle AFTER evaluator postconfig
            if others.get("oracle_after_postconfig"):
                # Run the save/export postconfig before applying the oracle, but
                # never mutate the task metadata. Final evaluation must still
                # execute postconfig again through the production path; several
                # sheet_print oracles intentionally replace the workbook and
                # remove stale CSV sidecars after this point.
                evaluator = copy.deepcopy(meta.get("evaluator", {}))
                from lite.gym.envs.lite.osworld.src.utils.dispatch import dispatch_actions
                try:
                    await asyncio.wait_for(
                        dispatch_actions(env._computer, evaluator.get("postconfig", [])),
                        timeout=oracle_timeout,
                    )
                except asyncio.TimeoutError:
                    return False, f"Oracle postconfig timed out after {oracle_timeout:.0f}s"

            # Kill apps that hold files the oracle needs to modify.
            #
            # SIGKILL (not SIGTERM): SIGTERM lets LibreOffice run its
            # save-on-close handler, which races with the oracle's wget and
            # silently overwrites the wgetted gold with LO's in-memory copy.
            # For small files (calc xlsx ~6KB) LO's save finishes before the
            # 1-second oracle sleep; for larger files (impress pptx ~6MB) it
            # doesn't, and the file ends up 34 bytes short of gold → eval FAIL.
            # SIGKILL skips the save handler entirely, leaving the on-disk file
            # untouched so wget produces a byte-identical gold.
            #
            # `killall -9` (psmisc) matches /proc/$pid/comm exactly, no
            # self-match (the pkill -f form killed the parent bash before the
            # pgrep-wait loop could run — cycle-45 LO calc regression).
            await env._computer.interface.run_command(
                "killall -9 -q chrome soffice.bin oosplash gimp gimp-2.10 "
                "vlc thunderbird code 2>/dev/null; "
                "true"
            )
            await asyncio.sleep(2)
            # Wait for apps to fully exit. SIGKILL'd processes are reaped by
            # the kernel almost immediately, but the kernel still needs a tick
            # to remove them from the proc table — keep the safety poll.
            await env._computer.interface.run_command(
                "for i in $(seq 1 20); do "
                "  { pgrep chrome || pgrep thunderbird || "
                "    pgrep soffice || pgrep gimp || "
                "    pgrep vlc || pgrep code; } "
                "    > /dev/null 2>&1 || break; "
                "  sleep 0.5; "
                "done; true"
            )

            from lite.gym.envs.lite.osworld.src.utils.dispatch import dispatch_action
            for i, action in enumerate(oracle_actions):
                t = action.get("type", "")
                logger.info("[%s] Oracle %d/%d: %s %s", task_id, i + 1, len(oracle_actions),
                            t, str(action.get("parameters", {}))[:60])
                try:
                    result = await asyncio.wait_for(
                        dispatch_action(env._computer, action),
                        timeout=oracle_timeout,
                    )
                except asyncio.TimeoutError:
                    cmd = action.get("parameters", {}).get("command", "")
                    if isinstance(cmd, list):
                        cmd = " ".join(cmd)
                    return False, f"Oracle action {i+1} timed out after {oracle_timeout:.0f}s: {cmd!r}"
                if t in ("execute", "command") and isinstance(result, dict):
                    rc = result.get("returncode", 0)
                    if rc != 0:
                        cmd = action.get("parameters", {}).get("command", "")
                        if isinstance(cmd, list):
                            cmd = " ".join(cmd)
                        return False, f"Oracle action {i+1} failed (rc={rc}): {cmd!r}\nstderr: {result.get('error', '')[:500]}"
        else:
            return False, "No oracle_actions"

        reward = await evaluate_final_fn(task, env._computer)
        if reward != 1.0:
            return False, f"Oracle scored {reward}, expected 1.0"

        return True, "ok"
    except Exception as e:
        return False, f"Exception: {e}"
    finally:
        await env.close()


async def main():
    # Parse first so `--help` / argument errors never touch Docker state.
    args = _parse_args()
    if args.concurrency < 1 or args.retries < 1:
        raise ValueError("--concurrency and --retries must be >= 1")

    # Force direct local-Docker mode. This validator launches its OWN
    # containers from the config image via gym.make; gym routes to a remote env-server
    # whenever CUA_LITE_ENV_SERVER_URL is set (registry.make/_import_env key on
    # it), so an ambient export in the shell would silently hijack every task
    # to a remote host. Clear it for this process's lifetime so validation
    # always runs against the local image being tested.
    os.environ.pop("CUA_LITE_ENV_SERVER_URL", None)
    # Scope Docker cleanup to this validation session. Do not clear or sweep
    # other sessions; concurrent validators must use distinct session ids.
    session_id = args.session_id or _default_session_id(args)
    os.environ["SESSION_ID"] = session_id
    swept = _sweep_validate_containers(session_id)
    if swept:
        logger.info("startup sweep: removed %d leftover validate container(s)", swept)
    _install_signal_cleanup(session_id)

    # Point the env at CUA_LITE_VALIDATE_IMAGE before any env import / gym.make
    # (via a temp LITE_OSWORLD_CONFIG) so every container boots from the tag
    # under test. Must run before validate_task_spec's lazy env import.
    _configure_image_override()

    # Load previously-passed results for resume support.
    resumed: dict[str, tuple[bool, str]] = {}
    if args.resume_from:
        try:
            with open(args.resume_from) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        if rec.get("passed"):
                            resumed[rec["task_id"]] = (True, rec.get("message", "resumed"))
            logger.info("Resume: loaded %d already-passed tasks from %s",
                        len(resumed), args.resume_from)
        except FileNotFoundError:
            logger.info("Resume file %s not found — starting fresh", args.resume_from)

    specs = []
    with open(args.fixtures) as f:
        for line in f:
            line = line.strip()
            if line:
                specs.append(json.loads(line))

    # Always skip infeasible / oracle-less rows: a row with no oracle_actions
    # cannot be replayed, and a row flagged others.exclude_reason (e.g.
    # "infeasible") is intentionally unsolvable inside Docker — both would score
    # 0 by construction and only add false failures. Skipping them is
    # unconditional (not an opt-in flag) so a plain run validates exactly the
    # solvable set.
    before = len(specs)
    specs = [
        s for s in specs
        if (s["metadata"].get("others") or {}).get("oracle_actions")
        and not s["metadata"].get("others", {}).get("exclude_reason")
    ]
    if before != len(specs):
        logger.info("Skipped %d infeasible/oracle-less rows (validating %d solvable)",
                    before - len(specs), len(specs))

    if args.filter:
        wanted = [w.strip() for w in args.filter.split(",") if w.strip()]
        before = len(specs)
        specs = [
            s for s in specs
            if any(w == s["task_id"] or w in s["task_id"] for w in wanted)
        ]
        logger.info("Filter %r: %d → %d rows", args.filter, before, len(specs))

    if args.limit is not None:
        before = len(specs)
        specs = specs[: args.limit]
        logger.info("Limit %d: %d → %d rows", args.limit, before, len(specs))

    # Skip tasks already confirmed-passed in a previous run.
    if resumed:
        before = len(specs)
        specs = [s for s in specs if s["task_id"] not in resumed]
        logger.info("Resume: skipping %d already-passed tasks (%d remaining)",
                    before - len(specs), len(specs))

    logger.info("Validating %d task(s) with concurrency=%d", len(specs), args.concurrency)

    sem = asyncio.Semaphore(args.concurrency)
    results: dict[str, tuple[bool, str]] = dict(resumed)  # seed with resumed passes

    # Report record format (one JSON object per line):
    #   {"task_id": str, "passed": bool, "trivial_pass": bool, "message": str}
    # Written incrementally — every completed task is flushed immediately so the
    # file is a valid checkpoint even if the process is killed mid-run.
    # Pass --resume-from <this file> on restart to skip already-passed tasks.
    def _write_record(f, tid: str, ok: bool, msg: str) -> None:
        f.write(json.dumps({
            "task_id": tid,
            "passed": ok,
            "trivial_pass": "trivial_pass:" in msg,
            "message": msg,
        }) + "\n")
        f.flush()

    report_f = open(args.report, "w") if args.report else None
    try:
        # Re-emit already-resumed passes so the file is self-contained.
        if report_f:
            for tid, (ok, msg) in resumed.items():
                _write_record(report_f, tid, ok, msg)

        async def _validate_one(spec):
            async with sem:
                tid = spec["task_id"]
                for attempt in range(1, args.retries + 1):
                    passed, msg = await validate_task_spec(spec, oracle_timeout=args.oracle_timeout)
                    if passed:
                        results[tid] = (True, f"ok (attempt {attempt}/{args.retries})")
                        logger.info("[%s] PASS — %s", tid, results[tid][1])
                        if report_f:
                            _write_record(report_f, tid, True, results[tid][1])
                        return
                    logger.info("[%s] attempt %d/%d FAIL — %s",
                                tid, attempt, args.retries, msg[:100])
                results[tid] = (False, f"all {args.retries} attempts failed: last={msg}")
                logger.info("[%s] FAIL (after %d) — %s",
                            tid, args.retries, msg[:100])
                if report_f:
                    _write_record(report_f, tid, False, results[tid][1])

        await asyncio.gather(*[_validate_one(s) for s in specs])
    finally:
        if report_f:
            report_f.close()
            logger.info("Per-task report → %s", args.report)
        # UNCONDITIONAL, unlike the startup line: the sweep is the last thing
        # that can still remove a container, so "removed 0" and "could not ask
        # docker" must not both print nothing. The line is always emitted, so its
        # ABSENCE means this finally never ran.
        swept = _sweep_validate_containers(session_id)
        logger.info("final sweep: removed %s validate container(s)",
                    "UNKNOWN (docker unreachable)" if swept is None else swept)

    n_pass = sum(1 for v in results.values() if v[0])
    n_fail = len(results) - n_pass
    logger.info("Results: %d/%d passed, %d failed", n_pass, len(results), n_fail)

    n_trivial = sum(1 for _, (ok, msg) in results.items()
                    if not ok and "trivial_pass:" in msg)
    if n_trivial:
        logger.info("Trivial-pass failures (%d):", n_trivial)
        for tid, (ok, msg) in results.items():
            if not ok and "trivial_pass:" in msg:
                logger.info("  %s: %s", tid, msg[:200])
    if n_fail > 0:
        n_other = n_fail - n_trivial
        if n_other:
            logger.info("Other failures (%d):", n_other)
            for tid, (ok, msg) in results.items():
                if not ok and "trivial_pass:" not in msg:
                    logger.info("  %s: %s", tid, msg[:200])
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
