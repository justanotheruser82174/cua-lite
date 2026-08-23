"""Replay curated lite.scalecua oracle fixtures in real containers.

For each fixture:
  1. reset a fresh env and run a no-op final eval; reward must be 0.0
  2. reset a fresh env, replay oracle actions, then run final eval; reward must
     match the fixture's expected reward, usually 1.0

This intentionally mirrors the lite.osworld oracle validator, but fixture rows
point to imported lite.scalecua `(split, task_id)` rows instead of embedding the
full task spec.

Run:
    uv run python devs/envs/lite.scalecua/validate/oracle/validate.py \
        --fixtures lite/gym/envs/lite/scalecua/data/oracle/rl.jsonl \
        --artifacts .exps/validate/lite.scalecua/oracle/<run-id> \
        --report .exps/validate/lite.scalecua/oracle/<run-id>.report.jsonl --concurrency 16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SESSION_ID_PREFIX = "lite_scalecua_oracle_validate"
_DEFAULT_CONFIG = _REPO_ROOT / "lite/gym/envs/lite/scalecua/configs/default.yaml"
_OSWORLD_DEFAULT_CONFIG = _REPO_ROOT / "lite/gym/envs/lite/osworld/configs/default.yaml"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _write_image_override(default_config: Path, image: str) -> str:
    """Write a temp copy of ``default_config`` with the container image swapped
    to ``image``; return the temp path."""
    cfg = yaml.safe_load(default_config.read_text())
    cfg["env_kwargs"]["computer"]["image"] = image
    fd, path = tempfile.mkstemp(prefix="lite_scalecua_validate_", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def _configure_image_override() -> None:
    """Point the shared lite.osworld container at ``CUA_LITE_VALIDATE_IMAGE``.

    Instead of mutating module globals at runtime, use the env's own config
    mechanism (``lite/gym/utils/config/defaults.py``): write a temp yaml that copies the
    env's ``configs/default.yaml`` with ``env_kwargs.computer.image`` swapped to
    ``$CUA_LITE_VALIDATE_IMAGE``, then point ``LITE_SCALECUA_CONFIG`` at it.
    ``config.load`` reads ``<PREFIX>_CONFIG`` once at first call (env-module
    import) and is ``lru_cache``'d, so the module-level ``_IMAGE`` and the
    derived ``_COMPUTER_CONFIG`` / ``_TASK_COMPUTER`` all resolve to the override
    with NO module patching. MUST be called before the first import of the env
    module. No-op unless the env var is set (the config default already carries
    the shipping tag).

    ``LiteScaleCuaEnv`` subclasses ``LiteOsworldEnv`` and reuses the lite.osworld
    container image, so ``lite.osworld``'s OWN module-level ``_IMAGE`` (driving
    its constructor-side ``_check_base_image`` freshness gate) must ALSO
    be pointed at the override — otherwise the shared osworld module resolves to
    the shipping ``:latest`` tag and the gate rejects it as stale the moment the
    refactor changes any baked source. Set BOTH ``LITE_SCALECUA_CONFIG`` and
    ``LITE_OSWORLD_CONFIG``.
    """
    image = os.environ.get("CUA_LITE_VALIDATE_IMAGE")
    if not image:
        return
    scalecua_path = _write_image_override(_DEFAULT_CONFIG, image)
    osworld_path = _write_image_override(_OSWORLD_DEFAULT_CONFIG, image)
    os.environ["LITE_SCALECUA_CONFIG"] = scalecua_path
    os.environ["LITE_OSWORLD_CONFIG"] = osworld_path
    logger.info(
        "Validating against image: %s (LITE_SCALECUA_CONFIG=%s, LITE_OSWORLD_CONFIG=%s)",
        image, scalecua_path, osworld_path,
    )


def _safe_session_fragment(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text)
    return safe.strip("_") or "run"


def _default_session_id(artifacts: Path) -> str:
    return f"{_SESSION_ID_PREFIX}_{_safe_session_fragment(artifacts.name)}"


def _container_name_filter(session_id: str) -> str:
    return f"lite-env-{session_id}-lite.scalecua"


def _sweep_own_containers(session_id: str) -> int | None:
    """Remove containers belonging to this oracle run only.

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
    function used to return ``len(found)`` regardless — reporting "removed 4"
    while four ``Exited (137)`` containers stayed on the host. Neither the exit
    code nor ``check=True`` can catch it: ``docker rm -f`` exits **0** on a failed
    removal (measured — ``-f`` swallows "No such container" into success), and
    prints nothing on stdout while the reason goes to stderr. So stdout is the
    only receipt, which is the same signal
    ``lite/gym/utils/backend/docker.py::docker_rm_f`` reads for its own 1/0.
    """
    name_filter = _container_name_filter(session_id)
    try:
        found = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"name={name_filter}"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).stdout.split()
        if not found:
            return 0
        rm = subprocess.run(
            ["docker", "rm", "-f", "-v", *found],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
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
            len(removed),
            len(found),
            name_filter,
            len(found) - len(removed),
            " ".join(sorted(set(found) - removed)),
            (rm.stderr or "").strip()[:1000] or "(none)",
        )
    return len(removed)


def _install_signal_cleanup(session_id: str) -> None:
    """On SIGTERM/SIGINT, reap this run's containers then exit.

    ``os._exit`` runs from a ``finally:``, so no sweep outcome can keep this
    process alive — a sweep that raised used to unwind past the exit call and
    leave the process running on SIGTERM. Residues that survive this handler by
    design: an uncatchable SIGKILL, and a container the daemon materialises from
    a ``docker run`` it had already accepted — so
    ``docker ps -a --filter name=<filter>`` AFTER exit is the authority.
    """
    def _handler(signum, _frame):
        try:
            removed = _sweep_own_containers(session_id)
            logger.warning("signal %d: removed %s oracle container(s)", signum,
                           "UNKNOWN (docker unreachable)" if removed is None else removed)
        finally:
            os._exit(130)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _write_png(path: Path, screenshot: bytes | None) -> None:
    if not screenshot:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(screenshot)


async def _capture_png(env: Any, path: Path) -> None:
    from lite.gym.sandbox.base import _take_screenshot

    _write_png(path, await _take_screenshot(env._computer))


def _response_action(text: str = "Done") -> list[dict[str, Any]]:
    """One env-internal ``response`` action — the final action both evals see.

    This oracle calls ``evaluate_final_fn(..., actions=...)`` directly, outside
    normal ``env.step`` ingress. Ingress projects canonical Lite calls to this
    bare env action shape before the verifier sees them; the oracle must do the
    same rather than passing a persisted Lite call.

    The text only reaches a verdict for an ``infeasible`` evaluator, which looks
    for ``[infeasible]`` in it; no other ``func`` reads it. Re-derive the census
    with ``coverage_inventory.py`` before assuming a fixture needs a custom
    ``noop_response`` / ``response_text``.
    """
    return [{"name": "response", "arguments": {"text": text}}]


def _reward_matches(actual: float, expected: float, *, tolerance: float = 1e-9) -> bool:
    return abs(float(actual) - float(expected)) <= tolerance


def _flush_counts_from_debug(debug: dict[str, Any]) -> dict[str, int]:
    if not isinstance(debug, dict):
        return {}
    direct = debug.get("flush_fired_counts")
    if isinstance(direct, dict):
        return {str(app): int(count or 0) for app, count in direct.items()}
    counters = debug.get("flush_counters")
    if isinstance(counters, dict):
        return {
            "vlc": int(counters.get("vlc_flush_fired", 0) or 0),
            "thunderbird": int(counters.get("thunderbird_flush_fired", 0) or 0),
        }
    stats = debug.get("flush_stats")
    if not isinstance(stats, dict):
        return {}
    counts: dict[str, int] = {}
    for app, item in stats.items():
        if not isinstance(item, dict):
            continue
        counts[str(app)] = int(item.get("needed", 0) or 0)
    return counts


def _flush_fired_aliases(counts: dict[str, int]) -> dict[str, int]:
    return {
        "vlc_flush_fired": int(counts.get("vlc", 0)),
        "thunderbird_flush_fired": int(counts.get("thunderbird", 0)),
    }


def _result_flush_counts(result: dict[str, Any]) -> dict[str, int]:
    counts = result.get("flush_fired_counts")
    if isinstance(counts, dict):
        return {str(app): int(count or 0) for app, count in counts.items()}
    replay = result.get("replay") or {}
    counts = replay.get("flush_fired_counts") if isinstance(replay, dict) else {}
    if isinstance(counts, dict):
        return {str(app): int(count or 0) for app, count in counts.items()}
    debug = replay.get("eval_debug") if isinstance(replay, dict) else {}
    return _flush_counts_from_debug(debug if isinstance(debug, dict) else {})


def _result_runtime_split(result: dict[str, Any]) -> str | None:
    replay = result.get("replay") or {}
    if isinstance(replay, dict) and replay.get("split") is not None:
        return str(replay.get("split"))
    if result.get("split") is not None:
        return str(result.get("split"))
    return None


def _build_summary(
    results: list[dict[str, Any]],
    *,
    require_rl_flush_fired: bool = False,
) -> dict[str, Any]:
    passed = sum(1 for result in results if result.get("passed"))
    failed = len(results) - passed
    flush_fired_counts: dict[str, int] = {}
    rl_flush_fired_counts: dict[str, int] = {}
    for result in results:
        counts = _result_flush_counts(result)
        for app, count in counts.items():
            flush_fired_counts[app] = flush_fired_counts.get(app, 0) + int(count or 0)
            if _result_runtime_split(result) == "rl":
                rl_flush_fired_counts[app] = (
                    rl_flush_fired_counts.get(app, 0) + int(count or 0)
                )
    missing = [
        f"rl_{app}_flush_fired"
        for app in ("thunderbird", "vlc")
        if rl_flush_fired_counts.get(app, 0) < 1
    ]
    weak_gate_4 = {
        "required": bool(require_rl_flush_fired),
        "passed": (not require_rl_flush_fired) or not missing,
        "missing": missing if require_rl_flush_fired else [],
        "rl_vlc_flush_fired": int(rl_flush_fired_counts.get("vlc", 0)),
        "rl_thunderbird_flush_fired": int(rl_flush_fired_counts.get("thunderbird", 0)),
    }
    return {
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "flush_fired_counts": dict(sorted(flush_fired_counts.items())),
        "rl_flush_fired_counts": dict(sorted(rl_flush_fired_counts.items())),
        **_flush_fired_aliases(flush_fired_counts),
        "weak_gate_4": weak_gate_4,
    }


def _summary_failed(summary: dict[str, Any]) -> bool:
    return bool(
        summary.get("failed", 0)
        or not (summary.get("weak_gate_4") or {}).get("passed", True)
    )


def _load_fixtures(path: Path, wanted: str | None) -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    wanted_parts = [part.strip() for part in wanted.split(",")] if wanted else []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            fixture = json.loads(line)
            if wanted_parts and not any(
                part == fixture.get("fixture_id")
                or part == fixture.get("task_id")
                or part in fixture.get("fixture_id", "")
                or part in fixture.get("task_id", "")
                for part in wanted_parts
            ):
                continue
            fixtures.append(fixture)
    return fixtures


def _find_task(split: str, task_id: str) -> dict[str, Any]:
    from lite.gym.envs.lite.scalecua.src.utils import dataset

    for _, row in dataset.iter_jsonl(dataset.catalog_path(split)):
        if row["task_id"] == task_id:
            reason = row["metadata"]["others"].get("exclude_reason")
            if reason:
                raise RuntimeError(f"{task_id} is excluded: {reason}")
            return row
    raise RuntimeError(f"{task_id!r} not found in split {split!r}")


def _make_env_kwargs(*, reset_timeout: float, debug: bool = False) -> dict[str, Any]:
    # expose_oracle=True: oracle-only metadata keys are stripped from the
    # exposed metadata by default (see LiteOsworldEnv._task_metadata, inherited
    # by lite.scalecua). This validator replays oracle from the fixture files +
    # env._task (both unaffected), but re-expose them for parity / any
    # metadata.others oracle reads.
    kwargs: dict[str, Any] = {"reset_timeout": reset_timeout, "expose_oracle": True}
    if debug:
        kwargs["debug"] = True
    return kwargs


async def _run_fixture_postconfig_once(
    env: Any,
    *,
    timeout: float,
) -> None:
    from lite.gym.envs.lite.scalecua.src.osworld.setup import dispatch_strict

    evaluator = env._task.metadata.get("evaluator", {})
    if evaluator.get("_postconfig_done"):
        return
    cache_dir = tempfile.mkdtemp(prefix="scalecua_oracle_postconfig_")
    for index, step in enumerate(evaluator.get("postconfig", []) or []):
        await asyncio.wait_for(
            dispatch_strict(
                env._computer,
                step,
                phase="postconfig",
                index=index,
                cache_dir=cache_dir,
            ),
            timeout=timeout,
        )
    evaluator["_postconfig_done"] = True


async def _quiesce_apps_for_oracle(env: Any) -> None:
    await env._computer.interface.run_command(
        "killall -9 -q chrome google-chrome chromium chromium-browser "
        "soffice.bin oosplash gimp gimp-2.10 vlc thunderbird code "
        "code-insiders 2>/dev/null; true"
    )
    await asyncio.sleep(2)
    await env._computer.interface.run_command(
        "for i in $(seq 1 20); do "
        "  { pgrep chrome || pgrep chromium || pgrep thunderbird || "
        "    pgrep soffice || pgrep gimp || pgrep vlc || pgrep code; } "
        "    > /dev/null 2>&1 || break; "
        "  sleep 0.5; "
        "done; true"
    )


async def _run_noop_precheck(
    fixture: dict[str, Any],
    *,
    artifacts: Path,
    reset_timeout: float,
) -> dict[str, Any]:
    """Score a fresh reset with NO oracle action; must equal ``expected_pre_reward``.

    Asks ``evaluate_final_fn`` directly, exactly as ``_run_oracle_replay`` does.
    Both halves ask ONE question — *what does this task's checker score on the
    container as it stands?* — and that function is its one owner, returning a
    float on every path, so "no verdict" is unrepresentable here.

    NOT via ``env.step``: ``LiteEnvStepResult.reward`` answers *what did the
    EPISODE score*, and is ``None`` whenever the step was not terminal
    (``SandboxBaseEnv._finalize_step_result`` evaluates only once
    ``terminated or truncated``) — deliberately, since a non-verdict must never
    be coalesced into ``reward=0.0``. The two diverge on this validator's own
    configuration: finish tools are gated by ``env_kwargs.extra_tools``, opt-in
    and defaulting to ``[]``, so a ``response`` call is swallowed as an inactive
    extra tool and no verdict is ever produced.
    """
    import lite.gym as gym
    from lite.gym.envs.lite.scalecua.src.osworld.verify import evaluate_final_fn

    split = fixture["split"]
    task_id = fixture["task_id"]
    row = _find_task(split, task_id)
    env = gym.make(
        f"lite.scalecua@{task_id}",
        **_make_env_kwargs(reset_timeout=reset_timeout),
    )
    try:
        reset = await env.reset()
        if fixture.get("oracle_after_postconfig"):
            await _run_fixture_postconfig_once(
                env,
                timeout=fixture.get("postconfig_timeout", reset_timeout),
            )
        _write_png(artifacts / "00_noop_reset.png", reset.image)
        # BEFORE the eval, not after: this is the state the checker is about to
        # read, which is what a trivial-pass triage needs to see.
        await _capture_png(env, artifacts / "01_noop_final.png")
        reward = await evaluate_final_fn(
            env._task,
            env._computer,
            actions=_response_action(fixture.get("noop_response", "Done")),
        )
        return {
            "task_id": task_id,
            "split": split,
            "instruction": row["instruction"],
            "reward": reward,
            "passed": _reward_matches(
                reward,
                fixture.get("expected_pre_reward", 0.0),
            ),
        }
    finally:
        await env.close()


async def _dispatch_oracle_actions(
    env: Any,
    fixture: dict[str, Any],
    *,
    timeout: float,
) -> list[dict[str, Any]]:
    from lite.gym.envs.lite.scalecua.src.osworld.setup import dispatch_strict

    trace: list[dict[str, Any]] = []
    actions = fixture["oracle_actions"]
    await _quiesce_apps_for_oracle(env)
    for index, action in enumerate(actions):
        result = await asyncio.wait_for(
            dispatch_strict(
                env._computer,
                action,
                phase="oracle",
                index=index,
            ),
            timeout=timeout,
        )
        if action.get("type") in {"execute", "command"} and isinstance(result, dict):
            rc = result.get("returncode", 0)
            if rc not in (0, None):
                raise RuntimeError(
                    f"oracle action {index} failed rc={rc}: "
                    f"{result.get('error', '')[:500]}"
                )
        trace.append(
            {
                "kind": "oracle_action",
                "index": index,
                "action": action,
                "result": result,
            }
        )
    return trace


async def _run_oracle_replay(
    fixture: dict[str, Any],
    *,
    artifacts: Path,
    reset_timeout: float,
    oracle_timeout: float,
) -> dict[str, Any]:
    import lite.gym as gym
    from lite.gym.envs.lite.scalecua.src.osworld.verify import evaluate_final_fn

    split = fixture["split"]
    task_id = fixture["task_id"]
    row = _find_task(split, task_id)
    env = gym.make(
        f"lite.scalecua@{task_id}",
        **_make_env_kwargs(reset_timeout=reset_timeout, debug=True),
    )
    try:
        reset = await env.reset()
        _write_png(artifacts / "10_oracle_reset.png", reset.image)
        if fixture.get("oracle_after_postconfig"):
            await _run_fixture_postconfig_once(
                env,
                timeout=oracle_timeout,
            )
        trace = await _dispatch_oracle_actions(
            env,
            fixture,
            timeout=oracle_timeout,
        )
        await _capture_png(env, artifacts / "11_oracle_after_actions.png")
        raw_eval = await evaluate_final_fn(
            env._task,
            env._computer,
            actions=_response_action(fixture.get("response_text", "Done")),
            debug=True,
        )
        if isinstance(raw_eval, tuple):
            reward, debug = raw_eval
        else:
            reward, debug = raw_eval, {}
        flush_fired_counts = _flush_counts_from_debug(debug)
        expected = fixture.get("expected_reward", 1.0)
        return {
            "task_id": task_id,
            "split": split,
            "instruction": row["instruction"],
            "reward": reward,
            "expected_reward": expected,
            "passed": _reward_matches(reward, expected),
            "eval_debug": debug,
            "flush_fired_counts": flush_fired_counts,
            **_flush_fired_aliases(flush_fired_counts),
            "trace": trace,
        }
    finally:
        await env.close()


async def _validate_one(
    fixture: dict[str, Any],
    *,
    artifacts_root: Path,
    reset_timeout: float,
    oracle_timeout: float,
) -> dict[str, Any]:
    fixture_id = fixture["fixture_id"]
    artifacts = artifacts_root / fixture_id
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "fixture.json").write_text(
        json.dumps(fixture, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    result: dict[str, Any] = {
        "fixture_id": fixture_id,
        "task_id": fixture["task_id"],
        "split": fixture.get("split"),
    }
    try:
        precheck = await _run_noop_precheck(
            fixture,
            artifacts=artifacts,
            reset_timeout=reset_timeout,
        )
        result["precheck"] = precheck
        if not precheck["passed"]:
            result["passed"] = False
            result["failure_type"] = "trivial_pass"
            expected_pre_reward = fixture.get("expected_pre_reward", 0.0)
            result["message"] = (
                f"precheck reward {precheck['reward']} != {expected_pre_reward}"
            )
            return result
        replay = await _run_oracle_replay(
            fixture,
            artifacts=artifacts,
            reset_timeout=reset_timeout,
            oracle_timeout=oracle_timeout,
        )
        result["replay"] = replay
        flush_fired_counts = replay.get("flush_fired_counts") or {}
        result["flush_fired_counts"] = flush_fired_counts
        result.update(_flush_fired_aliases(flush_fired_counts))
        result["passed"] = bool(replay["passed"])
        if not result["passed"]:
            result["failure_type"] = fixture.get("expected_failure_type", "needs_triage")
            result["message"] = (
                f"oracle reward {replay['reward']} != {replay['expected_reward']}"
            )
        else:
            result["message"] = "ok"
        return result
    except Exception as exc:
        result["passed"] = False
        result["failure_type"] = "exception"
        result["message"] = str(exc)
        return result
    finally:
        (artifacts / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


async def _amain(args: argparse.Namespace) -> int:
    if args.concurrency < 1 or args.retries < 1:
        raise ValueError("--concurrency and --retries must be >= 1")
    os.environ.pop("CUA_LITE_ENV_SERVER_URL", None)
    os.environ.pop("CUA_LITE_ENV_SERVER_TOKEN", None)
    # Point the env at CUA_LITE_VALIDATE_IMAGE before registration / gym.make
    # (via a temp LITE_SCALECUA_CONFIG), before any lazy env import below.
    _configure_image_override()
    session_id = args.session_id or _default_session_id(args.artifacts)
    os.environ["SESSION_ID"] = session_id

    swept = _sweep_own_containers(session_id)
    if swept:
        logger.info("startup sweep: removed %d oracle container(s)", swept)
    _install_signal_cleanup(session_id)

    fixtures = _load_fixtures(args.fixtures, args.filter)
    before = len(fixtures)
    fixtures = [fixture for fixture in fixtures if fixture.get("oracle_actions")]
    if before != len(fixtures):
        logger.info("skipped %d fixture(s) without oracle actions", before - len(fixtures))
    if args.limit is not None:
        fixtures = fixtures[: args.limit]
    if not fixtures:
        raise SystemExit("no executable fixtures selected")

    # Load previously-passed results for resume support. Keyed by fixture_id (the
    # stable per-fixture identity used for artifacts/logging); re-run any fixture
    # that did not pass so a crashed run's transient tail resumes cleanly.
    resumed: dict[str, dict[str, Any]] = {}
    if args.resume_from:
        try:
            with args.resume_from.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        if rec.get("passed"):
                            resumed[rec["fixture_id"]] = rec
            logger.info("resume: loaded %d already-passed fixture(s) from %s",
                        len(resumed), args.resume_from)
        except FileNotFoundError:
            logger.info("resume: file %s not found — starting fresh", args.resume_from)
    if resumed:
        before = len(fixtures)
        fixtures = [f for f in fixtures if f["fixture_id"] not in resumed]
        logger.info("resume: skipping %d already-passed fixture(s) (%d remaining)",
                    before - len(fixtures), len(fixtures))

    logger.info("validating %d fixture(s), concurrency=%d", len(fixtures), args.concurrency)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    report_f = args.report.open("w", encoding="utf-8") if args.report else None
    results: list[dict[str, Any]] = list(resumed.values())  # seed with resumed passes

    # Re-emit already-resumed passes so the fresh report is self-contained.
    if report_f:
        for rec in resumed.values():
            report_f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        report_f.flush()

    async def _guarded(fixture: dict[str, Any]) -> None:
        async with sem:
            # Retry the full precheck+replay on a non-pass. A default of 1
            # means a single attempt (behavior unchanged); >1 lets transient
            # noise self-heal, mirroring the lite.osworld validator.
            for attempt in range(1, args.retries + 1):
                result = await _validate_one(
                    fixture,
                    artifacts_root=args.artifacts,
                    reset_timeout=args.reset_timeout,
                    oracle_timeout=args.oracle_timeout,
                )
                if result.get("passed") or attempt == args.retries:
                    if args.retries > 1:
                        result["attempts"] = attempt
                    break
                logger.info("[%s] attempt %d/%d FAIL — %s; retrying",
                            fixture["fixture_id"], attempt, args.retries,
                            result.get("message"))
            results.append(result)
            if report_f:
                report_f.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                report_f.flush()
            status = "PASS" if result.get("passed") else "FAIL"
            logger.info("[%s] %s %s", fixture["fixture_id"], status, result.get("message"))

    try:
        await asyncio.gather(*[_guarded(fixture) for fixture in fixtures])
    finally:
        if report_f:
            report_f.close()
        # UNCONDITIONAL, unlike the startup line: the sweep is the last thing
        # that can still remove a container, so "removed 0" and "could not ask
        # docker" must not both print nothing. The line is always emitted, so its
        # ABSENCE means this finally never ran.
        removed = _sweep_own_containers(session_id)
        logger.info("final sweep: removed %s oracle container(s)",
                    "UNKNOWN (docker unreachable)" if removed is None else removed)

    summary = _build_summary(
        results,
        require_rl_flush_fired=bool(args.require_rl_flush_fired),
    )
    (args.artifacts / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "summary: %d/%d passed, %d failed",
        summary["passed"],
        len(results),
        summary["failed"],
    )
    return 1 if _summary_failed(summary) else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("lite/gym/envs/lite/scalecua/data/oracle/rl.jsonl"),
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(".exps/validate/lite.scalecua/oracle/smoke"),
    )
    parser.add_argument("--filter", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=_positive_int, default=4)
    parser.add_argument(
        "--retries", type=_positive_int, default=1,
        help="Max attempts per fixture. A fixture is FAIL only if all attempts fail. "
             "Lets transient container/app contention (VLC/Chrome races, boot flakes) "
             "self-heal. (default: 1)",
    )
    parser.add_argument("--reset-timeout", type=float, default=600.0)
    parser.add_argument("--oracle-timeout", type=float, default=180.0)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--resume-from", type=Path, default=None,
        help="Path to a previous --report JSONL. Fixtures that already passed "
             "(passed=true) are skipped; their results are merged into the new "
             "report. Use this to restart a validate after a crash or kill.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help=(
            "Direct-mode container session id. Defaults to a value derived from "
            "--artifacts so concurrent oracle sweeps do not reap each other."
        ),
    )
    parser.add_argument(
        "--require-rl-flush-fired",
        action="store_true",
        help=(
            "Fail unless the report summary includes at least one Thunderbird "
            "and one VLC config-flush decider hit. Intended for the Batch-5 rl "
            "oracle run."
        ),
    )
    args = parser.parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
