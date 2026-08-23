# OSWorld rollout audit plan

> 🛑 **Spawning a subagent? Brief them on this plan FIRST.**
>
> When the main agent dispatches a subagent for any audit-loop step (`scan` triage,
> `subagent.diagnose`, diagnostic tiers in `§subagent.diagnose`, fix verification,
> param-reduction passes), the dispatch prompt MUST:
>
> 1. **Reference this file by absolute path:** `devs/envs/lite.osworld/validate/rollout/plan.md`.
>    Tell the subagent to read it before starting and to re-read the sections marked
>    📌 SUBAGENT-MUST-READ (currently: `§group_by_root_cause`, `§subagent.diagnose`,
>    `§edit_generator_py`, `§delete_summaries_for_affected`, `§classify_blast_radius`,
>    `§Trigger taxonomy`).
> 2. **Name the subagent's tier explicitly** (tier-1 / tier-2 / tier-3 in the
>    `§subagent.diagnose` verification table). All persistent edits are owned by
>    the main/coordinator agent; the tier tells the subagent what evidence to
>    collect before returning a proposal.
> 3. **Distinguish oracle replay from agent replay** explicitly when a verify step
>    is requested. The oracle-vs-agent table in `§subagent.diagnose` is the
>    canonical reference — citing it prevents the false-negative trap where a
>    subagent oracle-replays an eval-strictness-vs-UI-output bug and reports "no
>    bug" when the agent-side failure is real.
> 4. **Bound scope** — give the subagent the exact list of candidate `task_id`s /
>    file paths / domain modules to inspect or probe, AND list what it MUST NOT touch
>    (typically `_BUG_TEMPLATE_IDS` / `_HARD_TEMPLATE_IDS` / `_DROPPED_TEMPLATE_IDS`
>    state lists, generated JSONL, sha256 locks, rollout roots, and source files —
>    main agent owns those; `.venv/` — never; sibling subagents' scratch files —
>    race risk).
> 5. **State the scratch-dir convention** (`/tmp/audit_*` only; never repo root).
>
> Subagents that skip this briefing tend to mis-classify, re-investigate
> already-known-dead-end clusters, or mutate persistent files outside coordinator
> control. The audit loop's throughput depends on each subagent inheriting the
> plan's vocabulary (FALSE_NEG / TURN_CEILING_HIT / family-cluster invariant /
> trigger taxonomy) — not re-inventing it per dispatch.

## Quick reference

| Item | Value |
|---|---|
| Default model | `gpt-5.5` (API model; no local server needed) |
| Rollout script | `scripts/rollout.py` — auto-routes API models vs local models on `--model-id`; for local-served models add `--sglang-server-url` |
| Env-server | Full sweep / re-roll batches run through a fresh dedicated `scripts/serve_env.py` env-server; direct env mode is only for smoke checks and replay probes |
| Rollout config | `scripts/configs/gpt/default/lite.osworld.yaml` |
| Rollout logs | `.logs/rollout/<model_id>/lite.osworld/<timestamp>/<split>/<task_id>/sample_00/` |
| Source pool | 369 eval + 2429 train (1722 synth + 707 perturb) in `lite/gym/envs/lite/osworld/data/{eval,train.synth,train.perturb}.jsonl` (the actual rolled set is the subset selected by `--splits`) |
| Concurrency | 16 (matches the example below; raise carefully against provider quota and host capacity) |
| Findings log | `devs/envs/lite.osworld/validate/rollout/logs.md` — append-only, one line per finding: `<task_id>: <symptom> [trigger=X \| severity=critical/regression/cosmetic]`, close with `→ fixed in <hash>` |
| Debug script | `devs/envs/lite.osworld/validate/rollout/replay_trajectory.py` |
| Build env image | `uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/install.sh rebuild` (canonical — the Dockerfile docstring `docker build … docker/` uses the wrong context; only `install.sh` knows the parent-dir scope) |
| Regen jsonl | `uv run python -m lite.gym.envs.lite.osworld.src.gen.train --track {synth\|perturb\|all} [--domain X]` |
| Oracle replay | `uv run python devs/envs/lite.osworld/validate/oracle/validate.py --fixtures <jsonl> --filter <task_id>` — proves gold/eval/postconfig are mutually consistent. **Does NOT prove the agent can reproduce the gold via the live UI** |
| Agent replay | `uv run python devs/envs/lite.osworld/validate/rollout/replay_trajectory.py <task_id> [--rollout $LOG_ROOT/<split>/<tid>]` — replays the agent's recorded actions against the same env. Only this settles "does my source-side fix actually let the agent reach reward=1.0" |
| Scratch dir | All diagnostic JSONL / md outputs land under `/tmp/audit_*` — **never** at repo root. Main agent cleans `/tmp/audit_*` at end of cycle |
| Persistent edit owner | Main/coordinator agent only: source edits, generated JSONL, sha256 locks, rollout-root mutation, state-list/log updates, and commits |

> **NEVER directly edit `.jsonl` files.** They are generated idempotently by generator scripts.
> Any hand-edit will be silently overwritten on the next regen and breaks the sha256 lock.
> Always fix the `.py` generator script, then regen.
>
> **Keep `.py` ↔ `.md` in sync.** When you edit a perturb generator under
> `lite/gym/envs/lite/osworld/src/gen/train/perturb/<domain>.py`, also update its
> co-evolved spec at `devs/envs/lite.osworld/perturb/<domain>.md` (per-task tables,
> archetype rows, paraphrase pools, infeasible lists, expected row counts). The two files
> are the source of truth together — drift between them is the most common cause of stale
> assumptions in later audit cycles.

---

## Audit loop

Every function called below has either an inline explanation here in the comment, or a dedicated section in [Function reference](#function-reference) explaining what it does and how to invoke it. If you see a name you don't recognize, search for `### \`<name>\`` below.

```python
# Two independent rollouts. Sweep 1 finds and fixes bugs; Sweep 2 is a fresh
# re-run on `main` that confirms no new bugs surface. See `Exit criteria`.
#
# CRITICAL — log-dir conventions:
#   • EACH sweep gets its OWN log-root (a brand-new timestamped directory).
#     SWEEP_1 and SWEEP_2 must NEVER share the same `$LOG_ROOT` — that would
#     intermix old (pre-fix) and new (post-fix) summaries and break every
#     scan signal (false_neg / homo_zero / variant homogeneity all bucket by
#     `task_id` and would conflate the two runs).
#   • WITHIN one sweep, the log-root is FIXED for the entire audit loop.
#     `launch_rollout` returns it once at sweep start; every subsequent
#     `wait_for_new_completions_or_stall(log_root)`, `scan(log_root)`,
#     `delete_summaries_for_affected(log_root, ...)`, `restart_rollout(log_root)`
#     re-uses the same path. The fix-and-re-roll cycle is in-place — failed
#     tasks have their `summary.json` deleted (see `delete_summaries_for_affected`)
#     and the rollout is restarted against the same `$LOG_ROOT`, so the new
#     attempt overwrites the old summary at the same `task_id` dir.
#   • Auxiliary "validate-only" rollouts (small filtered re-runs to verify a
#     specific fix immediately, without waiting for the full sweep) DO use a
#     separate fresh log-root — those are scratch validations, not part of
#     the sweep itself, and they don't replace the sweep's persistent log.
for sweep in [SWEEP_1, SWEEP_2]:

    # Start the rollout asynchronously. Returns the log-root path it will
    # write summaries into. The path is timestamped and unique per sweep —
    # `SWEEP_2`'s call lands in a new directory, never `SWEEP_1`'s.
    # See `launch_rollout` below.
    log_root = launch_rollout(splits=["eval", "train"])

    # `remaining_tasks` = count of task_id dirs under `$LOG_ROOT/<split>/`
    # that don't yet have a `sample_00/summary.json`. Inner loop runs once
    # per ~50 completions until everything is rolled.
    while remaining_tasks(log_root) > 0:

        # Block until `threshold` new summaries appear (or the rollout has
        # stalled for `stall_timeout`). Threshold caps at remaining_tasks
        # so we don't deadlock when only 17 tasks are left. Stall handling
        # is in `wait_for_new_completions_or_stall` below.
        wait_for_new_completions_or_stall(
            threshold=min(50, remaining_tasks(log_root)),
            stall_timeout=15 * 60,
        )

        # Read all `summary.json` + per-turn `02_response.txt` under
        # log_root and emit candidates. Categories (defined in `Trigger
        # taxonomy`): FALSE_NEG, FALSE_POS, TRIVIAL_PASS, TURN_CEILING_HIT,
        # VARIANT_HOMOGENEITY_ZERO, VARIANT_HOMOGENEITY_ONE,
        # INFEASIBLE_CLAIM_TRAIN, INFEASIBLE_CLAIM_EVAL_UNMARKED,
        # INFEASIBLE_CLAIM_EVAL_CORRECT (informational, excluded from
        # candidates), ERROR_NO_SUMMARY (low-priority: usually transient
        # LLM-API 429 noise that the outer --max-attempts loop reruns
        # automatically — not a dataset bug). All others — including
        # EVERY infeasibility claim on train, every false negative, every
        # false positive, and any base with all variants scoring 0 — need
        # fixing before the data ships. See `scan` below for the full
        # extraction script.
        candidates = scan(log_root)
        if not candidates:
            continue

        # Cluster candidates that likely share a root cause so a single
        # subagent can investigate them together. See `group_by_root_cause`.
        # Subagents are read-only — they propose fixes but NEVER edit files.
        proposals = []
        for group in group_by_root_cause(candidates):

            # `spawn_subagent`: fork a diagnostic agent (one per cluster).
            # Brief per the top-of-file 🛑 callout (absolute-path cite, tier,
            # oracle-vs-agent replay, bounded scope). Full procedure +
            # family-cluster invariant in `subagent.diagnose` and
            # `group_by_root_cause` below.
            subagent = spawn_subagent(group)
            proposals.append(subagent.diagnose())

        # All proposals are admitted as fixes — verification is informational,
        # not gating. Any proposed fix triggers the stop-restart cycle so the
        # next sweep batch sees the updated generator output. If a fix turns
        # out to be wrong, the post-restart `family_returns_since` check
        # surfaces it as still-failing and the next audit-loop iteration
        # re-diagnoses. `manually_verify_fix_path` (below) remains available
        # as an optional pre-flight sanity check for ambiguous proposals,
        # but its result does NOT gate fix admission. Rationale: pre-fix
        # verification is expensive (interactive container + manual probe);
        # better to land plausible fixes fast and let the next cycle's
        # rollout do the real verification at scale.
        fixes = list(proposals)
        if not fixes:
            continue

        # Apply the fixes. Never hand-edit JSONL files — the generators are
        # byte-deterministic Python, and the next regen will silently
        # overwrite any manual JSONL edit. See `edit_generator_py` below.
        for fix in fixes:

            # Tally the affected templates BEFORE editing so we know whether
            # the change is 🟢 local / 🟡 family / 🔴 global. 🔴 fixes
            # require human sign-off because they touch shared harness code
            # (e.g. `eval/metrics.py`, `utils/dispatch.py`, Dockerfile) and
            # can silently regress unrelated tasks. See `classify_blast_radius`.
            radius = classify_blast_radius(fix)
            if radius == "global":
                # `require_user_signoff` blocks until the user explicitly
                # approves the diff. Inline-defined: just ask in chat with
                # a summary of fix-targets and unaffected canaries.
                require_user_signoff(fix)

            # Edit the generator .py file at the location identified in the
            # proposal. See `edit_generator_py` below.
            edit_generator_py(fix)

        # Re-emit JSONL splits ONCE per cycle (not per fix) — generators are
        # deterministic in their source, so per-fix regen wastes runs without
        # changing the output. See `regen_affected_jsonl_files` below.
        regen_affected_jsonl_files()

        # OPTIONAL informational replay using the regenerated JSONL row.
        # Pass = strong signal the fix is robust. Fail = INCONCLUSIVE (the
        # recorded trajectory may simply not exercise the fix path); does
        # NOT reject the fix. See `replay_after_regen`.
        for fix in fixes:
            replay_after_regen(fix)

        # TERMINAL GATE — batched stop → delete → restart → re-roll → check.
        # All fixes go through ONE stop/restart cycle. Per-fix overhead is
        # ~5 min cleanup + ~2 min container init + ~5-30 min re-roll wait =
        # ~12-37 min/fix; for 5 fixes that's 1-3 hours of overhead the batch
        # avoids.
        #
        # RESTART-THRESHOLD RULE (verify before invoking this gate):
        # The terminal stop+restart forfeits ~16 in-flight containers + ~30
        # min warm wall time. Trigger ONLY when NOT restarting would
        # invalidate ALL subsequent in-flight rolls (i.e. they'd be scored
        # under stale semantics that materially flip pass/fail). Touching a
        # harness file is NOT the threshold — "would the in-flight rolls be
        # materially wrong if I let them finish?" is. The canonical decision
        # matrix lives in §classify_blast_radius ("🔴 ≠ MUST RESTART"
        # table); per-fix deletion eligibility (which uses the same rule) is
        # in §delete_summaries_for_affected. When in doubt: prefer NOT
        # restarting — one wasted restart costs more than one stale tail.

        # Drop a marker file at $LOG_ROOT/.audit_sentinel so we can later
        # `find -newer` exactly the summaries written after this point (i.e.
        # only this cycle's re-rolls, not the entire sweep). See `touch_sentinel`.
        sentinel = touch_sentinel(log_root)

        # Stop the running rollout cleanly: pkill this rollout's scripts,
        # ask the env-server to close only this SESSION_ID's lite.osworld
        # instances, then sleep 30 to drain in-flight summary writes. See
        # `stop_rollout`.
        stop_rollout()

        # Delete `sample_00/summary.json` for every task whose SOURCE was
        # edited in this cycle AND whose regen produces different JSONL
        # bytes (or whose live env now behaves differently). The runner's
        # idempotency layer (`get_pending`) skips tasks with a summary,
        # so deletion is what re-queues them. Re-rolling a task whose
        # source did not change is pure waste (same JSONL + same agent →
        # same trajectory). Full qualifies / does-not-qualify list and the
        # eval↔perturb cross-delete rule live in
        # §delete_summaries_for_affected.
        for fix in fixes:
            delete_summaries_for_affected(fix)

        # Re-launch the rollout with the SAME --log-root. Idempotent: only
        # tasks missing a summary (i.e. the ones we just deleted, plus any
        # that were never rolled) get rolled. See `launch_rollout`
        # (`restart_rollout` is the same command with the existing log-root).
        restart_rollout(log_root)

        # Per-fix verification gate. `family_returns_since(fix, sentinel)`
        # filters by mtime > sentinel — only this cycle's re-rolls count.
        # Pass = every affected family has ≥1 task at episode_return == 1.0.
        # If a fix's family still scores 0 after restart: the fix is wrong
        # or incomplete → `revert_and_repropose(fix)` rolls back ONLY that
        # fix (leaves others applied — they're independently verified) and
        # the next inner-loop iteration re-diagnoses. Never silently regen
        # a partially-failing fix. See `family_returns_since`.
        for fix in fixes:
            if not all(r == 1.0 for r in family_returns_since(fix, sentinel)):
                revert_and_repropose(fix)

    # END-OF-SWEEP GLOBAL SCAN — the per-batch ~50-task scans only saw
    # small windows; now scan EVERY summary under log_root to surface bugs
    # masked by small batches (late-variant family failures, fix-introduced
    # regressions outside the cycle's re-roll set). Rerun the inner-loop
    # fix body until `scan` returns empty.
    while scan(log_root):
        # Same group → diagnose → apply → regen → sentinel → stop → delete
        # → restart → verify cycle as the inner loop above.
        ...

    # `no_unresolved_eval_or_setup_bugs`: per-sweep exit gate. After the
    # end-of-sweep scan settles, every FALSE_NEG / FALSE_POS / TRIVIAL_PASS
    # candidate must be either (a) confirmed agent error and logged in
    # logs.md, or (b) fixed and verified through `family_returns_since`.
    # See `Exit criteria` below for the full freeze conditions across both
    # sweeps.
    assert no_unresolved_eval_or_setup_bugs(log_root)
```

---

## Function reference

### `launch_rollout`: start a rollout that writes summaries into `LOG_ROOT`

Default API-model full sweep. Each value passed to `--splits` becomes a top-level subdirectory under `$LOG_ROOT/`, so log paths follow `$LOG_ROOT/<split>/<task_id>/sample_00/`.

**Env-server and log-dir rules** (do not deviate):
- Full sweep / restart batches MUST use a fresh dedicated env-server. Start the
  server before `scripts/rollout.py`, pass `CUA_LITE_ENV_SERVER_URL` and
  `CUA_LITE_ENV_SERVER_TOKEN` to the rollout command, and stop that server after
  the batch drains. Direct env mode is only for one-off smoke checks and
  `replay_trajectory.py` probes.
- One **fresh, timestamped** `LOG_ROOT` per sweep — `SWEEP_1` and `SWEEP_2` MUST have different `$LOG_ROOT` paths. If you accidentally restart `SWEEP_2` against `SWEEP_1`'s log-dir, every `scan` signal silently mixes pre-fix and post-fix attempts and the audit becomes unreadable.
- Inside one sweep, `LOG_ROOT` is **persistent** for the duration of the audit loop. The `restart_rollout(log_root)` call uses the same `$LOG_ROOT` so the `scripts/rollout.py` command through `lite.infer.rollout.run_rollout` is idempotent — it skips tasks that already have a `summary.json` and only re-rolls the ones whose summary was deleted by `delete_summaries_for_affected` after a fix landed. This is how fix-then-re-roll works in place without losing context.
- A separate one-off "fix-validate" rollout (small `--filter` against a fresh `LOG_ROOT`, used for fast feedback on a specific patch) is OK — it's scratch verification, not part of the sweep, and shouldn't reuse or overwrite the sweep's log-dir.

```bash
# Pick a NEW timestamp at the start of each sweep — never alias to a previous one.
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT=".logs/rollout/gpt-5.5/lite.osworld/$RUN_ID"
ENV_SERVER_PORT=30320
HOST_IP=$(hostname -I | awk '{print $1}')
ENV_SERVER_TOKEN="lite-osworld-$RUN_ID"

env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
  nohup uv run python scripts/serve_env.py \
  --host 0.0.0.0 \
  --port "$ENV_SERVER_PORT" \
  --env-ids lite.osworld \
  --token "$ENV_SERVER_TOKEN" \
  --idle-ttl-sec 900 \
  > /tmp/rollout-env-server.log 2>&1 &

CUA_LITE_ENV_SERVER_URL="http://${HOST_IP}:$ENV_SERVER_PORT" \
CUA_LITE_ENV_SERVER_TOKEN="$ENV_SERVER_TOKEN" \
nohup uv run python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id lite.osworld \
  --config-path scripts/configs/gpt/default/lite.osworld.yaml \
  --splits eval train \
  --concurrency 16 \
  --save-data true \
  --log-root "$LOG_ROOT" \
  > /tmp/rollout.log 2>&1 &
```

Leave `--max-live-envs` unset for normal sweeps so the env-server derives its
admission cap from host capacity. Use `--max-live-envs <N>` only as an explicit
advanced override for a constrained-cap repro, and record the reason with the
artifact root.

**Rolling only the perturb subset** — `train.synth` and `train.perturb` are registered as **separate splits** (see `lite/gym/envs/lite/osworld/main.py:282-299`), so the cleanest way is to pass them directly. No filter needed.

```bash
HOST_IP=$(hostname -I | awk '{print $1}')

CUA_LITE_ENV_SERVER_URL="http://${HOST_IP}:$ENV_SERVER_PORT" \
CUA_LITE_ENV_SERVER_TOKEN="$ENV_SERVER_TOKEN" \
nohup uv run python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id lite.osworld \
  --config-path scripts/configs/gpt/default/lite.osworld.yaml \
  --splits train.perturb \
  --concurrency 16 \
  --save-data true \
  --log-root "$LOG_ROOT" \
  > /tmp/rollout.log 2>&1 &
```

**`--filter` for finer slicing** — `--filter` takes a Python lambda on `LiteBaseMetadata` (parsed by `lite.core.utils.filters.parse_filter` and wired through `lite.infer.rollout`), **not** a task_id substring. Prefer `m.dims` for routing dimensions; CUA-only filters may use `m.platform` / `m.task_type`. The catch-all `m.others` carries domain-specific fields like `domain`, `difficulty`, `exclude_reason`. Examples:

```bash
# Only chrome-domain tasks within the chosen splits
--filter "lambda m: m.others.get('domain') == 'chrome'"
# Skip eval tasks tagged as infeasible / excluded
--filter "lambda m: not m.others.get('exclude_reason')"
```

To roll a single task or one base's variants, prefer regenerating against a smaller scope or post-filter by directory after the run — `--filter` cannot match on task_id directly.

`restart_rollout(log_root)` is the same command with the existing `$LOG_ROOT` and the current dedicated env-server URL/token — idempotent: only tasks missing `sample_00/summary.json` get rolled.

**Local-served models** (e.g. Qwen): the same `scripts/rollout.py` auto-routes to the local backend from `--model-id`; add `--sglang-server-url http://localhost:<port>` pointing at the local SGLang/vLLM server. All other flags (`--env-id`, `--config-path`, `--splits`, `--concurrency`, `--save-data`, `--log-root`) apply unchanged.

**Splits**: `eval` and `train` are the two top-level JSONL files. `train` is the union of two byte-locked sub-files (`train.synth.jsonl` + `train.perturb.jsonl`). Pass any of these to `--splits` directly:

| `--splits` | Top-level subdirs created under `$LOG_ROOT/` | Use case |
|---|---|---|
| `eval train` | `eval/`, `train/` | Default — full sweep |
| `eval train.perturb` | `eval/`, `train.perturb/` | Skip synth; iterate on perturb generator only |
| `train.synth train.perturb` | `train.synth/`, `train.perturb/` | Skip eval (rare — only when iterating on training data alone) |

The `scan` and `family_returns_since` helpers iterate every subdir of `$LOG_ROOT/`, so they handle any splits choice transparently.

---

### `wait_for_new_completions_or_stall`: block until N new summaries appear or progress stalls

Track previous total; poll `find $LOG_ROOT -name summary.json | wc -l` until delta ≥ `threshold` or `stall_timeout` elapses. Don't oversize the threshold — every cycle of unaudited completions is API budget burnt on tasks that will be re-rolled anyway once the cycle's fixes land. Default 50 keeps pending audits ≤ ~50 + concurrency.

**On stall (no new completion in `stall_timeout`):**

```bash
ps aux | grep "scripts/rollout\.py" | grep -v grep # rollout process alive?
curl -s -H "Authorization: Bearer ${CUA_LITE_ENV_SERVER_TOKEN}" \
  "${CUA_LITE_ENV_SERVER_URL}/instances?session_id=${SESSION_ID:?set SESSION_ID}&env_id=lite.osworld" | jq .
tail -50 /tmp/rollout.log                                        # persistent errors?
```

- Process dead / persistent errors → `stop_rollout()` then `restart_rollout(log_root)`.
- One task hanging (rollout otherwise healthy) → escalate to user.

---

### `scan`: extract FALSE_NEG / FALSE_POS / TRIVIAL_PASS / TURN_CEILING / VARIANT_HOMOGENEITY / INFEASIBLE_CLAIM / ERROR_NO_SUMMARY from `summary.json` + trajectory + `error.txt`

**`ERROR_NO_SUMMARY`** (`sample_00/error.txt` exists, no `summary.json`) is usually transient provider 429 / 503 noise — the outer `--max-attempts 10` loop re-rolls them automatically with no code change. Only escalate if a single task accumulates ≥2 errors across attempts (likely a deterministic crash: bad config, OOM, infinite-loop heredoc). Lowest scan priority.

Iterates *every* split-subdir of `$LOG_ROOT` so it works regardless of `--splits` value used at launch. Reads each task's `summary.json` for status flags, AND walks `turn_*/02_response.txt` to detect `report_infeasible` tool calls (`summary.json` records only `terminated`/`truncated`/`episode_return`, never per-call tool names, so summaries alone can't distinguish "agent gave up" from "agent claimed infeasible").

Two scan-only signals require post-pass aggregation across tasks (they cannot be emitted while iterating one summary at a time):

- **TURN_CEILING_HIT** = `truncated ∧ episode_return==0` — agent exhausted `max_steps` (15 default, 30 for `multi_apps` per `scripts/configs/gpt/default/lite.osworld.yaml`) without completing. Family-wide ceiling hits (every variant of a base maxes out turns) usually mean the task is over-budgeted, the instruction is ambiguous enough to send the agent in circles, or a transient UI dialog steals every other turn. **DO NOT drop a task just because it ran out of steps.** `max_steps` is a tunable knob — if the task is *feasible* (oracle 1.0; instruction unambiguous; UI path navigable) and only fails because the budget is too tight (e.g. multi-step Convert/Save dialogs, deep "Show settings: All" trees, per-message email export), the right fix is to raise `max_steps` for that task family in the rollout config, not to delete the perturb archetype. Only drop when there's a **structural** infeasibility (eval contract impossible against the live environment, redirect rewrites synthetic URL params, IMAP gloda async race, model-level safety refusal, eval threshold tighter than the gold can satisfy, schema absent on the target desktop, etc.).
- **VARIANT_HOMOGENEITY** = a perturb base whose ≥3 variants all score the same way (all 0 or all 1). All-0 confirms a base-level setup/eval bug (single-variant agent error is far more likely than N independent identical agent errors); all-1 with low n_turns suggests the perturbation didn't actually change task difficulty — knob is a no-op, training signal is diluted. Detected by stripping the trailing `_<8hex>` knob suffix from each perturb `task_id` to recover the base, then bucketing scores.

```bash
export LOG_ROOT=".logs/rollout/gpt-5.5/lite.osworld/<YYYYMMDD_HHMMSS>"

python3 - <<'EOF'
import json, os, pathlib, re
from lite.infer.debug.log_layout import turn_dirs

log_root = pathlib.Path(os.environ["LOG_ROOT"])
data_dir = pathlib.Path("lite/gym/envs/lite/osworld/data")

# Ground-truth: which eval tasks are intentionally infeasible (eval.func=="infeasible")
gt_infeasible = set()
for line in (data_dir / "eval.jsonl").read_text().splitlines():
    d = json.loads(line)
    fn = d["metadata"]["evaluator"].get("func")
    if fn == "infeasible" or (isinstance(fn, list) and "infeasible" in fn):
        gt_infeasible.add(d["task_id"])

def claimed_infeasible(task_dir):
    """Return (turn_name, reason) if any turn's 02_response.txt invoked
    report_infeasible (the original tool call, before main.py mutates it)."""
    for turn_dir in turn_dirs(task_dir / "sample_00"):
        rt = turn_dir / "02_response.txt"
        if not rt.exists():
            continue
        text = rt.read_text(errors="ignore")
        if '"name": "report_infeasible"' in text or '"name":"report_infeasible"' in text:
            m = re.search(r'"reason"\s*:\s*"([^"]{0,200})', text)
            return (turn_dir.name, (m.group(1) if m else "")[:120])
    return None

false_neg, false_pos, trivial_pass, turn_ceiling = [], [], [], []
infeasible_train, infeasible_eval_unmarked, infeasible_eval_correct = [], [], []
# Per-base aggregation for VARIANT_HOMOGENEITY: only meaningful for perturb
# (eval has 1 task per base). task_id format: `perturb_<base>_<8hex_knob>`.
base_scores = {}  # base_id -> list[(tid, ret, n_turns)]
total = 0
for split_dir in sorted(p for p in log_root.iterdir() if p.is_dir()):
    split = split_dir.name
    for task_dir in sorted(split_dir.iterdir()):
        s = task_dir / "sample_00" / "summary.json"
        if not s.exists():
            continue
        total += 1
        d = json.loads(s.read_text())
        tid = f"{split}/{task_dir.name}"
        n = d.get("n_turns"); n_disp = n if n is not None else "?"
        ret = d.get("episode_return")

        # Standard scan signals
        if d.get("terminated") and ret == 0.0:
            false_neg.append(f"{tid}  (n_turns={n_disp})")
        elif d.get("truncated") and ret == 1.0:
            false_pos.append(f"TRUNC_PASS {tid}  (n_turns={n_disp})")
        elif d.get("truncated") and ret == 0.0:
            # Agent ran out of turns without solving. Family-wide ceiling hits
            # = task design suspect (over-budgeted / ambiguous instruction).
            turn_ceiling.append(f"{tid}  (n_turns={n_disp})")
        elif d.get("terminated") and ret == 1.0 and isinstance(n, int) and n <= 5:
            # Cycle-43 widened from <=3 to <=5 after f_calc_22__color_top_share was
            # caught manually: vacuous task slipped through at n=5 because the agent
            # spent 4 inspection turns + 1 terminate turn confirming "nothing to do".
            # Vacuous (Trigger B) tasks need turns to inspect/verify even when no
            # mutation is required, so an n<=3 cap is too tight.
            trivial_pass.append(f"{tid}  (n_turns={n_disp})")

        # Bucket perturb tasks by base for homogeneity analysis. Strip the
        # trailing `_<8hex>` knob suffix to recover the shared base.
        name = task_dir.name
        if name.startswith("perturb_") and len(name) > 17 and name[-9] == "_":
            base = name[:-9]  # drop `_<8hex>`
            base_scores.setdefault(f"{split}/{base}", []).append((tid, ret, n))

        # Infeasibility-claim signal (orthogonal — overlaps with FALSE_NEG)
        info = claimed_infeasible(task_dir)
        if info is None:
            continue
        turn, reason = info
        line = f"{tid}/{turn}  reason={reason!r}"
        if split.startswith("train"):
            # Train tasks should NEVER be infeasible by construction — strong signal.
            infeasible_train.append(line)
        else:  # eval
            # Cross-check ground truth: eval tasks with func="infeasible" expect this.
            if task_dir.name in gt_infeasible:
                infeasible_eval_correct.append(line)
            else:
                infeasible_eval_unmarked.append(line)

# VARIANT_HOMOGENEITY: surface only "interesting" bases — ≥3 variants AND
# all scoring identically. all-0 = base-level bug; all-1 with low median
# n_turns = perturbation may be vacuous (knob isn't actually changing the
# task). Single-variant or 2-variant bases have too few samples to flag.
homo_zero, homo_one = [], []
for base, rows in sorted(base_scores.items()):
    if len(rows) < 3:
        continue
    rets = [r for _, r, _ in rows]
    if all(r == 0.0 for r in rets):
        homo_zero.append(f"{base}  (variants={len(rows)})")
    elif all(r == 1.0 for r in rets):
        n_turns_list = [n for _, _, n in rows if isinstance(n, int)]
        median_n = sorted(n_turns_list)[len(n_turns_list)//2] if n_turns_list else "?"
        homo_one.append(f"{base}  (variants={len(rows)}, median_n_turns={median_n})")

print(f"Completed: {total}")
def _dump(label, lst):
    print(f"\n{label} ({len(lst)}):")
    [print(f"  {t}") for t in lst]
_dump("FALSE_NEG", false_neg)
_dump("FALSE_POS", false_pos)
_dump("TRIVIAL_PASS", trivial_pass)
_dump("TURN_CEILING_HIT  [truncated ∧ ret==0 — agent ran out of turns; family-wide hits = design suspect]", turn_ceiling)
_dump("VARIANT_HOMOGENEITY_ZERO  [≥3 variants of one perturb base all score 0 — base-level bug]", homo_zero)
_dump("VARIANT_HOMOGENEITY_ONE  [≥3 variants of one perturb base all score 1 with low n_turns — knob may be no-op]", homo_one)
_dump("INFEASIBLE_CLAIM_TRAIN  [generator bug suspect — train should never be infeasible]", infeasible_train)
_dump("INFEASIBLE_CLAIM_EVAL_UNMARKED  [agent says infeasible, but eval.func != infeasible — potential eval bug or agent over-cautious]", infeasible_eval_unmarked)
_dump("INFEASIBLE_CLAIM_EVAL_CORRECT  [aligned with ground truth — informational only]", infeasible_eval_correct)
EOF
```

**Triage tips:**

- **FALSE_NEG with `n_turns=1`** → likely trigger H (UI dialog blocked turn_0000, see Trigger taxonomy below); check `turn_0000/prompt_images/0000_reset.png` when debug artifacts are present, or the first `sample_00/images/*.png` trajectory image, before spawning a full subagent.
- **TRIVIAL_PASS verification — `n_turns ≤ 5` is a cheap heuristic, NOT a verdict.** Before classifying as legit (hint-driven 1-action) vs vacuous (Trigger B / VACUOUS_PREDICATE), count **mutating actions** across all turns by reading each `turn_*/03_actions.json`'s `executed_actions`. Mutating = anything that changes file bytes / app state (type, hotkey other than Ctrl+S, format-dialog confirm, save-as). **0 mutating actions + ret=1.0** = vacuous-pass (the agent figured out "nothing to do" and the eval was already satisfied in the initial state) — generator bug, propose source-side fix. ≥1 mutating action + ret=1.0 at low n = likely hint-driven legit. The n_turns filter alone is leaky (caught in cycle-43: `f_calc_22__color_top_share` slipped through at n=5 because the agent needed 4 inspection turns to confirm "no rows below 5% threshold" before terminating).
- **INFEASIBLE_CLAIM_TRAIN** is the strongest setup-bug signal in the whole scan: the perturb/synth generators promise feasibility by construction, so any train-side `report_infeasible` is a generator bug worth investigating immediately. Common causes: requested asset (font, app, file) not present in the docker image, contradictory instruction, missing dependency.
- **INFEASIBLE_CLAIM_EVAL_UNMARKED** ≠ ground truth (eval.func is not `"infeasible"`). Either (a) eval needs a `func=infeasible` patch upstream, or (b) the agent is being over-cautious on a hard-but-feasible task. Manual review needed.
- **INFEASIBLE_CLAIM_EVAL_CORRECT** is just bookkeeping — the agent did the right thing on an eval task that was meant to be infeasible.
- **TURN_CEILING_HIT** is a weak signal in isolation (one agent ran out of time) but very strong as a family pattern: if 3+ variants of one base all hit the ceiling, the task is over-budgeted or has a UI trap (recovery dialog reappears every N turns). Cross-reference with `VARIANT_HOMOGENEITY_ZERO` — the same base usually appears in both.
- **VARIANT_HOMOGENEITY_ZERO** is a near-certain base-level bug: N independent identical agent failures on N variants of the same base are unlikely. Treat as if the entire family were a single FALSE_NEG. **VARIANT_HOMOGENEITY_ONE** is softer — investigate only if median n_turns ≤ 3 (suggests knob doesn't change difficulty, perturbation is vacuous).
- The three signals **INITIAL_STATE_MISMATCH**, **EVAL_FILE_PATH_MISMATCH**, and **POSTCONFIG_NO_OP** in [Trigger taxonomy](#trigger-taxonomy) below are *root-cause triggers* surfaced during `subagent.diagnose`, not standalone scan categories — they require comparing screenshots/trajectory/JSONL fields against each other and don't decompose to a single `summary.json` predicate. The scan flags candidates via FALSE_NEG/TURN_CEILING/VARIANT_HOMOGENEITY; the subagent then assigns them one of these triggers as part of the hypothesis.

---

### `group_by_root_cause`: cluster candidates so one subagent handles related tasks   📌 SUBAGENT-MUST-READ

A "**family**" in this plan = a cluster of tasks that almost certainly share a root cause: typically one base eval task plus all of its perturb variants, or all tasks generated by the same helper function. Family-wide failures are the strongest eval/setup-bug signal — N independent agent failures on the same template is far less likely than one shared bug.

Heuristic — group by:
1. Same `<base_task_id>` (perturb variants of one base) → likely same generator bug.
2. Same `<domain>` + same `n_turns` magnitude (e.g. `chrome` family-wide n_turns=1) → likely same trigger.
3. Otherwise one subagent per candidate.

A single subagent investigating 3-5 related tasks costs less wall-clock than 3-5 parallel subagents that each repeat the same diagnosis.

**Family-cluster invariant**: when ≥3 variants share the failure pattern, the diagnose may NOT close as "agent error" or "capability ceiling, no fix". Three independent identical failures = signature of a setup/instruction/eval bug. The subagent must keep peeling layers until it surfaces a source-side fix candidate; if every layer truly comes back clean, escalate to the main agent for explicit sign-off rather than silently dropping the cluster.

---

### `subagent.diagnose`: investigation → fix proposal AND interactive verification   📌 SUBAGENT-MUST-READ

The subagent has **tiered verification responsibility** depending on the class
of change. Persistent edits are always serialized by the main/coordinator agent:
source files, generated JSONL, sha256 locks, rollout roots, state lists,
`logs.md`, and commits. Subagents may inspect logs/source, run read-only shell
commands, probe/replay throwaway containers, and write scratch artifacts under
`/tmp/audit_*`; they return proposals with evidence instead of editing the repo.
Picking the right tier matters: tier-1 proposals need no replay, tier-2
proposals need agent-replay evidence before "fixed" is claimed, and tier-3
proposals need a live fix-path plus coordinator re-validation.

| Tier | Class of change | Subagent output | Verify required |
|---|---|---|---|
| 1 | Instruction string text; drop a single `Param` or single `task_id` from a `FileTask`; inline comment / docstring; cosmetic generator code reorganisation (helper rename, dead-code removal) | Exact proposal for the coordinator to persist | None — the gold/eval contract is identical |
| 2 | `eval_kind` / `eval_args` / evaluator-rules tweak; switching evaluator helper (e.g. upstream -> lite-custom); relaxing a strict comparator flag; gold builder semantic change | Exact proposal plus **agent replay** evidence from a `/tmp` patched spec or from the coordinator-applied edit | YES — agent replay required before claiming "fixed" |
| 3 | New eval helper in `lite/gym/envs/lite/osworld/src/eval/metrics.py`; `Dockerfile` package add or env change; cross-domain refactor; new dataset-state list (e.g. a sibling of `_DROPPED_TEMPLATE_IDS`) | Verified hypothesis and live throwaway-container fix path; **main agent persists** | YES — main agent re-validates via `manually_verify_fix_path` or its own replay |

Subagent MAY interactively probe a live container with `docker exec -it`, type into the GUI through `replay_trajectory.py`, and apply *temporary* in-container changes (write a file, set a registry key, dismiss a dialog) to **prove the fix path reaches `score == 1.0`** — these mutations live and die with the throwaway container. For tier-2 changes specifically the subagent should agent-replay the actual recorded trajectory (NOT just oracle replay) using a `/tmp` patched JSONL/spec when practical; if the replay requires a persistent repo edit, the proposal must mark that replay as coordinator-owned and not claim "fixed" until the coordinator applies and verifies it.

**Critical distinction — Oracle replay vs Agent replay** (often confused; oracle alone is a false-negative trap for whole classes of bugs):

| | Oracle replay | Agent replay |
|---|---|---|
| Command | `validate/oracle/validate.py --fixtures <jsonl> --filter <task_id>` | `validate/rollout/replay_trajectory.py <task_id> --rollout $LOG_ROOT/<split>/<tid>` |
| What it does | `cp expected→source` then runs `setup + postconfig + eval` | Re-runs the agent's recorded `03_actions.json` against a fresh env, then `eval` |
| Settles | **gold/eval/postconfig are mutually self-consistent** (the contract is feasible by construction) | **a particular agent trajectory + any source-side fix actually scores 1.0** |
| Cannot settle | Whether the live UI can produce the byte-exact state `expected` requires (eval may demand more strictness than LO Writer / Calc / Impress / Chrome ever emits via UI) | Whether *another* agent / a *future* agent could do better (only one trajectory tested) |
| When to use | Tier-2 step 1: "is the contract sound?" | Tier-2 step 2: "does my fix unblock the actual failure?" — only this settles eval-strictness-vs-UI-output bugs |

For any bug hypothesis of the form "evaluator's exact comparison is stricter than what the UI can output after `Ctrl+S` round-trip" — oracle replay returns 1.0 (the contract is fine in vitro) but agent replay still returns 0 (UI never matches). Mis-applying oracle replay to such cases mis-classifies real bugs as "not a bug". Always pair the two: oracle establishes the contract is solvable; agent replay establishes the *agent* solves it.

**Inputs to read** (per candidate):

| What | Where |
|---|---|
| First screenshot — correct app/file open, no recovery dialog? | `sample_00/turn_0000/prompt_images/0000_reset.png` when debug artifacts are present; otherwise first `sample_00/images/*.png` |
| Last screenshot — what state did the agent leave things in? | latest `sample_00/turn_*/prompt_images/*.png` when present; otherwise latest `sample_00/images/*.png` |
| Trajectory — what did the agent actually do? | `sample_00/turn_*/01_prompt.txt`, `02_response.txt`, `03_actions.json`, matching `prompt_images_annotated/*.png`, `04_results.json`, and `05_timing.json` when present |
| Instruction + evaluator + setup | the JSONL row for that task_id |

**Targeted root-cause checks** (run all four on every candidate; the answer feeds the trigger label in the proposal):

1. **INITIAL_STATE_MISMATCH** — open `turn_0000/prompt_images/0000_reset.png` when present, otherwise the first canonical trajectory image, and re-read the instruction. Does the screenshot show the precondition the instruction assumes? ("empty xlsx" → workbook actually empty? "Chrome tab shows table X" → tab is loaded and populated?) Mismatch = setup bug. Cross-check with the JSONL `metadata.config` step ordering — common cause is one config step clobbering another (e.g. download after gold-py, or unfocused soffice when an `open` runs).
2. **EVAL_FILE_PATH_MISMATCH** — compare `metadata.evaluator.result.path` (where eval reads) against the file path the agent ended up writing to (parse trajectory `03_actions.json` for `save_file` / `file_path` mentions; check matching `prompt_images_annotated/*.png` of the save-as dialog if any). If the evaluator reads `/home/user/Desktop/X.docx` but the agent saved to `/home/user/Documents/X.docx`, eval will see stale or missing content. Often surfaces as "file mtime didn't advance during the run."
3. **POSTCONFIG_NO_OP** — read `metadata.evaluator.postconfig`. If it ends with a Ctrl+S (or programmatic save) intended to flush LO buffers, verify two things: (a) the right window was focused at end-of-trajectory (last screenshot), and (b) inside `manually_verify_fix_path`'s live container, run the postconfig steps and check the sink file's mtime — if mtime doesn't advance, the save was a no-op (window-focus race, modal dialog ate the keystrokes, or readonly mount).
4. **REFERENT_MISMATCH** — pull every quoted / named literal from the instruction (file basenames, document titles, deck names, sheet names, column headers, slide titles, target string values, URLs) and grep each one against the `metadata.config` heredoc that constructs the source state AND the `metadata.evaluator.expected` content. Every literal in the instruction must exist verbatim in either the source-build output OR the eval target — otherwise the agent is being asked to operate on something that doesn't exist in the world it's shown. Even when the operation succeeds technically (right shape index, right action), the agent often refuses / wanders / picks the wrong target because the named referent is invisible. This is a separate failure mode from INITIAL_STATE_MISMATCH (where state is wrong) — REFERENT_MISMATCH is where state is fine but the instruction's wording references a string the source doesn't contain. Common causes: parallel name-tables (TARGET-side dict and SOURCE-side dict drift apart after one is edited), instruction templated with `f"... '{SOMETHING_NEW}' ..."` where `SOMETHING_NEW` should have been `SOMETHING_EXISTING`.
5. **VACUOUS_PREDICATE** — for FileTasks whose `Param.parameters` carries a predicate (`rules_py` / threshold / filter clause) that operates on rows of the source data, evaluate the predicate against the source data **at audit time** and count matching rows. **0 matches = vacuous**: `expected.xlsx == source.xlsx` byte-equal, so any agent that saves without mutating (or even terminates without saving) passes eval. 1 match = fragile (any source-data tweak silently un-bugs or re-bugs the task). The audit signature: the agent's last 02_response.txt typically says "no rows match / nothing to do / already correct" + the trajectory shows 0 mutating actions (no type, no Format, no Ctrl+S that changes bytes). **Caught in cycle-43**: `f_calc_22__color_top_share` Param[1] predicate `row[1] < 0.05` but source data minimum Share was 0.12 → 0 matches → agent correctly observed "nothing to do" and eval passed vacuously at `n_turns=5` (slipped TRIVIAL_PASS's old `≤3` filter). Trigger label: **B**. Fix: change source data to include rows that satisfy the predicate, OR change the predicate threshold to one that has ≥2 matches.

**Repro the failure** in a fresh container — interactive form so the subagent can keep probing after the trajectory replays:

```bash
# Default: inherits env_kwargs from scripts/configs/gpt/default/lite.osworld.yaml
# (resolution=1024×768, max_steps=15, multi_apps domain_overrides=30, etc.) — must
# match the original rollout config or click coordinates land on wrong widgets at
# the env's default 1920×1080. Sleeps 1.0s between turns to mimic agent thinking
# pause (replay-without-pause races LO GUI updates, e.g. font picker dropdown
# still open when next click fires).
uv run python devs/envs/lite.osworld/validate/rollout/replay_trajectory.py <task_id>

# Specify rollout explicitly when default `<rollout>/{train,eval}/<tid>/sample_00` isn't found:
uv run python devs/envs/lite.osworld/validate/rollout/replay_trajectory.py <task_id> \
  --rollout "$LOG_ROOT/<split>/<task_id>"

# Override config / env_kwargs explicitly — useful when the original rollout used
# a non-default config (Claude / local Qwen) or the subagent wants to test a
# different resolution:
uv run python devs/envs/lite.osworld/validate/rollout/replay_trajectory.py <task_id> \
  --config-path scripts/configs/claude/default/lite.osworld.yaml
uv run python devs/envs/lite.osworld/validate/rollout/replay_trajectory.py <task_id> \
  --env-kwargs '{"resolution":[1024,768],"max_steps":15}'

# Stop after N turns (isolate a suspected bad step):
uv run python devs/envs/lite.osworld/validate/rollout/replay_trajectory.py <task_id> --max-turns 3

# Adjust inter-turn pacing if the GUI race issue surfaces (e.g. higher value for
# slow-rendering apps like Impress):
uv run python devs/envs/lite.osworld/validate/rollout/replay_trajectory.py <task_id> --inter-turn-sleep 2.0

# `--no-wait` skips the pause — only when the subagent doesn't need to inspect.
```

**Cycle 34 calibration note**: replay's env_kwargs / inter-turn-sleep MUST match the original rollout, otherwise the same agent actions reproduce a different end state. Symptom of mismatch: replay shows agent typing "Times New Roman" as document text (resolution 1920×1080 default; click(307,176) misses the font picker box) while the rollout's screenshot shows the font picker correctly populated — because the rollout ran at 1024×768 where (307,176) lands on the picker. After the cycle-34 fix the script defaults to inheriting the GPT rollout config; pass `--config-path ''` to disable inheritance only when reproducing a clean default-env scenario on purpose.

The script: loads JSONL spec → reads `turn_NNNN/03_actions.json` → resets fresh container with rollout-matched env_kwargs → steps through actions with `inter_turn_sleep` → runs `evaluate_final_fn` → prints score. The container stays alive at the "Press Enter to close" prompt; the subagent reuses it for the next steps. Run the replay in the background and tail the log so the subagent's shell stays free for `docker exec`.

**Inspect the live container** — interactive `docker exec` is allowed:

```bash
# Non-interactive one-shot inspection (snapshot a file, parse a config):
docker exec <container_name> bash -c "cat /home/user/.config/GIMP/2.10/gimprc"
docker exec <container_name> bash -c "python3 -c \"import json; print(json.load(open('/home/user/.config/google-chrome/Default/Preferences'))['profile'])\""

# Interactive shell when iterating on a hypothesis (modal dismissal sequences,
# xdotool window probes, GUI-state-dependent debugging):
docker exec -it <container_name> bash
# inside: xdotool search --name "Save"; sed -i 's/...' /etc/...; xdotool key Return; etc.
```

**Try the fix path live** — temporary in-container mutations are encouraged. Apply the candidate fix (write a file, dismiss a dialog, set a registry key, run an xdotool sequence) inside the live container, then re-trigger the evaluator without the agent acting again. Two practical mechanisms:

```bash
# (a) Replay only postconfig + eval, skipping all agent turns. Re-run the same
# trajectory with --max-turns 0 against a NEW container — useful when the fix
# is something that should be in the perturb's pre/post-config (write a file
# before agent acts, or extend LO_SAVE_POSTCONFIG); the subagent patches an
# in-memory JSONL row before launching, e.g. by writing a scratch copy under
# /tmp and pointing replay_trajectory.py at it.
uv run python devs/envs/lite.osworld/validate/rollout/replay_trajectory.py <task_id> --max-turns 0

# (b) Live container — keep the existing replay container alive (default mode
# pauses at "Press Enter"), exec into it, apply the candidate mutation, then
# call the evaluator directly via the same Python entry the script uses:
docker exec -it <container_name> bash
# inside container: <apply your in-place fix>
# (then exit and from host) re-attach replay_trajectory.py via its eval-only
# path, or run a small wrapper that imports osworld.eval and calls
# evaluate_final_fn(metadata['evaluator'], container=...).
```

If the score flips to 1.0 the proposed fix is verified. The container is throwaway — the change is gone on the next reset, and the persistent edit (Dockerfile / generator `.py` / eval template) is made by the main agent in `edit_generator_py`. If the live mutation does *not* flip the score, surface the discrepancy in the proposal: the hypothesis is wrong or incomplete.

**Output**: a written hypothesis ("LinkedIn cookie is in SQLite but invisible to Chrome — adding `--password-store=basic` should expose it") + the affected file/line + the **verified live fix-path log** showing score 1.0 in the throwaway container. The persistent code edit lives with the main agent (`edit_generator_py`), not the subagent.

---

### `manually_verify_fix_path`: optional pre-flight sanity check (NOT a gate)

A "**fix path**" = the concrete sequence of in-container operations (file writes, config edits, dialog dismissals, etc.) that, when applied by hand to a freshly-reset task container, makes `evaluate_final_fn` return `1.0`. The subagent already exercised this live in `subagent.diagnose` (see "Try the fix path live"); this section is the main agent's **optional** chance to re-verify before committing the persistent edit — useful when the subagent's container has drifted, or when porting the same fix to a sibling task.

**As of the latest plan update, this is informational, not gating.** All proposals proceed to `edit_generator_py` regardless of whether the main agent ran this check. The rationale: pre-fix manual verification is expensive (interactive container + probe); a wrong fix is cheaply caught post-restart by `family_returns_since` at scale. Invoke this only when a proposal is ambiguous and you want a quick sanity reality-check before editing.

The main agent re-runs the same interactive replay → applies the fix → confirms 1.0. **Pass = "looks right"** (proceed with confidence). **Fail = "looks wrong"** (still proceed, but flag the proposal in `logs.md` as risky so the next iteration's `family_returns_since` failure isn't a surprise). Skip when the subagent already produced a clean 1.0 log against a still-alive container the main agent can re-attach to.

```bash
uv run python devs/envs/lite.osworld/validate/rollout/replay_trajectory.py <task_id>   # interactive — pause before close
```

The container stays alive at the "Press Enter to close" prompt. While paused, `docker exec -it <container_name> bash` and:

| Hypothesis class | Manual exercise |
|---|---|
| **Eval bug** (eval rejects a correct final state) | Write the expected file/state by hand; re-run `evaluate_final_fn`; expect 1.0 |
| **Setup bug** (initial state makes the task unfeasible / vacuous) | Read the instruction, try to satisfy it as a competent human would, no oracle. If you can't → setup bug confirmed |
| **Instruction bug** (two reasonable readings, eval accepts only one) | Compare two readings against the evaluator output |

---

### `log_rejected_hypothesis`: record a rejected proposal in `logs.md`

Called when `manually_verify_fix_path` rejects a proposal (the fix path didn't reach 1.0 by hand, so the hypothesis is wrong). Append one line to `devs/envs/lite.osworld/validate/rollout/logs.md`:

```
<task_id>: hypothesis rejected at manually_verify_fix_path — <one-line why> [trigger=? | severity=cosmetic]
```

A rejected hypothesis is information for the next cycle (a different angle may surface real evidence). Never delete the line; subsequent cycles append, never overwrite.

---

### `classify_blast_radius`: 🟢 local / 🟡 family / 🔴 global   📌 SUBAGENT-MUST-READ (proposal/apply handoff)

"**Blast radius**" = the number of templates the fix touches. Count BEFORE editing — a fix that touches 3 templates is cheap to verify; one that touches 300 is a regression event waiting to happen and must be guarded by canary tests + sign-off.

```bash
grep -rl "_chrome_pref_postconfig" lite/gym/envs/lite/osworld/ | grep -v __pycache__   # example: helper grep
```

| Class | What it touches | Discipline |
|---|---|---|
| 🟢 Local | One template's params/eval/instruction | Coordinator fixes directly, then verifies on that one task |
| 🟡 Family | Helper used by 5–30 templates | Audit ≥3 family members before + after; coordinator spot-checks unaffected family canaries still score 1.0 |
| 🔴 Global | `utils/dispatch.py`, `eval/metrics.py`, `main.py`, `synth/noise.py`, Dockerfile | 5 sentinel tasks (2 fix-targets + 2 unrelated + 1 family canary); user sign-off required; one 🔴 coordinator commit |

**Hard rules for 🔴**: tasks that previously scored 1.0 must still score 1.0 (any flip → halt and reconsider). Never bundle harness changes with template changes in one commit. If `docker/*` changed, run `docker_rebuild` before restart.

**🔴 ≠ MUST RESTART.** Touching `eval/metrics.py` / `docker/*` / `harness/*` is a blast-radius classifier, NOT a restart trigger. The restart decision is independent (see audit-loop terminal gate's RESTART-THRESHOLD RULE):

| Harness change | Affects in-flight rolls? | Restart? |
|---|---|---|
| Added a NEW eval helper (`check_gitignore_has_entries`) that no existing JSONL row references | No — old rows still call old helpers | **No** |
| Added a NEW Dockerfile package not yet referenced by any generator | No — existing tasks don't import it | **No** |
| Changed semantics of an EXISTING eval helper that current rolls invoke | Yes — every in-flight pass/fail might flip | **Yes** |
| Dockerfile package upgrade that changes LO / GIMP / etc. behavior on already-rolled tasks | Yes — re-rolls would diverge from completed | **Yes** |

The rule: 🔴 changes added "for one feature" without behavior change to existing JSONL rows → DO NOT restart. Let the next sweep validate the new feature. Restarting for a single-feature harness add wastes the active sweep's progress without gaining signal.

---

### `edit_generator_py` + `regen_affected_jsonl_files`: never hand-edit JSONL   📌 SUBAGENT-MUST-READ (proposal/apply handoff)

JSONL files are byte-deterministic outputs of generator scripts. Hand-edits are silently overwritten on the next regen AND break the sha256 lock — three test cases (`test_eval_jsonl_byte_locked`, `test_train_synth_jsonl_byte_locked`, `test_train_perturb_jsonl_byte_locked`) verify the locks. The main/coordinator agent edits the generator `.py`, then regens, then updates the matching `.sha256`; subagents only propose the change and may use `/tmp` patched copies for diagnostics.

```bash
# Regenerate the affected splits:
uv run python -m lite.gym.envs.lite.osworld.src.gen.train --track synth     # train.synth.jsonl
uv run python -m lite.gym.envs.lite.osworld.src.gen.train --track perturb   # train.perturb.jsonl
uv run python -m lite.gym.envs.lite.osworld.src.gen.eval                    # eval.jsonl

# Update the matching sha256 lock in the same commit (only for splits whose
# JSONL bytes actually changed — `git diff --stat` to see which):
for split in eval train.synth train.perturb; do
  sha=$(sha256sum "lite/gym/envs/lite/osworld/data/${split}.jsonl" | cut -d' ' -f1)
  echo "$sha  lite/gym/envs/lite/osworld/data/${split}.jsonl" \
    > "lite/gym/envs/lite/osworld/data/.${split}.sha256"
done

# Verify the locks pass:
uv run --no-sync pytest tests/gym/envs/lite/osworld/test_lite_osworld.py -k byte_locked -q
```

Regenerate **once per cycle**, after all fixes are edited in. Per-fix-then-regen wastes generator runs without changing the output (regen is byte-deterministic in source).

---

### `replay_after_regen`: optional informational replay

Run after `regen_affected_jsonl_files()`. Pass = strong signal the fix is robust. Fail = **inconclusive**, do NOT reject the fix — common false-negatives:

- Recorded trajectory never reached the fix path (e.g. cookie-fix: agent never saw the seeded cookie pre-fix, recorded actions don't include "delete cookie").
- Setup-bug fix changes the initial state in a way old recorded actions don't match.
- Pixel-coordinate non-determinism — recorded clicks land on slightly different UI in a fresh container.

```bash
uv run python devs/envs/lite.osworld/validate/rollout/replay_trajectory.py <task_id> --no-wait
```

**Mandatory order: edit → regen → replay.** `replay_trajectory.py` reads the
task spec FROM THE JSONL ON DISK. If you edit a generator `.py` and immediately
run replay without `regen` in between, the replay uses the **stale JSONL** —
old gold / old eval — and silently re-tests the unfixed state. Symptoms: the
fix looks correct in source, but replay still returns the original score, and
you waste time inspecting eval semantics for a bug that isn't there.

**Single regen owner per cycle.** `train.synth.jsonl` / `train.perturb.jsonl` /
`eval.jsonl` are a shared resource — the main agent owns the regen call, and
runs it **exactly once per audit cycle** AFTER all source edits land. Reasons:

- Regen output is a deterministic function of the source `.py` files; multiple
  regens in one cycle produce the same bytes — wasted compute, no signal gain.
- Parallel regens from concurrent subagents would race on the
  output file; one would clobber the other mid-write and leave a torn JSONL
  that fails sha256 lock and silently corrupts downstream replay / rollout.
- Subagents are READ-ONLY auditors per `subagent.diagnose` — they propose
  fixes and verify them in throwaway containers (live in-container mutation,
  no source edit). They MUST NOT regen even when they have an Edit tool;
  surface the fix as a proposal and let the main agent serialise the regen.

So the workflow is: **subagents diagnose in parallel → main agent collects
proposals → main agent serially edits sources → main agent regens ONCE → then
verification (replay, family_returns_since)**. If a subagent's `subagent.diagnose`
needs to verify a fix that requires a regenerated JSONL (because the fix is in
the gold builder / initial config, not just postconfig / eval), the subagent
patches the JSONL row in-memory or under `/tmp/` and points `replay_trajectory.py`
at the patched copy — it never overwrites the shared file.

For the main agent's single-owner workflow:

```bash
# 1. Apply all source edits for the cycle
# 2. Regen jsonl + sha256 ONCE (per `edit_generator_py` section above)
uv run python -m lite.gym.envs.lite.osworld.src.gen.train --track synth
sha256sum lite/gym/envs/lite/osworld/data/train.synth.jsonl | cut -d' ' -f1 \
  > lite/gym/envs/lite/osworld/data/.train.synth.sha256
# 3. Quick check: does the new gold/eval bytes actually appear in jsonl?
grep "<task_id>" lite/gym/envs/lite/osworld/data/train.synth.jsonl \
  | jq -r '.metadata.config[].parameters.command // ""' \
  | grep -A5 "<distinguishing snippet from new gold>"
# 4. THEN replay each affected task
uv run python devs/envs/lite.osworld/validate/rollout/replay_trajectory.py <task_id> --no-wait
```

**Diagnostic vs configured-eval scores.** `replay_trajectory.py` prints TWO
distinct numbers and they mean different things:

- **`Final score`** — the score from running the task's CONFIGURED `evaluator`
  func (e.g. `compare_docx_files` text-only). This is what the actual rollout
  scores against. **This is the gate.**
- **`compare_docx_strict score`** — always-printed format-aware diagnostic
  that runs `compare_docx_strict` from `lite/gym/envs/lite/osworld/src/eval/metrics.py`
  on the result/expected pair regardless of configured eval. Useful for
  surfacing format/style mismatches the configured eval may ignore (e.g.
  `compare_docx_files` only checks paragraph text, so a font / alignment
  mismatch shows up here but not in the configured score). **This is
  informational** — a strict-score=0 with a configured-final-score=1 means
  the fix is fine for the actual eval; don't chase the strict diff unless
  the configured score is also 0.

When configured `Final score` fails inconclusively, rely on `family_returns_since` (post-restart re-roll) as the terminal gate.

**Family-cluster invariant — grep for siblings before declaring a cluster
fixed.** When a subagent identifies a "broad-gold-mutator" / "empty-rebuild" /
"TRIVIAL_PASS-shape" class of bug across N specific FileTasks, the cluster
may have **additional same-pattern siblings the audit did not flag** (because
those siblings happened to pass for unrelated reasons, weren't covered by the
sweep1 sample, or were filtered out before the audit ran). After applying the
mechanical cluster fix to the named templates, **grep the entire generator
for the same pattern**:

```bash
# Example: after replacing broad _gold_set_italic(idx) with _gold_set_title_italic(idx)
# in 9 audit-listed FileTasks, find any remaining broad-helper usage where the
# instruction is title-only:
grep -nE "Param\(_gold_set_(bold|italic|underline|text_alignment)\(" \
  lite/gym/envs/lite/osworld/src/gen/train/synth/libreoffice_impress.py \
  | grep -v "^.*:.*#"   # skip commented-out
# For each hit, read the surrounding FileTask: if its instruction is title-only
# (says "the title on slide N"), the broad helper is the same bug — fix it too.
```

This applies to any cluster-fix pattern: `_dst = Document()` empty-rebuild,
`compare_table` row-order sensitivity, ARGB color exact-match, etc.

---

### `touch_sentinel`: mark the deletion moment, return its path

```bash
SENTINEL="$LOG_ROOT/.audit_sentinel"
touch "$SENTINEL"
echo "$SENTINEL"   # the pseudocode binds this path to `sentinel`
```

Re-touched every cycle (so its mtime advances each time), so `find -newer "$SENTINEL"` in `family_returns_since` returns *only this cycle's* re-rolls. The pseudocode `sentinel = touch_sentinel(log_root)` binds the returned path to a variable that gets passed to `family_returns_since(fix, sentinel)` for the per-fix check.

Do **not** repurpose `$LOG_ROOT/run_info.txt` for this — it's written once at rollout launch, so `find -newer run_info.txt` matches every summary written since launch (the entire sweep), not just this cycle's batch.

---

### `stop_rollout`: drain a running rollout and env-server cleanly

```bash
: "${SESSION_ID:?set SESSION_ID before cleanup}"
: "${CUA_LITE_ENV_SERVER_URL:?set CUA_LITE_ENV_SERVER_URL before cleanup}"
: "${CUA_LITE_ENV_SERVER_TOKEN:?set CUA_LITE_ENV_SERVER_TOKEN before cleanup}"
curl -fsS -X DELETE -H "Authorization: Bearer ${CUA_LITE_ENV_SERVER_TOKEN}" \
  "${CUA_LITE_ENV_SERVER_URL}/instances?session_id=${SESSION_ID}&env_id=lite.osworld" || true
if [ -n "${ROLLOUT_PID_FILE:-}" ] && [ -f "${ROLLOUT_PID_FILE}" ]; then
  kill "$(cat "${ROLLOUT_PID_FILE}")" || true
fi
if [ -n "${ENV_SERVER_PID_FILE:-}" ] && [ -f "${ENV_SERVER_PID_FILE}" ]; then
  kill "$(cat "${ENV_SERVER_PID_FILE}")" || true
fi
sleep 30   # drain in-flight summary.json writes
```

Always called before `delete_summaries_for_affected` to avoid a fuzzy boundary between in-flight (stale-spec) and re-rolled (fixed-spec) tasks.

---

### `delete_summaries_for_affected`: queue tasks for re-roll   📌 SUBAGENT-MUST-READ (proposal/apply handoff)

The runner (`lite.infer.rollout.get_pending`) skips any task with an existing `sample_*/summary.json`. Deleting summaries enqueues those tasks; `restart_rollout` rolls them against the regenerated JSONL.

**Necessary AND sufficient condition for deletion: the task's SOURCE was edited in this cycle AND that edit changes the live env / eval semantics for this specific task.** "Source edited" alone is not enough — the second clause matters because `delete → restart` is what triggers the expensive 16-container teardown (see audit-loop terminal-gate RESTART-THRESHOLD RULE). Re-rolling a task whose source did not change is pure waste — same JSONL row + same agent → same trajectory — and burns provider quota plus 5–30 min wall-time per task. Over-deletion bleeds budget without improving signal.

**What qualifies (delete):**
1. Template's generator code was edited AND the regenerated JSONL row for this task has different bytes than the in-flight version (`src/gen/train/synth/<domain>.py` line that built this row's instruction / oracle / config / evaluator). Verify via `diff` of pre-/post-regen jsonl rows when in doubt.
2. Evaluator function semantics changed for code paths this task INVOKES (`src/eval/metrics.py` — but only when the helper is referenced by the task's `evaluator.func`). Adding a NEW unused helper to `metrics.py` does NOT qualify. Apply eval↔perturb cross-delete only when the shared code path is also reached by perturb variants.
3. Dockerfile / shared harness changed in a way that materially alters this task's env state (`docker/` — but only when the changed package / config affects this task's app stack). A new `gnome-settings-daemon-common` install affects `os` schema-dependent tasks; adding ffmpeg for a vlc-only feature does NOT trigger deletion of chrome/writer rolls.

**What does NOT qualify (do not delete):**
- `TURN_CEILING_HIT` with no source change. `max_steps` is fixed in the rollout YAML; bumping it applies to subsequent rolls automatically (since each new container reads the config at start), but **does NOT justify re-rolling already-completed tasks**. If the task genuinely needs more budget, lower its difficulty in the generator (fewer required actions, simpler instruction, drop a redundant step) — only then does the rule above kick in.
- `FALSE_NEG` / `FALSE_POS` / `TRIVIAL_PASS` whose `subagent.diagnose` concluded "agent error" / "capability ceiling" with no source-side fix. Log to `logs.md` as `[severity=cosmetic]` and move on.
- Pending-tail tasks (never rolled yet). `restart_rollout` picks them up automatically.
- Pure cosmetic changes (comment-only, doc-only edits in the generator).
- **Single-template eval-option flip (1-2 tasks)**. The cost of a restart-cycle (~30 min × 16 containers + provider quota) exceeds the value of 1-2 corrected datapoints. Log the affected task_ids to `logs.md` as `[severity=cosmetic | sweep-stale]`; the NEXT sweep auto-validates the fix.
- **New harness/eval/Dockerfile additions for an UPCOMING feature** that no in-flight roll references. Adding `check_gitignore_has_entries` to `metrics.py` doesn't retroactively change any roll's score → don't delete. The fix takes effect when the FEATURE templates land in a later cycle.

```bash
# Single task:
rm "$LOG_ROOT/<split>/<task_id>/sample_00/summary.json"

# Template family across all split subdirectories — delete only the
# templates whose generator was edited in this cycle:
for split_dir in "$LOG_ROOT"/*/; do
  for prefix in synth_chrome_set_pref synth_chrome_show_home; do
    find "$split_dir" -maxdepth 1 -type d -name "${prefix}_*" \
      -exec rm -f {}/sample_00/summary.json \;
  done
done
```

**Cleanup empty task dirs after a template is dropped** — when a template is
removed entirely (drop-list entry added, or generator rows deleted), the
re-rolled JSONL no longer contains those rows but the pre-existing task
directories at `$LOG_ROOT/<split>/<task_id>/` remain on disk. After fix-up,
delete the orphan directories themselves (not just their `summary.json`); empty
or near-empty folders pollute `find` output, mislead future audit scans
(making it look like coverage exists when it doesn't), and confuse anyone
reading the log tree.

```bash
# Drop orphan task dirs (no row in current JSONL):
: "${LOG_ROOT:?set LOG_ROOT before deleting task dirs}"
case "$LOG_ROOT" in
  .logs/rollout/*|"$PWD"/.logs/rollout/*|.exps/eval/*|"$PWD"/.exps/eval/*) ;;
  *) echo "Refusing to delete outside rollout/eval log roots: $LOG_ROOT" >&2; exit 1 ;;
esac
jq -r '.["task-id"]' "$JSONL" | sort -u > /tmp/current_tids.txt
for split_dir in "$LOG_ROOT"/*/; do
  for d in "$split_dir"*/; do
    tid=$(basename "$d")
    grep -qxF "$tid" /tmp/current_tids.txt || rm -rf "$d"
  done
done
```

**Cross-deletion rule** — symmetric: when fixed code is shared between an eval base task and its perturb variants, delete summaries on **both** sides.

- Eval task fixed (e.g. `osworld_chrome_<hash>` evaluator changed) → also delete `perturb_osworld_<domain>_<hash>_*` (shared evaluator/postconfig).
- Perturb generator fixed AND the diff touches `lite/gym/envs/lite/osworld/src/eval/*` (shared metric code) or `lite/gym/envs/lite/osworld/main.py` (shared task registration) → also delete the eval-side base. Check via `git diff --name-only main` against those paths.

If `lite/gym/envs/lite/osworld/docker/*` changed → run `docker_rebuild` first.

---

### `family_returns_since`: per-fix verification using the sentinel

Reports the new `episode_return` for every re-rolled task in a fix's affected list. Pass condition per fix: every affected family has ≥1 re-rolled task with `episode_return == 1.0`. With multiple variants per family, also check pass rate is not worse than untouched sibling families (no regression).

```bash
# Targeted: explicit task list
check_rerolled() {
  local LOG_ROOT="$1"; shift
  for tid in "$@"; do
    for split_dir in "$LOG_ROOT"/*/; do
      s="$split_dir$tid/sample_00/summary.json"
      [ -f "$s" ] && \
        echo "$(basename $split_dir)/$tid: $(python3 -c "import json; print(json.load(open('$s'))['episode_return'])")"
    done
  done
}

# Broad: every summary written since the sentinel (everything re-rolled this batch)
find "$LOG_ROOT" -name summary.json -newer "$LOG_ROOT/.audit_sentinel" \
  -exec python3 -c "
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
print(f'{p.parent.parent.parent.name}/{p.parent.parent.name}: {d[\"episode_return\"]}')
" {} \; | sort
```

If a fix's family still scores 0 after restart, three possibilities:

1. The fix is incomplete or wrong → revert just that fix; the other fixes stay applied (they're independently verified). The next inner-loop iteration's `scan` will surface this family again, route it through `group_by_root_cause` → a fresh `subagent.diagnose` for re-proposal.
2. Multiple root causes layered → peel off the next layer in the next iteration's `subagent.diagnose`.
3. Agent capability ceiling, not a bug → record the family in `logs.md` as `[severity=cosmetic]` and stop trying to fix it (move on to the next candidate; the rollout itself keeps running).

---

### `docker_rebuild`: rebuild the environment image (only when `docker/` changes)

```bash
git show --name-only HEAD | grep -q 'docker/' && echo "REBUILD NEEDED"

# Canonical — `install.sh` uses the right build context (osworld/ parent, not docker/).
# Running `docker build … docker/` directly silently fails because the Dockerfile
# does `COPY docker/server/ …` which only resolves against the parent.
uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/install.sh rebuild

docker images cua-lite/lite.osworld:latest --format '{{.ID}} {{.CreatedAt}}'
docker run --rm cua-lite/lite.osworld:latest bash -c "libreoffice --version && python3 -c 'import flask' && echo OK"
```

~5–10 min cold, ~3 min warm. Start in parallel with subagent diagnosis to hide the wait.

---

## Hard rules across the loop

- **One sweep = one log-root.** SWEEP_1 and SWEEP_2 must launch into different timestamped `$LOG_ROOT` directories. Within a sweep the log-root is fixed and re-used by every `restart_rollout` / `scan` / `delete_summaries_for_affected` / `family_returns_since` call — the in-place fix-and-re-roll cycle depends on this. Never alias SWEEP_2 onto SWEEP_1's directory: that intermixes pre-fix and post-fix attempts at the same `task_id` and silently corrupts every aggregated signal (homo_zero, variant homogeneity, family_returns_since).
- **Full rollout batches use env-server.** Sweep and restart batches run through a fresh dedicated env-server and the rollout command must pass its URL/token. Direct env mode is allowed only for smoke checks and replay probes.
- **Coordinator owns persistent edits.** Subagents can audit/probe in parallel, including throwaway container mutation and `/tmp/audit_*` scratch files, but the main/coordinator agent owns source edits, generated artifacts, rollout-root mutation, state-list/log updates, commits, and final "fixed" claims.
- Each replay or re-rollout launches a NEW container. Never `docker exec` into a leftover container from a prior attempt — stale state (cached cookies, profile dirs, running Chrome, X11 sockets) will mask real bugs.
- **No concurrent `replay_trajectory.py` while the sweep's containers are running.** The 16-concurrency rollout saturates host CPU/memory; spinning up an additional replay container during the active sweep triggers `EnvTimeoutError reset() 120s` in the production tasks (cycle-43 lost 5 tasks this way). If you MUST replay-verify mid-sweep, do it strictly sequentially (one replay container at a time, never parallel) AND understand each replay steals turns from the active sweep — prefer waiting for the sweep to drain. When dispatching subagents that may replay, tell them this rule explicitly in the spawn prompt.
- `replay_after_regen` and `family_returns_since` must use the **regenerated JSONL** (post-`regen_affected_jsonl_files`). Editing the generator `.py` without regen tests stale data.
- This plan assumes `--group-size 1` (single sample per task). All paths use `sample_00/`. With multi-sample, replace `sample_00` with `sample_*` in scan/delete/verify scripts and aggregate per-task scores yourself.
- Findings log is **append-only**: never delete a finding line; close it with `→ fixed in <hash>` instead.
- **Diagnostic outputs live under `/tmp/audit_*` only.** Subagent reports, intermediate JSONL patches, classification tables, and triage scratch files go to `/tmp/audit_<topic>.{md,jsonl}`. Never write scratch artifacts to the repo root, the rollout log-dir, or anywhere under `lite/` / `scripts/` / `devs/`. Main agent runs an end-of-cycle pass to remove `/tmp/audit_*` once the persistent fixes have landed. The repo root should contain only the long-lived `AGENTS.md` / `CLAUDE.md` / `README.md` documents.
- **Build commands come from `install.sh`, not from Dockerfile docstrings.** The Dockerfile's example build command uses the `docker/` directory as context, but the actual build needs the `osworld/` parent directory (where `COPY docker/server/` resolves). Always invoke `uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/install.sh rebuild`. A build that succeeds with `exit 0` but contained an inner `ERROR: failed to build` line is a hung-but-not-fatal partial layer — re-run via `install.sh` and inspect the tail of the build log if the image fingerprint hasn't moved.
- **Verify-before-remove SLA.** Removing an entry from a synth-side state list (`_BUG_TEMPLATE_IDS`, `_HARD_TEMPLATE_IDS`, `_DROPPED_TEMPLATE_IDS`) or marking a perturb row as `→ fixed` in `logs.md` follows the tier table in `subagent.diagnose`: after coordinator review, tier-1 changes can move directly; tier-2 changes must show a passing agent replay (or a passing fresh rollout sample if the recorded trajectory is too stale to replay); tier-3 changes must show the main agent's persistent edit + a smoke-test pass.
- **DO NOT prescribe agent actions in instructions.** An instruction may name the **target object + the target property + the target value** (e.g. "Bold the title on slide 1.", "Set the slide background to #FAF8F0."), but must NOT spell out the click-by-click UI procedure ("click the title textbox to enter text-edit mode, triple-click to select all text inside, then Ctrl+B"). Over-prescriptive instructions defeat the skill the eval is measuring — they reduce the task from "can the agent figure out how to bold a title in Impress" to "can the agent follow a literal recipe". **The rule of thumb**: if a competent human reading the instruction would know unambiguously WHAT to achieve and on WHICH artifact, leave the HOW to the agent. If the agent reliably fails the WHAT-clear instruction, that's a **CAPABILITY_CEILING** — classify it as `_HARD_TEMPLATE_IDS`, don't paper over it by hand-feeding UI steps.
   - What IS allowed: pinning down underspecified values (exact RGB hex when "blue" is too vague vs the eval's exact-match check; exact column letter when "the price column" is ambiguous; exact filename when multiple candidates exist). These resolve **instruction ambiguity**, not agent skill.
   - What is NOT allowed: spelling out widget selection sequences, keyboard shortcuts, menu paths, "do not retype X" guardrails, "use the size box not the dropdown", etc. If the agent's failure mode is "clicks wrong widget" or "retyped instead of selecting", that's a skill gap, not a generator bug.
   - Cycle-11 enforcement: the cycle-8 D-IMP-76 "click the existing title textbox..." rewrite and cycle-11 D-IMP-22 / D-IMP-25 same-pattern edits were **reverted** under this rule; those templates are now classified `_HARD` (free-floating textbox is a real UI complexity, but the instructions themselves are unambiguous — the agent just can't drive that particular UI cleanly).
- **State-list lifecycle (synth side).** The three lists in `lite/gym/envs/lite/osworld/src/gen/train/synth/catalog.py` are **active TODOs**, not historical archives — `logs.md` is the archive. Each list has a tight semantic role:

  | List | Filtered out of jsonl? | Role | Add when… | Remove when… |
  |---|---|---|---|---|
  | `_DROPPED_TEMPLATE_IDS` | YES | Task is fundamentally infeasible against the live env — eval contract impossible against the available UI / docker schema / upstream comparator with no clean fix path | Replay-verify subagent shows the gold/eval pair can never be satisfied (hard-coded literal in upstream eval, missing docker schema, network-dependent target, etc.) | Underlying obstruction is removed (e.g. new package added to Dockerfile, lite-custom evaluator written) and entry is re-validated |
  | `_HARD_TEMPLATE_IDS` | NO | Task is well-formed (oracle 1.0, instruction unambiguous, UI path reachable) but the current agent reliably fails to execute the procedure | ≥2 sweeps score 0 across multiple seeds AND a `subagent.diagnose` cycle confirms no source-side fix is appropriate (the task is testing a real skill the agent currently can't do) | A stronger model unblocks it, OR re-classification surfaces a previously-hidden BUG (in which case move to `_BUG_TEMPLATE_IDS` for an active fix) |
  | `_BUG_TEMPLATE_IDS` | NO | **Active TODO** for the current audit cycle — entry has an identified source-side fix path that hasn't landed yet | Subagent diagnoses a concrete fix (instruction edit, eval swap, gold builder rewrite, Dockerfile package, etc.) AND that fix has not yet been applied + verified | **Always** remove once the fix is applied AND verified per the tier table above. A `BUG` entry that is "fixed but kept for history" is wrong — `logs.md` is the historical archive, not the BUG list. If the fix turns out to NOT actually unblock the task, demote the entry to `_HARD_TEMPLATE_IDS` (agent-skill issue, not a generator bug) or `_DROPPED_TEMPLATE_IDS` (truly infeasible), don't leave it in BUG indefinitely |

  Concretely: the BUG list size at end-of-cycle should be **≤ remaining-unfixed-bugs-this-cycle**, never accumulate. The HARD list is allowed to accumulate (it's a snapshot of "what the current model can't do") but only with entries whose status has been re-confirmed each cycle. The DROPPED list reflects current docker / eval environment capability and should be revisited whenever infrastructure changes.

  Perturb has no equivalent state lists — perturb tracks per-row status inline in its own per-domain `.md` spec (and in `logs.md` via `→ fixed in <hash>` markers). The same lifecycle discipline applies: once a perturb row is fixed and verified, mark `→ fixed in <hash>` in `logs.md` and don't carry it forward.

---

## Trigger taxonomy   📌 SUBAGENT-MUST-READ

Scan signals — **every category here except `INFEASIBLE_CLAIM_EVAL_CORRECT` and `VARIANT_HOMOGENEITY_ONE` (when median `n_turns > 3`) is a bug to fix before shipping the dataset**:

| Signal | Source | Definition |
|---|---|---|
| FALSE_NEG | `summary.json` | `terminated=True ∧ episode_return==0` — agent self-stopped, eval gave 0 |
| FALSE_POS | `summary.json` | `truncated=True ∧ episode_return==1` — agent ran out of turns but eval gave 1 |
| TRIVIAL_PASS | `summary.json` | `terminated=True ∧ episode_return==1 ∧ n_turns ≤ 5` — eval may already be satisfied by initial state (cycle-43 widened from ≤3 after a vacuous task slipped through at n=5: agent inspected 4 turns + 1 terminate to confirm "nothing to do") |
| TURN_CEILING_HIT | `summary.json` | `truncated=True ∧ episode_return==0` — agent exhausted `max_steps` (15 default, 30 for `multi_apps`). Family-wide hits = task over-budgeted, instruction ambiguous, or recurring UI dialog steals turns |
| VARIANT_HOMOGENEITY_ZERO | aggregated across perturb variants of one base | ≥3 variants of a perturb base all score 0. N independent identical agent failures unlikely → almost certain base-level setup/eval bug |
| VARIANT_HOMOGENEITY_ONE | aggregated across perturb variants of one base | ≥3 variants all score 1 *and* median `n_turns ≤ 3`. Knob may be vacuous — perturbation didn't actually change task difficulty, training signal is diluted. Investigate but lower priority than _ZERO |
| INFEASIBLE_CLAIM_TRAIN | `02_response.txt` | Train-split task where the agent invoked `report_infeasible(reason=…)`. **Train tasks should NEVER be infeasible by construction** (synth filters infeasibility; perturb is built on feasible bases) — any hit is a setup/generator bug |
| INFEASIBLE_CLAIM_EVAL_UNMARKED | `02_response.txt` + `eval.jsonl` | Eval-split task where the agent invoked `report_infeasible` but `eval.jsonl[task].metadata.evaluator.func != "infeasible"`. Either the agent is over-cautious OR the upstream eval is missing an `infeasible` marker — needs review |
| INFEASIBLE_CLAIM_EVAL_CORRECT | `02_response.txt` + `eval.jsonl` | Agent invoked `report_infeasible` on an eval task whose `func == "infeasible"`. Aligned with ground truth — informational only, **not** a candidate |

Note on infeasibility plumbing: `LiteOsworldEnv.step` (`lite/gym/envs/lite/osworld/main.py`) intercepts `report_infeasible` and routes it straight to `evaluate_final_fn` with `terminated=True` — it does **not** rewrite the call, so the verbatim `report_infeasible` survives in `03_actions.json`'s `lite_message.tool_calls`. `summary.json` still cannot distinguish "agent gave up" from "agent claimed infeasible" (it carries no per-call tool names), so the `scan` script recovers that signal from `turn_*/02_response.txt` (equivalently `turn_*/03_actions.json`).

Root-cause triggers (what's actually wrong, once a candidate is reviewed). Triggers I/J/K are surfaced during `subagent.diagnose` and don't map to a single `summary.json` predicate — they need cross-referencing JSONL metadata + screenshots + trajectory:

| Trigger | Meaning |
|---|---|
| A | instruction-vs-eval asymmetry (instr asks X, eval checks Y) |
| B | setup doesn't fail the pre-condition (initial state already satisfies eval) |
| E | env / state mismatch — eval reads a path the setup never seeded; or seeded state is invisible to the agent (wrong profile, locale, keyring) |
| H | UI dialog hijacks `turn_0000/prompt_images/0000_reset.png` or the first canonical trajectory image state (recovery / welcome modal blocks the agent's first action) |
| N | task fundamentally requires more turns than `max_steps` vision-only |
| O | output-format brittleness — eval's exact-match rejects a semantically-correct answer (whitespace, case, locale formatting, autocorrect) |
| I | INITIAL_STATE_MISMATCH — `turn_0000/prompt_images/0000_reset.png` or the first canonical trajectory image doesn't show the precondition the instruction assumes ("empty xlsx" but populated, "Chrome tab loaded" but blank). Usually a config-step ordering bug: a later step clobbered an earlier seed (download after gold-py, base config writes after perturb's empty-step) |
| J | EVAL_FILE_PATH_MISMATCH — `evaluator.result.path` differs from the file the agent actually wrote to (save-as default dir vs. eval read dir, ext mismatch, wrong profile path). Eval reads stale or missing content even though the agent did the work |
| K | POSTCONFIG_NO_OP — evaluator's `postconfig` Ctrl+S (or programmatic save) silently does nothing because the wrong window has focus, a modal dialog ate the keystrokes, or the file is on a readonly mount. Sink file mtime doesn't advance during the run |
| L | REFERENT_MISMATCH — instruction names a literal (`"the X deck"`, `"the Y sheet"`, file basename, column header) that does NOT exist verbatim in the source state. Operationally distinct from I: state may be perfectly correct, but the instruction's nouns reference strings the source doesn't contain, so the agent has nothing matching to act on. Typically caused by parallel name-tables in the generator drifting apart (one dict edited, the sibling left stale) — e.g. an `{INSTRUCTION_LABELS}` dict holding human-readable display names while `{SOURCE_DECK_TITLES}` holds the actual literal strings written into the file. Cycle-2 caught this pattern in TEXT_FAMILIES (instruction referenced `_NEW_TITLES[fam]` but source slide 1 was `TEXT_FAMILIES[fam][0][0]`, a different string) |

---

## Sweep 2 preconditions

Sweep 2 is the independent confirmation of sweep 1. Before launching:

```bash
git log --oneline -10                                              # all sweep-1 fixes on main
uv run --no-sync pytest tests/gym/envs/lite/osworld/test_lite_osworld.py -q       # tests green
# Pick a NEW log-root timestamp (never reuse sweep-1's)
```

---

## Exit criteria

Freeze for SFT export when **both** sweeps satisfy:

1. **Sweep 1** (log-root A) — rollout completes all 1679 tasks; 0 unresolved eval bugs (every FALSE_NEG/FALSE_POS either confirmed agent error or fixed+verified within the sweep).
2. **Sweep 2** (log-root B) — fresh run from `main` after all sweep-1 fixes are committed; same condition.

If sweep 2 surfaces new eval bugs, fix them via the inner loop and run a sweep 3 as the new confirmation.
