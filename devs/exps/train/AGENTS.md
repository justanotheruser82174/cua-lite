# Training experiments

Standardized layout for repeatable, version-pinned training runs across all envs in the cua-lite matrix. Each `<env>/<recipe>/AGENTS.md` is a self-contained recipe; running it produces one committed eval snapshot per **(commit, run_id)** pair plus gitignored raw artifacts (trajectories + checkpoints) under `.exps/train/...`.

This README is **self-contained**: layout, container/SESSION_ID rules, recipe contract, snapshot template, and the agent workflow all live here. Pair it with [`devs/exps/eval/AGENTS.md`](/devs/exps/eval/AGENTS.md) — the structure is intentionally parallel (commits → runs → slugs) so a single mental model covers both.

---

## Layout

```
devs/exps/train/
├── AGENTS.md                              # this file — workflow spec + container rules + snapshot template
├── <env>/                                 # lite.osworld | osworld | androidlab | androidworld
│   └── <recipe>/                          # qwen3_vl_2b | qwen3_5_2b | ...
│       ├── AGENTS.md                      # the recipe (end-to-end teacher → SFT → RL → eval steps)
│       ├── run.sh                         # optional thin launcher (free for now; recipe README is source of truth)
│       ├── CHANGELOG                      # commit-pair diffs that affect this recipe's numbers
│       └── logs/
│           └── <commit-ts>_<commit>/
│               └── <run_id>.md            # one snapshot per (commit, run_id); multiple per commit allowed
└── ...

.exps/train/<env>/<recipe>/<commit-ts>_<commit>/<run_id>/<stage>/   # raw artifacts — gitignored
```

| Token | How resolved | Example |
|---|---|---|
| `<env>` | recipe's grandparent dir | `lite.osworld` |
| `<recipe>` | recipe's parent dir; agent family + size, lowercased | `qwen3_vl_2b`, `qwen3_5_2b` |
| `<commit-ts>_<commit>` | `git log -1 --date=format:'%Y-%m-%dT%H-%M' --format=%cd HEAD` + `_` + `git rev-parse --short HEAD`. The `<commit-ts>` prefix is the commit's committer time at minute precision — sortable chronologically and unique enough to distinguish two commits in the same minute via the sha suffix. | `2026-04-26T21-02_658c25b` |
| `<run_id>` | `$TRAIN_RUN_ID` — required, agent-chosen campaign id. Schema: `run_<N>` default (e.g. `run_0`, `run_1`), optionally `run_<N>_<label>` to name the campaign's intent. See [Run id contract](#run-id-contract-train_run_id). Unlike eval's `EVAL_RUN_ID`, the recipe READMEs treat `TRAIN_RUN_ID` as required (recipes are explicit about which `<run_id>` to use, and resume hooks span multiple stages — auto-resolution is too risky). | `run_0`, `run_1_lr-sweep` |
| `<stage>` | per-recipe step that emits artifacts | `teacher_rollout`, `sft`, `grpo`, `eval_base`, `eval_sft`, `eval_grpo` |

Same `(commit, run_id, recipe)` → resume; bumping any of them → fresh `.exps/train/...` subtree.

The layout intentionally mirrors [`devs/exps/eval/`](/devs/exps/eval/AGENTS.md) (`<env>/{run.sh, CHANGELOG, logs/<commit-ts>_<commit>/<run_id>.md}`), with one extra `<recipe>/` level under `<env>/` because each env hosts multiple training recipes.

> **Why `<commit-ts>_<commit>`?** Sortable chronologically (sha alone is opaque). Computed from `HEAD`'s committer time (not the wall clock), so the path stays stable across re-runs of the same commit — only `<run_id>` advances when you start a new campaign at the same commit.

---

## Env-server prerequisite

Every training recipe drives envs over HTTP via [`LiteEnvClient`](/lite/gym/remote/client.py) — the Slime container does NOT spawn envs as siblings via a mounted docker socket. Probe the URL you plan to use:

```bash
curl -sf -H "Authorization: Bearer $CUA_LITE_ENV_SERVER_TOKEN" \
  "$CUA_LITE_ENV_SERVER_URL/envs/$ENV_ID" | jq .
# → {"available": true,  "n_tasks": N, ...}   ready
# → {"available": false, "error": "..."}      env setup missing on server host
# → HTTP 404                                  env_id not registered on this server
```

No env-server reachable? Start a local one with [`serve_env.py`](/scripts/serve_env.py) — its host must already have the env's deps installed per the env's README:

```bash
uv run python scripts/serve_env.py --port 30100
# Use the host's external IP, NOT 'localhost'. The Slime container runs
# in rootless docker's bridge network — its 'localhost' is its own
# loopback, not the host's. For a remote env-server use
# ``http://<env-host>:30100``.
export CUA_LITE_ENV_SERVER_URL=http://$(hostname -I | awk '{print $1}'):30100
export CUA_LITE_ENV_SERVER_TOKEN=$(whoami)                  # any string; passthrough mode keys by sha256(token)[:6]
```

`scripts/train/slime/launch.sh` forwards `CUA_LITE_ENV_SERVER_URL` / `CUA_LITE_ENV_SERVER_TOKEN` into the Slime container; `run_grpo.sh` / `run_reinforce.sh` fail-fast via [`scripts/train/utils/preflight.sh`](/scripts/train/utils/preflight.sh) if the probe fails at training launch.

See [Env-server](/docs/envs.md#env-server) for details.

---

## ⚠️ Commit-first discipline + pipeline-aware path reuse

**Always `git commit` before launching a recipe.** Pipeline files (the set that, when edited, changes training/eval numerics) must be clean — see [eval's pipeline list](/devs/exps/eval/AGENTS.md#-commit-first-discipline--pipeline-aware-path-reuse), with the same logic applied per env. The point of the commit-keyed layout is **idempotent reproduction** — same pipeline state + same recipe → same numbers stage by stage.

But **doc-only commits don't fragment paths**. The recipe's `BASE_HOST` formula uses HEAD directly today (training is single-recipe-per-commit so the issue is rarer than for eval), but the eval-side `run.sh` invocations the recipe makes (Step 4 / Step 7 evals, `scripts/rollout.py` calls) inherit eval's pipeline-aware reuse — so a CHANGELOG/README commit between training stages doesn't move where the eval data lands.

For training this rule hits harder than for eval: stages span days and resume hooks (megatron `--load`, slime `iter_*` skips, parquet existence) carry forward checkpoints produced under the wrong code if you patch a pipeline file mid-campaign. Same `TRAIN_RUN_ID` + same pipeline state → same artifact tree → idempotent stage-by-stage reproduction. Anything else corrupts the chain.

**If you patch a pipeline file mid-campaign**: stop, commit, then decide:
- **safest**: bump to `run_<max+1>` and start fresh at the new commit (treat the prior tree as orphaned data)
- **acceptable** for cosmetic / non-numerical-impact changes: continue under the kickoff hash and add a `code drifted mid-campaign` note to the snapshot's `Notes`. See the `2026-04-27T23-21_828ea14` grounding snapshots for the example we now want to avoid.

---

## Run id contract (`TRAIN_RUN_ID`)

**The rule, in one sentence:** `TRAIN_RUN_ID` is the campaign identifier; **one campaign = one `TRAIN_RUN_ID`, used by every stage of the recipe AND every failure-restart of every stage, regardless of how many days it spans**. Bump only to start a brand-new campaign at the same commit.

This matters because resume relies on it:

```
.exps/train/<env>/<recipe>/<commit-ts>_<commit>/<run_id>/<stage>/
                                                  └─────────────────┘
                                                  same path → resume; different path → start over.
```

If you change `TRAIN_RUN_ID` mid-campaign (e.g. by picking a new value when restarting after a crash), the next stage writes to a *different* path and the recipe's resume hooks (megatron `--load`, slime `iter_*` skips, parquet existence) all break — you'd silently rerun teacher rollouts, re-export SFT data, and re-tokenize from scratch.

**Schema** (what to put in `TRAIN_RUN_ID`):

- Default: `run_<N>` (e.g. `run_0`, `run_1`, `run_2`) — incrementing per fresh campaign at the same commit.
- With label: `run_<N>_<label>` (e.g. `run_0_async2`, `run_1_lr-sweep`) — name the *intent* of the new campaign in the label. Free-form. Useful for variance re-runs, hyperparameter tweaks, recipe-knob experiments at the same commit.

### What "one campaign" means

- ✅ Day-1 starts teacher rollout + SFT, day-2 launches GRPO, day-3 evals all ckpts. **All same `TRAIN_RUN_ID`.**
- ✅ GRPO crashes with `ActorUnavailableError`; you fix the underlying issue and rerun. **Same `TRAIN_RUN_ID`** so it picks up the previously saved megatron ckpt.
- ✅ Mop-up pass to fill in missing eval-iter snapshots. **Same `TRAIN_RUN_ID`.**
- ❌ You ran a campaign last week, now want to re-run the same recipe at the same commit (e.g. variance check). **New `TRAIN_RUN_ID` — that's a separate campaign.**

### How to keep it stable

```bash
export TRAIN_RUN_ID="run_0"                # ← fixed string literal; reused across all stages and restarts
# NOT:
export TRAIN_RUN_ID="run_$(some-counter)"  # ← never compute it on the fly
```

Recover after a session ends: `ls .exps/train/<env>/<recipe>/<commit-ts>_<commit>/` — pick the existing `run_<N>` and re-`export TRAIN_RUN_ID` to that exact value (or `run_<max+1>` if you want a fresh campaign at the same commit).

---

## Matrix — which recipes exist per env

Mirrors [`devs/exps/eval/AGENTS.md#matrix`](/devs/exps/eval/AGENTS.md#matrix--which-agent-on-which-env) but rows are training recipes (not models) and columns are envs. ✅ = recipe written and runnable; ⚠️ = directory stubbed (`AGENTS.md` empty); — = not in scope.

| Recipe | lite.osworld | osworld | androidlab | androidworld |
|---|---|---|---|---|
| `qwen3_vl_2b` | ✅ | — | — | ✅ |
| `qwen3_5_2b`  | ✅ | — | — | ✅ |

To add a recipe: `mkdir -p devs/exps/train/<env>/<recipe>/{logs}` then write `AGENTS.md` against this framework — clone the closest existing recipe and edit. Empty stubs (`AGENTS.md` 0 bytes) are valid placeholders that signal intent.

---

## Running

> **GPU IDs are host-specific.** Use only the user-given allowed range; never hardcode. See [GPU reservation discipline](#gpu-reservation-discipline) for watchdog rules.

Each recipe's own `AGENTS.md` is the running guide — open `<env>/<recipe>/AGENTS.md` and execute Step 1 → Step 7 in order. The recipe assumes you've already done the kickoff per [Agent workflow checklist](#agent-workflow-checklist) below: `cd` to repo root, `export TRAIN_RUN_ID=<fixed>`, watchdog the allowed GPU range, leak survey, container launch + init.

```bash
cd /path/to/cua-lite
export TRAIN_RUN_ID="run_0"                                            # fixed; see Run id contract
SESSION_ID=train-<env>-<recipe> CUDA_VISIBLE_DEVICES=<gpus> \
  bash scripts/train/slime/launch.sh                                # only if a stage needs the slime container; auto-runs slime/init.sh inside
# then follow the recipe step by step
```

## SFT Export Evidence

Every recipe step that runs `python -m lite.train.export.export_sft` must record
the export evidence in that recipe's run snapshot before training consumes the
parquet. Required fields are: exact export command, source rollout/dataset path,
filter expression, export config path, `--model-id`, output parquet path, row
count, strict validation result, and one sample-inspection artifact or rendered
prompt note. This mirrors the real data promotion evidence contract in
[`devs/data/AGENTS.md`](/devs/data/AGENTS.md) and prevents a training run from
silently using an unvalidated or wrong-template SFT parquet.

---

## GPU reservation discipline

Same rules as [`devs/exps/eval/AGENTS.md#gpu-reservation-discipline`](/devs/exps/eval/AGENTS.md#gpu-reservation-discipline) — re-read it before launching anything. Training-specific deltas:

- **Long-hold pattern.** Training stages hold their GPUs continuously for hours (vs. eval's per-model burst+release loop). Watchdogs on **non-training** GPUs in the allowed range stay held for the entire campaign; only the training-stage GPUs flip from watchdog → training → watchdog at stage boundaries.
- **Reserve at the docker level.** `scripts/train/slime/launch.sh` passes `CUDA_VISIBLE_DEVICES` through to docker as `--gpus device=N`; the slime container only sees those physical GPUs, regardless of what `nvidia-smi` later reports on the host. Pass the **exact** allowed-range GPUs you intend to use; `--gpus all` (the script's fallback when `CUDA_VISIBLE_DEVICES` is unset) is never appropriate.
- **Inside the container, GPUs renumber to `0..N-1`.** Recipes use `CUDA_VISIBLE_DEVICES=0,1` inside `docker exec` — that's the docker remap, not the host range. The watchdog discipline is host-side; container-side GPU usage is fully described by `--gpus device=...`.
- **Stop-the-world for training-stage launches.** If a stage needs the entire allowed range (no headroom for held watchdogs), release them all the moment the launch starts and re-hold the moment training exits — same minimization rule as eval. With `ASYNC=1` (1 train + 1 rollout GPU) you can keep extra-allowed GPUs under watchdog throughout.

If no allowed range was given, **ask the user**.

---

## OOM escalation

Two layers, picked by where the OOM happens.

**(A) SFT / GRPO training (inside the slime container).** Two batching regimes, pick the matching lever:

- **qwen3_vl recipes** (`run_sft.sh` / `run_grpo.sh` — uses `--use-dynamic-batch-size`): the primary VRAM lever is `MAX_TOKENS_PER_GPU` (passes through to `--max-tokens-per-gpu`, default `2048`). Lower it to `1024` / `512` to shrink the per-GPU dynamic batch. Costs throughput, unblocks tight VRAM. `MBS` is **not consumed** by these scripts.
- **qwen3_5 recipes** (same `run_sft.sh` / `run_grpo.sh` with a `Qwen/Qwen3.5-*` `MODEL_ID` — the scripts auto-derive `MODEL_FAMILY=qwen3_5` and use fixed `--micro-batch-size ${MBS:-1}` because GDN's BSHD layout disallows dynamic batching): the lever is `MBS`: `1 → 2 → 4`. Each doubling halves activation memory of the largest forward pass. `MAX_TOKENS_PER_GPU` doesn't apply. (The old standalone `run_*_qwen3_5.sh` scripts were merged into the base scripts in the slime v0.3.0 migration.)

Additional cross-family levers (v0.3.0): `GRADS_FP32=0` (bf16 grad accumulation, ~2 bytes/param) and `OPTIM_CPU_OFFLOAD=1` (Adam state → host RAM, ~8 bytes/param) — see the script headers.

After exhausting the family-specific batching lever, escalate further (both families):

1. `TP_SIZE`: raise from the default `1` (needs ≥`TP_SIZE` GPUs in `NUM_TRAIN_GPUS`), up to the model's KV-head ceiling for qwen3_5 (Qwen3-VL has no ceiling — uncapped) — TP shards weights/activations/optimizer across ranks, cutting per-GPU memory. Note bumping `NUM_TRAIN_GPUS` alone only adds DP (model replicated, per-GPU memory unchanged).
2. `--recompute-num-layers <N>`: already on at `N=1` (in `BACKEND_ARGS` of the launch scripts). Bump to `2`/`3` only as a last resort — adds wall-clock overhead.
3. Reduce `ROLLOUT_BATCH_SIZE` and/or `N_SAMPLES_PER_PROMPT`: shrinks the rollout burst's KV cache. Don't drop `N_SAMPLES_PER_PROMPT` below `4` for GRPO — group_size < 4 makes advantage estimation noisy.

**(B) Host-side rollout / eval** (`scripts/rollout.py` for teacher rollout / SFT eval / GRPO eval): same `tp_size` doubling rule as eval — see [`devs/exps/eval/AGENTS.md#oom-escalation`](/devs/exps/eval/AGENTS.md#oom-escalation).

If a Ray actor dies with `ActorUnavailableError` mid-training, that's almost always a downstream OOM in either the rollout sglang engine or the train megatron actor — apply (A) on the training side, (B) on the rollout side.

---

## Leak survey + cleanup (between every stage)

> ⚠️ **Scope cleanup to your own `SESSION_ID` (`train.<env>.<recipe>`). Never kill what you didn't start.** Multi-tenant hosts may have concurrent `lite.slime-*` / `lite-env-*` / ray / sglang processes you don't own — leave them alone. No-arg `/cleanup` is safest (this session only); named-arg scopes widen — read [`.claude/skills/cleanup/SKILL.md`](/.claude/skills/cleanup/SKILL.md) first. Notable footguns: `/cleanup training` `pkill -9 python` is **container-only safe** (host invocation kills unrelated python); `/cleanup osworld` with `SESSION_ID` unset sweeps **every** `-osworld-` container on the daemon; `pkill`-pattern envs (`androidworld` / `webgym`) are host-global on the host. CLAUDE.md's rule applies: **never** stop, kill, or touch a Docker container you didn't start.

Aborted training stages leave more debris than eval: orphan `lite.slime-*` containers, leaked Ray workers / sglang servers (all on the Slime host), plus (during rollout / eval stages) the same `lite-env-*` / qemu / remote-VM debris **on the env-server host** (which may or may not be the same machine as Slime). They don't free RAM/CPU/VRAM on their own and silently starve the next stage.

Survey before / after each stage, and every 10-15 min during long ones — both layers:

```bash
# Slime-side (training stack):
echo "=== gpu ==="; nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo "=== docker slime containers ==="; \
  docker ps --filter "name=lite.slime-" --format '{{.Names}} ({{.Status}})'
echo "=== orphan rollout python ==="; ps -eo pid,etime,cmd | grep -E 'rollout/local\.py' | grep -v grep || echo "none"
echo "=== ray status (in your slime container) ==="; \
  docker exec lite.slime-${SESSION_ID:?} ray status 2>/dev/null || true

# Env-server-side (env stack — query the env-server instead of host docker):
curl -s -H "Authorization: Bearer ${CUA_LITE_ENV_SERVER_TOKEN}" \
     "${CUA_LITE_ENV_SERVER_URL}/instances" | jq .
# On the env host (only if you have shell there): docker ps --filter "name=lite-env-" ...
```

Anything **not** under your `SESSION_ID` — leave it alone. For your own debris, the relevant `/cleanup` scopes:

| Scope | Targets |
|---|---|
| `/cleanup` (no args) | only what *this session* launched (safest — review SKILL.md first) |
| HTTP `DELETE ${CUA_LITE_ENV_SERVER_URL}/instances?session_id=<sid>&env_id=<env>` | server-side reap of `lite-env-${SESSION_ID}-*` env containers on the env host (works even when env host ≠ Slime host) |
| `/cleanup <your env>` | tier-1 script — runs on the env host only (no-op from Slime host when env-server is remote); same scoping as the HTTP path |
| `/cleanup training` | sglang / ray / python — **container-only by default; host invocation will `pkill -9 python` and kill unrelated processes** |

The full table + cross-tenant warnings live in [`devs/exps/eval/AGENTS.md#leak-survey--cleanup-between-every-run`](/devs/exps/eval/AGENTS.md#leak-survey--cleanup-between-every-run).

**Never `docker stop` / `docker rm` a `lite.slime-*` container you didn't launch under your `SESSION_ID`** — it may belong to a concurrent training run and contain hours of un-snapshotted progress.

---

## Reading utilization

A single `nvidia-smi` snapshot at low util doesn't mean training is stuck. Sources of forward-progress signal, in order of preference:

1. **Wandb step counter** (`cua-lite-dev` by default; `cua-lite` for curated runs): the authoritative training pulse. If `_step` is monotonically advancing and `rollout/raw_reward` etc. are non-stale, training is alive.
2. **`data.py:219 - rollout N` log lines** (slime stdout): emitted at every train step inside the megatron actor. Rate ≈ 1 per `rollout_time` seconds (from the adjacent `perf` lines).
3. **`iter_<N>` ckpt cadence**: with `SAVE_INTERVAL=5` (default), expect a new HF ckpt under `${BASE_CTR}/<stage>/hf/iter_{4,9,14,...}` every ~5 train steps. Long gaps with no new iter dir = stuck.
4. **`nvidia-smi` heuristic**: training GPUs swing between high-util compute bursts (90-100%) and short collective-comm gaps (0-30%). Sample 3-5 readings over ~10s — a single 0% reading proves nothing. Same caveat as [eval's reading-utilization rule](/devs/exps/eval/AGENTS.md#reading-utilization).
5. **System-level resource pressure** (the silent killers, easy to forget):
   ```bash
   free -g | head -3                   # available RAM (should stay > 50 GiB on a 1.5 TiB host)
   top -bn1 | head -3                  # load avg vs cpu count
   df -h /tmp ${BASE_HOST%/*}          # disk: /tmp (DUMP=1) + .exps/ (ckpts + per-task PNGs)
   docker ps --filter "name=lite-env-" --format '{{.Names}}' | wc -l   # rollout fan-out
   ```
   Sustained low `available` (< 20 GiB) or load_avg ≫ NCPU is a setup for the **Linux OOM killer** to SIGKILL sglang/training without warning — that's the most common silent failure mode for training jobs that "just die". `df` for `/tmp` matters even without `DUMP=1` (sglang spills, ray plasma store).

What actually means stuck:

- **Ray `ActorUnavailableError`** in slime stdout — actor died, almost always a downstream OOM. Apply [OOM escalation](#oom-escalation), re-launch.
- **sglang `Gracefully exiting... Remaining number of requests N` looping** with GPU memory dropped — apply the [Graceful-exit / hung-rollout recipe](#agent-workflow-checklist) in Stage 3 of the agent workflow.
- **No new `data.py:219 rollout N` line in 30+ min** (assuming `rollout_time` was < ~10 min) — rollout cluster wedged. Run the leak survey, restart only **your** `SESSION_ID`'s stale `lite-env-*` containers via `/cleanup <env>`, re-launch.

---

## Where each stage runs

| Stage | Where | Driver |
|---|---|---|
| Teacher rollout | inside Slime container; envs reached via env-server | `docker exec -w /workspaces/cua-lite lite.slime-${SESSION_ID} python scripts/rollout.py` |
| SFT parquet export | inside Slime container | `python -m lite.train.export.export_sft` |
| SFT training | inside Slime container | `bash scripts/train/run_sft.sh` |
| GRPO / REINFORCE training | inside Slime container | `bash scripts/train/run_grpo.sh` |
| Eval (base / SFT / RL ckpts) | inside Slime container; envs reached via env-server | `docker exec -w /workspaces/cua-lite lite.slime-${SESSION_ID} python scripts/rollout.py --model-path ...` |

Architecture (current default): the training + sglang stack runs inside the Slime container, but envs do **not** spawn as siblings via a mounted docker socket. Instead the Slime container has `CUA_LITE_ENV_SERVER_URL` + `CUA_LITE_ENV_SERVER_TOKEN` exported; `LiteEnvClient` transparently routes every `gym.make` over HTTP to [`serve_env.py`](/scripts/serve_env.py), which spawns the `lite-env-*` env containers / KVM emulators / remote VMs on the **env host** (same machine as Slime by default, or a separate machine — set the URL to the env-server's external IP either way; rootless docker bridge means `localhost` from inside the Slime container would not reach the host). The training stack and the rollout stack share the same Megatron + sglang environment inside Slime; the env stack is fully decoupled. See [`docs/slime.md`](/docs/slime.md) + [`docs/envs.md#env-server`](/docs/envs.md#env-server) for the full rationale and the same-host vs remote-host trade-off.

---

## Container setup (one-time per recipe)

Every recipe that needs a Slime container creates exactly one. Follow [`docs/slime.md`](/docs/slime.md) for build / launch flow. The recipe-specific bits:

### `SESSION_ID` convention

**`SESSION_ID = train-{env}-{recipe}`**

- `env` matches the recipe's grandparent dir (`lite.osworld`, `androidworld`, ...).
- `recipe` is the recipe's parent dir name (`qwen3_vl_2b`, `qwen3_5_2b`, ...).

| Recipe | SESSION_ID | Container name |
|---|---|---|
| `devs/exps/train/lite.osworld/qwen3_vl_2b/AGENTS.md` | `train-lite.osworld-qwen3_vl_2b` | `lite.slime-train-lite.osworld-qwen3_vl_2b` |
| `devs/exps/train/androidworld/qwen3_5_2b/AGENTS.md` | `train-androidworld-qwen3_5_2b` | `lite.slime-train-androidworld-qwen3_5_2b` |

Why this shape:
- **Self-describing**: `docker ps` immediately tells you which env + recipe owns the container.
- **Cleanup-scoped**: `SESSION_ID` is forwarded to the env-server, which tags every `lite-env-*` container it spawns with this id; `/cleanup <env>` and `DELETE /instances?session_id=...&env_id=...` both target leaks via `--filter name=lite-env-${SESSION_ID}-`. Concurrent recipes for different envs don't collide.
- **Idempotent**: re-launching the same recipe attaches to the existing container if alive (or fails fast if you forgot to remove a stale one), matching the resume-friendly intent of the recipe markdown files.

### ~~⚠️ One container = one model family (qwen3_vl XOR qwen3_5)~~ — OBSOLETE since slime v0.3.0

**This restriction is gone.** On `slimerl/slime:v0.3.0` (transformers 5.6.0, native megatron-bridge 0.5.0 with built-in Qwen3.5/GDN support), the qwen3.5 path no longer pip-mutates the container — the old `pip install -U "transformers>=5.0"` + `Megatron-Bridge-slime@qwen35 --force-reinstall` poisoning flow was deleted along with the standalone `run_*_qwen3_5.sh` scripts. One container now serves **both** families; pick the family per run via `MODEL_ID` (the merged `run_grpo.sh`/`run_reinforce.sh`/`run_sft.sh` auto-derive `MODEL_FAMILY`).

(Historical note: on v0.2.4 the two families needed separate containers because the qwen3_5 scripts force-reinstalled transformers/megatron-bridge. Old run logs under `logs/` describe that flow — do not follow them on v0.3.0.)

### Launch + init

From the repo root, **on the host**:

```bash
# 1. Probe env-server has the env_id you'll train against.
curl -sf -H "Authorization: Bearer $CUA_LITE_ENV_SERVER_TOKEN" \
   "$CUA_LITE_ENV_SERVER_URL/envs/$ENV_ID" | jq .
# → {"available": true,  "n_tasks": N, ...}   ready
# → {"available": false, "error": "..."}      env setup missing on server host
# → HTTP 404                                  env_id not registered on this server

# 2. No env-server reachable? Start one (its host must already have the env's deps installed).
uv run python scripts/serve_env.py --port 30100
# Use the host's external IP, NOT 'localhost'. This URL is forwarded into the Slime
# container in step 3; from inside the container's bridge network 'localhost' is its
# own loopback. For a remote env-server, point at the remote host's IP.
export CUA_LITE_ENV_SERVER_URL=http://$(hostname -I | awk '{print $1}'):30100
export CUA_LITE_ENV_SERVER_TOKEN=$(whoami)                  # any string; passthrough mode keys by sha256(token)[:6]

# 3. Launch slime container.
SESSION_ID=train-<env>-<recipe> CUDA_VISIBLE_DEVICES=<gpus> \
  bash scripts/train/slime/launch.sh
```

`scripts/train/slime/launch.sh` forwards `CUA_LITE_ENV_SERVER_URL` / `CUA_LITE_ENV_SERVER_TOKEN` / `SESSION_ID` into the container, then runs `scripts/train/slime/init.sh` automatically (re-points editable installs at the bind-mounted source). The container uses default docker bridge networking — set `CUA_LITE_ENV_SERVER_URL` to the env-server's external IP (`localhost` from inside the container would not reach the host). `run_grpo.sh` / `run_reinforce.sh` fail-fast via [`scripts/train/utils/preflight.sh`](/scripts/train/utils/preflight.sh) if the probe fails at training launch.

See [Env-server](/docs/envs.md#env-server) for details.

**Pass `CUDA_VISIBLE_DEVICES` explicitly — never run launch without it.** `scripts/train/slime/launch.sh` now refuses to start without that env var unless `CUA_LITE_SLIME_ALL_VISIBLE_GPUS=1` is set for a dedicated host. The set you pass should be **exactly** the GPUs the recipe needs — read the recipe's `Step 6 — GRPO` (and the SFT step) to size it. Concretely:

- 2-GPU recipes (default for the qwen3_vl 2B recipes here — 1 train + 1 rollout under `ASYNC=1`): pass 2 GPUs, e.g. `CUDA_VISIBLE_DEVICES=0,1`.
- Recipes that bump `NUM_TRAIN_GPUS=4` for faster SFT but stay 2-GPU for GRPO: pass 4 at launch (the larger of the two stages), then internally only the SFT step uses all 4 — nothing wasted by passing 4 once.
- Recipes that mention `NUM_TRAIN_GPUS=N` only at one step: pass exactly that count, even if other steps use fewer.

The recipe markdown is the source of truth for how many GPUs each step needs; the launch CLI must envelope all of them. If a recipe doesn't say, default to **2 GPUs**.

Inside the container, physical GPUs `<gpus>` always appear as `0..N-1` (docker `--gpus device=...` remaps), so recipes use `CUDA_VISIBLE_DEVICES=0,1` etc. inside `docker exec` regardless of which host GPUs you reserved.

The recipe's training commands then run inside that container via `docker exec lite.slime-${SESSION_ID} bash -c '...'`.

---

## Where artifacts go

**Everything reproducibility-relevant lives under `.exps/train/...`** — task lists, derived parquets, training ckpts, eval rollouts. Nothing under `/root/datasets/...` (container-internal, transient) or other ad-hoc paths.

| Stage | Artifact | Path (host = `${BASE_HOST}` / container = `${BASE_CTR}`) |
|---|---|---|
| Step 1.1 | Pinned task pool | `${BASE_CTR}/teacher_pool.parquet` |
| Step 1.2 | Teacher rollout logs (per-task trajectories) | `${BASE_CTR}/teacher_rollout/` |
| Step 2 | SFT-input trajectory parquet | `${BASE_CTR}/sft_trajectory.parquet` |
| Step 3 | SFT megatron ckpt | `${BASE_CTR}/sft/megatron/` |
| Step 3 | SFT HF ckpts | `${BASE_CTR}/sft/hf/iter_<n>/` |
| Step 4 | Base / SFT eval rollouts | `${BASE_CTR}/eval_base/<slug>/`, `${BASE_CTR}/eval_sft/iter_<n>/<slug>/` |
| Step 5 | GRPO prompt parquets (anchor / random / merged / eval) | `${BASE_CTR}/grpo_pool_*.parquet`, `${BASE_CTR}/eval.parquet` |
| Step 6 | GRPO megatron ckpt | `${BASE_CTR}/grpo/megatron/` |
| Step 6 | GRPO HF ckpts | `${BASE_CTR}/grpo/hf/iter_<n>/` |
| Step 7 | GRPO eval rollouts | `${BASE_CTR}/eval_grpo/iter_<n>/<slug>/` |

Where `${BASE_HOST} = .exps/train/<env>/<recipe>/<commit-ts>_<commit>/<run_id>/` (set once per § Working directory + paths in each recipe), and `${BASE_CTR} = /workspaces/cua-lite/${BASE_HOST}` is the same path inside the slime container via the bind mount.

Concretely, every recipe must pass:
- `--log-root ${BASE_HOST}/<stage>/...` to host-side `scripts/rollout.py` invocations
- `-o ${BASE_HOST}/<file>.parquet` (host) or `-o ${BASE_CTR}/<file>.parquet` (container) for any data-export step
- `SAVE_DIR=${BASE_CTR}/<stage>/megatron`, `SAVE_HF_DIR=${BASE_CTR}/<stage>/hf/iter_{rollout_id}`, `PROMPT_DATA=${BASE_CTR}/<file>.parquet` to slime training scripts

Why under `.exps/` (dot-prefixed)? `.exps/` is gitignored (large blobs / 100s of GB of trajectories + ckpts). The committed counterpart is `exps/` (no dot) — README + recipe markdown + per-commit snapshots only.

> The legacy `.ckpts/` and ad-hoc `.logs/` paths from earlier runs (e.g. the `.ckpts/20260425_1149/...` SFT ckpts) **predate this convention**; new recipes follow `.exps/train/...`. Re-archive legacy ckpts under `.exps/train/...` only if you actively re-run them.

---

## Conventions every recipe must follow

1. **Working directory on host**: the repo root. Inside the container the same tree appears at `/workspaces/cua-lite`.
2. **GPU reservation**: respect the user's allowed GPU range. Never use any GPU outside it. Hold via `/watchdog` between job phases — see [GPU reservation discipline](#gpu-reservation-discipline) above for the full rules.
3. **Commit pinning**: every recipe is reproducible under the commit it was written against. If you re-run at a later commit and numbers drift, record both in `<env>/<recipe>/CHANGELOG` and start a new `<run_id>`.
4. **Distinct save paths per recipe + per stage**: avoid cross-run megatron-resume mishaps. The path scheme above (`/<env>/<recipe>/<commit-ts>_<commit>/<run_id>/<stage>/`) gives this for free.
5. **Failure recovery**: on `ActorUnavailableError` / docker / sglang crash, prefer resuming from the most recent megatron ckpt under the same `(commit, run_id, stage)`. If the underlying state is broken, bump `<run_id>` (treat the next run as a separate campaign) rather than mixing partial results.
6. **DUMP=1 caution**: slime's `DUMP=1` writes per-rollout debug tensors that have crashed disk before (5.3 TB filled `/tmp` → OOM). Default OFF. Enable only on a small smoke run where you need byte-exact replay.
7. **🛑 SFT-must-beat-base gate before GRPO** (hard requirement): every recipe whose pipeline runs SFT → GRPO must, between those stages, run a deterministic eval of base + the latest SFT ckpt and verify `mean_episode_return(sft) > mean_episode_return(base)` on the recipe's eval subset before any GRPO step starts. If the gate fails, **STOP the pipeline** — do not start GRPO from a ckpt that's worse than base. GRPO can usually claw back to base from a regressed SFT, but the lift you observe will be GRPO climbing out of a hole the SFT dug, not GRPO actually exceeding base. Recovery options when the gate fails: (a) re-run SFT with `NUM_EPOCH=5` if the recipe trained at `NUM_EPOCH=2` and undertrained; (b) inspect `sft_trajectory.parquet` row count + per-turn formatting against the teacher rollout source for a data-pipeline regression; (c) bypass SFT entirely and start GRPO from the base 2B (set `HF_CKPT=` to the base model). The gate is recipe-owned — each recipe README carries its own threshold (G1 in the acceptance gates table) and the eval commands the gate uses.

---

## Snapshot / progress log

Path: `<env>/<recipe>/logs/<commit-ts>_<commit>/<run_id>.md`.

This file is the **running progress log**, not just a wrap-up artifact. **Update it after every recipe step**: append rows to `Stages run` / `Eval results`, refresh `Last updated`, commit. The same `(commit, run_id)` markdown lives across the whole campaign — one file, edited multiple times. That way an interrupted campaign's state is visible in git, not just stuck in the agent's head.

The skeleton below is schema-only — generic across recipes. **Each recipe's README owns the concrete copy-paste template** (with this recipe's actual stage list, eval rows, etc.); start from that recipe's `## Snapshot template` section. Required fields only; omit a field rather than restate a default.

```markdown
# <env> · <recipe> @ <commit-ts>_<commit> · <run_id>

- **Recipe**: `<env>/<recipe>/AGENTS.md`
- **Commit**: `<short sha>` — `<subject>`
- **Host / GPUs**: `<hostname>` / `<e.g. 0-3>`
- **Container**: `lite.slime-train-<env>-<recipe>` (started `<YYYY-MM-DD HH:MM TZ>`)
- **wandb** — one URL per training stage that produced canonical numbers (recipe owns the stage keys; **skip mid-run failed-restart URLs**; if a stage was relaunched after a crash, paste only the URL of the run whose ckpts ended up in `Eval results`):
  - `<stage>`: `<url or "n/a">`
  - `<stage>`: `<url or "n/a">`
- **Artifacts**: `.exps/train/<env>/<recipe>/<commit-ts>_<commit>/<run_id>/`
- **Started**: `<YYYY-MM-DD HH:MM TZ>`
- **Last updated**: `<YYYY-MM-DD HH:MM TZ>` ← bump on every edit
- **Notes**: anything non-default — non-default lr / batch-size, mid-run crashes + resumes, recipe deviations

## Stages run

| Stage | Status | Output |
|---|---|---|
| <stage> | `not started` / `in_progress` / `done` / `failed (<reason>)` | <stage-specific output spec — see recipe README> |

Update each row as soon as the corresponding stage flips state. `failed` rows must include a one-line reason (e.g. `failed (ActorUnavailableError at iter_22; resumed as run_id=run_1)`).

## Eval results

| Ckpt | Eval set | Finished | Mean episode return | Δ vs base |
|---|---|---|---|---|
| base `<model-id>` | `<env>` eval | <nf>/<nt> | <mer> | — |
| <stage> iter_<n>  | `<env>` eval | <nf>/<nt> | <mer> | +/- <pp>pp |

Add rows incrementally as each eval finishes — **don't wait for the campaign to finish**. Sort within stage by iter ascending. Mark intentionally-skipped iters `⚠️ not eval'd`.

## Highlights

- short interpretive bullets — call out partial stages, surprising deltas, failure modes, key recipe knobs that mattered. Add bullets as observations come in; the wrap-up is just removing TODO markers.
```

Pull numbers from `.exps/train/<env>/<recipe>/<commit-ts>_<commit>/<run_id>/eval_<stage>/<slug>/summary.json`:

```bash
python3 -c "
import json, glob
for f in sorted(glob.glob('.exps/train/<env>/<recipe>/<commit-ts>_<commit>/<run_id>/eval_*/*/summary.json')):
    d = json.load(open(f))['stats']
    parts = f.split('/')
    print(f'  {parts[-3]:20s} {parts[-2]:42s}  {d[\"num_valid\"]}/{d[\"num_tasks\"]}  {d[\"mean_episode_return\"]:.4f}')
"
```

---

## CHANGELOG conventions — `<env>/<recipe>/CHANGELOG`

CHANGELOG is **commit-scoped**, not run-scoped — it tracks code-level deltas that change training numbers, regardless of how many recipe runs share each commit. Newest on top.

Record only changes that **plausibly affect this recipe's training numbers**:
- `lite/train/**` — rollout / GRPO / REINFORCE / loss code
- `lite/agents/**` — adapter, action space, prompt templates (eval-impactful)
- `lite/gym/envs/<env>/**` — env wrapper / reward shaping
- `scripts/train/**` — launch wrappers, default args
- `scripts/configs/<agent>/{default,compact}/<env>.yaml` — sampling, history sizes, env_kwargs
- slime / megatron / sglang version bumps

Skip pure docs / unrelated env code / cosmetic refactors.

```markdown
## <new_short> ← <prev_short>  (YYYY-MM-DD)

- one-liner about what changed and why it matters here
```

The very first entry has no predecessor; just write `## <new_short>  (YYYY-MM-DD)` and a one-liner noting it's the initial snapshot.

---

## Agent workflow checklist

For an agent given **(this README) + a recipe path** (e.g. `devs/exps/train/lite.osworld/qwen3_vl_2b/AGENTS.md`):

1. **Pre-flight**
   - `cd` to repo root.
   - **Commit first**: any code change you intend to train against must
     already be committed — `git diff` and `git diff --staged` both
     empty (untracked artifact dirs are fine). The artifact tree is
     `.exps/train/<env>/<recipe>/<commit-ts>_<commit>/<run_id>/<stage>/`,
     and `<commit-ts>_<commit>` is captured at kickoff. Mid-campaign
     code edits drift the actual training/eval distribution away from
     the recorded hash; resume hooks can then carry forward stale
     checkpoints produced by the wrong code. Either commit + restart,
     or accept the kickoff hash as approximate and document the drift
     in the snapshot's `Notes`. See the `2026-04-27T23-21_828ea14` grounding
     eval snapshots
     (`devs/exps/eval/{osworld_g,screenspot_pro}/logs/2026-04-27T23-21_828ea14/run_0.md`)
     for the example we want to stop reproducing.
   - Resolve `<commit-ts>_<commit>` = `git log -1 --format=%cs HEAD` + `_` + `git rev-parse --short HEAD`. (Recipe `BASE_HOST` does this for you.)
   - Pick `<run_id>` and `export TRAIN_RUN_ID=<run_id>` as a fixed string literal. Schema: `run_<N>` (e.g. `run_0` for the first campaign at this commit, `run_1` for the next), or `run_<N>_<label>` to name the campaign's intent. See [Run id contract](#run-id-contract-train_run_id).
   - Confirm allowed GPU range with the user (ask if not given). Hold every GPU in the range with `/watchdog`.
   - Run the [leak survey](#leak-survey--cleanup-between-every-stage); `/cleanup` if non-empty (mind the SESSION_ID safety callout there).
   - **Initialize the progress log** at `<env>/<recipe>/logs/<commit-ts>_<commit>/<run_id>.md` from the snapshot template. Fill the header (recipe / commit / host+GPUs / container / artifact path / Started timestamp). `git add` it; commit at the next stage boundary.

2. **Container kickoff**
   - `SESSION_ID=train-<env>-<recipe>` — derive from the recipe path.
   - Launch container + run init (commands in [Container setup](#container-setup-one-time-per-recipe) above).
   - Confirm container is up: `docker ps --filter name=lite.slime-${SESSION_ID}`.
   - In the progress log, refresh `Last updated`; commit.

3. **Stage execution** (follow recipe step-by-step). Per stage:
   - Mark its row in the progress log `in_progress`, refresh `Last updated`, commit.
   - Pre-pend artifact paths with `.exps/train/<env>/<recipe>/<commit-ts>_<commit>/<run_id>/<stage>/` per [Where artifacts go](#where-artifacts-go).
   - For long stages (training > 1h, multi-task rollouts): poll every 15-25 min — distinguish "still working" from "stuck" via [Reading utilization](#reading-utilization).
   - **Graceful-exit / hung-rollout recipe** (training-mode counterpart to eval's Phase 2):
     1. Diagnose: `ActorUnavailableError` in slime stdout, OR sglang `Gracefully exiting... Remaining number of requests N` looping with GPU mem dropped, OR `data.py:219 rollout N` line stale > 30 min, OR `subprocess.CalledProcessError: ... died with <Signals.SIGKILL: 9>` (external SIGKILL, often Linux OOM killer — re-check [Reading utilization](#reading-utilization) point 5 first).
     2. Kill the slime job: `docker exec lite.slime-${SESSION_ID} pkill -9 -f "ray::"`; if that doesn't free things, `docker exec ... pkill -9 -f train_async`.
     3. Run the leak survey; `/cleanup` only **your** `SESSION_ID`'s stragglers — see [Leak survey + cleanup](#leak-survey--cleanup-between-every-stage) for the safety rules. Cleanup may need 2 passes for racy docker rm; verify count == 0 before relaunching.
     4. **Do NOT `/watchdog hold` between for-loop attempts.** A host-side retry loop (e.g. `for attempt in 1 2 3; do ... done`) advances to the next attempt within seconds, and a watchdog launched in that gap will saturate VRAM and OOM the next attempt's sglang at startup. Watchdog only between **stages**, never inside a stage's retry loop.
     5. If the underlying cause was VRAM pressure, apply [OOM escalation](#oom-escalation) before re-launching.
     6. Re-launch the SAME stage at the SAME `(commit, run_id)`. Resume hooks (megatron `--load`, parquet existence, `iter_*` skips) pick up the rest. Note the crash + resume one-liner in the progress log's `Notes` field.
   - Run the leak survey + `/cleanup` between stages (scoped to your SESSION_ID).
   - **When the stage finishes** (or fails terminally): flip its row to `done <ckpt>` / `failed (<reason>)`, append any new eval rows (from the just-finished stage's `summary.json`), refresh `Last updated`, commit. Don't wait for the whole campaign — every stage boundary is a commit point.

4. **Wrap-up**
   - Final `/cleanup <env>` per [Leak survey + cleanup](#leak-survey--cleanup-between-every-stage) — for `osworld` campaigns this sweeps the VM-in-Docker containers (see [eval's full cleanup table](/devs/exps/eval/AGENTS.md#leak-survey--cleanup-between-every-run) for the unset-`SESSION_ID` footgun). `lite.osworld` cleanup is session-scoped sandbox containers only.
   - Verify allowed GPUs are fully held by watchdogs.
   - Re-read the progress log; fill in any pending `Highlights` bullets (delta vs base, length drift, mixed-group anomalies). Most number rows should already be in place from per-stage commits — the wrap-up is interpretation, not data extraction. Final commit.
   - If this is the first run at this commit and the diff vs the previous catalogued commit is non-trivial, append a `CHANGELOG` entry.

---

## Cross-references

- [`docs/slime.md`](/docs/slime.md) — Slime container build, launch, init
- [`docs/sft.md`](/docs/sft.md) — SFT recipes (ScaleCUA, Lite.OSWorld distill, Android-World distill)
- [`docs/grpo.md`](/docs/grpo.md) — GRPO knobs and tunables
- [`docs/examples/reinforce.md`](/docs/examples/reinforce.md) — REINFORCE / filtered-BC
- [`devs/exps/eval/AGENTS.md`](/devs/exps/eval/AGENTS.md) — sibling eval campaign workflow + GPU discipline + leak survey
