# Android-World: GRPO from base + 8B-Teacher Distill SFT → GRPO (MAI-UI-2B)

> **Read [`devs/exps/train/AGENTS.md`](/devs/exps/train/AGENTS.md) first** for layout, `TRAIN_RUN_ID` contract, Slime v0.3.0 container/`SESSION_ID` rules, GPU discipline, snapshot template, and the agent workflow checklist.
>
> **Env-server prereq.** An env-server must be reachable and expose `androidworld` before launch — see [`devs/exps/train/AGENTS.md#env-server-prerequisite`](/devs/exps/train/AGENTS.md#env-server-prerequisite).

End-to-end recipes for **MAI-UI-2B** (Alibaba Tongyi MAI's GUI-specialized 2B model) on androidworld. Unlike the qwen3_vl_2b / qwen3_5_2b recipes that distill from a separate 8B teacher because the 2B base is weak (~24% on this env), MAI-UI-2B is **already a strong GUI specialist** (base ≈ 41-42% on the 86-task subset) — so this recipe asks two angles, not one.

Two variants, picked at `TRAIN_RUN_ID` time:

1. **`run_0` — pure GRPO from base.** No SFT step. Tests whether RL alone on this strong base can match or beat the SFT-then-GRPO baseline that qwen3_vl_2b established.
2. **`run_1_sft_grpo` — SFT (2 epochs from 8B-teacher trajectories) then GRPO.** Reuses the **same teacher rollout** committed by [`../qwen3_vl_2b/AGENTS.md#step-1`](/devs/exps/train/androidworld/qwen3_vl_2b/AGENTS.md#step-1--teacher-rollout-8b-with-rollout-config) — never re-roll out the 8B teacher just for mai_ui. Tests whether SFT distillation on top of an already-GUI-specialist base still pays off.

| TRAIN_RUN_ID | Steps |
|---|---|
| `run_0` | Step 5 → Step 6 → Step 7 (no SFT — skip Steps 1-4) |
| `run_1_sft_grpo` | Step 1 → Step 2 → Step 3 → Step 4 → Step 5 → Step 6 → Step 7 |

Steps 5/6/7 are shared between variants; only `HF_CKPT` in Step 6 differs.

> **Why use the Qwen3-VL family configs?** MAI-UI-2B is a Qwen3-VL architecture variant, so its training path follows the Qwen3-VL model family. With Slime v0.3.0 the shared image selects the family from `MODEL_ID`; do not mutate the container environment for a specific family.

## ⚠️ Working directory + paths

> Host commands run from the repo root; container commands run inside the Slime container at `/workspaces/cua-lite`. See [`docs/slime.md`](/docs/slime.md) and [`devs/exps/train/AGENTS.md`](/devs/exps/train/AGENTS.md) for setup.

Per the [training experiments convention](/devs/exps/train/AGENTS.md#layout):

```bash
export TRAIN_RUN_ID="run_0"                                         # or "run_1_sft_grpo"
COMMIT=$(git rev-parse --short HEAD)
COMMIT_TS=$(git log -1 --date=format:'%Y-%m-%dT%H-%M' --format=%cd HEAD)
BASE_HOST=.exps/train/androidworld/mai_ui_2b/${COMMIT_TS}_${COMMIT}/${TRAIN_RUN_ID}
BASE_CTR=/workspaces/cua-lite/${BASE_HOST}                          # same path inside the slime container

# Slime container session id — shared with qwen3_vl_2b (same model family)
export SESSION_ID=train-androidworld-qwen3_vl_2b
```

`MODEL_ID=Tongyi-MAI/MAI-UI-2B` is mapped to Qwen3-VL-2B megatron args via `MODEL_ARGS_MAP` in [`scripts/train/utils/models.sh`](/scripts/train/utils/models.sh) and to the **`mai_ui` adapter** (mobile_use action schema, distinct from qwen3_vl's computer_use) — see [`scripts/configs/mai_ui/compact/androidworld.yaml`](/scripts/configs/mai_ui/compact/androidworld.yaml).

The HF model itself must be **pre-downloaded** (slime won't auto-download — it asserts `args.load` exists):

```bash
# inside the container, once
docker exec -w /workspaces/cua-lite lite.slime-${SESSION_ID} \
  huggingface-cli download Tongyi-MAI/MAI-UI-2B --local-dir /root/models/Tongyi-MAI/MAI-UI-2B
```

Then pass `HF_CKPT=/root/models/Tongyi-MAI/MAI-UI-2B` to SFT/GRPO as the warm start.

---

## Step 1 — Teacher rollout (REUSE companion qwen3_vl_2b run) [run_1 only]

> **Skip this step for `run_0`** (pure GRPO from base — no SFT, no teacher).

**Do not run a fresh teacher rollout in this recipe.** Run [`../qwen3_vl_2b/AGENTS.md#step-1`](/devs/exps/train/androidworld/qwen3_vl_2b/AGENTS.md#step-1--teacher-rollout-8b-with-rollout-config) **first** (in the qwen3-VL container), then point this recipe at its output.

```bash
# Default: same (commit-ts, commit, TRAIN_RUN_ID) — `qwen3_vl_2b/${BASE}/teacher_rollout`.
# Override with TEACHER_ROLLOUT_DIR (absolute container path) to reuse a
# rollout from a different commit / run_id — cheaper than re-rolling the
# 8B teacher (1-3h) when neither the env nor the qwen3_vl_2b recipe has
# changed since that rollout. Use sparingly: drifting too far from the
# current commit forfeits the reproducibility contract.
TEACHER_ROLLOUT="${TEACHER_ROLLOUT_DIR:-/workspaces/cua-lite/.exps/train/androidworld/qwen3_vl_2b/${COMMIT_TS}_${COMMIT}/run_0/teacher_rollout}"

# Sanity check on host. Bind-mount makes /workspaces/cua-lite/... and the
# repo-relative path the same physical dir; strip the prefix to get the
# host-side view.
TEACHER_ROLLOUT_HOST="${TEACHER_ROLLOUT#/workspaces/cua-lite/}"
echo "Teacher rollout count: $(find ${TEACHER_ROLLOUT_HOST} -name summary.json 2>/dev/null | wc -l) / 344"
```

If the count is < 95% (~327/344), finish the companion's Step 1 first (or point `TEACHER_ROLLOUT_DIR` at a more complete rollout). The companion uses Qwen3-VL-8B-Instruct as teacher (~55% success rate → ≈190 success-filtered trajectories from 344).

---

## Step 2 — Export SFT parquet (success-only, mai_ui format) [run_1 only]

Use `mai_ui/compact/androidworld.yaml` for export — this **converts** the teacher's qwen3-VL trajectory format (computer_use) into mai_ui's (mobile_use) at parquet write time. The output parquet is mai_ui-native; no further conversion at SFT time.

```bash
docker exec -w /workspaces/cua-lite lite.slime-${SESSION_ID} \
  python -m lite.train.export.export_sft \
    --config /workspaces/cua-lite/scripts/configs/mai_ui/compact/androidworld.yaml \
    --model-id Tongyi-MAI/MAI-UI-2B \
    --data-paths ${TEACHER_ROLLOUT} \
    --image-root /workspaces/cua-lite \
    --filter "lambda m: (m.others.get('episode_return') or 0) >= 1.0" \
    -o ${BASE_CTR}/sft_trajectory.parquet
```

`${TEACHER_ROLLOUT}` is the companion's directory (set in Step 1). The parquet is written to **this** recipe's `BASE_CTR`, keeping mai_ui artifacts isolated. Expected: ~190-200 rows depending on the 8B teacher's exact success rate.

---

## Step 3 — SFT (8B → MAI-UI-2B distill) [run_1 only]

```bash
# CUDA_VISIBLE_DEVICES + NUM_TRAIN_GPUS illustrative — match your allowed GPU range.
# Default is TP=1 (pure DP + ZeRO-1); 2→4 GPUs adds DP and cuts SFT wall-clock roughly in half.
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=2 \
  MODEL_ID=Tongyi-MAI/MAI-UI-2B \
  HF_CKPT=/root/models/Tongyi-MAI/MAI-UI-2B \
  SAVE=1 NO_SAVE_OPTIM=1 \
  NUM_EPOCH=2 \
  PROMPT_DATA=${BASE_CTR}/sft_trajectory.parquet \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/mai_ui/compact/androidworld.yaml \
  SAVE_DIR=${BASE_CTR}/sft/megatron \
  SAVE_HF_DIR=${BASE_CTR}/sft/hf/iter_{rollout_id} \
  bash /workspaces/cua-lite/scripts/train/run_sft.sh
```

`NUM_EPOCH=2` — at ~194 trajectories with the default global batch size → ~48 iters/epoch → saves `iter_47` (end epoch 1) + `iter_95` (end epoch 2). Aligned across `<env>/<recipe>` for consistent wall-clock. `lr=2e-6` (`run_sft.sh` default) matches the captured distill — re-tune only if SFT eval is flat.

---

## Step 4 — Eval base + SFT (sanity) [run_1 only; G1 gate]

Verify the 8B → MAI-UI-2B SFT moved the student above its already-strong base. Eval is **resume-safe** — re-running the same `--log-root` skips finished tasks. Loop 3× per ckpt to guarantee full 86-task coverage. Runs **inside the slime container**.

```bash
SLUG=Tongyi-MAI_MAI-UI-2B

# 4.1 base — 3 retries (same base for run_0 as well; eval once, reuse the
# numbers when both variants run at the same commit)
for attempt in 1 2 3; do
  echo "=== eval base attempt ${attempt}/3 ==="
  docker exec -w /workspaces/cua-lite -e CUDA_VISIBLE_DEVICES=0,1 lite.slime-${SESSION_ID} \
    python scripts/rollout.py \
      --model-id Tongyi-MAI/MAI-UI-2B --env-id androidworld \
      --splits eval --concurrency 16 \
      --filter "lambda m: m.others.get('complexity') <= 2.0" \
      --sampling-kwargs '{"temperature": 0, "top_p": 1}' \
      --config-path scripts/configs/mai_ui/compact/androidworld.yaml \
      --log-root ${BASE_CTR}/eval_base/${SLUG}
done

# 4.2 SFT — 3 retries per ckpt; replace iter_N with each ckpt you want to score
for attempt in 1 2 3; do
  echo "=== eval sft/iter_N attempt ${attempt}/3 ==="
  docker exec -w /workspaces/cua-lite -e CUDA_VISIBLE_DEVICES=0,1 lite.slime-${SESSION_ID} \
    python scripts/rollout.py \
      --model-id Tongyi-MAI/MAI-UI-2B --env-id androidworld \
      --model-path ${BASE_CTR}/sft/hf/iter_N \
      --splits eval --concurrency 16 \
      --filter "lambda m: m.others.get('complexity') <= 2.0" \
      --sampling-kwargs '{"temperature": 0, "top_p": 1}' \
      --config-path scripts/configs/mai_ui/compact/androidworld.yaml \
      --log-root ${BASE_CTR}/eval_sft/iter_N/${SLUG}
done
```

Pick the `iter_N` with the highest `mean_episode_return`; that becomes the GRPO warm start for run_1.

---

## Step 5 — Export GRPO data [both variants]

Just the train + eval parquets — no anchor / random rebalancing needed (androidworld only has 86 tasks; rollout_batch_size=16 resamples the pool fast enough).

```bash
docker exec -w /workspaces/cua-lite lite.slime-${SESSION_ID} \
  python -m lite.train.export.export_tasks --env-id androidworld --split train \
    --filter "lambda m: m.others.get('complexity') <= 2.0" \
    -o ${BASE_CTR}/train.parquet

docker exec -w /workspaces/cua-lite lite.slime-${SESSION_ID} \
  python -m lite.train.export.export_tasks --env-id androidworld --split eval \
    --filter "lambda m: m.others.get('complexity') <= 2.0" \
    -o ${BASE_CTR}/eval.parquet
```

Expected: 86 train tasks, 86 eval tasks (after the complexity filter). Deterministic — same filter on same env_id produces the same pool across commits / machines.

---

## Step 6 — GRPO [both variants — `HF_CKPT` differs]

`HF_CKPT` is the only thing that changes between variants:

| Variant | `HF_CKPT` |
|---|---|
| **run_0** (pure GRPO from base) | `/root/models/Tongyi-MAI/MAI-UI-2B` |
| **run_1_sft_grpo** | `${BASE_CTR}/sft/hf/iter_N` (the best from Step 4.2) |

ASYNC mode runs rollout (1 GPU, sglang) and train (1 GPU, megatron) concurrently.

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=1 NUM_ROLLOUT_GPUS=1 ASYNC=1 \
  MODEL_ID=Tongyi-MAI/MAI-UI-2B \
  HF_CKPT=<see table above per variant> \
  SAVE_DIR=${BASE_CTR}/grpo/megatron \
  SAVE_HF_DIR=${BASE_CTR}/grpo/hf/iter_{rollout_id} \
  ENV_ID=androidworld \
  PROMPT_DATA=${BASE_CTR}/train.parquet \
  EVAL_PROMPT_DATA=${BASE_CTR}/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/mai_ui/compact/androidworld.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh
```

Notes:
- GRPO config uses `full_history_size=1`, matching the SFT / eval protocol — student sees the same context shape at SFT, RL, and eval time.
- No per-split env_kwargs: train and eval rollouts share the `env_kwargs` in `mai_ui/compact/androidworld.yaml` (`loop_detect: 5`, `extra_tools`; no `max_steps` override → native `int(10 * task.complexity)`). `resolution: [720, 1600]` lives under `agent_kwargs` in the same file, not `env_kwargs`. A train-only override goes on the `export_tasks` call that builds `train.parquet` (`--env-kwargs '{...}'` → per-row `metadata.env_kwargs`, which deep-merges over the yaml).
- Saves a HF ckpt every `--save-interval` step (5 default → `iter_4 / iter_9 / iter_14 / iter_19` at 20 total iters).
- Each iter ≈ 7 min wall-clock (rollout 400-500s + train ~30s); 20 iters ≈ 2.3h. **With `ActorUnavailableError` retries (see [Stability notes](#stability-notes-ray-actorunavailableerror)), budget 3-6h.**
- **For run_1, the best ckpt is iter_4, NOT iter_19** — GRPO regresses past sem 28 (see [acceptance gates](#acceptance-gates) and the [captured numbers](#reference-captured-numbers-2026-05-01t02-50_eb9bdce4) below).

If the wrapper crashes with `ActorUnavailableError`, re-run the same command — slime resumes from the latest saved megatron iter automatically. A 6-attempt retry loop wrapped around the `bash run_grpo.sh` invocation suffices.

---

## Step 7 — Eval GRPO ckpts (≥ 3 retries per ckpt)

Runs **inside the slime container**.

```bash
SLUG=Tongyi-MAI_MAI-UI-2B
# Replace iter_M with each saved GRPO iter (4, 9, 14, 19, ...) to track the curve.
# For run_1: ALWAYS eval ALL four iters — best is NOT necessarily iter_19.
for attempt in 1 2 3; do
  echo "=== eval grpo/iter_M attempt ${attempt}/3 ==="
  docker exec -w /workspaces/cua-lite -e CUDA_VISIBLE_DEVICES=0,1 lite.slime-${SESSION_ID} \
    python scripts/rollout.py \
      --model-id Tongyi-MAI/MAI-UI-2B --env-id androidworld \
      --model-path ${BASE_CTR}/grpo/hf/iter_M \
      --splits eval --concurrency 16 \
      --filter "lambda m: m.others.get('complexity') <= 2.0" \
      --sampling-kwargs '{"temperature": 0, "top_p": 1}' \
      --config-path scripts/configs/mai_ui/compact/androidworld.yaml \
      --log-root ${BASE_CTR}/eval_grpo/iter_M/${SLUG}
done
```

In-training eval (the `eval N` lines in slime stdout / wandb) tracks the same metric automatically — re-run host-side eval only if you need byte-exact trajectories or want to evaluate ckpts past the auto-eval cadence.

---

## Acceptance gates

| Gate | What | Target | Hard stop? | Captured reference |
|---|---|---|---|---|
| **G1** | (**run_1 only**) SFT eval > base eval | **≥ +10pp on the 86-task eval split (T=0)** | **🛑 yes — pipeline must stop here if it fails** | run_1 iter_95: **+15.1pp** (41.86% → 56.98%) ✓ |
| G2 | GRPO mixed-group ratio | ≥ 50% (run_0) / ≥ 40% (run_1) | no | run_1 typically ~43% — borderline (see Highlights) |
| G3 | GRPO eval > start eval | ≥ +5pp at peak iter | no | run_0: +12.2pp at iter_19; run_1 SFT+GRPO: +7pp at iter_4 |
| G4 | Response-len mean stable across iters | within ±15% of start mean | no | run_0: stable; run_1: monotone-increasing past iter_4 (length-hacking signal — pair with G2 monitoring) |

**G1 enforcement (run_1 only)** — Step 4 is a hard gate. Don't start Step 6 until `sft_iter_N ≥ base + 0.10`.

If G1 underperforms on run_1 (SFT lift < 10pp): MAI-UI-2B base is already strong → SFT data may be too narrow. Inspect `sft_trajectory.parquet` against `${TEACHER_ROLLOUT}` for format-conversion regressions in the qwen3-VL → mai_ui (computer_use → mobile_use) translation. Other options: re-run SFT at `NUM_EPOCH=5`, or fall back to `run_0` (skip SFT entirely).

If G3 underperforms on run_1 past iter_4: that's the expected regression — **stop at iter_4** and treat it as best. For run_0, no such regression — iter_19 is typically the best.

---

## Snapshot template

Concrete copy-paste body for `logs/<commit-ts>_<commit>/<run_id>.md` (this recipe's stages: optional `teacher_rollout` reuse → optional `sft` → `grpo`). Schema rules and the "skip mid-run failed-restart URLs" semantics are in [`devs/exps/train/AGENTS.md`](/devs/exps/train/AGENTS.md#snapshot--progress-log).

```markdown
# androidworld · mai_ui_2b @ <commit-ts>_<commit> · <run_id>

- **Recipe**: `devs/exps/train/androidworld/mai_ui_2b/AGENTS.md`
- **Variant**: `run_0` (pure GRPO from base) | `run_1_sft_grpo` (SFT then GRPO)
- **Commit**: `<short sha>` — `<subject>`
- **Host / GPUs**: `<hostname>` / `<e.g. 4,5>`
- **Container**: `lite.slime-train-androidworld-qwen3_vl_2b` (shared with qwen3_vl family) (started <timestamp>)
- **TEACHER_ROLLOUT_DIR** (run_1 only): `<override path or "default — this commit's companion qwen3_vl_2b/run_0">`
- **wandb** (skip mid-run failed-restart URLs):
  - `sft` (run_1 only): `<url or "n/a">`
  - `grpo`: `<url or "n/a">`
- **Artifacts**: `.exps/train/androidworld/mai_ui_2b/<commit-ts>_<commit>/<run_id>/`
- **Started**: `<YYYY-MM-DD HH:MM TZ>`
- **Last updated**: `<YYYY-MM-DD HH:MM TZ>` ← bump on every edit
- **Notes**: anything non-default — TEACHER_ROLLOUT_DIR override rationale, non-default lr, mid-run crashes + resumes, recipe deviations

## Stages run

| Stage | Status | Output |
|---|---|---|
| teacher_rollout | (run_1 only) `reused from <companion qwen3_vl_2b path>` | (no rollout in this recipe) |
| sft             | (run_1 only) `not started` / `in_progress` / `done iter_<best>` / `failed` | `best ckpt eval <acc>% (vs base <acc>%)` once Step 4 has the numbers |
| grpo            | `not started` / `in_progress at iter_<n>` / `done iter_<best>` / `failed` | `best ckpt eval <acc>% (vs start <acc>%)` once Step 7 has the numbers |

## Eval results

| Ckpt | Eval set | Finished | Mean episode return | Δ vs base |
|---|---|---|---|---|
| base `Tongyi-MAI/MAI-UI-2B` | `androidworld` eval | <nf>/<nt> | <mer> | — |
| sft iter_<n> (run_1)        | `androidworld` eval | <nf>/<nt> | <mer> | +/- <pp>pp |
| grpo iter_<n>               | `androidworld` eval | <nf>/<nt> | <mer> | +/- <pp>pp |

For **run_1**: list ALL four GRPO iters (4/9/14/19) explicitly — best is NOT necessarily iter_19. Mark intentionally-skipped iters `⚠️ not eval'd`.

## Highlights

- short interpretive bullets — variant chosen + rationale, best ckpt found (call out if NOT iter_19), length-hacking incidents, mixed-group ratio drift, ActorUnavailableError retries + recovery iter.
```

---

## Stability notes: Ray ActorUnavailableError

Long GRPO runs on this recipe (especially run_1's 4h+ trajectory across multiple attempts) repeatedly hit `ray.exceptions.ActorUnavailableError: ... Socket closed rpc_code: 14` during the trainer's `train_wait` phase. Pattern across all observed crashes:

1. RolloutManager goes silent (no log lines) for **5 min** before the crash
2. Last RM log line is always an adb command (`shell input tap`, `shell settings`, etc.)
3. raylet/gcs logs show **no actor death** event — the actor process is alive
4. `dmesg` shows no OOM kill

**Diagnosis** (per the captured 2026-05-01 run's watchdog data):

- adb daemon hangs sometimes; with 32 parallel rollouts sharing one daemon, multiple worker threads block simultaneously
- RolloutManager's gRPC server thread can't get GIL → ray client at trainer side observes silence
- ray's **client-side gRPC channel idle timeout** (default ≈ 5 min) closes the channel → trainer's `ray.get` raises ActorUnavailableError

**Mitigations applied** (these affect **future** pipeline launches; not the in-flight one when the patch lands):

1. [`scripts/train/utils/ray.sh`](/scripts/train/utils/ray.sh) — exports `RAY_grpc_client_idle_timeout_ms=1800000` (30 min) and friends before `ray start`. **This is the patch that meaningfully helped** — captured run_1 attempt 5 (with this active) survived 4h21m and reached iter_19, vs attempts 1-4 (without it) crashing in 1-2h.
2. [`lite/gym/envs/androidworld/main.py`](/lite/gym/envs/androidworld/main.py) — monkey-patches `android_env.AdbController.execute_command` to cap timeout at 30s (was 120s default) so the worst-case `try + restart_server + retry` chain compresses from ~16 min (120s × 8 calls) to ~4 min (30s × 8). Marginal — the ray.sh patch is the load-bearing fix.

If you still hit ActorUnavailableError, wrap Step 6's `bash run_grpo.sh` invocation in a 6-attempt retry loop so transient crashes auto-resume from the latest GRPO ckpt.

---

## Reference: captured numbers (2026-05-01T02-50_eb9bdce4)

For comparison only — from a prior MAI-UI-2B campaign at commit `eb9bdce4`. Both variants in one campaign, reusing the 2026-04-30T01-21_e93ba3f8 qwen3_vl_2b teacher_rollout (set via `TEACHER_ROLLOUT_DIR`).

| Recipe | Final ckpt | Eval (in-training, T=0, 86-task) | Δ vs base 41.86% |
|---|---|---|---|
| base `Tongyi-MAI/MAI-UI-2B` | — | **41.86%** (mean of 2 indep measurements) | — |
| **run_0** (pure GRPO 19 iter) | iter_19 | 51.74% (in-train) / **54.07%** (indep, mean of 2) | **+12.2pp** |
| **run_1** SFT iter_95 alone | iter_95 | 56.98% (eval_0 of GRPO attempt 1) | **+15.1pp** |
| **run_1** SFT+GRPO peak | iter_4 | **63.96%** (mean of 2 in-train measurements) | **+22.1pp** |
| **run_1** SFT+GRPO final | iter_19 | (Step 7 deterministic eval pending) | — |

**Key finding**: SFT does most of the work (+15pp); GRPO on top adds another ~+7pp at peak, then **regresses** past sem 28 (over-fit / reward-hack on the small 86-task pool). For MAI-UI-2B specifically, **iter_4 is the best ckpt, not iter_19** — opposite of qwen3_vl_2b where iter_14/19 was best.

**Why does GRPO regress past sem 28?** MAI-UI-2B base is already GUI-specialized → `success_rate ≈ 50%` in early rollouts → only ~43% of GRPO groups have non-zero advantage (the rest are all-success or all-fail) → weak training signal + 86 tasks resampled rapidly → policy finds spurious shortcuts that succeed at temp=1.0 but fail at temp=0 deterministic eval. See `logs/2026-05-01T02-50_eb9bdce4/run_1_sft_grpo.md` for the full eval trajectory.

To independently re-eval the captured iter_4 peak:

```bash
# Inside the container. The captured ckpt was archived under a non-default
# name (iter_4_attempt4_semantic28_peak65pct) because retry-attempts would
# otherwise overwrite it.
docker exec -w /workspaces/cua-lite -e CUDA_VISIBLE_DEVICES=0,1 lite.slime-${SESSION_ID} \
  python scripts/rollout.py \
    --model-id Tongyi-MAI/MAI-UI-2B \
    --env-id androidworld \
    --model-path .exps/train/androidworld/mai_ui_2b/2026-05-01T02-50_eb9bdce4/run_1_sft_grpo/grpo/hf_archive/iter_4_attempt4_semantic28_peak65pct \
    --splits eval --concurrency 16 \
    --filter "lambda m: m.others.get('complexity') <= 2.0" \
    --sampling-kwargs '{"temperature": 0, "top_p": 1}' \
    --config-path scripts/configs/mai_ui/compact/androidworld.yaml \
    --log-root /tmp/eval_mai_ui_iter4_peak
```

---

## Notes / tuning

- **Pre-downloaded HF model.** Slime asserts `args.load` exists before doing anything — passing a HF hub id directly fails. The one-time `huggingface-cli download` in the working-dir section is mandatory.
- **Action schema is `mai_ui` (mobile_use), not qwen3_vl (computer_use).** Even though MAI-UI-2B uses the same transformer architecture as Qwen3-VL-2B-Instruct, its native function-calling output is the mobile_use schema (Alibaba's chosen format). `scripts/configs/mai_ui/compact/androidworld.yaml` and the `mai_ui` adapter handle the translation; SFT data from the qwen3-VL teacher is reformatted at export time (Step 2) — never passed through raw.
- **Two-variant rationale.** Most distillation recipes don't ask "what if we skip SFT?" because the base is too weak. MAI-UI-2B is the exception — its 41.86% base puts it within striking distance of the SFT+GRPO ceiling, making `run_0` a meaningful comparison rather than a strawman.
- **GRPO regression is a regime signal, not a bug.** When the base is strong enough that GRPO groups frequently fail the "advantage variance" condition (`mixed_group_ratio < 50%`), the policy will eventually find a spurious shortcut. The fix is to detect this early via G2 monitoring and stop at the peak iter — not to keep training.
- **No `noise=true` knob.** Unlike lite.osworld, androidworld has no `noise` env_kwarg. Native Pixel 6 resolution (`[1080, 2400]`; the teacher rollout config sets no `resolution`, so the agent sees native screenshots — downsampled to `[720, 1600]` via `agent_kwargs.resolution` for student RL/eval per `mai_ui/compact/androidworld.yaml`) and the native action set are the only sampling knobs.
