# Android-World: 8B-Teacher Distill SFT → GRPO (Qwen3-VL-2B)

> **Read [`devs/exps/train/AGENTS.md`](/devs/exps/train/AGENTS.md) first** for layout, `TRAIN_RUN_ID` contract, Slime v0.3.0 container/`SESSION_ID` rules, GPU discipline, snapshot template, and the agent workflow checklist.
>
> **Env-server prereq.** An env-server must be reachable and expose `androidworld` before launch — see [`devs/exps/train/AGENTS.md#env-server-prerequisite`](/devs/exps/train/AGENTS.md#env-server-prerequisite).

End-to-end recipe that uses **Qwen3-VL-8B-Instruct as the teacher** and **Qwen3-VL-2B-Instruct as the student**:

1. Teacher = 8B with `qwen3_vl/default/androidworld.yaml` (`full_history_size=4`, the protocol default) — rolls out the 86 train tasks (after `complexity <= 2.0` filter), `--group-size 4` for ~344 trajectories.
2. SFT student = 2B with `qwen3_vl/compact/androidworld.yaml` (`full_history_size=1`) on success-filtered teacher trajectories.
3. GRPO student = warm-started from the best SFT ckpt, trained on the **same 86 train tasks**.

> **Companion: qwen3.5 student.** [`../qwen3_5_2b/AGENTS.md`](/devs/exps/train/androidworld/qwen3_5_2b/AGENTS.md) reuses the **same** teacher rollout produced here — never re-roll out the 8B teacher just for qwen3.5. The two recipes share Step 1; Step 2 onwards diverges through qwen3.5 configs while the shared Slime v0.3.0 image selects the model family from `MODEL_ID`.

> Why no 50/50 anchor / random split here (vs the [`lite.osworld` recipe](/devs/exps/train/lite.osworld/qwen3_vl_2b/AGENTS.md))? The android-world train pool is only **86 tasks** (vs lite.osworld's 1376) → with `rollout_batch_size=16` per iter, every prompt is resampled within a few iters. Mixed-group ratio stays at ~70% out of the box; no need to rebalance the pool.

A prior 8B → 2B distill on this env hit **46.5% (SFT iter_93) → 58.1% (GRPO iter_19), +11.6pp / +25% relative** in 19 GRPO iters; this recipe is its codified form.

## ⚠️ Working directory + paths

> Host commands run from the repo root; container commands run inside the Slime container at `/workspaces/cua-lite`. See [`docs/slime.md`](/docs/slime.md) and [`devs/exps/train/AGENTS.md`](/devs/exps/train/AGENTS.md) for setup.

Per the [training experiments convention](/devs/exps/train/AGENTS.md#layout):

```bash
export TRAIN_RUN_ID="run_0"                                         # fixed string per Run id contract
COMMIT=$(git rev-parse --short HEAD)
COMMIT_TS=$(git log -1 --date=format:'%Y-%m-%dT%H-%M' --format=%cd HEAD)
BASE_HOST=.exps/train/androidworld/qwen3_vl_2b/${COMMIT_TS}_${COMMIT}/${TRAIN_RUN_ID}
BASE_CTR=/workspaces/cua-lite/${BASE_HOST}                          # same path inside the slime container

# Slime container session id (per training README)
export SESSION_ID=train-androidworld-qwen3_vl_2b
```

`BASE_CTR` is what container-side commands pass to `--log-root` / `SAVE_DIR` / `SAVE_HF_DIR` / `--model-path` / `-o`. Every step in this recipe runs inside the slime container; env containers spawn on the env-server's host (reached via `CUA_LITE_ENV_SERVER_URL`), not in this container — see [`docs/envs.md#env-server`](/docs/envs.md#env-server). `BASE_HOST` is the same physical path expressed as a host-relative string; it is only used for the post-rollout coverage `find` (`find` is a host shell builtin, no python deps).

---

## Step 1 — Teacher rollout (8B with rollout config)

Two sub-steps so the task pool is **pinned to a committed parquet** (reproducible across reruns and machines). **No `--env-kwargs` noise** — androidworld doesn't have a `noise` env_kwarg like lite.osworld; `qwen3_vl/default/androidworld.yaml`'s native settings produce enough action diversity.

> Runs in the qwen3-VL container (or directly on the host with `uv run python`) — **never** in the qwen3.5 container, which is poisoned with `transformers>=5.0` and breaks Qwen3-VL inference.

### 1.1 — pin the task pool

The complexity filter keeps 86 of 116 train tasks; commit them so different runs share the same pool. Runs **inside the slime container** — bind-mount makes `${BASE_CTR}` the same physical path as `${BASE_HOST}` on the host, so the parquet is visible from both sides.

```bash
docker exec -w /workspaces/cua-lite lite.slime-${SESSION_ID} \
  python -m lite.train.export.export_tasks \
    --env-id androidworld --split train \
    --filter "lambda m: m.others.get('complexity') <= 2.0" \
    -o ${BASE_CTR}/teacher_pool.parquet
```

### 1.2 — rollout from the pinned pool (≥ 3 retries)

`local.py` is resume-safe — re-running with the same `--log-root` + `--prompt-data` skips finished tasks. Loop 3× to ensure full coverage of the 86 × 4 = 344 expected rollouts.

Runs **inside the slime container**; env containers spawn on the env-server's host (reached via `CUA_LITE_ENV_SERVER_URL`), not in this container — see [`docs/envs.md#env-server`](/docs/envs.md#env-server). 8B fits across 2 GPUs via `dp_size=2`; if you only have 1 GPU available, drop the `--engine-kwargs` and pin `tp_size: 2` instead.

```bash
# CUDA_VISIBLE_DEVICES + --concurrency illustrative — scale to your allowed GPU range.
# Inside-container CUDA_VISIBLE_DEVICES uses the docker-remapped GPU indices (0..N-1).
for attempt in 1 2 3; do
  echo "=== teacher rollout attempt ${attempt}/3 ==="
  docker exec -w /workspaces/cua-lite -e CUDA_VISIBLE_DEVICES=0,1 lite.slime-${SESSION_ID} \
    python scripts/rollout.py \
      --model-id Qwen/Qwen3-VL-8B-Instruct --env-id androidworld \
      --prompt-data ${BASE_CTR}/teacher_pool.parquet --group-size 4 --concurrency 16 \
      --sampling-kwargs '{"temperature": 1.0, "top_p": 1}' \
      --engine-kwargs '{"dp_size": 2}' \
      --config-path scripts/configs/qwen3_vl/default/androidworld.yaml \
      --log-root ${BASE_CTR}/teacher_rollout \
      --group-shared-seed false
done

echo "Finished: $(find ${BASE_HOST}/teacher_rollout -name summary.json 2>/dev/null | wc -l) / 344"
```

Expected: 86 train tasks × 4 = 344 rollouts; ≥ 95% coverage after 3 retries. 8B teacher success rate ~55% (vs 2B self-distill's ~25%) → ≈190 success-filtered trajectories. The companion qwen3.5 recipe will read the **same** `teacher_rollout/` directory — do not duplicate this step.

---

## Step 2 — Export SFT parquet (success-only, variant A protocol)

Use `qwen3_vl/compact/androidworld.yaml` for export so SFT data matches the student's RL / eval distribution (`full_history_size=1`). This replicates the captured run's variant A; variant B (`qwen3_vl/default/androidworld.yaml`, `full_history_size=4`) gave essentially zero SFT lift on captured numbers (43.6% → 44.2%) so we skip it.

```bash
docker exec -w /workspaces/cua-lite lite.slime-${SESSION_ID} \
  python -m lite.train.export.export_sft \
    --config /workspaces/cua-lite/scripts/configs/qwen3_vl/compact/androidworld.yaml \
    --model-id Qwen/Qwen3-VL-2B-Instruct \
    --data-paths ${BASE_CTR}/teacher_rollout \
    --image-root /workspaces/cua-lite \
    --filter "lambda m: (m.others.get('episode_return') or 0) >= 1.0" \
    -o ${BASE_CTR}/sft_trajectory.parquet
```


---

## Step 3 — SFT (8B → 2B distill)

```bash
# CUDA_VISIBLE_DEVICES + NUM_TRAIN_GPUS illustrative — match your allowed GPU range.
# Default is TP=1 (pure DP + ZeRO-1); 2→4 GPUs adds DP and cuts SFT wall-clock roughly in half.
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=2 \
  MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
  SAVE=1 NO_SAVE_OPTIM=1 \
  NUM_EPOCH=2 \
  PROMPT_DATA=${BASE_CTR}/sft_trajectory.parquet \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_vl/compact/androidworld.yaml \
  SAVE_DIR=${BASE_CTR}/sft/megatron \
  SAVE_HF_DIR=${BASE_CTR}/sft/hf/iter_{rollout_id} \
  bash /workspaces/cua-lite/scripts/train/run_sft.sh
```

`NUM_EPOCH=2` — matches the captured 8B→2B distill (which peaked at iter_93, end of epoch 2); per-epoch HF ckpts give **2 candidates** for Step 4 to scan. Aligned across all `<env>/<recipe>` combinations for consistent wall-clock. `lr=2e-6` (run_sft.sh default) matches the captured distill — re-tune only if SFT eval is flat.

---

## Step 4 — Eval base + SFT (sanity)

Verify the 8B → 2B SFT moved the student above its base. The captured 8B → 2B distill achieved variant A's 47.1% (vs 20.9% base, +26pp); replicating that here should give a similar lift.

Eval is **resume-safe** — re-running the same `--log-root` skips finished tasks. Loop 3× per ckpt to guarantee full 86-task coverage. Runs **inside the slime container**.

```bash
SLUG=Qwen_Qwen3-VL-2B-Instruct

# 4.1 base — 3 retries
for attempt in 1 2 3; do
  echo "=== eval base attempt ${attempt}/3 ==="
  docker exec -w /workspaces/cua-lite -e CUDA_VISIBLE_DEVICES=0,1 lite.slime-${SESSION_ID} \
    python scripts/rollout.py \
      --model-id Qwen/Qwen3-VL-2B-Instruct --env-id androidworld \
      --splits eval --concurrency 16 \
      --filter "lambda m: m.others.get('complexity') <= 2.0" \
      --sampling-kwargs '{"temperature": 0, "top_p": 1}' \
      --config-path scripts/configs/qwen3_vl/compact/androidworld.yaml \
      --log-root ${BASE_CTR}/eval_base/${SLUG}
done

# 4.2 SFT — 3 retries per ckpt; replace iter_N with each ckpt you want to score
for attempt in 1 2 3; do
  echo "=== eval sft/iter_N attempt ${attempt}/3 ==="
  docker exec -w /workspaces/cua-lite -e CUDA_VISIBLE_DEVICES=0,1 lite.slime-${SESSION_ID} \
    python scripts/rollout.py \
      --model-id Qwen/Qwen3-VL-2B-Instruct --env-id androidworld \
      --model-path ${BASE_CTR}/sft/hf/iter_N \
      --splits eval --concurrency 16 \
      --filter "lambda m: m.others.get('complexity') <= 2.0" \
      --sampling-kwargs '{"temperature": 0, "top_p": 1}' \
      --config-path scripts/configs/qwen3_vl/compact/androidworld.yaml \
      --log-root ${BASE_CTR}/eval_sft/iter_N/${SLUG}
done
```

Pick the `iter_N` with the highest `mean_episode_return`; that becomes the GRPO warm start.

---

## Step 5 — Export GRPO data

Just the train + eval parquets — no anchor / random rebalancing needed (see header note).

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

Expected: 86 train tasks, 86 eval tasks (after the complexity filter).

---

## Step 6 — GRPO (warm-started from SFT)

Knobs match the prior 8B → 2B captured GRPO. ASYNC mode runs rollout (1 GPU, sglang) and train (1 GPU, megatron) concurrently.

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=1 NUM_ROLLOUT_GPUS=1 ASYNC=1 \
  MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
  HF_CKPT=${BASE_CTR}/sft/hf/iter_N \
  SAVE_DIR=${BASE_CTR}/grpo/megatron \
  SAVE_HF_DIR=${BASE_CTR}/grpo/hf/iter_{rollout_id} \
  ENV_ID=androidworld \
  PROMPT_DATA=${BASE_CTR}/train.parquet \
  EVAL_PROMPT_DATA=${BASE_CTR}/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_vl/compact/androidworld.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh
```

Notes:
- The GRPO config uses `full_history_size=1`, matching the SFT student protocol — student sees the same context shape at SFT, RL, and eval time.
- No per-split env_kwargs: train and eval rollouts share the `env_kwargs` in `qwen3_vl/compact/androidworld.yaml` (`loop_detect: 5`, `extra_tools`; no `max_steps` override → native `int(10 * task.complexity)`). `resolution: [720, 1600]` lives under `agent_kwargs` in the same file, not `env_kwargs`. A train-only override goes on the `export_tasks` call that builds `train.parquet` (`--env-kwargs '{...}'` → per-row `metadata.env_kwargs`, which deep-merges over the yaml).
- Slime saves a HF ckpt every `--save-interval` step (5 by default → `iter_4 / iter_9 / iter_14 / iter_19`). Eval is reported at the same cadence.
- Each iter ≈ 7 min wall-clock (rollout 400-500s + train ~30s) on the captured run. Total 20 iters ≈ 2.3h.
- The captured run hit `ActorUnavailableError` after ~3.5h on the 8B-warmed variant; expect similar instability on long runs. Best ckpt stayed at `iter_19` regardless.

---

## Step 7 — Eval GRPO ckpts (≥ 3 retries per ckpt)

Runs **inside the slime container**.

```bash
# Replace iter_M with each saved GRPO iter (4, 9, 14, 19, ...) to track the curve.
for attempt in 1 2 3; do
  echo "=== eval grpo/iter_M attempt ${attempt}/3 ==="
  docker exec -w /workspaces/cua-lite -e CUDA_VISIBLE_DEVICES=0,1 lite.slime-${SESSION_ID} \
    python scripts/rollout.py \
      --model-id Qwen/Qwen3-VL-2B-Instruct --env-id androidworld \
      --model-path ${BASE_CTR}/grpo/hf/iter_M \
      --splits eval --concurrency 16 \
      --filter "lambda m: m.others.get('complexity') <= 2.0" \
      --sampling-kwargs '{"temperature": 0, "top_p": 1}' \
      --config-path scripts/configs/qwen3_vl/compact/androidworld.yaml \
      --log-root ${BASE_CTR}/eval_grpo/iter_M/${SLUG}
done
```

In-training eval (the `eval N` lines in slime stdout / wandb) tracks the same metric automatically — re-run host-side eval only if you need byte-exact trajectories or want to evaluate ckpts past the auto-eval cadence.

---

## Acceptance gates

| Gate | What | Target | Hard stop? | Captured (8B-teacher) reference |
|---|---|---|---|---|
| **G1** | **SFT eval > base eval** | **≥ +20pp on the 86-task eval split (T=0)** | **🛑 yes — pipeline must stop here if it fails** | 8B → 2B variant A: +26.16pp; 2026-04-30 run: 23.84% → ~44% = **+20pp** |
| G2 | GRPO mixed-group ratio | ≥ 50% | no | typically ~70% on androidworld |
| G3 | GRPO eval > SFT eval | ≥ +3pp at peak iter | no | 8B-warmed: +11.6pp at iter_19; 2026-04-30 run: +10pp at iter_14 |
| G4 | Response-len mean stable across iters | within ±15% of SFT eval mean | no | 8B-warmed: 98 → 102 (+4%) |

**G1 enforcement** — Step 4 is a hard gate. Pipeline.sh must run Step 4 between SFT and GRPO and exit non-zero if `sft ≤ base + 0.20`. Don't start Step 6 until the gate passes.

If G1 fails substantially (SFT lift < 20pp), check teacher rollout success rate first — < 30% means the 8B is mode-collapsing on this task pool and re-rolling with different sampling kwargs may help. Other options: re-run SFT at `NUM_EPOCH=5`, inspect `sft_trajectory.parquet` for a data-pipeline regression, or last resort start GRPO directly from base 2B (set `HF_CKPT=Qwen/Qwen3-VL-2B-Instruct`).

## Snapshot template

Concrete copy-paste body for `logs/<commit-ts>_<commit>/<run_id>.md` (this recipe's stages: `teacher_rollout` → `sft` → `grpo`). Schema rules and the "skip mid-run failed-restart URLs" semantics are in [`devs/exps/train/AGENTS.md`](/devs/exps/train/AGENTS.md#snapshot--progress-log).

```markdown
# androidworld · qwen3_vl_2b @ <commit-ts>_<commit> · <run_id>

- **Recipe**: `devs/exps/train/androidworld/qwen3_vl_2b/AGENTS.md`
- **Commit**: `<short sha>` — `<subject>`
- **Host / GPUs**: `<hostname>` / `<e.g. 4,5>`
- **Container**: `lite.slime-train-androidworld-qwen3_vl_2b` (started <timestamp>)
- **wandb** (skip mid-run failed-restart URLs):
  - `sft`: `<url or "n/a">`
  - `grpo`: `<url or "n/a">`
- **Artifacts**: `.exps/train/androidworld/qwen3_vl_2b/<commit-ts>_<commit>/<run_id>/`
- **Started**: `<YYYY-MM-DD HH:MM TZ>`
- **Last updated**: `<YYYY-MM-DD HH:MM TZ>` ← bump on every edit
- **Notes**: anything non-default — non-default lr / batch-size, mid-run crashes + resumes, recipe deviations

## Stages run

| Stage | Status | Output |
|---|---|---|
| teacher_rollout | `not started` / `in_progress` / `done` / `failed (<reason>)` | `<n_traj> trajectories (<n_success> success-filtered)` once done (Step 1.2) |
| sft             | `not started` / `in_progress` / `done iter_<best>` / `failed` | `best ckpt eval <acc>% (vs base <acc>%)` once Step 4 has the numbers |
| grpo            | `not started` / `in_progress at iter_<n>` / `done iter_<best>` / `failed` | `best ckpt eval <acc>% (vs sft <acc>%)` once Step 7 has the numbers |

## Eval results

| Ckpt | Eval set | Finished | Mean episode return | Δ vs base |
|---|---|---|---|---|
| base `Qwen/Qwen3-VL-2B-Instruct` | `androidworld` eval | <nf>/<nt> | <mer> | — |
| sft iter_<n>                     | `androidworld` eval | <nf>/<nt> | <mer> | +/- <pp>pp |
| grpo iter_<n>                    | `androidworld` eval | <nf>/<nt> | <mer> | +/- <pp>pp |

Add rows incrementally — base after Step 4.1, each `sft iter_<n>` after Step 4.2, each `grpo iter_<n>` after Step 7. Sort within stage by iter ascending. Mark intentionally-skipped iters `⚠️ not eval'd`.

## Highlights

- short interpretive bullets — call out partial stages, surprising deltas vs the captured 8B→2B reference (+11.6pp), failure modes, key recipe knobs.
```

---

## Reference: captured 8B → 2B numbers

For comparison only — from a prior 8B → 2B distill on this env.

### SFT (variant A only, 86 train tasks, 2 epochs)

| ckpt | eval | Δ base |
|---|---|---|
| base Qwen3-VL-2B | 20.93% | — |
| variant A iter_93 (best) | **47.09%** | **+26.16pp** |

### GRPO from variant A iter_93

| eval iter | accuracy | Δ vs eval 0 |
|---|---|---|
| 0 (SFT base) | 46.51% (40/86) | — |
| 4 | 45.35% (39/86) | -1.16pp |
| 9 | 50.58% (43/86) | +4.07pp |
| 14 | 54.65% (47/86) | +8.14pp |
| 19 | **58.14% (50/86)** | **+11.63pp** ← best |

GRPO config: `lr=1e-6 const`, `eps_clip 0.2/0.28`, `kl=0`, `n_samples_per_prompt=8`, `rollout_batch_size=16`, `env_concurrency=32`. 18 generate() failures across 22 rollout iters (~0.6% trajectories), 0 ActorUnavailableError until ~3.5h mark.

## Notes / tuning

- **Teacher choice.** 8B teacher gives ~55% success rate (vs 2B self-distill's ~25%) → ≈190 success-filtered trajectories, materially stronger SFT lift. The companion `qwen3_5_2b` recipe reads the **same** `teacher_rollout/` — never re-roll out for the 5_2b student.
- **No `noise=true` knob.** Unlike lite.osworld, androidworld has no `noise` env_kwarg. Native Pixel 6 resolution (`[1080, 2400]`; the teacher rollout config sets no `resolution`, so the agent sees native screenshots — downsampled to `[720, 1600]` via `agent_kwargs.resolution` for student RL/eval per `qwen3_vl/compact/androidworld.yaml`) and the native action set are the only sampling knobs.
- **GRPO recipe stability.** The 8B-warmed run captured here was monotonic up to iter_19 (no eval drops). If your run shows eval regressing across iters (e.g. iter_9 < iter_4 < eval_0), check `mixed_group_ratio` — if < 30%, anchor pool is too small; rerun teacher with `--group-size 8` to densify.
- **Long-run failure mode.** Expect `ActorUnavailableError` somewhere past 3-5 hours of GRPO. If it crashes mid-eval, use the latest saved HF ckpt and re-launch with the same `(commit, run_id)` to pick up where megatron left off.
