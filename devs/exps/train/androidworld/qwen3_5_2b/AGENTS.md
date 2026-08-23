# Android-World: 8B-Teacher Distill SFT → GRPO (Qwen3.5-2B)

> **Slime v0.3.0 recipe.** Use `run_sft.sh` / `run_grpo.sh` /
> `run_reinforce.sh` with `MODEL_ID=Qwen/Qwen3.5-2B`; the family is
> auto-derived and the shared v0.3.0 image supports both Qwen3.5 and Qwen3-VL
> without container-local package mutation.


> **Read [`devs/exps/train/AGENTS.md`](/devs/exps/train/AGENTS.md) first** for layout, `TRAIN_RUN_ID` contract, Slime v0.3.0 container/`SESSION_ID` rules, GPU discipline, snapshot template, and the agent workflow checklist.
>
> **Env-server prereq.** An env-server must be reachable and expose `androidworld` before launch — see [`devs/exps/train/AGENTS.md#env-server-prerequisite`](/devs/exps/train/AGENTS.md#env-server-prerequisite).

End-to-end recipe that uses **Qwen3-VL-8B-Instruct as the teacher** (cross-family distill) and **Qwen/Qwen3.5-2B as the student**:

1. Teacher rollout — **reuse** the rollout produced by the companion [`../qwen3_vl_2b/AGENTS.md#step-1`](/devs/exps/train/androidworld/qwen3_vl_2b/AGENTS.md#step-1--teacher-rollout-8b-with-rollout-config) — same 86 train tasks (complexity ≤ 2.0), `--group-size 4`, ~344 trajectories. **Never re-roll out the 8B teacher just for qwen3.5** (same data either way; re-rolling is wasted compute).
2. SFT student = qwen3.5-2B with `qwen3_5/compact/androidworld.yaml` (`history_n=1`) on success-filtered teacher trajectories — **converted at export time** from the qwen3-VL trajectory format to qwen3.5's via the qwen3_5 export config.
3. GRPO student = warm-started from the best SFT ckpt, trained on the **same 86 train tasks**.

> Why no 50/50 anchor / random split here (vs the [`lite.osworld` recipe](/devs/exps/train/lite.osworld/qwen3_5_2b/AGENTS.md))? The android-world train pool is only **86 tasks** (vs lite.osworld's 1376) → with `rollout_batch_size=16` per iter, every prompt is resampled within a few iters. Mixed-group ratio stays at ~70% out of the box; no need to rebalance the pool.

**Why a distinct qwen3.5 path?** Qwen3.5 has native GatedDeltaNet (GDN) that needs `--qkv-format bshd` and fixed `--micro-batch-size` (no THD packing) — see [`run_grpo.sh`](/scripts/train/run_grpo.sh) (formerly `run_grpo_qwen3_5.sh`, merged in v0.3.0) header for the rationale. Since v0.3.0 the same script serves both families (`MODEL_FAMILY` auto-derived from `MODEL_ID`); no env mutation.

> **Cross-family distill caveat.** Qwen3.5 GDN is a different architecture from Qwen3-VL; even though `export_sft` converts the trajectory format via the qwen3_5 RL config, the student still has to learn the teacher's action distribution from scratch (no shared visual tokenizer / attention layout). Expect a slightly slower SFT ramp than the qwen3_vl peer at the same teacher rollout count.

## ⚠️ Working directory + paths

> Host commands run from the repo root; container commands run inside the Slime container at `/workspaces/cua-lite`. See [`docs/slime.md`](/docs/slime.md) and [`devs/exps/train/AGENTS.md`](/devs/exps/train/AGENTS.md) for setup.

Per the [training experiments convention](/devs/exps/train/AGENTS.md#layout):

```bash
export TRAIN_RUN_ID="run_0"                                         # fixed string per Run id contract
COMMIT=$(git rev-parse --short HEAD)
COMMIT_TS=$(git log -1 --date=format:'%Y-%m-%dT%H-%M' --format=%cd HEAD)
BASE_HOST=.exps/train/androidworld/qwen3_5_2b/${COMMIT_TS}_${COMMIT}/${TRAIN_RUN_ID}
BASE_CTR=/workspaces/cua-lite/${BASE_HOST}                          # same path inside the slime container

# Slime container session id
export SESSION_ID=train-androidworld-qwen3_5_2b
```

`BASE_CTR` is what container-side commands pass to `--log-root` / `SAVE_DIR` / `SAVE_HF_DIR` / `--model-path` / `-o`. Every step in this recipe runs inside the slime container; env containers spawn on the env-server's host (reached via `CUA_LITE_ENV_SERVER_URL`), not in this container — see [`docs/envs.md#env-server`](/docs/envs.md#env-server). `BASE_HOST` is the same physical path expressed as a host-relative string; the recipe only uses it for the post-rollout coverage `find` (`find` is a host shell builtin, no python deps).

---

## Step 1 — Teacher rollout (REUSE companion qwen3_vl_2b run)

**Do not run this step in this recipe.** Run [`../qwen3_vl_2b/AGENTS.md#step-1`](/devs/exps/train/androidworld/qwen3_vl_2b/AGENTS.md#step-1--teacher-rollout-8b-with-rollout-config) **first** (in the qwen3-VL container, not this one), then point this recipe at its output.

```bash
# Default: same (commit-ts, commit, TRAIN_RUN_ID) — `qwen3_vl_2b/${BASE}/teacher_rollout`.
# Override with TEACHER_ROLLOUT_DIR (absolute container path) to reuse a
# rollout from a different commit / run_id — cheaper than re-rolling the
# 8B teacher (1-3h) when neither the env nor the qwen3_vl_2b recipe has
# changed since that rollout. Use sparingly: drifting too far from the
# current commit forfeits the reproducibility contract.
TEACHER_ROLLOUT="${TEACHER_ROLLOUT_DIR:-/workspaces/cua-lite/.exps/train/androidworld/qwen3_vl_2b/${COMMIT_TS}_${COMMIT}/${TRAIN_RUN_ID}/teacher_rollout}"

# Sanity check on host. Bind-mount makes /workspaces/cua-lite/... and the
# repo-relative path the same physical dir; strip the prefix to get the
# host-side view.
TEACHER_ROLLOUT_HOST="${TEACHER_ROLLOUT#/workspaces/cua-lite/}"
echo "Teacher rollout count: $(find ${TEACHER_ROLLOUT_HOST} -name summary.json 2>/dev/null | wc -l) / 344"
```

If the count is < 95% (~327/344), go finish the companion's Step 1 first (or point `TEACHER_ROLLOUT_DIR` at a more complete rollout). The companion uses Qwen3-VL-8B-Instruct as teacher (~55% success rate, ≈190 success-filtered trajectories from 344).

---

## Step 2 — Export SFT parquet (success-only, qwen3.5 format)

Use `qwen3_5/compact/androidworld.yaml` for export — this **converts** the teacher's qwen3-VL trajectory format into qwen3.5's at parquet write time. The output parquet is qwen3.5-native; no further conversion needed at SFT time.

```bash
docker exec -w /workspaces/cua-lite lite.slime-${SESSION_ID} \
  python -m lite.train.export.export_sft \
    --config /workspaces/cua-lite/scripts/configs/qwen3_5/compact/androidworld.yaml \
    --model-id Qwen/Qwen3.5-2B \
    --data-paths ${TEACHER_ROLLOUT} \
    --image-root /workspaces/cua-lite \
    --filter "lambda m: (m.others.get('episode_return') or 0) >= 1.0" \
    -o ${BASE_CTR}/sft_trajectory.parquet
```

`${TEACHER_ROLLOUT}` is the companion's directory (set in Step 1). The parquet is written to **this** recipe's `BASE_CTR`, keeping qwen3.5 artifacts isolated.


---

## Step 3 — SFT (8B → qwen3.5-2B distill, qwen3.5 path)

v0.3.0: `run_sft.sh` auto-derives `MODEL_FAMILY=qwen3_5` from `MODEL_ID` (BSHD + fixed `MBS` path). No env mutation — the container is shareable across families.

```bash
# CUDA_VISIBLE_DEVICES + NUM_TRAIN_GPUS illustrative — match your allowed GPU range.
# Default is TP=1 (pure DP + ZeRO-1); 2→4 GPUs adds DP and cuts SFT wall-clock roughly in half.
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=2 \
  MODEL_ID=Qwen/Qwen3.5-2B \
  SAVE=1 NO_SAVE_OPTIM=1 \
  NUM_EPOCH=2 \
  PROMPT_DATA=${BASE_CTR}/sft_trajectory.parquet \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_5/compact/androidworld.yaml \
  SAVE_DIR=${BASE_CTR}/sft/megatron \
  SAVE_HF_DIR=${BASE_CTR}/sft/hf/iter_{rollout_id} \
  bash /workspaces/cua-lite/scripts/train/run_sft.sh
```

`NUM_EPOCH=2` mirrors the captured 8B → qwen3-VL-2B distill (peaked at iter_93, end of epoch 2) and is now unified across all `<env>/<recipe>` combinations for consistent wall-clock. `lr=2e-6` (run_sft.sh default) — re-tune only if SFT eval is flat.

---

## Step 4 — Eval base + SFT (sanity)

Verify the 8B → qwen3.5-2B SFT moved the student above its base. If SFT brings no gain, GRPO has no warm start to leverage.

Eval is **resume-safe** — re-running the same `--log-root` skips finished tasks. Loop 3× per ckpt to guarantee full 86-task coverage.

Runs **inside the slime container**.

```bash
SLUG=Qwen_Qwen3.5-2B

# 4.1 base — 3 retries
for attempt in 1 2 3; do
  echo "=== eval base attempt ${attempt}/3 ==="
  docker exec -w /workspaces/cua-lite -e CUDA_VISIBLE_DEVICES=0,1 lite.slime-${SESSION_ID} \
    python scripts/rollout.py \
      --model-id Qwen/Qwen3.5-2B --env-id androidworld \
      --splits eval --concurrency 16 \
      --filter "lambda m: m.others.get('complexity') <= 2.0" \
      --sampling-kwargs '{"temperature": 0, "top_p": 1}' \
      --config-path scripts/configs/qwen3_5/compact/androidworld.yaml \
      --log-root ${BASE_CTR}/eval_base/${SLUG}
done

# 4.2 SFT — 3 retries per ckpt; replace iter_N with each ckpt you want to score
for attempt in 1 2 3; do
  echo "=== eval sft/iter_N attempt ${attempt}/3 ==="
  docker exec -w /workspaces/cua-lite -e CUDA_VISIBLE_DEVICES=0,1 lite.slime-${SESSION_ID} \
    python scripts/rollout.py \
      --model-id Qwen/Qwen3.5-2B --env-id androidworld \
      --model-path ${BASE_CTR}/sft/hf/iter_N \
      --splits eval --concurrency 16 \
      --filter "lambda m: m.others.get('complexity') <= 2.0" \
      --sampling-kwargs '{"temperature": 0, "top_p": 1}' \
      --config-path scripts/configs/qwen3_5/compact/androidworld.yaml \
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

## Step 6 — GRPO (warm-started from SFT, qwen3.5 path)

v0.3.0: `run_grpo.sh` with `MODEL_ID=Qwen/Qwen3.5-2B` auto-selects the qwen3.5 path (`--qkv-format bshd` + fixed `--micro-batch-size` for native GatedDeltaNet correctness). ASYNC mode runs rollout (1 GPU, sglang) and train (1 GPU, megatron) concurrently.

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=1 NUM_ROLLOUT_GPUS=1 ASYNC=1 \
  MODEL_ID=Qwen/Qwen3.5-2B \
  HF_CKPT=${BASE_CTR}/sft/hf/iter_N \
  SAVE_DIR=${BASE_CTR}/grpo/megatron \
  SAVE_HF_DIR=${BASE_CTR}/grpo/hf/iter_{rollout_id} \
  ENV_ID=androidworld \
  PROMPT_DATA=${BASE_CTR}/train.parquet \
  EVAL_PROMPT_DATA=${BASE_CTR}/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_5/compact/androidworld.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh
```

Notes:
- The GRPO config uses `history_n=1`, matching the SFT student protocol — student sees the same context shape at SFT, RL, and eval time.
- No per-split env_kwargs: train and eval rollouts share the `env_kwargs` in `qwen3_5/compact/androidworld.yaml` (`loop_detect: 5`, `extra_tools`; no `max_steps` override → native `int(10 * task.complexity)`). `resolution: [720, 1600]` lives under `agent_kwargs` in the same file, not `env_kwargs`. A train-only override goes on the `export_tasks` call that builds `train.parquet` (`--env-kwargs '{...}'` → per-row `metadata.env_kwargs`, which deep-merges over the yaml).
- The qwen3_5 path uses `--qkv-format bshd` and `MBS=1` (default; bump only if VRAM allows). Don't try `--use-dynamic-batch-size` — it bypasses the GDN-correct path. `MAX_TOKENS_PER_GPU` from the qwen3_vl recipes does not apply here.
- Slime saves a HF ckpt every `--save-interval` step (5 by default → `iter_4 / iter_9 / iter_14 / iter_19`). Eval is reported at the same cadence.
- Each iter wall-clock is slower than the qwen3_vl peer recipe (BSHD + fixed MBS = no THD packing). Plan ~10-15 min/iter at MBS=1; total 20 iters ≈ 3-5h.

---

## Step 7 — Eval GRPO ckpts (≥ 3 retries per ckpt)

Runs **inside the slime container**.

```bash
# Replace iter_M with each saved GRPO iter (4, 9, 14, 19, ...) to track the curve.
for attempt in 1 2 3; do
  echo "=== eval grpo/iter_M attempt ${attempt}/3 ==="
  docker exec -w /workspaces/cua-lite -e CUDA_VISIBLE_DEVICES=0,1 lite.slime-${SESSION_ID} \
    python scripts/rollout.py \
      --model-id Qwen/Qwen3.5-2B --env-id androidworld \
      --model-path ${BASE_CTR}/grpo/hf/iter_M \
      --splits eval --concurrency 16 \
      --filter "lambda m: m.others.get('complexity') <= 2.0" \
      --sampling-kwargs '{"temperature": 0, "top_p": 1}' \
      --config-path scripts/configs/qwen3_5/compact/androidworld.yaml \
      --log-root ${BASE_CTR}/eval_grpo/iter_M/${SLUG}
done
```

In-training eval (the `eval N` lines in slime stdout / wandb) tracks the same metric automatically — re-run host-side eval only if you need byte-exact trajectories or want to evaluate ckpts past the auto-eval cadence.

---

## Acceptance gates

| Gate | What | Target | Hard stop? | Captured reference |
|---|---|---|---|---|
| **G1** | **SFT eval > base eval** | **≥ +20pp on the 86-task eval split (T=0)** | **🛑 yes — pipeline must stop here if it fails** | qwen3_vl peer: +26.16pp; 2026-04-30 run: 5.23% → ~50% = **+45pp** ✅ (best of the 4 recipes) |
| G2 | GRPO mixed-group ratio | ≥ 50% | no | typically ~70% on androidworld |
| G3 | GRPO eval > SFT eval | ≥ +3pp at peak iter | no | qwen3_vl peer: +11.6pp at iter_19; 2026-04-30 run: +12pp at iter_14 |
| G4 | Response-len mean stable across iters | within ±15% of SFT eval mean | no | qwen3_vl peer: stable |

**G1 enforcement** — Step 4 is a hard gate. Pipeline.sh must run Step 4 between SFT and GRPO and exit non-zero if `sft ≤ base + 0.20`. Don't start Step 6 until the gate passes.

If G1 underperforms (e.g. SFT lift < 10pp): the cross-family conversion (qwen3-VL teacher → qwen3.5 student format) is paying a higher tax than expected. Inspect a few rows of `sft_trajectory.parquet` against their teacher rollout source — same task, side-by-side — to spot whether action calls are being silently dropped or reformatted. Other options: re-run SFT at `NUM_EPOCH=5`, or last resort start GRPO directly from base (set `HF_CKPT=Qwen/Qwen3.5-2B`).

## Snapshot template

Concrete copy-paste body for `logs/<commit-ts>_<commit>/<run_id>.md` (this recipe's stages: `teacher_rollout` → `sft` → `grpo`). Schema rules and the "skip mid-run failed-restart URLs" semantics are in [`devs/exps/train/AGENTS.md`](/devs/exps/train/AGENTS.md#snapshot--progress-log).

```markdown
# androidworld · qwen3_5_2b @ <commit-ts>_<commit> · <run_id>

- **Recipe**: `devs/exps/train/androidworld/qwen3_5_2b/AGENTS.md`
- **Commit**: `<short sha>` — `<subject>`
- **Host / GPUs**: `<hostname>` / `<e.g. 4,5>`
- **Container**: `lite.slime-train-androidworld-qwen3_5_2b` (started <timestamp>)
- **wandb** (skip mid-run failed-restart URLs):
  - `sft`: `<url or "n/a">`
  - `grpo`: `<url or "n/a">`
- **Artifacts**: `.exps/train/androidworld/qwen3_5_2b/<commit-ts>_<commit>/<run_id>/`
- **Started**: `<YYYY-MM-DD HH:MM TZ>`
- **Last updated**: `<YYYY-MM-DD HH:MM TZ>` ← bump on every edit
- **Notes**: anything non-default — non-default lr / `MBS`, container-poisoning concerns, mid-run crashes + resumes, recipe deviations

## Stages run

| Stage | Status | Output |
|---|---|---|
| teacher_rollout | `reused from <companion qwen3_vl_2b run_id>` | (no rollout in this recipe) |
| sft             | `not started` / `in_progress` / `done iter_<best>` / `failed` | `best ckpt eval <acc>% (vs base <acc>%)` once Step 4 has the numbers |
| grpo            | `not started` / `in_progress at iter_<n>` / `done iter_<best>` / `failed` | `best ckpt eval <acc>% (vs sft <acc>%)` once Step 7 has the numbers |

## Eval results

| Ckpt | Eval set | Finished | Mean episode return | Δ vs base |
|---|---|---|---|---|
| base `Qwen/Qwen3.5-2B` | `androidworld` eval | <nf>/<nt> | <mer> | — |
| sft iter_<n>           | `androidworld` eval | <nf>/<nt> | <mer> | +/- <pp>pp |
| grpo iter_<n>          | `androidworld` eval | <nf>/<nt> | <mer> | +/- <pp>pp |

Add rows incrementally — base after Step 4.1, each `sft iter_<n>` after Step 4.2, each `grpo iter_<n>` after Step 7. Sort within stage by iter ascending. Mark intentionally-skipped iters `⚠️ not eval'd`.

## Highlights

- short interpretive bullets — Qwen3.5 GDN-specific quirks, length-drift, mixed-group collapse, container-poisoning incidents, comparison to the qwen3_vl peer numbers.
```

---

## Notes / tuning

- **Qwen3.5 GDN specifics.** Native GatedDeltaNet means slower per-step training (no THD packing) but stable convergence. Don't add `--use-dynamic-batch-size` — it bypasses the GDN-correct path.
- **~~Container poisoning~~ (obsolete since slime v0.3.0).** The old qwen3_5 scripts' `pip install` mutation is gone; one container serves both model families. See the OBSOLETE note in [`devs/exps/train/AGENTS.md`](/devs/exps/train/AGENTS.md).
- **Cross-family teacher reuse.** Qwen3-VL-8B teacher trajectories drive both this recipe and the [`../qwen3_vl_2b/AGENTS.md`](/devs/exps/train/androidworld/qwen3_vl_2b/AGENTS.md) recipe. The same physical `teacher_rollout/` directory is read by both; `export_sft` does the per-recipe format conversion. Running the 8B teacher rollout twice is forbidden waste.
- **No `noise=true` knob.** Unlike lite.osworld, androidworld has no `noise` env_kwarg. Native Pixel 6 resolution (`[1080, 2400]`; the teacher rollout config sets no `resolution`, so the agent sees native screenshots — downsampled to `[720, 1600]` via `agent_kwargs.resolution` for student RL/eval per `qwen3_5/compact/androidworld.yaml`) and the native action set are the only sampling knobs.
- **Long-run failure mode.** Expect `ActorUnavailableError` somewhere past 3-5 hours of GRPO (typically a downstream OOM or host-side resource exhaustion under heavy multi-tenant load — `__clone` segfault during `fork()` is a known symptom). If it crashes mid-eval, use the latest saved HF ckpt and re-launch with the same `(commit, run_id)` to pick up where megatron left off.
