# Lite.OSWorld: 8B-Teacher Distill SFT → GRPO (Qwen3-VL-2B)

> **Read [`devs/exps/train/AGENTS.md`](/devs/exps/train/AGENTS.md) first** for layout, `TRAIN_RUN_ID` contract, Slime v0.3.0 container/`SESSION_ID` rules, GPU discipline, snapshot template, and the agent workflow checklist.
>
> **Env-server prereq.** An env-server must be reachable and expose `lite.osworld` before launch — see [`devs/exps/train/AGENTS.md#env-server-prerequisite`](/devs/exps/train/AGENTS.md#env-server-prerequisite).

End-to-end recipe that uses **Qwen3-VL-8B-Instruct as the teacher** and **Qwen3-VL-2B-Instruct as the student**:

1. Teacher = 8B with `qwen3_vl/default/lite.osworld.yaml` (`full_history_size=4`, the protocol default) plus `--env-kwargs '{"noise": true}'` — rolls out 512 train tasks. (Reference: a prior 8B-noise rollout at sample=256 hit 82/256 = 32% success rate; at sample=512 expect ≈160 success-filtered trajectories.)
2. SFT student = 2B with `qwen3_vl/compact/lite.osworld.yaml` (`full_history_size=1`) on success-filtered teacher trajectories.
3. GRPO student = warm-started from the best SFT ckpt, trained on a **50/50 mixed prompt pool**:
   - 50% from prompts the teacher succeeded on (high non-degenerate group rate)
   - 50% randomly drawn from the full train split (exploration coverage)

The 50/50 split is the key difference from a vanilla GRPO recipe: with all-random prompts, ≥70% of GRPO groups end up all-zero on lite.osworld (zero gradient signal). Anchoring half the pool to "achievable" prompts pushes mixed-group ratio up, restoring per-iter learning signal.

> **Companion: qwen3.5 student.** [`../qwen3_5_2b/AGENTS.md`](/devs/exps/train/lite.osworld/qwen3_5_2b/AGENTS.md) reuses the **same** teacher rollout produced here — never re-roll out the 8B teacher just for qwen3.5. Step 1 is shared; Step 2 onwards diverges through qwen3.5 configs while the shared Slime v0.3.0 image selects the model family from `MODEL_ID`.

## ⚠️ Working directory + paths

> Host commands run from the repo root; container commands run inside the Slime container at `/workspaces/cua-lite`. See [`docs/slime.md`](/docs/slime.md) and [`devs/exps/train/AGENTS.md`](/devs/exps/train/AGENTS.md) for setup.

Per the [training experiments convention](/devs/exps/train/AGENTS.md#layout), all artifacts go under one campaign-scoped tree. Set these once at kickoff:

```bash
export TRAIN_RUN_ID="run_0"                                         # fixed string per Run id contract
COMMIT=$(git rev-parse --short HEAD)
COMMIT_TS=$(git log -1 --date=format:'%Y-%m-%dT%H-%M' --format=%cd HEAD)
BASE_HOST=.exps/train/lite.osworld/qwen3_vl_2b/${COMMIT_TS}_${COMMIT}/${TRAIN_RUN_ID}
BASE_CTR=/workspaces/cua-lite/${BASE_HOST}                          # same path inside the slime container

# Slime container session id (per training README)
export SESSION_ID=train-lite.osworld-qwen3_vl_2b
```

`BASE_CTR` is what container-side commands pass to `--log-root` / `SAVE_DIR` / `SAVE_HF_DIR` / `--model-path` / `-o`. Every step in this recipe runs inside the slime container; env containers spawn on the env-server's host (reached via `CUA_LITE_ENV_SERVER_URL`), not in this container — see [`docs/envs.md#env-server`](/docs/envs.md#env-server). `BASE_HOST` is the same physical path expressed as a host-relative string; the recipe only uses it for the post-rollout coverage `find` (`find` is a host shell builtin, no python deps).

---

## Step 1 — Teacher rollout (8B with rollout config)

Two sub-steps so the task pool is **pinned to a committed parquet** (reproducible across reruns and across machines, no `--sample` random drift):

> Runs in the qwen3-VL container (or directly on the host with `uv run python`) — **never** in the qwen3.5 container, which is poisoned with `transformers>=5.0` and breaks Qwen3-VL inference.

### 1.1 — pin the task pool

Runs **inside the slime container** — bind-mount makes `${BASE_CTR}` the same physical path as `${BASE_HOST}` on the host, so the parquet is visible from both sides.

```bash
docker exec -w /workspaces/cua-lite lite.slime-${SESSION_ID} \
  python -m lite.train.export.export_tasks \
    --env-id lite.osworld --split train --sample 512 \
    --filter "lambda m: not m.others.get('exclude_reason')" \
    -o ${BASE_CTR}/teacher_pool.parquet
```

512 task records (`{problem, metadata: {env_key, split, ...}}`), seeded random sample. The parquet lives under the campaign artifact root, so the same `(commit, run_id)` always rolls out the same task list.

### 1.2 — rollout from the pinned pool (≥ 3 retries)

`noise=true` stays critical: without it the success-only filter starves rare-action coverage and the student mode-collapses on eval (captured experiments noted dropped drag/double_click/right_click). No yaml sets it — the `--env-kwargs '{"noise": true}'` below is what turns it on (the env default is `noise: false`).

`local.py` is **resume-safe**: re-running with the same `--log-root` + `--prompt-data` skips tasks whose `<task_id>/sample_NN/summary.json` already exists, so a crash mid-rollout (env timeout / docker port collision / sglang OOM) only forfeits the in-flight task — re-launch picks up the rest. Loop the command 3 times to ensure full coverage of the 512-task pool. Runs **inside the slime container**; env containers spawn on the env-server's host (reached via `CUA_LITE_ENV_SERVER_URL`), not in this container — see [`docs/envs.md#env-server`](/docs/envs.md#env-server). 8B fits across 2 GPUs via `dp_size=2`:

```bash
# CUDA_VISIBLE_DEVICES + --concurrency illustrative — scale to your allowed GPU range.
# Inside-container CUDA_VISIBLE_DEVICES uses the docker-remapped GPU indices (0..N-1), regardless of host range.
for attempt in 1 2 3; do
  echo "=== teacher rollout attempt ${attempt}/3 ==="
  docker exec -w /workspaces/cua-lite -e CUDA_VISIBLE_DEVICES=0,1 lite.slime-${SESSION_ID} \
    python scripts/rollout.py \
      --model-id Qwen/Qwen3-VL-8B-Instruct --env-id lite.osworld \
      --prompt-data ${BASE_CTR}/teacher_pool.parquet --concurrency 32 \
      --sampling-kwargs '{"temperature": 1.0, "top_p": 1}' \
      --engine-kwargs '{"dp_size": 2}' \
      --env-kwargs '{"noise": true}' \
      --config-path scripts/configs/qwen3_vl/default/lite.osworld.yaml \
      --log-root ${BASE_CTR}/teacher_rollout
done

# Sanity check coverage — finished tasks should be close to 512 (host-side; same physical path as ${BASE_CTR})
echo "Finished: $(find ${BASE_HOST}/teacher_rollout -name summary.json 2>/dev/null | wc -l) / 512"
```

Expected after 3 retries: ≥ 95% coverage. A few env-flaky tasks (Markor / multi_apps perturbations) tend to hard-fail repeatedly — that's fine, success-filter in Step 2 drops them anyway. If coverage < 90%, run `/cleanup lite.osworld` (clears leaked `lite-env-*` containers) and loop a 4th attempt.

Teacher success rate ~30% on noise=true (prior 8B-noise reference: 82/256 = 32.2%; sample=512 expects ≈150-170 success-filtered trajectories). The companion qwen3.5 recipe will read the **same** `teacher_rollout/` directory — do not duplicate this step.

---

## Step 2 — Export SFT parquet (success-only)

Use `qwen3_vl/compact/lite.osworld.yaml` for export so the SFT data matches the student's RL / eval distribution (`full_history_size=1`).

```bash
docker exec -w /workspaces/cua-lite lite.slime-${SESSION_ID} \
  python -m lite.train.export.export_sft \
    --config /workspaces/cua-lite/scripts/configs/qwen3_vl/compact/lite.osworld.yaml \
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
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_vl/compact/lite.osworld.yaml \
  SAVE_DIR=${BASE_CTR}/sft/megatron \
  SAVE_HF_DIR=${BASE_CTR}/sft/hf/iter_{rollout_id} \
  bash /workspaces/cua-lite/scripts/train/run_sft.sh
```

`NUM_EPOCH=2` — unified across `<env>/<recipe>` combinations to minimize wall-clock. **Caveat**: a captured run on this env at `NUM_EPOCH=5` peaked at epoch 4 (+6.25pp), but the same recipe at `NUM_EPOCH=2` produced **−4.7pp** (regression) on the 2026-04-30 run. lite.osworld is reward-sparse and the SFT learning curve is non-monotonic in epochs (1ep was −14% in the captured run before crossing base at 2ep+). 2 epochs is the absolute minimum here — if you want certainty over speed, **start at `NUM_EPOCH=5` for this env** and rely on the G1 gate to catch undertraining. `DUMP` stays OFF (default).

---

## Step 4 — 🛑 Eval base + SFT (HARD GATE)

This is a **hard pipeline gate**, not a sanity check. After the SFT ckpts are saved, the pipeline must:

1. Eval the base model and the latest SFT ckpt(s) on the deterministic 128-task subset.
2. Parse `mean_episode_return` from each `summary.json`.
3. **Stop the pipeline if `sft_best ≤ base + 0.05`** (a `≥ +5pp` lift is the minimum acceptable signal).

GRPO from a regressed SFT just spends GPU climbing back to base — the +pp curve you'll see is the model recovering from the SFT hole, not actually learning beyond base. Don't start GRPO until G1 passes. See § Acceptance gates for recovery options if it fails.

Like Step 1.2, eval is **resume-safe** — re-running the same `--log-root` skips finished tasks. Loop 3× per ckpt to guarantee full 128-task coverage.

Runs **inside the slime container**; env containers spawn on the env-server's host (reached via `CUA_LITE_ENV_SERVER_URL`), not in this container — see [`docs/envs.md#env-server`](/docs/envs.md#env-server).

```bash
SLUG=Qwen_Qwen3-VL-2B-Instruct

# 4.1 base — 3 retries
for attempt in 1 2 3; do
  echo "=== eval base attempt ${attempt}/3 ==="
  docker exec -w /workspaces/cua-lite -e CUDA_VISIBLE_DEVICES=0,1 lite.slime-${SESSION_ID} \
    python scripts/rollout.py \
      --model-id Qwen/Qwen3-VL-2B-Instruct --env-id lite.osworld \
      --splits eval --sample 128 --concurrency 16 \
      --filter "lambda m: not m.others.get('exclude_reason')" \
      --sampling-kwargs '{"temperature": 0, "top_p": 1}' \
      --config-path scripts/configs/qwen3_vl/compact/lite.osworld.yaml \
      --log-root ${BASE_CTR}/eval_base/${SLUG}
done

# 4.2 SFT — by default score the LATEST SFT ckpt only (iter_N = highest iter saved). Replace iter_N below with that exact value. Optionally repeat the block for intermediate ckpts if you want the full epoch curve (extra eval cost).
for attempt in 1 2 3; do
  echo "=== eval sft/iter_N attempt ${attempt}/3 ==="
  docker exec -w /workspaces/cua-lite -e CUDA_VISIBLE_DEVICES=0,1 lite.slime-${SESSION_ID} \
    python scripts/rollout.py \
      --model-id Qwen/Qwen3-VL-2B-Instruct --env-id lite.osworld \
      --model-path ${BASE_CTR}/sft/hf/iter_N \
      --splits eval --sample 128 --concurrency 16 \
      --filter "lambda m: not m.others.get('exclude_reason')" \
      --sampling-kwargs '{"temperature": 0, "top_p": 1}' \
      --config-path scripts/configs/qwen3_vl/compact/lite.osworld.yaml \
      --log-root ${BASE_CTR}/eval_sft/iter_N/${SLUG}
done
```

Pick the `iter_N` with the highest `mean_episode_return` from `eval_sft/iter_N/${SLUG}/summary.json`; that becomes the GRPO warm start.

---

## Step 5 — Build the 50/50 GRPO prompt pool

Filter the teacher rollout logs to the **task_ids of successful trajectories** (the "anchor" half). Random-sample the same count from the full train split (excluding the anchor task_ids) for the "exploration" half. Concatenate, write to one parquet.

`export_tasks` (today's CLI) accepts only `--env-id / --split / --head / --sample / --seed / --filter / --env-kwargs / -o` — no `--paths` / `--exclude-paths`. We derive the anchor task_id set from the teacher rollout log tree, then inline it into a `--filter` lambda. (Filter reads `m.others.get('task_id')` from every metadata row.)

```bash

# 5.0 build the anchor task_id set (success-filtered teacher rollouts)
ANCHOR_TASK_IDS=$(python3 -c "
import json, glob, os
ids = set()
for p in glob.glob('${BASE_HOST}/teacher_rollout/*/sample_*/summary.json'):
    if json.load(open(p)).get('episode_return', 0) >= 1.0:
        ids.add(os.path.basename(os.path.dirname(os.path.dirname(p))))
print(repr(ids))
")
echo "Anchor pool size: $(echo $ANCHOR_TASK_IDS | python3 -c 'import ast, sys; print(len(ast.literal_eval(sys.stdin.read())))')"

# 5.1 export the anchor half (success task_ids only)
# --env-kwargs writes {"noise": true} into every row's metadata.env_kwargs, so the
# GRPO train rollouts get the teacher's noise injection. 5.4 (eval) omits it.
docker exec -w /workspaces/cua-lite lite.slime-${SESSION_ID} \
  python -m lite.train.export.export_tasks --env-id lite.osworld --split train \
    --filter "lambda m: not m.others.get('exclude_reason') and m.others.get('task_id') in ${ANCHOR_TASK_IDS}" \
    --env-kwargs '{"noise": true}' \
    -o ${BASE_CTR}/grpo_pool_anchor.parquet

# 5.2 export the exploration half (random from full train, excluding anchor task_ids)
# N = anchor row count (≈160 with --sample 512 in Step 1).
N=$(docker exec lite.slime-${SESSION_ID} \
      python -c "import pyarrow.parquet as pq; print(pq.read_table('${BASE_CTR}/grpo_pool_anchor.parquet').num_rows)")
docker exec -w /workspaces/cua-lite lite.slime-${SESSION_ID} \
  python -m lite.train.export.export_tasks --env-id lite.osworld --split train \
    --filter "lambda m: not m.others.get('exclude_reason') and m.others.get('task_id') not in ${ANCHOR_TASK_IDS}" \
    --sample ${N} \
    --env-kwargs '{"noise": true}' \
    -o ${BASE_CTR}/grpo_pool_random.parquet

# 5.3 merge into the final GRPO pool (50/50 by row count)
docker exec -w /workspaces/cua-lite lite.slime-${SESSION_ID} \
  python -m lite.data.merge \
    -i ${BASE_CTR}/grpo_pool_anchor.parquet \
       ${BASE_CTR}/grpo_pool_random.parquet \
    -o ${BASE_CTR}/grpo_pool_50_50.parquet

# 5.4 eval pool — same as the SFT recipe (--sample 128 holdout from the eval split)
docker exec -w /workspaces/cua-lite lite.slime-${SESSION_ID} \
  python -m lite.train.export.export_tasks --env-id lite.osworld --split eval --sample 128 \
    --filter "lambda m: not m.others.get('exclude_reason')" \
    -o ${BASE_CTR}/eval.parquet
```

> Future-friendlier: if `export_tasks` ever grows native `--paths` / `--exclude-paths` (mirroring `export_sft`), drop the inline-set trick. The `m.others.get('task_id')` filter route is a stable fallback either way.

---

## Step 6 — GRPO (warm-started from SFT)

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=1 NUM_ROLLOUT_GPUS=1 ASYNC=1 \
  MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
  HF_CKPT=${BASE_CTR}/sft/hf/iter_N \
  SAVE_DIR=${BASE_CTR}/grpo/megatron \
  SAVE_HF_DIR=${BASE_CTR}/grpo/hf/iter_{rollout_id} \
  ENV_ID=lite.osworld \
  PROMPT_DATA=${BASE_CTR}/grpo_pool_50_50.parquet \
  EVAL_PROMPT_DATA=${BASE_CTR}/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_vl/compact/lite.osworld.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh
```

Notes:
- The GRPO config uses `full_history_size=1`, matching the SFT student protocol — student sees the same context shape at SFT, RL, and eval time.
- Train rollouts get the same noise injection as the teacher because Step 5.1/5.2 exported the pool with `--env-kwargs '{"noise": true}'`: each row's `metadata.env_kwargs` deep-merges over the yaml at rollout (`prompt_data > args > yaml`). `eval.parquet` (Step 5.4) omits it, so eval stays deterministic at the env default `noise: false`. No yaml carries a `noise` key.
- Slime saves a HF ckpt every `--save-interval` step (5 by default → `iter_4 / iter_9 / ...`). Eval is reported at the same cadence (`eval 0`, `eval 4`, `eval 9`, ...).

---

## Step 7 — Eval GRPO ckpts (≥ 3 retries per ckpt)

Runs **inside the slime container**.

```bash
# Replace iter_M with each saved GRPO iter (4, 9, 14, ...) to track the curve.
for attempt in 1 2 3; do
  echo "=== eval grpo/iter_M attempt ${attempt}/3 ==="
  docker exec -w /workspaces/cua-lite -e CUDA_VISIBLE_DEVICES=0,1 lite.slime-${SESSION_ID} \
    python scripts/rollout.py \
      --model-id Qwen/Qwen3-VL-2B-Instruct --env-id lite.osworld \
      --model-path ${BASE_CTR}/grpo/hf/iter_M \
      --splits eval --sample 128 --concurrency 16 \
      --filter "lambda m: not m.others.get('exclude_reason')" \
      --sampling-kwargs '{"temperature": 0, "top_p": 1}' \
      --config-path scripts/configs/qwen3_vl/compact/lite.osworld.yaml \
      --log-root ${BASE_CTR}/eval_grpo/iter_M/${SLUG}
done
```

In-training eval (the `eval N` lines in slime stdout / wandb) tracks the same metric automatically — re-run the host-side eval only if you need byte-exact trajectories or want to evaluate ckpts past the auto-eval cadence.

---

## Acceptance gates

| Gate | What | Target | Hard stop? | Captured (8B-teacher) reference |
|---|---|---|---|---|
| **G1** | **SFT eval > base eval** | **≥ +5pp on this recipe's 128-task subset (T=0)** | **🛑 yes — pipeline must stop here if it fails** | prior 8B-noise distill peak: +6.25pp at iter_503 (4 epochs); current 2-epoch path produced **−4.7pp** (regression) on the 2026-04-30 run, confirming the gate is necessary |
| G2 | GRPO mixed-group ratio (early iters) | ≥ 50% (vs ~30% with all-random pool) | no | — |
| G3 | GRPO eval > SFT eval | ≥ +3pp at peak iter | no — but if G3 fails, the run is consuming GPU for nothing | (no captured GRPO follow-up) |
| G4 | Response-len mean stable across iters | within ±15% of SFT-iter eval mean | no | — |

**G1 enforcement** — Step 4 (eval base + SFT) is a hard gate, not a sanity check. The pipeline.sh wrapper for this recipe **must** run Step 4 on the latest SFT ckpt and the base model, parse `mean_episode_return` from each `summary.json`, and exit non-zero if `sft ≤ base + 0.05`. Do not start Step 6 (GRPO) until the gate passes. Treat it like a unit-test guard.

If G1 fails, in priority order:
1. **Re-run SFT with `NUM_EPOCH=5`** — at this env's 2-epoch budget, SFT is undertrained; the prior captured run only crossed base around iter_251 (2 epochs, +43%) and peaked at iter_503 (4 epochs, +57%). 2 epochs is borderline; 5 epochs gives the curve room.
2. **Inspect `sft_trajectory.parquet`** — row count + a sample row's `steps` field. If the new trajectory-centric pipeline (commit `e35f9618` and later) is dropping turns or rendering them differently from the teacher's rollout config (especially `full_history_size`), SFT learns a phantom prompt distribution. Check that the SFT samples' rendered prompt matches what the rollout config produces at eval time.
3. **Bypass SFT** — if neither (1) nor (2) recovers, set `HF_CKPT=Qwen/Qwen3-VL-2B-Instruct` (base model) and run GRPO directly. SFT is a warm-start, not a load-bearing gate; better to GRPO from base than from a regressed SFT.

If G2 doesn't hit ≥ 50% within 3 iters, the anchor / random ratio likely needs to bend toward more anchor (e.g., 75/25); see § Notes.

## Snapshot template

Concrete copy-paste body for `logs/<commit-ts>_<commit>/<run_id>.md` (this recipe's stages: `teacher_rollout` → `sft` → `grpo`). Schema rules and the "skip mid-run failed-restart URLs" semantics are in [`devs/exps/train/AGENTS.md`](/devs/exps/train/AGENTS.md#snapshot--progress-log).

```markdown
# lite.osworld · qwen3_vl_2b @ <commit-ts>_<commit> · <run_id>

- **Recipe**: `devs/exps/train/lite.osworld/qwen3_vl_2b/AGENTS.md`
- **Commit**: `<short sha>` — `<subject>`
- **Host / GPUs**: `<hostname>` / `<e.g. 4,5>`
- **Container**: `lite.slime-train-lite.osworld-qwen3_vl_2b` (started <timestamp>)
- **wandb** (skip mid-run failed-restart URLs):
  - `sft`: `<url or "n/a">`
  - `grpo`: `<url or "n/a">`
- **Artifacts**: `.exps/train/lite.osworld/qwen3_vl_2b/<commit-ts>_<commit>/<run_id>/`
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
| base `Qwen/Qwen3-VL-2B-Instruct` | `lite.osworld` eval | <nf>/<nt> | <mer> | — |
| sft iter_<n>                     | `lite.osworld` eval | <nf>/<nt> | <mer> | +/- <pp>pp |
| grpo iter_<n>                    | `lite.osworld` eval | <nf>/<nt> | <mer> | +/- <pp>pp |

Add rows incrementally — base after Step 4.1, each `sft iter_<n>` after Step 4.2, each `grpo iter_<n>` after Step 7. Sort within stage by iter ascending. Mark intentionally-skipped iters `⚠️ not eval'd`.

## Highlights

- short interpretive bullets — call out partial stages, surprising deltas (e.g. SFT helps but RL doesn't), failure modes (response_len drift, mixed-group collapse), key recipe knobs that mattered.
```

---

## Notes / tuning

- **Why 50/50?** All-anchor would over-fit the policy to the easy slice and lose exploration. All-random has the zero-gradient problem (≥70% all-zero groups on lite.osworld). 50/50 is the empirical sweet spot before tuning.
- **Teacher choice.** 8B teacher gives ~32% success rate on noise=true → ≈160 success-filtered trajectories from 512 rollouts. The companion `qwen3_5_2b` recipe reads the **same** `teacher_rollout/` — never re-roll out for the 5_2b student.
- **Length-drift watch.** Policy can drift toward longer responses on lite.osworld (observed: response_len mean grows iter-over-iter, truncated_ratio rises in lockstep). If you see mean response_len rise > 20% over the first 5 iters, lower `--lr` (e.g. 1e-6 → 5e-7), shrink `rollout-max-response-len`, or stop early at the best in-training eval iter.
- **Ablation sweep.** Anchor-only / random-only / 50-50 / 75-25 are the natural four-point sweep to validate the 50/50 choice on this env.
