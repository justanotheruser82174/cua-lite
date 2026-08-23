# Lite.OSWorld: 8B-Teacher Distill SFT → GRPO (Qwen3.5-2B)

> **Slime v0.3.0 recipe.** Use `run_sft.sh` / `run_grpo.sh` /
> `run_reinforce.sh` with `MODEL_ID=Qwen/Qwen3.5-2B`; the family is
> auto-derived and the shared v0.3.0 image supports both Qwen3.5 and Qwen3-VL
> without container-local package mutation.


> **Read [`devs/exps/train/AGENTS.md`](/devs/exps/train/AGENTS.md) first** for layout, `TRAIN_RUN_ID` contract, Slime v0.3.0 container/`SESSION_ID` rules, GPU discipline, snapshot template, and the agent workflow checklist.
>
> **Env-server prereq.** An env-server must be reachable and expose `lite.osworld` before launch — see [`devs/exps/train/AGENTS.md#env-server-prerequisite`](/devs/exps/train/AGENTS.md#env-server-prerequisite).

End-to-end recipe that uses **Qwen3-VL-8B-Instruct as the teacher** (cross-family distill) and **Qwen/Qwen3.5-2B as the student**:

1. Teacher rollout — **reuse** the rollout produced by the companion [`../qwen3_vl_2b/AGENTS.md#step-1`](/devs/exps/train/lite.osworld/qwen3_vl_2b/AGENTS.md#step-1--teacher-rollout-8b-with-rollout-config) — 512 train tasks with `noise=true`. **Never re-roll out the 8B teacher just for qwen3.5** (same data either way; re-rolling is wasted compute).
2. SFT student = qwen3.5-2B with `qwen3_5/compact/lite.osworld.yaml` (`history_n=1`) on success-filtered teacher trajectories — **converted at export time** from the qwen3-VL trajectory format to qwen3.5's via the qwen3_5 export config.
3. GRPO student = warm-started from the best SFT ckpt, trained on a **50/50 mixed prompt pool** (same anchor / random construction as the qwen3_vl recipe).

**Why a distinct qwen3.5 path?** Qwen3.5 has native GatedDeltaNet (GDN) that needs `--qkv-format bshd` and fixed `--micro-batch-size` (no THD packing) — see [`run_grpo.sh`](/scripts/train/run_grpo.sh) (formerly `run_grpo_qwen3_5.sh`, merged in v0.3.0) header for the rationale. Since v0.3.0 the same script serves both families (`MODEL_FAMILY` auto-derived from `MODEL_ID`); no env mutation.

> **Cross-family distill caveat.** Qwen3.5 GDN is a different architecture from Qwen3-VL; even though `export_sft` converts the trajectory format via the qwen3_5 RL config, the student still has to learn the teacher's action distribution from scratch (no shared visual tokenizer / attention layout). Expect a slightly slower SFT ramp than the qwen3_vl peer at the same teacher rollout count.

## ⚠️ Working directory + paths

> Host commands run from the repo root; container commands run inside the Slime container at `/workspaces/cua-lite`. See [`docs/slime.md`](/docs/slime.md) and [`devs/exps/train/AGENTS.md`](/devs/exps/train/AGENTS.md) for setup.

```bash
export TRAIN_RUN_ID="run_0"                                         # fixed string per Run id contract
COMMIT=$(git rev-parse --short HEAD)
COMMIT_TS=$(git log -1 --date=format:'%Y-%m-%dT%H-%M' --format=%cd HEAD)
BASE_HOST=.exps/train/lite.osworld/qwen3_5_2b/${COMMIT_TS}_${COMMIT}/${TRAIN_RUN_ID}
BASE_CTR=/workspaces/cua-lite/${BASE_HOST}                          # same path inside the slime container

# Slime container session id
export SESSION_ID=train-lite.osworld-qwen3_5_2b
```

`BASE_CTR` is what container-side commands pass to `--log-root` / `SAVE_DIR` / `SAVE_HF_DIR` / `--model-path` / `-o`. Every step in this recipe runs inside the slime container; env containers spawn on the env-server's host (reached via `CUA_LITE_ENV_SERVER_URL`), not in this container — see [`docs/envs.md#env-server`](/docs/envs.md#env-server). `BASE_HOST` is the same physical path expressed as a host-relative string; the recipe only uses it for the post-rollout coverage `find` (`find` is a host shell builtin, no python deps).

---

## Step 1 — Teacher rollout (REUSE companion qwen3_vl_2b run)

**Do not run this step in this recipe.** Run [`../qwen3_vl_2b/AGENTS.md#step-1`](/devs/exps/train/lite.osworld/qwen3_vl_2b/AGENTS.md#step-1--teacher-rollout-8b-with-rollout-config) **first** (in the qwen3-VL container, not this one), then point this recipe at its output.

```bash
# Set after the companion qwen3_vl_2b recipe has produced a teacher_rollout/ dir.
# Convention: same TRAIN_RUN_ID across companion recipes for a given (commit-ts, commit).
COMPANION_BASE=/workspaces/cua-lite/.exps/train/lite.osworld/qwen3_vl_2b/${COMMIT_TS}_${COMMIT}/${TRAIN_RUN_ID}
TEACHER_ROLLOUT=${COMPANION_BASE}/teacher_rollout

# Sanity check on host (BASE_HOST mirror of COMPANION_BASE)
COMPANION_BASE_HOST=.exps/train/lite.osworld/qwen3_vl_2b/${COMMIT_TS}_${COMMIT}/${TRAIN_RUN_ID}
echo "Teacher rollout count: $(find ${COMPANION_BASE_HOST}/teacher_rollout -name summary.json 2>/dev/null | wc -l) / 512"
```

If the companion's `teacher_rollout/` doesn't exist or has < 90% coverage, go finish that recipe's Step 1 before continuing here. The companion uses Qwen3-VL-8B-Instruct + `noise=true` (~32% success rate on lite.osworld, ≈160 success-filtered trajectories from 512).

---

## Step 2 — Export SFT parquet (success-only, qwen3.5 format)

Use `qwen3_5/compact/lite.osworld.yaml` for export — this **converts** the teacher's qwen3-VL trajectory format into qwen3.5's at parquet write time. The output parquet is qwen3.5-native; no further conversion needed at SFT time.

```bash
docker exec -w /workspaces/cua-lite lite.slime-${SESSION_ID} \
  python -m lite.train.export.export_sft \
    --config /workspaces/cua-lite/scripts/configs/qwen3_5/compact/lite.osworld.yaml \
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
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_5/compact/lite.osworld.yaml \
  SAVE_DIR=${BASE_CTR}/sft/megatron \
  SAVE_HF_DIR=${BASE_CTR}/sft/hf/iter_{rollout_id} \
  bash /workspaces/cua-lite/scripts/train/run_sft.sh
```

`NUM_EPOCH=2` — unified across `<env>/<recipe>` combinations. **Caveat for this env**: the 2026-04-30 run at NUM_EPOCH=2 produced a +3.13pp SFT lift (right at the G1 floor). lite.osworld is reward-sparse and qwen3_5 SFT data is small after cross-family filtering — when SFT undershoots G1, **re-run with `NUM_EPOCH=5`**. The companion `lite.osworld/qwen3_vl_2b` recipe regressed at NUM_EPOCH=2 (−4.7pp) but the captured 5-epoch run on the same env peaked at +6.25pp (4 epochs) — the longer ramp is needed on this env. `DUMP` stays OFF (default).

---

## Step 4 — 🛑 Eval base + SFT (HARD GATE)

This is a **hard pipeline gate**, not a sanity check. Run a deterministic eval of base + the latest SFT ckpt, parse `mean_episode_return` from each `summary.json`, and **stop the pipeline if `sft ≤ base + 0.03`** (a `≥ +3pp` lift is the minimum acceptable for this env / model combo). Captured 2026-04-30 result was exactly +3.13pp — borderline but enough; anything below this means SFT didn't really help and GRPO has nothing to amplify. See § Acceptance gates for recovery options if it fails.

Eval is **resume-safe** — re-running the same `--log-root` skips finished tasks. Loop 3× per ckpt to guarantee full 128-task coverage. Runs **inside the slime container**.

```bash
SLUG=Qwen_Qwen3.5-2B

# 4.1 base — 3 retries
for attempt in 1 2 3; do
  echo "=== eval base attempt ${attempt}/3 ==="
  docker exec -w /workspaces/cua-lite -e CUDA_VISIBLE_DEVICES=0,1 lite.slime-${SESSION_ID} \
    python scripts/rollout.py \
      --model-id Qwen/Qwen3.5-2B --env-id lite.osworld \
      --splits eval --sample 128 --concurrency 16 \
      --filter "lambda m: not m.others.get('exclude_reason')" \
      --sampling-kwargs '{"temperature": 0, "top_p": 1}' \
      --config-path scripts/configs/qwen3_5/compact/lite.osworld.yaml \
      --log-root ${BASE_CTR}/eval_base/${SLUG}
done

# 4.2 SFT — by default score the LATEST SFT ckpt only (iter_N = highest iter saved). Replace iter_N below with that exact value. Optionally repeat the block for intermediate ckpts if you want the full epoch curve (extra eval cost).
for attempt in 1 2 3; do
  echo "=== eval sft/iter_N attempt ${attempt}/3 ==="
  docker exec -w /workspaces/cua-lite -e CUDA_VISIBLE_DEVICES=0,1 lite.slime-${SESSION_ID} \
    python scripts/rollout.py \
      --model-id Qwen/Qwen3.5-2B --env-id lite.osworld \
      --model-path ${BASE_CTR}/sft/hf/iter_N \
      --splits eval --sample 128 --concurrency 16 \
      --filter "lambda m: not m.others.get('exclude_reason')" \
      --sampling-kwargs '{"temperature": 0, "top_p": 1}' \
      --config-path scripts/configs/qwen3_5/compact/lite.osworld.yaml \
      --log-root ${BASE_CTR}/eval_sft/iter_N/${SLUG}
done
```

Pick the `iter_N` with the highest `mean_episode_return`; that becomes the GRPO warm start.

---

## Step 5 — Build the 50/50 GRPO prompt pool

Same construction as the [qwen3_vl recipe Step 5](/devs/exps/train/lite.osworld/qwen3_vl_2b/AGENTS.md#step-5--build-the-5050-grpo-prompt-pool): derive the success task_id set from the teacher rollout log tree, inline into the `--filter` lambda. `export_tasks` accepts only `--env-id / --split / --head / --sample / --seed / --filter / --env-kwargs / -o` — no `--paths` / `--exclude-paths`.

```bash

# 5.0 build the anchor task_id set (success-filtered teacher rollouts)
ANCHOR_TASK_IDS=$(python3 -c "
import json, glob, os
ids = set()
for p in glob.glob('${COMPANION_BASE_HOST}/teacher_rollout/*/sample_*/summary.json'):
    if json.load(open(p)).get('episode_return', 0) >= 1.0:
        ids.add(os.path.basename(os.path.dirname(os.path.dirname(p))))
print(repr(ids))
")
echo "Anchor pool size: $(echo $ANCHOR_TASK_IDS | python3 -c 'import ast, sys; print(len(ast.literal_eval(sys.stdin.read())))')"

# 5.1 anchor half (success task_ids only)
# --env-kwargs writes {"noise": true} into every row's metadata.env_kwargs, so the
# GRPO train rollouts get the teacher's noise injection. 5.4 (eval) omits it.
docker exec -w /workspaces/cua-lite lite.slime-${SESSION_ID} \
  python -m lite.train.export.export_tasks --env-id lite.osworld --split train \
    --filter "lambda m: not m.others.get('exclude_reason') and m.others.get('task_id') in ${ANCHOR_TASK_IDS}" \
    --env-kwargs '{"noise": true}' \
    -o ${BASE_CTR}/grpo_pool_anchor.parquet

# 5.2 random half (full train minus anchor task_ids)
N=$(docker exec lite.slime-${SESSION_ID} \
      python -c "import pyarrow.parquet as pq; print(pq.read_table('${BASE_CTR}/grpo_pool_anchor.parquet').num_rows)")
docker exec -w /workspaces/cua-lite lite.slime-${SESSION_ID} \
  python -m lite.train.export.export_tasks --env-id lite.osworld --split train \
    --filter "lambda m: not m.others.get('exclude_reason') and m.others.get('task_id') not in ${ANCHOR_TASK_IDS}" \
    --sample ${N} \
    --env-kwargs '{"noise": true}' \
    -o ${BASE_CTR}/grpo_pool_random.parquet

# 5.3 merge into the final 50/50 pool
docker exec -w /workspaces/cua-lite lite.slime-${SESSION_ID} \
  python -m lite.data.merge \
    -i ${BASE_CTR}/grpo_pool_anchor.parquet \
       ${BASE_CTR}/grpo_pool_random.parquet \
    -o ${BASE_CTR}/grpo_pool_50_50.parquet

# 5.4 eval pool — --sample 128 holdout from eval split
docker exec -w /workspaces/cua-lite lite.slime-${SESSION_ID} \
  python -m lite.train.export.export_tasks --env-id lite.osworld --split eval --sample 128 \
    --filter "lambda m: not m.others.get('exclude_reason')" \
    -o ${BASE_CTR}/eval.parquet
```

---

## Step 6 — GRPO (warm-started from SFT)

v0.3.0: `run_grpo.sh` with `MODEL_ID=Qwen/Qwen3.5-2B` auto-selects the qwen3.5 path (`--qkv-format bshd` + fixed micro-batch-size for native GatedDeltaNet correctness). ASYNC mode runs rollout (1 GPU, sglang) and train (1 GPU, megatron) concurrently.

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=1 NUM_ROLLOUT_GPUS=1 ASYNC=1 \
  MODEL_ID=Qwen/Qwen3.5-2B \
  HF_CKPT=${BASE_CTR}/sft/hf/iter_N \
  SAVE_DIR=${BASE_CTR}/grpo/megatron \
  SAVE_HF_DIR=${BASE_CTR}/grpo/hf/iter_{rollout_id} \
  ENV_ID=lite.osworld \
  PROMPT_DATA=${BASE_CTR}/grpo_pool_50_50.parquet \
  EVAL_PROMPT_DATA=${BASE_CTR}/eval.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_5/compact/lite.osworld.yaml \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh
```

Notes:
- The GRPO config uses `history_n=1`, matching the SFT student protocol — student sees the same context shape at SFT, RL, and eval time.
- Train rollouts get the same noise injection as the teacher because Step 5.1/5.2 exported the pool with `--env-kwargs '{"noise": true}'`: each row's `metadata.env_kwargs` deep-merges over the yaml at rollout (`prompt_data > args > yaml`). `eval.parquet` (Step 5.4) omits it, so eval stays deterministic at the env default `noise: false`. No yaml carries a `noise` key.
- The qwen3_5 path uses `--qkv-format bshd` and `MBS=1` (default; bump only if VRAM allows). Don't try `--use-dynamic-batch-size` — it bypasses the GDN-correct path.
- Slime saves a HF ckpt every `--save-interval` step (5 by default → `iter_4 / iter_9 / ...`). Eval is reported at the same cadence.

---

## Step 7 — Eval GRPO ckpts (≥ 3 retries per ckpt)

Runs **inside the slime container**.

```bash
# Replace iter_M with each saved GRPO iter (4, 9, 14, ...) to track the curve.
for attempt in 1 2 3; do
  echo "=== eval grpo/iter_M attempt ${attempt}/3 ==="
  docker exec -w /workspaces/cua-lite -e CUDA_VISIBLE_DEVICES=0,1 lite.slime-${SESSION_ID} \
    python scripts/rollout.py \
      --model-id Qwen/Qwen3.5-2B --env-id lite.osworld \
      --model-path ${BASE_CTR}/grpo/hf/iter_M \
      --splits eval --sample 128 --concurrency 16 \
      --filter "lambda m: not m.others.get('exclude_reason')" \
      --sampling-kwargs '{"temperature": 0, "top_p": 1}' \
      --config-path scripts/configs/qwen3_5/compact/lite.osworld.yaml \
      --log-root ${BASE_CTR}/eval_grpo/iter_M/${SLUG}
done
```

In-training eval (the `eval N` lines in slime stdout / wandb) tracks the same metric automatically.

---

## Acceptance gates

| Gate | What | Target | Hard stop? | Captured reference |
|---|---|---|---|---|
| **G1** | **SFT eval > base eval** | **≥ +3pp on this recipe's 128-task subset (T=0)** | **🛑 yes — pipeline must stop here if it fails** | 2026-04-30 run: 4.69% → 7.81% = **+3.13pp** (right at the line) |
| G2 | GRPO mixed-group ratio (early iters) | ≥ 50% (vs ~30% with all-random pool) | no | observed 27–47% on the 2026-04-30 run |
| G3 | GRPO eval > SFT eval | ≥ +3pp at peak iter | no — but if G3 fails for 2 consecutive ckpts, abort GRPO; you're spending GPU for nothing | 2026-04-30 run: **0.0pp** (all 4 grpo ckpts tied at 7.81% with SFT) |
| G4 | Response-len mean stable across iters | within ±15% of SFT-iter eval mean | no | — |

**G1 enforcement** — Step 4 is a hard gate. Pipeline.sh must run Step 4 between SFT and GRPO and exit non-zero if `sft ≤ base + 0.03`. Don't start Step 6 until the gate passes.

If G1 fails, in priority order:
1. **Re-run SFT with `NUM_EPOCH=5`** — qwen3_5 SFT data is sparser (cross-family text rendering filters out trajectories that don't translate cleanly); 2 epochs may undertrain.
2. **Inspect `sft_trajectory.parquet`** — confirm the qwen3_5 chat template is being applied correctly during export. A regression in the export pipeline can leave training data in a format the rl-config rollout never sees.
3. **Bypass SFT** — set `HF_CKPT=Qwen/Qwen3.5-2B` and run GRPO directly. Note this loses the +3pp SFT lift but at least gives GRPO a clean starting point.

If G3 fails (GRPO ties or drops below SFT for 2 consecutive ckpts): on this env / model, GRPO has no reward gradient to follow. Kill the run early and save GPU for higher-yield experiments, or sweep the anchor-pool ratio to 75/25 to inject more positive-reward trajectories into the GRPO mix.

If G2 doesn't hit ≥ 50% within 3 iters, bend the anchor / random ratio toward more anchor (75/25); see § Notes.

## Snapshot template

Concrete copy-paste body for `logs/<commit-ts>_<commit>/<run_id>.md` (this recipe's stages: `teacher_rollout` → `sft` → `grpo`). Schema rules and the "skip mid-run failed-restart URLs" semantics are in [`devs/exps/train/AGENTS.md`](/devs/exps/train/AGENTS.md#snapshot--progress-log).

```markdown
# lite.osworld · qwen3_5_2b @ <commit-ts>_<commit> · <run_id>

- **Recipe**: `devs/exps/train/lite.osworld/qwen3_5_2b/AGENTS.md`
- **Commit**: `<short sha>` — `<subject>`
- **Host / GPUs**: `<hostname>` / `<e.g. 4,5>`
- **Container**: `lite.slime-train-lite.osworld-qwen3_5_2b` (started <timestamp>)
- **wandb** (skip mid-run failed-restart URLs):
  - `sft`: `<url or "n/a">`
  - `grpo`: `<url or "n/a">`
- **Artifacts**: `.exps/train/lite.osworld/qwen3_5_2b/<commit-ts>_<commit>/<run_id>/`
- **Started**: `<YYYY-MM-DD HH:MM TZ>`
- **Last updated**: `<YYYY-MM-DD HH:MM TZ>` ← bump on every edit
- **Notes**: anything non-default — non-default lr / batch-size / `MBS`, container-poisoning concerns, mid-run crashes + resumes

## Stages run

| Stage | Status | Output |
|---|---|---|
| teacher_rollout | `reused from <companion qwen3_vl_2b run_id>` | (no rollout in this recipe) |
| sft             | `not started` / `in_progress` / `done iter_<best>` / `failed` | `best ckpt eval <acc>% (vs base <acc>%)` once Step 4 has the numbers |
| grpo            | `not started` / `in_progress at iter_<n>` / `done iter_<best>` / `failed` | `best ckpt eval <acc>% (vs sft <acc>%)` once Step 7 has the numbers |

## Eval results

| Ckpt | Eval set | Finished | Mean episode return | Δ vs base |
|---|---|---|---|---|
| base `Qwen/Qwen3.5-2B` | `lite.osworld` eval | <nf>/<nt> | <mer> | — |
| sft iter_<n>          | `lite.osworld` eval | <nf>/<nt> | <mer> | +/- <pp>pp |
| grpo iter_<n>         | `lite.osworld` eval | <nf>/<nt> | <mer> | +/- <pp>pp |

Add rows incrementally — base after Step 4.1, each `sft iter_<n>` after Step 4.2, each `grpo iter_<n>` after Step 7. Sort within stage by iter ascending. Mark intentionally-skipped iters `⚠️ not eval'd`.

## Highlights

- short interpretive bullets — Qwen3.5 GDN-specific quirks, length-drift, mixed-group collapse, container-poisoning incidents.
```

---

## Notes / tuning

- **Qwen3.5 GDN specifics.** Native GatedDeltaNet means slower per-step training (no THD packing) but stable convergence. Don't add `--use-dynamic-batch-size` — it bypasses the GDN-correct path.
- **~~Container poisoning~~ (obsolete since slime v0.3.0).** The old qwen3_5 scripts' `pip install` mutation is gone; one container serves both model families. See the OBSOLETE note in [`devs/exps/train/AGENTS.md`](/devs/exps/train/AGENTS.md).
- **Cross-family teacher reuse.** Qwen3-VL-8B teacher trajectories drive both this recipe and the [`../qwen3_vl_2b/AGENTS.md`](/devs/exps/train/lite.osworld/qwen3_vl_2b/AGENTS.md) recipe. The same physical `teacher_rollout/` directory is read by both; `export_sft` does the per-recipe format conversion. Running the 8B teacher rollout twice is forbidden waste.
- **Length-drift watch.** Policy can drift toward longer responses iter-over-iter. If you see mean response_len rise > 20% over the first 5 iters, lower `--lr` (e.g. 1e-6 → 5e-7), shrink `rollout-max-response-len`, or stop early at the best in-training eval iter.
- **Ablation sweep.** Anchor-only / random-only / 50-50 / 75-25 are the natural four-point sweep to validate the 50/50 choice on this env.
