# androidworld · mai_ui_2b @ 2026-05-24T14-57_85afac84 · run_0

- **Recipe**: `devs/exps/train/androidworld/mai_ui_2b/AGENTS.md`
- **Variant**: `run_0` — pure GRPO from base (no SFT)
- **Commit**: `85afac84` — docker_reap: use ``docker ps -aq`` so Exited zombies are reaped too
- **Host / GPUs**: `gpublaze` / `6,7` (mapped to container's `0,1`)
- **Container**: `lite.slime-train-androidworld-qwen3_vl_2b`
- **TEACHER_ROLLOUT_DIR**: n/a (run_0 — pure GRPO from base, no SFT step)
- **wandb** (skip mid-run failed-restart URLs):
  - `grpo`: `https://wandb.ai/asap-zzhou/cua-lite-rl/runs/t365jjsa`
- **Artifacts**: `.exps/train/androidworld/mai_ui_2b/2026-05-24T14-57_85afac84/run_0/`
- **Started**: 2026-05-24 14:56 PDT
- **Last updated**: 2026-05-24 14:58 PDT
- **Notes**:
  - env-server: `http://169.229.219.180:30100` (rootless docker disable-host-loopback workaround — must use host external IP, not localhost)
  - K=20 for `_MAX_RESETS_PER_CONTAINER` (lowered from 50 this session)

## Stages run

| Stage | Status | Output |
|---|---|---|
| data | `done` | train.parquet (86 tasks), eval.parquet (86 tasks) |
| grpo | `in_progress` | (TBD — iter ckpts under `grpo/hf/iter_*`) |

## Eval results

| Ckpt | Eval set | Finished | Mean episode return | Δ vs base |
|---|---|---|---|---|
| base `Tongyi-MAI/MAI-UI-2B` | `androidworld` eval | — | — | — |
| grpo iter_4  | `androidworld` eval | — | — | — |
| grpo iter_9  | `androidworld` eval | — | — | — |
| grpo iter_14 | `androidworld` eval | — | — | — |
| grpo iter_19 | `androidworld` eval | — | — | — |

## Highlights

- (TBD — fill in after Step 6 + Step 7)
