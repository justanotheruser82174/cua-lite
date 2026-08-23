# Slime Container

The training container uses the upstream [Slime](https://github.com/cua-lite/slime) image ([`slimerl/slime:v0.3.0`](https://hub.docker.com/r/slimerl/slime)) — no custom Dockerfile, no local image build.

## Quick Start

### 1. Pull submodules

Brings `slime/` into the worktree so [`scripts/train/slime/init.sh`](/scripts/train/slime/init.sh) can `pip install -e` it.

```bash
git submodule update --init --recursive
```

### 2. Make sure an env-server is reachable (Optional: RL only)

Skip for pure SFT (no online rollouts). Otherwise the container reaches its envs over HTTP.
See [docs/envs.md#installation](/docs/envs.md#installation) for env installation.
See [docs/envs.md#env-server](/docs/envs.md#env-server) for env-server setup.

### 3. Launch the container

[`scripts/train/slime/launch.sh`](/scripts/train/slime/launch.sh) forwards exactly these host env vars into the container — set them on the host first:

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Gated-model downloads at training start |
| `WANDB_API_KEY` | Weights & Biases logging |
| `CUA_LITE_ENV_SERVER_URL` | Env-server URL — local or remote. Read by `LiteEnvClient` during rollout |
| `CUA_LITE_ENV_SERVER_TOKEN` | Bearer token for that server (any string in passthrough mode) |
| `SESSION_ID` | Launch-batch tag (auto-picked if unset); propagated to env-server for cleanup scoping |

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/train/slime/launch.sh           # specific GPUs
SESSION_ID=train-osworld-qwen3_vl_2b bash scripts/train/slime/launch.sh   # explicit name → lite.slime-<SESSION_ID>
```

`CUDA_VISIBLE_DEVICES` is required on shared hosts. On a dedicated host only,
set `CUA_LITE_SLIME_ALL_VISIBLE_GPUS=1` to launch with Docker's `--gpus all`.

### 4. Run training

| Method | Guide |
|--------|-------|
| GRPO (online RL) | [docs/grpo.md](/docs/grpo.md) |
| REINFORCE / filtered-BC (online RL) | [docs/examples/reinforce.md](/docs/examples/reinforce.md) |
| DAgger (online teacher-forcing) | [docs/examples/dagger.md](/docs/examples/dagger.md) |
| SFT (offline) | [docs/sft.md](/docs/sft.md) |

## Notes

### Cleanup if stale

If preflight or a new launch sees stale env-server sessions for the same run,
bulk-close that run's sessions before retrying.

```bash
curl -X DELETE \
  "${CUA_LITE_ENV_SERVER_URL}/instances?session_id=${SESSION_ID}&env_id=${ENV_ID}" \
  -H "Authorization: Bearer ${CUA_LITE_ENV_SERVER_TOKEN}"
```

If bulk close does not clear the problem, restart the env-server or ask the
env-node operator to run the relevant `cleanup.sh`.

### `SESSION_ID` and env-server cleanup scoping

`SESSION_ID` scopes env-server cleanup for a training run. Set it explicitly
when you want a stable run name; otherwise `scripts/train/slime/launch.sh` picks one.
