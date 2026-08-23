# REINFORCE / Filtered-BC Training for CUA-Lite Agents

> **Status**: Implemented and merged into `zzh`. See [docs/examples/reinforce.md](/docs/examples/reinforce.md) for usage instructions.
> This file is retained as implementation history; use
> `lite/train/rollout/{core,grpo,reinforce}.py` and the example docs as the
> source of truth for current code paths.
>
> **Prerequisites**: Follow [docs/slime.md](/docs/slime.md) to set up and run inside a Slime Docker container. Training requires the Slime container's infrastructure (Megatron-LM, SGLang, Ray).
>
> **GPU usage**: Use at least 4 GPUs. Same guidelines as [devs/train/grpo.md](/devs/train/grpo.md).

REINFORCE-style algorithm inspired by WebGym's "online filtered behavior cloning": rollout with current policy → filter to successful trajectories only → SFT on those → repeat. Mathematically equivalent to REINFORCE with binary reward and no baseline.

**Reference implementation:**
- `${CUA_LITE_REFERENCES_ROOT}/webgym` — `main` branch is the unmodified upstream reference (read-only). If you need to modify WebGym (patch action parsing, fix bugs, add features), create a `cua-lite` branch off `main` and work there. Keep `main` clean so we can diff against upstream and pull updates.
- **Paper:** https://arxiv.org/html/2601.02439v3 — *WebGym: Scaling Training Environments for Visual Web Agents with Realistic Tasks* (Bai et al., Microsoft/UIUC/CMU)

This task has two parts: (1) refactor the existing `train/` directory structure to share rollout code between GRPO and REINFORCE, and (2) implement the new REINFORCE algorithm.

## Architecture

### Target directory structure

```
lite/train/
├── rollout/
│   ├── grpo.py          # GRPO-specific: convert_samples_to_train_data + re-exports
│   ├── reinforce.py     # REINFORCE-specific: convert_samples_to_train_data + re-exports
│   ├── sft.py
│   └── engine.py        # shared rollout infra
└── export/
    ├── export_tasks.py
    └── export_sft.py
lite/data/               # split/merge are generic data CLIs (not train-specific)
├── split.py
└── merge.py

scripts/train/
├── rl/                      # renamed from grpo/
│   ├── run_grpo.sh          # moved from grpo/
│   ├── run_reinforce.sh     # new
│   ├── GRPO.md              # moved from grpo/README.md
│   ├── REINFORCE.md         # new
│   └── configs/             # shared env configs (moved from grpo/configs/)
│       ├── osworld.yaml
│       ├── webgym.yaml
│       └── androidworld.yaml
├── sft/                     # unchanged
└── utils/                   # unchanged
```

### Key renames

| Before | After |
|--------|-------|
| `lite/train/data_utils/` | `lite/train/utils/data/` |
| `scripts/train/grpo/` | `scripts/train/rl/` |
| `scripts/train/grpo/README.md` | `scripts/train/rl/GRPO.md` |

## Key Design Decisions

1. **Shared rollout code**: `generate()`, `generate_rollout()`, `abort()`, and all helpers are identical between GRPO and REINFORCE — extracted into `lite/train/rollout/core/engine.py`. Only `convert_samples_to_train_data()` differs (GRPO does group advantage normalization; REINFORCE does trajectory filtering).

2. **SFT loss for filtered BC**: For binary-reward filtered BC, SFT NLL loss = REINFORCE with reward=1. Use `--loss-type sft_loss` + `--disable-compute-advantages-and-returns` + `--calculate-per-token-loss`. Confirmed compatible with `slime/train.py` (online RL mode with SGLang engines).

3. **No group sampling**: GRPO needs `n_samples_per_prompt=8` for group normalization. REINFORCE uses `n_samples_per_prompt=1` — each prompt gets one rollout, failed ones are filtered out.

4. **Shared env configs**: GRPO and REINFORCE use identical env configs (same `agent_id`, `resolution`, `max_steps`). Algorithm-specific params like `reward_threshold` are added to the shared YAMLs (GRPO ignores them). Slime merges YAML keys into `args` via `setattr` (`arguments.py:1753`), so `getattr(args, "reward_threshold", 0.0)` reads it.

5. **Large rollout batch to compensate for filtering**: With `n_samples_per_prompt=1`, effective training batch = `ROLLOUT_BATCH_SIZE * nonzero_return_rate`. Default `ROLLOUT_BATCH_SIZE=64`; at a 25% nonzero-return rate, ~16 samples survive filtering.

---

## Implementation (in order)

**IMPORTANT — Global search-replace**: Steps 1-2 rename existing paths. Each rename requires a **repo-wide grep** to find and update ALL references — including test files (`patch()` paths and imports), docstrings, `.md` documentation, comments, and CLI examples. Do not rely on the file lists below being exhaustive; always grep to catch everything. After each step, run `uv run pytest` to confirm nothing breaks before proceeding.

### Step 1: Rename `lite/train/data_utils/` → `lite/train/utils/data/`

Create `lite/train/utils/__init__.py` and `lite/train/utils/data/__init__.py`, move all `.py` files from `data_utils/` into `utils/data/`, then delete the old `data_utils/` directory.

CLI module path changes: `python -m lite.train.data_utils.X` → `python -m lite.train.utils.data.X`

**Grep patterns**: `lite.train.data_utils`, `lite/train/data_utils`

Grep the entire repo and update all hits — docstrings in the moved files, CLI examples in READMEs, comments in eval scripts, etc.

**Verify**: `grep -rn 'data_utils' --include='*.py' --include='*.sh' --include='*.md' --include='*.yaml' .` → zero hits (excluding `__pycache__`).

### Step 2: Rename `scripts/train/grpo/` → `scripts/train/rl/`

Move all files. Rename `README.md` → `GRPO.md`.

**Grep patterns**: `scripts/train/grpo`, `train/grpo/`

Grep the entire repo and update all hits — shell script paths, doc links, comments, etc.

**Verify**: `grep -rn 'train/grpo' --include='*.py' --include='*.sh' --include='*.md' --include='*.yaml' .` → zero hits.

### Step 3: Extract shared rollout code into `lite/train/rollout/core/engine.py`

Extract from `lite/train/rollout/grpo.py` — everything except `convert_samples_to_train_data()`:

**Module-level state + helpers:**
- `_active_envs`, `register_env`, `unregister_env`, `AbortError`
- `_make_turn_sample()`, `_build_turn_samples()`, `_empty_sample()`, `_dummy_sample()`

**Rollout pipeline:**
- `generate()` — async multi-turn rollout with SGLang
- `abort()` — env cleanup + SGLang abort
- `generate_rollout_async()` — async loop collecting samples until `target_data_size`
- `_eval_rollout()` — evaluation rollout wrapper
- `generate_rollout()` — entry point dispatching train vs eval

**Shared `convert_samples_to_train_data` helpers** (avoid ~100 lines duplication):

```python
def regroup_trajectories(samples: list[Sample]) -> list[list[Sample]]:
    """Reconstruct per-trajectory grouping from flattened Sample list.
    Regroups by (group_index, index), sorts turns by turn_idx."""

def filter_errored_rollouts(
    rollouts: list[list[Sample]],
) -> tuple[list[list[Sample]], int]:
    """Drop rollouts where ALL turns have empty tokens.
    Returns (valid_rollouts, n_errored)."""

def get_episode_returns(rollouts: list[list[Sample]]) -> list[float]:
    """Extract episode_return from last turn's metadata for each rollout."""

def flatten_and_align(rollouts: list[list[Sample]], args) -> dict:
    """Flatten turns → fix response_length/log_prob alignment →
    batch-size alignment (pad with _dummy_sample if needed) →
    assemble train_data dict."""
```

After extraction, `lite/train/rollout/grpo.py` keeps only `convert_samples_to_train_data()` + re-exports of `generate` and `generate_rollout` (Slime resolves them via `--*-function-path`).

**Test + import updates (critical!)**: Moving functions from `lite/train/rollout/grpo.py` to `lite/train/rollout/core/engine.py` changes the module where names are looked up. This affects `patch()` targets in tests (must target the module where the name lives) and `from ... import` statements. **Grep pattern**: `lite\.train\.rollout\.grpo` — update all hits in `tests/`, docstrings, and any other Python files.

**Verify**: `uv run pytest tests/train/ -v` → all existing tests pass.

### Step 4: Implement `lite/train/rollout/reinforce.py`

Re-export `generate` and `generate_rollout` from `lite.train.rollout.core`.

New `convert_samples_to_train_data(args, samples) -> dict`:

1. `regroup_trajectories(samples)` — shared helper
2. `filter_errored_rollouts(rollouts)` — shared helper
3. `get_episode_returns(rollouts)` — shared helper
4. **Filtered-BC filtering** (REINFORCE-specific):
   - `min_episode_return = getattr(args, "min_episode_return", 1.0)` (from YAML config)
   - Default `1.0` keeps only success trajectories under binary rewards;
     lower the threshold to admit partially-rewarded ones.
   - Keep only rollouts where `episode_return >= min_episode_return`
   - Log: nonzero-return rate, n_kept, n_filtered
   - wandb log: `rollout/nonzero_return_rate`, `rollout/n_kept`, `rollout/n_filtered`,
     `rollout/n_trajs_{valid,errored,missing,expected}`, `rollout/n_truncated`
   - If 0% success → pad with `_dummy_sample` for no-op gradient step
5. Set `reward = 1.0` for all kept samples (uniform weight for SFT loss)
6. `flatten_and_align(rollouts, args)` — shared helper

### Step 5: Create `scripts/train/run_reinforce.sh`

Based on `run_grpo.sh`. Key differences:

| Setting | GRPO | REINFORCE |
|---------|------|-----------|
| `--n-samples-per-prompt` | 8 | 1 |
| `--loss-type` | policy_loss (default) | `sft_loss` |
| `--calculate-per-token-loss` | not set | set |
| `--disable-compute-advantages-and-returns` | not set | set |
| `--rollout-batch-size` | 8 | 64 (large — filtering drops most rollouts) |
| `--global-batch-size` | 64 | 64 |
| function paths | `lite.train.rollout.grpo.*` | `lite.train.rollout.reinforce.*` |
| `--wandb-project` | `cua-lite-grpo` | `cua-lite-reinforce` |
| `--advantage-estimator` | grpo | not set (irrelevant — `--disable-compute-advantages-and-returns` skips it) |
| KL/clipping args | set | not needed (SFT loss ignores them) |

Uses `slime/train.py` (not `train_async.py`) — needs SGLang engines for on-policy rollout.

### Step 6: Update shared env configs + create `REINFORCE.md`

Add `reward_threshold: 0.0` to each env YAML in `scripts/configs/qwen3_vl/compact/` (GRPO ignores it; REINFORCE reads it).

Create `scripts/train/rl/REINFORCE.md` documenting usage, env vars, and example commands (follow `GRPO.md` style).

---

## Development Workflow

### Step 0: Verify refactoring (Steps 1-3)

After completing the directory reorganization and code extraction:

1. `grep -rn 'data_utils\|train/grpo' --include='*.py' --include='*.sh' --include='*.md' --include='*.yaml' .` → zero stale refs
2. `uv run pytest` → all existing tests pass
3. Verify that `lite/train/rollout/grpo.py` still works correctly (it now imports from `lite/train/rollout/core/engine.py` instead of defining functions inline — behavior must be identical)

### Step 1: Dry run REINFORCE

Run with a small rollout to verify the pipeline connects end-to-end:

```bash
ROLLOUT_BATCH_SIZE=4 NUM_ROLLOUT=2 bash scripts/train/run_reinforce.sh
```

Verify:
1. Rollout generates trajectories and filters by reward threshold
2. wandb logs `rollout/nonzero_return_rate` and `rollout/n_kept`
3. SFT loss computes correctly on filtered samples
4. Weight sync works between rollout→train cycles

### Step 2: Stability testing

Same approach as [devs/train/grpo.md](/devs/train/grpo.md) Step 2 — run multiple consecutive rollout→train cycles, verify env fault isolation, training data integrity, and loop stability. Target: 5 consecutive cycles without errors.
