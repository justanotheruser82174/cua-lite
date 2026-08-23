# GRPO Training for CUA-Lite Agents

> **Historical design note.** This file is retained as implementation history, not
> as the current development guide. The current code path is
> `lite.train.rollout.grpo.{generate,generate_rollout,convert_samples_to_train_data}`
> plus `lite.train.rollout.core.engine` and `lite.train.rollout.core.segmenter`:
> `agent.sample()` produces `LiteRLStep`, the segmenter builds radix-packable
> `slime.Sample` rows, and slime handles static-path padding downstream.

> **Prerequisites**: Follow [docs/slime.md](/docs/slime.md) to set up and run inside a Slime Docker container. Training requires the Slime container's infrastructure (Megatron-LM, SGLang, Ray).
>
> **IMPORTANT — Worktree requirement:** All development MUST happen in a **git worktree** on a dedicated branch. Do NOT modify the main working directory (`zzh` branch). Create the worktree with `git worktree add ../cua-lite-grpo grpo`. The Slime container mounts the main repo at `/workspaces/cua-lite`; you need to additionally mount the worktree into the container (e.g. `-v ../cua-lite-grpo:/workspaces/cua-lite-grpo`) and work inside that path. Follow [docs/slime.md](/docs/slime.md) for the full container setup before starting.
>
> **GPU usage**: Use at least 4 GPUs for this task. Prefer using idle GPUs. If all GPUs are occupied but some have low utilization, it is acceptable to share those as well. If training OOMs despite low pre-launch memory usage, tune training parameters (e.g. `--max-tokens-per-gpu`, `--global-batch-size`, `--sglang-mem-fraction-static`) or using more GPUs.
>
> **Note**: This document is a tentative design plan. Feel free to override any decisions when you encounter difficulties or find a better design as you go. Do not blindly trust the reference implementations (geo3k multi-turn rollout, slime internals, sglang internals, megatron internals, et al.) — they may contain bugs or assumptions that don't apply here. Always analyze the low-level details and verify correctness yourself. If you discover bugs in reference code or deviate from this plan during implementation, document all design decisions and observations in a top-level `GRPO_DESIGN.md`. `devs/train/grpo.md` is read-only.

GRPO training pipeline using slime framework with cua-lite agent-env interaction. Follows `slime/examples/geo3k_vlm_multi_turn_unfold/rollout_abort_env.py`, `slime/examples/geo3k_vlm_multi_turn_unfold/run_geo3k_vlm_multi_turn_unfold_abort_env.sh` pattern — multi-turn trajectories are "unfolded" into per-step training rows. Each `StepRecord(prompt, response, images)` → one slime `Sample`.

## Architecture

```
lite/train/
  rollout/
    grpo.py       # GRPO-specific convert_samples_to_train_data() + re-exports
    engine.py     # shared: generate(), abort(), generate_rollout(), helpers
  export/
    export_tasks.py  # CLI: parquet of gym tasks (the rollout input)

scripts/train/run_grpo.sh                 # launch script (slime train.py)
scripts/configs/qwen3_vl/compact/<env>.yaml    # config for generate()
```

## Key Design Decisions

1. **Use `agent.sample()` directly, replace generate_fn**: Rollout calls `agent.sample()` which handles the full loop (env.reset → adapter pipeline → env.step → terminated/truncated → env.close). The only customization: `generate_fn` returns `{"response": str, "response_tokens": list, "response_log_probs": list}` — the SGLang HTTP call with `return_logprob=True`. Extra keys beyond `"response"` are automatically stored in `StepRecord.extras` and available in the `on_step` callback for building slime Samples.

2. **Why unfold (no think-stripping)**: Each step's prompt is dynamically constructed by the adapter with history protocol (e.g. summarized history). Step k's prompt is NOT a simple prefix of the full trajectory → must unfold into independent training rows. No `strip_think()` needed — geo3k unfolds specifically to strip `<think>` blocks from prior turns' context (the model shouldn't see its own reasoning from previous turns). CUA-lite unfolds for a different reason: the adapter's protocol (e.g. `Qwen3VLHistoryProtocol`) dynamically reconstructs each turn's prompt, so turns are inherently independent.

3. **Abort env support** (critical): CUA-lite envs involve Docker containers, browsers, etc. Design follows `rollout_abort_env.py`: env registry, cooperative abort checks in `generate_fn` (raises AbortError to exit `agent.sample()`), custom `abort()` with env cleanup, custom `generate_rollout()` entry point.

5. **Dynamic prompts + minimal dataset**: Prompts built at rollout time from env.reset() screenshot. Parquet only stores `{problem, metadata: {env_key}}`. `agent_key` derived at runtime from model family (args) + env metadata. Can be overwritten per-row.

6. **Env-based reward (no external RM)**: Unlike geo3k which uses `--rm-type math` + `--label-key answer` to grade outputs against ground truth, CUA-lite gets reward directly from `env.step()` → set `sample.reward` in `generate()` before returning. Slime's RM pipeline checks `if sample.reward is None` before calling RM (`sglang_rollout.py:generate_and_rm`), so pre-filled rewards are preserved as-is. Consequently, `--rm-type` and `--label-key` are **not needed** in `run_grpo.sh`. For GRPO advantage computation: `convert_samples_to_train_data()` uses the **terminal reward** (last turn's `sample.reward`) and propagates the normalized advantage to all per-turn samples in the same trajectory.

---

## Implementation (in order)

### 1. `lite/train/__init__.py`

Empty file.

### 2. `lite/train/export/export_tasks.py`

CLI tool to create parquet from gym task registry.

```bash
python -m lite.train.export.export_tasks --env-id lite.demo -o train.parquet
python -m lite.train.export.export_tasks --env-id lite.osworld --split train -o data/train.parquet
python -m lite.train.export.export_tasks --env-id lite.osworld --split eval -o data/eval.parquet
```

**Parquet schema:**

| Column | Type | Description |
|--------|------|-------------|
| problem | str | Instruction text (required by slime's `_build_messages`, not used by `generate()`) |
| metadata | dict | `{env_key: "lite.demo@create_file", split: "eval"}` — passed to `gym.make(env_key)` |

- `--split` filters tasks via `gym.registry.task_ids(env_id, split=split)`, default `"train"`
- Small datasets are fine — slime's `RolloutDataSource.get_samples()` auto-wraps + reshuffles across epochs
- `agent_key` NOT stored — derived at runtime:
  ```python
  agent_key = sample.metadata.get("agent_key") or compose_key(agent_id, *env.metadata.dims)
  ```
  `sample.metadata` here is Slime prompt/control metadata (`env_key`, `split`,
  optional explicit `agent_key`), not persisted Lite row metadata. The fallback
  routing dimensions come from the live env metadata.

### 3. `lite/train/rollout/grpo.py`

Core file. Contains all rollout logic.

#### Module-level env registry

```python
_active_envs: set = set()
register_env(env)      # called on env creation
unregister_env(env)    # called in finally block
```

#### `generate(args, sample, sampling_params) -> list[Sample]`

Abort-aware. Uses `agent.sample()` directly — the only customization is the `generate_fn` passed to the agent, which calls SGLang HTTP to capture response_tokens + response_log_probs.

```
1. GenerateState(args) → tokenizer, processor, SGLang URL
2. Define custom generate_fn(prompt, images) -> dict that:
   - CHECK state.aborted → raise AbortError
   - POST to SGLang /generate with return_logprob=True
     Use "text" key (not "input_ids") to avoid SGLang's image placeholder pre-expansion bug
   - Return {"response": response, "response_tokens": [...], "response_log_probs": [...]}
3. Create agent via AgentRegistry.get(agent_key, processor=state.processor, generate_fn=generate_fn)
4. Create env via gym.make(sample.metadata["env_key"])
5. register_env(env)
6. Define a SampleHook subclass that:
   - Reads response_tokens, response_log_probs from step_record.extras (populated by generate_fn)
   - Builds per-turn slime Sample via _make_turn_sample(prompt, response_tokens, response_log_probs, ...)
   - Sets sample.reward from step_result.reward (env-based, skips slime RM)
   - Appends to turn_samples list
7. result = await agent.sample(env, hooks=[hook])
   # agent.sample() handles: env.reset → predict → env.step → terminated/truncated → env.close
8. If aborted → mark last sample ABORTED
9. finally: unregister_env(env)
10. Return turn_samples
```

#### `abort(args, rollout_id) -> list[list[Sample]]`

Based on `rollout_abort_env.py:349-406`:
1. `state.aborted = True`
2. Clean up all registered envs via `asyncio.gather(*[env.close() for env in _active_envs])` for parallel teardown (CUA-lite envs are async, unlike geo3k's sync envs)
3. Standard SGLang abort (cancel pending requests)
4. Wait for pending tasks, collect partial samples

#### `generate_rollout(args, rollout_id, data_source, evaluation=False)`

Entry point using local `abort()`. Based on `rollout_abort_env.py:494-504`.
When `evaluation=True`, delegate to `eval_rollout()`.

#### `generate_rollout_async(args, rollout_id, data_source)`

Async rollout loop with local `abort()`. Based on `rollout_abort_env.py:415-486`.

#### `convert_samples_to_train_data(args, samples) -> dict`

Copy from `slime/.../geo3k_vlm_multi_turn_unfold/rollout.py:337-436`. GRPO advantage normalization at rollout level — all per-turn samples from same trajectory share same advantage.

### 4. `scripts/configs/qwen3_vl/compact/lite.osworld.yaml`

```yaml
agent_id: "qwen3_vl"
agent_kwargs:
  protocol_kwargs:
    full_history_size: 1
  resolution: [1280, 720]
env_id: "lite.osworld"
env_kwargs:
  max_steps: 30
  loop_detect: 5
```

`generate()` passes `env_kwargs` to `gym.make()` and `agent_kwargs` to the agent. Most `env_kwargs` become env `bind()` / constructor kwargs, while make-layer keys such as `step_timeout`, `reset_timeout`, `loop_detect`, and carried env-owned keys such as `cursor` are consumed or forwarded by `gym.make()` according to the registry config contract. `resolution` belongs in `agent_kwargs`; it lands on the adapter's `BaseAgentAdapter.resolution` field, where it gates the exact-stretch target-resize step inside `process_image`, and the adapter-specific `_process_image_after_target` hook then applies any model-architecture transforms — e.g. Qwen3-VL's 32-px alignment + max_pixels cap.

### 5. `scripts/train/run_grpo.sh`

Copy the full structure of `slime/examples/geo3k_vlm_multi_turn_unfold/run_geo3k_vlm_multi_turn_unfold_abort_env.sh` as a starting point — keep all sections (configuration, cleanup, NVLink detection, model download, arg blocks, Ray start, runtime env, `ray job submit`). Then apply the CUA-lite-specific changes listed below. Default model: `Qwen3-VL-4B-Instruct`.

Key changes from the geo3k script:

```bash
MODEL_ID=${SLIME_SCRIPT_MODEL_ID:-"Qwen/Qwen3-VL-4B-Instruct"}

ROLLOUT_ARGS=(
   --prompt-data ${DATA_ROOT}/train.parquet
   --input-key problem
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 3000
   --rollout-batch-size 64  # tune down if env resources (Docker/browser) are limited
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 1
   --global-batch-size 512
   --custom-generate-function-path lite.train.rollout.grpo.generate
   --custom-convert-samples-to-train-data-path lite.train.rollout.grpo.convert_samples_to_train_data
   --rollout-function-path lite.train.rollout.grpo.generate_rollout
   --custom-config-path scripts/configs/qwen3_vl/compact/lite.osworld.yaml
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data ${DATA_ROOT}/eval.parquet
   --n-samples-per-eval-prompt 4
   --eval-max-response-len 4096
   --eval-top-p 1
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.5
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 128)
   --sglang-server-concurrency 64  # controls max parallel generate() calls; default 512 is too high for CUA-lite
                                    # (each call spawns a Docker env + browser). Start at 64, reduce if OOM or container exhaustion.
)
```

GRPO_ARGS, OPTIMIZER_ARGS, MISC_ARGS, BACKEND_ARGS — use geo3k abort env script as reference. Always include `--dump-details`.

---

## Functions to Reuse

| Function | Source | Usage |
|----------|--------|-------|
| `_make_turn_sample()` | `slime/.../geo3k_vlm_multi_turn_unfold/rollout.py:102` | Build per-turn Sample in `on_step` callback (adapt) |
| `_run_text_inference()` | same file, line 76 | SGLang HTTP inference for custom `generate_fn` (adapt) |
| `convert_samples_to_train_data()` | same file, line 337 | Copy as-is |
| `AgentRegistry.get()` | `lite/agents/agent/base.py` | Create agent with custom `generate_fn` + `processor` |
| `agent.sample()` | `lite/agents/agent/base.py:220` | Full sampling loop (env.reset → predict → env.step → close) |
| `GenerateState` | `slime/slime/rollout/sglang_rollout.py` | Tokenizer, processor, abort state |
| `eval_rollout` | same file | Evaluation rollout delegation |
| `RolloutFnTrainOutput/EvalOutput` | `slime/slime/rollout/base_types.py` | Return types |
| `MetricGatherer`, `call_dynamic_filter` | `slime/slime/rollout/filter_hub/base_types.py` | Rollout filtering |
| `abort()` pattern | `slime/.../rollout_abort_env.py:349-406` | Env cleanup + SGLang abort |
| `generate_rollout()` pattern | same file, line 494-504 | Entry point with local abort |

---

## Development Workflow

### Step 0: Verify inference data flow

Before writing training code, run pure inference to establish a reference baseline:

```bash
uv run python scripts/rollout.py --model-id Qwen/Qwen3-VL-4B-Instruct --head 1
```

Produces `.logs/rollout/<model_slug>/<env_id>/<timestamp>/.../sample_NN/turn_NNNN/`
with prompts, responses, images, annotations, actions, results, and timing.

### Step 1: Implement & dump

Run the full end-to-end training script with `--dump-details`:

```bash
bash scripts/train/run_grpo.sh
```

The rollout data flow should match Step 0's inference output in **structure and format** (prompt template, tokenization, image handling), though exact content will differ due to sampling randomness. A structural mismatch = bug in `generate()`.

Verification checklist:
1. **Prompt/response structure**: Compare dumped rollout prompts/responses against Step 0 logs — same format and structure (not exact content, since sampling is stochastic).
2. **Trajectory health**: Visually inspect that actions cause reasonable changes in environment screenshots (e.g. a click action should change the page, a type action should produce visible text). Screenshots that don't change across steps, or change in nonsensical ways, indicate broken action execution or parsing.

### Step 2: Stability testing

Run the full end-to-end training script repeatedly to verify the pipeline is stable across consecutive rollout→train cycles:

```bash
bash scripts/train/run_grpo.sh
```

Iterate until all warnings and bugs are resolved. Between runs, always clean up zombie processes and orphan containers (`docker ps`, `nvidia-smi`, `/cleanup`).

#### 2.1 Env Fault Isolation

CUA-lite envs (Docker containers + browsers) are inherently unreliable at scale. The rollout must be robust to individual env failures:

1. **Env errors must not crash training**: If an env fails (timeout, Docker error, browser crash), the rollout worker should catch the exception, return partial/empty samples, and continue. Training must never abort due to a single env failure.
2. **No resource leaks**: Env containers and browser processes must be cleaned up even on failure or abort. After each rollout batch, verify with `docker ps` that no orphan containers remain.
3. **Abort path**: Trigger SGLang abort and confirm all registered envs are closed, no zombie containers are left, and the next batch starts fresh.

#### 2.2 Training Data Integrity

Bad rollout data silently corrupts training. Verify with `--dump-details`:

1. **Prompt/response match Step 0**: Dumped rollout prompts and responses must match the inference ground truth from Step 0 — same format, same tokenization.
2. **Token ID / log-prob alignment**: Confirm that `response_tokens` and `response_log_probs` returned by SGLang correspond exactly to the response text. Misalignment causes incorrect advantage estimates.
3. **Reward correctness**: Spot-check that reward values make sense (e.g. successful trajectories get higher reward than failed ones). Wrong rewards = training in the wrong direction.
4. **Sample count per prompt**: Each prompt should produce exactly `n_samples_per_prompt` samples. Missing samples (from env failures) should be handled gracefully by `convert_samples_to_train_data` without skewing GRPO advantage normalization.

#### 2.3 Training Loop Stability

Run at least 2–3 full rollout→train cycles end-to-end:

1. **OOM during training**: If Megatron OOMs on the training step, reduce `GLOBAL_BATCH_SIZE` / `MBS` or add GPUs (TP). The default path is BSHD+MBS with `--max-tokens-per-gpu` unset, so that knob only applies if you opted into THD via `MAX_TOKENS_PER_GPU`. Rollout-phase and train-phase memory compete on colocated GPUs.
2. **Weight sync**: After each training step, verify the updated weights are correctly loaded back into SGLang for the next rollout (check loss curve — it should not reset).
3. **Loss / gradient health**: Watch for NaN loss, gradient explosion, or loss plateau from the very first steps. These often indicate data pipeline bugs rather than hyperparameter issues.
4. **Throughput**: Log samples/sec and env utilization. If throughput is much lower than expected, profile whether the bottleneck is env creation, SGLang inference, or training.

#### 2.4 Iterate

Fix every warning and error that appears in logs. The goal is zero warnings under sustained parallel load across multiple consecutive rollout→train cycles. Once parameters are stable, the pipeline must complete 5 consecutive rollout→train cycles without any errors before this step is considered done.
