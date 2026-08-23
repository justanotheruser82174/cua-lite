# Multimodal SFT for CUA-Lite Agents

> **Prerequisites**: Follow [docs/slime.md](/docs/slime.md) to set up and run inside a Slime Docker container. Training requires the Slime container's infrastructure (Megatron-LM, Ray).
>
> **Status:** This is a contributor contract note for the current Slime SFT path, not the canonical user quickstart. Keep runnable examples aligned with [docs/sft.md](/docs/sft.md); use this file for implementation boundaries and verification checklists.
>
> **IMPORTANT - Worktree requirement:** Development campaigns should happen in a **git worktree** on a dedicated branch. The stock Slime launcher mounts the current repo at `/workspaces/cua-lite`; if you train from a separate worktree, mount that worktree into the container and run commands from that mounted path. Follow [docs/slime.md](/docs/slime.md) for the container setup before starting.
>
> **GPU usage**: Use at least 4 GPUs for full-size validation unless the recipe says otherwise. If SFT OOMs, tune the current SFT knobs (`GLOBAL_BATCH_SIZE`, `MBS`, `TP_SIZE`, `OPTIM_CPU_OFFLOAD`, or extra GPUs). `MAX_TOKENS_PER_GPU` explicitly opts into THD dynamic packing and is forbidden for Qwen3.5; SFT has no SGLang server.
>
> **Note**: Do not blindly trust reference implementations (`slime` SFT rollout, Geo3K VLM SFT, legacy TRL pipeline, et al.). Use them as context, then verify the current CUA-Lite export -> `LiteRLStep` -> Slime `Sample` contract directly.

---

## Background: Legacy vs Slime SFT

### Legacy pipeline (removed)

```
Parquet (CUA-Lite format)
  -> create_cua_dataset_from_config()     # load.py: discover + adapter convert + set_transform (coupled)
  -> dataset.train_test_split(0.1)        # random split at train time (not reproducible)
  -> LiteDataCollatorForVisionLanguageModeling
  -> SFTTrainer (TRL + Accelerate)
```

Problems:
- **Data prep and training are coupled**: `create_cua_dataset_from_config()` does discovery, adapter conversion, image resolution, tool schema expansion all in one call. No way to inspect intermediate results or reuse processed data across runs.
- **Train/eval split at runtime**: `dataset.train_test_split(test_size=0.1)` is not reproducible across runs and mixes data concerns into the training script.
- **Complex config**: YAML config with nested `adapter` / `paths` / `adapter_overrides` / regex patterns is hard to use and debug.
- **Different infrastructure from GRPO**: TRL + Accelerate vs slime + Ray + Megatron.

### Migration strategy

1. **Separate data prep from training**: Export preprocessed parquet offline (like GRPO does with `export_tasks` + `split`). Training script just reads parquet.
2. **Replace training loop**: TRL -> slime (same infrastructure as GRPO).
3. **Reuse adapter pipeline**: The conversion logic (`lite/data/load.py`, adapters, protocols) is sound. Just invoke it during data prep, not at training time.

---

## Architecture

```
lite/train/
  export/export_sft.py   # CLI: canonical parquet -> model-ready SFT parquet (offline)
  rollout/sft.py         # SFT rollout: read pre-tokenized steps -> Sample segments
lite/data/merge.py       # CLI: concatenate multiple parquets

scripts/train/run_sft.sh   # launch script (slime train_async.py with SFT args)
```

The SFT launch side ships `scripts/train/run_sft.sh`; there is no
`scripts/train/sft/README.md` or per-dataset SFT launch directory today.
Export SFT with the **rollout** config for the env the data came from
(`scripts/configs/<agent>/default/<env>.yaml`): `export_sft` re-renders every
step through the agent adapter, so exporting under a different history window or
resolution than the rollout used trains the model on prompts it will never see
at inference. `scripts/configs/<agent>/recipes/sft/` is the fallback for a
corpus with no rollout config — notably `scripts/configs/lite/recipes/sft/`,
since the cua-lite reference dialect has no Agent class and never rolls out.
Every file there states in its header what it deliberately differs on and why;
an undocumented divergence from `default/` is a bug, not a recipe. Per-dataset
config files can be added alongside `run_sft.sh` if a future dataset needs them. Keep runnable,
copy-paste-ready examples aligned with the top-level
[`README.md` SFT quickstart](/README.md#sft-any-cua-on-any-datasets) and
[`lite/data/README.md` dataset layout](/lite/data/README.md#consuming-with-export_sft).

### Data flow

```
Phase 1 (offline, per adapter):
  Raw Parquet (CUA-Lite format)
    |  export_sft --agent-id qwen3_vl --model-id <HF model>
    |    lite/data/load.py: discover rows; export_sft loads/coerces messages
    |    adapter.unroll(): trajectory -> ordered AgentStep records
    |    Each AgentStep -> LiteRLStep {prompt, image_indices, response,
    |      response_tokens, reward, status, prompt_tokens}
    |    raw_response fast path only for same-adapter assistant tool-call turns;
    |      no-tool finals train the canonical stored content
    v
  SFT Parquet (model-ready: processed_images + tokenized steps)
    |  merge (optional: combine task types)
    |  split
    v
  train.parquet / eval.parquet

Phase 2 (training):
  train.parquet
    |
  slime Dataset (--prompt-data, --input-key steps, --metadata-key processed_images)
    |  sample.prompt = serialized LiteRLStep list
    |  sample.metadata = PNG bytes for trajectory images
    v
  lite.train.rollout.sft.generate_rollout()
    |  deserialize steps -> LiteRLSample
    |  build_segment_samples(): retokenize stored prompt strings with ordered
    |    image_indices, materialize multimodal inputs, radix-pack when prefix-safe
    v
  Sample(tokens, loss_mask, response_length, response, reward=0.0,
         multimodal_train_inputs / multimodal_lazy_payloads)
    |
  slime train_async.py (Megatron backend, sft_loss)
```

---

## Implementation

### 1. `lite/train/export/export_sft.py`

CLI tool for offline data preparation. Reuses `lite/data/load.py` adapter pipeline.

#### Adapter selection

The `--agent-id` flag controls the adapter family used for rendering:

- **`--agent-id lite`**: Output stays in CUA-Lite native response grammar. The full adapter key is derived per-row with `compose_key(agent_id, *metadata.dims)` (CUA `dims=("desktop", "use")` gives `lite@desktop@use`). Use this for a Lite-format student or round-trip tests.
- **`--agent-id qwen3_vl`**: Output is converted to Qwen3-VL response grammar (e.g. `computer_use` / `mobile_use` tool calls, Qwen3-VL action enums). The full key is derived the same way (e.g. `qwen3_vl@desktop@use`). Use this when training that model family.

Different adapters produce different response grammars and system prompts (with inlined tool schemas), so the exported parquet is **model-specific** and cannot be reused across models. Re-export with a different `--agent-id` and matching `--model-id` to switch models.

#### Task types

| Task type | Step contract | Adapter behavior |
|-----------|----------|-----------------|
| **understanding** | prompt + ordered image refs → text response/tokens | Pass-through (`AsIsAdapter`), no tool conversion |
| **grounding.action**, **grounding.point**, **grounding.bbox** | prompt + ordered image refs → tool-call response/tokens | Single-step trajectory, tool_calls converted to model format, tool schemas inlined in system prompt |
| **use** | one or more prompt + ordered image refs → response/tokens steps | Multi-step trajectory; protocol decides each step's history/window, tool schemas inlined in system prompt |

#### What it does

1. Discovers raw parquet files via `discover_files_under_paths()`. If `--image-root` is set, relative image paths are resolved before image loading.
2. Validates message/image references before adapter conversion. Message-level image references must point into the raw image list.
3. Adapter conversion: CUA-Lite tool_calls -> model-specific format (via `AgentAdapterRegistry`). Understanding tasks use `AsIsAdapter` (pass-through). Grounding/use tasks convert tool_calls to model format.
4. `adapter.unroll(sample)` produces a trajectory-level `AgentSample` with `processed_images` plus rendered per-step prompts and assistant targets. This is a generic RL step contract, not tied to any env-state shape: each step carries `prompt`, ordered `image_indices`, `response`, and `response_tokens`.
5. Tokenizes each step into a `LiteRLStep`. `image_indices` are ordered exactly as the corresponding `<|image_pad|>` markers occur in `prompt`, and each index points into `processed_images`. The exporter checks the count; the SFT/GRPO radix path depends on the order.
6. Writes final parquet with `processed_images` (PNG bytes), `steps` (serialized `LiteRLStep` structs), and `metadata`. Training reads these rows directly; it does not re-render chat templates.

#### Output parquet schema

| Column | Type | Description |
|--------|------|-------------|
| `processed_images` | list[bytes] | PNG bytes, one per physical trajectory image after adapter processing |
| `steps` | list[struct] | Serialized `LiteRLStep`: `prompt`, ordered `image_indices`, `response`, `response_tokens`, `reward`, `status`, `prompt_tokens` |
| `metadata` | string or dict | Pass-through from the input parquet |

#### `merge` utility

Simple CLI to concatenate multiple parquet files:

```bash
python -m lite.data.merge \
  -i use.parquet grounding_action.parquet understanding.parquet \
  -o all.parquet
```

Separating export per task type allows inspecting each independently, controlling mixture ratios, and debugging adapter conversion in isolation.

### 2. `lite/train/rollout/sft.py`

VLM SFT rollout function. Reads model-ready parquet rows and emits the same `slime.Sample` segment shape as the online GRPO/REINFORCE rollout path.

**Why not use slime's `sft_rollout.py`?**

Slime's `sft_rollout.py` expects message rows and calls `MASK_GENERATOR.get_loss_mask(messages)`, which only uses the tokenizer. For VLM, images add vision tokens (`<|vision_start|><|image_pad|>...<|vision_end|>`) that the tokenizer doesn't produce — only the processor does. This causes token count misalignment and also bypasses CUA-Lite's shared radix segmentation.

**Our approach: tokenize response at export, materialize prompts with images at training time, then use the shared radix path.**

```python
"""VLM SFT rollout for CUA-Lite datasets.

Reads serialized LiteRLStep rows and builds the same Sample segment shape as:
- lite/train/rollout/core/segmenter.py build_segment_samples()

Usage:
  --rollout-function-path lite.train.rollout.sft.generate_rollout
"""

def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    # ... lazy init PROCESSOR ...

    samples = data_buffer.get_samples(args.rollout_batch_size)

    for i, sample in enumerate(samples):
        (sample,) = sample
        processed_images = decode_hf_images(sample.metadata or [])
        steps = [deserialize_rl_step(s) for s in (sample.prompt or [])]

        # Each step already has prompt + ordered image_indices + response/tokens.
        # `build_segment_samples` retokenizes prompt text with the selected
        # processed_images, checks image_pad count/order, materializes VLM
        # tensors, and radix-packs prefix-extensional steps.
        rl_sample = LiteRLSample(
            processed_images=processed_images,
            steps=steps,
            episode_return=0.0,
            terminated=True,
            truncated=False,
        )
        out.extend(build_segment_samples(rl_sample, sample, PROCESSOR, PROCESSOR.tokenizer, state))

    return out
```

Key design decisions:

1. **Same Sample contract as GRPO**: Both SFT and GRPO operate on `LiteRLStep` records with `prompt`, ordered `image_indices`, `response`, and `response_tokens`. `build_segment_samples()` turns those steps into `tokens`, `loss_mask`, `response_length`, `response`, and multimodal inputs.

2. **No train-time chat-template rendering**: `export_sft` freezes the rendered prompt/target boundary with the model's chat template. Training only retokenizes the stored prompt text with the ordered images so the processor can expand vision tokens correctly.

3. **Generic multimodal/text contract**: Text-only steps are valid (`image_indices=()` and `multimodal_train_inputs=None`). Image steps require one `image_indices` entry per `<|image_pad|>` in prompt order.

4. **Data loading handled by the custom rollout**: Slime passes the `steps` column as `sample.prompt` and the `processed_images` column as `sample.metadata`. `lite.train.rollout.sft` deserializes both and owns image decoding/materialization.

5. **Minimal `Sample` fields**:
   - `tokens` — segment prompt + final response tokens
   - `loss_mask` — multi-span mask over assistant response tokens only
   - `response_length` — segment suffix length after the first prompt
   - `response` — concatenated response text for the segment
   - `reward = 0.0` — SFT has no reward signal
   - `multimodal_train_inputs` or `multimodal_lazy_payloads` — processor outputs or lazy image refs

### 3. `scripts/train/run_sft.sh`

Launch script. Based on `slime/examples/geo3k_vlm/run_geo3k_vlm_sft.sh`, adapted with CUA-Lite conventions from `scripts/train/run_grpo.sh`.

Uses `train_async.py` (not `train.py`) because SFT has no inference server, so `--colocate` is irrelevant. This matches slime's own VLM SFT example.

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `Qwen/Qwen3-VL-4B-Instruct` | Model to train |
| `NUM_TRAIN_GPUS` | `4` | Number of training GPUs |
| `PROMPT_DATA` | (required) | Path to model-ready SFT parquet |
| `SAVE_DIR` | `/root/checkpoints/${MODEL_SLUG}/sft/<dataset>` | Checkpoint save dir (`MODEL_SLUG` = model id with `/`→`_`; `<dataset>` = `PROMPT_DATA`'s parent dir). Also the `--wandb-group`. |
| `SAVE_INTERVAL` | `1000` | Save every N rollouts |
| `NUM_EPOCH` | `2` | Number of data passes |
| `GLOBAL_BATCH_SIZE` | `4` | Trajectories per optimizer step |
| `ROLLOUT_BATCH_SIZE` | `${GLOBAL_BATCH_SIZE}` | Data-fetch granularity; not a training knob |
| `LR` | `2e-6` | Learning rate |
| `TP_SIZE` | `1` | Tensor parallelism (must be <= NUM_TRAIN_GPUS) |
| `MBS` | `1` | BSHD micro-batch size (default batching path) |
| `MAX_TOKENS_PER_GPU` | `(unset)` | Set to opt into THD dynamic packing; forbidden for Qwen3.5 |

**Key args:**

```bash
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-4}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-${GLOBAL_BATCH_SIZE}}

SFT_ARGS=(
   --rollout-function-path lite.train.rollout.sft.generate_rollout
   --prompt-data ${PROMPT_DATA}
   --input-key steps
   --metadata-key processed_images
   --rollout-shuffle
   --num-epoch ${NUM_EPOCH:-2}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --global-batch-size ${GLOBAL_BATCH_SIZE}

   # Disable RL components
   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only                        # no SGLang inference server
   --multimodal-lazy-expand-fn-path lite.train.utils.multimodal_expand.expand
)

# Entry point: train_async.py (not train.py — no --colocate)
ray job submit ... -- python3 ${CUA_LITE_ROOT}/slime/train_async.py ...
```

Everything from GRPO that is **not** present: `--custom-generate-function-path`, `--custom-convert-samples-to-train-data-path`, `--advantage-estimator grpo`, `--kl-loss-coef`, `--eps-clip*`, `--sglang-*`, `--rollout-num-gpus*`, `--colocate`, `--n-samples-per-prompt`, env cleanup/start scripts.

---

## Verification Workflow

### Step 0: Verify data export

`export_sft.py` and `merge.py` are the current offline data-prep path. Downloaded
HF dataset mirrors keep the `cua-lite/` namespace and the dataset basename; for
the rollout mirror, that basename is `Lite.ScaleCUA`. Point `--data-paths` at
the dataset root and slice with `--filter` on `LiteBaseMetadata` (`m.dims`,
`m.others.get('episode_return')`, `m.others`). CUA metadata also exposes
`.platform` / `.task_type` properties for CUA-only filters. Do not use the historical non-Lite ScaleCUA mirror
or lowercase desktop partition paths for Lite rollout data.

`--config` is the **rollout** config, not an SFT-only recipe under
`scripts/configs/*/recipes/sft/`: `export_sft` re-renders every step through the
agent adapter, so exporting under a different history window or resolution than
the rollout used trains the model on prompts it will never see at inference.
There is no `qwen3_vl/default/lite.scalecua.yaml` — `lite.scalecua` rides the
`lite.osworld` desktop substrate, and the only fields `export_sft` reads
(`agent_id`, `agent_kwargs`) are identical across the
family's `default/*.yaml`. The `lite` examples below are the one exception: the
cua-lite reference dialect has no Agent class and never rolls out, so its SFT
recipe is its only config.

For the current Lite.ScaleCUA rollout dataset smoke, use the root dataset path:

```bash
python -m lite.train.export.export_sft \
  --config scripts/configs/qwen3_vl/compact/lite.osworld.yaml \
  --agent-id qwen3_vl \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --data-paths "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA" \
  --image-root "${CUA_LITE_DATASETS_ROOT}" \
  --filter "lambda m: m.dims == ('desktop', 'use') and not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) > 0.5" \
  --head 10 \
  -o /tmp/test_qwen3vl_lite_scalecua_use.parquet
```

For the documented task/adapter smoke matrix against the canonical
Lite.ScaleCUA corpus, keep using root-level discovery plus metadata filters instead of
embedding partition paths in the command. The examples below cover `use`,
`understanding`, and `grounding.action` for each adapter:

```bash
python -m lite.train.export.export_sft \
  --config scripts/configs/qwen3_vl/compact/lite.osworld.yaml \
  --agent-id qwen3_vl \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --data-paths "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA" \
  --image-root "${CUA_LITE_DATASETS_ROOT}" \
  --filter "lambda m: m.dims == ('desktop', 'use')" \
  --head 10 \
  -o /tmp/test_qwen3vl_use.parquet

python -m lite.train.export.export_sft \
  --config scripts/configs/qwen3_vl/compact/lite.osworld.yaml \
  --agent-id qwen3_vl \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --data-paths "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA" \
  --image-root "${CUA_LITE_DATASETS_ROOT}" \
  --filter "lambda m: m.dims == ('desktop', 'grounding.action')" \
  --head 10 \
  -o /tmp/test_qwen3vl_grounding_action.parquet

python -m lite.train.export.export_sft \
  --config scripts/configs/qwen3_vl/compact/lite.osworld.yaml \
  --agent-id qwen3_vl \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --data-paths "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA" \
  --image-root "${CUA_LITE_DATASETS_ROOT}" \
  --filter "lambda m: m.dims == ('desktop', 'understanding')" \
  --head 10 \
  -o /tmp/test_qwen3vl_understanding.parquet

python -m lite.train.export.export_sft \
  --config scripts/configs/lite/recipes/sft/default.yaml \
  --agent-id lite \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --data-paths "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA" \
  --image-root "${CUA_LITE_DATASETS_ROOT}" \
  --filter "lambda m: m.dims == ('desktop', 'use')" \
  --head 10 \
  -o /tmp/test_cualite_use.parquet

python -m lite.train.export.export_sft \
  --config scripts/configs/lite/recipes/sft/default.yaml \
  --agent-id lite \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --data-paths "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA" \
  --image-root "${CUA_LITE_DATASETS_ROOT}" \
  --filter "lambda m: m.dims == ('desktop', 'grounding.action')" \
  --head 10 \
  -o /tmp/test_cualite_grounding_action.parquet

python -m lite.train.export.export_sft \
  --config scripts/configs/lite/recipes/sft/default.yaml \
  --agent-id lite \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --data-paths "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA" \
  --image-root "${CUA_LITE_DATASETS_ROOT}" \
  --filter "lambda m: m.dims == ('desktop', 'understanding')" \
  --head 10 \
  -o /tmp/test_cualite_understanding.parquet
```

For each, inspect and verify:
- **Use**: `steps` contains one or more `LiteRLStep` structs; each step has a rendered prompt, ordered `image_indices`, and assistant `response`/`response_tokens`. Tool-call responses use the correct adapter grammar (`computer_use` for qwen3_vl; for lite, `function.name` is `computer`/`mobile` and GUI actions live under `function.arguments.actions[]` — see [/lite/agents/core/action_space/base.py](/lite/agents/core/action_space/base.py)).
- **Grounding**: `steps` contains a single assistant-target step with the adapter's tool-call response grammar.
- **Understanding**: `steps` contains a single text response step, no tool_calls, and `response_tokens` encode the canonical plain QA answer.
- **Images**: for every step, `len(image_indices) == prompt.count("<|image_pad|>")`; the Nth index supplies the Nth image pad. This order is load-bearing for radix packing and lazy multimodal expansion.

Compare qwen3_vl vs cua-lite outputs for the same input to confirm tool_calls and tool schemas differ as expected.

### Step 1: Verify `lite/train/rollout/sft.py` and `run_sft.sh`

Verify the current rollout and script contract with `DUMP=1`:

```bash
CUDA_VISIBLE_DEVICES=4 NUM_TRAIN_GPUS=1 DUMP=1 \
  MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
  PROMPT_DATA=/tmp/test_qwen3vl_use.parquet \
  bash scripts/train/run_sft.sh
```

Inspect the dump directory printed by the script
(`/tmp/cua-lite/sft/<model-slug>/<timestamp>`):

1. **Prompt/response split correctness**: Decode the initial prompt prefix and masked response spans separately — prompt should already include the model's assistant generation prefix, response should contain tool-call text or the plain answer.
2. **Vision tokens present**: Decoded prompt should contain `<|vision_start|>` / `<|vision_end|>` for each referenced image.
3. **Image order invariant**: the decoded prompt's `<|image_pad|>` order must match `steps[*].image_indices`; do not sort or dedup the per-step indices.
4. **Loss mask alignment**: `response_length` should match the segment suffix length, and the `loss_mask` should be 1 only on assistant response spans.
5. **Cross-check with GRPO rollout**: Compare output structure against `lite.train.rollout.core.segmenter.build_segment_samples()` to confirm SFT and GRPO share the same step -> Sample contract.

### Step 2: Task/adapter smoke matrix

> **NON-NEGOTIABLE**: The documented smoke matrix must be tested end-to-end (export -> rollout -> train -> verify). Do not skip any listed combination:
>
> | | `qwen3_vl` | `cua-lite` |
> |---|---|---|
> | **use** | test | test |
> | **understanding** | test | test |
> | **grounding.action** | test | test |
>
> For each cell: export a small subset (`--head 5`), run SFT with `DUMP=1` for at least 3 gradient steps, decode and inspect the first sample's tokens/loss_mask, and confirm loss decreases. The pipeline is not done until these 6 listed combinations produce correct output and train without errors.
>
> Do not call this full generic grounding coverage unless the examples and runs also cover `grounding.point` and `grounding.bbox` for both adapters.

### Step 3: Stability testing

Run multiple epochs on a merged dataset (all task types combined). Watch for:
- OOM (default BSHD path: reduce `GLOBAL_BATCH_SIZE` / `MBS`, enable `OPTIM_CPU_OFFLOAD`, raise `TP_SIZE`, or add GPUs; `MAX_TOKENS_PER_GPU` only applies if you opted into THD via `MAX_TOKENS_PER_GPU`, and is forbidden for Qwen3.5)
- NaN loss (data pipeline bug — check for empty samples or malformed images)
- Loss not decreasing (loss mask bug — likely training on prompt tokens instead of response only)
- Loss immediately zero (loss mask all zeros — `response_length` computation wrong)

---

## Reference Implementations

| Component | Reference | What to take |
|-----------|-----------|--------------|
| `run_sft.sh` | `slime/examples/geo3k_vlm/run_geo3k_vlm_sft.sh` | Script structure, SFT args, `train_async.py` entry point |
| `run_sft.sh` | `scripts/train/run_grpo.sh` | CUA-Lite conventions: PYTHONPATH, model download, CKPT_ARGS |
| `README.md` / data docs | `/README.md#sft-any-cua-on-any-datasets`, `/lite/data/README.md#consuming-with-export_sft` | Current end-to-end SFT examples and canonical dataset-root layout |
| `configs/` (future) | `scripts/configs/qwen3_vl/compact/` | Per-dataset YAML config pattern (pattern only — SFT launch side currently has no per-dataset configs) |
| `lite/train/rollout/sft.py` | `lite/train/rollout/core/segmenter.py` `build_segment_samples()` | Shared LiteRLStep -> radix Sample path |
| `lite/train/rollout/sft.py` | `lite/train/rollout/core/segmenter.py` `build_segment_samples()` | processor -> multimodal_train_inputs / lazy payload pattern, Sample fields |
| `lite/train/rollout/sft.py` | `slime/slime/rollout/sft_rollout.py` | `generate_rollout()` function signature, data_buffer API |
| `export_sft.py` | `lite/data/load.py` | `discover_files_under_paths()`, `load_file_as_dataset()` |
| `export_sft.py` | `lite/train/export/export_tasks.py` | CLI pattern, `--head`, `--sample` flags |
