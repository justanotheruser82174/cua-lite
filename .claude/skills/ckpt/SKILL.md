---
name: ckpt
description: Manage training checkpoints — export from the slime container to the host .ckpts/ directory with a run_info.txt for full reproducibility.
---

Checkpoint operations. `$ARGUMENTS` starts with a subcommand.

## Arguments

| Subcommand | Action |
|---|---|
| `export <desc>` (or empty — defaults to `export`) | Copy a ckpt from the Slime container to `.ckpts/` and write a `run_info.txt` with git + config + eval metadata. See [## export](#export) below. |

`<desc>` is a free-form identifier the user types to locate the ckpt, e.g. `"iter_14 from android_world 4B GRPO"`. Ask for clarification if ambiguous.

## export

Save checkpoint `$ARGUMENTS` to `.ckpts/`.

### Directory structure

```
.ckpts/{YYYYMMDD_HHMM}/{model}/{env}/{method}/async/hf/{iter}
```

- `YYYYMMDD_HHMM`: timestamp of the eval that corresponds to this checkpoint
- `{model}`: e.g. `Qwen3-VL-4B-Instruct` (without `Qwen/` prefix)
- `{env}`: e.g. `android_world`, `lite.osworld`, `webgym`
- `{method}`: e.g. `grpo`, `sft`, `reinforce`
- `async/hf/{iter}`: mirrors the container save path structure

Example:

```
.ckpts/20260418_1819/Qwen3-VL-4B-Instruct/android_world/grpo/async/hf/iter_14
```

### Steps

1. **Identify the checkpoint** in the container:

   ```bash
   docker exec lite.slime-1 ls /root/models/{MODEL_ID}/{ENV_ID}/{method}/async/hf/
   ```

2. **Get the eval timestamp** for the checkpoint iter from training logs.

3. **Copy checkpoint** from container to host:

   ```bash
   mkdir -p .ckpts/{timestamp}/{model}/{env}/{method}/async/hf/{iter}
   docker cp lite.slime-1:/root/models/{MODEL_ID}/{ENV_ID}/{method}/async/hf/{iter}/. \
     .ckpts/{timestamp}/{model}/{env}/{method}/async/hf/{iter}/
   ```

4. **Write `run_info.txt`** in the `hf/` directory (one level above iter dirs — shared across all iters of the same run). Must contain all information needed for full reproducibility:

   ```
   # Command (run inside slime container)

   # 1. Data export
   <exact data export commands with filters>
   # Result: <N> train tasks, <N> eval tasks

   # 2. Train
   <exact launch command with all env vars>

   # Git
   commit: <hash>
   branch: <branch>
   message: <commit message>

   # Training config
   model: ...
   hf_checkpoint: ... (base model or SFT checkpoint path)
   env_id: ...
   method: ...
   mode: sync/async
   lr: ...
   normalize_by_turns: ...
   eval_temperature: ...
   <all other non-default parameters>

   # Data export (run inside slime container)
   <exact commands to regenerate the training/eval parquets>
   # Result: <N> train tasks, <N> eval tasks

   # Eval results
   eval  0: XX.X% (baseline)
   eval  N: XX.X% (+/-Xpp) ← this checkpoint
   ...

   # Notes
   <any important observations, fixes, or caveats>
   ```

## Rules

- Always include the **data export commands** — without them the run is not reproducible
- Always include the **git commit hash** — code changes between runs
- Always include **all eval results** up to and including this checkpoint
- Timestamp in directory name should match the eval timestamp for this iter, not the current time
- Do NOT include processor/tokenizer dumps (too verbose) — the model name is sufficient
