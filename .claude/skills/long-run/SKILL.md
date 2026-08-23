---
name: long-run
description: Run a long-running command or task in background and monitor it with minimal-thinking polls. Poll, detect errors, fix, restart — repeat forever until the user explicitly stops.
---

Run a long-running shell command or natural-language task in background, polling periodically with minimal thinking. Detect errors, fix them, restart — repeat until the user explicitly stops.

## Arguments

`$ARGUMENTS` specifies the task.

| Form | Action |
|---|---|
| Shell command | Launch and monitor |
| Natural language | Execute step by step in long-running mode |
| *(empty)* | Resume monitoring the active background task in this session; if none, wait for the user's next instruction |

## GPU hold discipline (overriding principle)

**Never leave the target GPUs idle.** The moment the task stops consuming them — any reason — launch `/watchdog hold` on the **target GPU IDs** (the IDs the workload itself was using, captured in Step 1). Release only in the single moment just before re-launching the workload.

Trigger `/watchdog hold` before: any pause for diagnosis/thinking (error, anomaly, unexpected output, mid-task question needing real thought); any gap between a run finishing and the next decision; `/cleanup`; long-run loop exit — **leave the hold running on exit**; the user explicitly releases.

## 1. Launch

Extract and record the **target GPU IDs** from the launch command (e.g. the `CUDA_VISIBLE_DEVICES` value, or the GPU IDs the workload will occupy). All later `/watchdog hold` calls use this ID list.

If the task is a shell command, start it with `run_in_background: true`.

## 2. Initialization phase (real-time reporting)

Stream the task output in real time until the workload is confirmed running (e.g. SGLang "fired up", first eval/rollout started, GPU memory loaded). Use `TaskOutput(task_id, block=true, timeout=30000)` or `tail -f` on the log file, and **report each observation back to the user immediately**. Only switch to steady-state polling once initialization is stable — never assume a launch succeeded without verifying.

## 3. Steady-state polling

Periodic observation, **at most every 5 minutes** (shorter is fine), with `TaskOutput(task_id, block=true, timeout=300000)`.

- Track previous output length; inspect only newly added lines each poll
- Each poll, capture system resource usage:
  ```bash
  top -bn1 | head -5
  nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.free --format=csv,noheader -i <TARGET_GPU_IDS>
  ```
- If the background task exits 0 but the workload lives inside a container (e.g. Ray), switch to `tail` on the log file
- If a poll fails (rate limit, API error), wait ~30s then retry — do NOT stop the loop
- **Report** one short line per poll with key numbers (step, loss, CPU%/mem, GPU util/mem). No analysis
- Scan for obvious errors only (traceback, CUDA OOM, NaN loss, assertion failed, connection refused). Don't analyze deeply

## 4. Handle errors and anomalies

Resume full thinking only when an error/anomaly is detected, CUDA OOM hits from GPU contention, or the script finishes/fails. Per GPU-hold discipline: hold first, think second.

- Error/anomaly → hold → diagnose, fix → release hold → restart
- **Contention** (other users' processes appeared on target GPUs) → hold grabs all remaining free memory; poll `nvidia-smi` until those processes disappear or **10 minutes** pass. Timeout → report and stop, leave hold running. Otherwise → release hold, restart
- Script finishes/fails → hold → summarize to user. Keep hold running until the user decides next steps

## 5. Restart

Before each restart: `/cleanup` (keep the hold through cleanup — `/cleanup` will not kill `gpu_watchdog`); release the hold only as the next run command launches.

## 6. Exit

Repeat forever until the user explicitly says to stop (e.g. "stop", "abort", "kill the loop", "停止"). On exit, leave `/watchdog hold` running unless the user explicitly asks you to release it.

## Rules

- Poll at most every 5 minutes in steady state; shorter is fine
- Never stop the poll loop on a transient failure (API error, rate limit) — retry after ~30s
- Target GPUs must never go idle — hold them the instant the workload pauses
- On exit (any reason), leave the GPU hold running unless the user explicitly releases it
- Resume full thinking only on error/anomaly/completion — during steady-state polling, report key numbers only, no analysis
