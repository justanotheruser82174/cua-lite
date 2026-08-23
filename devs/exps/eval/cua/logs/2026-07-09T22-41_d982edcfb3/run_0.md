# cua @ 2026-07-09T22-41_d982edcfb3 · run_0

- **Commit**: `d982edcfb3ef86ea6e3cbac155adc16a2d30338b` — `Correct WebHarbor WebVoyager run_0`
- **Host / GPUs**: `gpublaze` / `6,7`
- **Artifacts**: `.exps/eval/cua/2026-07-09T22-41_d982edcfb3/run_0/`
- **Started**: `2026-07-10 02:20 PDT`
- **Last updated**: `2026-07-10 18:02 PDT`
- **Notes**: `EVAL_SUITE=basic` only (`cua.bench.local.basic`, 68 tasks). Shared HF cache is tried first with `HF_HOME=/srv/share/huggingface`; failed missing-cache launches are retried against the local cache. Current CUA basic runner has config mappings only for `qwen3_vl`, `qwen3_5`, and `gpt`; this campaign runs the Qwen local families plus API-backed `gpt-5.5` because those are the families with `cua.bench/basic.yaml` support in this checkout. `Qwen/Qwen3-VL-32B-Instruct` and `Qwen/Qwen3.5-27B` were initially skipped while only one GPU was available, then resumed on GPUs `6,7` once both were free. `Qwen/Qwen3-VL-2B-Instruct` started at concurrency 8, then resumed at `EVAL_CONCURRENCY=2` after CUA env-server 404/500/timeout failures; later runs start at `EVAL_CONCURRENCY=2`. Rollouts stop at max attempts 3.

## Results

| Model | Finished | Mean episode return |
|---|---|---|
| ⚠️ `Qwen/Qwen3.5-27B` | _**55/68**_ | _**0.3455**_ |
| ⚠️ `Qwen/Qwen3.5-4B` | _**53/68**_ | _**0.3396**_ |
| ⚠️ `Qwen/Qwen3-VL-4B-Instruct` | _**57/68**_ | _**0.3333**_ |
| ⚠️ `Qwen/Qwen3-VL-32B-Instruct` | _**55/68**_ | _**0.3273**_ |
| ⚠️ `gpt-5.5` | _**59/68**_ | _**0.2712**_ |
| ⚠️ `Qwen/Qwen3.5-9B` | _**57/68**_ | _**0.2456**_ |
| ⚠️ `Qwen/Qwen3-VL-2B-Instruct` | _**62/68**_ | _**0.2097**_ |
| ⚠️ `Qwen/Qwen3.5-2B` | _**53/68**_ | _**0.2075**_ |
| ⚠️ `Qwen/Qwen3-VL-8B-Instruct` | _**55/68**_ | _**0.1818**_ |

## Highlights

- `Qwen/Qwen3-VL-4B-Instruct` is partial at 57/68 after max attempts; unresolved tasks hit CUA env-server 500s or remained incomplete after retries.
- `Qwen/Qwen3-VL-2B-Instruct` is partial at 62/68 after max attempts; the six misses are CUA env-server reset/step 500s or timeouts rather than model output parse failures.
- `Qwen/Qwen3-VL-8B-Instruct` is partial at 55/68 after max attempts; unresolved tasks repeatedly hit CUA env-server 500s.
- `Qwen/Qwen3.5-2B` is partial at 53/68 after max attempts; unresolved tasks repeatedly hit CUA env-server 500s.
- `Qwen/Qwen3.5-4B` is partial at 53/68 after max attempts; unresolved tasks repeatedly hit CUA env-server 500s.
- `Qwen/Qwen3.5-9B` is partial at 57/68 after max attempts; unresolved tasks repeatedly hit CUA env-server reset/step 500s or remained incomplete after retries.
- `gpt-5.5` is partial at 59/68 after max attempts; unresolved tasks repeatedly hit CUA env-server reset/step 500s or missing-instance 404s.
- `Qwen/Qwen3-VL-32B-Instruct` is partial at 55/68 after max attempts on GPUs `6,7`; unresolved tasks repeatedly hit CUA env-server reset/step 500s.
- `Qwen/Qwen3.5-27B` is partial at 55/68 after max attempts on GPUs `6,7`; unresolved tasks repeatedly hit CUA env-server reset/step 500s.
- Requested non-Qwen local agents are not started because this checkout has no `scripts/configs/<agent>/default/cua.bench/basic.yaml` config or `devs/exps/eval/cua/run.sh` mapping for them; `gpt-5.5` was run through the `gpt` config.
