---
name: watchdog
description: Hold / release GPUs by running a standard SGLang inference server on them (lab convention — occupy a card with a real serving job that fills ~all of its memory AND is driven at stable near-full utilization), instead of a dedicated GPU-squatter script.
---

Hold target GPUs by launching a **normal SGLang inference server** on each one (sized so its static
memory pool fills ~all of the card's VRAM) plus a small **load client** that drives it
continuously (so GPU utilization stays stably near-full). This is the lab convention: a held card
shows up in `nvidia-smi` as an ordinary `sglang`/`python` serving process under load — not a
recognizable "watchdog" — and both its memory *and* compute read as genuinely in-use.

One server + one driver is launched **per GPU** (`CUDA_VISIBLE_DEVICES=<g>`, `--tp-size 1`, its own
port), so holds and releases are per-card. Each is tracked by a pidfile so `release` only ever
stops **our own** holders — never another user's SGLang server.

## Arguments

`$ARGUMENTS` starts with a subcommand (`hold` / `release`), optionally followed by a GPU spec.

| Form | Action |
|---|---|
| `hold <gpu_spec>` | Launch a holder (server + driver) on those GPUs |
| `hold` (no spec) | Launch holders on **all free** visible GPUs |
| *(empty $ARGUMENTS)* | Same as `hold` |
| `release <gpu_spec>` | Release holders on those GPUs only |
| `release` (no spec) | Release **all** holders started by this skill |

**GPU spec forms** — normalize to a comma-separated list:

| Input | Normalized |
|---|---|
| `5` | `5` |
| `0-3` | `0,1,2,3` |
| `0,1,2,3` | `0,1,2,3` |
| `0-3,6` | `0,1,2,3,6` |
| `all` | every visible GPU (`nvidia-smi --query-gpu=index --format=csv,noheader \| paste -sd,`) |

If `$ARGUMENTS` starts with something other than `hold` / `release`, ask the user before acting.

Before holding, check `nvidia-smi` and only target GPUs that are actually free (memory ~0). Never
launch a holder onto a card another user is already computing on.

## Config (defaults; tune inline)

- `MODEL=Qwen/Qwen3-VL-2B-Instruct` — small, usually already cached (weights are tiny; the memory
  fill comes from the static pool below, not the model).
- `MEM_FRAC=0.95` — fraction of total VRAM reserved for the static pool. On an 80 GB card this
  fills ~78 GB (≈ all of it). Lower toward `0.90` if the server OOMs at boot.
- `CONCURRENCY=32` — in-flight requests the driver keeps per GPU. Higher → fuller/steadier util;
  32 keeps a 2B model's continuous-batching queue saturated (near-100%).
- `PORT_BASE=31500` — holder for GPU `g` listens on `127.0.0.1:$((PORT_BASE+g))` (localhost only;
  it's a hold, not a public endpoint). This dedicated range is how `release` finds our holders.
- `PY=.venv/bin/python` — the repo venv (has `sglang` + `requests`). Inside Slime Docker use bare `python`.
- The load driver is copied to a **neutral path** (`/tmp/sglang_client_<port>.py`) and launched from
  there, so `ps`/`top` show `python /tmp/sglang_client_<port>.py` rather than the skill dir. (The
  driver holds no GPU context, so it never appears in `nvidia-smi` at all — only the sglang server
  does, as `sglang::scheduler`.)

## hold

For each GPU in the normalized spec: (1) launch the server in its own session, (2) wait until it's
up, (3) launch the load driver against its port. Record a pidfile for each.

```bash
GPU_IDS="4"                              # <- normalized spec
MODEL="Qwen/Qwen3-VL-2B-Instruct"
MEM_FRAC=0.95
CONCURRENCY=32
PORT_BASE=31500
PY=.venv/bin/python
DRIVER_SRC=.claude/skills/watchdog/driver.py

for g in ${GPU_IDS//,/ }; do
  port=$((PORT_BASE + g))
  # (1) server — fills memory
  CUDA_VISIBLE_DEVICES=$g setsid nohup "$PY" -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port "$port" \
    --tp-size 1 --mem-fraction-static "$MEM_FRAC" --disable-cuda-graph \
    > "/tmp/sglang_gpu${g}.log" 2>&1 < /dev/null &
  echo $! > "/tmp/sglang_gpu${g}.pid"
  echo "GPU $g: server starting (PID $(cat /tmp/sglang_gpu${g}.pid), port $port)"
done

# (2) wait for each server to be ready (~30–60 s: loads model + fills pool)
for g in ${GPU_IDS//,/ }; do
  for _ in $(seq 1 60); do
    grep -q "fired up" "/tmp/sglang_gpu${g}.log" 2>/dev/null && { echo "GPU $g: server up"; break; }
    grep -qiE "error|traceback|out of memory" "/tmp/sglang_gpu${g}.log" 2>/dev/null && { echo "GPU $g: server FAILED — see log"; break; }
    sleep 2
  done
done

# (3) driver — drives util to near-full and keeps it there.
#     Run from a neutral /tmp path so ps/top don't show the skill dir.
for g in ${GPU_IDS//,/ }; do
  grep -q "fired up" "/tmp/sglang_gpu${g}.log" 2>/dev/null || continue
  port=$((PORT_BASE + g))
  client="/tmp/sglang_client_${port}.py"
  cp "$DRIVER_SRC" "$client"
  setsid nohup "$PY" "$client" --ports "$port" --concurrency "$CONCURRENCY" \
    > "/tmp/sglang_gpu${g}.client.log" 2>&1 < /dev/null &
  echo $! > "/tmp/sglang_gpu${g}.client.pid"
  echo "GPU $g: driver started (PID $(cat /tmp/sglang_gpu${g}.client.pid), running as $client)"
done
```

Then confirm both memory and util are up (give the driver ~10 s to saturate):

```bash
sleep 10
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv,noheader -i "$GPU_IDS"
```

Report per-GPU `utilization.gpu` (should be near-full) and `memory.used` (~78 GB). If a server
OOMs at boot, lower `MEM_FRAC` and retry that GPU; if util is too low, raise `CONCURRENCY`.

## release

Stop only holders started by this skill (matched via pidfiles), gracefully — driver first (stop
the load), then SIGTERM the server's process group so SGLang frees its CUDA context cleanly
(avoids the wedged-GPU failure mode from `kill -9` on a live CUDA process); SIGKILL only if it
doesn't exit.

```bash
GPU_IDS="4"        # normalized spec; for "release all": GPU_IDS=$(ls /tmp/sglang_gpu*.pid 2>/dev/null | grep -oE 'gpu[0-9]+' | grep -oE '[0-9]+' | sort -u | paste -sd,)
PORT_BASE=31500

for g in ${GPU_IDS//,/ }; do
  # driver first (stop the load), then remove its neutral /tmp copy
  dpf="/tmp/sglang_gpu${g}.client.pid"
  if [ -f "$dpf" ]; then
    dpid=$(cat "$dpf")
    kill -TERM "-$dpid" 2>/dev/null || kill -TERM "$dpid" 2>/dev/null
    rm -f "$dpf"
  fi
  rm -f "/tmp/sglang_client_$((PORT_BASE + g)).py"
  # then the server
  pf="/tmp/sglang_gpu${g}.pid"
  [ -f "$pf" ] || { echo "GPU $g: no holder pidfile — nothing to release"; continue; }
  pid=$(cat "$pf")
  kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  for _ in $(seq 1 15); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -KILL "-$pid" 2>/dev/null; kill -KILL "$pid" 2>/dev/null
  rm -f "$pf"
  echo "GPU $g: holder released"
done
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
```

Verify with `nvidia-smi` that the target GPUs are freed (util → 0, memory → ~0).

## Rules

- **Only hold free GPUs, and only release our own holders.** Never launch onto — or kill a server
  on — a card another user is using. `release` matches strictly by our pidfiles / `PORT_BASE`
  range; never blanket-`pkill sglang`.
- The **server** reserves memory (~78 GB); the **driver** keeps utilization stably near-full by
  continuously driving real inference. Both together make the card read as genuinely in-use.
- When the user wants to actually train/rollout on a held GPU, release it first (or release for
  them) — the hold occupies ~all memory and compute and will otherwise OOM/starve their job.
- Server startup takes ~30–60 s. If the log shows an OOM at boot, lower `MEM_FRAC` (e.g. 0.90); if
  util sits below target, raise `CONCURRENCY`.
- Holders look like ordinary SGLang servers under load, so generic GPU-cleanup tooling may treat
  them as real jobs — release is always explicit via this skill.
