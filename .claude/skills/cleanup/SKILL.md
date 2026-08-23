---
name: cleanup
description: Direct-mode cleanup of local docker containers, training processes, and GPU memory. For env-server based ops (HTTP DELETE /instances, server start/stop) use /server.
---

Clean up stale resources on the local host:

* env containers via each env's `lite/gym/envs/<env>/scripts/cleanup.sh`
* training processes (sglang / ray / python) via `scripts/train/utils/cleanup.sh`
* GPU memory by reaping compute apps (excluding `gpu_watchdog`)

**Direct-mode only.** This skill runs commands against the local docker daemon
and process table. For env-server-based cleanup (DELETE /instances, drain by
session, wipe everything under a token), use `/server` — it routes through
HTTP and doesn't need env deps locally.

Detection at the top of any run:

```bash
IN_CONTAINER=$([ -f /.dockerenv ] && echo 1 || echo 0)
RUN=$([ "$IN_CONTAINER" = "1" ] && echo "bash" || echo "uv run bash")
```

* Inside the slime container: `bash <script>` (venv already active).
* On the host: `uv run bash <script>` (activates the uv venv for tier-1 scripts).

## Arguments

`$ARGUMENTS` accepts one or more of:

| Form | Action |
|---|---|
| *(empty)* | **Only what this session launched** — `kill -9 <pid>` / `docker rm -f <name>` on IDs the agent can confidently attribute to itself; otherwise ask. |
| `android_world` | `$RUN lite/gym/envs/android_world/scripts/cleanup.sh` |
| `android_lab` | `$RUN lite/gym/envs/android_lab/scripts/cleanup.sh` |
| `browsergym <bench>` | `$RUN lite/gym/envs/browsergym/scripts/cleanup.sh <bench>` |
| `osworld` | `$RUN lite/gym/envs/osworld/scripts/cleanup.sh` (local Docker sweep, `SESSION_ID`-scoped when set) |
| `webgym` | container-only (no cleanup.sh): `docker rm -f webgym-<port>` (the shared backend container) |
| `lite.osworld` | `$RUN lite/gym/envs/lite/osworld/scripts/cleanup.sh` |
| `training` | `bash scripts/train/utils/cleanup.sh` — sglang / ray / python; **container-only by default** |
| `gpu` | nvidia-smi compute-app reap, excluding `gpu_watchdog` (from `/watchdog`) |
| `all` | every env above + `training` (if `IN_CONTAINER`) + `gpu` |
| PID / container-name / pattern | targeted `kill -9` / `docker rm -f` |

`lite.demo`, `osworld_g`, `screenspot_pro` have no tier-1 cleanup script
(stateless / atexit-hook handled at process exit). To clean orphan containers
from these, use the generic `docker rm -f` pattern under the PID / container
form above.

Named args = the user's implicit authorization to widen scope beyond
this-session-only. Show the candidate list before any destructive op.

### SESSION_ID scoping

Tier-1 scripts that filter docker containers by SESSION_ID:
`android_world`, `android_lab`, `osworld`, `lite.osworld`.

* Inside slime container — `$SESSION_ID` set by `scripts/train/slime/launch.sh`.
* On host — falls back to `local` (collision risk if two users share the host without setting it).
* `unset` is intentional widening: the script kills **every container of
  that env_id across the daemon** — the "env-server is dead, sweep
  everything" fallback. Warn the user before running unscoped on a shared
  host.

`browsergym` and `webgym` use different scoping (fixed container names and a
single shared backend container reaped via `docker rm -f <env>-<port>`,
respectively) — SESSION_ID doesn't apply.

### `training` host protection

`scripts/train/utils/cleanup.sh` issues `pkill -9 python`. Safe inside the
slime container (only training python lives there); dangerous on the host
(kills unrelated python). **Default: refuse on host** unless the user
re-confirms.

## 1. Survey

```bash
nvidia-smi
ps aux | grep -E 'sglang|ray|slime|emulator|qemu' | grep -v grep
ps aux | grep -E 'python.*(train|rollout|server|bench|stress)' | grep -v grep
# per-instance (lite-env-*) and shared-backend (webgym-<port> / mobilegym-<port>) containers
docker ps -a --filter "name=lite-env-" --filter "name=webgym-" --filter "name=mobilegym-" --format "{{.ID}} {{.Names}} {{.Status}}"
ps aux | awk '$8 ~ /Z/ {print}'      # zombies
```

Show the user a summary of what was found before any destructive action.

## 2. Clean (after user confirmation)

### Per-env

```bash
# env_id "lite.osworld" → lite/gym/envs/lite/osworld/scripts/cleanup.sh
for env_id in $ENV_IDS_TO_CLEAN; do
  $RUN "lite/gym/envs/${env_id//.//}/scripts/cleanup.sh" $EXTRA_ARGS
done
```

Trust each tier-1 script's scoping (SESSION_ID filters, env-specific container
patterns, etc.). If a command needs fixing, fix the tier-1 script — don't paper
over it here.

### Training processes

```bash
if [ "$IN_CONTAINER" = "1" ]; then
  bash scripts/train/utils/cleanup.sh
else
  echo "Refusing 'training' cleanup on host: 'pkill -9 python' would kill unrelated processes."
  echo "If you really want this, ask the user to confirm: uv run bash scripts/train/utils/cleanup.sh"
fi

# Zombie reap — only when parent matches our patterns
ps -eo pid,ppid,stat,cmd | awk '$3 ~ /Z/' | while read pid ppid stat cmd; do
  ps -p "$ppid" -o cmd= 2>/dev/null \
    | grep -qE 'sglang|ray|slime|python.*(train|rollout|server|bench|stress)' \
    && kill -9 "$ppid" 2>/dev/null
done
```

### GPU memory

```bash
# Kill GPU users EXCEPT gpu_watchdog (which holds the GPUs intentionally).
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  ps -p "$pid" -o cmd= 2>/dev/null | grep -q gpu_watchdog || kill -9 "$pid" 2>/dev/null
done
```

## 3. Verify

```bash
nvidia-smi
ps aux | grep -E 'sglang|ray|slime|emulator|qemu' | grep -v grep
docker ps -a --filter "name=lite-env-" --filter "name=webgym-" --filter "name=mobilegym-" --format "{{.ID}} {{.Names}} {{.Status}}"
```

Report before/after. If anything you tried to clean is still alive, list the
survivors and stop — don't declare cleanup successful.

## Rules

* Default scope (empty $ARGUMENTS) is **this-session-only** — review conversation context for IDs the agent itself launched. Don't run tier-1 scripts under default scope; they cross session boundaries.
* Tier-1 scripts are the source of truth for each env. Update them rather than the skill if a command needs fixing.
* Use `$RUN` (`bash` in container, `uv run bash` on host) — never hardcode either.
* Never run `training` on the host without explicit re-confirmation (`pkill -9 python`).
* Never kill the slime training container (`lite.slime-*`), the current shell, Claude Code, or IDE processes.
* Never kill processes owned by other users unless explicitly asked.
* **Never** stop, kill, exec into, or otherwise interfere with a running docker container that you did not start.
* For env-server REST-API ops (instance list / drain / wipe), use `/server`. This skill doesn't make HTTP calls to env-server.
