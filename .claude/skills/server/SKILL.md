---
name: server
description: Manage the cua-lite env server — start / stop / restart the process, list / drain / wipe live instances under your token, and read-only cross-token admin views (budget, tokens, usage). All env-server ops live here; for direct (no-server) cleanup of local docker / processes use /cleanup.
---

Manage the cua-lite env server (`scripts/serve_env.py`):

* **lifecycle** — start, stop, restart, kill the server process
* **state** (token-scoped) — survey, list, drain, or wipe **your own** live instances
* **admin** (cross-token, read-only) — budget pressure, token registry, usage breakdown, all-instance dump

For direct cleanup (local `docker rm`, env's own `cleanup.sh`, training / GPU
processes), use `/cleanup` — that path doesn't touch the env-server.

## Arguments

`$ARGUMENTS` starts with a subcommand. Optional key=value filters / opts follow.

| Form | Action |
|---|---|
| *(empty)* | Same as `status` |
| `status` | Liveness probe + instance count under your token |
| `list` | List instances under your token (JSON dump) |
| `list session=ID` / `list env-id=X` / `list session=ID env-id=X` | Same, filtered |
| `drain session=ID env-id=X` | DELETE matching the precise (session × env_id) pair — no `force` needed |
| `drain session=ID` (single filter) | DELETE all env_ids of that session — auto-adds `force=true` (still token-scoped) |
| `drain env-id=X` (single filter) | DELETE all sessions of that env_id (under your token) — auto-adds `force=true` |
| `wipe` | DELETE **everything** under your token (`force=true`, no filter); **confirm first** |
| `start` | Launch env-server in background (passthrough auth — any bearer accepted), wait for `/envs` ready |
| `start port=N env-ids=a,b,c` | Same, with overrides (env-ids comma-separated) |
| `stop` | SIGTERM (graceful — lifespan finally drains state) |
| `kill` | SIGKILL (force; orphans recovered on next `start`) |
| `restart` | `stop` + `start` (lifespan startup `recover_all` clears orphans) |
| `restart force` | `kill` + `start` (when graceful shutdown hangs) |
| `admin budget` | `GET /admin/budget` — cluster admission pressure (in_flight / max_live_envs / pct_used / host RAM+load / 503_total per layer) |
| `admin tokens` | `GET /admin/tokens` — every token the server has ever seen + instances_active / instances_created_total |
| `admin usage` / `admin usage token=T session=S env-id=X` | `GET /admin/usage` w/ filters — per (token, session, env_id) active-instance count |
| `admin list` / `admin list token=T session=S env-id=X` | `GET /admin/instances` w/ filters — cross-token instance rows (with raw token + IDs) |

Resolution at the top of any run:

```bash
URL=${CUA_LITE_ENV_SERVER_URL:?must be set; ask user before proceeding}
# TOKEN: identity, NOT a server password. Server runs in passthrough mode
# by default — any value works, identity = sha256(TOKEN)[:6] for per-user
# scoping. Default to "anonymous" if unset so client ops can proceed
# without a flag dance. Strict-token servers (rare) still need this to
# match exactly.
TOKEN=${CUA_LITE_ENV_SERVER_TOKEN:-anonymous}
AUTH="Authorization: Bearer $TOKEN"
PORT=${URL##*:}    # extract port from URL for local-side ops (assumes URL ends in :PORT)

# Parse filters from $ARGUMENTS (key=value pairs after the subcommand).
# ``token=X`` is admin-only — used by ``admin usage`` / ``admin list`` to
# filter rows. The token-scoped ops (list/drain/wipe) ignore it (they
# always filter to the caller's own AUTH bearer).
SESSION=""
ENV_ID=""
FILTER_TOKEN=""
for kv in $ARGUMENTS; do
  case "$kv" in
    session=*) SESSION="${kv#session=}" ;;
    env-id=*)  ENV_ID="${kv#env-id=}" ;;
    token=*)   FILTER_TOKEN="${kv#token=}" ;;
  esac
done

# Admin auth: only matters when server was started with --admin-token.
# Default to the same TOKEN so admin-open servers (passthrough + no
# --admin-token) work out of the box. Set CUA_LITE_ENV_SERVER_ADMIN_TOKEN
# to override for strict-admin deployments.
ADMIN_TOKEN=${CUA_LITE_ENV_SERVER_ADMIN_TOKEN:-$TOKEN}
ADMIN_AUTH="Authorization: Bearer $ADMIN_TOKEN"
```

If `$ARGUMENTS` starts with something not in the table, ask the user first.

**Auth: keep it loose.** `start` runs the server in **passthrough mode** — no
`--token`, any bearer accepted, identity = sha256 of whatever bearer the
client sent. Per-user scoping (containers, sessions, DELETE filters) all
work via that identity hash, so two tenants share one env-server without
interfering. **Avoid strict mode** (`--token X` on serve_env.py) — it
locks the server to one bearer and forces every client to know the same
secret, which is rarely what we want. The strict pattern is documented in
[serve_env.py](/scripts/serve_env.py) mode #3 for completeness; this skill
does not expose it. If you really need it, edit the launch command by hand.

## status

```bash
# Server reachable?  GET /envs returns a bare list of env_ids ([] when empty)
http_code=$(curl -so /dev/null -w '%{http_code}' --max-time 3 -H "$AUTH" "$URL/envs")
if [ "$http_code" = "200" ]; then
  curl -s -H "$AUTH" "$URL/envs" | python3 -c \
    "import json,sys; d=json.load(sys.stdin); print(f'up, envs: {d}')"
  # GET /instances returns {\"instances\": [...]} — len that key
  curl -s -H "$AUTH" "$URL/instances" | python3 -c \
    "import json,sys; d=json.load(sys.stdin); print(f'instances: {len(d[\"instances\"])}')"
else
  echo "env-server unreachable (HTTP $http_code)"
fi

# Local process (when running on the env node)
ps aux | grep -E 'serve_env\.py' | grep -v grep
```

## list

```bash
# All instances under your token
curl -sf -H "$AUTH" "$URL/instances" | python3 -m json.tool

# Filtered (session + env-id; either or both)
QUERY=""
[ -n "$SESSION" ] && QUERY+="&session_id=$SESSION"
[ -n "$ENV_ID" ] && QUERY+="&env_id=$ENV_ID"
curl -sf -H "$AUTH" "$URL/instances?${QUERY#&}" | python3 -m json.tool
```

## drain

Server's wide-net guard:

```
session_id + env_id   → precise scope, no force=true needed
session_id only       → all env_ids of that session — needs force=true
env_id only           → all sessions of that env_id (this token) — needs force=true
neither               → kill switch — use `wipe` instead (refuse here)
```

The skill mirrors that contract:

```bash
[ -z "$SESSION" ] && [ -z "$ENV_ID" ] && {
  echo "drain requires at least one filter; use 'wipe' for the unscoped case"; exit 1
}
QUERY=""
[ -n "$SESSION" ] && QUERY+="&session_id=$SESSION"
[ -n "$ENV_ID" ]  && QUERY+="&env_id=$ENV_ID"
# Single filter? auto-add force=true (still token-scoped — other tenants safe)
if [ -z "$SESSION" ] || [ -z "$ENV_ID" ]; then
  QUERY+="&force=true"
fi
curl -sf -X DELETE -H "$AUTH" "$URL/instances?${QUERY#&}" | python3 -m json.tool
# returns {"closed": [list of ids]}
```

`{"closed": []}` is success (idempotent — nothing matched the filter).

## wipe

**Destroys every instance under your token.** Other tenants' instances are
untouched (token-scoped), but every session you own goes. Confirm with the user
before sending.

```bash
curl -sf -X DELETE -H "$AUTH" "$URL/instances?force=true" | python3 -m json.tool
```

## admin

Read-only cross-token introspection. Independent of the token-scoped ops
above — `admin` views see EVERY caller's state. When the server was
started without `--admin-token` (default), these endpoints are OPEN;
with `--admin-token T`, set `CUA_LITE_ENV_SERVER_ADMIN_TOKEN=T`. The
`$ADMIN_AUTH` header (built at the top of the skill) selects the right
credential automatically.

### admin budget

```bash
curl -sf -H "$ADMIN_AUTH" "$URL/admin/budget" | python3 -m json.tool
```

Flat object: `{in_flight, max_live_envs, pct_used, host_ram_percent, host_ram_free_bytes, host_load_per_cpu, 503_total: {emergency, capacity, docker_sema, env_internal}}`.

### admin tokens

```bash
curl -sf -H "$ADMIN_AUTH" "$URL/admin/tokens" | python3 -m json.tool
```

`{tokens: [{token, token_hash, first_seen_at, last_seen_at, instances_created_total, instances_active}, …]}` sorted by current `instances_active` desc — most loaded caller first.

### admin usage / admin list

Same filter vocabulary (`token=` / `session=` / `env-id=`) — drill from
coarse to fine. Filters compose (logical AND).

```bash
QUERY=""
[ -n "$FILTER_TOKEN" ] && QUERY+="&token=$FILTER_TOKEN"
[ -n "$SESSION" ]      && QUERY+="&session_id=$SESSION"
[ -n "$ENV_ID" ]       && QUERY+="&env_id=$ENV_ID"
# admin usage — aggregated by (token, session, env_id)
curl -sf -H "$ADMIN_AUTH" "$URL/admin/usage?${QUERY#&}" | python3 -m json.tool
# admin list — one row per live instance (cross-token), with raw `token` + IDs
curl -sf -H "$ADMIN_AUTH" "$URL/admin/instances?${QUERY#&}" | python3 -m json.tool
```

- `admin usage` → `{usage: [{token, token_hash, session_id, env_id, n_active_instances}, …]}` sorted by `n_active_instances` desc.
- `admin list` → `{instances: [...]}` — same row shape as `list` plus the raw `token` field; cross-token analog of `list`.

## start

Parse opts from `$ARGUMENTS` (any of: `port=N`, `env-ids=a,b,c`). No `--token`
is passed — the server runs in passthrough mode (see "Auth: keep it loose"
above).

```bash
# Defaults (PORT already set at top from URL)
ENV_IDS="android_world android_lab lite.osworld"
for kv in $ARGUMENTS; do
  case "$kv" in
    port=*)    PORT="${kv#port=}" ;;
    env-ids=*) ENV_IDS=$(echo "${kv#env-ids=}" | tr ',' ' ') ;;
  esac
done

REPO_ROOT=${CUA_LITE_REPO_ROOT:-$(git rev-parse --show-toplevel)}
cd "$REPO_ROOT"
# NOTE: intentionally no --token → passthrough auth (recommended).
nohup uv run python scripts/serve_env.py \
  --port "$PORT" --env-ids $ENV_IDS \
  > /tmp/env_server.log 2>&1 &
echo "env-server PID=$!"

# Wait until one of the env-ids reports available (lifespan startup +
# page-cache warm complete). GET /envs default returns a bare list;
# /envs/<env_id> returns {"available": true/false, ...}.
PROBE_ENV=$(echo "$ENV_IDS" | awk '{print $1}')
until curl -s --max-time 2 -H "$AUTH" "http://localhost:$PORT/envs/$PROBE_ENV" 2>/dev/null \
        | grep -q '"available":true'; do
  sleep 5
done
echo "ready"
```

The full serve_env.py CLI supports the L2 cap override (`--max-live-envs`),
the idle TTL (`--idle-ttl-sec`), and the strict-auth knobs
`--token` (single-bearer main-API gate) and `--admin-token` (independent
secret for `/admin/*`). Tier-2 env vars (jitter, drift cycle, emergency
thresholds) bypass argparse — see `scripts/serve_env.py --help` or


**Orphan reap on startup**: `lifespan` first calls `recover_all(scope)`, which
runs each env's `EnvServices.reap(boot=True)` — orphan docker containers /
process trees / remote VMs left behind by a previous (crashed / SIGKILL'd)
env-server lifetime (scoped by `server_port`) are reaped before the first
request lands. **This is
the official orphan-cleanup pattern** — prefer `restart` over manual
`docker rm -f` when env-server died.

**Cold start cost**: when `--env-ids ...` is set, page-cache warming runs
synchronously before uvicorn opens the port. Budget 3-5 min on cold disk
(per-image `docker save | dd of=/dev/null`). Drop `--env-ids` to skip — the
server still works, just first env-spawn pays the docker-image read cost.

## stop

```bash
PID=$(pgrep -f "serve_env\.py.*--port $PORT" | tail -1)
[ -z "$PID" ] && { echo "no env-server on port $PORT"; exit 0; }
kill -TERM "$PID"
# Wait up to 60s for lifespan finally
for _ in $(seq 60); do
  ps -p "$PID" > /dev/null 2>&1 || break
  sleep 1
done
ps -p "$PID" > /dev/null 2>&1 && echo "still alive after 60s — try 'kill'"
```

Lifespan finally:
1. Cancels reaper tasks
2. Closes every `state.envs` in parallel (`env.close()` → docker rm)
3. Drains in-flight DELETE detached close tasks (bounded 120s)
4. `shutdown_all_services(port, token)` — env-specific shared resource teardown

## kill

SIGKILL — bypasses lifespan finally. Use when SIGTERM hangs. Orphans are
guaranteed; recover with next `start` (which auto-reaps).

```bash
PID=$(pgrep -f "serve_env\.py.*--port $PORT" | tail -1)
kill -KILL "$PID"
# Also kill the `uv run` wrapper if separate
PARENT=$(pgrep -f "uv run.*serve_env\.py.*--port $PORT" | head -1)
[ -n "$PARENT" ] && [ "$PARENT" != "$PID" ] && kill -KILL "$PARENT"
```

## restart

`restart` = `stop` followed by `start`. `restart force` = `kill` followed by
`start` (use when `stop` doesn't return in 60s).

`recover_all` on `start` catches anything `stop` couldn't drain.

## Rules

* `status` / `list` / `drain` / `wipe` are **token-scoped** via `Authorization: Bearer $CUA_LITE_ENV_SERVER_TOKEN`. Other tenants on the same env-server are never touched. `admin` reads are cross-token but **read-only** — there is no admin-scope mutation (cleanup of another tenant's instances is intentionally out of scope; if you must, ask them to `wipe` or `restart` the server).
* **Never pass `--token` to `start` by default.** Passthrough is the right
  shape for almost every workload (solo dev or multi-tenant). Strict mode
  forces every client to know the same secret and rarely buys anything;
  if a workload genuinely needs it, edit the launch command by hand.
* `drain` always needs at least one filter — refuse the wide-net case explicitly. Use `wipe` (with confirmation) for "kill everything under my token".
* After `kill`, follow with `start` — leaving the server dead lets orphans accumulate uncaught.
* Page-cache warm is unavoidable on `start --env-ids ...`. Skip `--env-ids` if you want a fast start without first-spawn cost.
* Never `kill` the slime training container (`lite.slime-*`); the env-server is a separate process.
* Without `$CUA_LITE_ENV_SERVER_URL` set, the skill refuses — use `/cleanup` instead for direct-mode (no-server) ops.
