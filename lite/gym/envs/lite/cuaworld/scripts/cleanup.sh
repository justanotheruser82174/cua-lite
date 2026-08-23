#!/usr/bin/env bash
# Lightweight per-run cleanup for cuaworld. Reaps leaked
# DIRECT-mode session containers named `lite-env-local-lite.cuaworld.*` (a crashed direct
# rollout can leave these). Idempotent — exits 0 when there's nothing to reap.
#
# Scoped to this checkout's SESSION_ID (default `local`). By default only removes
# NON-running containers (safe alongside a live run); pass `--force` to also kill
# running ones. Env-server mode does NOT need this — the server's ContainerServices
# reaper handles its `*-<port>` containers.
#
# NOTE: direct-mode containers share the `local` session, so `--force` with the
# default SESSION_ID=`local` removes EVERY local-session cuaworld container on the
# host (incl. a co-tenant's live direct rollout). Set SESSION_ID to scope to your run.
#
# Usage:
#   uv run --no-sync bash lite/gym/envs/lite/cuaworld/scripts/cleanup.sh [--force]
set -euo pipefail

FORCE=0; [ "${1:-}" = "--force" ] && FORCE=1
SESSION="${SESSION_ID:-local}"; SESSION="${SESSION//[^[:alnum:]_]/_}"
# `--filter name=` is a Go RE2 regex; `[.]` matches the literal dot before <software>.
PATTERN="lite-env-${SESSION}-lite[.]cuaworld[.]"

if [ "${FORCE}" = "1" ]; then
  ids="$(docker ps -aq --filter "name=${PATTERN}" 2>/dev/null || true)"
else
  # only stopped/exited/created (leave running direct sessions alone)
  ids="$(docker ps -aq --filter "name=${PATTERN}" --filter "status=exited" --filter "status=created" --filter "status=dead" 2>/dev/null || true)"
fi

if [ -z "${ids}" ]; then
  echo "no leaked ${PATTERN}* containers to reap"
  exit 0
fi

n=0
for c in ${ids}; do
  docker rm -fv "${c}" >/dev/null 2>&1 && n=$((n+1)) || true
done
echo "reaped ${n} ${PATTERN}* container(s)"
