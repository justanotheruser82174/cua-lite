#!/bin/bash
# Remove cua.bench.local containers AND their per-instance image-tag aliases. Both direct-
# and server-mode names share the inner ``-cua.bench.local.`` segment (like lite.osworld's
# ``-lite.osworld-``):
#   direct: lite-env-{session}-cua.bench.local.{dataset}-{env}-{variant}-{suffix}[_latest]
#   server: lite-env-{port}-{token}-{session}-cua.bench.local.{dataset}-{env}-{variant}-{suffix}[_latest]
#
# On a live env-server the drift-reaper reclaims the CONTAINERS; this is the "server is dead,
# sweep the daemon" fallback. It is ALSO the only thing that reclaims the leaked *image-tag*
# aliases: reset() tags trycua/cua-xfce → ``lite-env-…-cua.bench.local.…:latest`` and only a
# clean close() removes them, so a crashed process leaks the tag (not the layers).
#
# Scoping (mirrors lite.osworld/scripts/cleanup.sh):
#   SESSION_ID set   → narrow to that session.
#   SESSION_ID unset → sweep ALL cua.bench.local containers/tags on the daemon.
#     ⚠ that includes containers owned by other concurrent sessions/users — only run when
#     you own the host. Per-session cleanup on a live server should go through the HTTP path
#     (DELETE $CUA_LITE_ENV_SERVER_URL/instances?session_id=...&env_id=cua.bench.local.<dataset>).
#
# Usage:
#   bash lite/gym/envs/cua/scripts/cleanup.sh                       # all sessions
#   SESSION_ID=mysess bash lite/gym/envs/cua/scripts/cleanup.sh     # one session

SEG="-cua.bench.local."
# Match the producer BYTE-FOR-BYTE (main.py reset → format_container_name(...).lower()), which
# is TWO transforms, not one: lite.gym.utils.config.naming._sanitize_session_id collapses every
# non-alnum char (notably ``-`` and ``.``) to ``_``, and only then is the whole repo lowercased
# (docker image repos must be lowercase; a user's SESSION_ID may not be). The lowercase half
# alone silently reaped NOTHING for any session carrying a ``-``: SESSION_ID=my-run filtered
# ``my-run-…`` while the producer had already emitted ``lite-env-my_run-cua.bench.local.…``.
# Same expression as osworld/osworld_2 cleanup; derived-and-enforced by
# tests/gym/utils/config/test_naming.py::test_cleanup_script_filter_matches_format_container_name.
[ -n "${SESSION_ID:-}" ] && SEG="$(printf '%s' "$SESSION_ID" | tr -c '[:alnum:]' '_' | tr '[:upper:]' '[:lower:]')-cua.bench.local."

# containers (running or exited)
docker ps -aq --filter "name=${SEG}" | xargs -r docker rm -fv 2>/dev/null || true
# per-instance image-tag aliases (repository carries the same segment). `--` before the
# pattern: SEG starts with `-`, which grep would otherwise read as options.
docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
  | grep -F -- "${SEG}" | xargs -r docker rmi 2>/dev/null || true

echo "[cua cleanup] done" >&2
