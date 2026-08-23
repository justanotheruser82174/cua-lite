#!/bin/bash
# Remove mobileworld benchmark containers. Covers BOTH direct-mode and
# server-mode (env-server) naming via the shared inner segment
# ``{session_id}-mobileworld-``:
#
#   * direct mode: lite-env-{session_id}-mobileworld-{task}-{api_port}
#   * server mode: lite-env-{server_port}-{token_hash}-{session_id}-mobileworld-{task}-{api_port}
#
# Scoping:
#   * SESSION_ID set → narrow to that session after applying the same
#     session-id sanitization as lite.gym.utils.config.naming._sanitize_session_id;
#     both modes are covered by the single ``{safe_session}-mobileworld-``
#     substring.
#   * SESSION_ID unset → kill ALL mobileworld containers on the daemon.
#     Per-session granularity should go through env-server's HTTP path
#     (``DELETE $CUA_LITE_ENV_SERVER_URL/instances?session_id=...&env_id=mobileworld``);
#     this script is the "env-server is dead, sweep the daemon" fallback
#     so the unset case intentionally widens, not silently defaults to "local".
#
# ⚠ When SESSION_ID is unset, this kills EVERY mobileworld container on
# the docker daemon — including ones owned by other concurrent sessions /
# users. Only run that mode when you own the host or have coordinated
# with co-tenants. (Each container holds a nested dockerd; all of its
# inner containers die with it — nothing else to sweep.)
#
# Usage:
#   bash lite/gym/envs/mobileworld/scripts/cleanup.sh        # all sessions
#   SESSION_ID=mysess bash lite/gym/envs/mobileworld/scripts/cleanup.sh   # one session

if [ -n "${SESSION_ID:-}" ]; then
    safe_session="${SESSION_ID//[^[:alnum:]_]/_}"
    docker ps -aq --filter "name=${safe_session}-mobileworld-" \
      | xargs -r docker rm -fv 2>/dev/null || true
else
    docker ps -aq --filter "name=-mobileworld-" \
      | xargs -r docker rm -fv 2>/dev/null || true
fi
