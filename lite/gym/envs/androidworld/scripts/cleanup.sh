#!/bin/bash
# Remove androidworld emulator containers. Covers BOTH direct-mode and
# server-mode (env-server) naming via the shared inner segment
# ``{session_id}-androidworld-``:
#
#   * direct mode: lite-env-{session_id}-androidworld-{task}-{api_port}
#   * server mode: lite-env-{server_port}-{token_hash}-{session_id}-androidworld-{task}-{api_port}
#
# Scoping:
#   * SESSION_ID set → narrow to that session after applying the same
#     session-id sanitization as lite.gym.utils.config.naming._sanitize_session_id;
#     both modes are covered by the single ``{safe_session}-androidworld-``
#     substring.
#   * SESSION_ID unset → kill ALL androidworld containers on the daemon.
#     Per-session granularity should go through env-server's HTTP path
#     (``DELETE $CUA_LITE_ENV_SERVER_URL/instances?session_id=...&env_id=androidworld``);
#     this script is the "env-server is dead, sweep the daemon" fallback
#     so the unset case intentionally widens, not silently defaults to "local".
#
# ⚠ When SESSION_ID is unset, this kills EVERY androidworld container on
# the docker daemon — including ones owned by other concurrent sessions /
# users. Only run that mode when you own the host or have coordinated
# with co-tenants.
#
# Usage:
#   bash lite/gym/envs/androidworld/scripts/cleanup.sh        # all sessions
#   SESSION_ID=mysess bash lite/gym/envs/androidworld/scripts/cleanup.sh   # one session

if [ -n "${SESSION_ID:-}" ]; then
    safe_session="${SESSION_ID//[^[:alnum:]_]/_}"
    docker ps -aq --filter "name=${safe_session}-androidworld-" \
      | xargs -r docker rm -fv 2>/dev/null || true
else
    docker ps -aq --filter "name=-androidworld-" \
      | xargs -r docker rm -fv 2>/dev/null || true
fi
