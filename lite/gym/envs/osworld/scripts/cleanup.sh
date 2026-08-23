#!/bin/bash
# Remove osworld VM-in-Docker containers — tier-1 daemon sweep, env-server-independent
# (the SIGKILL/OOM backstop for BOTH direct and server mode). Covers both naming
# forms via the shared inner segment ``{session_id}-osworld-``:
#
#   * direct mode: lite-env-{session_id}-osworld-{task}-{suffix}
#   * server mode: lite-env-{server_port}-{token_hash}-{session_id}-osworld-{task}-{suffix}
#
# The ``-osworld-`` filter does NOT match lite.osworld containers (``.osworld-`` —
# the char before ``osworld`` is a dot, not a hyphen). Also disjoint from osworld_2 (``-osworld_2-``:
# an ``_`` follows ``osworld``, not ``-``). So all three envs reap only their own containers.
#
# Scoping:
#   * SESSION_ID set   → narrow to that session (both modes).
#   * SESSION_ID unset → ALL osworld containers on the daemon. ⚠ own-host only —
#     this kills containers owned by other sessions/users too; coordinate with
#     co-tenants. Per-session LIVE cleanup should go through the env-server:
#       DELETE $CUA_LITE_ENV_SERVER_URL/instances?session_id=...&env_id=osworld
#
# Usage:
#   bash lite/gym/envs/osworld/scripts/cleanup.sh                     # all sessions
#   SESSION_ID=mysess bash lite/gym/envs/osworld/scripts/cleanup.sh   # one session

if [ -n "${SESSION_ID:-}" ]; then
    # Match the container name: lite.gym.utils.config.naming._sanitize_session_id
    # collapses every non-alnum char (incl. - and .) → _, and
    # OSWorldContainerFactory._make_name lowercases the whole name. So
    # "My-Run.1" → "my_run_1".
    _SID=$(printf '%s' "$SESSION_ID" | tr -c '[:alnum:]' '_' | tr '[:upper:]' '[:lower:]')
    docker ps -aq --filter "name=${_SID}-osworld-" \
      | xargs -r docker rm -fv 2>/dev/null || true
else
    docker ps -aq --filter "name=-osworld-" \
      | xargs -r docker rm -fv 2>/dev/null || true
fi
