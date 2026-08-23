#!/bin/bash
# Remove osworld_2 VM-in-Docker containers — tier-1 daemon sweep, env-server-independent
# (the SIGKILL/OOM backstop for BOTH direct and server mode). Covers both naming forms
# via the shared inner segment ``{session_id}-osworld_2-``:
#
#   * direct mode: lite-env-{session_id}-osworld_2-{task}-{suffix}
#   * server mode: lite-env-{server_port}-{token_hash}-{session_id}-osworld_2-{task}-{suffix}
#
# The ``-osworld_2-`` filter is disjoint from v1 osworld (``-osworld-``: after ``osworld``
# comes ``_`` not ``-``) and lite.osworld (``.osworld-``), so the three envs never reap
# each other's containers.
#
# Scoping:
#   * SESSION_ID set   → narrow to that session (both modes).
#   * SESSION_ID unset → ALL osworld_2 containers on the daemon. ⚠ own-host only.
#     Per-session LIVE cleanup should go through the env-server:
#       DELETE $CUA_LITE_ENV_SERVER_URL/instances?session_id=...&env_id=osworld_2
#
# Usage:
#   bash lite/gym/envs/osworld_2/scripts/cleanup.sh                     # all sessions
#   SESSION_ID=mysess bash lite/gym/envs/osworld_2/scripts/cleanup.sh   # one session

if [ -n "${SESSION_ID:-}" ]; then
    # Match the container name: lite.gym.utils.config.naming._sanitize_session_id
    # collapses every non-alnum char (incl. - and .) → _, and the factory lowercases the whole name.
    # So "My-Run.1" → "my_run_1".
    _SID=$(printf '%s' "$SESSION_ID" | tr -c '[:alnum:]' '_' | tr '[:upper:]' '[:lower:]')
    docker ps -aq --filter "name=${_SID}-osworld_2-" \
      | xargs -r docker rm -fv 2>/dev/null || true
else
    docker ps -aq --filter "name=-osworld_2-" \
      | xargs -r docker rm -fv 2>/dev/null || true
fi
