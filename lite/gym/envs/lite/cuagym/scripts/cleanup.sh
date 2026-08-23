#!/bin/bash
# Remove lite.cuagym containers. Covers BOTH direct-mode and server-mode
# (env-server) naming via the shared inner segment:
#
#   * direct mode: lite-env-{session_id}-lite.cuagym-{task}-{suffix}
#   * server mode: lite-env-{server_port}-{token_hash}-{session_id}-lite.cuagym-{task}-{suffix}
#
# Scoping:
#   * SESSION_ID set   -> narrow to that session, both modes.
#   * SESSION_ID unset -> ALL lite.cuagym containers on the daemon. Only run
#     this mode when you own the host or have coordinated with co-tenants.
#
# Usage:
#   bash lite/gym/envs/lite/cuagym/scripts/cleanup.sh
#   SESSION_ID=mysess bash lite/gym/envs/lite/cuagym/scripts/cleanup.sh

if [ -n "${SESSION_ID:-}" ]; then
    safe_session="${SESSION_ID//[^[:alnum:]_]/_}"
    docker ps -aq --filter "name=${safe_session}-lite.cuagym-" \
      | xargs -r docker rm -fv 2>/dev/null || true
else
    docker ps -aq --filter "name=-lite.cuagym-" \
      | xargs -r docker rm -fv 2>/dev/null || true
fi
