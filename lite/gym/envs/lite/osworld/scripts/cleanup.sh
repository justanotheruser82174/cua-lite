#!/bin/bash
# Remove lite.osworld emulator containers. Covers BOTH direct-mode and
# server-mode (env-server) naming via the shared inner segment
# ``{session_id}-lite.osworld-``:
#
#   * direct mode: lite-env-{session_id}-lite.osworld-{task}-{suffix}
#   * server mode: lite-env-{server_port}-{token_hash}-{session_id}-lite.osworld-{task}-{suffix}
#
# (The trailing ``{suffix}`` is the random hex tail from
# lite/gym/sandbox/base.py:_attempt_boot_computer — replaces the
# pre-exec-stdio ``{api_port}`` segment that the cua-Computer path used.
# The filter below greps on the inner ``-lite.osworld-`` substring so
# both old and new shapes match without code changes.)
#
# Scoping:
#   * SESSION_ID set → narrow to that session after applying the same
#     session-id sanitization as lite.gym.utils.config.naming._sanitize_session_id;
#     both modes are covered by the single ``{SESSION_SAFE}-lite.osworld-``
#     substring.
#   * SESSION_ID unset → kill ALL lite.osworld containers on the daemon.
#     Per-session granularity should go through env-server's HTTP path
#     (``DELETE $CUA_LITE_ENV_SERVER_URL/instances?session_id=...&env_id=lite.osworld``);
#     this script is the "env-server is dead, sweep the daemon" fallback
#     so the unset case intentionally widens, not silently defaults to "local".
#
# ⚠ When SESSION_ID is unset, this kills EVERY lite.osworld container on
# the docker daemon — including ones owned by other concurrent sessions /
# users. Only run that mode when you own the host or have coordinated
# with co-tenants.
#
# Usage:
#   bash lite/gym/envs/lite/osworld/scripts/cleanup.sh        # all sessions
#   SESSION_ID=mysess bash lite/gym/envs/lite/osworld/scripts/cleanup.sh   # one session

if [ -n "${SESSION_ID:-}" ]; then
    SESSION_SAFE="$(printf '%s' "$SESSION_ID" | sed 's/[^[:alnum:]_]/_/g')"
    docker ps -aq --filter "name=${SESSION_SAFE}-lite.osworld-" \
      | xargs -r docker rm -fv 2>/dev/null || true
else
    docker ps -aq --filter "name=-lite.osworld-" \
      | xargs -r docker rm -fv 2>/dev/null || true
fi
