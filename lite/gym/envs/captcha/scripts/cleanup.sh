#!/bin/bash
# Kill leftover captcha processes (per-episode Flask servers + their /tmp
# files) and reap leaked Chromium subprocesses. Does NOT touch .cache/
# downloads — that's uninstall.sh.
#
# Normally unnecessary: episodes clean up after themselves (env.close()), and
# the env-server's EnvServices.reap (boot + steady) handles leaks. This is the
# manual hammer for a wedged host.
#
# Session scope: both the Flask reap and the /tmp GC key off
# CUA_LITE_ENV_SERVER_PORT (the env server's listen port), "direct" when unset
# — the SAME tag main.py's _server_scope() / _server_marker() bake into the
# subprocess cmdline + /tmp filenames. So this NEVER touches a co-resident
# env-server's captcha resources (different port).
#
# Usage:
#   bash lite/gym/envs/captcha/scripts/cleanup.sh
#   CUA_LITE_ENV_SERVER_PORT=30180 bash lite/gym/envs/captcha/scripts/cleanup.sh

set -euo pipefail

log() { echo "[captcha cleanup] $*" >&2; }

# Scope = this env-server's listen port, "direct" when unset. Matches
# main.py _server_marker(scope) == "CUA_LITE_CAPTCHA_SERVER_es<scope>_ = 1".
SCOPE="${CUA_LITE_ENV_SERVER_PORT:-direct}"
MARKER="CUA_LITE_CAPTCHA_SERVER_es${SCOPE}_ = 1"

# 1. Per-episode Flask servers of THIS scope (cmdline tagged by main.py's
#    Popen), this user only. Never a host-wide match — a concurrent
#    env-server's Flask procs carry a different _es<scope>_ infix.
if pgrep -u "$(id -u)" -f "$MARKER" >/dev/null 2>&1; then
    pkill -9 -u "$(id -u)" -f "$MARKER" || true
    log "killed leftover captcha Flask server(s) (scope=${SCOPE})"
else
    log "no captcha Flask servers running (scope=${SCOPE})"
fi

# 2. Reap leaked Chromium subprocesses (Playwright launches with
#    --user-data-dir=/tmp/playwright_chromiumdev_*). Under normal teardown
#    LocalCaptchaEnv.close kills them; they only leak on SIGKILL of the
#    supervisor. The marker is unscoped (Playwright owns the dir name), so we
#    add an orphan-or-cwd filter: only kill chromiums
#    whose supervisor has died (PPid=1) OR is still inside our cua-lite
#    checkout — never a live co-tenant's browser.
killed_chromium=0
for pid in $(pgrep -u "$(id -u)" -f 'playwright_chromiumdev' 2>/dev/null); do
    [ -d "/proc/$pid" ] || continue
    ppid=$(awk '/^PPid:/ {print $2}' "/proc/$pid/status" 2>/dev/null) || continue
    pcwd=$(readlink "/proc/$ppid/cwd" 2>/dev/null) || pcwd=""
    if [ "$ppid" = "1" ] || [[ "$pcwd" == *cua-lite* ]]; then
        if kill -9 "$pid" 2>/dev/null; then
            killed_chromium=$((killed_chromium + 1))
        fi
    fi
done
if [ "$killed_chromium" -gt 0 ]; then
    log "reaped $killed_chromium chromium subprocess(es)"
fi

# 3. Per-instance result/log files of THIS scope in /tmp (only ours are
#    writable). Names are captcha_<scope>_<flaskport>.{json,log} (main.py reset).
rm -f "/tmp/captcha_${SCOPE}_"*.json "/tmp/captcha_${SCOPE}_"*.log 2>/dev/null || true

log "done"
