#!/bin/sh
# Move the named CLIs off the agent PATH into /opt/env/bin (separation is by PATH,
# not a uid wall). Vendor-first: resolve the REAL binary (readlink -f follows the
# alternatives chain), move it, symlink an alt name if the basename differs; a
# non-/usr/bin real (e.g. a JRE launcher) is symlinked in place. Re-open the a+rX wall.
# Does NOT remove packages and does NOT relocate payloads — callers own those, AFTER this.
# Idempotent. Usage: relocate-clis.sh <cli> [<cli> …]
set -eu
for c in "$@"; do
  src="$(command -v "$c")" || continue
  case "$src" in /*) ;; *) continue ;; esac   # skip shell builtins / bare names (else rm would nuke /usr/bin/<c>)
  real="$(readlink -f "$src")"; b="$(basename "$real")"
  update-alternatives --remove-all "$c" >/dev/null 2>&1 || true
  case "$real" in
    /usr/bin/*) mv "$real" "/opt/env/bin/$b"
                [ "$b" != "$c" ] && ln -sf "/opt/env/bin/$b" "/opt/env/bin/$c" || true ;;
    *)          ln -sf "$real" "/opt/env/bin/$c" ;;
  esac
  rm -f "/usr/bin/$c"
done
chmod -R a+rX /opt/env
