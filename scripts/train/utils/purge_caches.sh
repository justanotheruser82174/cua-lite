#!/usr/bin/env bash
set -euo pipefail

#
# Purge Slime-container model/dataset caches.
#
# Intended for manual use inside a disposable training container. Outside a
# Docker container, require an explicit override to avoid deleting host caches.
#
if [ ! -f /.dockerenv ] && [ "${CUA_LITE_ALLOW_HOST_CACHE_PURGE:-0}" != "1" ]; then
   echo "Refusing to purge /root caches outside Docker." >&2
   echo "Set CUA_LITE_ALLOW_HOST_CACHE_PURGE=1 to override." >&2
   exit 1
fi

rm -rf -- /root/models/ /root/datasets/
