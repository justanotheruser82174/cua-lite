#!/bin/bash
for _ in $(seq 1 150); do
    [ -f /tmp/gnome-failed ] && exit 1
    [ -f /tmp/gnome-ready ]  && break
    sleep 1
done
[ -f /tmp/gnome-ready ] || exit 1
rm -f "$HOME/.config/Code/SingletonSocket" \
      "$HOME/.config/Code/SingletonLock" 2>/dev/null || true
# LibreOffice first-run suppression: cp the canonical, baked xcu into the live
# profile. This is load-bearing (not a build-time no-op) because eval re-inits
# the profile, wiping the build-time COPY — we restore it on every boot. The
# single source of truth is docker/software/libreoffice/registrymodifications.xcu,
# baked read-only at the path below (see Dockerfile).
mkdir -p "$HOME/.config/libreoffice/4/user"
cp -f /usr/local/share/osworld/libreoffice/registrymodifications.xcu \
      "$HOME/.config/libreoffice/4/user/registrymodifications.xcu"
