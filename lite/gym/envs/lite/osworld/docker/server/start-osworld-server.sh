#!/bin/bash
set -e
for _ in $(seq 1 150); do
    [ -f /tmp/gnome-failed ] && { echo "FATAL: GNOME failed, aborting osworld-server" >&2; exit 1; }
    [ -f /tmp/gnome-ready ]  && break
    sleep 1
done
[ -f /tmp/gnome-ready ] || { echo "FATAL: /tmp/gnome-ready never appeared" >&2; exit 1; }
export DBUS_SESSION_BUS_ADDRESS=$(cat /tmp/dbus-session-bus-address)
export GTK_MODULES=gail:atk-bridge
export DISPLAY=:1
exec /usr/bin/python3 /home/user/server/main.py
