#!/bin/bash
# Install androidworld's benchmark apps onto the base AVD.
#
# Invoked by install.sh via `docker exec` against a builder container
# spawned from cua-lite/androidworld:base. The builder container is started
# with `--device /dev/kvm --privileged` so this script (which boots a
# real x86_64 emulator) has hardware acceleration. After this script
# returns successfully, install.sh `docker commit`s the container
# state as cua-lite/androidworld:latest.
#
# Steps:
#   1. Boot a non-read-only emulator (so changes persist to the base AVD).
#   2. Wait for sys.boot_completed.
#   3. Call `android_world.env.env_launcher.load_and_setup_env(emulator_setup=True)`
#      — this installs ~20 third-party APKs and grants permissions.
#   4. Gracefully shut down the emulator. AVD state is now seeded;
#      subsequent containers spawn `-read-only` emulators that inherit it.
set -euxo pipefail

AVD_NAME="${ANDROID_AVD_NAME:-lite_avd_androidworld}"
ADB=/root/Android/Sdk/platform-tools/adb
EMU_LOG=/tmp/emu_setup.log

# 1. Boot emulator (non-read-only). No -read-only flag = writes to the AVD.
emulator -avd "$AVD_NAME" \
    -no-snapshot \
    -no-window \
    -no-metrics \
    -no-audio \
    -gpu swiftshader_indirect \
    -ports 5554,5555 -grpc 8554 \
    >"$EMU_LOG" 2>&1 &
EMU_PID=$!

# 2. Wait for boot. 5 minutes hard deadline.
DEVICE=emulator-5554
boot_ok=0
for _ in $(seq 1 150); do
    sleep 2
    if [ $(("$(stat -c %s "$EMU_LOG" 2>/dev/null || echo 0)" >= 4096)) -eq 1 ] && \
       grep -q "Successfully loaded snapshot\|emulator: INFO: boot completed" "$EMU_LOG"; then
        :  # progress signal from emulator
    fi
    if "$ADB" -s "$DEVICE" shell getprop sys.boot_completed 2>/dev/null | grep -q '^1'; then
        boot_ok=1
        break
    fi
done
if [ "$boot_ok" -ne 1 ]; then
    echo "ERROR: emulator did not boot within 300s" >&2
    echo "--- emulator log tail ---" >&2
    tail -100 "$EMU_LOG" >&2
    exit 1
fi
echo "[apps] emulator booted; running env_launcher setup..."

# 3. Install apps. The cua-lite androidworld fork's env_launcher knows
# which APKs to install + which permissions to grant.
python - <<'PY'
from android_world.env import env_launcher
env = env_launcher.load_and_setup_env(
    console_port=5554,
    grpc_port=8554,
    adb_path="/root/Android/Sdk/platform-tools/adb",
    emulator_setup=True,    # install + grant permissions
)
env.close()
PY
echo "[apps] apps installed; baking hide_automation_ui state..."

# 3a-pre. Bake ``hide_automation_ui`` into the snapshot. ``android_env``
# defaults ``show_pointer_location=True`` / ``show_touches=True`` (see
# config_classes.py) and device_settings.py applies these on every
# env construction. ``androidworld``'s ``hide_automation_ui()`` clears
# pointer_location, but only at /init time on the host server. Since
# the snapshot is loaded on EVERY snapshot-reuse reset and would restore
# whatever state was captured here, we must explicitly disable the
# overlay BEFORE saving so all future reloads come up clean. Without
# this, the GPU pointer-trace overlay (rendered by SurfaceFlinger
# when pointer_location=1) appears in obs.image after snapshot
# reload, and the agent's hard-coded swipe coordinates land on
# different UI elements vs the cold-spawn baseline → reward=0.
"$ADB" -s "$DEVICE" shell settings put system pointer_location 0
"$ADB" -s "$DEVICE" shell settings put system show_touches 0
echo "[apps] saving Quick Boot snapshot..."

# 3a. Bake a Quick Boot snapshot. Subsequent container spawns will use
# ``-snapshot default_boot -no-snapshot-save`` (in container.py)
# to load this state in ~5 s instead of cold-booting (~30-60 s).
# IMPORTANT: must happen BEFORE ``emu kill`` so qemu can write the
# snapshot file. ``avd snapshot save`` is synchronous and takes
# ~10-30 s — the disk write is the long pole.
"$ADB" -s "$DEVICE" emu avd snapshot save default_boot
echo "[apps] snapshot saved; shutting emulator down..."

# 4. Graceful shutdown so writes flush. `emu kill` is the recommended path
# (faster than SIGTERM and avoids leaving locks behind in the AVD).
"$ADB" -s "$DEVICE" emu kill || true
# Give qemu up to 30s to flush + exit.
for _ in $(seq 1 15); do
    if ! kill -0 "$EMU_PID" 2>/dev/null; then break; fi
    sleep 2
done
# Force-kill if it's still hanging.
if kill -0 "$EMU_PID" 2>/dev/null; then
    kill -9 "$EMU_PID" || true
fi
wait "$EMU_PID" 2>/dev/null || true

echo "[apps] AVD '$AVD_NAME' seeded; image build will commit this state."
