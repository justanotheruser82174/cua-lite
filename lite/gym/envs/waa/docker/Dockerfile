# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=trycua/cua-qemu-windows@sha256:d9b0465f75b6cf801bade52e4066260e356e65b5ea0d3dab10508cd6f1ea855c
FROM python:3.9-slim-bookworm AS python

FROM ${BASE_IMAGE}

ARG WAA_COMMIT=6d39ed88c545a0d40a7a02e39b928e278df7332b

LABEL org.opencontainers.image.title="CUA-Lite WindowsAgentArena QEMU Runner" \
      org.opencontainers.image.source="https://github.com/microsoft/WindowsAgentArena" \
      org.opencontainers.image.revision="${WAA_COMMIT}"

# Snapshot support (a present ready-snapshot is always used; see prepare_snapshot.py):
# to migrate/restore a VM, QEMU needs a migratable CPU and dockur must not kill an
# -incoming restore as a failed boot. Both edits are inert for cold boots.
#  1. Drop the non-migratable `+invtsc` CPU flag dockur adds on AMD hosts (guest
#     falls back to the standard migratable clocksource — fine for GUI automation).
#  2. Make dockur's serial-PTY boot watchdog (power.sh boot()) treat a restore
#     marker as "booted", so it doesn't SIGTERM a resumed-from-snapshot VM whose
#     PTY is silent. Only triggers when /storage/.waa-restore is present.
RUN sed -i 's/CPU_FEATURES+=",+invtsc"/: # cua-lite: dropped for snapshot migratability/' /run/proc.sh \
    && sed -i '/^boot() {/a\  [ -f /storage/.waa-restore ] && return 0   # cua-lite: skip boot-watchdog on snapshot restore' /run/power.sh \
    && ! grep -q '+invtsc' /run/proc.sh \
    && bash -n /run/proc.sh && bash -n /run/power.sh \
    && grep -q 'waa-restore' /run/power.sh

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      curl \
      ffmpeg \
      git \
      libgl1 \
      libglib2.0-0 \
      libsndfile1 \
      poppler-utils \
      tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY --from=python /usr/local /usr/local
RUN ln -sf /usr/local/bin/python /usr/bin/python \
    && ln -sf /usr/local/bin/pip /usr/bin/pip

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN git clone https://github.com/microsoft/WindowsAgentArena.git /opt/waa \
    && git -C /opt/waa checkout --detach "$WAA_COMMIT"
COPY docker/patches /tmp/patches
RUN git -C /opt/waa apply /tmp/patches/lazy-heavy-imports.patch \
    && git -C /opt/waa apply /tmp/patches/browser-startup-compat.patch \
    && git -C /opt/waa apply /tmp/patches/evaluator-runtime-compat.patch \
    && rm -rf /opt/waa/.git /tmp/patches

COPY docker/bridge.py /opt/waa-bridge/bridge.py
COPY data/assets.json /opt/waa-bridge/assets.json
COPY docker/entrypoint.sh /usr/local/bin/waa-entrypoint
RUN ln -s /opt/waa/src/win-arena-container/client/desktop_env \
      /opt/waa-bridge/desktop_env \
    && chmod 0755 /usr/local/bin/waa-entrypoint

WORKDIR /opt/waa-bridge
ENV PYTHONPATH="/opt/waa/src/win-arena-container/client" \
    WAA_GUEST_IP="20.20.20.21" \
    WAA_BRIDGE_PORT="5050" \
    HOST_PORTS="5050" \
    ARGUMENTS="-qmp tcp:0.0.0.0:7200,server,nowait"

RUN python -c "from desktop_env.envs.desktop_env import DesktopEnv; assert DesktopEnv"

EXPOSE 5050 8006
HEALTHCHECK --interval=10s --timeout=3s --start-period=15m --retries=3 \
  CMD curl --fail --silent "http://127.0.0.1:5050/health" >/dev/null || exit 1

ENTRYPOINT ["/usr/bin/tini", "-s", "--", "/usr/local/bin/waa-entrypoint"]
