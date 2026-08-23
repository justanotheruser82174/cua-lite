# cua-lite/mobileworld — MobileWorld benchmark container (DinD: Android emulator
# + self-hosted app backends + task/eval HTTP server on :6800).
#
# Why we build FROM the upstream prebuilt image instead of the repo's own
# docker/Dockerfile: that Dockerfile needs build inputs that are NOT in git —
# the Pixel_8_API_34_x86_64.avd snapshot dir (the baked "init_state" device
# snapshot every task resets to), docker/skins, docker/images/*.tar (Mattermost/
# Mastodon/Mall backend images loaded by the entrypoint) — so a from-source build
# is impossible. Instead we pin reproducibility at the CODE level: overlay the
# upstream source cloned at a fixed SHA over the prebuilt image (the same pattern
# upstream itself uses in docker/Dockerfile.update).
#
# Build (see scripts/install.sh — do not docker-build this by hand):
#   uv run --no-sync bash lite/gym/envs/mobileworld/scripts/install.sh
#
# Run time (NOT build time) requires KVM + privileged + nested docker:
#   docker run --device /dev/kvm --privileged ... cua-lite/mobileworld:latest

FROM ghcr.io/tongyi-mai/mobile_world@sha256:b680380eac98a7ad064707f9653772af18554d201a3e6e7cf8f15d58cdc73240

ARG MOBILEWORLD_REPO=https://github.com/Tongyi-MAI/MobileWorld.git
ARG MOBILEWORLD_SHA=8ae506487bf87785292d6cad101c49955d704d39

# git for the pinned clone; socat for the ADB relay (already present in newer
# prebuilt images — idempotent either way).
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git socat && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Overlay the pinned upstream source + runtime scripts over the image's baked
# copy, then re-sync the in-image venv against the pinned lockfile.
RUN git clone --no-checkout "$MOBILEWORLD_REPO" /tmp/mobileworld && \
    cd /tmp/mobileworld && \
    git checkout "$MOBILEWORLD_SHA" && \
    rm -rf /app/service/src && \
    cp -r src /app/service/src && \
    cp pyproject.toml uv.lock README.md /app/service/ && \
    cp docker/entrypoint.sh /usr/local/bin/entrypoint.sh && \
    cp docker/start_emulator.sh docker/start_novnc.sh docker/proxy_chain.py /app/docker/ && \
    chmod +x /usr/local/bin/entrypoint.sh /app/docker/start_emulator.sh /app/docker/start_novnc.sh && \
    rm -rf /tmp/mobileworld
RUN cd /app/service && uv sync --no-cache

# Default to the nftables iptables backend — required for dockerd-in-docker on
# 6.x+ host kernels that dropped legacy iptable_nat. The pinned entrypoint.sh
# re-detects at runtime and falls back to legacy where nft is unavailable.
RUN update-alternatives --set iptables /usr/sbin/iptables-nft && \
    update-alternatives --set ip6tables /usr/sbin/ip6tables-nft

# ENTRYPOINT (entrypoint.sh) is inherited from the base image; keep upstream's
# log-tailing CMD so `docker logs` surfaces emulator/server/dockerd output.
CMD ["bash", "-c", "tail -f /var/log/emulator.log /var/log/server.log /var/log/dockerd.err.log"]
