# cua-lite/webgym — the whole OmniBoxes pool (redis + master + node + M instance_server
# Chromiums) in ONE container per env-server.
#
# Self-contained + FORK-FREE (plan D4/D5/D5a): this image clones the OmniBoxes
# tree from the pinned UPSTREAM commit and applies the cua-lite deltas from
# docker/patches/ INSIDE the build — there is no host `.cache` staging and no
# dependency on any cua-lite fork repo at runtime.
#
# Build:  uv run --no-sync bash lite/gym/envs/webgym/scripts/install.sh   (just
#         `docker build -f docker/Dockerfile -t cua-lite/webgym:latest docker/`)
# Run:    WebGymContainerServices.ensure → docker run -d --init --memory <cap>
#         -p <host>:7000 cua-lite/webgym:latest   (only the master port is published;
#         redis/node/instance ports stay in the container's private netns)
#
# Build context = this `docker/` dir (so `COPY patches/` resolves).
#
# Single-stage (androidlab/docker is the install.sh template). Layer order keeps
# the heavy `playwright install chromium` BEFORE the cheap clone+patch so editing
# a patch doesn't rebuild the Chromium layer.

FROM python:3.12-bookworm

# OmniBoxes upstream pin: a FULL commit SHA, never a branch.
# 574a205e is the head of upstream's `webgym` branch. Every cua-lite delta ships
# as docker/patches/ (the 7 whole-file copies listed below), so the build clones
# UPSTREAM directly — no cua-lite fork is involved at build or run time.
ARG WEBGYM_REPO=https://github.com/microsoft/webgym.git
ARG WEBGYM_SHA=574a205e1176a295decde84879d94057ca763946

# redis-server binary (the OmniBoxes lease pool). Shipping it in the image means
# the host never needs redis (the host process-pool path was removed). `git` is
# needed for the shallow clone below.
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends redis-server ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

# Heavy + cache-stable layer FIRST: the Chromium binary + its OS libs (~400 MB).
# `--with-deps` installs the apt libs Chromium needs (libnss3, libgbm1, ...).
#
# The version is PINNED, not floating, because two patches import from playwright's
# PRIVATE module `playwright._impl._errors` — `playwright.async_api` re-exports
# neither `TargetClosedError` nor the base `Error`, so there is no public path:
#   patches/playwright_instance.py     → TargetClosedError            (dead-browser detection)
#   patches/_playwright_controller.py  → Error, TimeoutError, TargetClosedError
# A floating resolve could move or rename those symbols, and the patch would then
# fail at MODULE IMPORT — taking the whole instance server down instead of
# degrading. The pin is what makes that state unrepresentable; bump it deliberately
# and re-check the four symbols above. It also fixes the Chromium revision the
# `playwright install` CLI below downloads (the CLI version picks the revision).
RUN pip install --no-cache-dir playwright==1.62.0 && playwright install --with-deps chromium

# Clone the pinned UPSTREAM tree (shallow, single-SHA fetch — D5a) into
# /opt/omniboxes, then overlay the cua-lite patches.
RUN git init /opt/omniboxes \
 && cd /opt/omniboxes \
 && git remote add origin "${WEBGYM_REPO}" \
 && git fetch --depth 1 origin "${WEBGYM_SHA}" \
 && git checkout FETCH_HEAD \
 && mkdir -p /opt/omniboxes/omniboxes/deploy/redis-data   # git omits the empty runtime dir redis.conf --dir expects

# Apply the cua-lite deltas: each patch is a whole-file vendored copy `cp`'d over
# its upstream path. See docker/patches/<file> for each patch's rationale.
#   node_server.py             → omniboxes/node/server.py                          (lease-TTL reaper; forwards the instance error body verbatim)
#   master_server.py           → omniboxes/master/server.py                         (API key + cursor query + preserves the instance_dead flag)
#   instance_server.py         → omniboxes/node/instance_server.py                  (API key + cursor query)
#   playwright_instance.py     → omniboxes/node/instances/playwright_instance.py   (launch hardening + structured instance_dead flag)
#   process_manager.py         → omniboxes/deploy/process_manager.py               (fast-port-clear)
#   _playwright_controller.py  → omniboxes/node/instances/_playwright_controller.py (fast-nav)
#   evaluator.py               → webgym/models/evaluator.py                        (Azure OpenAI judge)
COPY patches/ /tmp/webgym-patches/
RUN set -eu \
 && cp /tmp/webgym-patches/node_server.py        /opt/omniboxes/omniboxes/node/server.py \
 && cp /tmp/webgym-patches/master_server.py      /opt/omniboxes/omniboxes/master/server.py \
 && cp /tmp/webgym-patches/instance_server.py    /opt/omniboxes/omniboxes/node/instance_server.py \
 && cp /tmp/webgym-patches/playwright_instance.py /opt/omniboxes/omniboxes/node/instances/playwright_instance.py \
 && cp /tmp/webgym-patches/process_manager.py    /opt/omniboxes/omniboxes/deploy/process_manager.py \
 && cp /tmp/webgym-patches/_playwright_controller.py /opt/omniboxes/omniboxes/node/instances/_playwright_controller.py \
 && cp /tmp/webgym-patches/evaluator.py          /opt/omniboxes/webgym/models/evaluator.py \
 && rm -rf /tmp/webgym-patches

# `[omnibox]` extra pulls Pillow/redis/Requests/psutil/httpx/fastapi (install_requires
# is []). Its `playwright` requirement is UNVERSIONED, but it is already satisfied by
# the pinned install above, and pip does not upgrade a satisfied requirement without
# `--upgrade` — so the pin above is the single source of the playwright version.
RUN pip install --no-cache-dir -e "/opt/omniboxes[omnibox]"

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# PID1 is the entrypoint; zombie-reaping of the M Chromiums' forked children is
# handled by `docker run --init` (tini), NOT a tini ENTRYPOINT (no double-init).
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
