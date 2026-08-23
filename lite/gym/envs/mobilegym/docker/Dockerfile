# cua-lite/mobilegym — the mobilegym browser pool + vite UI + an in-container FastAPI
# RPC server that runs the FULL task lifecycle (setup → step → evaluate →
# teardown), ONE container per env-server (SINGLETON). The host
# is a thin JSON-RPC client (RemoteMobileGymEnv); the Playwright pool + the task
# lifecycle live in here. Wire payloads are plain JSON primitives (NO pickle) —
# the host imports ZERO bench_env.
#
# Self-contained + fork-free (D4/D5a): clones UPSTREAM Purewhiter/mobilegym at a
# pinned FULL SHA INSIDE the image (no host .cache staging, no cua-lite fork). A
# node stage builds the vite static site (`npm ci && npm run build` → dist/); the
# python stage installs Playwright + FastAPI and copies the cloned repo tree (for
# the container's own `import bench_env`) + the built dist/. No patches dir —
# server.py is our own file, COPY'd directly. There is NO server.py patch and the
# host-only data/tasks.json + tasks.sh are NOT needed in the image.
#
# Companion media dataset (~1.4GB) is BUNDLED into dist/cdn/ at build time so the
# container serves `/cdn/...` LOCALLY (the app's os/utils/cdn.ts falls back to
# `/cdn`). This keeps the image self-contained — image-bearing apps (Bilibili /
# Ebay / RedBook) get real media with NO runtime CDN egress. The dataset is
# gitignored upstream (the clone omits it), so we fetch the pinned release tarball
# with retries and FAIL the build if it can't complete (never ship a half /cdn).
#
# Build:  uv run --no-sync bash lite/gym/envs/mobilegym/scripts/install.sh   (docker build -f
#         docker/Dockerfile -t cua-lite/mobilegym:latest docker/)
# Run:    MobileGymContainerServices.ensure → docker run -d --init --memory <cap>
#         -p <host>:8000 cua-lite/mobilegym:latest   (only the RPC port is published;
#         the vite static server stays on the container-internal 4173)

# Pin: Purewhiter/mobilegym @ 36ba5f7 ("ci: sync main into demo via PR").
ARG MOBILEGYM_SHA=36ba5f730d9bbc795c21c08121231427242c3da4

# ---------------------------------------------------------------------------
# Stage 1 (node): clone upstream @ pinned SHA, npm ci, build the vite dist/.
# ---------------------------------------------------------------------------
FROM node:22-bookworm AS web
ARG MOBILEGYM_SHA
WORKDIR /opt/mobilegym
# Shallow fetch of the EXACT pinned commit (D5a) — no branch, no full history.
RUN git init -q \
 && git fetch -q --depth 1 https://github.com/Purewhiter/mobilegym.git "${MOBILEGYM_SHA}" \
 && git checkout -q "${MOBILEGYM_SHA}"
RUN npm ci && npm run build      # → /opt/mobilegym/dist

# ---------------------------------------------------------------------------
# Stage 2 (python): Chromium + RPC deps, the cloned repo tree, the built dist/.
# ---------------------------------------------------------------------------
FROM python:3.12-bookworm
ARG MOBILEGYM_SHA

RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends ca-certificates git curl \
 && rm -rf /var/lib/apt/lists/*

# Heavy + cache-stable layer FIRST: Chromium + its OS libs (~400 MB) + RPC deps.
# (No httpx: the in-container server is a FastAPI service; the host RPC client uses
# urllib, and the in-container env talks to the local vite UI via Playwright.)
RUN pip install --no-cache-dir playwright fastapi "uvicorn[standard]" \
 && playwright install --with-deps chromium

# Clone the SAME pinned upstream tree for `import bench_env` + its sibling data
# dirs (apps/, os/, system/ — bench_env reads e.g. apps/RedBook/data/defaults.json
# at import). Drop node_modules/.git to keep the layer lean. bench_env's
# requirements.txt omits `requests` (a transitive need of task/utils.py), so
# install it explicitly alongside.
RUN git init -q /opt/mobilegym && cd /opt/mobilegym \
 && git fetch -q --depth 1 https://github.com/Purewhiter/mobilegym.git "${MOBILEGYM_SHA}" \
 && git checkout -q "${MOBILEGYM_SHA}" \
 && rm -rf .git \
 && pip install --no-cache-dir -r bench_env/requirements.txt requests

# Built vite static site from the node stage.
COPY --from=web /opt/mobilegym/dist /opt/mobilegym/dist

# Bundle the companion media dataset into dist/cdn/ so the dist http.server serves
# `/cdn/...` locally (self-contained; no runtime CDN egress). Idempotent + robust
# so install.sh completes on any node, in any situation: the curl loop retries +
# resumes (-C -), the gzip is integrity-checked, and the build FAILS LOUDLY (the
# trailing `test -d`) rather than shipping a half-populated /cdn. The tarball is
# extracted with --strip-components=1 (drops the `mobilegym-data/` prefix → /cdn/
# maps to <app>/...) and --exclude='._*' (macOS AppleDouble junk).
ARG MOBILEGYM_DATA_URL=https://github.com/Purewhiter/mobilegym/releases/download/data-v0.1.0/mobilegym-data-v0.1.0.tar.gz
RUN set -eu \
 && mkdir -p /opt/mobilegym/dist/cdn \
 && for attempt in 1 2 3 4 5; do \
        echo "[mobilegym-data] download attempt ${attempt}/5 ..." >&2 \
     && curl -fL --retry 5 --retry-delay 5 --retry-connrefused --connect-timeout 30 \
          -C - -o /tmp/mgym-data.tar.gz "${MOBILEGYM_DATA_URL}" \
     && tar -tzf /tmp/mgym-data.tar.gz >/dev/null 2>&1 \
     && break \
     || { echo "[mobilegym-data] attempt ${attempt} failed; retrying in 15s" >&2; sleep 15; }; \
    done \
 && tar -tzf /tmp/mgym-data.tar.gz >/dev/null \
 && tar -xzf /tmp/mgym-data.tar.gz --strip-components=1 --exclude='._*' -C /opt/mobilegym/dist/cdn \
 && rm -f /tmp/mgym-data.tar.gz \
 && test -d /opt/mobilegym/dist/cdn/bilibili \
 && test -d /opt/mobilegym/dist/cdn/redbook \
 && echo "[mobilegym-data] bundled $(du -sh /opt/mobilegym/dist/cdn | cut -f1) into dist/cdn" >&2

ENV PYTHONPATH=/opt/mobilegym

COPY server.py /usr/local/bin/server.py
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# PID1 is the entrypoint; `docker run --init` (tini) reaps the chromium children +
# the vite http.server subprocess (no tini ENTRYPOINT → no double-init).
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
