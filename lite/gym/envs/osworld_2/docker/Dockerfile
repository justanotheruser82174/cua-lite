# cua-lite/osworld_2 — OSWorld-V2 VM-in-Docker image with the eval stack baked IN.
#
# Why a derived image (vs osworld v1's PULL-only host-attach): V2's `desktop_env` and v1's
# share the one import name `desktop_env`, so they can't coexist in the host's single uv.
# Baking V2's desktop_env into THIS image (the container's Debian userspace, outside the
# guest VM) means the cua-lite HOST needs ZERO desktop_env — osworld_2's eval runs here,
# driven by the host over a thin HTTP-RPC (docker/server.py). v1 osworld + lite.osworld
# keep the v1 dist on the host, untouched. (androidworld's docker/server.py pattern.)
#
# The base `happysixd/osworld-docker` (Debian trixie) boots the QEMU/KVM guest VM via its
# entrypoint `tini → /run/entry.sh`; we DON'T change that. We only add Python + V2
# desktop_env + server.py; the host `docker exec -d`'s server.py once the VM's :5000 answers.
#
# Build context = this dir. install.sh stages the local OSWorld-V2 checkout into
# `_vendor/OSWorld-V2/` (gitignored) first — no dependency on the pending V2 git tag.
#
# Build: uv run --no-sync bash lite/gym/envs/osworld_2/scripts/install.sh
FROM happysixd/osworld-docker@sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9

# ── 1. Python + build tools (base image has none) ────────────────────────────
# build-essential/ninja/dev headers: a few eval libs (pyacoustid/formulas/…) may build
# from source; without them pip fails on the meson/C builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip python3-dev unzip curl sudo \
      build-essential ninja-build pkg-config \
 && rm -rf /var/lib/apt/lists/*

# venv to sidestep Debian trixie's PEP 668 externally-managed marker + isolate the eval
# stack from the base image's own python usage.
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# ── 2. OSWorld-V2 — the FULL official install (`pip install .`) ───────────────
# Follow the official setup completely: install the osworld package with ALL its declared
# dependencies (pyproject is upstream's dependency source of truth; requirements.txt is just
# `-e .`). No hand-curation — that only breaks the version-consistent set upstream tests
# (e.g. pydrive's unpinned oauth2client). --prefer-binary favors wheels (avoids needless
# source builds); build tools (step 1) cover the rest. The one required pin is `oauth2client
# <4`: pydrive imports `CredentialsFileSymbolicLinkError`, removed in oauth2client 4.x — an
# upstream-unpinned transitive break that any working install must constrain.
COPY _vendor/OSWorld-V2 /opt/osworld-v2
RUN pip install --no-cache-dir --prefer-binary "oauth2client<4" /opt/osworld-v2 \
      fastapi "uvicorn[standard]"

# ── 2b. Runtime system libs the eval stack needs (opencv → libGL etc.) ───────
# Separate layer AFTER pip so tweaking system libs doesn't re-run the ~5-min pip install.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# ── 2c. Smoke-test the full eval stack (fail the build early if an import breaks) ─
RUN python -c "from desktop_env.desktop_env import DesktopEnv; \
               from desktop_env.controllers.python import PythonController; \
               from desktop_env.controllers.setup import SetupController; \
               from desktop_env.task_base import BaseTask; \
               import cv2, fastapi, uvicorn; print('V2 desktop_env eval stack OK')"

# ── 3. In-container eval server (bind-mounted over at run time for hot edits) ─
COPY server.py /usr/local/bin/server.py
RUN chmod +x /usr/local/bin/server.py

# The eval server listens here (fixed in-container); host maps it to a dynamic port and
# `docker exec -d`'s it after the VM boots. The base image's VM ports (5000/9222/8080/8006)
# are unchanged.
EXPOSE 6000
