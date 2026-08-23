# cua-lite/androidworld — emulator image for the AndroidWorld benchmark.
#
# Each rollout worker spawns one container from this image. The contained
# emulator boots a Pixel 6 AVD running Android 13 (API 33), with the 20+
# benchmark apps + apps install grants already applied via androidworld's
# env_launcher (run once at image build time inside this Dockerfile).
#
# Why containerize (vs the older host-direct qemu in EmulatorPool):
#   1. Env-node setup collapses to `docker pull` — no Android SDK / AVD /
#      apps on the host filesystem.
#   2. Per-worker isolation is native — containers replace per-worker AVD
#      copies + boot-serializer flock.
#   3. Cleanup becomes a one-line `docker rm -f` filter.
#
# Apps install needs KVM at build time, but buildx's `security.insecure`
# entitlement does NOT bind-mount /dev/kvm into the buildkit step's
# container. Instead, install.sh runs a two-stage build:
#   1. `docker build` this Dockerfile → cua-lite/androidworld:base (no apps).
#   2. `docker run --device /dev/kvm --privileged` a builder container,
#      `docker exec` it to run apps.sh (which boots the emulator
#      and installs the benchmark apps), then `docker commit` the result
#      as cua-lite/androidworld:latest.
#
# Build: bash lite/gym/envs/androidworld/scripts/install.sh
# Run:   spawned per-worker by AndroidWorldContainerPool.
#
# Build context is just this directory; no large pre-bundled assets like
# androidlab's docker-file.zip — everything is fetched from upstream
# during build.

FROM python:3.11-bullseye

# ── 1. apt deps ──────────────────────────────────────────────────────────────
# JDK 17 (sdkmanager + emulator are happy with 17, unlike androidlab which
# pins JDK 8 for its older toolchain). The rest is the qemu/emulator
# runtime library set — same family as androidlab's apt block.
RUN apt-get update && apt-get install -y --no-install-recommends \
      openjdk-17-jdk-headless \
      wget unzip curl ca-certificates \
      libpulse0 libnss3 libxcomposite1 libxdamage1 libxrandr2 libxtst6 \
      libgbm1 libasound2 libdbus-1-3 libdrm2 libglib2.0-0 libgl1 \
      libfontconfig1 libxcb-image0 libxcb-icccm4 libxcb-keysyms1 \
      libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-shm0 \
      libxcb-sync1 libxcb-xfixes0 libxcb-xinerama0 libxcb-xkb1 \
      libxkbcommon-x11-0 libxkbcommon0 libegl1 libgles2 libglu1-mesa \
      libxi6 libxcursor1 libxrender1 libgoogle-perftools4 \
      ffmpeg \
 && rm -rf /var/lib/apt/lists/*
# ffmpeg is consumed by androidworld's per-task initialize_task for tasks
# that generate audio fixtures (e.g. RetroSavePlaylist seeds .mp3 files
# via pydub, which shells out to ffmpeg). Without it,
# user_data_generation._create_test_mp3 raises FileNotFoundError and
# /task/initialize returns 500, leaving the host with self._task=None
# and the trajectory's reward stuck at None.

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV ANDROID_SDK_ROOT=/root/Android/Sdk
ENV ANDROID_AVD_HOME=/root/.android/avd
ENV PATH=$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/emulator:$PATH
ENV ANDROID_AVD_NAME=lite_avd_androidworld

# ── 2. Android SDK cmdline-tools ─────────────────────────────────────────────
RUN mkdir -p "$ANDROID_SDK_ROOT/cmdline-tools" && cd /tmp \
 && curl -sLO https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip \
 && unzip -q commandlinetools-linux-11076708_latest.zip -d "$ANDROID_SDK_ROOT/cmdline-tools" \
 && mv "$ANDROID_SDK_ROOT/cmdline-tools/cmdline-tools" "$ANDROID_SDK_ROOT/cmdline-tools/latest" \
 && rm commandlinetools-linux-11076708_latest.zip

# ── 3. Platform-tools + emulator + AVD system image ──────────────────────────
# build-tools;33.0.0 matches the upstream androidworld Dockerfile so aapt
# / apksigner are on PATH if any benchmark task helper needs them.
RUN yes | sdkmanager --licenses >/dev/null 2>&1 || true \
 && sdkmanager "platform-tools" "emulator" "build-tools;33.0.0" \
                "platforms;android-33" \
                "system-images;android-33;google_apis;x86_64"

# ── 4. Create AVD ────────────────────────────────────────────────────────────
# pixel_6: matches the upstream host-direct install.sh. Switch to pixel_5 if
# upstream androidworld ever changes the canonical device profile.
RUN echo "no" | avdmanager create avd \
      --name "$ANDROID_AVD_NAME" \
      --package "system-images;android-33;google_apis;x86_64" \
      --device "pixel_6"

# ── 5. android-world + android-env + in-container-server deps ────────────────
# android-world / android-env: needed at image-build time so apps.sh
# can call ``env_launcher.load_and_setup_env(emulator_setup=True)``, and
# at runtime by ``server.py`` (the in-container env-server) which hosts
# env_launcher inside the container so the a11y forwarder's reverse
# gRPC stays in-container.
# fastapi + uvicorn: HTTP RPC server consumed by the host's
# ``_RemoteAndroidWorldEnv`` proxy.
#
# Fork-free build (D5/D5a): clone UPSTREAM google-research/android_world at
# the pinned full SHA, then apply the cua-lite deltas from docker/patches/
# (two reviewable build-config patches — pyproject.toml + setup.py — that
# relax the upstream dependency pins and drop the legacy ``pkg_resources``
# import). No dependency on any cua-lite fork repo.
#
# Pin setuptools<70 and use --no-build-isolation:
# * upstream android-world's setup.py imports ``pkg_resources`` (legacy
#   setuptools API). setuptools 70+ moved pkg_resources to a separate
#   ``pkg_resources`` distribution; its absence breaks the wheel build.
#   (The setup.py patch also removes that import, but keep the pin as a
#   belt-and-suspenders for any other transitive legacy use.)
# * Pinning setuptools<70 in the system is necessary BUT not sufficient
#   because PEP 517 builds spin up an ISOLATED build env that ships its
#   own modern setuptools. ``--no-build-isolation`` makes pip reuse the
#   system setuptools (the pinned <70 one) → the build sees
#   pkg_resources → succeeds.
COPY patches/ /tmp/patches/
RUN mkdir -p /opt/android_world && cd /opt/android_world \
 && git init -q \
 && git fetch -q --depth 1 https://github.com/google-research/android_world.git \
        d9c569f764b3a5629321858de03ff653d0f24056 \
 && git checkout -q d9c569f764b3a5629321858de03ff653d0f24056 \
 && git apply /tmp/patches/pyproject.toml.patch /tmp/patches/setup.py.patch \
 # androidworld's setup.py generates information_retrieval/proto/*_pb2.py via
 # grpc_tools.protoc. Under --no-build-isolation pip ignores its setup_requires,
 # so grpc_tools must already be importable — else the protoc step is silently
 # skipped and `import state_pb2` blows up at task time. Pre-install it here.
 && pip install --no-cache-dir "grpcio-tools>=1.62.0" \
 && pip install --no-cache-dir --no-build-isolation -e . \
 && pip install --no-cache-dir \
      'android-env==1.2.3' \
      imageio-ffmpeg \
      fastapi \
      'uvicorn[standard]' \
      Pillow \
      Faker \
      python-dateutil \
      pytz

# ── 5b. Pre-bake accessibility_forwarder.apk + patch wrapper ─────────────────
# android_env's a11y_grpc_wrapper._get_accessibility_forwarder_apk()
# downloads a 4.5 MB APK from
# storage.googleapis.com on EVERY ``/init`` call at runtime — no cache.
# Under c=32 concurrent container boots this floods slirp4netns's
# single-thread DNS forwarder (rootless-docker /etc/resolv.conf points to
# 10.0.2.3, a slirp4netns userspace stub), all 32 boots get
# "Temporary failure in name resolution" → init 500 → main.py's defensive
# retry destroys + recreates each container 3× → all 32 tasks fail.
# Empirically observed in v14 rollout: 30+ task FAILs from this exact path.
#
# Fix: bake the APK into the image (one curl at build time, on this
# layer) and patch the wrapper to read it from local disk instead of
# calling urlopen. Runtime is then fully offline — no DNS, no HTTPS,
# no slirp4netns DNS contention.
# Pinned to the same APK version (2024.05.13) that upstream android_env
# 1.2.3 expects. ``curl --retry 5`` covers a transient Google Storage
# blip at build time.
RUN curl -sLf --retry 5 --retry-delay 3 \
        -o /usr/local/bin/accessibility_forwarder.apk \
        https://storage.googleapis.com/android_env-tasks/2024.05.13-accessibility_forwarder.apk

# Apply the a11y dep-patch via a reviewable script (docker/patches/) instead
# of an inline heredoc — the regex-substitution of
# ``_get_accessibility_forwarder_apk`` (pinned to android_env==1.2.3) lives in
# docker/patches/a11y_patch.py, which keeps the build-time self-assert.
# (patches/ was already COPY'd into /tmp/patches/ at step 5.)
RUN python3 /tmp/patches/a11y_patch.py

# Smoke-test the patch — fail the build if the function no longer reads
# the local file (e.g. upstream android_env changes the function name).
RUN python3 -c "from android_env.wrappers.a11y_grpc_wrapper import _get_accessibility_forwarder_apk as f; \
                assert f() == open('/usr/local/bin/accessibility_forwarder.apk','rb').read(), 'patched _get_accessibility_forwarder_apk returned unexpected bytes'; \
                print('a11y_grpc_wrapper patch verified')"

# ── 6. Stage apps.sh + server.py into the image ────────────────────────
# apps.sh is invoked once at image-build time via ``docker exec`` against
# a privileged container that has /dev/kvm passed through; after commit it
# stays in the image as a no-op artifact (~4 KB).
# server.py is the runtime FastAPI app; container_pool launches it as a
# long-lived ``docker exec -d`` after the emulator boots.
COPY apps.sh /usr/local/bin/apps.sh
COPY server.py /usr/local/bin/server.py
RUN chmod +x /usr/local/bin/apps.sh /usr/local/bin/server.py

# ── 7. Standard Android emulator port triple ─────────────────────────────────
# Internal ports are fixed; the host-side ports are dynamic and assigned
# by AndroidWorldContainerPool at `docker run -p` time.
EXPOSE 5554 5555 8554
