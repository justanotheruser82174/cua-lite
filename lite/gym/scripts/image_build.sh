#!/usr/bin/env bash
# Shared helpers for env install.sh scripts: stamp an image with its build-source
# hash, check freshness against it, pull/push it from GHCR — and enforce the one
# INVOCATION CONTRACT every install.sh shares (see below). The runtime freshness check
# (lite.gym.utils.backend.freshness.ContainerImage.ensure_runnable) reads the label
# this stamps, computed by the SAME python hash, so build and launch agree on what
# "fresh" means.
#
# Usage in an env's install.sh:
#   source "<repo>/lite/gym/scripts/image_build.sh"
#   build() {
#       image_is_fresh "$IMAGE" <env_id> && { echo "up to date"; return; }
#       image_rm "$IMAGE"                                        # drop stale tag
#       docker build -t "$IMAGE" --label "$(src_label <env_id>)" -f .../Dockerfile <ctx>
#   }
# <env_id> is any id accepted by backend.freshness.image_for() through an
# env-local image_spec.py provider.
# Docker-free envs (captcha, cua, screenspot_pro, browsergym) source this too, for
# ``require_uv`` alone — the contract below is theirs as well.
#
# ─── THE INVOCATION CONTRACT, stated ONCE for every install.sh ────────────────
# Every install.sh is invoked as ``uv run --no-sync bash <script>`` (CLAUDE.md /
# docs/envs.md#invocation). Two consequences used to be re-derived — and re-argued
# at length — in each script; they live here now:
#
#   * ``python`` is the venv interpreter, and is the ONLY spelling these scripts
#     may use. ``python3`` / ``${PYTHON:-python3}`` can resolve to a DIFFERENT
#     interpreter than the venv ``uv run`` activated (this host: a 3.10 outside,
#     3.12 inside), which is how a hash computed against one interpreter's ``lite``
#     tree gets stamped on an image the runtime freshness check then re-checks with another's.
#     The check below is only an EXISTENCE test — it catches "ran the script bare on
#     a host with no `python` at all", not "ran it bare on a host that happens to
#     have one". Obeying the contract is still the caller's job; this just turns the
#     most common violation into a message instead of `command not found`.
#   * ``uv pip``, never a bare ``pip`` — see ``require_uv``.
command -v python >/dev/null 2>&1 || {
    echo "FATAL: python not found. Run the install script as: uv run --no-sync bash <script>" >&2
    exit 1
}

# Guard a ``uv pip install``: call it immediately before one. Deliberately NOT a
# source-time check like ``python`` above — an install.sh that only builds an image
# needs no uv, and a missing tool should fail where it is actually used.
#
# ``uv pip``, never a bare-pip fallback: this venv ships NO pip, so a bare ``pip``
# resolves to /usr/local/bin/pip — a python3.10 interpreter — and installs into
# system dist-packages while every consumer runs .venv's python3.12. The old
# ``command -v uv || pip install`` fallback therefore exited 0 with the deps still
# MISSING, silently: webgym's host judge absent means every reward is 0.0, and
# cuaworld's verifier parsers absent means 141 tasks score a degraded check. uv is
# present by construction under the contract above; if it isn't, fail loud rather
# than install into the wrong interpreter.
require_uv() {
    command -v uv >/dev/null 2>&1 || {
        echo "FATAL: uv not found. Run this as: uv run --no-sync bash $0" >&2
        exit 1
    }
}

# Build-source hash for an env — the SINGLE source of truth (python), so
# install.sh and the runtime freshness check can't drift.
src_hash() { python -m lite.gym.utils.backend.freshness hash "$1"; }

# The ``--label`` argument to bake into ``docker build``:
#   docker build ... --label "$(src_label <env_id>)"
#
# An empty hash must abort the SCRIPT, not just this subshell: ``exit`` inside ``$( )``
# ends only the substitution, and bash discards its status when it is an argument word,
# so the caller would get ``--label ""`` and carry on. ``docker build --label ""``
# succeeds, and an image with no ``lite.src_hash`` reads STALE FOREVER — a full rebuild
# on every launch (androidlab: ~45 min / ~32 GB). Hence ``kill -s TERM $$``, which
# aborts during word expansion, before docker is exec'd. Doing it here instead of asking
# all 15 call sites to hoist the assignment is what makes the guard hold everywhere.
# (``errexit`` cannot do this job: command substitutions run with -e unset unless
# ``shopt -s inherit_errexit``. ``image_is_fresh`` needs no such guard — an empty
# ``want`` there just means "not fresh", which fails safe.)
src_label() {
    local h
    h="$(src_hash "$1")"
    [ -n "$h" ] || {
        echo "FATAL: empty lite.src_hash for env '$1' — refusing to stamp an unlabeled" >&2
        echo "  image (it would read STALE forever and rebuild on every launch)." >&2
        kill -s TERM $$     # abort the whole install.sh, not just this subshell
        exit 1
    }
    echo "lite.src_hash=$h"
}

# Drop an image tag, NEVER with ``-f``. This host is shared: ``docker image rm -f``
# on an image another tenant's container references deletes the image out from
# under that container. The container keeps running, but its image reference
# dangles — ``docker image inspect`` on it says "No such image" — so freshness
# checks and cleanup stop recognising it, and nobody who did not already know the
# container name can find it. Plain ``docker image rm`` untags when another tag
# remains and lets the daemon REFUSE while a container still holds the image,
# which costs nothing here: ``docker build -t`` re-points the tag anyway, and an
# uninstall that leaves one image behind is recoverable. Always exits 0; names
# the holders so an operator can decide.
image_rm() {
    docker image rm "$1" >/dev/null 2>&1 && return 0
    docker image inspect "$1" >/dev/null 2>&1 || return 0   # absent: nothing to drop
    echo "[image_build] kept $1: in use by container(s)" \
         "$(docker ps -a --filter "ancestor=$1" --format '{{.Names}}' | tr '\n' ' ')" >&2
}

# Read-only: returns 0 iff IMAGE ($1) exists AND was built from the current
# sources of env ($2). No side effects — the caller decides what to do (build()
# rebuilds, status() reports). A no-arg ``install.sh`` skips the build when this
# is true, so it auto-rebuilds exactly when a Dockerfile/patch/source changed.
image_is_fresh() {
    local have want
    # A missing image (or one without the label) yields an empty ``have`` — which
    # can't equal the non-empty ``want`` — so no separate existence probe is needed.
    have="$(docker image inspect --format '{{index .Config.Labels "lite.src_hash"}}' "$1" 2>/dev/null || true)"
    want="$(src_hash "$2")"
    [ -n "$want" ] && [ "$have" = "$want" ]
}

# Read-only status formatter for install.sh scripts. It deliberately treats hash
# computation errors as STALE instead of printing Python tracebacks into a human
# status command; build/push still call image_is_fresh directly and fail loud.
image_status_line() {
    local label="$1" image="$2" env_id="$3" state sz
    if image_is_fresh "$image" "$env_id" >/dev/null 2>&1; then
        state="FRESH"
    elif docker image inspect "$image" >/dev/null 2>&1; then
        state="STALE"
    else
        state="MISSING"
    fi
    if [ "$state" = "MISSING" ]; then
        echo "${label}: MISSING (${image})"
        return 1
    fi
    sz="$(docker image inspect --format='{{.Size}}' "$image" 2>/dev/null \
        | awk '{printf "%.1fGB", $1/1024/1024/1024}')"
    echo "${label}: ${state} (${image}${sz:+, $sz})"
    [ "$state" = "FRESH" ]
}

# ─── GHCR distribution (pull instead of build / publish) ──────────────────────
# The GHCR ref is just the local name with a registry prefix: local
# ``cua-lite/<env>:latest`` ↔ remote ``ghcr.io/cua-lite/<env>:latest``. The
# ``lite.src_hash`` label rides inside the image (survives push/pull/tag), so a
# pulled image is gate-checkable exactly like a locally-built one — no per-build
# hash tags needed.
_ghcr_ref() { echo "ghcr.io/${1}"; }   # $1 = local IMAGE (cua-lite/<env>:tag)

# do_push IMAGE ENV_ID — tag and push only if the local image is fresh for ENV_ID.
# Requires ``docker login ghcr.io`` with a write:packages token.
do_push() {
    local img="$1" env_id="${2:?do_push requires env_id}" remote want have
    remote="$(_ghcr_ref "$img")"
    want="$(src_hash "$env_id")"
    [ -n "$want" ] || {
        echo "FATAL: empty lite.src_hash for env '$env_id' — refusing to push $img" >&2
        return 1
    }
    docker image inspect "$img" >/dev/null 2>&1 \
        || { echo "ERROR: $img not present locally — build it first (install.sh)." >&2; return 1; }
    have="$(docker image inspect --format '{{index .Config.Labels "lite.src_hash"}}' "$img" 2>/dev/null || true)"
    if [ -z "$have" ]; then
        echo "ERROR: $img has no lite.src_hash label — refusing to push unlabeled image." >&2
        return 1
    fi
    if [ "$have" != "$want" ]; then
        echo "ERROR: $img is stale (${have:0:12}) vs current $env_id sources (${want:0:12}); rebuild first." >&2
        return 1
    fi
    docker tag "$img" "$remote"
    echo "[image_build] pushing $remote ..." >&2
    docker push "$remote"
}

# do_push_unchecked IMAGE — explicit escape hatch for non-env data images that do
# not carry lite.src_hash (currently WAA's qcow2 data image). Do not use for
# runnable env images.
do_push_unchecked() {
    local img="$1" remote; remote="$(_ghcr_ref "$img")"
    docker image inspect "$img" >/dev/null 2>&1 \
        || { echo "ERROR: $img not present locally." >&2; return 1; }
    docker tag "$img" "$remote"
    echo "[image_build] pushing unchecked data image $remote ..." >&2
    docker push "$remote"
}

# do_pull IMAGE ENV_ID — adopt the published image ONLY if its lite.src_hash
# matches the caller's local sources (else the runtime freshness check would flag it STALE).
# Reads the remote label without downloading layers when buildx is available.
do_pull() {
    local img="$1" env_id="$2" remote want have
    remote="$(_ghcr_ref "$img")"
    want="$(src_hash "$env_id")"
    [ -n "$want" ] || {
        echo "FATAL: empty lite.src_hash for env '$env_id' — refusing to pull $remote" >&2
        return 1
    }
    # Cheap pre-check: read the remote config label via buildx (no layer download).
    have="$(docker buildx imagetools inspect "$remote" \
              --format '{{ index .Image.Config.Labels "lite.src_hash" }}' 2>/dev/null || true)"
    if [ -n "$have" ] && [ "$have" != "$want" ]; then
        echo "[image_build] $remote was built from different sources (${have:0:12}) than your" >&2
        echo "  checkout (${want:0:12}); it would be STALE. Build locally: install.sh" >&2
        return 1
    fi
    docker pull "$remote" || { echo "ERROR: pull $remote failed (not published, or need docker login ghcr.io)." >&2; return 1; }
    # Definitive check on the actual pulled image (covers the no-buildx fallback).
    have="$(docker image inspect --format '{{index .Config.Labels "lite.src_hash"}}' "$remote" 2>/dev/null || true)"
    if [ -z "$have" ]; then
        docker image rm "$remote" >/dev/null 2>&1 || true
        echo "[image_build] pulled $remote is missing lite.src_hash; refusing to adopt." >&2
        return 1
    fi
    if [ "$have" != "$want" ]; then
        docker image rm "$remote" >/dev/null 2>&1 || true
        echo "[image_build] pulled $remote (${have:0:12}) != your sources (${want:0:12}); build locally." >&2
        return 1
    fi
    docker tag "$remote" "$img"     # adopt as the canonical local name the env runs
    echo "[image_build] pulled $img (fresh, src_hash ${want:0:12})" >&2
}
