#!/bin/bash
# Install deps for the Cua envs (cua.sandbox + cua.bench.*).
#
# There is NO cua-lite-owned image here — the sandboxes are Cua-managed (cloud
# or local via Cua's own runtimes). So this just pip-installs the SDKs and
# checks auth. (Per-env install.sh convention; mirrors screenspot_pro/captcha
# which pip-install their own third-party deps rather than using a pyproject
# extra — keeps the base install lean.)
#
# Usage:
#   uv run --no-sync bash lite/gym/envs/cua/scripts/install.sh
#
# Cloud (cua.ai) needs CUA_API_KEY. cua.bench's default `webtop` provider
# (local Playwright) and cua.sandbox `local=True` need no key.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# No image here, but the `uv run --no-sync bash` invocation contract (bare `python`
# + `uv pip`, never bare pip) is the same one every install.sh signs; it and
# ``require_uv`` live in the shared helper.
source "$SCRIPT_DIR/../../../scripts/image_build.sh"

# cua-lite uses the cua-sandbox SDK + drives cua-bench's env directly (make/reset/step/evaluate).
# It does NOT use cua-agent (cua's model-driven runner) — cua-lite has its own agents. But the `cua`
# meta-package pulls `cua-agent[cloud]`, AND cua-bench hard-declares `cua-agent` as a core dep; both
# drag in transformers/torch/tokenizers that CLASH with cua-lite's pinned qwen ML stack
# (transformers>=5 / sglang / torch<2.11) in the shared venv. So install the SDK + cua-bench WITHOUT
# cua-agent: cua-bench via --no-deps, then its own declared deps minus cua-agent (cua_agent is only
# imported lazily inside cua-bench's agent runners / CLI, never on the env task path cua-lite uses).
# Idempotent (mirrors webgym's install_host_evaluator): skip if both are already importable.
if python -c "import cua_sandbox, cua_bench" >/dev/null 2>&1; then
    echo "[cua install] cua-sandbox + cua-bench already importable; skipping pip install." >&2
else
    echo "[cua install] installing cua-sandbox SDK + cua-bench (excluding cua-agent) ..." >&2
    require_uv
    uv pip install -q cua-sandbox cua-computer
    uv pip install -q --no-deps cua-bench
    # cua-bench's CORE deps ONLY, minus cua-agent (read from metadata so the list never drifts).
    # CRITICAL: skip every `extra ==` optional dep — the `rl` extra pulls torch>=2.12.1 +
    # transformers, `agents` pulls cua-agent/anthropic, etc., which would re-drag the exact
    # torch/transformers clash with cua-lite's pinned qwen stack this whole dance avoids. Keep the
    # full requirement string (incl. non-extra python_version/platform markers) for pip to evaluate.
    CB_REQ="$(mktemp)"
    python -c "
import importlib.metadata as m
for r in (m.requires('cua-bench') or []):
    if 'extra ==' in r or 'cua-agent' in r.lower():
        continue
    print(r.strip())
" > "$CB_REQ"
    uv pip install -q -r "$CB_REQ"
    rm -f "$CB_REQ"
fi

# Download the cua-bench task datasets (basic / KiCad / workflows, ~3 MB) into .cache/
# so cua.bench.* registers with NO $CUA_BENCH_DATASET_ROOT export needed (mirrors
# osworld_g's .cache clone). Idempotent: skip if already present. The env var still
# overrides the default to point at any single dataset dir.
CACHE_DIR="$SCRIPT_DIR/../.cache/datasets"
if [ -d "$CACHE_DIR" ] && [ -n "$(ls -A "$CACHE_DIR" 2>/dev/null)" ]; then
    echo "[cua install] datasets already present ($CACHE_DIR) — skip" >&2
else
    echo "[cua install] downloading cua-bench datasets into $CACHE_DIR ..." >&2
    TMP="$(mktemp -d)"
    if git clone --depth 1 --filter=blob:none --sparse https://github.com/trycua/cua "$TMP" 2>/dev/null \
       && git -C "$TMP" sparse-checkout set libs/cua-bench/datasets 2>/dev/null; then
        mkdir -p "$CACHE_DIR"
        cp -R "$TMP"/libs/cua-bench/datasets/. "$CACHE_DIR/"
        echo "[cua install] datasets ready: $(ls "$CACHE_DIR" | tr '\n' ' ')" >&2
    else
        echo "[cua install] WARN: could not fetch cua-bench datasets — set \$CUA_BENCH_DATASET_ROOT manually." >&2
    fi
    rm -rf "$TMP"
fi

# cua.bench's default `webtop`/`simulated` provider is a Playwright HTML desktop.
# pip installs the `playwright` package but NOT the browser binary — download it,
# else webtop reset() fails with "Executable doesn't exist ... playwright install".
echo "[cua install] installing Playwright chromium (cua.bench webtop/simulated) ..." >&2
python -m playwright install chromium || \
    echo "[cua install] WARN: 'playwright install chromium' failed — webtop needs it; rerun manually." >&2

# Native cua.bench (linux-docker) + cua.sandbox local=True boot a real desktop in
# Docker (trycua/cua-xfce). The SDK pulls it lazily on first use; pre-pull here so
# the first run isn't a multi-GB surprise. Best-effort — skip if Docker is absent.
if command -v docker >/dev/null 2>&1; then
    echo "[cua install] pre-pulling trycua/cua-xfce (native cua.bench + cua.sandbox local) ..." >&2
    docker pull trycua/cua-xfce:latest || \
        echo "[cua install] WARN: could not pre-pull trycua/cua-xfce — SDK will pull on first native/local use." >&2
else
    echo "[cua install] NOTE: docker not found — native cua.bench + cua.sandbox local=True need Docker" \
         "(QEMU suites also need KVM). webtop (default) and cloud don't." >&2
fi

if [ -z "${CUA_API_KEY:-}" ]; then
    echo "[cua install] NOTE: CUA_API_KEY is unset — needed for Cua cloud. " \
         "cua.bench webtop (default) and cua.sandbox local=True work without it." >&2
fi

echo "[cua install] done." >&2
