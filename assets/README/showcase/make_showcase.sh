#!/usr/bin/env bash
# Reproduce the 5 benchmark showcase GIFs in the top-level README.
#
# Each GIF is one gpt-5.5 trajectory captured with rollout `--save-gif`.
# One task per benchmark, picked to be visually representative:
#
#   lite.demo      create_file                          desktop  (terminal: make a file)
#   lite.osworld   osworld_libreoffice_impress_05dd4c1d desktop  (LibreOffice Impress slides)
#   browsergym.webarena  task 21                         browser  (One Stop Market storefront)
#   androidworld  ContactsAddContact                   mobile   (Material "Create contact")
#   mobilegym      spotify.AddToQueueAndPlay            mobile   (Spotify: queue + play a song)
#
# Run from the repo root:
#   uv run bash assets/README/showcase/make_showcase.sh
#
# Prereqs:
#   - Each env installed (see docs/envs.md). lite.demo / lite.osworld /
#     androidworld run in-process ("direct" mode) — one local container each.
#   - OPENAI_API_KEY + OPENAI_BASE_URL exported (gpt-5.5; Azure or OpenAI).
#   - WebArena ONLY: needs an env-server; --warm-singleton prewarms the
#     WebArena app stack, which direct mode can't start. The server process must
#     also see OPENAI_API_KEY (the WebArena task evaluator reads it at
#     env.reset()). See the WebArena section below.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # showcase -> README -> assets -> repo root
cd "$ROOT"
OUT="assets/README/showcase"
LOGS=".logs/rollout/showcase"
MODEL="gpt-5.5"

: "${OPENAI_API_KEY:?export OPENAI_API_KEY for gpt-5.5}"

run() {  # run <env_id> <task_id> <out_name> [extra rollout args...]
  local env_id="$1" task_id="$2" name="$3"; shift 3
  echo "### $name : $env_id@$task_id"
  uv run python scripts/rollout.py --model-id "$MODEL" --env-id "$env_id" --task-id "$task_id" \
    --save-gif true --save-video false --log-root "$LOGS/$name" "$@"
  cp "$LOGS/$name"/task/*/sample_00/trajectory.gif "$OUT/$name.gif"
  echo "  -> $OUT/$name.gif"
}

# ── Desktop + mobile: direct mode (no env-server) ───────────────────────────
# Unset any env-server vars so these run in-process against a local container.
unset CUA_LITE_ENV_SERVER_URL CUA_LITE_ENV_SERVER_TOKEN || true

run lite.demo     create_file                          lite_demo     --config-path scripts/configs/gpt/default/lite.demo.yaml
run lite.osworld  osworld_libreoffice_impress_05dd4c1d lite_osworld  --config-path scripts/configs/gpt/default/lite.osworld.yaml
run androidworld ContactsAddContact                   androidworld --config-path scripts/configs/gpt/default/androidworld.yaml
run mobilegym     spotify.AddToQueueAndPlay            mobilegym     --config-path scripts/configs/gpt/default/mobilegym.yaml

# ── WebArena: via an env-server (singleton prewarm) ─────────────────────────
# WebArena's app stack (shopping/gitlab/forum/wikipedia) is brought up by the
# env-server, NOT by direct mode. Start one on a free port; its process MUST
# export OPENAI_API_KEY (the task evaluator needs it at reset). Example:
#
#   OPENAI_API_KEY="$OPENAI_API_KEY" OPENAI_BASE_URL="$OPENAI_BASE_URL" \
#     uv run python scripts/serve_env.py --port 30911 --token "$TOK" \
#       --env-ids browsergym.webarena --warm-singleton &
#   # singleton prewarm boots the stack (gitlab is the slow one, ~2-3 min)
#
# Then point the client at it and run (gpt config = coord action space + the
# WebArena homepage hint):
WA_URL="${CUA_LITE_ENV_SERVER_URL:-}"
WA_TOK="${CUA_LITE_ENV_SERVER_TOKEN:-}"
if [[ -n "$WA_URL" && -n "$WA_TOK" ]]; then
  echo "### webarena : browsergym.webarena@21 (via $WA_URL)"
  uv run python scripts/rollout.py --model-id "$MODEL" --env-id browsergym.webarena \
    --config-path scripts/configs/gpt/default/browsergym.webarena/default.yaml --task-id 21 \
    --save-gif true --save-video false --max-attempts 12 --log-root "$LOGS/webarena"
  cp "$LOGS/webarena"/task/*/sample_00/trajectory.gif "$OUT/webarena.gif"
  echo "  -> $OUT/webarena.gif"
else
  echo "### webarena : SKIPPED — set CUA_LITE_ENV_SERVER_URL + CUA_LITE_ENV_SERVER_TOKEN"
  echo "    (see the WebArena section in this script)"
fi

# Post-process every gif: normalize size, burn a thin gray border in, and
# compress. Desktop/browser -> 720px wide (uniform, so they line up in the README's
# left column); mobile -> 480px (portrait). The border must be baked in (GitHub
# strips <img> CSS), and a 128-color no-dither palette keeps the files small
# (dithering bloats flat UI screenshots) — without changing the README layout.
uv run python - "$OUT" <<'PY'
import sys, glob, os
from PIL import Image, ImageSequence, ImageOps
for path in sorted(glob.glob(f"{sys.argv[1]}/*.gif")):
    im = Image.open(path); frames=[]; durs=[]
    for fr in ImageSequence.Iterator(im):
        durs.append(fr.info.get("duration", 1100))
        f = fr.convert("RGB")
        w = 720 if f.width >= f.height else 480   # landscape (uniform) vs portrait (mobile)
        if f.width != w:
            f = f.resize((w, round(f.height * w / f.width)), Image.LANCZOS)
        f = ImageOps.expand(f, border=3, fill=(153, 153, 153))
        frames.append(f.quantize(colors=128, method=Image.MEDIANCUT, dither=Image.NONE))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durs, loop=0, optimize=True)
    print(f"  packed {os.path.basename(path)} -> {frames[0].size} {os.path.getsize(path)//1024}K")
PY

echo "done -> $OUT/*.gif"
