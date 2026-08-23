#!/bin/bash
# Regenerate the checked-in Online-Mind2Web task manifest (data/tasks.json) that
# the host reads at module-import time. The 300-task list lives in the GATED HF
# dataset osunlp/Online-Mind2Web, so the host never network-loads it at import
# (webgym/webharbor.webvoyager pattern) — this script is the single, reproducible source.
#
# Re-run this only when the upstream pin (server_kwargs.hf_revision in
# configs/default.yaml) moves. Output is deterministic: sorted keys, no
# output-changing knobs, so a clean re-run produces a byte-identical file
# (the manifest drift-check).
#
# Usage:
#   uv run --no-sync bash lite/gym/envs/online_mind2web/scripts/utils/tasks.sh
#
# Pre-req: huggingface_hub installed (a dev/generation-time dep, NOT a host
# runtime dep) and an HF token with access to the gated osunlp/Online-Mind2Web
# dataset (`huggingface-cli login` or export HF_TOKEN).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd -- "$SCRIPT_DIR/../.." &>/dev/null && pwd)"
OUT="$ENV_DIR/data/tasks.json"

python - "$ENV_DIR" "$OUT" <<'PY'
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

from lite.gym.utils import config

env_dir, out_path = sys.argv[1], sys.argv[2]
cfg = config.load(env_dir)
repo_id = cfg.server_kwargs["hf_repo_id"]
filename = cfg.server_kwargs.get("hf_filename", "Online_Mind2Web.json")
revision = cfg.server_kwargs.get("hf_revision") or None

src = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, revision=revision)
rows = json.loads(Path(src).read_text(encoding="utf-8"))

# Faithful upstream fields, keyed by task_id; split is "eval" (whole benchmark
# is a held-out eval set). _convert_task() in main.py normalizes at load time.
manifest: dict[str, dict] = {}
for row in rows:
    tid = str(row["task_id"]).strip()
    if not tid:
        raise SystemExit(f"empty task_id in upstream row: {row!r}")
    if tid in manifest:
        raise SystemExit(f"duplicate task_id {tid!r} in upstream manifest")
    manifest[tid] = {
        "task_id": tid,
        "confirmed_task": row["confirmed_task"],
        "website": row["website"],
        "reference_length": row.get("reference_length"),
        "level": row.get("level"),
        "split": "eval",
    }

Path(out_path).write_text(
    json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(f"wrote {len(manifest)} tasks -> {out_path}")
PY
