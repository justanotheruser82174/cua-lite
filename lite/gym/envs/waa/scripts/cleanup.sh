#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RUNTIME_ROOT="$(python "$SCRIPT_DIR/utils/config_value.py" server_kwargs runtime_root)"
RUNTIME_ROOT="$(realpath -m "${RUNTIME_ROOT/#\~/$HOME}")"

if [[ -n "${SESSION_ID:-}" ]]; then
  filter="${SESSION_ID//[^[:alnum:]_]/_}-waa-"
else
  filter="-waa-"
  echo "WARNING: removing all WindowsAgentArena containers on this Docker daemon." >&2
fi

mapfile -t containers < <(docker ps -aq --filter "name=$filter")
if (( ${#containers[@]} )); then
  docker rm -fv "${containers[@]}" >/dev/null
fi

python - "$RUNTIME_ROOT" "${SESSION_ID:-}" <<'PY'
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser().resolve() / "slots"
session = re.sub(r"[^A-Za-z0-9_]", "_", sys.argv[2])
if not root.is_dir():
    raise SystemExit(0)
result = subprocess.run(
    ["docker", "ps", "-a", "--format", "{{.Names}}"],
    check=True,
    capture_output=True,
    text=True,
)
live = {line.strip() for line in result.stdout.splitlines() if line.strip()}
removed = 0
for slot in root.iterdir():
    if not slot.is_dir() or "-waa-" not in slot.name:
        continue
    if session and f"-{session}-waa-" not in slot.name:
        continue
    if slot.name in live:
        continue
    shutil.rmtree(slot)
    removed += 1
print(f"removed {removed} orphan WindowsAgentArena runtime slots")
PY
