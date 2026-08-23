#!/usr/bin/env bash
# Generate and verify lite.scalecua runtime task catalogs.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$ENV_DIR/../../../../.." && pwd)"

run_py() {
    PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python "$@"
}

generate() {
    run_py "$SCRIPT_DIR/import_tasks.py" "$@"
}

check() {
    run_py - <<'PY'
from lite.gym.envs.lite.scalecua.src.utils import dataset

dataset.validate_catalog_lock()
dataset.validate_runtime_cache()
dataset.validate_all(strict_counts=True)
print("catalogs and runtime cache fresh")
PY
}

refresh_lock() {
    run_py - <<'PY'
from lite.gym.envs.lite.scalecua.src.utils import dataset

dataset.write_catalog_lock()
dataset.validate_catalog_lock()
PY
}

case "${1:-check}" in
    generate) shift; generate "$@" ;;
    check) check ;;
    refresh-lock) refresh_lock ;;
    status) check ;;
    *)
        echo "Usage: $0 [generate|check|refresh-lock|status]" >&2
        exit 1
        ;;
esac
