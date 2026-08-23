#!/bin/bash

#
# Build Ray runtime-env JSON from KEY=VALUE pairs with proper JSON escaping.
# Ray workers do not inherit the launcher shell, so train launchers pass a
# narrow env_vars map through --runtime-env-json.
#
build_runtime_env_json() {
  python - "$@" <<'PY'
import json
import os
import re
import sys

env_vars = {}
for item in sys.argv[1:]:
    if "=" not in item:
        raise SystemExit(f"runtime-env entry must be KEY=VALUE, got {item!r}")
    key, value = item.split("=", 1)
    if not key:
        raise SystemExit(f"runtime-env entry has an empty key: {item!r}")
    env_vars[key] = value

extra_names = os.environ.get("CUA_LITE_RAY_ENV_VARS", "")
for name in re.split(r"[,\s:]+", extra_names.strip()):
    if not name:
        continue
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise SystemExit(f"CUA_LITE_RAY_ENV_VARS entry must be an env var name, got {name!r}")
    if name in os.environ:
        env_vars[name] = os.environ[name]

print(json.dumps({"env_vars": env_vars}, indent=2))
PY
}
