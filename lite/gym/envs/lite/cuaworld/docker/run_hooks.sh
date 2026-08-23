#!/usr/bin/env bash
# Runs a gym-anything software's upstream install/setup hooks INSIDE the image
# build (COPY'd to /workspace by docker/Dockerfile, then RUN). This is a
# committed, reviewable script so the Dockerfile can stay static
# (container hook contract).
#
# Reads the software's env.json (fetched from the materials repo into the build
# context) and runs its pre_start hook against the in-container desktop unix
# user `ga`. post_start belongs to the running environment lifecycle and is
# executed once after each container boots, when the X desktop is available.
#
# NOTE: gym-anything's original hooks target an unprivileged user named `ga`
# (`su - ga -c`, `/home/ga`). The base image is built with `--build-arg USER=ga`
# (cua-lite/lite.cuaworld.base) so the container's desktop user IS `ga`,
# matching the assets. There is no ga->user rewrite; any CUA-Lite integration
# adaptation has already happened in the staged build context before this script
# runs it.
set -euo pipefail
cd /workspace
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

[ -x /usr/bin/python3 ] || { echo "FATAL: system python3 absent in base image" >&2; exit 1; }
[ -f env.json ] || { echo "FATAL: env.json missing from build context" >&2; exit 1; }

read_hook() { /usr/bin/python3 -c "import json,sys;print(json.load(open('env.json'))['hooks'].get(sys.argv[1],''))" "$1"; }
PRE_B="$(basename "$(read_hook pre_start)")"
[ -n "${PRE_B}" ] || { echo "FATAL: env.json has no pre_start hook" >&2; exit 1; }

echo "==> pre_start: scripts/${PRE_B}"
DEBIAN_FRONTEND=noninteractive bash "/workspace/scripts/${PRE_B}"

chown -R ga:ga /home/ga 2>/dev/null || true

# Optional per-env post_build (from materials), e.g. a boot-time app launcher.
if [ -f /workspace/post_build.sh ]; then
  echo "==> post_build.sh"
  bash /workspace/post_build.sh
fi
echo "==> hooks complete"
