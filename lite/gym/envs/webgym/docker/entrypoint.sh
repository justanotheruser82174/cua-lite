#!/bin/bash
# Entrypoint for the cua-lite/webgym container: run the OmniBoxes pool in the
# foreground (PID1). M = WEBGYM_INSTANCES (set by WebGymContainerServices.ensure
# from derive_pool_size_from_env). All ports are the OmniBoxes upstream defaults
# inside the container's private netns; only the master (7000) is published.
#
# redis 6379 (NOT the host's 6380): 6380 exists only to dodge Ray's GCS on the
# host; the container's private netns has no Ray, so the upstream default 6379 is
# correct. Do NOT "fix" it back to 6380.
set -euo pipefail
M="${WEBGYM_INSTANCES:-64}"
echo "[webgym container] launching OmniBoxes pool: M=${M} (redis=6379 master=7000 node=8080 instance-start=9000)" >&2
exec python /opt/omniboxes/omniboxes/deploy/deploy.py "${M}" \
  --redis-port 6379 \
  --master-port 7000 \
  --node-port 8080 \
  --instance-start-port 9000
