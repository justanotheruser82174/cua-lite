#!/bin/bash
#
# Training cleanup is deliberately opt-in. On shared hosts, broad pkill patterns
# can stop unrelated jobs from the same Unix user; run with
# CUA_LITE_TRAIN_BROAD_CLEANUP=1 only inside a dedicated training container.
#

if [ "${CUA_LITE_TRAIN_BROAD_CLEANUP:-0}" != "1" ]; then
  echo "Skipping broad training cleanup; set CUA_LITE_TRAIN_BROAD_CLEANUP=1" \
       "inside a dedicated container to enable it." >&2
  return 0 2>/dev/null || exit 0
fi

pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
# wandb may leave a non-python GPU monitor behind; match exact binary names so
# unrelated processes whose cmdline merely mentions wandb are left alone.
pkill -9 -x wandb-core 2>/dev/null || true
pkill -9 -x wandb-xpu 2>/dev/null || true
sleep 3
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
pkill -9 -x wandb-core 2>/dev/null || true
pkill -9 -x wandb-xpu 2>/dev/null || true
rm -rf /tmp/ray/session_* 2>/dev/null || true
