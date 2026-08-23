#!/usr/bin/env bash

#
# Detect NVLink for NCCL_NVLS_ENABLE in the Ray runtime env.
#
# Sourced by run_grpo.sh / run_reinforce.sh / run_dagger.sh / run_sft.sh:
#
#   source "${CUA_LITE_ROOT}/scripts/train/utils/nvlink.sh"
#
# Sets HAS_NVLINK to 1 if `nvidia-smi topo -m` reports any NVLink edges.

_NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$_NVLINK_COUNT" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $_NVLINK_COUNT NVLink references)"
unset _NVLINK_COUNT
