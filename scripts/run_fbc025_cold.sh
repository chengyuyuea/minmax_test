#!/usr/bin/env bash
# FBCache 0.25 cold single-video runs under SDPA and Sage, so the guide's main tables use
# the same measurement (cold, one video, no --reuse-text-embeds) as every other column.
set -u
cd /mnt/workspace
P='A cinematic shot of a red fox trotting through a snowy pine forest at dawn, soft volumetric light, shallow depth of field'
for backend in native sage; do
  if [ "$backend" = sage ]; then pre=sage_; else pre=; fi
  id="${pre}fbcache025-$(date +%Y%m%d-%H%M%S)"
  env DIFFUSERS_ATTN_BACKEND=$backend PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 scripts/run_h3.py --prompt "$P" --steps 50 --seed 0 --drop-page-cache \
      --cache-dit --cache-dit-threshold 0.25 --tag "${pre}fbcache025-cold" --run-id "$id" \
      > "logs/${id}.log" 2>&1
  echo "$id exit=$?" >> logs/fbc025_cold.done
done
