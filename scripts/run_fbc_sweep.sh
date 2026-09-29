#!/usr/bin/env bash
# Sage + FBCache threshold sweep, cold start, same fox prompt / seed as every other
# single-video run, so each point differs from sage_fbcache025 only in the threshold.
set -u
cd /mnt/workspace
P='A cinematic shot of a red fox trotting through a snowy pine forest at dawn, soft volumetric light, shallow depth of field'
for t in 015 020; do
  id="sage_fbcache${t}-$(date +%Y%m%d-%H%M%S)"
  env DIFFUSERS_ATTN_BACKEND=sage PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 scripts/run_h3.py --prompt "$P" --steps 50 --seed 0 --drop-page-cache \
      --cache-dit --cache-dit-threshold "0.${t:1}" --tag "sage-fbcache${t}-cold" --run-id "$id" \
      > "logs/${id}.log" 2>&1
  echo "$id exit=$?" >> logs/fbc_sweep.done
done
