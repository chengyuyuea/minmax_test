#!/usr/bin/env bash
# Kernel attribution for the Sage write-up: one denoise forward (--steps 2) under SDPA
# and under Sage, so attention's share of step time and Sage's kernel speedup are both
# measured on the same machine as every other number in the guide.
set -u
cd /mnt/workspace
# Wait for the FBCache sweep to release the GPU; polls a file, not pgrep, so the loop
# can never match its own command line.
until [ "$(wc -l < logs/fbc_sweep.done 2>/dev/null || echo 0)" -ge 2 ]; do sleep 30; done
P='A cinematic shot of a red fox trotting through a snowy pine forest at dawn, soft volumetric light, shallow depth of field'
for backend in native sage; do
  if [ "$backend" = sage ]; then pre=sage_; else pre=; fi
  id="${pre}steps2-$(date +%Y%m%d-%H%M%S)"
  env DIFFUSERS_ATTN_BACKEND=$backend PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 scripts/run_h3.py --prompt "$P" --steps 2 --seed 0 --profile \
      --tag "profile-$backend" --run-id "$id" > "logs/${id}.log" 2>&1
  echo "$id exit=$?" >> logs/profile.done
done
