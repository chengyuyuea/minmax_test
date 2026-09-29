#!/usr/bin/env bash
# Load-time calibration: same prompt, steps=2, page cache evicted before each run.
# iter #0 = standard cold load, iter #1 (reuse text embeds) = standard warm load.
set -euo pipefail
cd /mnt/workspace
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PROMPT='A cinematic shot of a red fox trotting through a snowy pine forest at dawn, soft volumetric light, shallow depth of field'
for k in 1 2; do
  id="steps2_repeat2_reuse-$(date +%Y%m%d-%H%M%S)"
  python3 scripts/run_h3.py --prompt "$PROMPT" --steps 2 --seed 0 \
    --tag load-calib --reuse-text-embeds --repeat 2 --drop-page-cache \
    --run-id "$id" > "logs/$id.log" 2>&1
done
echo ALL_DONE
