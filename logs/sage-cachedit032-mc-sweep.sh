#!/bin/bash
# SGLang resident50 + Sage + Cache-DiT R0.32 at MC6/MC9.
cd /mnt/workspace
for mc in 6 9; do
  echo "=== sage cachedit thr=0.32 mc=$mc $(date +%T)"
  python3 scripts/run_sglang_h3.py --dit-resident-layers 50 \
    --dit-attention-backend sage_attn \
    --cache-dit --cache-dit-threshold 0.32 --cache-dit-mc $mc --drop-page-cache
  echo "=== exit $? $(date +%T)"
done
