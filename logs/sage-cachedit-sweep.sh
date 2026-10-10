#!/bin/bash
# SGLang resident50 + Sage + Cache-DiT: R0.24/R0.32 at MC3, R0.24 at MC6/MC9.
cd /mnt/workspace
for cfg in "0.24 3" "0.32 3" "0.24 6" "0.24 9"; do
  set -- $cfg
  extra=""
  [ "$2" != "3" ] && extra="--cache-dit-mc $2"
  echo "=== sage cachedit thr=$1 mc=$2 $(date +%T)"
  python3 scripts/run_sglang_h3.py --dit-resident-layers 50 \
    --dit-attention-backend sage_attn \
    --cache-dit --cache-dit-threshold "$1" $extra --drop-page-cache
  echo "=== exit $? $(date +%T)"
done
