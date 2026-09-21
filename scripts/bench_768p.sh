#!/usr/bin/env bash
# Regression fixture for MiniMax-H3 T2VA 768p.
#
# Locks the delivery-baseline geometry so every run stays comparable and only the
# lever under test varies through "$@". Locked: 1344x768, 124 frames, 50 steps,
# seed 0 (fps=24 is the pipeline default, not a CLI flag).
#
# Usage:
#   scripts/bench_768p.sh --tag baseline    --reuse-text-embeds --repeat 2
#   scripts/bench_768p.sh --tag fbcache     --cache-dit --reuse-text-embeds --repeat 2
#   scripts/bench_768p.sh --tag steps20 --steps 20   # later --steps wins over locked 50
#
# PITFALLS (expert-handoff §4), both structurally avoided here:
#   1. tag is NOT a positional argument. Always pass it as `--tag NAME` inside
#      "$@"; the original positional form let `--tag foo` collapse TAG into the
#      literal "--tag". Here everything is forwarded verbatim to run_h3.py.
#   2. --steps is set in the locked defaults FIRST; argparse lets a later value in
#      "$@" win, so appending `--steps N` overrides the locked 50 cleanly.
#
# run.log / run.json / metrics.csv / timeline.png are produced by RunMonitor
# inside run_h3.py, so no external tee is needed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_PROMPT="A cinematic shot of a red fox trotting through a snowy pine forest at dawn, soft volumetric light, shallow depth of field"

# Locked geometry first, caller overrides ("$@") last so argparse's last-wins
# semantics let any lever be flipped without editing this file.
exec python3 "$HERE/run_h3.py" \
    --prompt "${H3_BENCH_PROMPT:-$DEFAULT_PROMPT}" \
    --height 768 --width 1344 --num-frames 124 --steps 50 --seed 0 \
    "$@"
