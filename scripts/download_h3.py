#!/usr/bin/env python3
"""Selective download of MiniMax/MiniMax-H3 (T2VA components only) from ModelScope.

Skips FL2VA / Ref2VA / transformer_ref / assets (~330 GiB) which are only needed
for image-to-video / reference variants. Downloads ~134 GiB: text_encoder,
transformer, vae, audio_vae, schedulers, tokenizer, processor + root configs.
Resumable: re-running continues from partial files.
"""
import sys, time
from modelscope import snapshot_download

MODEL_ID = "MiniMax/MiniMax-H3"
LOCAL_DIR = "/mnt/workspace/MiniMax-H3"
IGNORE = ["FL2VA/*", "Ref2VA/*", "transformer_ref/*", "assets/*"]

t0 = time.time()
print(f"[download] model={MODEL_ID} -> {LOCAL_DIR}", flush=True)
print(f"[download] ignore={IGNORE}", flush=True)
path = snapshot_download(
    MODEL_ID,
    local_dir=LOCAL_DIR,
    ignore_patterns=IGNORE,
    max_workers=8,
)
dt = time.time() - t0
print(f"[download] DONE in {dt/60:.1f} min -> {path}", flush=True)
