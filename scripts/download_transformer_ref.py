#!/usr/bin/env python3
"""Download ONLY the transformer_ref/ component (~62 GiB) from ModelScope.

transformer_ref is the reference-conditioning transformer used by the diffusers
ModularPipeline for the Ref2VA workflow. It was skipped in the initial T2VA-only
download. FL2VA needs no extra weights (it reuses the main transformer), and the
self-contained FL2VA/ Ref2VA/ dirs (134 GiB each) are vLLM-Omni serving copies
that the diffusers path never loads -- so we fetch just this one component.
Resumable: re-running continues from partial files.
"""
import time
from modelscope import snapshot_download

MODEL_ID = "MiniMax/MiniMax-H3"
LOCAL_DIR = "/mnt/workspace/MiniMax-H3"
ALLOW = ["transformer_ref/*"]

t0 = time.time()
print(f"[dl] {MODEL_ID} allow={ALLOW} -> {LOCAL_DIR}", flush=True)
path = snapshot_download(
    MODEL_ID,
    local_dir=LOCAL_DIR,
    allow_patterns=ALLOW,
    max_workers=8,
)
print(f"[dl] DONE in {(time.time()-t0)/60:.1f} min -> {path}", flush=True)
