#!/usr/bin/env python3
"""Download only the essential Turbo LoRA files (main safetensors + config + scripts)."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import hf_hub_download

local_dir = "/mnt/workspace/explore/turbo-lora-A"
os.makedirs(local_dir, exist_ok=True)

repo_id = "larryvrh/MiniMax-H3-Turbo-Lora"

# Files to download — only what we need
files = [
    "minimax_h3_turbo_4step.safetensors",      # ~780 MB - main LoRA
    "minimax_h3_t2v_turbo.json",                # config
    "generate.py",                               # usage reference
    "README.md",                                 # docs
    "requirements.txt",                          # deps info
]

for fname in files:
    print(f"Downloading {fname}...")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=fname,
        local_dir=local_dir,
    )
    size_mb = os.path.getsize(path) / 1e6
    print(f"  → {path} ({size_mb:.1f} MB)")

print("\nDone! All files saved to:", local_dir)
