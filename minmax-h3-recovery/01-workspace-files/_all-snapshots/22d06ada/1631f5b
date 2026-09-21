#!/usr/bin/env python3
"""Selectively download Turbo LoRA weights (safetensors + config only, skip huge .bin files)."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

local_dir = "/mnt/workspace/explore/turbo-lora-A"
os.makedirs(local_dir, exist_ok=True)

# Only download safetensors, json, py, txt, md — skip the huge experimental_*.bin files
print(f"Downloading larryvrh/MiniMax-H3-Turbo-Lora (safetensors+config only) → {local_dir}")
path = snapshot_download(
    repo_id="larryvrh/MiniMax-H3-Turbo-Lora",
    local_dir=local_dir,
    ignore_patterns=["*.bin"],  # skip 10+ GB experimental checkpoints
)
print(f"Done. Files saved to: {path}")
