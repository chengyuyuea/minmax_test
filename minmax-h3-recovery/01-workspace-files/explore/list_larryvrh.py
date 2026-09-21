#!/usr/bin/env python3
"""List files in larryvrh/MiniMax-H3-Turbo-Lora."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import list_repo_tree

print("Files in larryvrh/MiniMax-H3-Turbo-Lora:")
print("-" * 80)
total = 0
for item in list_repo_tree("larryvrh/MiniMax-H3-Turbo-Lora", recursive=True):
    size = getattr(item, "size", None)
    rtype = type(item).__name__
    if size is not None:
        total += size
        size_str = f"{size / 1e9:.2f} GB" if size > 1e9 else f"{size / 1e6:.1f} MB" if size > 1e6 else f"{size / 1e3:.1f} KB" if size > 1e3 else f"{size} B"
        print(f"  {item.path:<60} {size_str:>12}  [{rtype}]")
    else:
        print(f"  {item.path:<60} {'(dir)':>12}  [{rtype}]")
print("-" * 80)
print(f"Total file size: {total / 1e9:.2f} GB")
