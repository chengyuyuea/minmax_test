"""Method-based run names shared by run_h3.py and the rename of older runs.

A run is named `<method>-<YYYYMMDD>-<HHMMSS>`. <method> lists every lever that differs
from a plain 50-step T2VA generation, joined by `_` in a fixed order:

    batch<N>  fl2va  sage  fbcache<ttt>  turbolora  compile  tf32  nodevmap
    steps<S>  repeat<R>  reuse

fbcache<ttt> is the threshold x100, zero-padded (0.25 -> fbcache025). A run with no
lever at all is `raw`. Measurement conditions such as --drop-page-cache are not
methods and stay in run.json.
"""

from __future__ import annotations


def method_name(args: dict, attn_backend: str | None, n_prompts: int | None = None) -> str:
    parts: list[str] = []
    if n_prompts:
        parts.append(f"batch{n_prompts}")
    if args.get("image"):
        parts.append("fl2va")
    levers: list[str] = []
    if attn_backend == "sage":
        levers.append("sage")
    if args.get("cache_dit"):
        levers.append(f"fbcache{round(float(args['cache_dit_threshold']) * 100):03d}")
    if args.get("lora"):
        levers.append("turbolora")
    if args.get("compile"):
        levers.append("compile")
    if args.get("tf32"):
        levers.append("tf32")
    if args.get("no_load_opt"):
        levers.append("nodevmap")
    if args.get("steps", 50) != 50:
        levers.append(f"steps{args['steps']}")
    if args.get("repeat", 1) > 1:
        levers.append(f"repeat{args['repeat']}")
    if args.get("reuse_text_embeds"):
        levers.append("reuse")
    return "_".join(parts + (levers or ["raw"]))
