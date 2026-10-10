"""Method-based run names shared by run_h3.py and the rename of older runs.

A run is named `<method>-<YYYYMMDD>-<HHMMSS>`. <method> starts with the framework and
then lists every lever that differs from a plain 50-step T2VA generation, joined by `_`
in a fixed order:

    framework   diffusers | turbolora          (run_sglang_h3.py uses sglang)
    batch       batch<N>
    attention   sage
    cache       fbcache<ttt>                   (SGLang: cachedit<ttt>)
    other       fl2va  v4_8eval  compile  tf32  nodevmap  steps<S>  repeat<R>  reuse

fbcache<ttt> is the threshold x100, zero-padded (0.25 -> fbcache025). A run with no
lever at all is `<framework>_raw`. Measurement conditions such as --drop-page-cache are
not methods and stay in run.json.
"""

from __future__ import annotations


def method_name(args: dict, attn_backend: str | None, n_prompts: int | None = None) -> str:
    parts: list[str] = ["turbolora" if args.get("lora") else "diffusers"]
    if n_prompts:
        parts.append(f"batch{n_prompts}")
    if attn_backend == "sage":
        parts.append("sage")
    if args.get("cache_dit"):
        parts.append(f"fbcache{round(float(args['cache_dit_threshold']) * 100):03d}")
    if args.get("image"):
        parts.append("fl2va")
    if args.get("lora"):
        parts.append("v4_8eval")
    if args.get("compile"):
        parts.append("compile")
    if args.get("tf32"):
        parts.append("tf32")
    if args.get("no_load_opt"):
        parts.append("nodevmap")
    if args.get("steps", 50) != 50 and not args.get("lora"):
        parts.append(f"steps{args['steps']}")
    if args.get("repeat", 1) > 1:
        parts.append(f"repeat{args['repeat']}")
    if args.get("reuse_text_embeds"):
        parts.append("reuse")
    if len(parts) == 1:
        parts.append("raw")
    return "_".join(parts)
