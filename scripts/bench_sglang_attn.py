#!/usr/bin/env python3
"""Time one MiniMax-H3 DiT self-attention call under each kernel path.

Shape matches the 1344x768 / 124-frame T2VA fixture: one packed document of
37736 live tokens, 56 heads x 128 dims, BF16. SGLang pads the packed sequence
to a multiple of 64 and passes cu_seqlens=(0, used, total).

Run with the SGLang PYTHONPATH:
  PYTHONPATH=.sglang-runtime:.sglang-probe python3 scripts/bench_sglang_attn.py
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F
from sageattention import sageattn

from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func
from sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn import (
    SageAttentionImpl,
)

USED, HEADS, DIM = 37736, 56, 128
TOTAL = (USED + 63) // 64 * 64
SCALE = DIM**-0.5


def bench(fn, iters: int = 10) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def main() -> None:
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(TOTAL, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    )
    cu = torch.tensor([0, USED, TOTAL], device="cuda", dtype=torch.int32)
    bounds = (0, USED, TOTAL)
    sage_impl = SageAttentionImpl(
        HEADS, DIM, causal=False, softmax_scale=SCALE, packed_trailing_padding=True
    )
    q1, k1, v1 = (x[:USED].unsqueeze(0) for x in (q, k, v))
    qh, kh, vh = (x.transpose(1, 2).contiguous() for x in (q1, k1, v1))

    results = {
        # SGLang default DiT path (FlashAttentionImpl.forward_varlen, ver=3).
        "sglang_fa_varlen": bench(
            lambda: flash_attn_varlen_func(
                q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                max_seqlen_q=USED, max_seqlen_k=USED,
                softmax_scale=SCALE, causal=False, ver=3,
            )
        ),
        # SGLang sage_attn path: slice live rows, sageattn, zero-pad tail.
        "sglang_sage_varlen": bench(
            lambda: sage_impl.forward_varlen(
                q, k, v, cu_seqlens=cu, max_seqlen=USED, cu_seqlens_host=bounds
            )
        ),
        # Bare kernel both frameworks reach (diffusers _sage_attention).
        "sageattn_nhd": bench(
            lambda: sageattn(q1, k1, v1, tensor_layout="NHD", sm_scale=SCALE)
        ),
        # diffusers native backend on this machine (torch flash kernel).
        "torch_sdpa": bench(lambda: F.scaled_dot_product_attention(qh, kh, vh)),
    }
    fa = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=USED,
        max_seqlen_k=USED, softmax_scale=SCALE, causal=False, ver=3,
    )
    fa = (fa[0] if isinstance(fa, tuple) else fa)[:USED].float()
    sage = sage_impl.forward_varlen(
        q, k, v, cu_seqlens=cu, max_seqlen=USED, cu_seqlens_host=bounds
    )[:USED].float()
    results["sage_vs_fa_cosine"] = F.cosine_similarity(
        fa.flatten(), sage.flatten(), dim=0
    ).item()
    results["shape"] = {"used": USED, "total": TOTAL, "heads": HEADS, "dim": DIM}
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v
                      for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
