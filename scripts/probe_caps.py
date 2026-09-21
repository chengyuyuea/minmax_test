"""Hardware-capability probe: decide which accel candidates are silicon-gated
on this box vs merely a CUDA/build problem. Read-only, no env mutation."""
import torch

print("torch", torch.__version__)
print("cuda runtime", torch.version.cuda)
cc = torch.cuda.get_device_capability(0)
print("device", torch.cuda.get_device_name(0), "CC", cc)
sm = cc[0] * 10 + cc[1]

# FP8 tensor-core GEMM (needs sm_89 Ada / sm_90 Hopper)
try:
    a = torch.randn(16, 16, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(16, 16, device="cuda").to(torch.float8_e4m3fn).t()
    sa = torch.ones(1, device="cuda")
    sb = torch.ones(1, device="cuda")
    torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    print("FP8 scaled_mm: OK")
except Exception as e:
    print("FP8 scaled_mm: FAILED ->", str(e)[:140])

print("--- capability gates ---")
print(f"sm_{sm}")
print("FP8 tensor core (sm>=89):", sm >= 89)
print("FlashAttention-3 (sm==90 Hopper):", sm == 90)
print("NVFP4 / SageAttn3 FP4 (sm>=120 Blackwell):", sm >= 120)
print("bf16 (sm>=80):", sm >= 80)
