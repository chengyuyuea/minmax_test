#!/usr/bin/env python
"""Phased single-GPU runner for MiniMax-H3 (t2va / fl2va).

The FL2VA half of the checkpoint is ~134 GiB of bf16 weights, which fits neither the
79 GiB of HBM on one A100-80GB nor the 122 GB of host RAM on this box, so neither
"everything on GPU" nor the usual "everything in RAM + model CPU offload" works here.

MiniMaxH3Blocks is a five-stage sequential pipeline and each stage only touches its own
components, so this driver runs the stages one at a time against a shared PipelineState
and frees each component group before loading the next. Peak residency becomes the
largest single component (text_encoder or transformer, ~62 GiB) instead of the whole
checkpoint.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h3_monitor import RunMonitor  # noqa: E402
from run_naming import method_name  # noqa: E402

MODEL_DIR = os.environ.get("H3_MODEL_DIR", "/mnt/workspace/MiniMax-H3")

# Stage -> components that must be resident while that stage runs.
STAGE_COMPONENTS = {
    "before_encode": ["image_processor"],
    "text_encoder": ["text_encoder", "tokenizer", "processor"],
    "vae_encoder": ["vae", "audio_vae"],
    "denoise": ["transformer", "scheduler", "audio_scheduler"],
    "decode": ["vae", "audio_vae", "video_processor"],
}


def host_used_gib() -> float:
    info = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, _, v = line.partition(":")
            info[k] = int(v.split()[0])
    return (info["MemTotal"] - info["MemAvailable"]) / 1024 / 1024


def page_cache_gib() -> float:
    info = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, _, v = line.partition(":")
            info[k] = int(v.split()[0])
    return (info.get("Cached", 0) + info.get("Buffers", 0)) / 1024 / 1024


def drop_model_page_cache(model_dir: str) -> dict:
    """Evict the checkpoint's file pages so every run starts from the same cold state.

    Load time is set by how much of the ~196 GiB snapshot the page cache still holds
    from whatever ran before, not by the lever under test: the same transformer read
    took 10 s warm and 160+ s cold. POSIX_FADV_DONTNEED on the model files needs no
    root (unlike drop_caches) and leaves the rest of the page cache alone.
    """
    before = page_cache_gib()
    t0 = time.time()
    n = 0
    for root, _, files in os.walk(model_dir):
        for name in files:
            fd = os.open(os.path.join(root, name), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            n += 1
    after = page_cache_gib()
    print(f"[cache] evicted {n} model files: page cache {before:.1f} -> {after:.1f} GiB "
          f"in {time.time() - t0:.1f}s", flush=True)
    return {"files": n, "cached_before_gib": round(before, 2), "cached_after_gib": round(after, 2)}


def report(tag: str) -> None:
    hbm = torch.cuda.memory_allocated() / 2**30
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"[mem] {tag:<28} host_used={host_used_gib():6.1f} GiB  "
        f"hbm={hbm:6.1f} GiB  hbm_peak={peak:6.1f} GiB",
        flush=True,
    )


def repoint_to_local(pipe, model_dir: str) -> None:
    """Make every component load from the local snapshot instead of the Hub.

    modular_model_index.json hardcodes `MiniMaxAI/MiniMax-H3`, so a --local-dir
    download would otherwise be ignored and re-fetched.
    """
    for spec in pipe._component_specs.values():
        if spec.pretrained_model_name_or_path is not None:
            spec.pretrained_model_name_or_path = model_dir


# nn.Module component names in MiniMax-H3 that accept device_map / .to().
_MODULE_COMPONENTS = {"text_encoder", "transformer", "transformer_ref", "vae", "audio_vae"}


def load_group(pipe, names: list[str], dtype: torch.dtype, use_device_map: bool = False) -> None:
    wanted = [n for n in names if getattr(pipe, n, None) is None]
    if not wanted:
        return
    t0 = time.time()
    # dtype is a hint only: components whose config pins one keep it -- text_encoder
    # and transformer are bfloat16 on disk, both VAEs are float32.
    if use_device_map:
        # Build a per-component device_map dict so only nn.Module components get
        # device_map="cuda" (tokenizers / schedulers / processors don't accept it).
        dm = {n: "cuda" for n in wanted if n in _MODULE_COMPONENTS}
        pipe.load_components(names=wanted, dtype=dtype, **(dict(device_map=dm) if dm else {}))
    else:
        pipe.load_components(names=wanted, dtype=dtype)
    for name in wanted:
        comp = getattr(pipe, name, None)
        if isinstance(comp, torch.nn.Module):
            if not use_device_map:
                comp.to("cuda")
            comp.eval()
            # Break the count down by dtype: these checkpoints are mixed (the bf16
            # transformer still keeps fp32 norms), so sizing off the first parameter
            # alone overstates the transformer as 123 GiB instead of its real ~62 GiB.
            by_dtype: dict[torch.dtype, list[int]] = {}
            for x in comp.parameters():
                slot = by_dtype.setdefault(x.dtype, [0, 0])
                slot[0] += x.numel()
                slot[1] += x.numel() * x.element_size()
            params = sum(v[0] for v in by_dtype.values())
            if params:
                mix = ", ".join(
                    f"{str(dt).replace('torch.', '')} {n / 1e9:.2f}B/{b / 2**30:.1f}GiB"
                    for dt, (n, b) in sorted(by_dtype.items(), key=lambda kv: -kv[1][1])
                )
                print(f"[load] {name}: {params / 1e9:.2f}B params -> {mix}", flush=True)
    torch.cuda.synchronize()
    missing = [n for n in names if getattr(pipe, n, None) is None]
    if missing:
        raise RuntimeError(f"components still unset after load: {missing}")
    print(f"[load] {wanted} in {time.time() - t0:.1f}s", flush=True)


# Config-only components: no weights, no VRAM, and load_components cannot rebuild
# them once they have been set to None -- there is nothing on disk to reload. Freeing
# them is invisible within a single generation but breaks the next one, which is how
# --repeat first surfaced it: iteration 1 reached the decoder with video_processor=None.
_STICKY_COMPONENTS = {"image_processor", "processor", "video_processor", "tokenizer"}


def free_group(pipe, names: list[str]) -> None:
    dropped = []
    for name in names:
        if name in _STICKY_COMPONENTS:
            continue
        if getattr(pipe, name, None) is not None:
            setattr(pipe, name, None)
            dropped.append(name)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if dropped:
        print(f"[free] {dropped}", flush=True)


def seed_state(pipe, kwargs: dict):
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    state = PipelineState()
    passed = dict(kwargs)
    for param in pipe.blocks.inputs:
        name = param.name
        if name is None:
            continue
        if name in passed:
            state.set(name, passed.pop(name), param.kwargs_type)
        elif name not in state.values:
            state.set(name, param.default, param.kwargs_type)
    if passed:
        raise ValueError(f"inputs not accepted by MiniMaxH3Blocks: {sorted(passed)}")
    return state


def write_mp4(videos, audio, sampling_rate: int, out_path: str, fps: int) -> None:
    """Mux the decoded frames and stereo soundtrack into one H.264/AAC MP4."""
    import imageio
    import numpy as np
    import soundfile as sf

    frames = videos[0] if isinstance(videos, list) else videos
    if torch.is_tensor(frames):
        frames = frames.float().clamp(0, 1).mul(255).byte().cpu().numpy()
        if frames.shape[1] in (1, 3):  # (T, C, H, W) -> (T, H, W, C)
            frames = frames.transpose(0, 2, 3, 1)
    else:
        frames = np.stack([np.asarray(f) for f in frames])

    wav = audio
    if torch.is_tensor(wav):
        wav = wav.float().cpu().numpy()
    wav = np.asarray(wav)
    while wav.ndim > 2:
        wav = wav[0]
    if wav.ndim == 2 and wav.shape[0] in (1, 2):
        wav = wav.T  # (channels, samples) -> (samples, channels)

    with tempfile.TemporaryDirectory() as tmp:
        v_tmp = os.path.join(tmp, "video.mp4")
        a_tmp = os.path.join(tmp, "audio.wav")
        imageio.mimwrite(v_tmp, frames, fps=fps, quality=8, macro_block_size=1)
        sf.write(a_tmp, wav, int(sampling_rate))
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", v_tmp, "-i", a_tmp,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest", out_path,
            ],
            check=True,
        )
    print(f"[out ] {out_path} frames={len(frames)} audio={wav.shape} sr={sampling_rate}", flush=True)


def numbered_path(path: str, idx: int) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}-{idx}{ext}"


def run_batch(pipe, prompt_list: list[dict], args, mon, run_id: str) -> list[str]:
    """Stage-major batch execution for variable prompts.

    Each component group is loaded once, all N prompts are processed through
    that stage, then the group is freed -- so the 62 GiB text encoder and
    62 GiB transformer each make a single trip to GPU instead of N.

    Text embeds stay on their original device (some are GPU, some are CPU) and
    are carried in each PipelineState across stages -- no cache/restore needed.
    """
    N = len(prompt_list)
    use_dm = not args.no_load_opt
    out_dir = os.path.join(args.outputs_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)
    outputs: list[str] = []
    wall = time.time()

    # ---- build per-prompt run_kwargs and seed initial states ----
    base_kwargs = {
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.steps,
    }
    states = []
    for i, p in enumerate(prompt_list):
        rk = dict(base_kwargs)
        rk["prompt"] = p["prompt"]
        seed_i = p.get("seed", args.seed + i if args.seed is not None else i)
        rk["generator"] = torch.Generator(device="cuda").manual_seed(seed_i)
        states.append(seed_state(pipe, rk))

    # ================================================================ stage 1: before_encode
    be_block = pipe.blocks.sub_blocks["before_encode"]
    with mon.phase("load:before_encode"):
        load_group(pipe, STAGE_COMPONENTS["before_encode"], torch.bfloat16, use_device_map=use_dm)
    for i in range(N):
        with mon.phase(f"run:before_encode:{i}"):
            _, states[i] = be_block(pipe, states[i])
    with mon.phase("free:before_encode"):
        free_group(pipe, STAGE_COMPONENTS["before_encode"])

    # ================================================================ stage 2: text_encoder
    te_block = pipe.blocks.sub_blocks["text_encoder"]
    with mon.phase("load:text_encoder"):
        load_group(pipe, STAGE_COMPONENTS["text_encoder"], torch.bfloat16, use_device_map=use_dm)
    report("loaded for text_encoder")

    for i in range(N):
        t0 = time.time()
        with mon.phase(f"run:text_encoder:{i}"), torch.no_grad():
            _, states[i] = te_block(pipe, states[i])
        print(f"[run ] text_encoder:{i} took {time.time() - t0:.1f}s", flush=True)
    report("ran text_encoder")

    with mon.phase("free:text_encoder"):
        free_group(pipe, STAGE_COMPONENTS["text_encoder"])

    # ================================================================ stage 3: vae_encoder
    # Always run even for t2va: the block may set defaults the denoise stage expects.
    ve_block = pipe.blocks.sub_blocks["vae_encoder"]
    with mon.phase("load:vae_encoder"):
        load_group(pipe, STAGE_COMPONENTS["vae_encoder"], torch.bfloat16, use_device_map=use_dm)
    report("loaded for vae_encoder")
    for i in range(N):
        t0 = time.time()
        with mon.phase(f"run:vae_encoder:{i}"), torch.no_grad():
            _, states[i] = ve_block(pipe, states[i])
        print(f"[run ] vae_encoder:{i} took {time.time() - t0:.1f}s", flush=True)
    report("ran vae_encoder")
    with mon.phase("free:vae_encoder"):
        free_group(pipe, STAGE_COMPONENTS["vae_encoder"])

    # ================================================================ stage 4: denoise
    dn_block = pipe.blocks.sub_blocks["denoise"]
    with mon.phase("load:denoise"):
        load_group(pipe, STAGE_COMPONENTS["denoise"], torch.bfloat16, use_device_map=use_dm)
    report("loaded for denoise")

    # LoRA (once)
    if args.lora:
        with mon.phase("load:lora"):
            pipe.load_lora_weights(args.lora)
            pipe.fuse_lora(lora_scale=args.lora_scale)
            pipe.unload_lora_weights()
            report(f"fused LoRA from {args.lora} (scale={args.lora_scale})")

    # compile (once)
    if args.compile and getattr(pipe, "transformer", None) is not None:
        with mon.phase("compile:transformer"):
            pipe.transformer.compile_repeated_blocks(
                mode="max-autotune-no-cudagraphs",
                fullgraph=args.compile_fullgraph,
                dynamic=False,
            )
        print("[compile] regional compilation applied to transformer blocks", flush=True)

    # FBCache (once)
    fbc_applied = False
    if args.cache_dit and getattr(pipe, "transformer", None) is not None:
        from diffusers.hooks._helpers import TransformerBlockRegistry, TransformerBlockMetadata
        from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig
        from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock
        if MiniMaxH3TransformerBlock not in TransformerBlockRegistry._registry:
            TransformerBlockRegistry.register(
                model_class=MiniMaxH3TransformerBlock,
                metadata=TransformerBlockMetadata(
                    return_hidden_states_index=0,
                    return_encoder_hidden_states_index=None,
                ),
            )
        apply_first_block_cache(
            pipe.transformer,
            FirstBlockCacheConfig(threshold=args.cache_dit_threshold),
        )
        from diffusers.hooks import HookRegistry
        HookRegistry.check_if_exists_or_initialize(pipe.transformer)._set_context("fbc_inference")
        print(f"[fbc ] FBCache enabled: threshold={args.cache_dit_threshold}", flush=True)
        fbc_applied = True

    # instrument (once)
    if getattr(pipe, "transformer", None) is not None:
        mon.instrument(pipe.transformer, "transformer")

    for i in range(N):
        # reset FBCache state so the previous video's residuals don't leak
        if fbc_applied and i > 0:
            from diffusers.hooks import HookRegistry
            _fbc_reg = HookRegistry.check_if_exists_or_initialize(pipe.transformer)
            _fbc_reg.reset_stateful_hooks(recurse=True)
            _fbc_reg._set_context("fbc_inference")
            del _fbc_reg

        t0 = time.time()
        with mon.phase(f"run:denoise:{i}"), torch.no_grad():
            _, states[i] = dn_block(pipe, states[i])
        print(f"[run ] denoise:{i} took {time.time() - t0:.1f}s", flush=True)
        report(f"ran denoise:{i}")

        # offload heavy latents to CPU to free HBM for the next video's denoise
        for key in ("latents", "audio_latents"):
            val = states[i].get(key)
            if val is not None and torch.is_tensor(val):
                states[i].set(key, val.cpu())
        torch.cuda.empty_cache()

    # FBCache cleanup (remove hooks before freeing transformer)
    if fbc_applied and getattr(pipe, "transformer", None) is not None:
        from diffusers.hooks import HookRegistry
        from diffusers.hooks.first_block_cache import _FBC_LEADER_BLOCK_HOOK, _FBC_BLOCK_HOOK
        _fbc_cleanup_reg = HookRegistry.check_if_exists_or_initialize(pipe.transformer)
        _fbc_cleanup_reg.remove_hook(_FBC_LEADER_BLOCK_HOOK, recurse=True)
        _fbc_cleanup_reg.remove_hook(_FBC_BLOCK_HOOK, recurse=True)
        del _fbc_cleanup_reg

    with mon.phase("free:denoise"):
        free_group(pipe, STAGE_COMPONENTS["denoise"])

    # ================================================================ stage 5: decode
    dc_block = pipe.blocks.sub_blocks["decode"]
    with mon.phase("load:decode"):
        load_group(pipe, STAGE_COMPONENTS["decode"], torch.bfloat16, use_device_map=use_dm)
    report("loaded for decode")

    for i in range(N):
        # bring latents back to GPU
        for key in ("latents", "audio_latents"):
            val = states[i].get(key)
            if val is not None and torch.is_tensor(val):
                states[i].set(key, val.cuda())

        t0 = time.time()
        with mon.phase(f"run:decode:{i}"), torch.no_grad():
            _, states[i] = dc_block(pipe, states[i])
        print(f"[run ] decode:{i} took {time.time() - t0:.1f}s", flush=True)
        report(f"ran decode:{i}")

        # write mp4
        out_path = os.path.join(out_dir, f"video-{i}.mp4")
        with mon.phase(f"encode_mp4:{i}"):
            write_mp4(
                states[i].get("videos"),
                states[i].get("audio"),
                states[i].get("sampling_rate"),
                out_path,
                fps=pipe.fps,
            )
        outputs.append(os.path.abspath(out_path))

        # release decoded tensors to free memory for the next video
        states[i] = None
        torch.cuda.empty_cache()

    with mon.phase("free:decode"):
        free_group(pipe, STAGE_COMPONENTS["decode"])

    print(f"[done] batch of {N} videos, wall time {time.time() - wall:.1f}s", flush=True)
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--image", default=None, help="first keyframe (switches to fl2va)")
    ap.add_argument("--last-image", default=None, help="last keyframe")
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--width", type=int, default=1344)
    # H3 targets 5-15s at 24 fps and its video VAE only encodes 17*n + 5 frames,
    # so the usable range is 124..345.
    ap.add_argument("--num-frames", type=int, default=124)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="output video; defaults to <outputs-dir>/<run-id>/video.mp4")
    ap.add_argument("--outputs-dir", default="/mnt/workspace/outputs")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--lora", default=None,
                    help="path or HF repo for LoRA weights (e.g. larryvrh/MiniMax-H3-Turbo-Lora)")
    ap.add_argument("--lora-scale", type=float, default=1.0,
                    help="LoRA adapter weight scale")
    ap.add_argument("--run-id", default=None,
                    help="name for this run's telemetry dir; defaults to a timestamp")
    ap.add_argument("--runs-dir", default="/mnt/workspace/runs")
    ap.add_argument("--monitor-hz", type=float, default=2.0)
    ap.add_argument("--tag", default="baseline",
                    help="what is being tested, e.g. 'tf32' or 'compile' -- one lever per run")
    ap.add_argument("--no-load-opt", action="store_true",
                    help="disable device_map=cuda direct-to-GPU loading (for A/B baseline)")
    ap.add_argument("--parallel-load", action="store_true",
                    help="set HF_ENABLE_PARALLEL_LOADING=1 (8-worker threadpool shard loading)")
    ap.add_argument("--profile", action="store_true",
                    help="capture a kernel trace of the denoise stage; use with --steps 2, "
                         "a 50-step trace is tens of GB and no more informative")
    ap.add_argument("--cache-dit", action="store_true",
                    help="enable FBCache (First Block Cache) residual caching in denoise")
    ap.add_argument("--cache-dit-threshold", type=float, default=0.15,
                    help="FBCache residual L1 threshold (higher = more skips, lower quality)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="generate N videos in one process. Iteration 0 pays cold disk reads; "
                         "later ones show what a warm page cache actually buys, which is the "
                         "only honest way to compare against a resident server")
    ap.add_argument("--reuse-text-embeds", action="store_true",
                    help="encode the prompt once and replay it on later --repeat "
                         "iterations, so the 62 GiB text_encoder is loaded once per "
                         "process instead of once per video")
    ap.add_argument("--compile", action="store_true",
                    help="apply torch.compile regional compilation to transformer "
                         "blocks before denoising. First forward triggers compilation "
                         "(2-5 min); use --repeat 2 to isolate warmup cost")
    ap.add_argument("--compile-fullgraph", action="store_true",
                    help="use fullgraph=True for torch.compile (stricter, may fail "
                         "if any graph break exists, but enables more aggressive optimization)")
    ap.add_argument("--prompt-file", default=None,
                    help="JSON file with a list of prompt objects. Each object must have 'prompt' (str). "
                         "Optional keys: 'seed' (int). "
                         "When set, --prompt/--image/--last-image are ignored. "
                         "Execution uses stage-major ordering: each component loads once for all prompts.")
    ap.add_argument("--batch-limit", type=int, default=None,
                    help="use only the first N entries from --prompt-file; useful for N-scaling benchmarks")
    ap.add_argument("--drop-page-cache", action="store_true",
                    help="evict the model files from the page cache before starting, so load:* "
                         "phases measure a reproducible cold read instead of leftover cache state")
    ap.add_argument("--tf32", action="store_true",
                    help="enable TF32 for float32 matmuls (torch.backends.cuda.matmul.allow_tf32=True). "
                         "A100 TF32 tensor cores provide 312 TFLOPS vs FP32 19.5 TFLOPS. "
                         "Only affects fp32 operations; bf16 matmuls use bf16 tensor cores regardless")
    args = ap.parse_args()

    # ---- prompt-file vs single-prompt validation ----
    prompt_list: list[dict] | None = None
    if args.prompt_file:
        if args.prompt:
            print("[warn] --prompt ignored when --prompt-file is set", flush=True)
        if args.image or args.last_image:
            raise SystemExit("--image/--last-image are not supported with --prompt-file (t2va only)")
        if args.repeat != 1:
            raise SystemExit("--repeat and --prompt-file are mutually exclusive")
        if args.reuse_text_embeds:
            raise SystemExit("--reuse-text-embeds and --prompt-file are mutually exclusive "
                             "(batch mode already encodes once)")
        import json as _json
        with open(args.prompt_file) as f:
            prompt_list = _json.load(f)
        assert isinstance(prompt_list, list) and len(prompt_list) > 0, \
            "prompt-file must be a non-empty JSON list"
        if args.batch_limit is not None:
            if args.batch_limit < 1:
                raise SystemExit("--batch-limit must be >= 1")
            if args.batch_limit > len(prompt_list):
                raise SystemExit(
                    f"--batch-limit {args.batch_limit} exceeds prompt-file size {len(prompt_list)}"
                )
            prompt_list = prompt_list[:args.batch_limit]
        for i, p in enumerate(prompt_list):
            assert "prompt" in p, f"prompt_list[{i}] missing 'prompt' key"
        print(f"[batch] loaded {len(prompt_list)} prompts from {args.prompt_file}", flush=True)
    elif not args.prompt:
        raise SystemExit("either --prompt or --prompt-file is required")

    if args.parallel_load:
        os.environ["HF_ENABLE_PARALLEL_LOADING"] = "1"

    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")

    if args.reuse_text_embeds and args.repeat == 1:
        print("[warn] --reuse-text-embeds has nothing to replay at --repeat 1", flush=True)

    if args.compile and args.cache_dit:
        print("[warn] --compile and --cache-dit are not validated together; "
              "FBCache hooks may cause graph breaks. Use separately for now.", flush=True)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; check the torch build against the driver")

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        print(f"[tf32] enabled: matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}, "
              f"cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}, "
              f"precision={torch.get_float32_matmul_precision()}", flush=True)

    run_id = args.run_id or (
        method_name(vars(args), os.environ.get("DIFFUSERS_ATTN_BACKEND"),
                    len(prompt_list) if prompt_list is not None else None)
        + time.strftime("-%Y%m%d-%H%M%S")
    )
    run_dir = os.path.join(args.runs_dir, run_id)
    # runs/<id> and outputs/<id> share one name, <method>-<date>-<time>, so the
    # directory listing alone says which levers each run pulled (see run_naming.py).
    if args.out is None:
        args.out = os.path.join(args.outputs_dir, run_id, "video.mp4")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.cuda.reset_peak_memory_stats()

    # The invocation is the experiment. run.json carries the parsed args, but only
    # raw argv shows how the run was really launched, and it is written up front so
    # a crashed run still says what it was trying to do. The first 768p video was
    # lost this way: its prompt was never recorded anywhere.
    with open(os.path.join(run_dir, "cmd.txt"), "w") as fh:
        fh.write(" ".join(shlex.quote(a) for a in sys.argv) + "\n\n")
        fh.write(json.dumps(vars(args), indent=2, ensure_ascii=False, default=str) + "\n")

    page_cache = {"dropped": False, "cached_at_start_gib": round(page_cache_gib(), 2)}
    if args.drop_page_cache:
        page_cache = {"dropped": True} | drop_model_page_cache(args.model_dir)

    mon = RunMonitor(run_dir, run_id, hz=args.monitor_hz).start()
    print(f"[mon ] telemetry -> {run_dir}", flush=True)

    from diffusers import ModularPipeline

    with mon.phase("init"):
        print(f"[init] loading pipeline config from {args.model_dir}", flush=True)
        pipe = ModularPipeline.from_pretrained(args.model_dir)
        repoint_to_local(pipe, args.model_dir)
    report("after config load")

    # ---- batch mode: stage-major ordering ----
    if prompt_list is not None:
        outputs = run_batch(pipe, prompt_list, args, mon, run_id)
        summary = mon.stop(extra={
            "tag": args.tag,
            "args": vars(args) | {"generator": None},
            "output": outputs[0] if outputs else None,
            "outputs": outputs,
            "fps": getattr(pipe, "fps", None),
            "batch_size": len(prompt_list),
            "page_cache": page_cache,
        })
        print("[mon ] " + json.dumps(summary["phase_seconds"], ensure_ascii=False), flush=True)
        print(f"[mon ] step median {summary['step_ms_median']} ms over "
              f"{summary['steps_counted']} transformer calls", flush=True)
        print(f"[mon ] peaks: torch_alloc={summary['peak_torch_alloc_gib']} GiB  "
              f"nvml={summary['peak_nvml_used_gib']} GiB  host={summary['peak_host_used_gib']} GiB",
              flush=True)
        return

    # ---- single-prompt mode (original code path) ----
    run_kwargs = {
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.steps,
        "generator": torch.Generator(device="cuda").manual_seed(args.seed),
    }
    if args.image or args.last_image:
        from diffusers.utils import load_image

        if args.image:
            run_kwargs["image"] = load_image(args.image)
        if args.last_image:
            run_kwargs["last_image"] = load_image(args.last_image)

    outputs = []
    # A few MB of prompt_embeds produced by a 62 GiB encoder. Keeping it drops the
    # per-iteration working set from ~135 GiB to ~72 GiB, and 72 GiB is the first size
    # that fits the ~112 GiB of page cache on this box -- so the win is not only the
    # stage that stops running, it is the transformer reads that stop being evicted.
    text_embeds_cache: dict | None = None
    for it in range(args.repeat):
        # Phase names carry the iteration index so the monitor keeps them apart.
        # That separation is the measurement: load:denoise#0 is a cold disk read,
        # load:denoise#1 is whatever the page cache still held onto.
        sfx = f"#{it}" if args.repeat > 1 else ""
        # Re-seeded per iteration, so every iteration should produce a
        # bit-identical video -- a free determinism check across the repeat.
        run_kwargs["generator"] = torch.Generator(device="cuda").manual_seed(args.seed)
        state = seed_state(pipe, run_kwargs)
        out_path = numbered_path(args.out, it) if args.repeat > 1 else args.out
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        wall = time.time()
        for stage, comps in STAGE_COMPONENTS.items():
            block = pipe.blocks.sub_blocks[stage]
            if stage == "text_encoder" and text_embeds_cache is not None:
                # kwargs_type is read back off the block's own declaration rather than
                # hardcoded: denoise looks prompt_embeds up by its kwargs_type
                # ('denoiser_input_fields'), not by name, so setting the value without
                # it would leave the denoiser with no text conditioning at all.
                with mon.phase(f"reuse:text_encoder{sfx}"):
                    for param in block.intermediate_outputs:
                        state.set(param.name, text_embeds_cache[param.name], param.kwargs_type)
                print(f"[reuse] text_encoder skipped, replayed {sorted(text_embeds_cache)}",
                      flush=True)
                continue
            # load / compute / free are timed separately: weight loading is I/O plus
            # host-to-device copy, compute is GPU-bound. They respond to completely
            # different optimisations, so a merged number would hide which one moved.
            with mon.phase(f"load:{stage}{sfx}"):
                load_group(pipe, comps, torch.bfloat16, use_device_map=not args.no_load_opt)
            report(f"loaded for {stage}")
            if stage == "denoise" and args.lora:
                with mon.phase(f"load:lora{sfx}"):
                    pipe.load_lora_weights(args.lora)
                    pipe.fuse_lora(lora_scale=args.lora_scale)
                    pipe.unload_lora_weights()
                    report(f"fused LoRA from {args.lora} (scale={args.lora_scale})")
            if stage == "denoise" and getattr(pipe, "transformer", None) is not None:
                if args.compile:
                    # Regional compilation: compiles only the repeated blocks (50 MiniMaxH3TransformerBlock
                    # + 2 MiniMaxH3TokenRefinerBlock). These share a single compiled graph, so compilation
                    # cost is paid once, not 52x.  We avoid full-model compile because the top-level forward
                    # contains index_select/index_copy that complicate tracing.
                    #
                    # mode="max-autotune-no-cudagraphs": enable Triton autotuning without CUDA Graphs
                    # (graph capture would need extra headroom on an already tight 80 GiB card).
                    # dynamic=False: sequence length is fixed within one generation (768p/124 frames).
                    with mon.phase(f"compile:transformer{sfx}"):
                        pipe.transformer.compile_repeated_blocks(
                            mode="max-autotune-no-cudagraphs",
                            fullgraph=args.compile_fullgraph,
                            dynamic=False,
                        )
                    print(f"[compile] regional compilation applied to transformer blocks", flush=True)
                if args.cache_dit:
                    from diffusers.hooks._helpers import TransformerBlockRegistry, TransformerBlockMetadata
                    from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig
                    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock
                    # Register MiniMaxH3TransformerBlock (guard against double registration)
                    if MiniMaxH3TransformerBlock not in TransformerBlockRegistry._registry:
                        TransformerBlockRegistry.register(
                            model_class=MiniMaxH3TransformerBlock,
                            metadata=TransformerBlockMetadata(
                                return_hidden_states_index=0,
                                return_encoder_hidden_states_index=None,
                            ),
                        )
                    apply_first_block_cache(
                        pipe.transformer,
                        FirstBlockCacheConfig(threshold=args.cache_dit_threshold),
                    )
                    # FBCache hooks use StateManager which needs a context to be set.
                    # The denoise loop doesn't call cache_context(), so set it manually.
                    from diffusers.hooks import HookRegistry
                    HookRegistry.check_if_exists_or_initialize(pipe.transformer)._set_context("fbc_inference")
                    print(f"[fbc ] FBCache enabled: threshold={args.cache_dit_threshold}", flush=True)
                mon.instrument(pipe.transformer, "transformer")
            t0 = time.time()
            # A trace tells you which kernels the time is in; the monitor only tells you
            # which phase it is in. Scoped to denoise because that is the only stage worth
            # tracing, and bounded by --steps because trace size grows with step count.
            prof_ctx = contextlib.nullcontext()
            if args.profile and stage == "denoise":
                prof_ctx = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_stack=False,  # stacks balloon the trace and Inductor frames are unreadable
                )
            with mon.phase(f"run:{stage}{sfx}"), prof_ctx as prof, torch.no_grad():
                _, state = block(pipe, state)
            if prof is not None:
                trace = os.path.join(run_dir, "denoise_trace.json.gz")
                prof.export_chrome_trace(trace)
                table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30)
                with open(os.path.join(run_dir, "denoise_kernels.txt"), "w") as fh:
                    fh.write(table + "\n")
                print(f"[prof] {trace}\n{table}", flush=True)
            print(f"[run ] {stage} took {time.time() - t0:.1f}s", flush=True)
            report(f"ran {stage}")
            if stage == "text_encoder" and args.reuse_text_embeds and text_embeds_cache is None:
                # Cloned: these tensors outlive the state that produced them, and an
                # in-place write downstream would silently poison every later iteration.
                text_embeds_cache = {}
                for param in block.intermediate_outputs:
                    value = state.get(param.name)
                    text_embeds_cache[param.name] = (
                        value.clone() if torch.is_tensor(value) else value
                    )
                cached_mib = sum(
                    v.numel() * v.element_size()
                    for v in text_embeds_cache.values() if torch.is_tensor(v)
                ) / 2**20
                print(f"[reuse] cached {sorted(text_embeds_cache)} ({cached_mib:.1f} MiB); "
                      f"text_encoder will not be loaded again", flush=True)
            # Free everything, including the VAEs decode needs again. Holding them through
            # denoise costs 10.3 GiB and left only ~7 GiB for activations; reloading them
            # for decode takes ~21s and is the better trade on a single 80 GiB card.
            with mon.phase(f"free:{stage}{sfx}"):
                if stage == "denoise" and args.cache_dit and getattr(pipe, "transformer", None) is not None:
                    # Remove FBCache hooks from all blocks before freeing the transformer.
                    # Without this, the hook chain creates circular references that keep
                    # the 62 GiB transformer alive through the decode stage → OOM.
                    from diffusers.hooks import HookRegistry
                    from diffusers.hooks.first_block_cache import _FBC_LEADER_BLOCK_HOOK, _FBC_BLOCK_HOOK
                    _fbc_cleanup_reg = HookRegistry.check_if_exists_or_initialize(pipe.transformer)
                    _fbc_cleanup_reg.remove_hook(_FBC_LEADER_BLOCK_HOOK, recurse=True)
                    _fbc_cleanup_reg.remove_hook(_FBC_BLOCK_HOOK, recurse=True)
                    del _fbc_cleanup_reg
                free_group(pipe, comps)

        print(f"[done] iteration {it} wall time {time.time() - wall:.1f}s", flush=True)

        with mon.phase(f"encode_mp4{sfx}"):
            write_mp4(
                state.get("videos"),
                state.get("audio"),
                state.get("sampling_rate"),
                out_path,
                fps=pipe.fps,
            )
        report(f"final it={it}")
        outputs.append(os.path.abspath(out_path))

    summary = mon.stop(extra={
        "tag": args.tag,
        "args": vars(args) | {"generator": None},
        "output": outputs[0],
        "outputs": outputs,
        "fps": getattr(pipe, "fps", None),
        "page_cache": page_cache,
    })
    print("[mon ] " + json.dumps(summary["phase_seconds"], ensure_ascii=False), flush=True)
    print(f"[mon ] step median {summary['step_ms_median']} ms over "
          f"{summary['steps_counted']} transformer calls", flush=True)
    print(f"[mon ] peaks: torch_alloc={summary['peak_torch_alloc_gib']} GiB  "
          f"nvml={summary['peak_nvml_used_gib']} GiB  host={summary['peak_host_used_gib']} GiB",
          flush=True)


if __name__ == "__main__":
    main()
