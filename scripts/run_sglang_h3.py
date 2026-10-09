#!/usr/bin/env python3
"""Run a reproducible single-A100 MiniMax-H3 experiment with SGLang.

The sampling fixture matches bench_768p.sh: 1344x768, 124 aligned frames,
24 fps, 50 sigma grid points (49 DiT evaluations), prompt and seed 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h3_monitor import RunMonitor  # noqa: E402

WORKSPACE = Path(__file__).resolve().parents[1]
SGLANG_RUNTIME = WORKSPACE / ".sglang-runtime"
SGLANG_SOURCE = WORKSPACE / ".sglang-probe"
DEFAULT_MODEL = WORKSPACE / "MiniMax-H3-SGLang"
DEFAULT_PROMPT = (
    "A cinematic shot of a red fox trotting through a snowy pine forest at dawn, "
    "soft volumetric light, shallow depth of field"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--tag")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--runs-dir", default=str(WORKSPACE / "runs"))
    parser.add_argument("--outputs-dir", default=str(WORKSPACE / "outputs"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--duration-seconds", type=float, default=5.0)
    parser.add_argument("--drop-page-cache", action="store_true")
    parser.add_argument("--cache-dit", action="store_true")
    parser.add_argument("--cache-dit-threshold", type=float, default=0.24)
    parser.add_argument(
        "--attention-backend",
        choices=("fa", "torch_sdpa", "sage_attn", "sage_attn_3"),
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--dit-resident-layers", type=float, default=0.0)
    parser.add_argument("--dit-prefetch-size", type=float, default=0.0)
    parser.add_argument("--monitor-hz", type=float, default=2.0)
    return parser.parse_args()


def method_name(args: argparse.Namespace) -> str:
    parts = ["sglang"]
    if args.attention_backend:
        parts.append(args.attention_backend.replace("_attn", ""))
    if args.cache_dit:
        parts.append(f"cachedit{round(args.cache_dit_threshold * 100):03d}")
    if args.compile:
        parts.append("compile")
    if args.dit_resident_layers:
        value = args.dit_resident_layers
        label = str(int(value)) if value >= 1 else str(value).replace(".", "p")
        parts.append(f"resident{label}")
    if args.dit_prefetch_size:
        value = args.dit_prefetch_size
        label = str(int(value)) if value >= 1 else str(value).replace(".", "p")
        parts.append(f"prefetch{label}")
    if args.steps != 50:
        parts.append(f"steps{args.steps}")
    if len(parts) == 1:
        parts.append("raw")
    return "_".join(parts)


def page_cache_gib() -> float:
    values: dict[str, int] = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            key, _, value = line.partition(":")
            values[key] = int(value.split()[0])
    return (values.get("Cached", 0) + values.get("Buffers", 0)) / 1024 / 1024


def drop_model_page_cache(model_path: str) -> dict:
    before = page_cache_gib()
    started = time.time()
    files = 0
    seen: set[tuple[int, int]] = set()
    for root, _, names in os.walk(model_path):
        for name in names:
            path = os.path.join(root, name)
            try:
                stat = os.stat(path)
                identity = (stat.st_dev, stat.st_ino)
                if identity in seen:
                    continue
                seen.add(identity)
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
                files += 1
            except OSError as exc:
                print(f"[cache] skip {path}: {exc}", flush=True)
    after = page_cache_gib()
    result = {
        "dropped": True,
        "files": files,
        "cached_before_gib": round(before, 2),
        "cached_after_gib": round(after, 2),
        "seconds": round(time.time() - started, 2),
    }
    print(
        f"[cache] evicted {files} files: {before:.1f} -> {after:.1f} GiB "
        f"in {result['seconds']:.1f}s",
        flush=True,
    )
    return result


def update_json(path: Path, updates: dict) -> None:
    with path.open() as fh:
        data = json.load(fh)
    data.update(updates)
    with path.open("w") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")


def sglang_config(args: argparse.Namespace, output_file: Path) -> dict:
    config = {
        "model_path": args.model_path,
        "model_variant": "fl2va",
        "pipeline_class_name": "MiniMaxH3Pipeline",
        "backend": "sglang",
        "performance_mode": "memory",
        "layerwise_offload_components": ["dit", "text_encoder", "vae"],
        "layerwise_resident_layers": {"video_vae": 36},
        "dit_layerwise_resident_layers": args.dit_resident_layers,
        "dit_offload_prefetch_size": args.dit_prefetch_size,
        "enable_torch_compile": args.compile,
        "component_paths": {
            "transformer": str(WORKSPACE / "MiniMax-H3" / "transformer")
        },
        "prompt": args.prompt,
        "task": "t2va",
        "conditions": [],
        "target": {
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": args.duration_seconds,
        },
        "num_inference_steps": args.steps,
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
        "seed": args.seed,
        "save_output": True,
        "output_path": str(output_file.parent),
        "output_file_name": output_file.name,
    }
    if args.cache_dit:
        config["enable_cache_dit"] = True
        config["cache_dit_params"] = {
            "residual_diff_threshold": args.cache_dit_threshold
        }
    else:
        config["enable_cache_dit"] = False
    if args.attention_backend:
        config["attention_backend_override"] = args.attention_backend
    return config


def perf_summary(perf: dict, process_seconds: float) -> dict:
    stage_ms = {
        row["name"]: float(row["duration_ms"])
        for row in perf.get("steps", [])
    }
    request_seconds = float(perf.get("total_duration_ms", 0.0)) / 1000.0
    denoise_ms = [float(row["duration_ms"]) for row in perf.get("denoise_steps_ms", [])]
    peak_mb = max(
        (
            float(snapshot.get("peak_reserved_mb", 0.0))
            for snapshot in perf.get("memory_checkpoints", {}).values()
        ),
        default=0.0,
    )
    phase_seconds = {
        "load:pipeline": round(max(0.0, process_seconds - request_seconds), 3),
        "run:text_encoder": round(stage_ms.get("MiniMaxH3TextEncodingStage", 0.0) / 1000, 3),
        "run:vae_encoder": round(
            (
                stage_ms.get("MiniMaxH3VisualEncodingStage", 0.0)
                + stage_ms.get("MiniMaxH3AudioEncodingStage", 0.0)
            )
            / 1000,
            3,
        ),
        "run:prepare": round(
            (
                stage_ms.get("MiniMaxH3LatentPreparationStage", 0.0)
                + stage_ms.get("MiniMaxH3TimestepPreparationStage", 0.0)
            )
            / 1000,
            3,
        ),
        "run:denoise": round(stage_ms.get("MiniMaxH3DenoisingStage", 0.0) / 1000, 3),
        "run:decode": round(stage_ms.get("MiniMaxH3DecodingStage", 0.0) / 1000, 3),
    }
    return {
        "phase_seconds": phase_seconds,
        "load_seconds_total": phase_seconds["load:pipeline"],
        "run_seconds_total": round(request_seconds, 3),
        "step_ms_median": round(statistics.median(denoise_ms), 1) if denoise_ms else None,
        "step_ms_mean": round(statistics.fmean(denoise_ms), 1) if denoise_ms else None,
        "steps_counted": len(denoise_ms),
        "peak_torch_alloc_gib": round(peak_mb / 1024, 2),
        "sglang_request_seconds": round(request_seconds, 3),
        "sglang_perf": perf,
    }


def main() -> int:
    args = parse_args()
    method = args.tag or method_name(args)
    run_id = args.run_id or f"{method}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(args.runs_dir) / run_id
    output_dir = Path(args.outputs_dir) / run_id
    output_file = output_dir / "video.mp4"
    perf_file = run_dir / "perf.json"
    config_file = run_dir / "config.json"
    run_dir.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir(parents=True, exist_ok=False)

    config = sglang_config(args, output_file)
    with config_file.open("w") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    with (run_dir / "cmd.txt").open("w") as fh:
        fh.write(" ".join(shlex.quote(value) for value in sys.argv) + "\n\n")
        fh.write(json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n")

    page_cache = (
        drop_model_page_cache(args.model_path)
        if args.drop_page_cache
        else {"dropped": False, "cached_at_start_gib": round(page_cache_gib(), 2)}
    )
    monitor = RunMonitor(str(run_dir), run_id, hz=args.monitor_hz).start()
    update_json(
        run_dir / "run.json",
        {
            "backend": "sglang",
            "args": vars(args),
            "sampling": config,
            "page_cache": page_cache,
            "output": str(output_file),
            "outputs": [str(output_file)],
            "batch_size": 1,
        },
    )

    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = os.pathsep.join(
        [str(SGLANG_RUNTIME), str(SGLANG_SOURCE), child_env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    child_env["HF_HUB_OFFLINE"] = "1"
    child_env["TRANSFORMERS_OFFLINE"] = "1"
    child_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    command = [
        sys.executable,
        "-m",
        "sglang.multimodal_gen.runtime.entrypoints.cli.main",
        "generate",
        "--config",
        str(config_file),
        "--perf-dump-path",
        str(perf_file),
        "--output-file-path",
        str(output_file),
    ]
    print(f"[run ] {run_id}", flush=True)
    print("[cmd ] " + " ".join(shlex.quote(value) for value in command), flush=True)
    started = time.time()
    returncode = 1
    try:
        with monitor.phase("sglang:subprocess"):
            process = subprocess.Popen(
                command,
                cwd=WORKSPACE,
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
            returncode = process.wait()
    except BaseException as exc:
        print(f"[fail] {type(exc).__name__}: {exc}", flush=True)
        summary = monitor.stop(
            extra={
                "status": "failed",
                "backend": "sglang",
                "args": vars(args),
                "sampling": config,
                "page_cache": page_cache,
                "output": str(output_file),
                "outputs": [str(output_file)] if output_file.exists() else [],
                "returncode": returncode,
            }
        )
        raise

    process_seconds = time.time() - started
    extra = {
        "status": "success" if returncode == 0 and output_file.exists() else "failed",
        "backend": "sglang",
        "args": vars(args),
        "sampling": config,
        "page_cache": page_cache,
        "output": str(output_file),
        "outputs": [str(output_file)] if output_file.exists() else [],
        "batch_size": 1,
        "returncode": returncode,
        "process_seconds": round(process_seconds, 3),
    }
    if perf_file.exists():
        with perf_file.open() as fh:
            extra.update(perf_summary(json.load(fh), process_seconds))
    summary = monitor.stop(extra=extra)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
