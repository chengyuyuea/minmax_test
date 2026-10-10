#!/usr/bin/env python3
"""Report MiniMax-H3 performance runs and reproducible media quality metrics.

Performance rows expose measured wall/load/compute values. Only `cold` rows have
comparable measured load; use stage_table.py for the report's unified L_cold
comparison. Quality mode compares encoded MP4 outputs from identical prompt/seed
runs and reports video PSNR/SSIM plus decoded-audio waveform SNR.

Usage:
  scripts/h3_report.py
  scripts/h3_report.py --tag fbcache --sort pervideo
  scripts/h3_report.py --quality-reference diffusers_raw-20260929-103055 \
      --quality-target diffusers_sage-20260928-165335 --quality-json logs/quality.json
"""
import argparse
from array import array
import glob
import json
import math
import os
import re
import subprocess
import sys


def load_runs(runs_dir):
    runs = []
    for rj in sorted(glob.glob(os.path.join(runs_dir, "*", "run.json"))):
        try:
            with open(rj) as fh:
                doc = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if doc.get("status") == "running" or doc.get("wall_seconds") is None:
            continue  # crashed / in-flight run, no usable summary
        runs.append(doc)
    return runs


def n_videos(doc):
    outs = doc.get("outputs")
    if isinstance(outs, list) and outs:
        return len(outs)
    if doc.get("batch_size"):
        return int(doc["batch_size"])
    return int((doc.get("args") or {}).get("repeat", 1) or 1)


def levers(doc):
    a = doc.get("args") or {}
    flags = []
    if (doc.get("env") or {}).get("attn_backend"):
        flags.append(str(doc["env"]["attn_backend"]))
    if a.get("cache_dit"):
        flags.append(f"fbc@{a.get('cache_dit_threshold', '?')}")
    if a.get("reuse_text_embeds"):
        flags.append("reuse")
    if a.get("compile"):
        flags.append("compile")
    if a.get("lora"):
        flags.append("lora")
    if a.get("tf32"):
        flags.append("tf32")
    if (doc.get("page_cache") or {}).get("dropped"):
        flags.append("cold")  # load:* measured from an evicted page cache
    return ",".join(flags) or "-"


def phase_totals(doc):
    """Separate I/O-sensitive loading from per-video execution."""
    phases = doc.get("phase_seconds") or {}
    load = sum(float(v) for k, v in phases.items() if k.startswith("load:"))
    compute = sum(
        float(v)
        for k, v in phases.items()
        if k.startswith("run:") or k.startswith("encode_mp4")
    )
    load_read = sum(
        float(stats.get("read_gib", 0.0))
        for name, stats in (doc.get("phase_io") or {}).items()
        if name.startswith("load:")
    )
    return load, compute, load_read


def _run_output(runs_dir, run_id):
    path = os.path.join(runs_dir, run_id, "run.json")
    with open(path) as fh:
        doc = json.load(fh)
    outputs = doc.get("outputs") or [doc.get("output")]
    if not outputs or not outputs[0]:
        raise ValueError(f"run {run_id} has no output")
    output = outputs[0]
    if not os.path.isfile(output):
        raise FileNotFoundError(output)
    return output


def _video_quality(reference, target):
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostats", "-i", target, "-i", reference,
            "-lavfi", "[0:v][1:v]psnr;[0:v][1:v]ssim", "-f", "null", "-",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip())
    psnr = re.search(r"PSNR .* average:([0-9.+-]+|inf)", proc.stderr)
    ssim = re.search(r"SSIM .* All:([0-9.+-]+)", proc.stderr)
    if not psnr or not ssim:
        raise RuntimeError("ffmpeg did not report PSNR/SSIM")
    return float(psnr.group(1)), float(ssim.group(1))


def _decode_audio(path):
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0", "-vn",
            "-ac", "2", "-ar", "32000", "-f", "f32le", "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode(errors="replace").strip())
    samples = array("f")
    samples.frombytes(proc.stdout)
    return samples


def _audio_snr(reference, target):
    ref = _decode_audio(reference)
    test = _decode_audio(target)
    if len(ref) != len(test):
        raise ValueError(f"decoded audio length differs: {len(ref)} != {len(test)}")
    signal = math.fsum(sample * sample for sample in ref)
    noise = math.fsum((sample - other) ** 2 for sample, other in zip(ref, test))
    if noise == 0:
        return math.inf, len(ref)
    if signal == 0:
        return -math.inf, len(ref)
    return 10.0 * math.log10(signal / noise), len(ref)


def _quality_report(args):
    reference = _run_output(args.runs_dir, args.quality_reference)
    rows = []
    failed = False
    gate_enabled = any(
        value is not None
        for value in (args.min_psnr, args.min_ssim, args.min_audio_snr)
    )
    for run_id in args.quality_target:
        target = _run_output(args.runs_dir, run_id)
        psnr, ssim = _video_quality(reference, target)
        audio_snr, samples = _audio_snr(reference, target)
        passed = None if not gate_enabled else (
            (args.min_psnr is None or psnr >= args.min_psnr)
            and (args.min_ssim is None or ssim >= args.min_ssim)
            and (args.min_audio_snr is None or audio_snr >= args.min_audio_snr)
        )
        failed |= passed is False
        rows.append({
            "run_id": run_id,
            "reference_run_id": args.quality_reference,
            "video_psnr_db": round(psnr, 6),
            "video_ssim": round(ssim, 6),
            "audio_snr_db": round(audio_snr, 6),
            "audio_samples": samples,
            "passed": passed,
        })

    print(f"{'run_id':<46} {'PSNR(dB)':>10} {'SSIM':>10} {'audio SNR(dB)':>14} {'gate':>8}")
    print("-" * 94)
    for row in rows:
        gate = "MEASURED" if row["passed"] is None else ("PASS" if row["passed"] else "FAIL")
        print(
            f"{row['run_id']:<46} {row['video_psnr_db']:>10.3f} "
            f"{row['video_ssim']:>10.6f} {row['audio_snr_db']:>14.3f} {gate:>8}"
        )
    if args.quality_json:
        payload = {
            "reference_run_id": args.quality_reference,
            "comparison_basis": "encoded MP4; decoded video and 32 kHz stereo audio",
            "thresholds": {
                "min_psnr_db": args.min_psnr,
                "min_ssim": args.min_ssim,
                "min_audio_snr_db": args.min_audio_snr,
            },
            "results": rows,
        }
        with open(args.quality_json, "w") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
    return 2 if failed else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="/mnt/workspace/runs")
    ap.add_argument("--baseline", type=float, default=1271.1,
                    help="per-video reference seconds for delta (default: unified raw T_cold)")
    ap.add_argument("--tag", default=None, help="substring filter on tag")
    ap.add_argument("--sort", choices=["run_id", "wall", "pervideo"], default="run_id")
    ap.add_argument("--quality-reference", metavar="RUN_ID")
    ap.add_argument("--quality-target", metavar="RUN_ID", action="append", default=[])
    ap.add_argument("--quality-json", metavar="PATH")
    ap.add_argument("--min-psnr", type=float)
    ap.add_argument("--min-ssim", type=float)
    ap.add_argument("--min-audio-snr", type=float)
    args = ap.parse_args()

    if args.quality_reference:
        if not args.quality_target:
            ap.error("--quality-reference requires at least one --quality-target")
        return _quality_report(args)
    if args.quality_target or args.quality_json:
        ap.error("quality options require --quality-reference")

    runs = load_runs(args.runs_dir)
    if args.tag:
        runs = [d for d in runs if args.tag in str(d.get("tag", ""))]
    if not runs:
        print(f"no completed runs under {args.runs_dir}", file=sys.stderr)
        return 1

    rows = []
    for d in runs:
        a = d.get("args") or {}
        nv = n_videos(d)
        wall = float(d["wall_seconds"])
        per = wall / nv if nv else wall
        load, compute, load_read = phase_totals(d)
        rows.append({
            "run_id": d.get("run_id", "?"),
            "tag": str(d.get("tag", "-"))[:18],
            "steps": a.get("steps", "?"),
            "nv": nv,
            "wall": wall,
            "per": per,
            "load": load,
            "compute": compute,
            "compute_per": compute / nv if nv else compute,
            "load_read": load_read,
            "multi": nv > 1,
            "step_ms": d.get("step_ms_median"),
            "nsteps": d.get("steps_counted"),
            "nvml": d.get("peak_nvml_used_gib"),
            "levers": levers(d),
        })

    if args.sort == "wall":
        rows.sort(key=lambda r: r["wall"])
    elif args.sort == "pervideo":
        rows.sort(key=lambda r: r["per"])
    else:
        rows.sort(key=lambda r: r["run_id"])

    base = args.baseline
    hdr = (f"{'run_id':<46} {'tag':<18} {'stp':>3} {'n':>2} "
           f"{'wall_s':>8} {'per_vid':>8} {'load_s':>7} {'run_s':>7} "
           f"{'run/vid':>8} {'read_G':>7} {'Δbase':>7} {'stp_ms':>7} {'nvml':>6}  levers")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        per_s = f"{r['per']:.1f}{'*' if r['multi'] else ' '}"
        delta = r["per"] - base
        dstr = f"{delta:+.1f}"
        step_ms = f"{r['step_ms']:.1f}" if r["step_ms"] is not None else "-"
        nvml = f"{r['nvml']:.1f}" if r["nvml"] is not None else "-"
        print(f"{r['run_id']:<46} {r['tag']:<18} {str(r['steps']):>3} {r['nv']:>2} "
              f"{r['wall']:>8.1f} {per_s:>8} {r['load']:>7.1f} {r['compute']:>7.1f} "
              f"{r['compute_per']:>8.1f} {r['load_read']:>7.1f} {dstr:>7} "
              f"{step_ms:>7} {nvml:>6}  {r['levers']}")

    if any(r["multi"] for r in rows):
        print("\n* multi-video run: per_vid = wall / n_videos and still includes one "
              "cold iteration; treat as an upper bound on warm steady state.")
    print("run_s sums run:* and encode_mp4 phases; run/vid removes shared load cost. "
          "read_G is process storage read GiB during load phases (new runs only).")
    print("Only rows flagged `cold` (--drop-page-cache) have comparable load_s; every other "
          "load_s depends on what the page cache held when the run started.")
    print(f"Δbase is per-video seconds minus the {base:.0f}s reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
