#!/usr/bin/env python3
"""Aggregate runs/*/run.json into a comparison table for MiniMax-H3 experiments.

Reporting discipline (00-REPRODUCE §5):
  - Compare acceleration changes against the DELIVERY BASELINE 188s (FBCache+reuse,
    warm), not the original 1212.5s.
  - Known defect (Run 11 lesson): a --repeat / batch run reports total wall for N
    videos. Comparing that raw total to a single-video baseline is wrong, so this
    tool always splits out per-video = wall / n_videos and flags multi-video runs
    with `*` — the per-video number still mixes one cold iteration with warm ones,
    so treat `*` rows as an upper bound on the warm steady state.

Usage:
  scripts/h3_report.py                     # table for /mnt/workspace/runs
  scripts/h3_report.py --tag fbcache       # only runs whose tag contains "fbcache"
  scripts/h3_report.py --sort pervideo     # sort by per-video seconds
  scripts/h3_report.py --baseline 188      # delta reference (default 188s)
"""
import argparse
import glob
import json
import os
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
    return ",".join(flags) or "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="/mnt/workspace/runs")
    ap.add_argument("--baseline", type=float, default=188.0,
                    help="per-video reference seconds for delta (default 188s delivery baseline)")
    ap.add_argument("--tag", default=None, help="substring filter on tag")
    ap.add_argument("--sort", choices=["run_id", "wall", "pervideo"], default="run_id")
    args = ap.parse_args()

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
        rows.append({
            "run_id": d.get("run_id", "?"),
            "tag": str(d.get("tag", "-"))[:18],
            "steps": a.get("steps", "?"),
            "nv": nv,
            "wall": wall,
            "per": per,
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
    hdr = (f"{'run_id':<16} {'tag':<18} {'stp':>3} {'n':>2} "
           f"{'wall_s':>8} {'per_vid':>8} {'Δbase':>7} {'stp_ms':>7} {'nvml':>6}  levers")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        per_s = f"{r['per']:.1f}{'*' if r['multi'] else ' '}"
        delta = r["per"] - base
        dstr = f"{delta:+.1f}"
        step_ms = f"{r['step_ms']:.1f}" if r["step_ms"] is not None else "-"
        nvml = f"{r['nvml']:.1f}" if r["nvml"] is not None else "-"
        print(f"{r['run_id']:<16} {r['tag']:<18} {str(r['steps']):>3} {r['nv']:>2} "
              f"{r['wall']:>8.1f} {per_s:>8} {dstr:>7} {step_ms:>7} {nvml:>6}  {r['levers']}")

    if any(r["multi"] for r in rows):
        print("\n* multi-video run: per_vid = wall / n_videos and still includes one "
              "cold iteration; treat as an upper bound on warm steady state.")
    print(f"\nΔbase is per-video seconds minus the {base:.0f}s reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
