"""Per-video stage breakdown in the guide's table style (§16), straight from run.json.

    python3 scripts/stage_table.py <run_id> [<run_id> ...]

Every value is seconds per video. Load rows are what the run measured; the
`L (unified)` row replaces them with the standard constant L_cold = 396.7 s (or
L_cold / N for a batch) for every run, cold or not, so the T rows of two columns
differ only by M, i.e. only by the lever under test (§7.4).
"""

from __future__ import annotations

import json
import os
import re
import sys

RUNS = "/mnt/workspace/runs"
L_COLD = 396.7
LOADS = [("text_encoder", "load:text_encoder"), ("vae", "load:vae_encoder"),
         ("transformer", "load:denoise"), ("decode vae", "load:decode")]
RUNS_ = [("text encode", "run:text_encoder"), ("denoise", "run:denoise"),
         ("decode", "run:decode"), ("mp4", "encode_mp4")]


def breakdown(run_id: str) -> dict:
    r = json.load(open(os.path.join(RUNS, run_id, "run.json")))
    ps = r["phase_seconds"]
    n = r.get("batch_size") or r.get("args", {}).get("repeat", 1) or 1

    def total(prefix: str) -> float:
        # "run:denoise", "run:denoise#1" (--repeat), "run:denoise:7" (batch)
        pat = re.compile(re.escape(prefix) + r"(?:[#:]\d+)?$")
        return sum(v for k, v in ps.items() if pat.match(k))

    wall = r["wall_seconds"]
    out = {"n": n, "wall": wall, "cold": bool(r.get("page_cache", {}).get("dropped"))}
    load_sum = 0.0
    for name, key in LOADS:
        v = total(key)
        load_sum += v
        out[f"load {name}"] = v / n
    out["load (measured)"] = load_sum / n
    timed = load_sum
    for name, key in RUNS_:
        v = total(key)
        timed += v
        out[name] = v / n
    free = sum(v for k, v in ps.items() if k.startswith("free:"))
    out["free"] = free / n
    out["other"] = (wall - timed - free) / n
    out["M"] = (wall - load_sum) / n
    # Only a batch amortizes load; a --repeat run stands in for single videos.
    batch = r.get("batch_size")
    out["L (unified)"] = L_COLD / batch if batch else L_COLD
    out["T"] = out["L (unified)"] + out["M"]
    out["step_ms_median"] = r.get("step_ms_median")
    out["nvml_peak"] = r.get("peak_nvml_used_gib")
    return out


def main() -> None:
    ids = sys.argv[1:]
    rows = [breakdown(i) for i in ids]
    keys = [k for k in rows[0] if k not in ("n", "wall", "cold")]
    print("| stage | " + " | ".join(ids) + " |")
    print("|---|" + "---|" * len(ids))
    for k in ["n", "cold", "wall"] + keys:
        vals = []
        for r in rows:
            v = r[k]
            vals.append(f"{v:.1f}" if isinstance(v, float) else str(v))
        print(f"| {k} | " + " | ".join(vals) + " |")


if __name__ == "__main__":
    main()
