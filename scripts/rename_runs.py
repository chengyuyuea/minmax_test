"""Rename run ids to `<framework>_[batch]_[sage]_[cache]_[other]-<stamp>` (see run_naming.py).

Dry run by default; pass --apply to rename. Each run is located under runs/ or runs/bak/
(outputs/ mirrors it), the directory pair and logs/<id>.log are moved, the old id is
appended to run.json `renamed_from`, and every derived text file that mentions an old id
is rewritten in one pass. Raw evidence (cmd.txt, *.log) is left untouched.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess

ROOT = "/mnt/workspace"
MAP = {
    # diffusers, standard 50-sigma
    "raw-20260929-103055": "diffusers_raw-20260929-103055",
    "sage-20260928-165335": "diffusers_sage-20260928-165335",
    "fbcache025-20260928-203358": "diffusers_fbcache025-20260928-203358",
    "sage_fbcache015-20260928-200056": "diffusers_sage_fbcache015-20260928-200056",
    "sage_fbcache020-20260928-201152": "diffusers_sage_fbcache020-20260928-201152",
    "sage_fbcache025-20260928-204352": "diffusers_sage_fbcache025-20260928-204352",
    "batch5_sage_fbcache025-20260928-105413": "diffusers_batch5_sage_fbcache025-20260928-105413",
    "batch10_sage_fbcache025-20260928-111414": "diffusers_batch10_sage_fbcache025-20260928-111414",
    "batch20_sage_fbcache025-20260928-114735": "diffusers_batch20_sage_fbcache025-20260928-114735",
    # Turbo LoRA
    "sage_turbolora_v4_8eval-20261008-160115": "turbolora_sage_v4_8eval-20261008-160115",
    # SGLang
    "sglang_resident50_cachedit024-20261008-122639": "sglang_cachedit024_resident50-20261008-122639",
    "sglang_sage_cachedit024_mc6_resident50-20261009-215257": "sglang_sage_cachedit024_resident50_mc6-20261009-215257",
    "sglang_sage_cachedit024_mc9_resident50-20261009-220305": "sglang_sage_cachedit024_resident50_mc9-20261009-220305",
    "sglang_sage_cachedit032_mc6_resident50-20261009-223939": "sglang_sage_cachedit032_resident50_mc6-20261009-223939",
    # bak: diffusers
    "raw-20260920-231616": "diffusers_raw-20260920-231616",
    "fl2va_raw-20260920-234316": "diffusers_fl2va-20260920-234316",
    "batch3_sage_fbcache025-20260920-213701": "diffusers_batch3_sage_fbcache025-20260920-213701",
    "batch5_sage_fbcache025_compile-20260928-125513": "diffusers_batch5_sage_fbcache025_compile-20260928-125513",
    "fbcache025_repeat2_reuse-20260920-181316": "diffusers_fbcache025_repeat2_reuse-20260920-181316",
    "sage_fbcache025_repeat2_reuse-20260920-211122": "diffusers_sage_fbcache025_repeat2_reuse-20260920-211122",
    "steps2-20260920-172251": "diffusers_steps2-20260920-172251",
    "steps2-20260928-202235": "diffusers_steps2-20260928-202235",
    "sage_steps2-20260928-202608": "diffusers_sage_steps2-20260928-202608",
    "steps2_repeat2_reuse-20260928-145944": "diffusers_steps2_repeat2_reuse-20260928-145944",
    "steps2_repeat2_reuse-20260928-150823": "diffusers_steps2_repeat2_reuse-20260928-150823",
    "nodevmap_steps2-20260928-152320": "diffusers_nodevmap_steps2-20260928-152320",
    "probe_parallel_off-20260929-01": "diffusers_parallel_off_steps2_probe-20260929-01",
    "probe_parallel_on-20260929-01": "diffusers_parallel_on_steps2_probe-20260929-01",
    # bak: SGLang
    "sglang-smoke-20261007-01": "sglang_steps2_smoke-20261007-01",
    "probe_sglang_resident025_steps4-20261008-112439": "sglang_resident025_steps4_probe-20261008-112439",
    "probe_sglang_resident050_steps4-20261008-113440": "sglang_resident050_steps4_probe-20261008-113440",
    "probe_sglang_resident075_steps4-20261008-114343": "sglang_resident075_steps4_probe-20261008-114343",
    "probe_sglang_resident50_steps4-20261008-115211": "sglang_resident50_steps4_probe-20261008-115211",
    "sglang_resident50_cachedit016-20261009-174251": "sglang_cachedit016_resident50-20261009-174251",
    "sglang_resident50_cachedit020-20261009-173134": "sglang_cachedit020_resident50-20261009-173134",
    "sglang_resident50_cachedit032-20261009-174546": "sglang_cachedit032_resident50-20261009-174546",
    "sglang_resident50_cachedit024_mc6-20261009-181334": "sglang_cachedit024_resident50_mc6-20261009-181334",
    "sglang_resident50_cachedit024_mc9-20261009-182346": "sglang_cachedit024_resident50_mc9-20261009-182346",
}
PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(k) for k in sorted(MAP, key=len, reverse=True)) + r")(?![0-9])"
)
TEXT_GLOBS = [
    "README.md", "recovery_guide.md", "docs/minimax-h3-inference-guide.md",
    "scripts/*.py", "scripts/*.sh", "logs/*.json", "logs/*.py", "logs/*.sh",
    "runs/*/*.json", "runs/*/*.csv", "runs/bak/*/*.json", "runs/bak/*/*.csv",
]


def tracked(path: str) -> bool:
    out = subprocess.run(["git", "ls-files", path], cwd=ROOT, capture_output=True, text=True).stdout
    return bool(out.strip())


def move(src: str, dst: str) -> None:
    assert not os.path.exists(dst), dst
    if tracked(os.path.relpath(src, ROOT)):
        subprocess.run(["git", "mv", src, dst], cwd=ROOT, check=True)
    else:
        os.rename(src, dst)


def locate(rid: str) -> str | None:
    for level in ("", "bak"):
        if os.path.isdir(os.path.join(ROOT, "runs", level, rid)):
            return level
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    assert len(set(MAP.values())) == len(MAP), "name collision"

    plan = []
    for old, new in MAP.items():
        level = locate(old)
        if level is None:
            if locate(new) is None:
                print(f"MISSING {old}")
            continue
        plan.append((old, new, level))
        print(f"{level or 'main':<4} {old:<55} -> {new}")
    if not args.apply:
        return

    for old, new, level in plan:
        for sub in ("runs", "outputs"):
            src = os.path.join(ROOT, sub, level, old)
            if os.path.isdir(src):
                move(src, os.path.join(ROOT, sub, level, new))
        log = os.path.join(ROOT, "logs", f"{old}.log")
        if os.path.exists(log):
            move(log, os.path.join(ROOT, "logs", f"{new}.log"))

    for pattern in TEXT_GLOBS:
        for path in glob.glob(os.path.join(ROOT, pattern)):
            if os.path.basename(path) == "rename_runs.py":
                continue
            with open(path) as fh:
                s = fh.read()
            t = PATTERN.sub(lambda m: MAP[m.group(1)], s)
            if t != s:
                with open(path, "w") as fh:
                    fh.write(t)

    for old, new, level in plan:
        rj = os.path.join(ROOT, "runs", level, new, "run.json")
        if not os.path.exists(rj):
            continue
        r = json.load(open(rj))
        prev = r.get("renamed_from")
        history = prev if isinstance(prev, list) else ([prev] if prev else [])
        r["renamed_from"] = history + [old]
        with open(rj, "w") as fh:
            json.dump(r, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
