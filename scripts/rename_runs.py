"""One-off: rename runs/, outputs/, logs/ from ad-hoc ids to <method>-<date>-<time>.

Dry run by default; pass --apply to rename. The old id is kept in run.json as
`renamed_from`, and every text file that mentions it is rewritten.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_naming import method_name  # noqa: E402

ROOT = "/mnt/workspace"
TEXT_REFS = ["docs/minimax-h3-inference-guide.md", "scripts/calib_load.sh"]


def new_id(old: str) -> str:
    r = json.load(open(os.path.join(ROOT, "runs", old, "run.json")))
    a = r.get("args", {})
    n = None
    if a.get("prompt_file"):
        n = len(json.load(open(os.path.join(ROOT, a["prompt_file"]))))
        if a.get("batch_limit"):
            n = min(n, a["batch_limit"])
    m = re.search(r"(\d{8}-\d{6})$", old)
    stamp = m.group(1) if m else r["started_at"].replace("-", "").replace(":", "").replace("T", "-")
    return f"{method_name(a, r.get('env', {}).get('attn_backend'), n)}-{stamp}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    plan = {}
    for old in sorted(os.listdir(os.path.join(ROOT, "runs"))):
        rj = os.path.join(ROOT, "runs", old, "run.json")
        if not os.path.exists(rj) or "args" not in json.load(open(rj)):
            continue  # still running; it was named by run_naming already
        new = new_id(old)
        if new != old:
            plan[old] = new
    assert len(set(plan.values())) == len(plan), "name collision"
    for old, new in plan.items():
        print(f"{old:<30} -> {new}")
    if not args.apply:
        return

    # Longest first, so e.g. batch5-compile-... is replaced before batch5-...
    order = sorted(plan, key=len, reverse=True)

    def rewrite(path: str) -> None:
        with open(path) as fh:
            s = fh.read()
        t = s
        for old in order:
            t = t.replace(old, plan[old])
        if t != s:
            with open(path, "w") as fh:
                fh.write(t)

    for old, new in plan.items():
        for sub in ("runs", "outputs"):
            src = os.path.join(ROOT, sub, old)
            if os.path.isdir(src):
                os.rename(src, os.path.join(ROOT, sub, new))
        log = os.path.join(ROOT, "logs", f"{old}.log")
        if os.path.exists(log):
            os.rename(log, os.path.join(ROOT, "logs", f"{new}.log"))
        rd = os.path.join(ROOT, "runs", new)
        for f in os.listdir(rd):
            if f.endswith((".json", ".txt", ".jsonl", ".csv", ".log")):
                rewrite(os.path.join(rd, f))
        rj = os.path.join(rd, "run.json")
        r = json.load(open(rj))
        r["renamed_from"] = old
        with open(rj, "w") as fh:
            json.dump(r, fh, indent=2, ensure_ascii=False)
    for f in TEXT_REFS:
        rewrite(os.path.join(ROOT, f))
    for f in os.listdir(os.path.join(ROOT, "logs")):
        rewrite(os.path.join(ROOT, "logs", f))


if __name__ == "__main__":
    main()
