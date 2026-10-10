import json, os, sys
sys.path.insert(0, "scripts"); import h3_report as h
REF = "sglang_resident50-20261008-120139"
# Sage Cache-DiT run -> (same-config FA run or None, Sage 0.24 run with the same MC or None)
SAGE = {
    "sglang_sage_cachedit024_resident50-20261009-213120": ("sglang_cachedit024_resident50-20261008-122639", None),
    "sglang_sage_cachedit032_resident50-20261009-214209": ("sglang_cachedit032_resident50-20261009-174546", "sglang_sage_cachedit024_resident50-20261009-213120"),
    "sglang_sage_cachedit024_resident50_mc6-20261009-215257": ("sglang_cachedit024_resident50_mc6-20261009-181334", None),
    "sglang_sage_cachedit024_resident50_mc9-20261009-220305": ("sglang_cachedit024_resident50_mc9-20261009-182346", None),
    "sglang_sage_cachedit032_resident50_mc6-20261009-223939": (None, "sglang_sage_cachedit024_resident50_mc6-20261009-215257"),
    "sglang_sage_cachedit032_resident50_mc9-20261009-224947": (None, "sglang_sage_cachedit024_resident50_mc9-20261009-220305"),
}
def mp4(rid):
    base = "outputs/bak" if os.path.exists(f"outputs/bak/{rid}") else "outputs"
    return f"{base}/{rid}/video.mp4"
def cmp(ref, tgt):
    psnr, ssim = h._video_quality(mp4(ref), mp4(tgt)); snr, n = h._audio_snr(mp4(ref), mp4(tgt))
    return {"run_id": tgt, "reference_run_id": ref, "video_psnr_db": round(psnr, 6), "video_ssim": round(ssim, 6),
            "audio_snr_db": round(snr, 6), "audio_samples": n, "passed": None}
fmt = lambda r: "%.3f/%.4f/%.3f" % (r["video_psnr_db"], r["video_ssim"], r["audio_snr_db"])
for sage, (fa, sage024) in SAGE.items():
    q = {"reference_run_id": REF, "comparison_basis": "encoded MP4; decoded video and 32 kHz stereo audio",
         "thresholds": {"min_psnr_db": None, "min_ssim": None, "min_audio_snr_db": None},
         "results": [cmp(REF, sage)]}
    if fa: q["vs_same_cachedit_fa"] = cmp(fa, sage)
    if sage024: q["vs_sage_cachedit024_same_mc"] = cmp(sage024, sage)
    json.dump(q, open(f"runs/{sage}/quality.json", "w"), indent=2)
    print(sage, "| vs raw", fmt(q["results"][0]),
          "| vs FA", fmt(q["vs_same_cachedit_fa"]) if fa else "-",
          "| vs sage024", fmt(q["vs_sage_cachedit024_same_mc"]) if sage024 else "-")
