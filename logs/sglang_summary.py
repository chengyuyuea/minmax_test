import json, sys, hashlib, glob, os
sys.path.insert(0, "scripts"); import h3_report as h
REF = "outputs/sglang_resident50-20261008-120139/video.mp4"
def row(rid, ref=REF):
    base = "runs/bak" if os.path.exists(f"runs/bak/{rid}") else "runs"
    obase = base.replace("runs", "outputs")
    r = json.load(open(f"{base}/{rid}/run.json"))
    if not os.path.exists(f"{base}/{rid}/perf.json"):
        print(f"{rid} | status {r.get('status')} | no perf.json"); return
    p = json.load(open(f"{base}/{rid}/perf.json"))
    d = [x["duration_ms"] / 1000 for x in p["denoise_steps_ms"]]
    full = sum(1 for x in d if x > 5)
    v = f"{obase}/{rid}/video.mp4"
    psnr, ssim = h._video_quality(ref, v); snr, _ = h._audio_snr(ref, v)
    sha = hashlib.sha256(open(v, "rb").read()).hexdigest()[:16]
    ph = r["phase_seconds"]
    print(f"{rid} | full {full} | load {ph['load:pipeline']} | text {ph.get('run:text_encoder')} | den {ph['run:denoise']} | dec {ph.get('run:decode')} "
          f"| req {r['run_seconds_total']} | wall {r.get('wall_seconds')} | step_med {r.get('step_ms_median')} | nvml {r.get('peak_nvml_used_gib')} "
          f"| {psnr:.3f}/{ssim:.4f}/{snr:.3f} | {sha}")
for rid in sys.argv[1:]:
    row(rid)
