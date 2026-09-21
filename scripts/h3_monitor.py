#!/usr/bin/env python
"""Per-run telemetry for run_h3.py.

Rebuilt from the interface run_h3.py depends on (it imports RunMonitor at module
top, so nothing runs without this file) plus the run.json/metrics.csv/timeline.png
shape the analysis tooling expects. Design notes worth keeping:

* The five-stage driver frees each 62 GiB component group before loading the next,
  so a single wall-clock number hides everything. Timing is per-phase, and phases
  carry an iteration suffix (load:denoise#0 cold-disk vs #1 page-cache) -- keeping
  those apart *is* the cold/warm measurement, so the sampler tags every sample with
  whatever phase is currently open.

* denoise is the only GPU-bound stage worth a per-step number. instrument() wraps
  the transformer's forward with a pair of CUDA events per call; the median of those
  is step_ms_median. Events are read back once at stop() so the hot loop never
  synchronizes.

* run.json is written up front (env fingerprint + argv) and rewritten at stop().
  A crashed run still leaves its environment on disk -- the first 768p video was
  lost precisely because nothing about how it ran was recorded before it ran.
"""

from __future__ import annotations

import atexit
import contextlib
import csv
import json
import os
import socket
import sys
import threading
import time
import weakref

import torch

try:
    import pynvml
    _NVML_OK = True
except Exception:  # nvml missing -> nvml_used stays None, run still works
    _NVML_OK = False


def _host_used_gib() -> float:
    """Host RAM in use, read straight from /proc so it matches run_h3.report()."""
    info = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, _, v = line.partition(":")
            info[k] = int(v.split()[0])
    return (info["MemTotal"] - info["MemAvailable"]) / 1024 / 1024


def env_fingerprint() -> dict:
    """GPU / driver / torch / diffusers / CUDA versions into run.json's env section.

    This is what makes a run.json reproducible after the box is rebuilt -- the whole
    reason the 2026-09 recovery needed a version diff was that the old fingerprints
    lived only in these files. DIFFUSERS_ATTN_BACKEND is captured because 'sage'
    vs unset is the difference between the 178s and 188s baselines with no arg change.
    """
    env: dict = {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "attn_backend": os.environ.get("DIFFUSERS_ATTN_BACKEND"),
    }
    try:
        import diffusers
        env["diffusers"] = diffusers.__version__
    except Exception:
        env["diffusers"] = None
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        try:
            major, minor = torch.cuda.get_device_capability(0)
            env["gpu_capability"] = f"{major}.{minor}"
        except Exception:
            pass
    if _NVML_OK:
        try:
            drv = pynvml.nvmlSystemGetDriverVersion()
            env["driver"] = drv.decode() if isinstance(drv, bytes) else drv
        except Exception:
            pass
    for pkg in ("flash_attn", "sageattention", "triton"):
        try:
            mod = __import__(pkg)
            env[pkg] = getattr(mod, "__version__", "?")
        except Exception:
            pass
    return env


class _Tee:
    """Mirror a stream to run.log without swallowing the console.

    run.log is one of the four expected artifacts, and the only one that used to
    depend on the launcher piping through `tee`. Owning it here means a bare
    `python run_h3.py` still leaves a full log next to run.json/metrics.csv.
    """

    def __init__(self, stream, logfile):
        self._stream = stream
        self._log = logfile

    def write(self, data):
        self._stream.write(data)
        try:
            self._log.write(data)
        except Exception:
            pass
        return len(data)

    def flush(self):
        self._stream.flush()
        try:
            self._log.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


class RunMonitor:
    """Samples memory at `hz`, tags each sample with the open phase, and times
    per-phase wall clock plus per-step transformer latency.

    Usage (as run_h3.py calls it):
        mon = RunMonitor(run_dir, run_id, hz=2.0).start()
        with mon.phase("load:denoise#0"):
            ...
        mon.instrument(pipe.transformer, "transformer")
        summary = mon.stop(extra={...})   # writes run.json / metrics.csv / timeline.png
    """

    def __init__(self, run_dir: str, run_id: str, hz: float = 2.0) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.hz = hz if hz and hz > 0 else 2.0
        os.makedirs(run_dir, exist_ok=True)

        self._t0 = time.time()
        self._cur_phase = "init"
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None

        # phase name -> accumulated seconds (duplicate names sum, e.g. if a stage
        # is ever re-entered); order preserved for a readable run.json.
        self._phase_seconds: dict[str, float] = {}
        # sampler rows: (t_rel, phase, hbm_alloc, hbm_peak, host_used, nvml_used)
        self._samples: list[tuple] = []
        # (start_event, end_event) pairs per transformer forward, read at stop()
        self._step_events: list[tuple] = []
        # id(module) -> module, held WEAKLY so instrumenting the transformer never
        # keeps it alive: run_h3.py sets pipe.transformer=None then gc.collect() to
        # free 62 GiB between --repeat iterations, and a strong ref here would pin the
        # old transformer, so iteration 1's load collides at 62+62 GiB -> OOM.
        self._instrumented: "weakref.WeakValueDictionary[int, object]" = weakref.WeakValueDictionary()

        self._peak_host = 0.0
        self._peak_nvml = 0.0

        # run.log capture (installed in start(), restored in stop())
        self._log_fh = None
        self._orig_stdout = None
        self._orig_stderr = None

        self._nvml_handle = None
        if _NVML_OK:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(
                    torch.cuda.current_device() if torch.cuda.is_available() else 0
                )
            except Exception:
                self._nvml_handle = None

    # ---- lifecycle ----------------------------------------------------------

    def start(self) -> "RunMonitor":
        # Tee stdout/stderr to run.log up front so even a crash keeps its console.
        # Restored on stop(), and atexit as a safety net if stop() is never reached.
        self._log_fh = open(os.path.join(self.run_dir, "run.log"), "w", buffering=1)
        self._orig_stdout, self._orig_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(self._orig_stdout, self._log_fh)
        sys.stderr = _Tee(self._orig_stderr, self._log_fh)
        atexit.register(self._restore_streams)
        # Write run.json before anything runs so a crash still leaves the env behind.
        self._write_run_json(dict(status="running"))
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def _restore_streams(self) -> None:
        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
            self._orig_stdout = None
        if self._orig_stderr is not None:
            sys.stderr = self._orig_stderr
            self._orig_stderr = None
        if self._log_fh is not None:
            try:
                self._log_fh.flush()
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None

    def _sample_loop(self) -> None:
        period = 1.0 / self.hz
        while not self._stop_evt.is_set():
            self._take_sample()
            self._stop_evt.wait(period)
        # one last sample so the timeline reaches the end of the final phase
        self._take_sample()

    def _take_sample(self) -> None:
        hbm_alloc = hbm_peak = 0.0
        if torch.cuda.is_available():
            hbm_alloc = torch.cuda.memory_allocated() / 2**30
            hbm_peak = torch.cuda.max_memory_allocated() / 2**30
        host = _host_used_gib()
        nvml_used = None
        if self._nvml_handle is not None:
            try:
                nvml_used = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle).used / 2**30
            except Exception:
                nvml_used = None
        with self._lock:
            phase = self._cur_phase
            self._peak_host = max(self._peak_host, host)
            if nvml_used is not None:
                self._peak_nvml = max(self._peak_nvml, nvml_used)
            self._samples.append(
                (round(time.time() - self._t0, 3), phase,
                 round(hbm_alloc, 3), round(hbm_peak, 3),
                 round(host, 3), round(nvml_used, 3) if nvml_used is not None else None)
            )

    @contextlib.contextmanager
    def phase(self, name: str):
        with self._lock:
            prev = self._cur_phase
            self._cur_phase = name
        t0 = time.time()
        try:
            yield
        finally:
            dt = time.time() - t0
            with self._lock:
                self._phase_seconds[name] = self._phase_seconds.get(name, 0.0) + dt
                self._cur_phase = prev

    # ---- per-step transformer timing ---------------------------------------

    def instrument(self, module, name: str = "transformer") -> None:
        """Time every forward of `module` with CUDA events.

        run_h3.py re-instruments after each --repeat iteration because the
        transformer is freed and reloaded (a fresh object each time); events just
        keep accumulating into one step-latency population, which is what we want.
        Guard against double-hooking the *same* object so re-entry is harmless.
        """
        if not torch.cuda.is_available():
            return
        key = id(module)
        if key in self._instrumented:
            return
        self._instrumented[key] = module

        def _pre_hook(_mod, _inp):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            _mod._h3_step_start = start

        def _post_hook(_mod, _inp, _out):
            start = getattr(_mod, "_h3_step_start", None)
            if start is None:
                return
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._step_events.append((start, end))
            _mod._h3_step_start = None

        module.register_forward_pre_hook(_pre_hook)
        module.register_forward_hook(_post_hook)

    def _collect_step_ms(self) -> list[float]:
        if not self._step_events:
            return []
        torch.cuda.synchronize()
        out = []
        for start, end in self._step_events:
            try:
                out.append(start.elapsed_time(end))  # milliseconds
            except Exception:
                pass
        return out

    # ---- teardown + artifacts ----------------------------------------------

    def stop(self, extra: dict | None = None) -> dict:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

        step_ms = self._collect_step_ms()
        step_ms_sorted = sorted(step_ms)
        n = len(step_ms_sorted)
        step_ms_median = round(step_ms_sorted[n // 2], 1) if n else None
        step_ms_mean = round(sum(step_ms_sorted) / n, 1) if n else None

        peak_torch_alloc = (
            round(torch.cuda.max_memory_allocated() / 2**30, 2)
            if torch.cuda.is_available() else None
        )

        summary = {
            "run_id": self.run_id,
            "hz": self.hz,
            "wall_seconds": round(time.time() - self._t0, 1),
            "phase_seconds": {k: round(v, 1) for k, v in self._phase_seconds.items()},
            "step_ms_median": step_ms_median,
            "step_ms_mean": step_ms_mean,
            "steps_counted": n,
            "peak_torch_alloc_gib": peak_torch_alloc,
            "peak_nvml_used_gib": round(self._peak_nvml, 2) if self._peak_nvml else None,
            "peak_host_used_gib": round(self._peak_host, 1),
        }
        if extra:
            summary.update(extra)

        self._write_metrics_csv()
        self._write_run_json(summary)
        self._write_timeline_png()
        self._restore_streams()
        return summary

    def _write_run_json(self, payload: dict) -> None:
        doc = {
            "run_id": self.run_id,
            "argv": sys.argv,
            "env": env_fingerprint(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self._t0)),
        }
        doc.update(payload)
        if payload.get("status") != "running":
            doc["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(os.path.join(self.run_dir, "run.json"), "w") as fh:
            json.dump(doc, fh, indent=2, ensure_ascii=False, default=str)
            fh.write("\n")

    def _write_metrics_csv(self) -> None:
        with self._lock:
            rows = list(self._samples)
        with open(os.path.join(self.run_dir, "metrics.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t_rel", "phase", "hbm_alloc_gib", "hbm_peak_gib",
                        "host_used_gib", "nvml_used_gib"])
            w.writerows(rows)

    def _write_timeline_png(self) -> None:
        """Memory-over-time with phase bands -- the visual the guide calls timeline.png.

        Best-effort: a plotting failure must never sink an otherwise good run, so
        everything here is guarded and the CSV remains the source of truth.
        """
        with self._lock:
            rows = list(self._samples)
        if not rows:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            t = [r[0] for r in rows]
            hbm_alloc = [r[2] for r in rows]
            hbm_peak = [r[3] for r in rows]
            host = [r[4] for r in rows]

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(t, hbm_alloc, label="HBM allocated (GiB)", lw=1.4)
            ax.plot(t, hbm_peak, label="HBM peak (GiB)", lw=1.0, ls="--", alpha=0.7)
            ax.plot(t, host, label="host used (GiB)", lw=1.0, alpha=0.7)

            # shade phase spans and label the load/run/denoise ones so the plot reads
            # as a stage timeline rather than a raw memory trace.
            phases = [r[1] for r in rows]
            span_start = 0
            for i in range(1, len(phases) + 1):
                if i == len(phases) or phases[i] != phases[span_start]:
                    name = phases[span_start]
                    x0, x1 = t[span_start], t[min(i, len(t) - 1)]
                    if name and (name.startswith(("load", "run", "denoise", "compile", "reuse"))):
                        ax.axvspan(x0, x1, alpha=0.05, color="tab:blue")
                        ax.text(x0, ax.get_ylim()[1] * 0.98, name, rotation=90,
                                va="top", ha="left", fontsize=6, alpha=0.6)
                    span_start = i

            ax.set_xlabel("seconds since start")
            ax.set_ylabel("GiB")
            ax.set_title(f"{self.run_id} memory timeline")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(self.run_dir, "timeline.png"), dpi=110)
            plt.close(fig)
        except Exception as e:
            print(f"[mon ] timeline.png skipped: {type(e).__name__}: {e}", flush=True)
