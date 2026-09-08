"""Inference wrapper the daemon uses: classify one channelized sub-band with
the ENABLED subset of models (dashboard checklist; any of vtcnn2/resnet1d/
tprime, hot enable/disable at runtime). Per-model resource accounting rides
along: parameter count, GPU memory measured at load, analytic/profiled FLOPs
per window, and rolling latency — the dashboard's compute-cost story.

Windows are cut evenly across the band signal, RMS-normalized per window,
and the per-window softmaxes are median-aggregated (bands shorter than a
model's window get cyclically tiled).
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from .models import CLASSES, WINDOW, build, rms_normalize, to_tensor

N_WINDOWS = 4
ALL_MODELS = ("vtcnn2", "resnet1d", "tprime")


class ModelResult:
    __slots__ = ("name", "label", "conf", "probs", "ms")

    def __init__(self, name, label, conf, probs, ms):
        self.name, self.label, self.conf, self.probs, self.ms = \
            name, label, conf, probs, ms


def _profile_flops(net, x) -> float:
    """FLOPs for one forward of x via torch.profiler (0.0 if unavailable)."""
    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     with_flops=True) as prof:
            with torch.no_grad():
                net(x)
        return float(sum(e.flops for e in prof.key_averages() if e.flops))
    except Exception:
        return 0.0


class AmcClassifier:
    def __init__(self, weights_dir: str | None = None, device: str | None = None,
                 gate: str = "tprime", enabled: tuple[str, ...] | None = None):
        self.weights_dir = weights_dir or os.path.join(os.path.dirname(__file__), "weights")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gate = gate
        self.models: dict[str, torch.nn.Module] = {}
        # name -> dict(params, gpu_mb, flops_per_window, ms_sum, calls, windows)
        self.resources: dict[str, dict] = {}
        for name in enabled if enabled is not None else ALL_MODELS:
            self._load(name)

    # ------------------------------------------------------------ lifecycle --

    def _load(self, name: str):
        if name in self.models or name not in ALL_MODELS:
            return
        if self.device == "cuda":
            torch.cuda.synchronize()
            mem0 = torch.cuda.memory_allocated()
        net = build(name)
        state = torch.load(os.path.join(self.weights_dir, f"{name}.pt"),
                           map_location="cpu", weights_only=True)
        net.load_state_dict({k: v.float() for k, v in state.items()})
        net.eval().to(self.device)
        n = WINDOW[name]
        warm = torch.from_numpy(np.stack(
            [to_tensor(rms_normalize(np.ones(n, np.complex64)), name)] * N_WINDOWS)).to(self.device)
        with torch.no_grad():
            net(warm)
        flops = _profile_flops(net, warm[:1])
        if self.device == "cuda":
            torch.cuda.synchronize()
            gpu_mb = (torch.cuda.memory_allocated() - mem0) / 1e6
        else:
            gpu_mb = 0.0
        self.models[name] = net
        self.resources[name] = dict(
            params=sum(p.numel() for p in net.parameters()),
            gpu_mb=round(gpu_mb, 1),
            flops_per_window=flops,
            recent_ms=[], calls=0, windows=0)

    def _unload(self, name: str):
        if name not in self.models:
            return
        del self.models[name]
        self.resources.pop(name, None)
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def set_enabled(self, names) -> bool:
        """Reconcile the loaded model set with the checklist. Returns True if
        the set changed. Unloading frees GPU memory (visible in the HUD)."""
        want = [n for n in ALL_MODELS if n in set(names)]
        changed = False
        for n in list(self.models):
            if n not in want:
                self._unload(n)
                changed = True
        for n in want:
            if n not in self.models:
                self._load(n)
                changed = True
        return changed

    # ------------------------------------------------------------ inference --

    def _windows(self, iq: np.ndarray, n: int) -> np.ndarray:
        if iq.size < n:
            iq = np.tile(iq, int(np.ceil(n / iq.size)))
        # Live detection boxes bridge guard gaps, so an evenly-spread window
        # can land on dead air and outvote the signal windows: sample more
        # candidates, keep those within ~6 dB of the strongest.
        k = min(2 * N_WINDOWS, max(1, iq.size // n))
        starts = np.linspace(0, iq.size - n, k).astype(int)
        wins = [iq[s:s + n] for s in starts]
        e = np.array([float(np.mean(np.abs(w) ** 2)) for w in wins])
        kept = [w for w, ei in zip(wins, e) if ei >= 0.25 * e.max()]
        if len(kept) > N_WINDOWS:
            idx = np.linspace(0, len(kept) - 1, N_WINDOWS).astype(int)
            kept = [kept[i] for i in idx]
        return np.stack([to_tensor(rms_normalize(w), self._name) for w in kept])

    def classify(self, iq: np.ndarray) -> dict[str, ModelResult]:
        iq = np.asarray(iq, dtype=np.complex64)
        out: dict[str, ModelResult] = {}
        with torch.no_grad():
            for name, net in self.models.items():
                t0 = time.time()
                self._name = name
                x = torch.from_numpy(self._windows(iq, WINDOW[name])).to(self.device)
                # median across windows: a lone outlier window cannot outvote
                probs = torch.softmax(net(x), dim=1).median(0).values.cpu().numpy()
                if self.device == "cuda":
                    torch.cuda.synchronize()
                ms = (time.time() - t0) * 1e3
                k = int(probs.argmax())
                out[name] = ModelResult(name, CLASSES[k], float(probs[k]), probs, ms)
                r = self.resources[name]
                r["recent_ms"].append(ms)
                if len(r["recent_ms"]) > 32:
                    r["recent_ms"].pop(0)
                r["calls"] += 1
                r["windows"] += int(x.shape[0])
        return out

    def resource_report(self) -> dict:
        """Static + cumulative per-model stats for the dashboard (which turns
        the cumulative windows counter into windows/s from deltas)."""
        rep = {}
        for name, r in self.resources.items():
            recent = r["recent_ms"]
            rep[name] = dict(params=r["params"], gpu_mb=r["gpu_mb"],
                             gflops_per_window=round(r["flops_per_window"] / 1e9, 3),
                             ms_avg=round(sum(recent) / len(recent), 2) if recent else None,
                             calls=r["calls"], windows=r["windows"])
        return rep
