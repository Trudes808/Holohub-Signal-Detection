"""Inference wrapper the decode daemon uses: classify one channelized
sub-band with all three models; the caller routes on the gate model's answer.

Windows are cut evenly across the band signal, RMS-normalized per window,
and the per-window softmaxes are averaged (bands shorter than a model's
window get cyclically tiled — pycodec bursts are tiled back-to-back anyway).
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from .models import CLASSES, WINDOW, build, rms_normalize, to_tensor

N_WINDOWS = 4


class ModelResult:
    __slots__ = ("name", "label", "conf", "probs", "ms")

    def __init__(self, name, label, conf, probs, ms):
        self.name, self.label, self.conf, self.probs, self.ms = \
            name, label, conf, probs, ms


class AmcClassifier:
    def __init__(self, weights_dir: str | None = None, device: str | None = None,
                 gate: str = "tprime"):
        weights_dir = weights_dir or os.path.join(os.path.dirname(__file__), "weights")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gate = gate
        self.models: dict[str, torch.nn.Module] = {}
        for name in ("vtcnn2", "resnet1d", "tprime"):
            net = build(name)
            state = torch.load(os.path.join(weights_dir, f"{name}.pt"),
                               map_location="cpu", weights_only=True)
            net.load_state_dict({k: v.float() for k, v in state.items()})
            net.eval().to(self.device)
            self.models[name] = net
        # one warmup pass so the first live snippet doesn't pay CUDA init
        self.classify(np.zeros(WINDOW["tprime"], dtype=np.complex64))

    def _windows(self, iq: np.ndarray, n: int) -> np.ndarray:
        if iq.size < n:
            iq = np.tile(iq, int(np.ceil(n / iq.size)))
        k = min(N_WINDOWS, max(1, iq.size // n))
        starts = np.linspace(0, iq.size - n, k).astype(int)
        return np.stack([to_tensor(rms_normalize(iq[s:s + n]), self._name)
                         for s in starts])

    def classify(self, iq: np.ndarray) -> dict[str, ModelResult]:
        iq = np.asarray(iq, dtype=np.complex64)
        out: dict[str, ModelResult] = {}
        with torch.no_grad():
            for name, net in self.models.items():
                t0 = time.time()
                self._name = name
                x = torch.from_numpy(self._windows(iq, WINDOW[name])).to(self.device)
                probs = torch.softmax(net(x), dim=1).mean(0).cpu().numpy()
                if self.device == "cuda":
                    torch.cuda.synchronize()
                k = int(probs.argmax())
                out[name] = ModelResult(name, CLASSES[k], float(probs[k]),
                                        probs, (time.time() - t0) * 1e3)
        return out
