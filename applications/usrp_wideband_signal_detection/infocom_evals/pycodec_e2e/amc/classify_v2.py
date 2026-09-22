"""v2 classifier wrapper — drop-in replacement for amc.classify.AmcClassifier.

Ships the release_v2 architectures (genesys-neu/t-prime lineage) and the v2
input contract: EVERY model is win=8192, and each window is normalized to
mean(I^2 + Q^2) == 1. The snippet handed to `classify` is ALREADY the isolated
signal at occupied_bw + 10 MHz (the snipper's frequency mode with an ADDITIVE
5 MHz/side guard), so there is no Welch/resample here — just window + normalize.

6 classes (classes.json order): qpsk, bfm, he, nr5g, ble, noise.

Interface parity with amc.classify.AmcClassifier so rt_decode_daemon can use
either: set_enabled(names), classify(iq)->{name: ModelResult}, resource_report(),
plus .models/.device/.gate and .color_model (which model tints the overlay).
"""
from __future__ import annotations

import json
import math
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WIN = 8192
N_WINDOWS = 4
# checklist names -> checkpoint file stem in weights_v2/
ALL_MODELS = ("resnet1d", "cnn1d", "tprime_lg")
_CKPT = {"resnet1d": "resnet1d_w8192.pt",
         "cnn1d": "cnn1d_w8192.pt",
         "tprime_lg": "tprime_lg_w8192.pt"}

_HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = json.load(open(os.path.join(_HERE, "weights_v2", "classes.json")))["classes"]


# --------------------------------------------------------------- architectures
# Ported verbatim from release_v2/ml/models.py (only the three shipped nets).

def _iq_interleave(x):
    out = x.new_empty(x.shape[0], 2 * x.shape[2])
    out[:, 0::2] = x[:, 0]
    out[:, 1::2] = x[:, 1]
    return out


class CNN1D(nn.Module):
    def __init__(self, win, n_classes=6, cap=256):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 7)
        self.conv2 = nn.Conv1d(64, 128, 3)
        self.cap = cap
        with torch.no_grad():
            nf = self._feat(torch.zeros(1, 2, win)).shape[1]
        self.fc1 = nn.Linear(nf, 256)
        self.fc2 = nn.Linear(256, n_classes)

    def _feat(self, x):
        x = _iq_interleave(x).unsqueeze(1)
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = F.adaptive_max_pool1d(x, min(self.cap, x.shape[-1]))
        return torch.flatten(x, 1)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(self._feat(x))))


class _ResidualStack(nn.Module):
    def __init__(self, cin, cout, kernel, pool, norm=False, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 1, padding='same')
        self.conv2 = nn.Conv2d(cout, cout, kernel, padding='same')
        self.conv3 = nn.Conv2d(cout, cout, kernel, padding='same')
        self.conv4 = nn.Conv2d(cout, cout, kernel, padding='same')
        self.conv5 = nn.Conv2d(cout, cout, kernel, padding='same')
        self.pool = pool
        g = math.gcd(groups, cout)
        self.n2, self.n3, self.n4, self.n5 = (
            (nn.GroupNorm(g, cout) if norm else nn.Identity()) for _ in range(4))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x + self.n3(self.conv3(F.relu(self.n2(self.conv2(x))))))
        x = F.relu(x + self.n5(self.conv5(F.relu(self.n4(self.conv4(x))))))
        return F.max_pool2d(x, self.pool, self.pool)


class ResNet1D(nn.Module):
    def __init__(self, win, n_classes=6, kernel=5, pool=2, nk=32, cap=16,
                 nstacks=None, norm=None):
        super().__init__()
        if nstacks is None:
            nstacks = max(6, math.ceil(math.log2(max(1, win / cap))))
        if norm is None:
            norm = nstacks > 6
        self.nstacks, self.norm = nstacks, norm
        self.s1 = _ResidualStack(1, nk, (kernel, 2), (pool, 2), norm)
        self.rest = nn.ModuleList(
            [_ResidualStack(nk, nk, (kernel, 1), (pool, 1), norm)
             for _ in range(nstacks - 1)])
        self.cap = cap
        self.fc1 = nn.Linear(nk * cap, 128)
        self.drop = nn.AlphaDropout(0.3)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = self.s1(x)
        for s in self.rest:
            x = s(x)
        x = F.adaptive_avg_pool2d(x, (self.cap, 1))
        x = torch.flatten(x, 1)
        return self.fc2(self.drop(F.relu(self.fc1(x))))


class TPrime(nn.Module):
    def __init__(self, seq_len=64, slice_len=128, n_classes=6, nlayers=2, nhead=8):
        super().__init__()
        d = 2 * slice_len
        self.seq_len, self.slice_len, self.d = seq_len, slice_len, d
        self.norm = nn.LayerNorm(d)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, nhead, batch_first=True), nlayers)
        self.pre = nn.Linear(d * seq_len, d)
        self.drop = nn.Dropout(0.5)
        self.cls = nn.Linear(d, n_classes)

    def forward(self, x):
        src = _iq_interleave(x).view(x.shape[0], self.seq_len, self.d)
        t = self.enc(self.norm(src))
        return self.cls(self.drop(F.relu(self.pre(torch.flatten(t, 1)))))


def _build_from_ckpt(b, n_classes):
    """Reconstruct the net from a release_v2 checkpoint dict {model,win,state_dict}."""
    name, win, sd = b["model"], b["win"], b["state_dict"]
    if name == "cnn1d":
        return CNN1D(win, n_classes)
    if name == "tprime_lg":
        return TPrime(64, 128, n_classes)
    if name == "resnet1d":
        idx = {int(m.group(1)) for k in sd
               for m in [re.match(r"rest\.(\d+)\.", k)] if m}
        nstacks = (max(idx) + 2) if idx else 6
        norm = "s1.n2.weight" in sd
        return ResNet1D(win, n_classes, nstacks=nstacks, norm=norm)
    raise ValueError(f"unknown v2 model {name}")


# ------------------------------------------------------------------- wrapper

class ModelResult:
    __slots__ = ("name", "label", "conf", "probs", "ms")

    def __init__(self, name, label, conf, probs, ms):
        self.name, self.label, self.conf, self.probs, self.ms = \
            name, label, conf, probs, ms


def _profile_flops(net, x) -> float:
    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     with_flops=True) as prof:
            with torch.no_grad():
                net(x)
        return float(sum(e.flops for e in prof.key_averages() if e.flops))
    except Exception:
        return 0.0


def _win_tensor(w: np.ndarray) -> np.ndarray:
    """complex window -> (2, WIN) float32, normalized to mean(I^2+Q^2)==1."""
    x = np.stack([w.real, w.imag]).astype(np.float32)      # (2, WIN)
    x /= (np.sqrt((x ** 2).sum(0).mean()) + 1e-12)
    return x


class AmcClassifier:
    V2 = True   # daemon marker: classify the whole isolated snippet, no re-channelize

    def __init__(self, weights_dir: str | None = None, device: str | None = None,
                 gate: str = "resnet1d", enabled: tuple[str, ...] | None = None):
        self.weights_dir = weights_dir or os.path.join(_HERE, "weights_v2")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gate = gate if gate in ALL_MODELS else "resnet1d"
        # which model's prediction tints the "Color Mask by Class" overlay
        self.color_model = self.gate
        self.models: dict[str, torch.nn.Module] = {}
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
        b = torch.load(os.path.join(self.weights_dir, _CKPT[name]),
                       map_location="cpu", weights_only=False)
        net = _build_from_ckpt(b, len(CLASSES))
        net.load_state_dict({k: v.float() for k, v in b["state_dict"].items()})
        net.eval().to(self.device)
        warm = torch.from_numpy(np.stack(
            [_win_tensor(np.ones(WIN, np.complex64))] * N_WINDOWS)).to(self.device)
        with torch.no_grad():
            net(warm)
        # resnet1d is Conv2d(padding='same'): torch's flop counter can't handle it and floods
        # stderr with warnings for no useful number, so skip profiling it (params+latency remain).
        flops = 0.0 if name == "resnet1d" else _profile_flops(net, warm[:1])
        if self.device == "cuda":
            torch.cuda.synchronize()
            gpu_mb = (torch.cuda.memory_allocated() - mem0) / 1e6
        else:
            gpu_mb = 0.0
        self.models[name] = net
        self.resources[name] = dict(
            params=sum(p.numel() for p in net.parameters()),
            gpu_mb=round(gpu_mb, 1), flops_per_window=flops,
            recent_ms=[], calls=0, windows=0)

    def _unload(self, name: str):
        if name not in self.models:
            return
        del self.models[name]
        self.resources.pop(name, None)
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def set_enabled(self, names) -> bool:
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
    def _windows(self, iq: np.ndarray) -> np.ndarray:
        """Cut up to N_WINDOWS energy-gated 8192-sample windows from the
        isolated signal; tile if shorter than one window."""
        if iq.size < WIN:
            iq = np.tile(iq, int(np.ceil(WIN / iq.size)))
        k = min(2 * N_WINDOWS, max(1, iq.size // WIN))
        starts = np.linspace(0, iq.size - WIN, k).astype(int)
        wins = [iq[s:s + WIN] for s in starts]
        e = np.array([float(np.mean(np.abs(w) ** 2)) for w in wins])
        kept = [w for w, ei in zip(wins, e) if ei >= 0.25 * e.max()]
        if len(kept) > N_WINDOWS:
            idx = np.linspace(0, len(kept) - 1, N_WINDOWS).astype(int)
            kept = [kept[i] for i in idx]
        return np.stack([_win_tensor(w) for w in kept])

    def classify(self, iq: np.ndarray) -> dict[str, ModelResult]:
        iq = np.asarray(iq, dtype=np.complex64)
        out: dict[str, ModelResult] = {}
        with torch.no_grad():
            for name, net in self.models.items():
                t0 = time.time()
                x = torch.from_numpy(self._windows(iq)).to(self.device)
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
        rep = {}
        for name, r in self.resources.items():
            recent = r["recent_ms"]
            rep[name] = dict(params=r["params"], gpu_mb=r["gpu_mb"],
                             gflops_per_window=round(r["flops_per_window"] / 1e9, 3),
                             ms_avg=round(sum(recent) / len(recent), 2) if recent else None,
                             calls=r["calls"], windows=r["windows"])
        return rep
