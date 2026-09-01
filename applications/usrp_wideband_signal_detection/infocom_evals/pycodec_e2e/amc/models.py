"""AMC model zoo — three architectures from the literature, adapted to a
4-family + noise problem (PSK / QAM / FSK / OFDM / NOISE) on channelized IQ.

- VTCNN2   : O'Shea, Corgan, Clancy, "Convolutional Radio Modulation
             Recognition Networks" (2016). 2-conv CNN on a 2x128 IQ window.
- ResNet1D : O'Shea, Roy, Clancy, "Over-the-Air Deep Learning Based Radio
             Signal Classification" (2018). Six residual stacks on 2x1024 IQ.
- TPrime   : Belgiovine et al., "T-PRIME: Transformer-based Protocol
             Identification for Machine-learning at the Edge" (INFOCOM 2024,
             github.com/genesys-neu/t-prime). LG variant: 64 tokens of 128
             complex samples (interleaved I/Q -> d_model 256), 2 encoder
             layers, 8 heads, flatten head. No positional encoding (their
             default), LayerNorm on input.

All models take an RMS-normalized complex64 window and share the tensorizers
below so training and daemon inference agree exactly.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

CLASSES = ["PSK", "QAM", "FSK", "OFDM", "NOISE"]

WINDOW = {"vtcnn2": 128, "resnet1d": 1024, "tprime": 8192}
TPRIME_SEQ, TPRIME_SLICE = 64, 128  # LG: 64 tokens x 128 complex samples


def rms_normalize(iq: np.ndarray) -> np.ndarray:
    r = np.sqrt(np.mean(np.abs(iq) ** 2))
    return (iq / r).astype(np.complex64) if r > 0 else iq.astype(np.complex64)


def to_tensor(iq: np.ndarray, model: str) -> np.ndarray:
    """Complex window (already RMS-normalized, length WINDOW[model]) -> float32
    array in the model's input layout (no batch dim)."""
    if model == "tprime":
        s = iq.reshape(TPRIME_SEQ, TPRIME_SLICE)
        feat = np.empty((TPRIME_SEQ, 2 * TPRIME_SLICE), dtype=np.float32)
        feat[:, 0::2] = s.real  # chan2sequence interleave, per token
        feat[:, 1::2] = s.imag
        return feat
    x = np.stack([iq.real, iq.imag]).astype(np.float32)   # [2, N]
    if model == "vtcnn2":
        return x[None]                                     # [1, 2, 128]
    return x                                               # [2, 1024]


class VTCNN2(nn.Module):
    def __init__(self, classes: int = len(CLASSES)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 256, (1, 3), padding=(0, 2))
        self.conv2 = nn.Conv2d(256, 80, (2, 3), padding=(0, 2))
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(80 * 1 * (WINDOW["vtcnn2"] + 4), 256)
        self.fc2 = nn.Linear(256, classes)

    def forward(self, x):
        x = self.drop(torch.relu(self.conv1(x)))
        x = self.drop(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.drop(torch.relu(self.fc1(x)))
        return self.fc2(x)


class _ResUnit(nn.Module):
    def __init__(self, ch: int = 32):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv1d(ch, ch, 3, padding=1)

    def forward(self, x):
        y = torch.relu(self.c1(x))
        y = self.c2(y)
        return torch.relu(x + y)


class _ResStack(nn.Module):
    def __init__(self, in_ch: int, ch: int = 32):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, ch, 1)   # linear 1x1
        self.u1, self.u2 = _ResUnit(ch), _ResUnit(ch)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        return self.pool(self.u2(self.u1(self.proj(x))))


class ResNet1D(nn.Module):
    def __init__(self, classes: int = len(CLASSES)):
        super().__init__()
        self.stacks = nn.Sequential(
            _ResStack(2), *[_ResStack(32) for _ in range(5)])
        n_flat = 32 * (WINDOW["resnet1d"] // 2 ** 6)
        self.fc1 = nn.Linear(n_flat, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, classes)
        self.drop = nn.AlphaDropout(0.1)

    def forward(self, x):
        x = torch.flatten(self.stacks(x), 1)
        x = self.drop(torch.selu(self.fc1(x)))
        x = self.drop(torch.selu(self.fc2(x)))
        return self.out(x)


class TPrime(nn.Module):
    def __init__(self, classes: int = len(CLASSES), d_model: int = 2 * TPRIME_SLICE,
                 seq_len: int = TPRIME_SEQ, nlayers: int = 2, nhead: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.pre_classifier = nn.Linear(d_model * seq_len, d_model)
        self.drop = nn.Dropout(0.5)
        self.classifier = nn.Linear(d_model, classes)

    def forward(self, x):
        x = self.encoder(self.norm(x))
        x = torch.flatten(x, 1)
        x = self.drop(torch.relu(self.pre_classifier(x)))
        return self.classifier(x)


class TPrimeC(nn.Module):
    """Compressed-domain T-PRIME (codec-per-model experiment): tokens carry 256
    interleaved mantissa-normalized I/Q reals + 2 window-relative block
    exponents (matlab_ds.featurize), projected to d_model before the standard
    T-PRIME encoder. Identical architecture for every codec; only the input's
    mantissa precision differs."""

    def __init__(self, classes: int = 10, in_feats: int = 258,
                 d_model: int = 2 * TPRIME_SLICE, seq_len: int = TPRIME_SEQ,
                 nlayers: int = 2, nhead: int = 8):
        super().__init__()
        self.inproj = nn.Linear(in_feats, d_model)
        self.norm = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.pre_classifier = nn.Linear(d_model * seq_len, d_model)
        self.drop = nn.Dropout(0.5)
        self.classifier = nn.Linear(d_model, classes)

    def forward(self, x):
        x = self.encoder(self.norm(self.inproj(x)))
        x = torch.flatten(x, 1)
        x = self.drop(torch.relu(self.pre_classifier(x)))
        return self.classifier(x)


def build(name: str) -> nn.Module:
    return {"vtcnn2": VTCNN2, "resnet1d": ResNet1D, "tprime": TPrime}[name]()
