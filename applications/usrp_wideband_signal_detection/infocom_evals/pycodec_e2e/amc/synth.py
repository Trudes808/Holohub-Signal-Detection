"""Synthetic training data for the AMC classifier: windows generated through
the SAME front-end the daemon uses at inference, so there is no domain gap.

Per example:
  framed pycodec burst (cyclic -> tiled) at 245.76 MSps
  -> snipper-style decimation to snip_fs = 245.76/2^k   (k = 0..4)
  -> residual center error (find_subbands centers on coarse FFT boxes)
  -> optional neighbor signal at slot spacing (stacked-slot leakage)
  -> AWGN at U[0, 30] dB in-band SNR (added BEFORE the channel filter,
     exactly like noise inside a real snippet)
  -> pycodec.frame.channelize() with a jittered bandwidth estimate
     (find_subbands' -22 dB box edges are not exact)
  -> 8192-sample window (smaller models slice their prefix)

NOISE class: white noise through channelize() at a random plausible
bandwidth — precisely what a spurious sub-band looks like.

Family members (class = family, one-of-N member per example):
  PSK  : BPSK, QPSK, 8PSK    (composite carries BPSK)
  QAM  : 16QAM
  FSK  : 2FSK, 4FSK          (composite carries 4FSK)
  OFDM : payload QPSK, 16QAM (composite carries payload QPSK)
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys

import numpy as np
from scipy.signal import resample_poly

PYCODEC_ROOT = os.environ.get(
    "PYCODEC_ROOT", os.path.expanduser("~/Documents/holoscan_waveform_generation"))
sys.path.insert(0, PYCODEC_ROOT)

from pycodec.frame import channelize  # noqa: E402

from .models import CLASSES  # noqa: E402

WIN = 8192          # longest model window (T-PRIME LG); others slice a prefix
PAD = 768           # channelize()/window margin
FS0 = 245.76e6
K_MAX = 4           # snipper decimation ladder: snip_fs = FS0 / 2^k
CFO_FRAC_OCC = 0.05  # residual center error, fraction of the signal bandwidth
SNR_DB = (0.0, 45.0)   # Clean runs at 50 dB; keep the top of the range near it
NEIGHBOR_P = 0.35   # chance of a stacked-slot neighbor leaking into the band
# Payload length AND content are randomized per burst as a stratified
# (content kind x length) grid: every nuisance dimension of the frame must
# carry NO class information AND every joint combination must exist for
# every member, or the models learn the gap. All three leaks were observed
# on real snips: (a) fixed 2048-bit payloads made frame period a class cue;
# (b) pure-random training bits made PN9's 511-bit periodicity OOD, so real
# PN9-payload 16QAM classified PSK at 1.0 confidence; (c) with only 4
# random realizations per member the (PN9 x 4096) cell was usually missing,
# so the frame-spanning T-PRIME window still failed on real QAM while the
# 1024-sample ResNet (local stats only) was fixed.
PAYLOAD_BITS = [512, 1024, 2048, 4096]
PAYLOAD_KINDS = ["pn9", "pn9mid", "ascii", "random"]
REALIZATIONS = len(PAYLOAD_KINDS) * len(PAYLOAD_BITS)   # full 4x4 grid


def _payload_bits(kind: str, nbits: int, rng: np.random.Generator) -> np.ndarray:
    from pycodec.pn9 import bits_from_bytes, pn9_reference
    if kind == "pn9":         # PN9 from seed, exactly like the composite entries
        return pn9_reference(nbits)
    if kind == "pn9mid":      # PN9 sampled mid-stream
        seq = pn9_reference(nbits + 511)
        off = int(rng.integers(1, 511))
        return seq[off:off + nbits].copy()
    if kind == "ascii":       # repeating ASCII text (framedtext-style payloads)
        n = int(rng.integers(8, 40))
        txt = bytes(int(rng.integers(32, 127)) for _ in range(n))
        b = bits_from_bytes(txt * (nbits // (8 * n) + 2))
        return np.asarray(b[:nbits], dtype=np.uint8)
    return rng.integers(0, 2, nbits).astype(np.uint8)

MEMBERS = [  # (class, member-tag)
    ("PSK", "BPSK"), ("PSK", "QPSK"), ("PSK", "8PSK"),
    ("QAM", "16QAM"),
    ("FSK", "2FSK"), ("FSK", "4FSK"),
    ("OFDM", "OFDM-QPSK"), ("OFDM", "OFDM-16QAM"),
]

_POOL: dict[str, dict] = {}   # member -> {"occ": hz, "bufs": {k: [long buffers]}}


def _chan_decim(snip_fs: float, bw: float) -> int:
    """Mirror pycodec.frame.channelize's decimation choice."""
    d = 1
    while d * 2 <= 16 and snip_fs / (d * 2) >= 2.2 * bw:
        d *= 2
    return d


def _build_burst(member: str, kind: str, nbits: int, rng: np.random.Generator):
    from pycodec.frame import build_frame_iq
    from pycodec.fsk import FskSpec, build_frame_fsk_iq
    from pycodec.ofdm import OfdmSpec, build_frame_ofdm_iq
    from pycodec.psk import PskSpec

    bits = _payload_bits(kind, nbits, rng)
    if member.startswith("OFDM"):
        spec = OfdmSpec(native_fs_hz=30.72e6)
        iq = build_frame_ofdm_iq(bits, member.split("-")[1], spec, pn9_payload=False)
        occ = spec.occupied_bw_hz
    elif member.endswith("FSK"):
        spec = FskSpec(symbol_rate_hz=1.92e6, sps=8, h=0.5, bt=0.5)
        iq = build_frame_fsk_iq(bits[:min(nbits, 1024)], member, spec, pn9_payload=False)
        occ = spec.symbol_rate_hz * (1.0 + spec.h) * 1.4
    else:
        spec = PskSpec(modulation="BPSK", symbol_rate_hz=15.36e6, sps=8,
                       pulse_shape="rrc", rolloff=0.35)
        iq = build_frame_iq(bits, member, spec, pn9_payload=False)
        occ = spec.symbol_rate_hz * 1.35
    return iq.astype(np.complex64), float(occ)


def _get_pool():
    global _POOL
    if _POOL:
        return _POOL
    rng = np.random.default_rng(0xA3C0)
    for _, member in MEMBERS:
        bufs: dict[int, list[np.ndarray]] = {}
        occ = None
        for i in range(REALIZATIONS):
            kind = PAYLOAD_KINDS[i % len(PAYLOAD_KINDS)]
            nbits = PAYLOAD_BITS[(i // len(PAYLOAD_KINDS)) % len(PAYLOAD_BITS)]
            burst, occ = _build_burst(member, kind, nbits, rng)
            for k in range(K_MAX + 1):
                snip_fs = FS0 / 2 ** k
                if occ / snip_fs > 0.70:
                    continue
                d = _chan_decim(snip_fs, occ)
                need = 2 * (WIN + PAD) * d + 4096
                if k == 0:
                    buf = np.tile(burst, int(np.ceil(need / burst.size)))[:need]
                else:
                    m = int(np.ceil((need + 64) * 2 ** k / burst.size))
                    buf = resample_poly(np.tile(burst, m), 1, 2 ** k
                                        ).astype(np.complex64)[:need]
                bufs.setdefault(k, []).append(buf)
        _POOL[member] = {"occ": occ, "bufs": bufs}
    return _POOL


def _slice_signal(member: str, k: int, L: int, rng) -> tuple[np.ndarray, float]:
    e = _get_pool()[member]
    buf = e["bufs"][k][rng.integers(len(e["bufs"][k]))]
    if buf.size < L + 1:   # another member's decim geometry may need more
        buf = np.tile(buf, int(np.ceil((L + 1) / buf.size)))
    start = rng.integers(buf.size - L)
    return buf[start:start + L], e["occ"]


def _one_signal(member: str, rng: np.random.Generator) -> np.ndarray:
    e = _get_pool()[member]
    k = int(rng.choice(list(e["bufs"].keys())))
    snip_fs = FS0 / 2 ** k
    occ = e["occ"]
    bw_est = occ * rng.uniform(0.85, 1.25)
    d = _chan_decim(snip_fs, bw_est)
    L = (WIN + PAD) * d
    x, _ = _slice_signal(member, k, L, rng)
    n = np.arange(L, dtype=np.float64)
    df = rng.uniform(-CFO_FRAC_OCC, CFO_FRAC_OCC) * occ
    x = x * np.exp(1j * (2 * np.pi * df * n / snip_fs + rng.uniform(0, 2 * np.pi)))
    p_sig = float(np.mean(np.abs(x) ** 2))
    if rng.random() < NEIGHBOR_P:   # stacked-slot neighbor leaking through the filter
        om = MEMBERS[rng.integers(len(MEMBERS))][1]
        oe = _get_pool()[om]
        if k in oe["bufs"]:
            df_n = (0.5 * (occ + oe["occ"]) + rng.uniform(1e6, 6e6)) * \
                (1 if rng.random() < 0.5 else -1)
            if abs(df_n) + oe["occ"] / 2 < 0.47 * snip_fs:
                y, _ = _slice_signal(om, k, L, rng)
                g = 10 ** (rng.uniform(-3, 3) / 20.0)
                x = x + g * y * np.exp(1j * (2 * np.pi * df_n * n / snip_fs +
                                             rng.uniform(0, 2 * np.pi)))
    # burst gating: live content transmits ~57%-duty bursts with raised-cosine
    # edges (snr_staircase_4class); continuous-only training windows made the
    # on/off transitions out-of-distribution
    if rng.random() < 0.6:
        period = int(rng.uniform(0.35e-3, 1.6e-3) * snip_fs / d)
        duty = rng.uniform(0.4, 0.75)
        on_n = max(256, int(period * duty))
        env = np.zeros(L, dtype=np.float32)
        ramp = 64
        w = (0.5 - 0.5 * np.cos(np.pi * np.arange(ramp) / ramp)).astype(np.float32)
        pos = int(rng.integers(0, max(1, period)))
        while pos < L:
            e = min(pos + on_n, L)
            env[pos:e] = 1.0
            if e - pos > 2 * ramp:
                env[pos:pos + ramp] = w
                env[e - ramp:e] = w[::-1]
            pos += period
        if env.max() > 0:
            x = x * env
    snr = rng.uniform(*SNR_DB)
    noise_total = p_sig * (snip_fs / occ) / 10 ** (snr / 10.0)
    sigma = np.sqrt(noise_total / 2.0)
    x = x + sigma * (rng.standard_normal(L) + 1j * rng.standard_normal(L))
    ch, _ = channelize(x.astype(np.complex64), snip_fs, 0.0, bw_est)
    w = ch[PAD // 2:PAD // 2 + WIN]
    if w.size < WIN:
        w = np.tile(w, int(np.ceil(WIN / max(1, w.size))))[:WIN]
    return w.astype(np.complex64)


def _one_noise(rng: np.random.Generator) -> np.ndarray:
    k = int(rng.integers(0, K_MAX + 1))
    snip_fs = FS0 / 2 ** k
    bw = rng.uniform(3e6, 0.45 * snip_fs)
    d = _chan_decim(snip_fs, bw)
    L = (WIN + PAD) * d
    x = (rng.standard_normal(L) + 1j * rng.standard_normal(L)).astype(np.complex64)
    ch, _ = channelize(x, snip_fs, 0.0, bw)
    w = ch[PAD // 2:PAD // 2 + WIN]
    if w.size < WIN:
        w = np.tile(w, int(np.ceil(WIN / max(1, w.size))))[:WIN]
    return w.astype(np.complex64)


def _worker(args):
    seed, count = args
    rng = np.random.default_rng(seed)
    x = np.empty((count, WIN), dtype=np.complex64)
    y = np.empty(count, dtype=np.uint8)
    for i in range(count):
        k = rng.integers(len(CLASSES))
        y[i] = k
        if CLASSES[k] == "NOISE":
            x[i] = _one_noise(rng)
        else:
            members = [m for c, m in MEMBERS if c == CLASSES[k]]
            x[i] = _one_signal(members[rng.integers(len(members))], rng)
    return x, y


def make_dataset(n: int, seed: int = 0, workers: int | None = None):
    """-> (windows [n, WIN] complex64, labels [n] uint8), reproducible by seed."""
    workers = workers or max(1, (os.cpu_count() or 4) - 2)
    chunk = int(np.ceil(n / workers))
    jobs = [(seed * 7919 + j, min(chunk, n - j * chunk))
            for j in range(workers) if j * chunk < n]
    _get_pool()  # build bursts once in the parent; workers inherit via fork
    with mp.get_context("fork").Pool(workers) as pool:
        parts = pool.map(_worker, jobs)
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    return x, y
