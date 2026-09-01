"""Compressed-domain dataset from the ORIGINAL MATLAB waveform library.

Builds T-PRIME training windows for the codec-per-model experiment (compression
branch): 9 protocol classes + noise, windows synthesized to look like STORED
SNIPS (decimated to the snipper's keep-bw rate, AWGN at a drawn in-band SNR,
then codec-quantized), and featurized in the COMPRESSED DOMAIN.

Unified input representation (all codecs, so models are architecturally
identical and only mantissa precision differs):

    per 64-complex block:  mantissa-normalized I/Q  in [-1, 1]
                           + the block's exponent, window-relative
    per 128-complex token: 256 interleaved mantissa reals + 2 exponents = 258

  - bfp8/bfp12: the stored ints ARE the mantissas; exponents are the stored
    int8s. No dequantization anywhere (true compressed-domain).
  - sc16: reconstruct (int16 x scale is exact), then the same block transform;
    effective mantissa precision ~15 bits after block renorm.
  - none: float "mantissas" (infinite precision) — the control.

Window geometry: WIN = 8192 complex = 128 BFP blocks = 64 tokens x 128 complex
(one token = exactly 2 blocks). Snip payload block grids align because the
CUDA encoder also blocks from sample 0 of each snippet.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

LIB_ROOT = os.path.expanduser(
    "~/Documents/holoscan_waveform_generation/generated_waveforms_24576")

CLASSES10 = ["BPSK", "QPSK", "16QAM", "OFDM", "5G_Downlink", "802_11ax",
             "Bluetooth", "Narrowband_FM", "Broadband_FM", "NOISE"]
WIN = 8192          # complex samples per window
BLOCK = 64          # complex samples per BFP block (matches snippet_compression.cu)
TOK = 128           # complex samples per T-PRIME token (= 2 blocks)
FEATS = 2 * TOK + TOK // BLOCK   # 258 features per token
FS = 245.76e6
OVERSAMPLE = 1.25   # snipper keep_bw = occ * (1 + oversample_percent/100)


# --------------------------------------------------------------- library ----

def snip_decim(occ_hz: float) -> int:
    """Mirror signal_snip_core.cu: decim = floor(fs / keep_bw), min 1."""
    keep = max(occ_hz * OVERSAMPLE, 1.0)
    return max(1, int(np.floor(FS / keep)))


def scan_library(root: str = LIB_ROOT) -> dict[str, list[dict]]:
    """class -> [{iq (decimated, complex64), rate, occ, name}]. Loads + decimates
    every record once (whole library ~1 GB raw; decimated much smaller)."""
    import scipy.io as sio
    from scipy.signal import resample_poly
    lib: dict[str, list[dict]] = {}
    for cls in CLASSES10[:-1]:
        recs = []
        for mp in sorted(glob.glob(os.path.join(root, cls, "*.mat"))):
            j = json.load(open(mp.replace(".mat", ".json")))
            occ = float(j.get("designedOccupiedBandwidthHz") or
                        j.get("requestedOccupiedBandwidthHz") or FS / 2)
            m = sio.loadmat(mp, squeeze_me=True)
            iq = np.asarray(m["f_sig"], dtype=np.complex64)
            d = snip_decim(occ)
            if d > 1:
                iq = resample_poly(iq, 1, d).astype(np.complex64)
            recs.append(dict(iq=iq, rate=FS / d, occ=occ,
                             name=j.get("waveformName", os.path.basename(mp))))
        if not recs:
            raise FileNotFoundError(f"no records for class {cls} under {root}")
        lib[cls] = recs
    return lib


# ------------------------------------------------------- window synthesis ----

def _tile_to(x: np.ndarray, n: int) -> np.ndarray:
    if x.size >= n:
        return x[:n]
    reps = int(np.ceil(n / x.size))
    return np.tile(x, reps)[:n]


def make_windows(lib: dict, n_per_class: int, seed: int,
                 snr_db: tuple[float, float] = (-10.0, 30.0)):
    """-> (windows complex64 (N, WIN), labels int64, snrs float32).
    Each window: random record slice (tiled if short), random phase + tiny CFO,
    AWGN at the drawn in-band SNR (noise sized against the signal's occupied
    fraction of the snip rate). NOISE class: AWGN only."""
    rng = np.random.default_rng(seed)
    xs, ys, ss = [], [], []
    for ci, cls in enumerate(CLASSES10):
        for _ in range(n_per_class):
            snr = float(rng.uniform(*snr_db))
            if cls == "NOISE":
                w = (rng.standard_normal(WIN) + 1j * rng.standard_normal(WIN)) \
                    .astype(np.complex64)
            else:
                rec = lib[cls][rng.integers(len(lib[cls]))]
                iq = rec["iq"]
                if iq.size <= WIN:
                    seg = _tile_to(iq, WIN).copy()
                else:
                    s0 = int(rng.integers(0, iq.size - WIN))
                    seg = iq[s0:s0 + WIN].copy()
                # phase + tiny CFO jitter (front-end realism; lesson from rounds 1-6)
                cfo = rng.uniform(-2e-5, 2e-5)
                ph = rng.uniform(0, 2 * np.pi)
                seg *= np.exp(1j * (ph + 2 * np.pi * cfo * np.arange(WIN))) \
                    .astype(np.complex64)
                p_sig = float(np.mean(np.abs(seg) ** 2))
                if p_sig <= 0:
                    continue
                occ_frac = min(1.0, rec["occ"] / rec["rate"])
                # in-band SNR: noise power falling inside the occupied band
                sigma2 = p_sig / (10 ** (snr / 10.0)) / occ_frac
                noise = np.sqrt(sigma2 / 2) * (rng.standard_normal(WIN) +
                                               1j * rng.standard_normal(WIN))
                w = (seg + noise).astype(np.complex64)
            xs.append(w)
            ys.append(ci)
            ss.append(snr)
    x = np.stack(xs)
    y = np.asarray(ys, np.int64)
    s = np.asarray(ss, np.float32)
    perm = rng.permutation(len(y))
    return x[perm], y[perm], s[perm]


# ------------------------------------------------- compressed-domain feats ----

def _block_stats(w: np.ndarray):
    """w (N, WIN) complex -> per-block max|scalar| (N, nblocks)."""
    n, win = w.shape
    nb = win // BLOCK
    r = w.reshape(n, nb, BLOCK)
    return np.maximum(np.abs(r.real).max(axis=2), np.abs(r.imag).max(axis=2))


def quantize_windows(w: np.ndarray, codec: str):
    """Apply the codec to float windows -> (mant complex float in [-1,1],
    e_rel float per block). Mirrors snippet_compression.cu exactly for bfp."""
    n, win = w.shape
    nb = win // BLOCK
    if codec == "sc16":
        peak = np.abs(np.stack([w.real, w.imag])).max(axis=(0, 2), keepdims=False)
        scale = np.where(peak > 0, peak / 32767.0, 1.0)[:, None]
        q = np.clip(np.rint(w.real / scale), -32767, 32767) + \
            1j * np.clip(np.rint(w.imag / scale), -32767, 32767)
        w = (q * scale).astype(np.complex64)   # exact reconstruction
        codec = "none"                          # then the float block transform
    if codec == "none":
        bm = _block_stats(w)
        e = np.where(bm > 0, np.ceil(np.log2(np.maximum(bm, 1e-30))), -127.0)
        scale = np.exp2(e)[:, :, None]
        mant = w.reshape(n, nb, BLOCK) / scale                  # in [-1, 1]
        e_rel = e - e.max(axis=1, keepdims=True)
        return mant.reshape(n, win).astype(np.complex64), e_rel.astype(np.float32)
    if codec in ("bfp8", "bfp12"):
        mant_bits = 12 if codec == "bfp12" else 8
        qmax = float((1 << (mant_bits - 1)) - 1)
        bm = _block_stats(w)
        with np.errstate(divide="ignore"):
            e = np.ceil(np.log2(np.maximum(bm, 1e-30) / qmax))
        e = np.clip(np.where(bm > 0, e, -127.0), -127, 127)
        scale = np.exp2(e)[:, :, None]
        r = w.reshape(n, nb, BLOCK)
        mi = np.clip(np.rint(r.real / scale), -qmax, qmax)
        mq = np.clip(np.rint(r.imag / scale), -qmax, qmax)
        mant = (mi + 1j * mq) / qmax                            # in [-1, 1]
        e_rel = e - e.max(axis=1, keepdims=True)
        return mant.reshape(n, win).astype(np.complex64), e_rel.astype(np.float32)
    raise ValueError(f"unknown codec {codec}")


def featurize(mant: np.ndarray, e_rel: np.ndarray) -> np.ndarray:
    """(mant (N, WIN) complex, e_rel (N, nblocks)) -> (N, 64, 258) float32.
    Token = 128 complex: 256 interleaved I/Q mantissas + its 2 block exponents/16."""
    n, win = mant.shape
    seq = win // TOK
    t = mant.reshape(n, seq, TOK)
    feat = np.empty((n, seq, FEATS), np.float32)
    feat[:, :, 0:2 * TOK:2] = t.real
    feat[:, :, 1:2 * TOK:2] = t.imag
    feat[:, :, 2 * TOK:] = e_rel.reshape(n, seq, TOK // BLOCK) / 16.0
    return feat


# --------------------------------------------- stored-snip payload readers ----

def payload_to_mant(codec: str, raw: np.ndarray, n_iq: int, ann: dict):
    """Stored snip payload bytes -> (mant complex in [-1,1], e_rel per block),
    WITHOUT dequantization for bfp (the ints are used as-is). For sc16/cf32 the
    float block transform is applied to the (exact) reconstruction. Windows must
    be sliced on BLOCK boundaries afterward to stay on the stored block grid."""
    if codec in ("bfp8", "bfp12"):
        B = int(ann.get("wfgt:comp_block", BLOCK))
        mant_bits = 12 if codec == "bfp12" else 8
        qmax = float((1 << (mant_bits - 1)) - 1)
        bb = 1 + (2 * B * mant_bits) // 8
        nblocks = raw.size // bb
        m = raw[: nblocks * bb].reshape(nblocks, bb)
        e = m[:, 0].view(np.int8).astype(np.float32)
        if codec == "bfp8":
            scal = m[:, 1:].view(np.int8).astype(np.float32)
        else:
            p = m[:, 1:].reshape(nblocks, B, 3).astype(np.uint16)
            m0 = ((((p[..., 0].astype(np.int32) << 4) | (p[..., 1] >> 4)) ^ 0x800) - 0x800)
            m1 = (((((p[..., 1] & 0xF).astype(np.int32) << 8) | p[..., 2]) ^ 0x800) - 0x800)
            pair = np.stack([m0, m1], axis=-1)
            scal = pair.reshape(nblocks, 2 * B).astype(np.float32)
        mant = (scal[:, 0::2] + 1j * scal[:, 1::2]).astype(np.complex64) / qmax
        mant = mant.reshape(-1)[:n_iq]
        e_rel = e - e.max() if e.size else e
        return mant, e_rel
    # sc16 / cf32_le: exact float reconstruction, then the shared block transform
    if codec == "sc16":
        scale = float(ann.get("wfgt:comp_scale", 1.0))
        x = raw[: n_iq * 4].view("<i2").astype(np.float32) * scale
        iq = (x[0::2] + 1j * x[1::2]).astype(np.complex64)
    else:
        iq = raw[: n_iq * 8].view(np.complex64).copy()
    nb = iq.size // BLOCK
    iq = iq[: nb * BLOCK]
    m2, e2 = quantize_windows(iq[None, :], "none")
    return m2[0], e2[0]
