"""Baseband modulators + the rate-agnostic placement primitives used by gen_scenes.py.

Every modulator returns (x, r0, bw_hz): a unit-rms complex64 baseband waveform x at its own native rate r0,
with (approximate 99 %) occupied bandwidth bw_hz.  `Placer` then puts such a waveform on a canvas at an
arbitrary X410 rate R:

  * interior placement (|fc| + bw/2 well inside +/-R/2): rational resample r0 -> R, mix to fc, add.
  * band-edge / wider-than-band placement: resample r0 -> 2R, mix at 2R (alias-free), then an anti-alias
    decimate-by-2 back to R.  The decimation low-pass IS the receiver's DDC: whatever part of the emitter
    lies beyond +/-R/2 is removed, so signals cut off by the band edge look the way the radio shows them.

Signal level is specified as PSD SNR ("spectrogram SNR"): the signal's in-band power spectral density over
the noise PSD (noise power 1.0 per sample -> PSD 1/R).  That makes a 10 dB signal look 10 dB above the
floor at ANY sample rate and bandwidth, which is what the segmenter sees.  The older gen_491 convention
(total-power SNR, wfgt:power_db) is recorded alongside for compatibility.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
import math
import numpy as np
from scipy.signal import resample_poly, fftconvolve, firwin

C64 = np.complex64
TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- helpers
def rms_norm(x: np.ndarray) -> np.ndarray:
    p = math.sqrt(float(np.mean(np.abs(x) ** 2))) + 1e-12
    return (x / p).astype(C64)


def rrc_taps(beta: float, span: int, sps: int) -> np.ndarray:
    n = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    h = np.zeros_like(n)
    for i, t in enumerate(n):
        if abs(t) < 1e-9:
            h[i] = 1 - beta + 4 * beta / math.pi
        elif beta > 0 and abs(abs(4 * beta * t) - 1.0) < 1e-9:
            h[i] = (beta / math.sqrt(2)) * ((1 + 2 / math.pi) * math.sin(math.pi / (4 * beta)) +
                                            (1 - 2 / math.pi) * math.cos(math.pi / (4 * beta)))
        else:
            h[i] = (math.sin(math.pi * t * (1 - beta)) + 4 * beta * t * math.cos(math.pi * t * (1 + beta))) / \
                   (math.pi * t * (1 - (4 * beta * t) ** 2))
    return h / math.sqrt(np.sum(h ** 2))


def _qam_points(M: int) -> np.ndarray:
    if M == 2:
        return np.array([-1, 1], dtype=np.complex128)
    if M == 8:
        return np.exp(1j * (np.arange(8) * math.pi / 4 + math.pi / 8))
    if M == 4:
        return np.exp(1j * (np.arange(4) * math.pi / 2 + math.pi / 4))
    k = int(round(math.sqrt(M)))
    ax = np.arange(k) * 2 - (k - 1)
    pts = (ax[:, None] + 1j * ax[None, :]).ravel()
    return pts / math.sqrt(np.mean(np.abs(pts) ** 2))


# --------------------------------------------------------------------------- modulators
def mod_linear(rng, M: int, rs_hz: float, dur_s: float, beta: float = 0.35, sps: int = 4,
               oqpsk: bool = False):
    """PSK/QAM with RRC pulse shaping. Returns (x, r0, bw)."""
    n_sym = max(16, int(math.ceil(dur_s * rs_hz)) + 10)
    pts = _qam_points(M)
    syms = pts[rng.integers(0, M, n_sym)]
    up = np.zeros(n_sym * sps, dtype=np.complex128)
    up[::sps] = syms
    if oqpsk:
        q = np.zeros_like(up); q[sps // 2:] = up.imag[: -sps // 2]
        up = up.real + 1j * q.real
    h = rrc_taps(beta, 10, sps)
    x = fftconvolve(up, h, mode="same")
    return rms_norm(x), rs_hz * sps, rs_hz * (1.0 + beta)


def mod_gfsk(rng, rs_hz: float, dur_s: float, h: float = 0.5, bt: float = 0.5, M: int = 2, sps: int = 8):
    """Gaussian FSK (BT classic h=0.32 BT=0.5; BLE h=0.5; generic 2/4-FSK). Returns (x, r0, bw)."""
    n_sym = max(16, int(math.ceil(dur_s * rs_hz)) + 4)
    if M == 2:
        lv = rng.integers(0, 2, n_sym) * 2.0 - 1.0
    else:
        lv = rng.integers(0, 4, n_sym) * 2.0 - 3.0
    imp = np.repeat(lv, sps).astype(np.float64)
    if bt > 0:
        span = 4
        t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
        sig = math.sqrt(math.log(2)) / (TWO_PI * bt)
        g = np.exp(-0.5 * (t / sig) ** 2); g /= g.sum()
        imp = fftconvolve(imp, g, mode="same")
    phase = np.cumsum(imp) * (math.pi * h / sps)
    x = np.exp(1j * phase)
    dev = h * rs_hz / 2.0 * (M - 1)
    bw = 2.0 * (dev + rs_hz / 2.0)              # Carson
    return rms_norm(x), rs_hz * sps, bw


def mod_ofdm(rng, nfft: int, n_used: int, scs_hz: float, cp_frac: float, dur_s: float, mod_M: int = 4,
             load: float = 1.0, ctrl_symbols_per_slot: int = 0, slot_symbols: int = 14,
             center_stripe_sc: int = 0, stripe_every_slots: int = 0, pilot_boost_db: float = 0.0,
             p_idle: float = 0.0, load_jitter: float = 0.0):
    """Generic CP-OFDM (WiFi / LTE / NR / DVB flavours).

    load < 1 emulates a lightly loaded cellular downlink: each slot picks a random contiguous RB-group
    allocation covering `load` of the used subcarriers for its data symbols, while the first
    `ctrl_symbols_per_slot` symbols of every slot always span the full carrier (CRS/PDCCH) and an always-on
    `center_stripe_sc`-wide centre block (PSS/SSS/PBCH) appears every `stripe_every_slots` slots.
    Returns (x, r0 = nfft*scs, bw = n_used*scs)."""
    r0 = nfft * scs_hz
    cp = max(1, int(round(cp_frac * nfft)))
    sym_len = nfft + cp
    n_sym = max(2, int(math.ceil(dur_s * r0 / sym_len)) + 1)
    pts = _qam_points(mod_M)
    half = n_used // 2
    used = np.concatenate([np.arange(-half, 0), np.arange(1, n_used - half + 1)])  # skip DC
    bins = used % nfft
    X = np.zeros((n_sym, nfft), dtype=np.complex128)
    data = pts[rng.integers(0, mod_M, (n_sym, n_used))]
    if load >= 0.999 and ctrl_symbols_per_slot == 0 and p_idle <= 0 and load_jitter <= 0:
        X[:, bins] = data
    else:
        mask = np.zeros((n_sym, n_used), dtype=bool)
        for s0 in range(0, n_sym, slot_symbols):
            s1 = min(n_sym, s0 + slot_symbols)
            mask[s0:s0 + ctrl_symbols_per_slot, :] = True
            # per-slot traffic: idle slots (control symbols only) and jittered load -> time-varying occupancy
            if p_idle > 0 and rng.random() < p_idle:
                l_s = 0.0
            else:
                l_s = float(min(1.0, max(0.05, load + (rng.normal(0.0, load_jitter) if load_jitter > 0 else 0.0))))
            if l_s > 0:
                w = max(1, int(round(l_s * n_used)))
                # 1-3 allocation chunks
                nchunk = int(rng.integers(1, 4))
                for _ in range(nchunk):
                    cw = max(1, w // nchunk)
                    c0 = int(rng.integers(0, max(1, n_used - cw + 1)))
                    mask[s0 + ctrl_symbols_per_slot:s1, c0:c0 + cw] = True
            if center_stripe_sc > 0 and stripe_every_slots > 0 and ((s0 // slot_symbols) % stripe_every_slots == 0):
                c0 = max(0, n_used // 2 - center_stripe_sc // 2)
                mask[s0:min(s1, s0 + 4), c0:c0 + center_stripe_sc] = True
        X[:, bins] = data * mask
    if pilot_boost_db > 0:
        X[:, bins[::6]] *= 10 ** (pilot_boost_db / 20)
    x = np.fft.ifft(X, axis=1) * math.sqrt(nfft)
    x = np.concatenate([x[:, -cp:], x], axis=1).ravel()
    return rms_norm(x), r0, n_used * scs_hz


def mod_chirp(rng, bw_hz: float, t_sym_s: float, dur_s: float, up_frac: float = 1.0, osf: float = 2.0,
              sync_down: int = 0):
    """Linear chirps (LoRa-like repeating up-chirps with optional down-chirp sync, or FMCW sawtooth).
    Returns (x, r0 = osf*bw, bw)."""
    r0 = osf * bw_hz
    n = max(8, int(round(t_sym_s * r0)))
    t = np.arange(n) / r0
    k = bw_hz / t_sym_s
    up = np.exp(1j * TWO_PI * (-bw_hz / 2 * t + 0.5 * k * t * t))
    down = np.conj(up)
    n_sym = max(1, int(math.ceil(dur_s / t_sym_s)) + 1)
    parts = []
    for i in range(n_sym):
        if i < sync_down:
            parts.append(down)
        elif rng.random() < up_frac:
            # LoRa data symbols = cyclically shifted up-chirps
            sh = int(rng.integers(0, n))
            parts.append(np.roll(up, sh))
        else:
            parts.append(down)
    return rms_norm(np.concatenate(parts)), r0, bw_hz


def mod_fm(rng, dev_hz: float, msg_bw_hz: float, dur_s: float, osf: float = 1.5):
    """Analog FM of band-limited noise (voice-like NBFM or wideband/video FM). Carson bw = 2(dev + msg_bw)."""
    bw = 2.0 * (dev_hz + msg_bw_hz)
    r0 = max(osf * bw, 8.0 * msg_bw_hz)
    n = max(64, int(math.ceil(dur_s * r0)) + 64)
    m = rng.standard_normal(n)
    taps = firwin(129, min(0.99, msg_bw_hz / (r0 / 2)))
    m = fftconvolve(m, taps, mode="same")
    m /= (np.std(m) + 1e-12)
    phase = np.cumsum(m) * (TWO_PI * dev_hz / r0)
    return rms_norm(np.exp(1j * phase)), r0, bw


def mod_noise_band(rng, bw_hz: float, dur_s: float, osf: float = 1.5):
    """Band-limited complex noise (jammer / wideband interference / microwave-oven-like)."""
    r0 = osf * bw_hz
    n = max(256, int(math.ceil(dur_s * r0)) + 256)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    taps = firwin(257, min(0.99, bw_hz / r0))
    x = fftconvolve(x, taps, mode="same")
    return rms_norm(x), r0, bw_hz


def mod_cw(rng, dur_s: float, r0: float = 1.0e6, drift_hz: float = 0.0):
    n = max(16, int(math.ceil(dur_s * r0)) + 16)
    t = np.arange(n) / r0
    x = np.exp(1j * (TWO_PI * (drift_hz / (2 * max(dur_s, 1e-9))) * t * t))
    return rms_norm(x), r0, max(1.0, abs(drift_hz))


# --------------------------------------------------------------------------- impairments
def apply_ramp(x: np.ndarray, n_ramp: int) -> np.ndarray:
    n_ramp = int(min(max(n_ramp, 2), x.size // 4))
    if n_ramp < 2:
        return x
    w = 0.5 - 0.5 * np.cos(np.pi * np.arange(n_ramp) / n_ramp)
    x = x.copy()
    x[:n_ramp] *= w.astype(np.float32)
    x[-n_ramp:] *= w[::-1].astype(np.float32)
    return x


def apply_fading(rng, x: np.ndarray, rate: float, sigma_db: float, tcorr_s: float) -> np.ndarray:
    """Slow log-normal level variation (mobile emitter), AR(1) at 2 kHz then interpolated."""
    fs_env = 2000.0
    n_env = max(4, int(x.size / rate * fs_env) + 2)
    a = math.exp(-1.0 / max(1.0, tcorr_s * fs_env))
    e = np.empty(n_env); e[0] = rng.standard_normal()
    for i in range(1, n_env):
        e[i] = a * e[i - 1] + math.sqrt(1 - a * a) * rng.standard_normal()
    g_db = sigma_db * e
    g = 10 ** (g_db / 20)
    env = np.interp(np.arange(x.size) * (fs_env / rate), np.arange(n_env), g).astype(np.float32)
    return (x * env).astype(C64)


def apply_multipath(rng, x: np.ndarray, r0: float, tau_max_s: float, n_taps: int | None = None) -> np.ndarray:
    """Static frequency-selective channel: 2-6 Rayleigh taps with exponentially decaying power over
    [0, tau_max] -> notches / ripple across a wideband emitter (static over one scene; the slow fading
    envelope handles level changes). Power-normalised, rms re-normalised."""
    n_taps = int(n_taps or rng.integers(2, 7))
    d = np.sort(np.r_[0.0, rng.uniform(0.0, tau_max_s, n_taps - 1)])
    g = (rng.standard_normal(n_taps) + 1j * rng.standard_normal(n_taps)) * np.exp(-d / max(tau_max_s / 2.0, 1e-12))
    g[0] = abs(g[0]) + 0.3                     # a dominant first arrival keeps the notch depth realistic
    L = int(math.ceil(tau_max_s * r0)) + 1
    h = np.zeros(max(L, 1), dtype=np.complex128)
    for di, gi in zip(d, g):
        h[min(int(round(di * r0)), L - 1)] += gi
    if h.size <= 1:
        return x
    h /= math.sqrt(np.sum(np.abs(h) ** 2))
    y = fftconvolve(x, h, mode="full")[: x.size]
    return rms_norm(y)


def apply_pa(x: np.ndarray, backoff_db: float, p: float = 2.0) -> np.ndarray:
    """Rapp-model PA compression: input backoff relative to the rms -> spectral regrowth shoulders
    (~ -30..-45 dBc for 2-8 dB backoff on OFDM). rms re-normalised."""
    A = math.sqrt(float(np.mean(np.abs(x) ** 2))) * 10 ** (backoff_db / 20.0)
    r = np.abs(x) / max(A, 1e-12)
    y = x / (1.0 + r ** (2 * p)) ** (1.0 / (2 * p))
    return rms_norm(y)


def apply_phase_noise(rng, x: np.ndarray, rate: float, linewidth_hz: float) -> np.ndarray:
    """Wiener phase noise with the given Lorentzian linewidth (chunked for memory)."""
    out = np.empty_like(x)
    step = math.sqrt(TWO_PI * linewidth_hz / rate)
    phi0 = 0.0
    B = 1 << 22
    for i in range(0, x.size, B):
        n = min(B, x.size - i)
        dphi = rng.standard_normal(n) * step
        phi = phi0 + np.cumsum(dphi)
        phi0 = float(phi[-1])
        out[i:i + n] = x[i:i + n] * np.exp(1j * phi).astype(C64)
    return out


def mix_inplace(y: np.ndarray, f_norm: float, phi0: float = 0.0, n0: int = 0) -> None:
    """y *= exp(j(2 pi f_norm n + phi0)) with float64 phase, in 4M-sample chunks (complex64 result)."""
    if abs(f_norm) < 1e-15 and phi0 == 0.0:
        return
    B = 1 << 22
    for i in range(0, y.size, B):
        n = np.arange(n0 + i, n0 + i + min(B, y.size - i), dtype=np.float64)
        ph = TWO_PI * f_norm * n + phi0
        ph = np.mod(ph, TWO_PI)
        y[i:i + n.size] *= np.exp(1j * ph).astype(C64)


def rational(ratio: float, max_den: int = 4096) -> tuple[int, int]:
    fr = Fraction(ratio).limit_denominator(max_den)
    return fr.numerator, fr.denominator


def resample_to(x: np.ndarray, r0: float, r1: float) -> np.ndarray:
    """Rational resample r0 -> r1 in two stages so the anti-alias FIR stays small: an integer stage
    (up or down by floor(ratio)) followed by a near-unity fraction (limit_denominator 1024 -> <= 20k taps).
    A single-stage Fraction(r1/r0) for an arbitrary native rate can demand 10^7:1 terms, and scipy then
    designs a 10^8-tap Kaiser FIR (seconds per call)."""
    ratio = r1 / r0
    if abs(ratio - 1.0) < 1e-9:
        return x.astype(C64, copy=False)
    y = x
    if ratio >= 2.0:
        k = int(math.floor(ratio))
        y = resample_poly(y, k, 1)
        ratio /= k
    elif ratio <= 0.5:
        k = int(math.floor(1.0 / ratio))
        y = resample_poly(y, 1, k)
        ratio *= k
    if abs(ratio - 1.0) > 1e-7:
        up, dn = rational(ratio, 1024)
        y = resample_poly(y, up, dn)
    return y.astype(C64)


# --------------------------------------------------------------------------- placement
@dataclass
class Burst:
    """One emission the GT will box: content placed at (fc, t0, dur) with a level."""
    label: str                 # class label (BPSK, OFDM, 802_11ax, Bluetooth, LTE, NR, ...)
    emitter: str               # behaviour tag (wifi, bt, ble, zigbee, lte_dl, nr_dl, drone_video, hopper, ...)
    emitter_id: int
    fc_hz: float               # centre vs canvas centre (baseband)
    t0_s: float
    dur_s: float
    bw_hz: float
    snr_psd_db: float
    scenario: str = ""
    # filled by the placer
    sample_start: int = 0
    sample_count: int = 0
    total_snr_db: float = 0.0


class Placer:
    """Adds unit-rms native-rate content to a complex64 canvas at rate R (noise power 1 -> PSD 1/R)."""

    def __init__(self, canvas: np.ndarray, rate_hz: float, rng: np.random.Generator):
        self.c = canvas; self.R = float(rate_hz); self.rng = rng
        self.T = canvas.size

    # level: PSD SNR -> amplitude for unit-rms content of in-band bandwidth bw
    def amp_for(self, snr_psd_db: float, bw_hz: float, wide_bin_hz: float) -> tuple[float, float]:
        b_eff = min(max(bw_hz, wide_bin_hz), self.R)
        total_snr = snr_psd_db - 10.0 * math.log10(self.R / b_eff)
        return 10.0 ** (total_snr / 20.0), total_snr

    def _impair(self, seg: np.ndarray, r0: float, multipath, pa_db) -> np.ndarray:
        if multipath:
            seg = apply_multipath(self.rng, seg, r0, float(multipath[0]), multipath[1] if len(multipath) > 1 else None)
        if pa_db is not None and pa_db is not False:
            seg = apply_pa(seg, float(pa_db))
        return seg

    def _prep(self, x: np.ndarray, r0: float, fc: float, bw: float, dur_s: float, ramp_s: float,
              interior: bool, phi0: float, multipath=None, pa_db=None) -> np.ndarray:
        """native content -> canvas-rate complex64 segment of ~dur_s at fc (no level applied)."""
        target = self.R if interior else 2.0 * self.R
        # crop/tile the native content to dur_s (+ a little) BEFORE resampling (cost ~ duration)
        need = int(math.ceil(dur_s * r0)) + 8
        if x.size < need:
            x = np.tile(x, int(math.ceil(need / x.size)))
        s0 = int(self.rng.integers(0, max(1, x.size - need)))
        seg = self._impair(x[s0:s0 + need], r0, multipath, pa_db)
        seg = apply_ramp(seg, int(ramp_s * r0))
        y = resample_to(seg, r0, target)
        mix_inplace(y, fc / target, phi0)
        if not interior:
            y = resample_poly(y, 1, 2).astype(C64)       # DDC anti-alias: drops the out-of-band part
        n = int(round(dur_s * self.R))
        return y[:n]

    def is_interior(self, fc: float, bw: float) -> bool:
        return (abs(fc) + bw / 2.0) <= 0.49 * self.R and bw <= 0.98 * self.R

    def place(self, b: Burst, x: np.ndarray, r0: float, wide_bin_hz: float, ramp_s: float = 2e-6,
              fading: tuple | None = None, phase_noise_hz: float = 0.0, multipath=None, pa_db=None) -> bool:
        """Render burst b with content (x, r0) onto the canvas. Returns False if nothing landed in [0,T)."""
        t_end = b.t0_s + b.dur_s
        if t_end <= 0 or b.t0_s * self.R >= self.T:
            return False
        # clip the burst to the canvas in time (keeps content realistic: a cut burst has no ramp at the cut)
        t0 = max(0.0, b.t0_s); t1 = min(t_end, self.T / self.R)
        dur = t1 - t0
        if dur <= 0:
            return False
        interior = self.is_interior(b.fc_hz, b.bw_hz)
        if not interior and b.bw_hz >= self.R / 2.0:
            # keep the mixed signal alias-free at 2R: content beyond 1.5R wraps back inside the canvas
            lim = max(0.0, self.R - b.bw_hz / 2.0)
            b.fc_hz = float(np.clip(b.fc_hz, -lim, lim))
        phi0 = float(self.rng.uniform(0, TWO_PI))
        y = self._prep(x, r0, b.fc_hz, b.bw_hz, dur, ramp_s, interior, phi0, multipath, pa_db)
        if fading is not None:
            y = apply_fading(self.rng, y, self.R, *fading)
        if phase_noise_hz > 0:
            y = apply_phase_noise(self.rng, y, self.R, phase_noise_hz)
        amp, total = self.amp_for(b.snr_psd_db, b.bw_hz, wide_bin_hz)
        i0 = int(round(t0 * self.R)); n = min(y.size, self.T - i0)
        if n <= 0:
            return False
        self.c[i0:i0 + n] += (y[:n] * np.float32(amp))
        b.sample_start = i0; b.sample_count = n; b.total_snr_db = total
        return True

    def place_train(self, bursts: list[Burst], x: np.ndarray, r0: float, wide_bin_hz: float,
                    ramp_s: float = 2e-6, fading: tuple | None = None, multipath=None, pa_db=None) -> list[Burst]:
        """Many bursts sharing one centre frequency (a packet train): resample the content ONCE."""
        if not bursts:
            return []
        fc, bw = bursts[0].fc_hz, bursts[0].bw_hz
        interior = self.is_interior(fc, bw)
        if not interior and bw >= self.R / 2.0:
            lim = max(0.0, self.R - bw / 2.0); fc = float(np.clip(fc, -lim, lim))
            for b in bursts: b.fc_hz = fc
        target = self.R if interior else 2.0 * self.R
        on_s = sum(max(0.0, min(b.t0_s + b.dur_s, self.T / self.R) - max(0.0, b.t0_s)) for b in bursts)
        need = int(math.ceil((on_s + 1e-4) * r0)) + 64
        if x.size < need:
            x = np.tile(x, int(math.ceil(need / x.size)))
        s0 = int(self.rng.integers(0, max(1, x.size - need)))
        y_all = resample_to(self._impair(x[s0:s0 + need], r0, multipath, pa_db), r0, target)
        env = None
        if fading is not None:
            env_sig, env_t = fading
        out, pos = [], 0
        for b in bursts:
            t_end = b.t0_s + b.dur_s
            if t_end <= 0 or b.t0_s * self.R >= self.T:
                continue
            t0 = max(0.0, b.t0_s); t1 = min(t_end, self.T / self.R); dur = t1 - t0
            n_t = int(round(dur * target))
            if n_t <= 8 or pos + n_t > y_all.size:
                if pos + n_t > y_all.size:
                    pos = 0
                if n_t <= 8:
                    continue
            seg = y_all[pos:pos + n_t].copy(); pos += n_t
            seg = apply_ramp(seg, int(ramp_s * target))
            i0 = int(round(t0 * self.R))
            mix_inplace(seg, fc / target, float(self.rng.uniform(0, TWO_PI)), n0=int(round(t0 * target)))
            if not interior:
                seg = resample_poly(seg, 1, 2).astype(C64)
            if fading is not None:
                seg = apply_fading(self.rng, seg, self.R, env_sig, env_t)
            amp, total = self.amp_for(b.snr_psd_db, b.bw_hz, wide_bin_hz)
            n = min(seg.size, self.T - i0)
            if n <= 0:
                continue
            self.c[i0:i0 + n] += seg[:n] * np.float32(amp)
            b.sample_start = i0; b.sample_count = n; b.total_snr_db = total
            out.append(b)
        return out

    def place_hops(self, bursts: list[Burst], x: np.ndarray, r0: float, wide_bin_hz: float,
                   ramp_s: float = 2e-6, multipath=None, pa_db=None) -> list[Burst]:
        """A frequency hopper: bursts with individual centres. Content resampled once (interior hops);
        hops that straddle the band edge go through the 2R path individually."""
        if not bursts:
            return []
        on_s = sum(max(0.0, min(b.t0_s + b.dur_s, self.T / self.R) - max(0.0, b.t0_s)) for b in bursts)
        need = int(math.ceil((on_s + 1e-4) * r0)) + 64
        if x.size < need:
            x = np.tile(x, int(math.ceil(need / x.size)))
        s0 = int(self.rng.integers(0, max(1, x.size - need)))
        x_imp = self._impair(x[s0:s0 + need], r0, multipath, pa_db)
        y_all = resample_to(x_imp, r0, self.R)
        out, pos = [], 0
        for b in bursts:
            t_end = b.t0_s + b.dur_s
            if t_end <= 0 or b.t0_s * self.R >= self.T:
                continue
            t0 = max(0.0, b.t0_s); t1 = min(t_end, self.T / self.R); dur = t1 - t0
            n_t = int(round(dur * self.R))
            if n_t <= 8:
                continue
            i0 = int(round(t0 * self.R))
            if self.is_interior(b.fc_hz, b.bw_hz):
                if pos + n_t > y_all.size:
                    pos = 0
                seg = y_all[pos:pos + n_t].copy(); pos += n_t
                seg = apply_ramp(seg, int(ramp_s * self.R))
                mix_inplace(seg, b.fc_hz / self.R, float(self.rng.uniform(0, TWO_PI)), n0=i0)
            else:
                if abs(b.fc_hz) - b.bw_hz / 2.0 >= self.R / 2.0:
                    continue                                   # entirely outside the band
                seg = self._prep(x, r0, b.fc_hz, b.bw_hz, dur, ramp_s, False, float(self.rng.uniform(0, TWO_PI)), multipath, pa_db)
            amp, total = self.amp_for(b.snr_psd_db, b.bw_hz, wide_bin_hz)
            n = min(seg.size, self.T - i0)
            if n <= 0:
                continue
            self.c[i0:i0 + n] += seg[:n] * np.float32(amp)
            b.sample_start = i0; b.sample_count = n; b.total_snr_db = total
            out.append(b)
        return out


# --------------------------------------------------------------------------- receiver model
def receiver_response_taps(profile_db_1024: np.ndarray, n_taps: int = 513) -> np.ndarray:
    """Linear-phase FIR whose magnitude follows a (dB, 1024-bin, fftshifted) floor profile."""
    mag = 10.0 ** (np.asarray(profile_db_1024, dtype=np.float64) / 20.0)
    # profile is over [-R/2, R/2); build the complex frequency response on an n-point grid
    n = 2048
    xf = np.linspace(-0.5, 0.5, mag.size, endpoint=False)
    grid = (np.arange(n) / n) - 0.5
    H = np.interp(grid, xf, mag)
    H = np.fft.ifftshift(H)                       # to natural FFT order
    h = np.fft.ifft(H)                            # zero-phase, circular
    h = np.fft.fftshift(h)
    c = n // 2; half = n_taps // 2
    h = h[c - half: c + half + 1]
    w = np.hanning(h.size)
    h = h * w
    h /= np.sqrt(np.sum(np.abs(h) ** 2))          # unit noise gain (keeps the median floor at 1.0)
    return h.astype(np.complex64)


def apply_receiver(canvas: np.ndarray, taps: np.ndarray) -> np.ndarray:
    from scipy.signal import oaconvolve
    return oaconvolve(canvas, taps, mode="same").astype(C64)


def add_dc_spur(canvas: np.ndarray, rate: float, fft_size: int, spur_db: float, rng) -> float:
    """A CW line at DC (+/- a fraction of a wide bin) spur_db above the per-bin noise floor. Returns the
    spur frequency (Hz) so the scene can LABEL it: receiver spurs count as signals (user convention
    2026-11-03: flag every emission in the capture, filter spurs downstream)."""
    # hann-rms window: tone peak-bin power ~ A^2 N^2 * 0.667, noise per bin ~ N (sigma^2 = 1)
    A = math.sqrt(10 ** (spur_db / 10.0) / (0.667 * fft_size))
    f = rng.uniform(-0.25, 0.25) * (rate / fft_size)
    y = np.full(canvas.size, A, dtype=C64)
    mix_inplace(y, f / rate, float(rng.uniform(0, TWO_PI)))
    canvas += y
    return float(f)


def synthetic_floor_profile(rng) -> np.ndarray:
    """Random plausible receiver floor shape (dB over 1024 bins): tilt + edge roll-off + ripple."""
    x = np.linspace(-0.5, 0.5, 1024, endpoint=False)
    prof = rng.uniform(-2.0, 2.0) * x                       # tilt
    for side in (-1, 1):
        w = rng.uniform(0.01, 0.12)                          # roll-off width (fraction of band)
        depth = rng.uniform(2.0, 18.0)
        d = (0.5 - side * x) if side > 0 else (0.5 + x)       # distance from that edge
        prof -= depth * np.clip(1.0 - d / w, 0, 1) ** 2
    nr = int(rng.integers(0, 4))
    for _ in range(nr):
        prof += rng.uniform(0.1, 0.6) * np.sin(TWO_PI * rng.uniform(2, 30) * x + rng.uniform(0, TWO_PI))
    return prof - np.median(prof)
