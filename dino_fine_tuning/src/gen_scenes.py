"""Scenario-based synthetic scene generator for the DINO-FT multi-rate retrain (successor of gen_491.py).

A *scene* = one complex64 IQ canvas at an X410 sample rate R (5-500 MHz, rates.x410_rates) with per-burst
ground-truth boxes, built from library waveforms (the 287 real MATLAB entries) and synthetic emitters
(synth_mods.py) arranged by a scenario sampler:

  legacy_dense / legacy_sparse  the gen_491 recipes (random emitters), now at any rate + band-edge placement
  cellular_adjacent             2-6 LTE/NR/5G/CDMA/GSM/LMR carriers packed on a raster (5+10+10+20 MHz ...),
                                near/far power spread, FDD-continuous or TDD-bursty, light-to-full load
  ism_collisions                2.4 GHz band: WiFi packet trains on ch 1/6/11 (+bonded), BT classic / BLE
                                hoppers, Zigbee, LoRa/FSK IoT, microwave oven; everything colliding
  drone_ism                     ism_collisions + a drone link (OFDM/FM video 10-40 MHz, 25-50 dB, 80-95 %
                                duty, hopping control bursts, fading, optional ADC clipping)
  hoppers_focus                 BT / BLE / slow full-band tone hoppers / fast generic hoppers
  band_edge_focus               emitters straddling both band edges (incl. wider than the band)
  radar_pulses                  radar pulse trains (CW / chirp pulses) with an optional LTE neighbour
  wideband_fill                 one carrier wider than the band (+ narrow emitters on top)
  noise_only                    receiver noise only (floor shape + DC spur)

Realism knobs come from data/real_stats (real_stats.py): receiver floor templates measured on the X410 at
491.52 / 245.76 / 30.72 MSps (applied as an FIR to signal+noise), DC spur, and the OTA component statistics
(duration / bandwidth / level-above-floor) that the ISM and drone scenarios are matched to.

Levels are PSD SNR (dB above the noise floor in the spectrogram). GT = filled box per burst (per hop, per
packet, per TDD burst) with the designed occupied bandwidth, clipped to the band by the rasterizer.

CLI (preview one scene):  PYTHONPATH=src .venv-ml/bin/python src/gen_scenes.py --scenario drone_ism --rate 491.52e6 --png /tmp/x.png
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json, math, os, time
from pathlib import Path
import numpy as np
from scipy.io import loadmat

import synth_mods as sm
from synth_mods import Burst
import rates as R
import real_backgrounds as rb

# v5 (2026-11-05): probability that a scene at <= 250 MSps is laid over a REAL terminated-input N320 recording
# (data/real_backgrounds) instead of synthetic noise + a floor template. Override with REALBG_P=0 to disable.
P_REALBG = float(os.environ.get("REALBG_P", "0.5"))

HERE = Path(__file__).resolve().parents[1]
LIB = Path(os.environ.get("ARGUS_WAVEFORM_LIB",
                          "~/Documents/holoscan_waveform_generation/generated_waveforms_24576")).expanduser()
FS_LIB = 245.76e6
LIB_CLASSES = ["BPSK", "QPSK", "16QAM", "OFDM", "802_11ax", "5G_Downlink", "Bluetooth", "Broadband_FM", "Narrowband_FM"]
SCENARIOS = ["legacy_dense", "legacy_sparse", "cellular_adjacent", "cellular_uplink", "ism_collisions", "drone_ism",
             "hoppers_focus", "lmr_band", "band_edge_focus", "radar_pulses", "wideband_fill", "broadcast_fm", "noise_only"]
# v2 (2026-11-03): + cellular_uplink (FDD/TDD uplink bands, many UEs), lmr_band (narrowband FH / TDMA land-mobile,
# pagers), broadcast_fm; ism_collisions gained near-far / co-channel collisions, the 5 GHz plan and LAA;
# cellular_adjacent gained UE-model uplink in TDD gaps, time-varying load and in-band interferers.
# v3 (2026-11-03 pm, after the N320 OTA stress test): wideband_fill always keeps a carrier edge in view (a band
# filled edge-to-edge by a noise-like carrier is indistinguishable from an empty band after per-frame robust
# normalisation and taught M6-S/M7-S to paint whole tiles on quiet high-gain frames); noise_only doubled and
# half of it is now a "quiet band" with a few narrow persistent lines; DC spur labelled as a signal.
SCENARIO_WEIGHTS = {"legacy_dense": .08, "legacy_sparse": .10, "cellular_adjacent": .13, "cellular_uplink": .10,
                    "ism_collisions": .13, "drone_ism": .10, "hoppers_focus": .08, "lmr_band": .07,
                    "band_edge_focus": .08, "radar_pulses": .02, "wideband_fill": .01, "broadcast_fm": .02, "noise_only": .08}
TWO_PI = 2 * math.pi


# --------------------------------------------------------------------------- library
@dataclass
class LibEntry:
    path: Path
    cls: str
    bw_hz: float
    n: int
    dur_s: float


_LIB_INDEX: list[LibEntry] | None = None
_LIB_CACHE: dict[Path, np.ndarray] = {}


def lib_index() -> list[LibEntry]:
    global _LIB_INDEX
    if _LIB_INDEX is None:
        out = []
        for cls in LIB_CLASSES:
            d = LIB / cls
            if not d.is_dir():
                continue
            for j in sorted(d.glob("*.json")):
                try:
                    md = json.loads(j.read_text())
                except Exception:
                    continue
                bw = md.get("designedOccupiedBandwidthHz"); n = md.get("numOutputSamples")
                if not bw or not n or not j.with_suffix(".mat").exists():
                    continue
                out.append(LibEntry(j.with_suffix(".mat"), cls, float(bw), int(n), int(n) / FS_LIB))
        _LIB_INDEX = out
    return _LIB_INDEX


def _active_segment(x: np.ndarray, win: int = 2048, rel: float = 0.1) -> np.ndarray:
    """Longest contiguous run where the smoothed envelope exceeds rel x its max (drops idle padding)."""
    from scipy.ndimage import uniform_filter1d
    p = uniform_filter1d(np.abs(x) ** 2, win)
    act = p > rel * p.max()
    if act.mean() >= 0.9 or not act.any():      # whole entries with short real inter-frame gaps stay intact
        return x
    edges = np.flatnonzero(np.diff(np.r_[0, act.astype(np.int8), 0]))
    starts, stops = edges[0::2], edges[1::2]
    k = int(np.argmax(stops - starts))
    s0, s1 = int(starts[k]), int(stops[k])
    return x[s0:s1] if (s1 - s0) >= 1024 else x


def lib_wave(e: LibEntry) -> np.ndarray:
    w = _LIB_CACHE.get(e.path)
    if w is None:
        raw = np.asarray(loadmat(str(e.path))["f_sig"]).ravel().astype(np.complex64)
        w = sm.rms_norm(_active_segment(raw))
        if len(_LIB_CACHE) > 64:
            _LIB_CACHE.pop(next(iter(_LIB_CACHE)))
        _LIB_CACHE[e.path] = w
    return w


def lib_active_dur(e: LibEntry) -> float:
    return lib_wave(e).size / FS_LIB


def lib_pick(rng, cls: str | None = None, bw_max: float | None = None, bw_min: float = 0.0) -> LibEntry | None:
    cands = [e for e in lib_index() if (cls is None or e.cls == cls) and (bw_max is None or e.bw_hz <= bw_max) and e.bw_hz >= bw_min]
    return cands[rng.integers(len(cands))] if cands else None


# --------------------------------------------------------------------------- helpers
def logu(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def draw_snr(rng, strong=(15, 35), medium=(6, 15), weak=(-2, 6), p=(0.25, 0.40, 0.35)) -> float:
    k = rng.choice(3, p=p)
    lo, hi = (strong, medium, weak)[k]
    return float(rng.uniform(lo, hi))


def choose(rng, items: dict):
    ks = list(items.keys()); w = np.asarray([items[k] for k in ks], dtype=float); w /= w.sum()
    return ks[rng.choice(len(ks), p=w)]


@dataclass
class Job:
    """One emitter to render: content factory + bursts + placement mode."""
    mode: str                      # single | train | hops
    content: callable              # () -> (x, r0, bw)
    bursts: list
    ramp_s: float = 2e-6
    fading: tuple | None = None
    phase_noise_hz: float = 0.0
    multipath: object = "auto"     # None | "auto" | (tau_max_s,) | (tau_max_s, n_taps)
    pa_db: object = "auto"         # None | "auto" | input backoff in dB (PA shoulders)


@dataclass
class Scene:
    scenario: str
    rate_hz: float
    n_frames: int
    seed: int
    canvas: np.ndarray
    bursts: list
    meta: dict = field(default_factory=dict)

    @property
    def fft_size(self): return R.app_fft_size(self.rate_hz)
    @property
    def frame_samples(self): return R.frame_samples(self.rate_hz)


# --------------------------------------------------------------------------- emitter factories
def lib_content(e: LibEntry):
    return lambda: (lib_wave(e), FS_LIB, e.bw_hz)


def synth_content(rng, kind: str, bw_hint: float | None = None, dur_s: float = 5e-3):
    """Return (factory, bw, label) for a synthetic emitter kind."""
    s = int(rng.integers(0, 2**31))
    if kind == "psk":
        M = int(choose(rng, {2: .3, 4: .4, 8: .15, 16: .1, 64: .05}))
        beta = float(rng.uniform(0.15, 0.5))
        bw = bw_hint or logu(rng, 10e3, 40e6)
        rs = bw / (1 + beta)
        lab = {2: "BPSK", 4: "QPSK", 8: "8PSK", 16: "16QAM", 64: "64QAM"}[M]
        return (lambda: sm.mod_linear(np.random.default_rng(s), M, rs, dur_s, beta)), bw, lab
    if kind == "gfsk":
        bw = bw_hint or logu(rng, 5e3, 4e6)
        h = float(rng.uniform(0.3, 1.0)); M = 2 if rng.random() < .75 else 4
        rs = bw / (1 + h * (M - 1) / 1.0)
        return (lambda: sm.mod_gfsk(np.random.default_rng(s), rs, dur_s, h, float(rng.uniform(0.3, 0.6)), M)), bw, ("GFSK" if M == 2 else "4FSK")
    if kind == "ofdm":
        bw = bw_hint or logu(rng, 200e3, 60e6)
        nfft = int(choose(rng, {64: .2, 128: .1, 256: .2, 512: .15, 1024: .2, 2048: .15}))
        n_used = int(nfft * rng.uniform(0.6, 0.95)); scs = bw / n_used
        cp = float(choose(rng, {0.0625: .3, 0.07: .3, 0.125: .2, 0.25: .2}))
        M = int(choose(rng, {4: .4, 16: .4, 64: .2}))
        return (lambda: sm.mod_ofdm(np.random.default_rng(s), nfft, n_used, scs, cp, dur_s, M)), bw, "OFDM"
    if kind == "chirp":
        bw = bw_hint or float(choose(rng, {125e3: .3, 250e3: .2, 500e3: .2, 2e6: .1, 10e6: .1, 30e6: .1}))
        if bw <= 500e3:
            sf = int(rng.integers(7, 12)); tsym = (2 ** sf) / bw
            return (lambda: sm.mod_chirp(np.random.default_rng(s), bw, tsym, dur_s, 0.95, 2.0, sync_down=2)), bw, "LoRa"
        tsym = logu(rng, 10e-6, 1e-3)
        return (lambda: sm.mod_chirp(np.random.default_rng(s), bw, tsym, dur_s, 1.0, 2.0)), bw, "FMCW"
    if kind == "fm":
        if rng.random() < 0.6:
            dev = float(choose(rng, {2.5e3: .4, 5e3: .4, 75e3: .2})); mbw = 3e3 if dev < 50e3 else 15e3
            lab = "Narrowband_FM"
        else:
            dev = logu(rng, 0.5e6, 8e6); mbw = logu(rng, 0.1e6, 5e6); lab = "Broadband_FM"
        bw = 2 * (dev + mbw)
        return (lambda: sm.mod_fm(np.random.default_rng(s), dev, mbw, dur_s)), bw, lab
    if kind == "noise":
        bw = bw_hint or logu(rng, 100e3, 80e6)
        return (lambda: sm.mod_noise_band(np.random.default_rng(s), bw, dur_s)), bw, "NoiseBand"
    if kind == "cw":
        return (lambda: sm.mod_cw(np.random.default_rng(s), dur_s)), 1.0, "CW"
    raise ValueError(kind)


def random_fc(rng, R_hz: float, bw: float, p_edge: float = 0.2) -> float:
    """Centre frequency: interior (whole signal inside) or straddling one band edge."""
    if bw >= 0.98 * R_hz:
        return float(rng.uniform(-0.3, 0.3) * R_hz)
    if rng.random() < p_edge:
        side = 1 if rng.random() < 0.5 else -1
        return side * (R_hz / 2.0 + rng.uniform(-0.45, 0.45) * bw)
    lim = R_hz / 2.0 - bw / 2.0
    return float(rng.uniform(-lim, lim))


# --------------------------------------------------------------------------- scenarios
def scen_legacy(rng, R_hz, T, dense: bool, tag) -> list[Job]:
    jobs = []
    if dense:
        n = int(rng.uniform(12, 40) * (R_hz / 491.52e6) ** 0.3 * max(0.5, T / 0.1) ** 0.5)
        n = max(6, min(n, 120))
    else:
        n = int(rng.integers(2, 9))
    for i in range(n):
        dur = logu(rng, 80e-6, min(T, 60e-3))
        t0 = float(rng.uniform(-dur, T))
        snr = draw_snr(rng)
        if rng.random() < 0.6:
            e = lib_pick(rng, bw_max=2.0 * R_hz)
            if e is None:
                continue
            if e.cls in ("802_11ax", "Bluetooth"):
                dur = lib_active_dur(e) * int(rng.integers(1, 3))
            fac, bw, lab = lib_content(e), e.bw_hz, e.cls
        else:
            kind = choose(rng, {"psk": .3, "gfsk": .15, "ofdm": .25, "chirp": .1, "fm": .1, "noise": .05, "cw": .05})
            fac, bw, lab = synth_content(rng, kind, None, dur)
        fc = random_fc(rng, R_hz, bw, p_edge=0.2)
        b = Burst(lab, "legacy", i, fc, t0, dur, bw, snr, tag)
        jobs.append(Job("single", fac, [b], ramp_s=logu(rng, 1e-6, 10e-6)))
    return jobs


LTE_PLAN = {1.4e6: (128, 72), 3e6: (256, 180), 5e6: (512, 300), 10e6: (1024, 600), 15e6: (1536, 900), 20e6: (2048, 1200)}
NR30_PLAN = {10e6: (512, 288), 15e6: (512, 456), 20e6: (1024, 612), 40e6: (2048, 1272), 50e6: (2048, 1596), 100e6: (4096, 3276)}
NR15_PLAN = {5e6: (512, 300), 10e6: (1024, 624), 15e6: (1536, 948), 20e6: (2048, 1272)}


def carrier_job(rng, R_hz, T, fc, bw, tech, base_snr, tag, eid) -> list[Job]:
    """One cellular carrier (continuous FDD or TDD-bursty) as Jobs with GT per on-interval."""
    jobs = []
    s = int(rng.integers(0, 2**31))
    snr = base_snr
    if tech == "lte":
        plan_bw = min(LTE_PLAN, key=lambda k: abs(k - bw)); nfft, nu = LTE_PLAN[plan_bw]; scs = 15e3
        load = float(rng.uniform(0.3, 1.0)); lab = "LTE"; slot_syms, ctrl, stripe, every = 14, 2, 72, 5
        bw_eff = nu * scs
    elif tech == "nr":
        if rng.random() < 0.6 and bw >= 10e6:
            plan_bw = min(NR30_PLAN, key=lambda k: abs(k - bw)); nfft, nu = NR30_PLAN[plan_bw]; scs = 30e3; every = 40
        else:
            plan_bw = min(NR15_PLAN, key=lambda k: abs(k - bw)); nfft, nu = NR15_PLAN[plan_bw]; scs = 15e3; every = 20
        load = float(rng.uniform(0.2, 1.0)); lab = "NR"; slot_syms, ctrl, stripe = 14, 2, 240
        bw_eff = nu * scs
    elif tech == "lib5g":
        e = lib_pick(rng, "5G_Downlink", bw_max=bw * 1.15, bw_min=bw * 0.8) or lib_pick(rng, "5G_Downlink", bw_max=bw * 1.5)
        if e is None:
            return []
        fac = lib_content(e); lab = e.cls; bw_eff = e.bw_hz
    elif tech == "libofdm":
        e = lib_pick(rng, "OFDM", bw_max=bw * 1.2, bw_min=bw * 0.7) or lib_pick(rng, "OFDM", bw_max=bw * 1.5)
        if e is None:
            return []
        fac = lib_content(e); lab = e.cls; bw_eff = e.bw_hz
    elif tech == "cdma":
        fac = (lambda: sm.mod_linear(np.random.default_rng(s), 4, 1.2288e6, min(T, 60e-3), 0.2)); lab = "CDMA"; bw_eff = 1.25e6
    elif tech == "umts":
        fac = (lambda: sm.mod_linear(np.random.default_rng(s), 4, 3.84e6, min(T, 60e-3), 0.22)); lab = "UMTS"; bw_eff = 4.7e6
    else:
        raise ValueError(tech)
    if tech in ("lte", "nr"):
        p_idle = float(rng.uniform(0.0, 0.6)) if rng.random() < 0.4 else 0.0      # bursty traffic: idle slots
        fac = (lambda: sm.mod_ofdm(np.random.default_rng(s), nfft, nu, scs, 0.07, min(T, 60e-3), int(choose(rng, {4: .5, 16: .3, 64: .2})),
                                   load=load, ctrl_symbols_per_slot=ctrl, slot_symbols=slot_syms,
                                   center_stripe_sc=stripe, stripe_every_slots=every, p_idle=p_idle, load_jitter=0.15))
    # time pattern
    tdd = rng.random() < 0.4 and tech in ("lte", "nr", "lib5g", "libofdm")
    bursts = []
    if not tdd:
        bursts.append(Burst(lab, f"{tech}_dl", eid, fc, -1e-3, T + 2e-3, bw_eff, snr, tag))
    else:
        period = 10e-3 if (tech == "lte" or rng.random() < 0.5) else 5e-3
        dl_frac = float(rng.uniform(0.4, 0.8))
        t = -float(rng.uniform(0, period))
        ul_windows = []
        while t < T:
            on = period * dl_frac
            bursts.append(Burst(lab, f"{tech}_dl_tdd", eid, fc, t, on, bw_eff, snr, tag))
            ul_windows.append((t + on + 0.2e-3, t + period - 0.1e-3))
            t += period
        if rng.random() < 0.7:   # the UEs answering in the UL slots: sporadic, narrow, 25 dB power spread
            jobs += ue_jobs(rng, R_hz, T, fc, bw_eff, "lte" if tech == "lte" else "nr", snr + float(rng.uniform(-15, 10)),
                            tag, eid * 100 + 1000, windows=ul_windows)
    fading = (float(rng.uniform(0.5, 3.0)), logu(rng, 5e-3, 100e-3)) if rng.random() < 0.4 else None
    jobs.append(Job("train", fac, bursts, ramp_s=5e-6, fading=fading))
    return jobs


def scen_cellular(rng, R_hz, T, tag) -> list[Job]:
    jobs = []; cars = []
    n_car = int(rng.integers(2, 7)) if R_hz >= 30e6 else int(rng.integers(1, 4))
    bws = []
    for _ in range(n_car):
        bws.append(float(choose(rng, {1.4e6: .04, 3e6: .04, 5e6: .2, 10e6: .25, 15e6: .12, 20e6: .2, 40e6: .05, 50e6: .03, 100e6: .05, 1.25e6: .02})))
    guard = float(choose(rng, {0.0: .3, 0.1e6: .2, 0.3e6: .2, 1e6: .2, 2e6: .1}))
    pos, centers = 0.0, []
    for i, bw in enumerate(bws):
        if i > 0 and rng.random() < 0.2:
            pos += logu(rng, 3e6, 30e6)      # a gap in the plan
        centers.append(pos + bw / 2.0); pos += bw + guard
    W = pos
    off = float(rng.uniform(-0.5, 0.5)) * max(R_hz, W) - W / 2.0  # random alignment -> edges straddle
    base = float(rng.uniform(8, 35))
    for i, (bw, c) in enumerate(zip(bws, centers)):
        fc = c + off
        if abs(fc) - bw / 2.0 > R_hz / 2.0:
            continue
        if bw <= 1.3e6:
            tech = "cdma"
        else:
            tech = choose(rng, {"lte": .35, "nr": .3, "lib5g": .15, "libofdm": .1, "umts": .1 if bw <= 5e6 else 0.0})
        snr = base + float(rng.uniform(-20, 5))
        jobs += carrier_job(rng, R_hz, T, fc, bw, tech, snr, tag, i); cars.append((fc, bw, snr))
    # narrowband neighbours: LMR / paging / GSM bursts in gaps or on top
    if rng.random() < 0.6:
        for k in range(int(rng.integers(1, 7))):
            kind = choose(rng, {"lmr": .4, "gsm": .3, "psk": .3})
            s = int(rng.integers(0, 2**31))
            if kind == "lmr":
                dev = float(choose(rng, {2.5e3: .5, 5e3: .5})); bw = 2 * (dev + 3e3); lab = "LMR"
                fac = (lambda s=s, dev=dev: sm.mod_fm(np.random.default_rng(s), dev, 3e3, 50e-3))
                dur = logu(rng, 50e-3, 2.0); t0 = float(rng.uniform(-dur, T))
                jobs.append(Job("single", fac, [Burst(lab, "lmr", 100 + k, random_fc(rng, R_hz, bw, 0.1), t0, dur, bw, draw_snr(rng), tag)]))
            elif kind == "gsm":
                bw = 200e3; lab = "GSM"
                fac = (lambda s=s: sm.mod_gfsk(np.random.default_rng(s), 270.833e3, 30e-3, 0.5, 0.3, 2))
                fc = random_fc(rng, R_hz, bw, 0.1); n_slots = int(rng.integers(1, 9)); snr = draw_snr(rng)
                bursts = []
                t = -float(rng.uniform(0, 4.615e-3))
                while t < T:
                    for sl in range(8):
                        if sl < n_slots:
                            bursts.append(Burst(lab, "gsm", 200 + k, fc, t + sl * 576.9e-6, 546e-6, bw, snr, tag))
                    t += 4.615e-3
                jobs.append(Job("train", fac, bursts, ramp_s=10e-6))
            else:
                fac, bw, lab = synth_content(rng, "psk", logu(rng, 12.5e3, 500e3), 20e-3)
                dur = logu(rng, 1e-3, 100e-3); t0 = float(rng.uniform(-dur, T))
                jobs.append(Job("single", fac, [Burst(lab, "nb", 300 + k, random_fc(rng, R_hz, bw, 0.1), t0, dur, bw, draw_snr(rng), tag)]))
    # narrowband interferer sitting inside a carrier (CW spur / FSK / data) above the carrier's PSD
    if cars and rng.random() < 0.3:
        for k in range(int(rng.integers(1, 3))):
            fc_c, bw_c, snr_c = cars[rng.integers(len(cars))]
            kind = choose(rng, {"cw": .4, "gfsk": .3, "psk": .3})
            fac, bwi, lab = synth_content(rng, kind, logu(rng, 5e3, 200e3) if kind != "cw" else None, 50e-3)
            fci = fc_c + float(rng.uniform(-0.45, 0.45)) * bw_c
            dur = logu(rng, 5e-3, 1.0); t0 = float(rng.uniform(-dur, T))
            jobs.append(Job("single", fac, [Burst(lab, "inband_interferer", 500 + k, fci, t0, dur, bwi, snr_c + float(rng.uniform(0, 15)), tag)]))
    return jobs


WIFI_CH = {ch: 2412e6 + 5e6 * (ch - 1) for ch in range(1, 14)}; WIFI_CH[14] = 2484e6


def wifi_train(rng, R_hz, T, fc, bw, snr, tag, eid, duty, lib_ok=True) -> list[Job]:
    s = int(rng.integers(0, 2**31))
    use_lib = lib_ok and rng.random() < 0.4
    if use_lib:
        e = lib_pick(rng, "802_11ax", bw_max=bw * 1.05, bw_min=bw * 0.95)
        if e is None:
            use_lib = False
    if use_lib:
        fac = lib_content(e); lab = "802_11ax"; pkt_dur_fixed = lib_active_dur(e)
    else:
        ax = rng.random() < 0.5
        if ax:
            nfft, nu, scs, cp = 256, 234, 78.125e3, 0.1
        else:
            nfft, nu, scs, cp = 64, 52, 312.5e3, 0.25
        mult = int(round(bw / 20e6)); nfft *= mult; nu = nu * mult + (mult - 1) * 2 if mult > 1 else nu
        M = int(choose(rng, {2: .15, 4: .3, 16: .3, 64: .25}))
        fac = (lambda: sm.mod_ofdm(np.random.default_rng(s), nfft, nu, scs, cp, 40e-3, M)); lab = "802_11ax" if ax else "802_11g"
        pkt_dur_fixed = None
        bw = nu * scs                                 # GT = occupied bandwidth (16.25 / 18.3 MHz per 20 MHz)
    bursts, t = [], -float(rng.uniform(0, 2e-3))
    sta_snr = snr + float(rng.uniform(-20, 8))      # the "other side" of the link (uplink packets)
    while t < T:
        dur = pkt_dur_fixed if pkt_dur_fixed else logu(rng, 40e-6, 3e-3)
        who = snr if rng.random() < 0.7 else sta_snr
        bursts.append(Burst(lab, "wifi", eid, fc, t, dur, bw, who + float(rng.uniform(-2, 2)), tag))
        t += dur
        if rng.random() < 0.6:                       # SIFS + ACK from the other side
            t += 16e-6
            ack = float(rng.uniform(28e-6, 60e-6))
            bursts.append(Burst(lab, "wifi_ack", eid, fc, t, ack, bw, (sta_snr if who == snr else snr), tag))
            t += ack
        gap = 34e-6 + logu(rng, 9e-6, 20e-3) * (1.0 - duty) ** 2 * 4
        t += gap
    return [Job("train", fac, bursts, ramp_s=1.5e-6)]


def hopper_job(rng, R_hz, T, kind, tag, eid, center_rel=0.0, span=None, snr=None) -> list[Job]:
    """kind: bt | ble | slow_tone | fast | wide"""
    s = int(rng.integers(0, 2**31)); snr = draw_snr(rng, (12, 32), (4, 12), (-2, 4)) if snr is None else snr
    bursts = []
    if kind == "bt":
        chans = np.array([center_rel + (2402e6 + 1e6 * k - 2441e6) for k in range(79)])
        bw = 1e6
        e = lib_pick(rng, "Bluetooth", bw_max=1.1e6)
        if e is not None and rng.random() < 0.5:
            fac = lib_content(e); lab = "Bluetooth"
        else:
            fac = (lambda: sm.mod_gfsk(np.random.default_rng(s), 1e6, 30e-3, 0.32, 0.5, 2)); lab = "Bluetooth"
        t = -float(rng.uniform(0, 625e-6)); active = float(rng.uniform(0.2, 1.0))
        while t < T:
            slots = int(choose(rng, {1: .7, 3: .2, 5: .1}))
            if rng.random() < active:
                dur = {1: 366e-6, 3: 1622e-6, 5: 2870e-6}[slots]
                bursts.append(Burst(lab, "bt_hop", eid, float(chans[rng.integers(79)]), t, dur, bw, snr + float(rng.uniform(-1, 1)), tag))
            t += 625e-6 * slots
    elif kind == "ble":
        chans = np.array([center_rel + (2402e6 + 2e6 * k - 2441e6) for k in range(40)])
        bw = 2e6
        fac = (lambda: sm.mod_gfsk(np.random.default_rng(s), 1e6, 30e-3, 0.5, 0.5, 2)); lab = "BLE"
        t = -float(rng.uniform(0, 5e-3)); interval = logu(rng, 7.5e-3, 50e-3)
        adv = rng.random() < 0.4
        while t < T:
            if adv:
                for k in (0, 12, 39):
                    dur = float(rng.uniform(100e-6, 400e-6))
                    bursts.append(Burst(lab, "ble_adv", eid, float(chans[k]), t, dur, bw, snr, tag)); t += dur + 150e-6
                t += logu(rng, 20e-3, 200e-3)
            else:
                n_pk = int(rng.integers(1, 4))
                ch = float(chans[rng.integers(40)])
                for _ in range(n_pk):
                    dur = logu(rng, 80e-6, 2.1e-3)
                    bursts.append(Burst(lab, "ble_data", eid, ch, t, dur, bw, snr, tag)); t += dur + 150e-6
                t += interval
    elif kind == "slow_tone":
        bw = logu(rng, 1e3, 50e3); span = span or float(rng.uniform(0.3, 1.0)) * R_hz
        n_ch = int(rng.integers(10, 200)); chans = center_rel + np.linspace(-span / 2, span / 2, n_ch)
        if rng.random() < 0.5:
            fac = (lambda: sm.mod_cw(np.random.default_rng(s), 100e-3)); lab = "CW"
        else:
            fac = (lambda: sm.mod_gfsk(np.random.default_rng(s), bw / 2, 100e-3, 0.7, 0.5, 2)); lab = "GFSK"
        dwell = logu(rng, 20e-3, 200e-3); gap = logu(rng, 1e-3, 300e-3)
        t = -float(rng.uniform(0, dwell + gap))
        while t < T:
            bursts.append(Burst(lab, "slow_hop", eid, float(chans[rng.integers(n_ch)]), t, dwell, bw, snr, tag)); t += dwell + gap
    elif kind == "fast":
        bw = logu(rng, 25e3, 2e6); span = span or float(rng.uniform(0.2, 1.0)) * R_hz
        n_ch = int(rng.integers(5, 100)); chans = center_rel + np.linspace(-span / 2, span / 2, n_ch)
        if rng.random() < 0.5:
            fac = (lambda: sm.mod_gfsk(np.random.default_rng(s), bw / 1.5, 100e-3, 0.5, 0.5, 2)); lab = "GFSK"
        else:
            fac = (lambda: sm.mod_linear(np.random.default_rng(s), 4, bw / 1.35, 100e-3, 0.35)); lab = "QPSK"
        dwell = logu(rng, 0.2e-3, 20e-3); period = dwell * float(rng.uniform(1.0, 1.6))
        t = -float(rng.uniform(0, period))
        while t < T:
            bursts.append(Burst(lab, "fast_hop", eid, float(chans[rng.integers(n_ch)]), t, dwell, bw, snr, tag)); t += period
    elif kind == "wide":
        bw = logu(rng, 2e6, 10e6); span = span or R_hz
        n_ch = int(rng.integers(4, 40)); chans = center_rel + np.linspace(-span / 2 + bw / 2, span / 2 - bw / 2, n_ch)
        fac = (lambda: sm.mod_linear(np.random.default_rng(s), 4, bw / 1.3, 100e-3, 0.3)); lab = "QPSK"
        dwell = logu(rng, 0.1e-3, 5e-3); period = dwell * float(rng.uniform(1.0, 1.3))
        t = -float(rng.uniform(0, period))
        while t < T:
            bursts.append(Burst(lab, "wide_hop", eid, float(chans[rng.integers(n_ch)]), t, dwell, bw, snr, tag)); t += period
    else:
        raise ValueError(kind)
    return [Job("hops", fac, bursts, ramp_s=2e-6)]


def scen_ism(rng, R_hz, T, tag, drone=False) -> list[Job]:
    jobs = []
    band = "5g" if (not drone and rng.random() < 0.35) else "2g4"
    if band == "2g4":
        c0 = float(rng.uniform(2.400e9, 2.485e9))      # canvas RF centre within the 2.4 GHz band
    else:
        c0 = float(rng.uniform(5.17e9, 5.85e9))        # UNII-1..3
    vis = lambda f, bw: abs(f - c0) - bw / 2.0 < R_hz / 2.0
    chan_map = wifi_channels(band)
    # WiFi APs
    n_ap = int(rng.integers(1, 5))
    if band == "2g4":
        chans = list(rng.choice([1, 6, 11], size=min(n_ap, 3), replace=False)) if rng.random() < 0.6 else list(rng.integers(1, 14, size=n_ap))
    else:
        near = [ch for ch, f in chan_map.items() if vis(f, 80e6)] or list(chan_map.keys())
        chans = list(rng.choice(near, size=min(n_ap, len(near)), replace=False))
    ap_centers = []
    for i, ch in enumerate(chans):
        f = chan_map[int(ch)]
        if band == "2g4":
            bw = 20e6
            if rng.random() < 0.2 and R_hz >= 40e6:
                bw = 40e6; f += 10e6 * (1 if rng.random() < 0.5 else -1)
        else:
            bw = float(choose(rng, {20e6: .3, 40e6: .3, 80e6: .35, 160e6: .05}))
            if bw == 40e6: f += 10e6 * (1 if rng.random() < 0.5 else -1)
            elif bw == 80e6: f += float(rng.choice([-30e6, -10e6, 10e6, 30e6]))
            elif bw == 160e6: f += float(rng.choice([-70e6, -50e6, -30e6, -10e6, 10e6, 30e6, 50e6, 70e6]))
        if not vis(f, bw):
            continue
        ap_centers.append(f - c0)
        snr = draw_snr(rng, (18, 32), (6, 18), (-2, 6), (0.30, 0.50, 0.20)); duty = float(rng.uniform(0.1, 0.85))
        jobs += wifi_train(rng, R_hz, T, f - c0, bw, snr, tag, 10 + i, duty, lib_ok=(bw <= 40e6))
        # near-far partner: same channel (co-channel collisions) or an overlapping channel, 15-40 dB apart
        if rng.random() < 0.5:
            f2 = f if rng.random() < 0.5 else f + float(rng.choice([-10e6, -5e6, 5e6, 10e6] if band == "2g4" else [-20e6, 20e6]))
            snr2 = snr - float(rng.uniform(15, 40)) if rng.random() < 0.6 else snr + float(rng.uniform(10, 30))
            jobs += wifi_train(rng, R_hz, T, f2 - c0, bw, snr2, tag, 100 + i, float(rng.uniform(0.1, 0.8)), lib_ok=(bw <= 40e6))
        if rng.random() < 0.25:   # hidden node: a third station on the same channel, high duty -> frequent collisions
            jobs += wifi_train(rng, R_hz, T, f - c0, bw, snr + float(rng.uniform(-10, 10)), tag, 200 + i, float(rng.uniform(0.4, 0.9)), lib_ok=(bw <= 40e6))
    if band == "5g":
        if rng.random() < 0.3 and ap_centers:        # LAA / NR-U: LBT bursts of 2-10 ms on a WiFi channel
            jobs += laa_train(rng, R_hz, T, float(ap_centers[rng.integers(len(ap_centers))]) + float(rng.choice([-20e6, 0.0, 20e6])), tag, 300)
        if rng.random() < 0.1:                       # DFS radar pulses in the band
            jobs += radar_train(rng, R_hz, T, tag, 310)
    # Bluetooth classic / BLE hoppers (channel maps relative to 2441 MHz -> shift by c0)
    rel = 2441e6 - c0
    n24 = (lambda hi: int(rng.integers(0, hi)) if band == "2g4" else 0)
    for k in range(n24(4)):
        jobs += hopper_job(rng, R_hz, T, "bt", tag, 20 + k, center_rel=rel)
    for k in range(n24(4)):
        jobs += hopper_job(rng, R_hz, T, "ble", tag, 30 + k, center_rel=rel)
    # Zigbee
    for k in range(n24(3)):
        ch = int(rng.integers(11, 27)); f = 2405e6 + 5e6 * (ch - 11); bw = 2e6
        if not vis(f, bw):
            continue
        s = int(rng.integers(0, 2**31)); snr = draw_snr(rng)
        fac = (lambda s=s: sm.mod_linear(np.random.default_rng(s), 4, 1e6, 20e-3, 0.8, oqpsk=True))
        bursts, t = [], -float(rng.uniform(0, 20e-3))
        while t < T:
            dur = logu(rng, 0.35e-3, 4e-3); bursts.append(Burst("Zigbee", "zigbee", 40 + k, f - c0, t, dur, bw, snr, tag))
            t += dur + logu(rng, 2e-3, 100e-3)
        jobs.append(Job("train", fac, bursts, ramp_s=4e-6))
    # IoT narrowband: LoRa / FSK
    for k in range(n24(4)):
        kind = "chirp" if rng.random() < 0.5 else "gfsk"
        dur = logu(rng, 5e-3, 300e-3)
        fac, bw, lab = synth_content(rng, kind, (float(choose(rng, {125e3: .5, 250e3: .3, 500e3: .2})) if kind == "chirp" else logu(rng, 20e3, 500e3)), dur)
        fc = random_fc(rng, R_hz, bw, 0.1); t0 = float(rng.uniform(-dur, T))
        jobs.append(Job("single", fac, [Burst(lab, "iot", 50 + k, fc, t0, dur, bw, draw_snr(rng), tag)]))
    # microwave oven: swept band-limited noise, 50 % duty at mains rate
    if band == "2g4" and rng.random() < 0.12:
        s = int(rng.integers(0, 2**31)); bw = logu(rng, 3e6, 10e6); swing = logu(rng, 5e6, 25e6)
        f0 = float(rng.uniform(-0.4, 0.4)) * R_hz; snr = float(rng.uniform(5, 30))
        fac = (lambda s=s: sm.mod_noise_band(np.random.default_rng(s), bw, 20e-3))
        bursts, t = [], -float(rng.uniform(0, 16.7e-3))
        while t < T:
            for j in range(8):                       # 8 x ~1 ms sub-bursts tracing the sweep
                fcj = f0 + swing * (j / 8.0 - 0.5)
                bursts.append(Burst("Microwave", "oven", 60, fcj, t + j * 1.04e-3, 1.04e-3, bw, snr, tag))
            t += 16.7e-3
        jobs.append(Job("hops", fac, bursts, ramp_s=50e-6))
    if band == "2g4" and rng.random() < 0.15:
        jobs += fhss_network_jobs(rng, R_hz, T, tag, 900)
    if drone:
        jobs += drone_jobs(rng, R_hz, T, tag, ap_centers)
    return jobs


def drone_jobs(rng, R_hz, T, tag, ap_centers) -> list[Job]:
    jobs = []
    if ap_centers and rng.random() < 0.5:
        fc = float(ap_centers[rng.integers(len(ap_centers))]) + float(rng.uniform(-3e6, 3e6))
    else:
        fc = float(rng.uniform(-0.45, 0.45)) * R_hz
    vbw = float(choose(rng, {10e6: .4, 20e6: .4, 40e6: .2 if R_hz >= 60e6 else 0.0}))
    vsnr = float(rng.uniform(25, 50)); s = int(rng.integers(0, 2**31))
    fading = (float(rng.uniform(1.0, 3.0)), logu(rng, 10e-3, 60e-3))
    if rng.random() < 0.3:   # analog FM video (library wideband FM) - continuous
        e = lib_pick(rng, "Broadband_FM", bw_max=25e6, bw_min=15e6)
        if e is not None:
            jobs.append(Job("train", lib_content(e), [Burst("Broadband_FM", "drone_video_fm", 70, fc, -1e-3, T + 2e-3, e.bw_hz, vsnr, tag)], fading=fading))
    else:
        nfft = 1024 if vbw <= 10e6 else (2048 if vbw <= 20e6 else 4096); nu = int(nfft * 0.6); scs = vbw / nu
        fac = (lambda: sm.mod_ofdm(np.random.default_rng(s), nfft, nu, scs, 0.07, 30e-3, int(choose(rng, {4: .4, 16: .4, 64: .2}))))
        bursts, t = [], -float(rng.uniform(0, 1e-3))
        while t < T:
            dur = float(rng.uniform(0.7e-3, 2.2e-3)); bursts.append(Burst("DroneVideo", "drone_video", 70, fc, t, dur, vbw, vsnr + float(rng.uniform(-1, 1)), tag))
            t += dur + float(rng.uniform(50e-6, 300e-6))
        jobs.append(Job("train", fac, bursts, ramp_s=3e-6, fading=fading))
    # control / telemetry hopper around the video link (or across the band)
    cbw = float(choose(rng, {1e6: .5, 2e6: .5})); span = min(R_hz, float(rng.uniform(20e6, 80e6)))
    csnr = float(rng.uniform(15, 40)); s2 = int(rng.integers(0, 2**31))
    fac = (lambda: sm.mod_gfsk(np.random.default_rng(s2), cbw / 1.5, 50e-3, 0.5, 0.5, 2))
    n_ch = int(rng.integers(8, 60)); chans = fc + np.linspace(-span / 2, span / 2, n_ch)
    bursts, t = [], -float(rng.uniform(0, 5e-3)); period = logu(rng, 2e-3, 20e-3)
    while t < T:
        dur = float(rng.uniform(100e-6, 500e-6)); bursts.append(Burst("DroneCtrl", "drone_ctrl", 71, float(chans[rng.integers(n_ch)]), t, dur, cbw, csnr, tag))
        t += period
    jobs.append(Job("hops", fac, bursts, ramp_s=2e-6))
    return jobs


def fhss_network_jobs(rng, R_hz, T, tag, eid0) -> list[Job]:
    """v6: a dense FHSS network (smart-meter / sensor mesh): 8-30 narrowband hoppers (25-200 kHz, GFSK) over a
    shared span, each hopping every 1-50 ms with 20-90 % duty and its own level. Many simultaneous narrow bursts on
    the same time rows -- the pattern that still made M10-S paint a whole tile on the 900 MHz ISM capture."""
    jobs = []
    n = int(rng.integers(8, 31)); span = min(0.95 * R_hz, logu(rng, 2e6, 60e6))
    center = float(rng.uniform(-0.3, 0.3)) * R_hz
    bw = logu(rng, 25e3, 200e3); n_ch = int(rng.integers(16, 100)); chans = center + np.linspace(-span / 2, span / 2, n_ch)
    base = float(rng.uniform(4, 25))
    for k in range(n):
        s = int(rng.integers(0, 2**31)); rs = bw / 1.5
        fac = (lambda s=s, rs=rs: sm.mod_gfsk(np.random.default_rng(s), rs, 100e-3, 0.5, 0.5, 2))
        snr = float(np.clip(rng.normal(base, 6.0), -2, 35)); dwell = logu(rng, 1e-3, 50e-3); duty = float(rng.uniform(0.2, 0.9))
        bursts, t = [], -float(rng.uniform(0, dwell / duty))
        while t < T:
            bursts.append(Burst("GFSK", "fhss_net", eid0 + k, float(chans[rng.integers(n_ch)]), t, dwell, bw, snr, tag))
            t += dwell / duty
        jobs.append(Job("hops", fac, bursts, ramp_s=5e-6, multipath=None))
    return jobs


def scen_hoppers(rng, R_hz, T, tag) -> list[Job]:
    jobs = []
    if rng.random() < 0.35:
        jobs += fhss_network_jobs(rng, R_hz, T, tag, 900)
    for k in range(int(rng.integers(1, 5))):
        kind = choose(rng, {"bt": .2 if R_hz >= 10e6 else .05, "ble": .15, "slow_tone": .15, "fast": .2, "wide": .1 if R_hz >= 20e6 else 0.0, "nbfh": .2})
        if kind == "nbfh":
            jobs += nb_fh_job(rng, R_hz, T, tag, 80 + k, force_hop=True); continue
        center = float(rng.uniform(-0.3, 0.3)) * R_hz if kind in ("bt", "ble") else float(rng.uniform(-0.2, 0.2)) * R_hz
        jobs += hopper_job(rng, R_hz, T, kind, tag, 80 + k, center_rel=center)
    jobs += scen_legacy(rng, R_hz, T, False, tag)[: int(rng.integers(0, 4))]
    return jobs


def scen_band_edge(rng, R_hz, T, tag) -> list[Job]:
    jobs = []
    for i in range(int(rng.integers(3, 11))):
        bw = logu(rng, 50e3, 2.0 * R_hz)
        side = 1 if rng.random() < 0.5 else -1
        fc = side * (R_hz / 2.0 + float(rng.uniform(-0.45, 0.45)) * bw)
        dur = logu(rng, 100e-6, min(T, 80e-3)); t0 = float(rng.uniform(-dur, T)); snr = draw_snr(rng)
        if rng.random() < 0.5:
            e = lib_pick(rng, bw_max=bw * 1.5, bw_min=bw * 0.5) or lib_pick(rng)
            fac, bwe, lab = lib_content(e), e.bw_hz, e.cls
            fc = side * (R_hz / 2.0 + float(rng.uniform(-0.45, 0.45)) * bwe); bw = bwe
        else:
            kind = choose(rng, {"psk": .3, "ofdm": .4, "gfsk": .1, "noise": .1, "fm": .1})
            fac, bw, lab = synth_content(rng, kind, bw, dur)
        jobs.append(Job("single", fac, [Burst(lab, "edge", i, fc, t0, dur, bw, snr, tag)]))
    jobs += scen_legacy(rng, R_hz, T, False, tag)[: int(rng.integers(0, 4))]
    return jobs


def scen_radar(rng, R_hz, T, tag) -> list[Job]:
    jobs = []
    for k in range(int(rng.integers(1, 4))):
        pw = logu(rng, 0.5e-6, 100e-6); pri = logu(rng, max(2 * pw, 50e-6), 5e-3)
        bw = logu(rng, 1e6, min(40e6, R_hz)); chirp = rng.random() < 0.6
        s = int(rng.integers(0, 2**31))
        if chirp:
            fac = (lambda s=s, bw=bw, pw=pw: sm.mod_chirp(np.random.default_rng(s), bw, pw, pw * 20, 1.0, 2.0)); lab = "RadarChirp"
        else:
            fac = (lambda s=s: sm.mod_cw(np.random.default_rng(s), 20e-3)); lab = "RadarCW"; bw = max(1.0 / pw, 1.0)
        fc = random_fc(rng, R_hz, bw, 0.1); snr = float(rng.uniform(10, 45))
        bursts, t = [], -float(rng.uniform(0, pri))
        while t < T:
            bursts.append(Burst(lab, "radar", 90 + k, fc, t, pw, bw, snr, tag)); t += pri
        jobs.append(Job("train", fac, bursts, ramp_s=0.1e-6))
    if rng.random() < 0.6:   # CBRS-like LTE neighbour
        bw = float(choose(rng, {5e6: .25, 10e6: .35, 15e6: .2, 20e6: .2}))
        jobs += carrier_job(rng, R_hz, T, random_fc(rng, R_hz, bw, 0.15), bw, "lte", float(rng.uniform(5, 30)), tag, 95)
    return jobs


def scen_wideband_fill(rng, R_hz, T, tag) -> list[Job]:
    jobs = []
    # a very wide carrier, but with at least one edge inside the band (edge-to-edge fill is ambiguous)
    bw = float(rng.uniform(0.6, 0.95)) * R_hz; fc = float(rng.uniform(-0.5, 0.5)) * (R_hz - bw) * 0.9 + float(rng.choice([-1, 1])) * 0.0
    fc = float(rng.uniform(-(R_hz - bw) / 2 * 0.9, (R_hz - bw) / 2 * 0.9)) if rng.random() < 0.5 else float(rng.choice([-1, 1])) * (R_hz - bw) / 2 * float(rng.uniform(1.0, 1.4))
    snr = float(rng.uniform(5, 30))
    s = int(rng.integers(0, 2**31))
    if rng.random() < 0.6:
        nfft = 4096; nu = int(nfft * 0.9); scs = bw / nu
        fac = (lambda: sm.mod_ofdm(np.random.default_rng(s), nfft, nu, scs, 0.07, 10e-3, 16)); lab = "OFDM"
    else:
        fac = (lambda: sm.mod_noise_band(np.random.default_rng(s), bw, 5e-3)); lab = "NoiseBand"
    jobs.append(Job("train", fac, [Burst(lab, "fill", 0, fc, -1e-3, T + 2e-3, bw, snr, tag)]))
    if rng.random() < 0.5:
        for k in range(int(rng.integers(1, 5))):
            fac2, bw2, lab2 = synth_content(rng, choose(rng, {"psk": .5, "gfsk": .3, "cw": .2}), logu(rng, 10e3, 2e6), 20e-3)
            dur = logu(rng, 1e-3, 50e-3)
            jobs.append(Job("single", fac2, [Burst(lab2, "on_fill", 1 + k, random_fc(rng, R_hz, bw2, 0.0), float(rng.uniform(-dur, T)), dur, bw2, snr + float(rng.uniform(8, 25)), tag)]))
    return jobs


def scenario_jobs(rng, scenario, R_hz, T) -> list[Job]:
    if scenario == "legacy_dense": return scen_legacy(rng, R_hz, T, True, scenario)
    if scenario == "legacy_sparse": return scen_legacy(rng, R_hz, T, False, scenario)
    if scenario == "cellular_adjacent": return scen_cellular(rng, R_hz, T, scenario)
    if scenario == "ism_collisions": return scen_ism(rng, R_hz, T, scenario, drone=False)
    if scenario == "drone_ism": return scen_ism(rng, R_hz, T, scenario, drone=True)
    if scenario == "hoppers_focus": return scen_hoppers(rng, R_hz, T, scenario)
    if scenario == "band_edge_focus": return scen_band_edge(rng, R_hz, T, scenario)
    if scenario == "radar_pulses": return scen_radar(rng, R_hz, T, scenario)
    if scenario == "wideband_fill": return scen_wideband_fill(rng, R_hz, T, scenario)
    if scenario == "cellular_uplink": return scen_cellular_uplink(rng, R_hz, T, scenario)
    if scenario == "lmr_band": return scen_lmr_band(rng, R_hz, T, scenario)
    if scenario == "broadcast_fm": return scen_broadcast_fm(rng, R_hz, T, scenario)
    if scenario == "noise_only": return scen_quiet(rng, R_hz, T, scenario)
    raise ValueError(scenario)



# --------------------------------------------------------------------------- v2 samplers
def resolve_impairments(rng, j: Job):
    """Default impairment policy: multipath notches on wideband content (>= 500 kHz, half the emitters),
    PA shoulders on strong emitters (>= 18 dB, half of them). Jobs can pin either field."""
    bw = j.bursts[0].bw_hz if j.bursts else 0.0
    snr = max((b.snr_psd_db for b in j.bursts), default=0.0)
    mp = j.multipath; pa = j.pa_db
    if isinstance(mp, str):
        mp = (logu(rng, 50e-9, 1.5e-6),) if (bw >= 500e3 and rng.random() < 0.5) else None
    if isinstance(pa, str):
        pa = float(rng.uniform(2.0, 8.0)) if (snr >= 18.0 and bw >= 100e3 and rng.random() < 0.5) else None
    return mp, pa


def wifi_channels(band: str) -> dict:
    if band == "2g4":
        return dict(WIFI_CH)
    chs = list(range(36, 65, 4)) + list(range(100, 145, 4)) + list(range(149, 166, 4))
    return {ch: 5000e6 + 5e6 * ch for ch in chs}


def laa_train(rng, R_hz, T, fc, tag, eid) -> list[Job]:
    """LAA / NR-U: listen-before-talk bursts of a 20 MHz LTE-like carrier, 2-10 ms channel occupancy."""
    s = int(rng.integers(0, 2**31))
    fac = (lambda: sm.mod_ofdm(np.random.default_rng(s), 2048, 1200, 15e3, 0.07, 40e-3, int(choose(rng, {4: .4, 16: .4, 64: .2})),
                               load=float(rng.uniform(0.5, 1.0)), ctrl_symbols_per_slot=2, slot_symbols=14))
    snr = draw_snr(rng, (15, 32), (6, 15), (-2, 6), (.4, .4, .2))
    bursts, t = [], -float(rng.uniform(0, 5e-3))
    while t < T:
        dur = logu(rng, 2e-3, 10e-3)
        bursts.append(Burst("LAA", "laa", eid, fc, t, dur, 18e6, snr, tag)); t += dur + logu(rng, 0.1e-3, 5e-3)
    return [Job("train", fac, bursts, ramp_s=5e-6)]


def radar_train(rng, R_hz, T, tag, eid, fc=None) -> list[Job]:
    pw = logu(rng, 0.5e-6, 50e-6); pri = logu(rng, max(2 * pw, 0.1e-3), 4e-3)
    bw = logu(rng, 1e6, min(20e6, R_hz)); s = int(rng.integers(0, 2**31))
    if rng.random() < 0.6:
        fac = (lambda: sm.mod_chirp(np.random.default_rng(s), bw, pw, pw * 20, 1.0, 2.0)); lab = "RadarChirp"
    else:
        fac = (lambda: sm.mod_cw(np.random.default_rng(s), 20e-3)); lab = "RadarCW"; bw = max(1.0 / pw, 1.0)
    fc = random_fc(rng, R_hz, bw, 0.1) if fc is None else fc
    snr = float(rng.uniform(10, 45)); bursts, t = [], -float(rng.uniform(0, pri))
    while t < T:
        bursts.append(Burst(lab, "radar", eid, fc, t, pw, bw, snr, tag)); t += pri
    return [Job("train", fac, bursts, ramp_s=0.1e-6, multipath=None, pa_db=None)]


def ue_jobs(rng, R_hz, T, fc, bw, tech, base_snr, tag, eid0, windows=None) -> list[Job]:
    """Sporadic uplink traffic of several UEs inside one carrier: PUSCH bursts (1-8 TTIs, 1-100 PRB at random /
    persistent / hopping positions, 25 dB near-far spread), PUCCH bursts hopping between the carrier edges,
    PRACH preambles (6 PRB, 1-3 ms), wideband single-symbol SRS. `windows` restricts activity to UL slots."""
    scs = 15e3 if (tech == "lte" or rng.random() < 0.5) else 30e3
    tti = 1e-3 if scs == 15e3 else 0.5e-3
    prb = 12 * scs
    n_prb_car = max(6, int(bw * 0.9 / prb))
    lab = "LTE_UL" if tech == "lte" else "NR_UL"
    jobs = []

    def in_windows(t0, dur):
        if not windows:
            return (t0, dur)
        for (w0, w1) in windows:
            if w0 <= t0 < w1:
                return (t0, min(dur, w1 - t0))
        return None

    def ofdm_fac(n_sc):
        nfft = 16
        while nfft < n_sc * 1.25:
            nfft *= 2
        s = int(rng.integers(0, 2**31)); M = int(choose(rng, {4: .6, 16: .3, 64: .1}))
        return (lambda: sm.mod_ofdm(np.random.default_rng(s), nfft, n_sc, scs, 0.07, 30e-3, M))

    n_ue = int(rng.integers(2, 13)) if bw >= 3e6 else int(rng.integers(1, 5))
    for u in range(n_ue):
        snr = base_snr + float(rng.uniform(-25, 8))
        n_prb = int(choose(rng, {1: .2, 2: .15, 3: .1, 4: .12, 6: .1, 8: .1, 12: .08, 16: .06, 25: .05, 50: .03, 100: .01}))
        n_prb = min(n_prb, n_prb_car)
        fac = ofdm_fac(n_prb * 12); ubw = n_prb * prb
        mode = choose(rng, {"random": .5, "persistent": .3, "hopping": .2})
        duty = logu(rng, 0.02, 0.4)
        p0 = int(rng.integers(0, n_prb_car - n_prb + 1))
        bursts, t, k_i = [], -float(rng.uniform(0, 20e-3)), 0
        while t < T:
            k = int(choose(rng, {1: .5, 2: .2, 3: .1, 4: .1, 8: .1})); dur = k * tti
            if mode == "random":
                pp = int(rng.integers(0, n_prb_car - n_prb + 1))
            elif mode == "persistent":
                pp = p0
            else:
                pp = p0 if k_i % 2 == 0 else (n_prb_car - n_prb - p0)
            fcb = fc - bw * 0.9 / 2 + (pp + n_prb / 2.0) * prb
            w = in_windows(t, dur)
            if w is not None:
                bursts.append(Burst(lab, "pusch", eid0 + u, fcb, w[0], w[1], ubw, snr + float(rng.uniform(-1.5, 1.5)), tag))
            t += dur + logu(rng, tti, 4 * tti / duty); k_i += 1
        if bursts:
            jobs.append(Job("train" if mode == "persistent" else "hops", fac, bursts, ramp_s=5e-6, multipath=None))
    if rng.random() < 0.8:   # PUCCH: 1 PRB at the carrier edges, lower edge then upper edge within each TTI
        fac = ofdm_fac(12)
        for q in range(int(rng.integers(1, 6))):
            snr = base_snr + float(rng.uniform(-22, 5)); period = logu(rng, 5e-3, 80e-3)
            lo = fc - bw * 0.9 / 2 + prb * (0.5 + q); hi = fc + bw * 0.9 / 2 - prb * (0.5 + q)
            bursts, t = [], -float(rng.uniform(0, period))
            while t < T:
                for (f_, tt) in ((lo, t), (hi, t + tti / 2)):
                    w = in_windows(tt, tti / 2)
                    if w is not None:
                        bursts.append(Burst(lab, "pucch", eid0 + 50 + q, f_, w[0], w[1], prb, snr, tag))
                t += period * float(rng.uniform(0.7, 1.3))
            if bursts:
                jobs.append(Job("hops", fac, bursts, ramp_s=5e-6, multipath=None))
    if rng.random() < 0.6 and n_prb_car >= 6:   # PRACH preambles
        fac = ofdm_fac(72); pp = int(rng.integers(0, n_prb_car - 6 + 1)); fcb = fc - bw * 0.9 / 2 + (pp + 3) * prb
        bursts, t = [], -float(rng.uniform(0, 20e-3))
        while t < T:
            dur = float(choose(rng, {1e-3: .6, 2e-3: .25, 3e-3: .15})); w = in_windows(t, dur)
            if w is not None:
                bursts.append(Burst(lab, "prach", eid0 + 60, fcb, w[0], w[1], 6 * prb, base_snr + float(rng.uniform(-15, 10)), tag))
            t += logu(rng, 10e-3, 80e-3)
        if bursts:
            jobs.append(Job("train", fac, bursts, ramp_s=5e-6, multipath=None))
    if rng.random() < 0.5:   # SRS: wideband single-symbol sounding
        s = int(rng.integers(0, 2**31)); sbw = bw * 0.9 * float(choose(rng, {1.0: .6, 0.5: .4}))
        fac = (lambda: sm.mod_noise_band(np.random.default_rng(s), sbw, 5e-3))
        bursts, t = [], -float(rng.uniform(0, 10e-3)); period = logu(rng, 5e-3, 40e-3)
        while t < T:
            w = in_windows(t, 71e-6)
            if w is not None:
                bursts.append(Burst(lab, "srs", eid0 + 70, fc, w[0], w[1], sbw, base_snr + float(rng.uniform(-10, 5)), tag))
            t += period
        if bursts:
            jobs.append(Job("train", fac, bursts, ramp_s=1e-6, multipath=None))
    return jobs


def wideband_burst_jobs(rng, R_hz, T, tag, eid0=950) -> list[Job]:
    """v7: short EDGE-TO-EDGE bursts (a full-carrier PUSCH or PRACH seen through a narrow look, a wideband pulse):
    1-8 bursts of 0.3-5 ms spanning 0.98 of the band, 3-20 dB, each labelled on its own rows only. Teaches
    "a band-wide burst is painted on its rows, not over the whole tile" (the narrow 850 MHz uplink look)."""
    jobs = []
    bw = 0.98 * R_hz; base = float(rng.uniform(3, 20))
    for k in range(int(rng.integers(1, 9))):
        dur = logu(rng, 0.3e-3, 5e-3); t0 = float(rng.uniform(-dur / 2, T)); snr = float(np.clip(rng.normal(base, 4), 2, 30))
        s = int(rng.integers(0, 2**31))
        if rng.random() < 0.6:
            nfft = 2048; nu = int(nfft * 0.9); scs = bw / nu
            fac = (lambda s=s, nfft=nfft, nu=nu, scs=scs, dur=dur: sm.mod_ofdm(np.random.default_rng(s), nfft, nu, scs, 0.07, dur + 1e-3, 4)); lab = "LTE_UL"
        else:
            fac = (lambda s=s, bw=bw, dur=dur: sm.mod_noise_band(np.random.default_rng(s), bw, dur + 1e-3)); lab = "NoiseBand"
        jobs.append(Job("single", fac, [Burst(lab, "wide_burst", eid0 + k, 0.0, t0, dur, bw, snr, tag)], ramp_s=20e-6, multipath=None))
    return jobs


def scen_cellular_uplink(rng, R_hz, T, tag) -> list[Job]:
    """An FDD uplink band: 1-4 carriers of sporadic UE traffic (no downlink), NB-IoT / LTE-M in the guard
    bands or in-band, a few land-mobile neighbours."""
    jobs = []
    if rng.random() < (0.25 if R_hz <= 20e6 else 0.08):
        jobs += wideband_burst_jobs(rng, R_hz, T, tag)
    n_car = int(rng.integers(1, 5)) if R_hz >= 20e6 else int(rng.integers(1, 3))
    bws = [float(choose(rng, {1.4e6: .05, 3e6: .05, 5e6: .25, 10e6: .3, 15e6: .15, 20e6: .2})) for _ in range(n_car)]
    guard = float(choose(rng, {0.0: .2, 0.1e6: .2, 0.5e6: .2, 1e6: .2, 2e6: .2}))
    pos, centers = 0.0, []
    for i, bw in enumerate(bws):
        if i > 0 and rng.random() < 0.3:
            pos += logu(rng, 2e6, 20e6)
        centers.append(pos + bw / 2.0); pos += bw + guard
    off = float(rng.uniform(-0.5, 0.5)) * max(R_hz, pos) - pos / 2.0
    base = float(rng.uniform(5, 30))
    cars = []
    for i, (bw, c) in enumerate(zip(bws, centers)):
        fc = c + off
        if abs(fc) - bw / 2.0 > R_hz / 2.0:
            continue
        tech = "lte" if rng.random() < 0.55 else "nr"
        jobs += ue_jobs(rng, R_hz, T, fc, bw, tech, base + float(rng.uniform(-10, 5)), tag, 100 * i)
        cars.append((fc, bw))
    if rng.random() < 0.5:   # NB-IoT / LTE-M devices: guard band or in-band, very sporadic
        for k in range(int(rng.integers(1, 5))):
            s = int(rng.integers(0, 2**31))
            if rng.random() < 0.6:
                lab, bwd = "NB-IoT", 180e3
                fac = (lambda s=s: sm.mod_ofdm(np.random.default_rng(s), 16, 12, 15e3, 0.07, 30e-3, 4)) if rng.random() < 0.5 else \
                      (lambda s=s: sm.mod_linear(np.random.default_rng(s), 4, 15e3, 30e-3, 0.22))
            else:
                lab, bwd = "LTE-M", 1.08e6
                fac = (lambda s=s: sm.mod_ofdm(np.random.default_rng(s), 128, 72, 15e3, 0.07, 30e-3, 4))
            if cars and rng.random() < 0.5:
                fc_c, bw_c = cars[rng.integers(len(cars))]
                fcd = fc_c + float(rng.choice([-1, 1])) * (bw_c / 2.0 + bwd / 2.0 + float(rng.uniform(0, 300e3)))   # guard band
            else:
                fcd = random_fc(rng, R_hz, bwd, 0.1)
            snr = draw_snr(rng, (12, 30), (4, 12), (-2, 4))
            bursts, t = [], -float(rng.uniform(0, 0.5))
            while t < T:
                dur = logu(rng, 1e-3, 32e-3); bursts.append(Burst(lab, "nbiot", 600 + k, fcd, t, dur, bwd, snr, tag))
                t += dur + logu(rng, 20e-3, 2.0)
            jobs.append(Job("train", fac, bursts, ramp_s=5e-6, multipath=None))
    if rng.random() < 0.4:
        for k in range(int(rng.integers(1, 5))):
            jobs += nb_fh_job(rng, R_hz, T, tag, 700 + k)
    return jobs


def nb_fh_job(rng, R_hz, T, tag, eid, snr=None, force_hop=None) -> list[Job]:
    """Narrowband land-mobile radio: DMR / TETRA / P25 / NXDN / analog / generic FSK on a 6.25-25 kHz raster,
    push-to-talk transmissions of 0.3-5 s, optional TDMA slot structure (DMR 2 x 30 ms, TETRA 4 x 14.17 ms),
    optionally frequency hopping with 10-500 ms dwells over 4-64 channels."""
    std = choose(rng, {"dmr": .3, "tetra": .15, "p25": .15, "nxdn": .1, "analog": .15, "fsk": .15})
    s = int(rng.integers(0, 2**31)); R0 = np.random.default_rng(s)
    tdma = None
    if std == "dmr":
        fac = (lambda: sm.mod_gfsk(R0, 4800, 0.6, 0.27, 0.2, 4)); chan = 12.5e3; lab = "DMR"
        if rng.random() < 0.6: tdma = (30e-3, 2, [int(rng.integers(0, 2))])
    elif std == "tetra":
        fac = (lambda: sm.mod_linear(R0, 4, 18e3, 0.6, 0.35)); chan = 25e3; lab = "TETRA"
        if rng.random() < 0.5: tdma = (14.17e-3, 4, [int(rng.integers(0, 4))])
    elif std == "p25":
        fac = (lambda: sm.mod_gfsk(R0, 4800, 0.6, 0.25, 0.2, 4)); chan = 12.5e3; lab = "P25"
    elif std == "nxdn":
        fac = (lambda: sm.mod_gfsk(R0, 2400, 0.6, 0.29, 0.2, 4)); chan = 6.25e3; lab = "NXDN"
    elif std == "analog":
        fac = (lambda: sm.mod_fm(R0, 2.5e3, 3e3, 0.6)); chan = 12.5e3; lab = "Narrowband_FM"
    else:
        rs = logu(rng, 1.2e3, 19.2e3); h = float(rng.uniform(0.5, 1.0))
        fac = (lambda: sm.mod_gfsk(R0, rs, 0.6, h, 0.5, 2)); chan = 25e3 if rs * (1 + h) > 12.5e3 else 12.5e3; lab = "FSK"
    snr = draw_snr(rng, (15, 40), (5, 15), (-2, 5), (.35, .4, .25)) if snr is None else snr
    hop = (rng.random() < 0.55) if force_hop is None else bool(force_hop)
    raster = float(choose(rng, {12.5e3: .6, 25e3: .4})) if chan <= 12.5e3 else 25e3
    if hop:
        dwell = logu(rng, 10e-3, 0.5); gap_frac = float(rng.uniform(0.0, 0.1))
        n_ch = int(rng.integers(4, 65)); span = min(0.9 * R_hz, n_ch * raster * float(rng.uniform(1, 8)))
        chans = float(rng.uniform(-0.3, 0.3)) * R_hz + np.linspace(-span / 2, span / 2, n_ch)
    else:
        fc0 = random_fc(rng, R_hz, chan, 0.1)
    bursts = []

    def emit(a, b_, f_):
        """one on-air interval [a,b) at frequency f_, sliced by the TDMA pattern if any"""
        if tdma is None:
            bursts.append(Burst(lab, "nbfh" if hop else ("tdma" if std in ("dmr", "tetra") else "lmr"), eid, f_, a, b_ - a, chan, snr, tag)); return
        slot, n_slots, active = tdma; period = slot * n_slots
        t = a - (a % period) if a >= 0 else a
        while t < b_:
            for sl in active:
                t0 = t + sl * slot; t1 = t0 + slot - 1e-3
                if t1 > a and t0 < b_:
                    bursts.append(Burst(lab, "tdma", eid, f_, max(a, t0), min(b_, t1) - max(a, t0), chan, snr, tag))
            t += period

    t = -float(rng.uniform(0, 3.0))
    while t < T:
        tx = logu(rng, 0.3, 5.0); end = t + tx
        if hop:
            tt = t
            while tt < end:
                f_ = float(chans[rng.integers(n_ch)]); d = min(dwell, end - tt)
                emit(tt, tt + d * (1 - gap_frac), f_); tt += d
        else:
            emit(t, end, fc0)
        t = end + logu(rng, 0.2, 3.0)
    bursts = [b for b in bursts if b.dur_s > 0 and b.t0_s < T and b.t0_s + b.dur_s > 0]
    if not bursts:
        return []
    return [Job("hops" if hop else "train", fac, bursts, ramp_s=0.5e-3, multipath=None, pa_db=None)]


def pager_job(rng, R_hz, T, tag, eid) -> list[Job]:
    """FLEX/POCSAG-class paging transmitter: strong 25 kHz 4-FSK bursts of 0.1-2 s."""
    s = int(rng.integers(0, 2**31)); R0 = np.random.default_rng(s)
    fac = (lambda: sm.mod_gfsk(R0, 1600, 0.6, 1.0, 0.5, 4))
    fc = random_fc(rng, R_hz, 25e3, 0.1); snr = float(rng.uniform(15, 45))
    bursts, t = [], -float(rng.uniform(0, 2.0))
    while t < T:
        dur = logu(rng, 0.1, 2.0); bursts.append(Burst("Pager", "pager", eid, fc, t, dur, 25e3, snr, tag)); t += dur + logu(rng, 0.1, 2.0)
    return [Job("train", fac, bursts, ramp_s=1e-3, multipath=None, pa_db=None)]


def scen_lmr_band(rng, R_hz, T, tag) -> list[Job]:
    """A land-mobile / paging band (150 / 450 / 900 MHz): narrowband PTT, TDMA and hopping radios, pagers,
    narrowband data, 40 dB power spread, one wideband neighbour now and then."""
    jobs = []
    if rng.random() < 0.3:
        jobs += fhss_network_jobs(rng, R_hz, T, tag, 900)
    if rng.random() < 0.08:
        jobs += wideband_burst_jobs(rng, R_hz, T, tag)
    for k in range(int(rng.integers(3, 16))):
        kind = choose(rng, {"nbfh": .55, "pager": .15, "nbdata": .1, "ptt": .2})
        if kind == "nbfh":
            jobs += nb_fh_job(rng, R_hz, T, tag, 300 + k)
        elif kind == "pager":
            jobs += pager_job(rng, R_hz, T, tag, 400 + k)
        elif kind == "nbdata":
            fac, bw, lab = synth_content(rng, "psk", logu(rng, 5e3, 50e3), 0.1)
            dur = logu(rng, 5e-3, 2.0)
            jobs.append(Job("single", fac, [Burst(lab, "nbdata", 450 + k, random_fc(rng, R_hz, bw, 0.1), float(rng.uniform(-dur, T)), dur, bw, draw_snr(rng), tag)], multipath=None))
        else:
            s = int(rng.integers(0, 2**31)); dev = float(choose(rng, {2.5e3: .6, 5e3: .4})); bw = 2 * (dev + 3e3)
            fac = (lambda s=s, dev=dev: sm.mod_fm(np.random.default_rng(s), dev, 3e3, 0.6))
            dur = logu(rng, 0.3, 5.0)
            jobs.append(Job("single", fac, [Burst("Narrowband_FM", "ptt", 480 + k, random_fc(rng, R_hz, bw, 0.1), float(rng.uniform(-dur, T)), dur, bw,
                                              draw_snr(rng, (15, 45), (5, 15), (-2, 5), (.4, .35, .25)), tag)], ramp_s=1e-3, multipath=None, pa_db=None))
    if rng.random() < 0.3:
        bw = float(choose(rng, {1.4e6: .3, 3e6: .3, 5e6: .4}))
        jobs += carrier_job(rng, R_hz, T, random_fc(rng, R_hz, bw, 0.4), bw, choose(rng, {"lte": .6, "cdma": .4}), float(rng.uniform(5, 30)), tag, 90)
    return jobs


def scen_broadcast_fm(rng, R_hz, T, tag) -> list[Job]:
    """The FM broadcast band: 8-40 stations on the 200 kHz raster, 75 kHz deviation stereo composites,
    continuous, 40 dB power spread."""
    c0 = float(rng.uniform(88e6, 108e6)); jobs = []
    grid = np.arange(88.1e6, 107.95e6, 200e3)
    n = min(int(rng.integers(8, 41)), grid.size)
    for i, f in enumerate(rng.choice(grid, size=n, replace=False)):
        fc = float(f - c0)
        if abs(fc) - 100e3 > R_hz / 2.0:
            continue
        s = int(rng.integers(0, 2**31))
        fac = (lambda s=s: sm.mod_fm(np.random.default_rng(s), 75e3, 53e3, 30e-3))
        snr = draw_snr(rng, (20, 45), (8, 20), (0, 8), (.4, .4, .2))
        jobs.append(Job("train", fac, [Burst("Broadcast_FM", "fm_station", i, fc, -1e-3, T + 2e-3, 2 * (75e3 + 53e3), snr, tag)], ramp_s=1e-3, multipath=None))
    return jobs


def scen_quiet(rng, R_hz, T, tag) -> list[Job]:
    """Noise-only half the time; otherwise a quiet band with 1-4 narrow persistent lines (CW spurs, narrow
    carriers, a lone pager) -- what a high-gain look at an empty band really shows."""
    extra = wideband_burst_jobs(rng, R_hz, T, tag) if rng.random() < 0.10 else []
    if rng.random() < 0.5:
        return extra
    jobs = list(extra)
    for k in range(int(rng.integers(1, 5))):
        kind = choose(rng, {"cw": .5, "nbfsk": .3, "pager": .2})
        if kind == "pager":
            jobs += pager_job(rng, R_hz, T, tag, 800 + k); continue
        s = int(rng.integers(0, 2**31))
        if kind == "cw":
            fac = (lambda s=s: sm.mod_cw(np.random.default_rng(s), 0.1, drift_hz=float(np.random.default_rng(s).uniform(0, 2e3)))); bw = 2e3; lab = "CW"
        else:
            rs = logu(rng, 1.2e3, 9.6e3)
            fac = (lambda s=s, rs=rs: sm.mod_gfsk(np.random.default_rng(s), rs, 0.1, 0.7, 0.5, 2)); bw = 2 * rs; lab = "FSK"
        dur = logu(rng, 0.05, 5.0)
        jobs.append(Job("single", fac, [Burst(lab, "quiet_line", 800 + k, random_fc(rng, R_hz, bw, 0.05), float(rng.uniform(-dur, T)), dur, bw,
                                              draw_snr(rng, (12, 35), (4, 12), (-2, 4), (.4, .4, .2)), tag)], multipath=None, pa_db=None))
    return jobs


# --------------------------------------------------------------------------- receiver model
_FLOOR_TEMPLATES: list[np.ndarray] | None = None


def floor_templates() -> list[np.ndarray]:
    global _FLOOR_TEMPLATES
    if _FLOOR_TEMPLATES is None:
        _FLOOR_TEMPLATES = []
        p = HERE / "data/real_stats/floor_profiles.json"
        if p.exists():
            for name, r in json.loads(p.read_text()).items():
                if name.startswith("x410_"):
                    _FLOOR_TEMPLATES.append(np.asarray(r["profile_db_1024"], dtype=np.float64))
    return _FLOOR_TEMPLATES


def sample_receiver(rng, scenario: str) -> dict:
    tmpl = floor_templates()
    u = rng.random()
    if tmpl and u < 0.45:
        prof = tmpl[rng.integers(len(tmpl))] * float(rng.uniform(0.6, 1.8))
        if rng.random() < 0.5:
            prof = prof[::-1].copy()
        kind = "measured"
    elif u < 0.85:
        prof = sm.synthetic_floor_profile(rng); kind = "synthetic"
    else:
        prof = np.zeros(1024); kind = "flat"
    dc = float(rng.uniform(3, 20)) if rng.random() < 0.6 else 0.0
    p_clip = {"drone_ism": 0.3, "ism_collisions": 0.08, "cellular_adjacent": 0.08}.get(scenario, 0.04)
    clip_db = float(rng.uniform(0, 6)) if rng.random() < p_clip else None
    return {"profile_kind": kind, "profile": prof, "dc_spur_db": dc, "clip_over_db": clip_db,
            "level_db": float(rng.uniform(-40, 10))}


# --------------------------------------------------------------------------- scene assembly
def n_frames_for(rate_hz: float, rng=None) -> int:
    if rate_hz >= 400e6: return 2
    if rate_hz >= 150e6: return 3
    return 4


def make_scene(scenario: str, rate_hz: float, seed: int, n_frames: int | None = None, verbose=False) -> Scene:
    rng = np.random.default_rng(seed)
    n_frames = n_frames or n_frames_for(rate_hz)
    F = R.app_fft_size(rate_hz); T_samp = n_frames * R.ROWS_WIDE * F; T = T_samp / rate_hz
    canvas = np.zeros(T_samp, dtype=np.complex64)
    P = sm.Placer(canvas, rate_hz, rng)
    wide_bin = rate_hz / F
    jobs = scenario_jobs(rng, scenario, rate_hz, T)
    placed: list[Burst] = []
    t0 = time.time()
    for j in jobs:
        try:
            x, r0, bw_c = j.content()
        except Exception as ex:   # a modulator edge case must not kill the scene
            if verbose: print("content failed:", ex)
            continue
        mp, pa = resolve_impairments(rng, j)
        # chunk long trains so the one-shot resample stays bounded (~30 ms of content per chunk)
        if j.mode == "single":
            b = j.bursts[0]
            if P.place(b, x, r0, wide_bin, j.ramp_s, j.fading, j.phase_noise_hz, mp, pa):
                placed.append(b)
        elif j.mode == "train":
            chunk, on = [], 0.0
            for b in j.bursts:
                chunk.append(b); on += b.dur_s
                if on >= 30e-3:
                    placed += P.place_train(chunk, x, r0, wide_bin, j.ramp_s, j.fading, mp, pa); chunk, on = [], 0.0
            if chunk:
                placed += P.place_train(chunk, x, r0, wide_bin, j.ramp_s, j.fading, mp, pa)
        elif j.mode == "hops":
            chunk, on = [], 0.0
            for b in j.bursts:
                chunk.append(b); on += b.dur_s
                if on >= 30e-3:
                    placed += P.place_hops(chunk, x, r0, wide_bin, j.ramp_s, mp, pa); chunk, on = [], 0.0
            if chunk:
                placed += P.place_hops(chunk, x, r0, wide_bin, j.ramp_s, mp, pa)
    t_sig = time.time() - t0
    # receiver: noise (unit power) -> response shaping -> DC spur -> level -> optional ADC clipping
    rx = sample_receiver(rng, scenario)
    real_bg = None
    if rate_hz <= 250.01e6 and P_REALBG > 0 and rng.random() < P_REALBG:
        real_bg = rb.pick(rng, rate_hz, T_samp)
    if real_bg is not None:
        # REAL receiver noise (terminated-input N320 recording, unit power at the scene rate) under the synthetic
        # emitters. The emitters still get the receiver's pass-band shaping; the background already has its own.
        bg, e, lines = real_bg
        if rx["profile_kind"] != "flat":
            canvas = sm.apply_receiver(canvas, sm.receiver_response_taps(rx["profile"], 513))
        canvas += bg; del bg
        rx["profile_kind"] = "real_bg"; rx["dc_spur_db"] = 0.0
        rx["real_bg"] = {"path": Path(e["path"]).name, "rate_hz": e["rate_hz"], "gain_db": e["gain_db"], "dc_excess_db": e["dc_excess_db"], "n_lines": len(lines)}
        # flag-everything convention: the recording's DC line and persistent spur lines are labelled as signals
        lab_w = max(2.0 * rate_hz / F, rate_hz / R.NFFT_MODEL)
        if e["dc_excess_db"] > 3.0:
            b = Burst("DC_spur", "rx_dc", 9000, 0.0, 0.0, T, lab_w, e["dc_excess_db"], scenario)
            b.sample_start, b.sample_count, b.total_snr_db = 0, T_samp, e["dc_excess_db"]; placed.append(b)
        for k, l in enumerate(lines):
            b = Burst("RX_spur", "rx_spur", 9100 + k, l["f_hz"], 0.0, T, max(l["bw_hz"], lab_w), l["excess_db"], scenario)
            b.sample_start, b.sample_count, b.total_snr_db = 0, T_samp, l["excess_db"]; placed.append(b)
    else:
        noise = (rng.standard_normal(T_samp, dtype=np.float32) + 1j * rng.standard_normal(T_samp, dtype=np.float32)).astype(np.complex64)
        canvas += noise * np.float32(1 / math.sqrt(2)); del noise
        if rx["profile_kind"] != "flat":
            canvas = sm.apply_receiver(canvas, sm.receiver_response_taps(rx["profile"], 513))
        if rx["dc_spur_db"] > 0:
            f_dc = sm.add_dc_spur(canvas, rate_hz, F, rx["dc_spur_db"], rng)
            # the spur is labelled as a signal (one model bin wide, whole scene) -- "flag everything, filter later"
            b = Burst("DC_spur", "rx_dc", 9000, f_dc, 0.0, T, max(2.0 * rate_hz / F, rate_hz / R.NFFT_MODEL), rx["dc_spur_db"], scenario)
            b.sample_start, b.sample_count, b.total_snr_db = 0, T_samp, rx["dc_spur_db"]
            placed.append(b)
    # v5 (2026-11-05): spur combs. A quiet band at high gain shows an irregular comb of faint persistent lines; on
    # real N320 captures M7-S/M8-S painted such tiles wholesale (they resemble the resolved-subcarrier comb of an OFDM
    # carrier at low sample rates, which IS labelled as one filled box). Irregular, stationary combs are therefore
    # generated explicitly and labelled line by line ("flag everything"), so the model has both classes.
    p_comb = 0.5 if scenario == "noise_only" else 0.06
    if rng.random() < p_comb:
        lab_w = max(2.0 * rate_hz / F, rate_hz / R.NFFT_MODEL)
        wide_bin = rate_hz / F; lines = []
        regular = rng.random() < 0.45
        if regular:
            # v6 (2026-11-06): REGULAR combs. The 900 MHz ISM N320 capture that every retrain still painted whole
            # carries 30 persistent lines on an exact 1.00 MHz grid (~4 dB, up to 10 dB): clock / reference
            # harmonics reaching the antenna. A regular grid of weak lines is the resolved-subcarrier signature the
            # models fill, and the irregular combs below never showed it with line-by-line labels.
            spacing = float(np.clip(logu(rng, 0.2e6, 2.5e6), max(4 * wide_bin, 0.96 * rate_hz / 60), 0.96 * rate_hz / 4))
            off = float(rng.uniform(-0.5, 0.5)) * spacing
            fs_ = np.arange(-0.48 * rate_hz, 0.48 * rate_hz, spacing) + off
            fs_ = fs_[np.abs(fs_) < 0.48 * rate_hz]
            keep = rng.random(len(fs_)) > 0.1                           # a few teeth missing
            fs_ = fs_[keep]
            extra = rng.uniform(-0.48, 0.48, int(rng.integers(0, 4))) * rate_hz   # plus a few unrelated spurs
            fs_ = np.sort(np.r_[fs_, extra])
            base = float(rng.uniform(3, 9)); ripple = float(rng.uniform(0, 4)); cyc = float(rng.uniform(0.5, 3)); ph = float(rng.uniform(0, 2 * np.pi))
            n_ = max(len(fs_), 1)
            lvls = np.clip(base + ripple * np.sin(2 * np.pi * cyc * np.arange(n_) / n_ + ph) + rng.normal(0, 1.5, n_), 2.0, 25.0)
            drift_all = float(rng.uniform(-1e3, 1e3)) if rng.random() < 0.2 else 0.0
        else:
            n_lines = int(rng.integers(5, 61))
            fs_ = np.sort(rng.uniform(-0.48, 0.48, n_lines) * rate_hz)
            fs_ = fs_[np.r_[True, np.diff(fs_) > 3 * wide_bin]]       # keep lines >= 3 wide bins apart
            base = float(rng.uniform(3, 12)); lvls = None; drift_all = 0.0
        for k, f_ in enumerate(fs_):
            lvl = float(lvls[k]) if lvls is not None else float(np.clip(rng.normal(base, 5.0), 2.0, 30.0))
            drift = drift_all if regular else (float(rng.uniform(-2e3, 2e3)) if rng.random() < 0.3 else 0.0)
            gate = None if regular else ((logu(rng, 10e-3, 0.5), logu(rng, 10e-3, 0.5)) if rng.random() < 0.25 else None)
            lines.append((float(f_), lvl, drift, gate))
            b = Burst("RX_spur", "spur_comb", 9200 + k, float(f_), 0.0, T, max(lab_w, abs(drift) + wide_bin), lvl, scenario)
            b.sample_start, b.sample_count, b.total_snr_db = 0, T_samp, lvl; placed.append(b)
        sm.add_cw_lines(canvas, rate_hz, F, lines, rng)
        rx["spur_comb"] = len(lines); rx["spur_comb_regular"] = bool(regular)
    canvas *= np.float32(10 ** (rx["level_db"] / 20))
    # v4 (2026-11-05): live-ingest artefacts. M7-S paints 24-54 % of a frame under 30 % zero-filled packet loss and
    # 16-35 % under a 20 dB mid-frame gain step (test_1 experiments); none of this was in the training data.
    ingest = {"zero_ranges": []}
    u = rng.random()
    if u < 0.10:                                           # random CHDR packet (1024-sample) zero-fill over a window
        loss = logu(rng, 0.005, 0.40)
        w0 = int(rng.integers(0, T_samp)); w1 = min(T_samp, w0 + int(logu(rng, 5e-3, 0.5) * rate_hz)) if rng.random() < 0.6 else T_samp
        if rng.random() < 0.4: w0 = 0
        n_pk = (w1 - w0) // 1024
        drop = np.flatnonzero(rng.random(n_pk) < loss)
        for k in drop:
            canvas[w0 + k * 1024: w0 + (k + 1) * 1024] = 0
        ingest["zero_packet_loss"] = float(loss); ingest["zero_window"] = [int(w0), int(w1)]
    elif u < 0.15:                                         # contiguous outage (zeros) of 2-128 wide rows -> GT cleared there
        n_rows = int(logu(rng, 2, 128)); r0 = int(rng.integers(0, max(1, T_samp // F - n_rows)))
        s0, s1 = r0 * F, (r0 + n_rows) * F
        canvas[s0:s1] = 0; ingest["zero_ranges"].append([int(s0), int(s1)])
    # v7 (2026-11-06): the gain step is back. v6 (no gain step) made M11-S/M12-S paint whole tiles around band-wide
    # burst rows (narrow 850 MHz uplink look) and the DC phase-noise pedestal of a quiet band -- the gain-step
    # negatives were what taught M10-S "an elevated floor region without structure is background". Policy: that
    # lesson is worth more than the (rare) band-filling TDD carrier at a narrow look it makes ambiguous.
    if rng.random() < 0.05:                                # gain step mid-scene (AGC / manual gain change)
        g_db = float(rng.choice([-1, 1])) * float(rng.uniform(6, 25)); k0 = int(rng.integers(int(0.1 * T_samp), int(0.9 * T_samp)))
        canvas[k0:] *= np.float32(10 ** (g_db / 20)); ingest["gain_step_db"] = g_db; ingest["gain_step_sample"] = k0
    if rx["clip_over_db"] is not None and placed:
        peak = float(np.max(np.maximum(np.abs(canvas.real), np.abs(canvas.imag))))
        g = (10 ** (rx["clip_over_db"] / 20)) / max(peak, 1e-12)
        canvas *= np.float32(g)
        np.clip(canvas.real, -1, 1, out=canvas.real); np.clip(canvas.imag, -1, 1, out=canvas.imag)
    meta = {"scenario": scenario, "rate_hz": rate_hz, "n_frames": n_frames, "seed": seed, "fft_size": F,
            "frame_samples": R.ROWS_WIDE * F, "duration_s": T, "n_bursts": len(placed), "n_jobs": len(jobs),
            "receiver": {k: (v if k != "profile" else None) for k, v in rx.items()}, "gen_seconds": time.time() - t0,
            "signal_seconds": t_sig, "ingest": ingest}
    if verbose:
        print(f"[scene] {scenario} @ {rate_hz/1e6:.2f} MSps: {len(jobs)} jobs -> {len(placed)} bursts, "
              f"{T*1e3:.1f} ms, gen {meta['gen_seconds']:.1f}s (signals {t_sig:.1f}s), rx {rx['profile_kind']} dc {rx['dc_spur_db']:.0f} clip {rx['clip_over_db']}")
    return Scene(scenario, rate_hz, n_frames, seed, canvas, placed, meta)


def annotations_of(scene: Scene) -> list[dict]:
    out = []
    for b in scene.bursts:
        if b.sample_count <= 0:
            continue
        out.append({"core:sample_start": int(b.sample_start), "core:sample_count": int(b.sample_count),
                    "core:freq_lower_edge": float(b.fc_hz - b.bw_hz / 2), "core:freq_upper_edge": float(b.fc_hz + b.bw_hz / 2),
                    "core:label": b.label, "wfgt:kind": "waveform", "wfgt:occupied_bw_hz": float(b.bw_hz),
                    "wfgt:power_db": float(b.total_snr_db), "wfgt:snr_psd_db": float(b.snr_psd_db),
                    "wfgt:emitter": b.emitter, "wfgt:emitter_id": int(b.emitter_id), "wfgt:scenario": scene.scenario})
    return out


def write_sigmf(scene: Scene, stem: Path, center_hz: float = 0.0) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    # NOTE: not Path.with_suffix -- a stem like "..._491.52M_s3" would lose ".52M_s3" and collide
    data_p = stem.parent / (stem.name + ".sigmf-data"); meta_p = stem.parent / (stem.name + ".sigmf-meta")
    scene.canvas.astype(np.complex64).tofile(str(data_p))
    meta = {"global": {"core:datatype": "cf32_le", "core:sample_rate": float(scene.rate_hz), "core:version": "1.0.0",
                       "core:num_channels": 1, "core:description": f"synthetic scene {scene.scenario} seed {scene.seed} (gen_scenes.py)",
                       "wfgt:scenario": scene.scenario, "wfgt:seed": scene.seed, "wfgt:fft_size": scene.fft_size,
                       "wfgt:frame_samples": scene.frame_samples, "wfgt:receiver": scene.meta.get("receiver"),
                       "wfgt:ingest": scene.meta.get("ingest")},
            "captures": [{"core:sample_start": 0, "core:frequency": center_hz}],
            "annotations": annotations_of(scene)}
    meta_p.write_text(json.dumps(meta))


# --------------------------------------------------------------------------- preview
def preview_png(scene: Scene, out_png: Path, frames: int | None = None, device: str = "cuda", with_gt=True):
    import torch, frontend as fe
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cfg = fe.FrontEndCfg(); F = scene.fft_size; FS = scene.frame_samples
    nfr = frames or min(scene.n_frames, 2)
    anns = annotations_of(scene)
    fig, axes = plt.subplots(nfr, 1, figsize=(14, 4.2 * nfr), squeeze=False)
    for i in range(nfr):
        fr = torch.from_numpy(scene.canvas[i * FS:(i + 1) * FS]).to(device).unsqueeze(0)
        out = fe.compute_front_end(fr, scene.rate_hz, cfg, fft_size=F)
        img = out.resized[0].clamp(0, 1).cpu().numpy()
        gt = fe.rasterize_gt(anns, i * FS, F, R.ROWS_WIDE, cfg.nfft, scene.rate_hz).numpy()
        ax = axes[i, 0]
        ax.imshow(img, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1,
                  extent=[-scene.rate_hz / 2e6, scene.rate_hz / 2e6, 0, R.ROWS_WIDE * F / scene.rate_hz * 1e3])
        if with_gt:
            ax.contour(np.linspace(-scene.rate_hz / 2e6, scene.rate_hz / 2e6, cfg.nfft), np.linspace(0, R.ROWS_WIDE * F / scene.rate_hz * 1e3, R.ROWS_WIDE),
                       gt, levels=[0.5], colors="red", linewidths=0.6)
        ax.set_title(f"{scene.scenario} @ {scene.rate_hz/1e6:.2f} MSps  frame {i}  (robust-normalized model input, GT in red; occ {100*gt.mean():.1f}%)")
        ax.set_xlabel("MHz"); ax.set_ylabel("ms")
    fig.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out_png, dpi=90); plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="ism_collisions", choices=SCENARIOS)
    ap.add_argument("--rate", type=float, default=491.52e6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--png", default=None); ap.add_argument("--sigmf", default=None)
    a = ap.parse_args()
    sc = make_scene(a.scenario, a.rate, a.seed, a.frames, verbose=True)
    if a.sigmf: write_sigmf(sc, Path(a.sigmf))
    if a.png: preview_png(sc, Path(a.png))
