#!/usr/bin/env python3
"""Real-time decode stage: consume signal_snipper SigMF output, classify each
sub-band with three AMC models (VT-CNN2 / ResNet1D / T-PRIME), let the gate
model route the decoder, and publish live BER + classification metrics.

    detector -> signal_snipper -> sigmf_file_sink --(SigMF packs on disk)-->
        THIS DAEMON: find_subbands -> channelize -> AMC (all 3 models) ->
        gate model routes: NOISE->skip, FSK->discriminator, OFDM->OFDM chain,
        PSK/QAM->linear -> frame sync/header/CRC32 -> PN9 BER

Metrics (rt_metrics.json, atomic): the classic decode aggregates, plus
  - per-model classification accuracy vs truth (--truth-meta) and latency
  - ber_attempted: bit errors / bits over frames we actually decoded
  - ber_whole:     lost bits count 100% wrong — expected bits come from the
                   composite TX annotations (frames = placement//entry length)

Without AMC weights (or torch), falls back to the blind family cascade.

Usage:
    .venv-ml/bin/python rt_decode_daemon.py --snips <dir> [--poll 0.5] [--once]
        [--idle-exit 30] [--metrics-out <json>] [--no-amc] [--gate tprime]
        [--truth-meta <composite.sigmf-meta>] [--truth-lib <library root>]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

PYCODEC_ROOT = os.environ.get(
    "PYCODEC_ROOT", os.path.expanduser("~/Documents/holoscan_waveform_generation"))
sys.path.insert(0, PYCODEC_ROOT)

from pycodec.frame import (channelize, decode_frames,  # noqa: E402
                           estimate_symbol_rate, find_subbands)
from pycodec.fsk import (decode_frames_fsk, envelope_cv,  # noqa: E402
                         estimate_symbol_rate_fsk)
from pycodec.ofdm import decode_frames_ofdm, snap_profile_rate  # noqa: E402

PROFILE = dict(sps=8, pulse_shape="rrc", rolloff=0.35, span_symbols=10)
FSK_PROFILE = dict(sps=8, h=0.5, bt=0.5)

# Frequency-plan truth for the staircase-family replay captures (loop-invariant:
# no time alignment needed). Center -> family, plus the burst geometry
# (period samples @245.76M, payload bits per period) for whole-band BER:
# a band that is misclassified / misrouted / undecoded gets its expected
# bits charged at 100%.
# (center, family, burst period samples @245.76M, payload bits/period, symbol rate)
BAND_TRUTH_PLAN = [(-60e6, "PSK", 119616, 4096, 15.36e6),
                   (-20e6, "FSK", 150976, 1024, 1.92e6),
                   (0.0, "QAM", 86784, 8192, 15.36e6),
                   (60e6, "OFDM", 109056, 8192, 30.72e6)]
BAND_TRUTH_TOL_HZ = 12e6
BAND_TRUTH_FS = 245.76e6


def band_truth(f_center: float):
    """(family, burst period seconds, bits per period, symbol rate) from the
    fixed staircase plan, or (None, ...) off-plan (NOISE truth)."""
    for f, fam, period, pbits, rs in BAND_TRUTH_PLAN:
        if abs(f_center - f) <= BAND_TRUTH_TOL_HZ:
            return fam, period / BAND_TRUTH_FS, pbits, rs
    return None, 0.0, 0, 0.0


def decode_band_oracle(ch, chfs, fam, rs):
    """Oracle receiver: correct branch AND known symbol rate (blind rate
    estimation is unreliable on short single-burst boxes, and that is not a
    channel impairment)."""
    if fam == "FSK":
        return decode_frames_fsk(ch, chfs, rs, **FSK_PROFILE)
    if fam == "OFDM":
        frames = decode_frames_ofdm(ch, chfs, rs)
        for f in frames:
            f.payload_mod = f"OFDM-{f.payload_mod}"
        return frames
    return decode_frames(ch, chfs, rs, **PROFILE)


def family_of(mod: str) -> str:
    """Map a modulation / entry class tag to its AMC family."""
    m = (mod or "").upper()
    if m.startswith("OFDM"):
        return "OFDM"
    if m.endswith("FSK") or m == "GFSK":
        return "FSK"
    if "QAM" in m:
        return "QAM"
    if m in ("BPSK", "QPSK", "8PSK") or "PSK" in m:
        return "PSK"
    return "NOISE"


class TruthScorer:
    """Ground truth from the composite TX annotations + waveform library:
    per-band family labels (classification accuracy) and expected PN9 bits
    (whole-BER denominator: a bit never decoded counts 100% wrong)."""

    def __init__(self, meta_path: str, lib_root: str):
        meta = json.load(open(meta_path))
        self.placements = []
        self.sync_regions = []   # ZC sync + metadata bursts: excluded from scoring
        for a in meta.get("annotations", []):
            if a.get("wfgt:kind") in ("zadoff_chu", "metadata"):
                self.sync_regions.append(dict(
                    start=int(a["core:sample_start"]), count=int(a["core:sample_count"]),
                    f_lo=float(a.get("core:freq_lower_edge", -1e12)),
                    f_hi=float(a.get("core:freq_upper_edge", 1e12))))
                continue
            if a.get("wfgt:kind") != "waveform":
                continue
            if "wfgt:source_mat" in a:   # composer placements: look up the library entry
                src = a["wfgt:source_mat"]
                j = json.load(open(os.path.join(lib_root, src.replace(".mat", ".json"))))
                frame_bits = int((j.get("pycodecFrame") or {}).get("payload_len_bits", 0))
                entry_len = int(a.get("wfgt:original_length_samples")
                                or j["numOutputSamples"])
                pn9 = "framedtext" not in j.get("waveformName", "")
            else:   # synthetic captures (SNR staircase) carry the frame geometry inline
                frame_bits = int(a.get("wfgt:frame_payload_bits", 0))
                entry_len = int(a.get("wfgt:frame_len_samples", 0)) or 1
                pn9 = bool(a.get("wfgt:pn9", True))
            self.placements.append(dict(
                start=int(a["core:sample_start"]), count=int(a["core:sample_count"]),
                f_lo=float(a["core:freq_lower_edge"]),
                f_hi=float(a["core:freq_upper_edge"]),
                family=family_of(a.get("wfgt:class") or a.get("wfgt:modulation")),
                frames=int(a["core:sample_count"]) // entry_len,
                frame_bits=frame_bits,
                label=str(a.get("core:label", "?")),
                pn9=pn9))
        self.bits_expected = sum(p["frames"] * p["frame_bits"]
                                 for p in self.placements if p["pn9"])
        self.frames_expected = sum(p["frames"] for p in self.placements if p["pn9"])
        self.by_family_expected: dict[str, int] = {}
        for p in self.placements:
            if p["pn9"]:
                self.by_family_expected[p["family"]] = \
                    self.by_family_expected.get(p["family"], 0) + p["frames"] * p["frame_bits"]

    def lookup(self, orig_start: int, orig_count: int, f_center: float,
               margin_hz: float = 3e6) -> tuple[str, int | None]:
        """(truth family, placement index) for a band; family SYNC (excluded
        from scoring) when the band only matches a composer sync/metadata
        burst. Exact frequency containment beats margin matches so neighbor
        slots don't steal bands."""
        for margin in (0.0, margin_hz):
            best, best_ov = None, 0
            for i, p in enumerate(self.placements):
                ov = min(orig_start + orig_count, p["start"] + p["count"]) - \
                    max(orig_start, p["start"])
                if ov <= 0:
                    continue
                if f_center < p["f_lo"] - margin or f_center > p["f_hi"] + margin:
                    continue
                if ov > best_ov:
                    best, best_ov = (i, p), ov
            if best is not None:
                return best[1]["family"], best[0]
        for r in self.sync_regions:
            if (min(orig_start + orig_count, r["start"] + r["count"]) >
                    max(orig_start, r["start"]) and
                    r["f_lo"] - margin_hz <= f_center <= r["f_hi"] + margin_hz):
                return "SYNC", None
        return "NOISE", None


def load_classifier(weights: str | None, device: str | None, gate: str):
    try:
        from amc.classify import AmcClassifier
        clf = AmcClassifier(weights_dir=weights, device=device, gate=gate)
        print(f"amc: 3-model classifier ready on {clf.device}, gate={gate}", flush=True)
        return clf
    except Exception as e:
        print(f"amc: classifier unavailable ({e}) — falling back to blind cascade",
              flush=True)
        return None


def decode_band(ch, chfs, bw):
    """Legacy blind family cascade (no classifier): constant envelope -> FSK;
    otherwise linear framed; if that finds nothing, OFDM at the snapped rate."""
    if envelope_cv(ch) < 0.15:
        rs = estimate_symbol_rate_fsk(ch, chfs, lo_hz=max(0.2e6, 0.1 * bw),
                                      hi_hz=max(1e6, min(bw, 0.45 * chfs)))
        return decode_frames_fsk(ch, chfs, rs, **FSK_PROFILE), f"fsk/rs{rs/1e6:.2f}"
    rs = estimate_symbol_rate(ch, chfs, lo_hz=max(1e6, 0.3 * bw / 1.35),
                              hi_hz=max(2e6, min(1.2 * bw, 0.45 * chfs)))
    frames = decode_frames(ch, chfs, rs, **PROFILE)
    if frames:
        return frames, f"lin/rs{rs/1e6:.2f}"
    frames = decode_frames_ofdm(ch, chfs, snap_profile_rate(bw))
    for f in frames:
        f.payload_mod = f"OFDM-{f.payload_mod}"
    return frames, f"ofdm/fs{snap_profile_rate(bw)/1e6:.2f}"


def decode_band_routed(ch, chfs, bw, family):
    """Classifier-informed routing: the predicted family picks the decode
    branch outright (no fallback — a wrong route honestly loses those bits,
    which is exactly what ber_whole is for)."""
    if family == "NOISE":
        return [], "amc:skip"
    if family == "FSK":
        rs = estimate_symbol_rate_fsk(ch, chfs, lo_hz=max(0.2e6, 0.1 * bw),
                                      hi_hz=max(1e6, min(bw, 0.45 * chfs)))
        return decode_frames_fsk(ch, chfs, rs, **FSK_PROFILE), f"amc-fsk/rs{rs/1e6:.2f}"
    if family == "OFDM":
        frames = decode_frames_ofdm(ch, chfs, snap_profile_rate(bw))
        for f in frames:
            f.payload_mod = f"OFDM-{f.payload_mod}"
        return frames, f"amc-ofdm/fs{snap_profile_rate(bw)/1e6:.2f}"
    rs = estimate_symbol_rate(ch, chfs, lo_hz=max(1e6, 0.3 * bw / 1.35),
                              hi_hz=max(2e6, min(1.2 * bw, 0.45 * chfs)))
    return decode_frames(ch, chfs, rs, **PROFILE), f"amc-lin/rs{rs/1e6:.2f}"


class Metrics:
    def __init__(self, truth: TruthScorer | None = None, gate: str = "tprime"):
        self.snips = 0
        self.snips_with_frames = 0
        self.frames = 0
        self.crc_ok = 0
        self.pn9_bits = 0
        self.pn9_errors = 0
        self.by_mod: dict[str, int] = {}
        self.by_family: dict[str, dict] = {}
        self.started = time.time()
        self.recent: list[dict] = []
        self.last_payload_text = ""
        self.truth = truth
        self.gate = gate
        self.cls: dict[str, dict] = {}
        self.by_placement: dict[int, dict] = {}
        self.snr_bucket = "clean"          # DEMO CONTROLS snr selection
        self.by_snr: dict[str, dict] = {}  # bucket -> per-class gate acc + BER
        self.data_bytes = 0                # snippet bytes written to disk
        self.data_files = 0
        self.data_snips = 0
        self.stream_rate_hz = 245.76e6     # for the full-capture baseline
        self.comp_codec = None             # snippet_compression codec (from pack annotations)
        self.comp_logical = 0              # decompressed cf32 bytes represented
        self.comp_stored = 0               # bytes actually stored

    def note_decode(self, f_hz: float, mod: str, crc_ok: bool):
        self.recent.append({"f_hz": f_hz, "mod": mod, "crc_ok": bool(crc_ok), "t": time.time()})
        self.recent = self.recent[-16:]

    def note_frame_bits(self, mod: str, bits: int, errors: int,
                        pidx: int | None = None):
        fam = family_of(mod)
        d = self.by_family.setdefault(fam, {"bits": 0, "errors": 0})
        d["bits"] += bits
        d["errors"] += errors
        if pidx is not None:
            p = self.by_placement.setdefault(pidx, {"bits": 0, "errors": 0, "frames": 0})
            p["bits"] += bits
            p["errors"] += errors
            p["frames"] += 1

    def _bucket(self) -> dict:
        return self.by_snr.setdefault(self.snr_bucket,
                                      {"acc": {}, "per_gate": {}, "bytes": 0,
                                       "files": 0, "snips": 0})

    def note_data(self, new_bytes: int, new_files: int = 0, new_snips: int = 0):
        """Snippet-sink output accounting (bytes/files/snippets), attributed
        to the active SNR selection — the data-reduction story."""
        b = self._bucket()
        b["bytes"] += new_bytes
        b["files"] += new_files
        b["snips"] += new_snips
        self.data_bytes += new_bytes
        self.data_files += new_files
        self.data_snips += new_snips

    def note_compression(self, codec: str, logical_bytes: int, stored_bytes: int):
        """Codec economics per processed annotation: logical (decompressed cf32)
        vs stored bytes. cf32_le members of a compressed container count too
        (ratio-1 contributions keep the aggregate honest)."""
        self.comp_codec = codec
        self.comp_logical += logical_bytes
        self.comp_stored += stored_bytes

    def note_band_snr(self, truth, labels: dict, gate: str,
                      bits: int, errors: int, nframes: int, wexp: int = 0):
        """Per-SNR footer stats. Classification accuracy accumulates for ALL
        models simultaneously (each classifies every band); decode stats
        (BER/frames) accumulate under the ACTIVE gate, since routing decides
        what decodes — dwell at an SNR under each gate to compare."""
        b = self._bucket()
        gstats = b["per_gate"].setdefault(gate, {"bits": 0, "errors": 0, "frames": 0,
                                                 "wbits": 0, "werr": 0})
        gstats["bits"] += bits
        gstats["errors"] += errors
        gstats["frames"] += nframes
        if wexp > 0:   # whole-band BER: undelivered bits count 100% wrong
            gstats["wbits"] += wexp
            gstats["werr"] += errors + max(0, wexp - bits)
        if truth not in (None, "SYNC"):
            for name, label in labels.items():
                acc = b["acc"].setdefault(name, [0, 0])
                acc[1] += 1
                if label == truth:
                    acc[0] += 1

    def note_band_channel(self, bits: int, errors: int, wexp: int):
        """Oracle-routed decode of the same band (correct branch regardless of
        the gate): gate-independent channel/modem whole-BER. Classifier cost
        = wBER(gate) - chBER, visible directly on the table."""
        b = self._bucket()
        c = b.setdefault("channel", {"wbits": 0, "werr": 0})
        c["wbits"] += wexp
        c["werr"] += errors + max(0, wexp - bits)

    def note_cls(self, name: str, label: str, ms: float, truth_fam: str | None):
        c = self.cls.setdefault(name, {"n": 0, "ms_sum": 0.0, "truth_n": 0,
                                       "correct": 0, "preds": {}, "confusion": {}})
        c["n"] += 1
        c["ms_sum"] += ms
        c["preds"][label] = c["preds"].get(label, 0) + 1
        # SYNC bands (composer ZC/metadata bursts) are neither of the 4
        # families nor noise — the classifier never saw them, so skip scoring.
        if truth_fam is not None and truth_fam != "SYNC":
            c["truth_n"] += 1
            if label == truth_fam:
                c["correct"] += 1
            key = f"{truth_fam}>{label}"
            c["confusion"][key] = c["confusion"].get(key, 0) + 1

    def as_dict(self):
        d = {
            "snippets_seen": self.snips,
            "snippets_with_frames": self.snips_with_frames,
            "frames_decoded": self.frames,
            "frames_crc_ok": self.crc_ok,
            "pn9_bits_compared": self.pn9_bits,
            "pn9_bit_errors": self.pn9_errors,
            "pn9_ber": (self.pn9_errors / self.pn9_bits) if self.pn9_bits else None,
            "ber_attempted": (self.pn9_errors / self.pn9_bits) if self.pn9_bits else None,
            "frames_by_modulation": self.by_mod,
            "uptime_s": round(time.time() - self.started, 1),
            "recent_decodes": [{"f_hz": r["f_hz"], "mod": r["mod"], "crc_ok": r["crc_ok"],
                                "age_s": round(time.time() - r["t"], 1)}
                               for r in self.recent if time.time() - r["t"] < 30.0],
            "last_payload_text": self.last_payload_text,
        }
        if self.cls:
            d["classifier"] = {"gate": self.gate, "models": {
                name: {"n": c["n"],
                       "acc": (c["correct"] / c["truth_n"]) if c["truth_n"] else None,
                       "avg_ms": round(c["ms_sum"] / c["n"], 2) if c["n"] else None,
                       "preds": c["preds"], "confusion": c["confusion"]}
                for name, c in self.cls.items()}}
        if self.truth is not None:
            exp = self.truth.bits_expected
            lost = max(0, exp - self.pn9_bits)
            d["bits_expected"] = exp
            d["bits_lost"] = lost
            d["frames_expected"] = self.truth.frames_expected
            d["ber_whole"] = ((self.pn9_errors + lost) / exp) if exp else None
            d["ber_by_family"] = {}
            for fam, fam_exp in self.truth.by_family_expected.items():
                got = self.by_family.get(fam, {"bits": 0, "errors": 0})
                fam_lost = max(0, fam_exp - got["bits"])
                d["ber_by_family"][fam] = {
                    "attempted": (got["errors"] / got["bits"]) if got["bits"] else None,
                    "whole": ((got["errors"] + fam_lost) / fam_exp) if fam_exp else None,
                    "bits_lost": fam_lost, "bits_expected": fam_exp}
            # per-label rollup (labels like "16QAM@12dB" give the staircase table)
            steps: dict[str, dict] = {}
            for i, p in enumerate(self.truth.placements):
                if not p["pn9"]:
                    continue
                s = steps.setdefault(p["label"], {"bits": 0, "errors": 0, "frames": 0,
                                                  "bits_expected": 0})
                got = self.by_placement.get(i, {"bits": 0, "errors": 0, "frames": 0})
                s["bits"] += got["bits"]
                s["errors"] += got["errors"]
                s["frames"] += got["frames"]
                s["bits_expected"] += p["frames"] * p["frame_bits"]
            d["ber_by_label"] = {
                lbl: {"attempted": (s["errors"] / s["bits"]) if s["bits"] else None,
                      "whole": ((s["errors"] + max(0, s["bits_expected"] - s["bits"]))
                                / s["bits_expected"]) if s["bits_expected"] else None,
                      "frames": s["frames"], "bits_expected": s["bits_expected"]}
                for lbl, s in steps.items()}
        if self.by_snr:
            d["by_snr"] = {}
            for lbl, b in self.by_snr.items():
                row = {"snips": b.get("snips", 0),
                       "files": b.get("files", 0),
                       "gb": round(b.get("bytes", 0) / 1e9, 6)}
                for name in ("vtcnn2", "resnet1d", "tprime"):
                    c = b.get("acc", {}).get(name)
                    row[f"acc_{name}"] = (c[0] / c[1]) if c and c[1] else None
                    g = b.get("per_gate", {}).get(name)
                    row[f"ber_{name}"] = ((g["errors"] / g["bits"])
                                          if g and g["bits"] else None)
                    row[f"wber_{name}"] = ((g["werr"] / g["wbits"])
                                           if g and g.get("wbits") else None)
                    row[f"frames_{name}"] = g["frames"] if g else 0
                ch = b.get("channel")
                row["chber"] = ((ch["werr"] / ch["wbits"])
                                if ch and ch["wbits"] else None)
                d["by_snr"][lbl] = row
        # data-reduction story: what the snipper stored vs capturing the full
        # stream (cf32) for the daemon's whole uptime
        d["data_saved_gb"] = round(self.data_bytes / 1e9, 3)
        d["data_files"] = self.data_files
        d["data_snips"] = self.data_snips
        d["full_capture_gb"] = round(
            (time.time() - self.started) * self.stream_rate_hz * 8.0 / 1e9, 2)
        if self.comp_codec is not None and self.comp_stored > 0:
            d["comp_codec"] = self.comp_codec
            d["comp_ratio"] = round(self.comp_logical / self.comp_stored, 3)
            d["comp_logical_gb"] = round(self.comp_logical / 1e9, 3)
            d["comp_stored_gb"] = round(self.comp_stored / 1e9, 3)
        return d


args_no_band_truth = False
args_center_hz = 2.4e9   # channel tune: live snips tag ABSOLUTE RF centers


def decode_snip_payload(codec: str, raw: np.ndarray, n_iq: int, a: dict) -> np.ndarray:
    """Dequantize one compressed snippet payload (uint8 array) back to complex64.
    Codecs mirror snippet_compression.cu: sc16 (per-snippet scale int16 I/Q),
    bfp8/bfp12 (per-block int8 power-of-2 exponent + two's-complement mantissas)."""
    if codec == "cf32_le":
        return raw[: n_iq * 8].view(np.complex64).copy()
    if codec == "sc16":
        scale = float(a.get("wfgt:comp_scale", 1.0))
        x = raw[: n_iq * 4].view("<i2").astype(np.float32) * scale
        return (x[0::2] + 1j * x[1::2]).astype(np.complex64)
    if codec in ("bfp8", "bfp12"):
        B = int(a.get("wfgt:comp_block", 64))
        block_bytes = 1 + (2 * B if codec == "bfp8" else 3 * B)
        nblocks = raw.size // block_bytes
        if nblocks == 0:
            return np.empty(0, np.complex64)
        m = raw[: nblocks * block_bytes].reshape(nblocks, block_bytes)
        e = m[:, 0].view(np.int8).astype(np.float32)
        if codec == "bfp8":
            scal = m[:, 1:].view(np.int8).astype(np.float32)
        else:
            p = m[:, 1:].reshape(nblocks, B, 3).astype(np.uint16)
            m0 = (p[..., 0] << 4) | (p[..., 1] >> 4)
            m1 = ((p[..., 1] & 0xF) << 8) | p[..., 2]
            pair = np.empty((nblocks, B, 2), np.int32)
            pair[..., 0] = ((m0.astype(np.int32) ^ 0x800) - 0x800)  # sign-extend 12-bit
            pair[..., 1] = ((m1.astype(np.int32) ^ 0x800) - 0x800)
            scal = pair.reshape(nblocks, 2 * B).astype(np.float32)
        flat = (scal * np.exp2(e)[:, None]).reshape(-1)[: 2 * n_iq]
        return (flat[0::2] + 1j * flat[1::2]).astype(np.complex64)
    return np.empty(0, np.complex64)  # unknown codec: skip rather than mis-decode


class PackReader:
    """Per-slice np.fromfile reads instead of np.memmap. The sigmf sink
    rewrites pack files in place as they accumulate snippets, which SIGBUSes
    a live memmap (the daemon died twice this way under the snip flood);
    plain reads just come back short, which the callers already tolerate."""

    def __init__(self, path):
        self.path = path
        if not os.path.exists(path):
            raise OSError(path)

    def size_bytes(self) -> int:
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    def __getitem__(self, sl):
        start = int(sl.start or 0)
        stop = int(sl.stop if sl.stop is not None else start)
        n = max(0, stop - start)
        try:
            return np.fromfile(self.path, dtype="<c8", count=n, offset=start * 8)
        except OSError:
            return np.empty(0, dtype="<c8")

    def read_compressed(self, a: dict) -> np.ndarray:
        """Read one compressed-container chunk by its wfgt:comp_byte_* range and
        dequantize to complex64 (see decode_snip_payload)."""
        codec = a.get("wfgt:compression", "cf32_le")
        off = int(a.get("wfgt:comp_byte_offset", 0))
        nb = int(a.get("wfgt:comp_byte_count", 0))
        n_iq = int(a.get("core:sample_count", 0))
        try:
            raw = np.fromfile(self.path, dtype=np.uint8, count=nb, offset=off)
        except OSError:
            return np.empty(0, np.complex64)
        if raw.size < nb:
            return np.empty(0, np.complex64)  # pack still flushing; caller skips
        return decode_snip_payload(codec, raw, n_iq, a)


def process_annotation(a, data, metrics: Metrics, clf=None) -> str:
    snip_fs = float(a.get("wfgt:snippet_sample_rate",
                          a.get("wfgt:orig_sample_rate", 245.76e6)))
    metrics.stream_rate_hz = max(metrics.stream_rate_hz,
                                 float(a.get("wfgt:orig_sample_rate", 0.0)))
    start, count = int(a["core:sample_start"]), int(a["core:sample_count"])
    if "wfgt:comp_byte_offset" in a and hasattr(data, "read_compressed"):
        # Compressed container: chunks are byte-addressed; core:sample_* stay logical.
        iq = data.read_compressed(a)
        metrics.note_compression(a.get("wfgt:compression", "cf32_le"),
                                 logical_bytes=count * 8,
                                 stored_bytes=int(a.get("wfgt:comp_byte_count", 0)))
    else:
        iq = np.asarray(data[start:start + count], dtype=np.complex64)
    if iq.size < 4096:
        return None
    t0 = time.time()
    snip_center = float(a.get("wfgt:center_frequency", 0.0))
    orig_start = int(a.get("wfgt:orig_sample_start", 0))
    orig_count = int(a.get("wfgt:orig_sample_count",
                           count * int(a.get("wfgt:decimation_factor", 1))))
    frames = []
    band_info = []
    try:
        for center, bw in find_subbands(iq, snip_fs):
            ch, chfs = channelize(iq, snip_fs, center, bw)
            if bw / chfs > 0.55:
                # tight per-signal snips (snipper decimates to ~1.5x occ) leave
                # <2.4 samples/symbol: the |iq|^2 rate line sits at/over the
                # search cap and RRC timing starves — give the band headroom
                from scipy.signal import resample_poly
                ch = resample_poly(ch, 2, 1).astype(np.complex64)
                chfs *= 2.0
            truth_fam, truth_idx = (metrics.truth.lookup(orig_start, orig_count,
                                                         snip_center + center)
                                    if metrics.truth else (None, None))
            if clf is not None:
                res = clf.classify(ch)
                pred = res[clf.gate].label
                got, how = decode_band_routed(ch, chfs, bw, pred)
                # truth priority: TX annotations (offline) > fixed frequency
                # plan (staircase-family replay, loop-invariant) >
                # decode-verified (>=3 CRC-ok frames prove the family)
                t = truth_fam
                wexp = 0
                plan_rs = 0.0
                if t is None and not args_no_band_truth:
                    # live snips tag absolute RF; the plan is baseband
                    f_band = snip_center + center
                    if abs(f_band) > 1e9:
                        f_band -= args_center_hz
                    plan_fam, period_s, pbits, plan_rs = band_truth(f_band)
                    if plan_fam is not None:
                        t = plan_fam
                        # charge whole burst periods; a box exists because a
                        # burst is there, so charge at least one
                        dur = iq.size / snip_fs
                        wexp = max(1, int(round(dur / period_s))) * pbits
                    else:
                        t = "NOISE"
                if t is None:
                    okf = [f for f in got if f.payload_crc_ok]
                    if len(okf) >= 3:
                        t = family_of(okf[0].payload_mod)
                for name, r in res.items():
                    metrics.note_cls(name, r.label, r.ms, t)
                mark = ("" if t is None else
                        "~sync" if t == "SYNC" else
                        "=" if pred == t else f"!={t}")
                how = f"{how}[{pred}{mark}]"
                metrics.note_band_snr(
                    t, {name: r.label for name, r in res.items()}, clf.gate,
                    sum(f.payload_len_bits for f in got if f.pn9_payload),
                    sum(f.bit_errors for f in got if f.pn9_payload),
                    len(got), wexp)
                if wexp > 0 and t in ("PSK", "QAM", "FSK", "OFDM"):
                    try:   # oracle: correct branch AND known symbol rate
                        cgot = decode_band_oracle(ch, chfs, t, plan_rs)
                    except Exception:
                        cgot = []
                    metrics.note_band_channel(
                        sum(f.payload_len_bits for f in cgot if f.pn9_payload),
                        sum(f.bit_errors for f in cgot if f.pn9_payload), wexp)
            else:
                got, how = decode_band(ch, chfs, bw)
            frames.extend(got)
            band_info.append(f"{center/1e6:+.1f}MHz/{how}:{len(got)}")
            for f in got:
                metrics.note_decode(snip_center + center, f.payload_mod, f.payload_crc_ok)
                if f.pn9_payload:
                    metrics.note_frame_bits(f.payload_mod, f.payload_len_bits,
                                            f.bit_errors, truth_idx)
                if f.payload_crc_ok and not f.pn9_payload and f.payload_bits is not None:
                    from pycodec.pn9 import bytes_from_bits
                    raw = bytes_from_bits(f.payload_bits)
                    text = "".join(chr(b) if 32 <= b < 127 else "" for b in raw)[:160]
                    if len(text) > 8:
                        metrics.last_payload_text = text
    except Exception as e:
        return f"decode error: {e}"
    dt_ms = (time.time() - t0) * 1e3

    metrics.snips += 1
    if not frames:
        return (f"snip frame#{a.get('wfgt:frame_number', '?')} "
                f"{count/snip_fs*1e3:6.2f} ms  no frames (bands {' '.join(band_info) or '-'})  [{dt_ms:.0f} ms]")
    metrics.snips_with_frames += 1
    ok = [f for f in frames if f.payload_crc_ok]
    err = sum(f.bit_errors for f in frames if f.pn9_payload)
    bits = sum(f.payload_len_bits for f in frames if f.pn9_payload)
    metrics.frames += len(frames)
    metrics.crc_ok += len(ok)
    metrics.pn9_bits += bits
    metrics.pn9_errors += err
    for f in frames:
        metrics.by_mod[f.payload_mod] = metrics.by_mod.get(f.payload_mod, 0) + 1
    mods = ",".join(sorted({f.payload_mod for f in frames}))
    ber = f"{err}/{bits}" if bits else "-"
    return (f"snip frame#{a.get('wfgt:frame_number', '?')} {count/snip_fs*1e3:6.2f} ms  "
            f"bands[{' '.join(band_info)}]  frames {len(frames)} (crc_ok {len(ok)}) "
            f"mod {mods}  pn9_err {ber}  [{dt_ms:.0f} ms]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snips", required=True)
    ap.add_argument("--poll", type=float, default=0.5)
    ap.add_argument("--once", action="store_true", help="process what exists, then exit")
    ap.add_argument("--idle-exit", type=float, default=None,
                    help="exit after this many seconds without new snippets")
    ap.add_argument("--metrics-out", default=None,
                    help="where to write rt_metrics.json (default: inside --snips); "
                         "the HoloViz LIVE DECODE panel watches this path")
    ap.add_argument("--no-amc", action="store_true", help="disable the classifier stage")
    ap.add_argument("--amc-weights", default=None)
    ap.add_argument("--amc-device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--gate", default="tprime", choices=["tprime", "resnet1d", "vtcnn2"],
                    help="which model's prediction routes the decoder")
    ap.add_argument("--control-json", default=None,
                    help="demo_control.json written by the dashboard's DEMO CONTROLS "
                         "panel; the 'gate' field is applied live "
                         "(default: demo_control.json next to --metrics-out)")
    ap.add_argument("--truth-meta", default=None,
                    help="composite TX .sigmf-meta for classification accuracy + whole BER")
    ap.add_argument("--truth-lib", default=os.path.join(PYCODEC_ROOT,
                                                        "generated_waveforms_framed"),
                    help="waveform library root (entry jsons) for expected-bits math")
    ap.add_argument("--center-hz", type=float, default=2.4e9,
                    help="channel tune frequency (converts live snips' absolute "
                         "RF band centers to baseband for the frequency-plan truth)")
    ap.add_argument("--no-band-truth", action="store_true",
                    help="disable the fixed staircase frequency-plan truth "
                         "(use when replaying content with a different plan)")
    args = ap.parse_args()
    metrics_path = args.metrics_out or os.path.join(args.snips, "rt_metrics.json")
    control_path = args.control_json or os.path.join(os.path.dirname(metrics_path),
                                                     "demo_control.json")

    global args_no_band_truth, args_center_hz
    args_no_band_truth = args.no_band_truth
    args_center_hz = args.center_hz
    truth = TruthScorer(args.truth_meta, args.truth_lib) if args.truth_meta else None
    if truth:
        print(f"truth: {len(truth.placements)} placements, "
              f"{truth.frames_expected} PN9 frames / {truth.bits_expected} bits expected",
              flush=True)
    clf = None if args.no_amc else load_classifier(args.amc_weights, args.amc_device,
                                                   args.gate)
    metrics = Metrics(truth=truth, gate=args.gate)
    seen: dict[str, int] = {}   # pack meta path -> annotations processed
    pack_sizes: dict[str, int] = {}     # data path -> bytes seen (sink accounting)
    meta_mtimes: dict[str, float] = {}
    meta_counts: dict[str, int] = {}

    last_scan = [0.0]
    primed = [False]

    def scan_data_stats():
        """Account every byte/file/snippet the sink writes, independent of
        what the decoder gets to; prune tracking for janitored files.
        Time-throttled and called per-annotation: one pack can take a minute
        to decode, and bytes written meanwhile must land in the SNR bucket
        that was active when they were written, not in a later lump."""
        if time.time() - last_scan[0] < 1.0:
            return
        last_scan[0] = time.time()
        # first scan only PRIMES the baselines: pre-existing packs (from an
        # earlier daemon run) must not lump into the current SNR bucket
        attribute = primed[0]
        live = set()
        for dp in glob.glob(os.path.join(args.snips, "*.sigmf-data")):
            live.add(dp)
            try:
                sz = os.path.getsize(dp)
            except OSError:
                continue
            prev = pack_sizes.get(dp)
            if prev is None:
                if attribute:
                    metrics.note_data(sz, new_files=1)
                pack_sizes[dp] = sz
            elif sz > prev:
                if attribute:
                    metrics.note_data(sz - prev)
                pack_sizes[dp] = sz
        for gone in set(pack_sizes) - live:
            del pack_sizes[gone]
        for mp2 in glob.glob(os.path.join(args.snips, "*.sigmf-meta")):
            try:
                mt2 = os.path.getmtime(mp2)
            except OSError:
                continue
            if meta_mtimes.get(mp2) == mt2:
                continue
            meta_mtimes[mp2] = mt2
            try:
                n_ann = len(json.load(open(mp2)).get("annotations", []))
            except (OSError, json.JSONDecodeError):
                continue
            if n_ann > meta_counts.get(mp2, 0):
                if attribute:
                    metrics.note_data(0, new_snips=n_ann - meta_counts.get(mp2, 0))
                meta_counts[mp2] = n_ann
        for gone in set(meta_mtimes) - set(glob.glob(os.path.join(args.snips, "*.sigmf-meta"))):
            meta_mtimes.pop(gone, None)
            meta_counts.pop(gone, None)
        primed[0] = True
    last_new = time.time()
    print(f"rt_decode_daemon: watching {args.snips} "
          f"({'AMC-routed' if clf else 'blind cascade'}, profile {PROFILE})", flush=True)
    control_state = {"mtime": 0.0}

    def check_control():
        """Apply DEMO CONTROLS gate switches. Called per-snippet: a live
        backlog can keep one outer loop iteration busy for minutes."""
        if clf is None:
            return
        try:
            mt = os.path.getmtime(control_path)
            if mt == control_state["mtime"]:
                return
            control_state["mtime"] = mt
            ctl = json.load(open(control_path))
            snr = str(ctl.get("snr", "clean"))
            if snr != metrics.snr_bucket:
                metrics.snr_bucket = snr
                print(f"[{time.strftime('%H:%M:%S')}] === DEMO CONTROL: snr bucket -> {snr} ===",
                      flush=True)
            gate = ctl.get("gate")
            if gate in clf.models and gate != clf.gate:
                clf.gate = gate
                metrics.gate = gate
                print(f"[{time.strftime('%H:%M:%S')}] === DEMO CONTROL: gate -> {gate} ===",
                      flush=True)
                try:  # reflect the '>' marker immediately, before new snips
                    tmp = metrics_path + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump(metrics.as_dict(), f, indent=1)
                    os.replace(tmp, metrics_path)
                except OSError:
                    pass
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    while True:
        check_control()
        scan_data_stats()
        new_work = False
        packs_this_pass = 0
        # newest first: under a live looped replay the daemon cannot drain the
        # backlog, so track NOW and let the janitor age out what we skip
        for mp in sorted(glob.glob(os.path.join(args.snips, "*.sigmf-meta")),
                         key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0,
                         reverse=True):
            try:
                meta = json.load(open(mp))
            except (json.JSONDecodeError, OSError):
                continue  # pack still being written
            anns = meta.get("annotations", [])
            done = seen.get(mp, 0)
            if len(anns) <= done:
                continue
            dp = mp.replace(".sigmf-meta", ".sigmf-data")
            try:
                data = PackReader(dp)
            except (OSError, ValueError):
                continue
            processed = done
            for a in anns[done:]:
                # pack data still being flushed: retry the remainder next pass
                if "wfgt:comp_byte_offset" in a:
                    end_bytes = int(a["wfgt:comp_byte_offset"]) + int(a.get("wfgt:comp_byte_count", 0))
                else:
                    end_bytes = (int(a.get("core:sample_start", 0)) +
                                 int(a.get("core:sample_count", 0))) * 8
                if end_bytes > data.size_bytes():
                    break
                check_control()
                scan_data_stats()
                line = process_annotation(a, data, metrics, clf=clf)
                processed += 1
                if line:
                    print(f"[{time.strftime('%H:%M:%S')}] {line}", flush=True)
                # stream metrics per snippet so the HoloViz panel updates live
                try:
                    tmp = metrics_path + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump(metrics.as_dict(), f, indent=1)
                    os.replace(tmp, metrics_path)
                except OSError:
                    pass
            seen[mp] = processed
            new_work = True
            packs_this_pass += 1
            if packs_this_pass >= 3:
                break   # re-glob so freshly written packs jump the queue
        if new_work:
            last_new = time.time()
            m = metrics.as_dict()
            ber = m["pn9_ber"]
            extra = ""
            if "ber_whole" in m and m["ber_whole"] is not None:
                extra = f" whole {m['ber_whole']:.2e} (lost {m['bits_lost']})"
            if "classifier" in m:
                accs = {n: (f"{v['acc']:.3f}" if v["acc"] is not None else "-")
                        for n, v in m["classifier"]["models"].items()}
                extra += f" cls_acc {accs}"
            print(f"[{time.strftime('%H:%M:%S')}] === LIVE: frames {m['frames_decoded']} "
                  f"(crc_ok {m['frames_crc_ok']}) by_mod {m['frames_by_modulation']} "
                  f"PN9 BER {ber if ber is None else f'{ber:.2e}'} "
                  f"({m['pn9_bit_errors']}/{m['pn9_bits_compared']}){extra} ===", flush=True)
            try:
                tmp = metrics_path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(m, f, indent=1)
                os.replace(tmp, metrics_path)  # atomic: the HoloViz panel polls this file
            except OSError:
                pass
        if args.once and not new_work:
            break
        if args.idle_exit and (time.time() - last_new) > args.idle_exit:
            print("idle timeout, exiting", flush=True)
            break
        time.sleep(args.poll)

    m = metrics.as_dict()
    print("\nFINAL:", json.dumps(m, indent=1), flush=True)
    return 0 if (m["frames_decoded"] > 0 and
                 m["frames_crc_ok"] == m["frames_decoded"] and
                 m["pn9_bit_errors"] == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
