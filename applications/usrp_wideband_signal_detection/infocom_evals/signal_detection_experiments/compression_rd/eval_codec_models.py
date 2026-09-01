#!/usr/bin/env python3
"""Codec-model matrix eval on the ORIGINAL MATLAB composite.

For each composite variant (clean + AWGN 30/15/0/-10 nominal) and each
detection lane:

  dino : DINO-FT (cuda_dino_finetuned) offline runs -> real pipeline snips
         (codec none), truth by exact sample/freq join against the variant's
         annotations (offline indices are file-exact).
  gt   : ground-truth snipping -- every TX annotation sliced from the file,
         mixed to baseband, decimated by the snipper's law (absolute-truth
         boxes; no detector in the loop).

then the 4x4 storage-codec x model matrix: each snip is stored under codec X
(numpy round trip, bit-consistent with the CUDA kernels), transcoded to model
Y's compressed-domain representation (matlab_ds.quantize_windows/featurize),
and classified by the tprime_<Y> model with median-softmax aggregation over
block-aligned windows.

Outputs: codec_model_matrix.csv (per lane/variant/class/actual-SNR-bin cell
counts) + detection_recall.csv (DINO-FT recall per class per variant).

Run AFTER training (amc/train_matlab.py) and make_noisy_composites.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
APP_DIR = THIS.parents[2]
PYCODEC_E2E = APP_DIR / "infocom_evals/pycodec_e2e"
sys.path.insert(0, str(PYCODEC_E2E))

from amc.matlab_ds import (BLOCK, CLASSES10, WIN, featurize, quantize_windows,  # noqa: E402
                           snip_decim)

COMPOSITES = Path.home() / "Documents/holoscan_waveform_generation/composition/composites"
SCRATCH = Path("/tmp/usrp_spectrograms/codec_model_eval")
VARIANTS = {"clean": "comprehensive_ordered",
            "30": "comprehensive_ordered_awgn30db",
            "15": "comprehensive_ordered_awgn15db",
            "0": "comprehensive_ordered_awgn0db",
            "-10": "comprehensive_ordered_awgnm10db"}
CODECS = ["none", "sc16", "bfp12", "bfp8"]
WINDOWS_PER_SNIP = 4
FS = 245.76e6


# ------------------------------------------------------------ codec helpers ----

def codec_roundtrip(w: np.ndarray, codec: str) -> np.ndarray:
    """Store-and-reconstruct (N, WIN) windows under `codec` (absolute scale)."""
    if codec == "none":
        return w
    if codec == "sc16":
        peak = np.abs(np.stack([w.real, w.imag])).max(axis=(0, 2))
        scale = np.where(peak > 0, peak / 32767.0, 1.0)[:, None]
        qi = np.clip(np.rint(w.real / scale), -32767, 32767)
        qq = np.clip(np.rint(w.imag / scale), -32767, 32767)
        return ((qi + 1j * qq) * scale).astype(np.complex64)
    mant_bits = 12 if codec == "bfp12" else 8
    qmax = float((1 << (mant_bits - 1)) - 1)
    n, win = w.shape
    nb = win // BLOCK
    r = w.reshape(n, nb, BLOCK)
    bm = np.maximum(np.abs(r.real).max(axis=2), np.abs(r.imag).max(axis=2))
    with np.errstate(divide="ignore"):
        e = np.clip(np.where(bm > 0, np.ceil(np.log2(np.maximum(bm, 1e-30) / qmax)), -127), -127, 127)
    scale = np.exp2(e)[:, :, None]
    mi = np.clip(np.rint(r.real / scale), -qmax, qmax)
    mq = np.clip(np.rint(r.imag / scale), -qmax, qmax)
    return ((mi + 1j * mq) * scale).reshape(n, win).astype(np.complex64)


def snip_windows(iq: np.ndarray) -> np.ndarray:
    """Block-aligned WIN windows from one snip payload (tile if short)."""
    if iq.size < WIN:
        reps = int(np.ceil(WIN / iq.size))
        return np.tile(iq, reps)[None, :WIN]
    starts = np.linspace(0, iq.size - WIN, min(WINDOWS_PER_SNIP, 1 + (iq.size - WIN) // WIN),
                         dtype=np.int64)
    starts = (starts // BLOCK) * BLOCK
    return np.stack([iq[s:s + WIN] for s in starts])


# ------------------------------------------------------------------- lanes ----

def dino_lane_snips(variant: str, stem: str):
    """Offline DINO-FT run (codec none) -> [(iq, truth_class, snr_db)]."""
    out_root = SCRATCH / f"dino_{variant}"
    snips_dir = out_root / "snippets"
    if not (snips_dir.exists() and any(snips_dir.glob("*.sigmf-meta"))):
        cfg = make_dino_config()
        print(f"[eval] DINO-FT offline run: {variant}", flush=True)
        subprocess.run([sys.executable, str(APP_DIR / "run_cuda_dino_offline_file.py"),
                        str(COMPOSITES / f"{stem}.sigmf-data"),
                        "--detector", "cuda_dino_finetuned", "--config", str(cfg),
                        "--snippets-only", "--output-root", str(out_root)],
                       check=True, cwd=APP_DIR, stdout=subprocess.DEVNULL,
                       stderr=subprocess.STDOUT)
    truth = load_truth(stem)
    out = []
    for mp in sorted(snips_dir.glob("*.sigmf-meta")):
        m = json.load(open(mp))
        raw = np.fromfile(str(mp).replace(".sigmf-meta", ".sigmf-data"), np.complex64)
        for a in m["annotations"]:
            s, n = int(a["core:sample_start"]), int(a["core:sample_count"])
            iq = raw[s:s + n]
            if iq.size < 1024:
                continue
            o0 = int(a.get("wfgt:orig_sample_start", 0))
            oc = int(a.get("wfgt:orig_sample_count", 0))
            fc = float(a.get("wfgt:center_frequency", 0.0))
            cls, snr = truth_join(truth, o0, oc, fc)
            out.append((iq, cls, snr))
    return out


def make_dino_config() -> Path:
    """DINO-FT offline base config with the snipper enabled and codec none."""
    src = (APP_DIR / "config_dino_finetuned_viz_demo.yaml").read_text()
    import re
    src = re.sub(r'(^\s*codec:\s*)"[a-z0-9]*"', r'\g<1>"none"', src, count=1, flags=re.M)
    src = re.sub(r'(^\s*control_json_path:\s*)"[^"]*"', r'\g<1>""', src, count=1, flags=re.M)
    src = re.sub(r"(^\s*pack_frames:\s*)\d+", r"\g<1>1", src, count=1, flags=re.M)
    out = APP_DIR / "generated_configs" / "codec_model_eval_dino.yaml"
    out.parent.mkdir(exist_ok=True)
    out.write_text(src)
    return out


def load_truth(stem: str):
    meta = json.load(open(COMPOSITES / f"{stem}.sigmf-meta"))
    rows = []
    for a in meta["annotations"]:
        if a.get("wfgt:kind") != "waveform":
            continue
        rows.append((int(a["core:sample_start"]),
                     int(a["core:sample_start"]) + int(a["core:sample_count"]),
                     float(a["core:freq_lower_edge"]), float(a["core:freq_upper_edge"]),
                     str(a.get("wfgt:class", "?")), float(a.get("wfgt:snr_db", np.nan)),
                     float(a.get("wfgt:occupied_bw_hz", 0.0))))
    return rows


def truth_join(rows, o0: int, oc: int, fc: float, margin=3e6):
    best, best_ov = ("NOISE", np.nan), 0
    for t0, t1, flo, fhi, cls, snr, _occ in rows:
        ov = min(o0 + oc, t1) - max(o0, t0)
        if ov <= 0 or fc < flo - margin or fc > fhi + margin:
            continue
        if ov > best_ov:
            best, best_ov = (cls, snr), ov
    return best


def gt_lane_snips(variant: str, stem: str, max_len=2_000_000):
    """Absolute-truth snipping: slice+mix+decimate every annotation."""
    from scipy.signal import resample_poly
    truth = load_truth(stem)
    data = np.memmap(COMPOSITES / f"{stem}.sigmf-data", dtype=np.complex64, mode="r")
    out = []
    for t0, t1, flo, fhi, cls, snr, occ in truth:
        n = min(t1 - t0, max_len)
        x = np.asarray(data[t0:t0 + n]).copy()
        fc = (flo + fhi) / 2
        x *= np.exp(-2j * np.pi * fc / FS * np.arange(n)).astype(np.complex64)
        d = snip_decim(occ if occ > 0 else (fhi - flo) / 1.35)
        if d > 1:
            x = resample_poly(x, 1, d).astype(np.complex64)
        if x.size >= 256:
            out.append((x.astype(np.complex64), cls, snr))
    return out


# ------------------------------------------------------------------- eval ----

def load_models(weights_dir: Path, device):
    import torch
    from amc.models import TPrimeC
    models = {}
    for codec in CODECS:
        net = TPrimeC(classes=len(CLASSES10))
        sd = torch.load(weights_dir / f"tprime_{codec}.pt", map_location="cpu")
        net.load_state_dict({k: v.float() for k, v in sd.items()})
        models[codec] = net.to(device).eval()
    return models


def main() -> int:
    import torch
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=str(PYCODEC_E2E / "amc/weights_matlab"))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--lanes", nargs="+", default=["dino", "gt"])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_models(Path(args.weights), device)
    SCRATCH.mkdir(parents=True, exist_ok=True)

    cells: dict[tuple, list[int]] = {}   # key -> [n, correct]
    recall_rows = []
    for variant in args.variants:
        stem = VARIANTS[variant]
        if not (COMPOSITES / f"{stem}.sigmf-data").exists():
            print(f"[eval] missing {stem}, skipping", flush=True)
            continue
        for lane in args.lanes:
            snips = (dino_lane_snips if lane == "dino" else gt_lane_snips)(variant, stem)
            print(f"[eval] {lane}/{variant}: {len(snips)} snips", flush=True)
            if lane == "dino":
                truth = load_truth(stem)
                covered = {}
                for t0, t1, flo, fhi, cls, _s, _o in truth:
                    covered.setdefault(cls, [0, 0])[0] += 1
                hit = {}
                for _iq, cls, _snr in snips:
                    if cls != "NOISE":
                        hit[cls] = hit.get(cls, 0) + 1
                for cls, (tot, _z) in covered.items():
                    recall_rows.append(dict(variant=variant, cls=cls, annotations=tot,
                                            snips_matched=hit.get(cls, 0)))
            for iq, cls, snr in snips:
                w = snip_windows(iq)
                ci = CLASSES10.index(cls) if cls in CLASSES10 else CLASSES10.index("NOISE")
                snr_bin = ("clean" if variant == "clean" or not np.isfinite(snr)
                           else str(int(np.clip(np.round(snr / 5) * 5, -20, 50))))
                for store in CODECS:
                    ws = codec_roundtrip(w, store)
                    for mdl in CODECS:
                        mant, e_rel = quantize_windows(ws, mdl)
                        x = torch.from_numpy(featurize(mant, e_rel)).to(device)
                        with torch.no_grad():
                            p = torch.softmax(models[mdl](x), dim=1).median(dim=0).values
                        pred = int(p.argmax())
                        key = (lane, variant, cls, snr_bin, store, mdl)
                        c = cells.setdefault(key, [0, 0])
                        c[0] += 1
                        c[1] += int(pred == ci)

    with open(THIS / "codec_model_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lane", "variant", "class", "snr_bin", "storage_codec", "model_codec",
                    "n", "correct"])
        for key, (n, c) in sorted(cells.items()):
            w.writerow(list(key) + [n, c])
    with open(THIS / "detection_recall.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "cls", "annotations", "snips_matched"])
        w.writeheader()
        w.writerows(recall_rows)
    print(f"[eval] wrote codec_model_matrix.csv ({len(cells)} cells) + detection_recall.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
