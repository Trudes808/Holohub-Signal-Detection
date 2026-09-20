#!/usr/bin/env python3
"""Ground-truth sweep of adaptive_span_db x adaptive_floor_frac for M3_491.

For each grid point, run the held-out SigMF composites through the front-end (with that span/floor_frac,
everything else at the deployed baseline), threshold at 0.6, and score the predicted [512,1024] mask
against a ground-truth mask built from the generator annotations. Reports pixel Precision / Recall / F1 /
IoU (the recall-vs-FP tradeoff the operator comment warns about) plus region-detection recall overall and
for the hard sparse low-SNR case. Baseline = span 34.0, floor_frac 0.12.

Run: PYTHONPATH=/home/genesys-dgx1/Documents/dinov3 <.venv-ml python> ab_span_floor_sweep.py
"""
import sys
sys.path.insert(0, "src")
import json
from pathlib import Path
import numpy as np
import torch
import frontend as fe
from eval_heldout import load_model, FS, FSAMP, FFT, ROWS, SNR_EDGES

DEVICE = "cuda"
THR = 0.6
SIGMF = Path("data/dataset_491/heldout_sigmf")
SPANS = [24.0, 30.0, 34.0, 40.0, 48.0]
FLOORS = [0.08, 0.12, 0.16, 0.20]
LOW = ("-3..3", "3..10")

model = load_model("checkpoints/M3_491/best.pt", "configs/train_491.yaml", DEVICE)
NFFT = fe.FrontEndCfg().nfft
print(f"model loaded; thr={THR}; grid span={SPANS} x floor_frac={FLOORS}; grid={ROWS}x{NFFT}\n")


def bucket_of(snr):
    for lo, hi in SNR_EDGES:
        if lo <= snr < hi:
            return f"{lo}..{hi}"
    return ">=32" if snr >= 32 else "<-3"


# --- preload each frame's IQ (on CPU) + GT mask + annotation boxes (independent of the swept params) ---
frames = []  # list of dict(iq_cpu, gt[ROWS,NFFT] bool, boxes=[(r0,r1,c0,c1,bucket)])
for meta_p in sorted(SIGMF.glob("*.sigmf-meta")):
    meta = json.loads(meta_p.read_text())
    anns = meta["annotations"]
    iq = np.fromfile(str(meta_p.with_suffix(".sigmf-data")), dtype=np.complex64)
    nfr = iq.size // FSAMP
    for fidx in range(nfr):
        f0 = fidx * FSAMP
        gt = np.zeros((ROWS, NFFT), dtype=bool)
        boxes = []
        for a in anns:
            s = int(a["core:sample_start"]); c = int(a["core:sample_count"])
            if s >= f0 + FSAMP or s + c <= f0:
                continue
            r0 = max(0, (s - f0) // FFT); r1 = min(ROWS, -(-(s + c - f0) // FFT))
            lo = float(a["core:freq_lower_edge"]); hi = float(a["core:freq_upper_edge"])
            c0 = max(0, int(np.floor((lo + FS / 2) / FS * NFFT)))
            c1 = min(NFFT, int(np.ceil((hi + FS / 2) / FS * NFFT)))
            if r1 <= r0 or c1 <= c0:
                continue
            gt[r0:r1, c0:c1] = True
            boxes.append((r0, r1, c0, c1, bucket_of(float(a.get("wfgt:power_db", 0.0)))))
        frames.append(dict(iq=iq[f0:f0 + FSAMP].copy(), gt=gt, boxes=boxes))
print(f"{len(frames)} held-out frames; {sum(len(f['boxes']) for f in frames)} annotated signals\n")


def eval_cfg(span, floor):
    cfg = fe.FrontEndCfg(adaptive_span_db=span, adaptive_floor_frac=floor)
    tp = fp = fn = 0
    rdet = rtot = 0
    ldet = ltot = 0
    for fr in frames:
        iq = torch.from_numpy(fr["iq"]).to(DEVICE).unsqueeze(0)
        out = fe.compute_front_end(iq, FS, cfg, fft_size=FFT)
        tiles = out.tiles.to(DEVICE)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pr = torch.sigmoid(model(tiles).float()).squeeze(1)
        pred = (pr >= THR).to(torch.uint8).cpu().numpy().reshape(ROWS, NFFT).astype(bool)
        gt = fr["gt"]
        tp += int((pred & gt).sum()); fp += int((pred & ~gt).sum()); fn += int((~pred & gt).sum())
        for (r0, r1, c0, c1, bkt) in fr["boxes"]:
            d = int(pred[r0:r1, c0:c1].any())
            rtot += 1; rdet += d
            if bkt in LOW:
                ltot += 1; ldet += d
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / max(P + R, 1e-9); IoU = tp / max(tp + fp + fn, 1)
    return dict(span=span, floor=floor, P=P, R=R, F1=F1, IoU=IoU,
                rdet=100 * rdet / max(rtot, 1), lowdet=100 * ldet / max(ltot, 1))


rows = []
for span in SPANS:
    for floor in FLOORS:
        rows.append(eval_cfg(span, floor))
        r = rows[-1]
        print(f"span={span:4.0f} floor={floor:.2f} | P={r['P']:.3f} R={r['R']:.3f} "
              f"F1={r['F1']:.3f} IoU={r['IoU']:.3f} | region={r['rdet']:4.0f}% lowSNR={r['lowdet']:4.0f}%")

print("\n=== F1 grid (rows=span, cols=floor_frac) ===")
print("span\\floor " + "  ".join(f"{f:>6.2f}" for f in FLOORS))
for span in SPANS:
    cells = [next(x for x in rows if x['span'] == span and x['floor'] == f)['F1'] for f in FLOORS]
    print(f"{span:9.0f} " + "  ".join(f"{c:6.3f}" for c in cells))

base = next(x for x in rows if x['span'] == 34.0 and x['floor'] == 0.12)
bestF1 = max(rows, key=lambda x: x['F1'])
bestIoU = max(rows, key=lambda x: x['IoU'])
print(f"\nbaseline (34,0.12): F1={base['F1']:.3f} IoU={base['IoU']:.3f} region={base['rdet']:.0f}% lowSNR={base['lowdet']:.0f}%")
print(f"best F1  : span={bestF1['span']:.0f} floor={bestF1['floor']:.2f}  F1={bestF1['F1']:.3f} IoU={bestF1['IoU']:.3f} region={bestF1['rdet']:.0f}% lowSNR={bestF1['lowdet']:.0f}%")
print(f"best IoU : span={bestIoU['span']:.0f} floor={bestIoU['floor']:.2f}  F1={bestIoU['F1']:.3f} IoU={bestIoU['IoU']:.3f} region={bestIoU['rdet']:.0f}% lowSNR={bestIoU['lowdet']:.0f}%")
Path("checkpoints/M3_491").mkdir(parents=True, exist_ok=True)
Path("checkpoints/M3_491/span_floor_sweep.json").write_text(json.dumps(rows, indent=2))
