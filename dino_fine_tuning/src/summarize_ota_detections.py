"""Markdown summary of run_ota_detections.py outputs across capture folders (occupancy per model, agreement,
synthetic-overlay region detection where eval_scenes results exist).

  PYTHONPATH=src .venv-ml/bin/python src/summarize_ota_detections.py --root ~/Datasets/ARGUS/OOD_captures/OTA --out <md>
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); ap.add_argument("--glob", default="n320_*")
    a = ap.parse_args()
    root = Path(a.root).expanduser()
    rows = []
    for d in sorted(root.glob(a.glob)):
        dj = d / "detections.json"
        if not dj.exists():
            continue
        det = json.loads(dj.read_text()); meta = json.loads(sorted(d.glob("*.sigmf-meta"))[0].read_text())["global"]
        rows.append((d.name, det, meta))
    names = list(rows[0][1]["models"].keys()) if rows else []
    md = ["# N320 OTA stress captures — detector occupancy", "",
          f"{len(rows)} captures under `{root}`; masks for {', '.join(names)} in `<capture>/masks/<model>.npz`, panels in `<capture>/panels/`.",
          "Occupancy = fraction of the 512 x 1024 model grid marked signal, averaged over the evaluated frames (threshold 0.6).",
          "Agreement = IoU of the two masks over those frames.", "",
          "| capture | rate MSps | gain dB | frames | " + " | ".join(f"occ {n}" for n in names) + " | " + " | ".join(f"IoU {k}" for k in rows[0][1]["agreement_iou"]) + " | expected |",
          "|---|---|---|---|" + "---|" * len(names) + "---|" * len(rows[0][1]["agreement_iou"]) + "---|"]
    for name, det, meta in rows:
        occ = " | ".join(f"{100*det['models'][n]['occupancy_mean']:.2f}%" for n in names)
        agr = " | ".join(f"{v:.2f}" for v in det["agreement_iou"].values())
        md.append(f"| `{name}` | {det['rate_hz']/1e6:.2f} | {meta.get('argus:gain_db','')} | {len(det['frame_index'])}/{det['frames_total']} | {occ} | {agr} | {meta.get('argus:expected_waveforms','')} |")
    Path(a.out).write_text("\n".join(md) + "\n"); print("wrote", a.out, len(rows), "captures")


if __name__ == "__main__":
    main()
