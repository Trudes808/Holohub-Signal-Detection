"""Export a trained YOLO26 detector to TorchScript for the native yolo_detector operator.

The operator front-end mirrors yolo_training/src/yolo_infer.py: build the dB spectrogram (nfft=1024),
db_to_uint8, 256-row tiles, replicate gray->3ch, letterbox to imgsz, /255 -> run TorchScript -> decode
+ NMS -> scale boxes back to tile coords -> fill boxes into the binary mask -> stitch. TorchScript
module = the YOLO model (Ultralytics bakes the detection head in); decode/NMS live in the operator.

Usage: python export_yolo_torchscript.py --ckpt runs/detect/yolo26m_signal/weights/best.pt \
           --out weights/yolo26m.torchscript --imgsz 1024
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    m = YOLO(a.ckpt)
    names = m.names
    exported = m.export(format="torchscript", imgsz=a.imgsz, batch=1)  # -> best.torchscript next to ckpt
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), str(out))
    sidecar = out.with_suffix(".meta.json")
    sidecar.write_text(json.dumps({
        "ckpt": str(a.ckpt), "imgsz": a.imgsz, "conf": a.conf,
        "class_names": names, "num_classes": len(names),
        "input": f"float[1,3,{a.imgsz},{a.imgsz}] in [0,1] (letterboxed RGB tile)",
        "output": "Ultralytics detection head raw preds -> decode + NMS in the operator",
        "tile_rows": 256, "nfft": 1024, "preproc": "db->uint8 (dataset_meta vmin/vmax), gray->3ch, letterbox, /255",
        "post": "decode + NMS (conf, iou) -> boxes xyxy scaled to tile -> fill into mask",
    }, indent=2))
    print(f"saved {out} ({out.stat().st_size/1e6:.1f} MB) + {sidecar.name}  classes={names}")

if __name__ == "__main__":
    main()
