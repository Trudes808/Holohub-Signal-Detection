"""Train an Ultralytics YOLO26 model on an RF-spectrogram dataset.

Detection (boxes) example:
  python src/train_yolo.py --model yolo26s.pt --data configs/dataset.yaml --epochs 100 --imgsz 1024
Instance segmentation (native masks) — use a *-seg model + polygon labels:
  python src/train_yolo.py --model yolo26s-seg.pt --data configs/dataset_seg.yaml

Run with the 'yolo' conda env python.
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26s.pt",
                    help="pretrained YOLO26 weights; use a -seg.pt variant for instance masks")
    ap.add_argument("--data", required=True, help="dataset YAML (train/val paths + class names)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1024, help="square train size; spectrograms are wide -> tile/resize upstream")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="yolo26_rf")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                project=args.project, name=args.name, device=args.device)


if __name__ == "__main__":
    main()
