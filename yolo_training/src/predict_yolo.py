"""Run a trained/pretrained YOLO26 model on image(s).
  python src/predict_yolo.py --model runs/yolo26_rf/weights/best.pt --source <img_or_dir> --save
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--source", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    YOLO(args.model).predict(source=args.source, imgsz=args.imgsz, conf=args.conf, save=args.save)


if __name__ == "__main__":
    main()
