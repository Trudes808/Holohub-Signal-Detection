"""Materialize a fine-tuned YOLO26 detector's masks into a batch-eval-format run dir,
so the CANONICAL eval_detector_masks.py scores it identically to the DINO/deployed
detectors (same metrics, same source-SigMF buckets, all frames).

Mirrors dino_fine_tuning/src/gen_finetuned_run.py. For each capture it builds
  <out-root>/<detector-name>/<stem>/ with:
    - frame_manifest.csv, gt_masks/, gt_annotations/  (symlinked from --ref-detector: GT is
      detector-independent)
    - mask_arrays/mask_ch0_f{N}_{R}x{C}.packed.npz    (this YOLO model's mask per frame)
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np

FT = Path("/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning")
EVAL_DIR = Path("/home/bqn82/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection"
                "/infocom_evals/signal_detection_experiments")
YT = Path("/home/bqn82/Holohub-Signal-Detection/yolo_training")
for p in (str(FT / "src"), str(EVAL_DIR), str(YT / "src")):
    sys.path.insert(0, p)
import eval_viz as v                # noqa: E402
from yolo_infer import YoloDetector, to_display_grid   # noqa: E402


def log(m): print(f"[yolorun] {m}", flush=True)


def save_packed(path: Path, mask: np.ndarray):
    R, C = mask.shape
    np.savez_compressed(path, packed=np.packbits(mask.astype(np.uint8).ravel()), rows=R, cols=C)


def _link(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src.resolve(), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="/tmp/usrp_spectrograms/batch_eval/sweep_20260630")
    ap.add_argument("--out-root", default=str(YT / "eval/sweeps/sweep_all"))
    ap.add_argument("--captures-dir", default="/home/bqn82/captures")
    ap.add_argument("--yolo-ckpt", required=True, help="path to best.pt")
    ap.add_argument("--detector-name", required=True, help="run-dir name (e.g. yolo26s, yolo26m)")
    ap.add_argument("--dataset-meta", default=str(FT / "data/dataset/dataset_meta.json"),
                    help="dB calibration (db_vmin/vmax) the model trained on")
    ap.add_argument("--ref-detector", default="cuda_dino", help="detector to mirror GT/manifest from")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--stems", default=None)
    args = ap.parse_args()

    sweep = Path(args.sweep); out_root = Path(args.out_root); caps = [Path(args.captures_dir)]
    out_root.mkdir(parents=True, exist_ok=True)
    meta = json.loads(Path(args.dataset_meta).read_text())
    det = YoloDetector(args.yolo_ckpt, meta, conf=args.conf, name=args.detector_name)
    log(f"[{args.detector_name}] ckpt={args.yolo_ckpt} conf={args.conf} vmin/vmax={det.vmin:.1f}/{det.vmax:.1f}")

    ref = args.ref_detector
    stems = (args.stems.split(",") if args.stems else
             sorted(p.name for p in (sweep / ref).iterdir() if p.is_dir()))
    for stem in stems:
        src_run = sweep / ref / stem
        dst_run = out_root / args.detector_name / stem
        (dst_run / "mask_arrays").mkdir(parents=True, exist_ok=True)
        for name in ("frame_manifest.csv", "gt_masks", "gt_annotations"):
            _link(src_run / name, dst_run / name)
        dp = v.find_capture_data(stem, caps)
        manifest = list(csv.DictReader(open(src_run / "frame_manifest.csv")))
        made = skipped = 0
        for row in manifest:
            nsamp = int(row.get("complex_samples_read") or 0)
            R, C = int(row["fft_rows"]), int(row["fft_cols"])
            fnum = int(row["frame_number"])
            out_npz = dst_run / "mask_arrays" / f"mask_ch0_f{fnum}_{R}x{C}.packed.npz"
            if out_npz.exists():
                skipped += 1; continue
            if nsamp < det.nfft:
                continue
            iq = v.read_frame_iq(dp, int(row["local_file_offset_complex"]), nsamp)
            mask = to_display_grid(det.mask_for_iq(iq), R, C)
            save_packed(dst_run / "mask_arrays" / f"mask_ch0_f{fnum}_{R}x{C}.packed", mask)
            made += 1
        log(f"{stem}: {made} masks made, {skipped} present ({len(manifest)} frames)")
    log(f"DONE -> {out_root}/{args.detector_name}")


if __name__ == "__main__":
    main()
