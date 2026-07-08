# %% [markdown]
# # Three-detector comparison: Coherent Power vs Zero-shot DINOv3 vs Fine-tuned DINOv3
#
# Side-by-side evaluation on the INFOCOM captures, built on the same
# `eval_viz` / `mask_eval_metrics` machinery as `batch_eval_review.ipynb`.
#
# - **Coherent Power** and **Zero-shot DINOv3** (`cuda_dino`) masks come from the
#   deployed detectors' batch-eval run (`sweep_20260630`, 512×10240 frames).
# - **Fine-tuned DINOv3** is our `M1_ft` model, run offline on each frame's raw IQ
#   at its native front-end (nfft=1024, 256-row tiles) and resized onto the display
#   grid — the same way the other detectors' masks are resampled for comparison.
#
# **Read the caveats at the bottom before quoting numbers.** In particular, the
# ground truth marks *filled annotation boxes even where the signal is buried below
# the noise floor at high attenuation*, and pixel-IoU rewards a detector for
# reproducing that box-filling convention. The fair headline metric is
# **region detection rate** (a region counts as found if ≥30 % of its box is
# covered), which is what we plot vs attenuation.

# %%
from pathlib import Path
import sys, json, warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---- paths (edit if your layout differs) ----
DINO_REPO   = Path.home() / "dinov3"
FT_ROOT     = Path.home() / "Holohub-Signal-Detection/dino_fine_tuning"
EVAL_DIR    = Path.home() / ("Holohub-Signal-Detection/applications/usrp_wideband_signal_detection"
                             "/infocom_evals/signal_detection_experiments")
BATCH_ROOT  = Path("/tmp/usrp_spectrograms/batch_eval/sweep_20260630")
CAPTURE_DIRS = [Path.home() / "captures"]
FIG_DIR     = FT_ROOT / "reports/figs_compare"; FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- which fine-tuned model to compare ----
FT_CKPT   = FT_ROOT / "checkpoints/M1_ft/best.pt"     # best overall IoU/F1; swap to M2_ft for best 45-50 dB
FT_EVALMETA = FT_ROOT / "eval_out/M1_ft/eval_meta.json"
DET_THRESHOLD = 0.30   # region "detected" if coverage >= this (matches report.md)

for p in (DINO_REPO, FT_ROOT / "src", EVAL_DIR):
    sys.path.insert(0, str(p))

import eval_viz as v
import mask_eval_metrics as mem
import finetuned_infer as fi
import yaml

# pretty display names + fixed colors/order for the three detectors
DET_ORDER  = ["coherent_power", "cuda_dino", "finetuned_dino"]
DET_LABEL  = {"coherent_power": "Coherent Power",
              "cuda_dino": "Zero-shot DINOv3",
              "finetuned_dino": "Fine-tuned DINOv3"}
DET_COLOR  = {"coherent_power": "#d95f02", "cuda_dino": "#7570b3", "finetuned_dino": "#1b9e77"}

# %%
# Load the fine-tuned detector once.
train_cfg = yaml.safe_load(open(FT_ROOT / "configs/train.yaml"))
ds_meta   = json.loads((FT_ROOT / "data/dataset/dataset_meta.json").read_text())
THRESH    = fi.load_threshold(FT_EVALMETA)
detector  = fi.FinetunedDetector(str(FT_CKPT), train_cfg, ds_meta, threshold=THRESH)
print(f"Fine-tuned model: {FT_CKPT.parent.name}  (decision threshold={THRESH:.2f})")

ATTEN = {  # stem -> attenuation dB
    "attenuation_dB_0": 0, "attenuation_dB_5": 5, "attenuation_dB_10": 10,
    "attenuation_dB_15": 15, "attenuation_dB_20": 20, "attenuation_dB_25": 25,
    "attenuation_dB_30": 30, "attenuation_dB_35": 35, "attenuation_dB_40": 40,
    "attenuation_dB_45": 45, "attenuation_dB_50": 50, "attenuation_dB_55": 55,
    "attenuation_dB_60": 60,
}


# %%
def add_finetuned(bundle, stem):
    """Run the fine-tuned model on this frame's IQ and attach its mask (display grid)."""
    dp = v.find_capture_data(stem, CAPTURE_DIRS)
    row = next(r for r in mem.load_manifest(BATCH_ROOT / "coherent_power" / stem)
               if int(r["frame_number"]) == bundle.frame_number)
    iq = v.read_frame_iq(dp, int(row["local_file_offset_complex"]), int(row["complex_samples_read"]))
    m_native = detector.mask_for_iq(iq)                       # (rows, 1024)
    # max-pool onto the display/GT grid so thin short-time detections survive the
    # 10x time downsample (nearest would drop Zadoff-Chu-style pulses).
    bundle.detector_masks["finetuned_dino"] = fi.to_display_grid(
        m_native, bundle.fft_rows, bundle.fft_cols)
    return bundle


def load_all(stem, frame):
    b = v.load_frame_bundle_smart(BATCH_ROOT, frame, file_stem=stem, capture_dirs=CAPTURE_DIRS)
    return add_finetuned(b, stem)


def most_visible_frame(stem):
    """Pick an annotated frame that actually contains detectable energy (largest
    coherent-power ON area) so the visual comparison is informative rather than a
    fully-buried box."""
    ann, _ = v.classify_frames(BATCH_ROOT, stem)
    run = BATCH_ROOT / "coherent_power" / stem
    best, best_area = ann[len(ann) // 2] if ann else 1, -1
    manifest = {int(r["frame_number"]): r for r in mem.load_manifest(run)}
    for f in ann:
        r = manifest.get(f)
        if not r or not r.get("mask_npy"):
            continue
        m = mem.load_mask_any(run / r["mask_npy"])
        if m is None:
            continue
        area = int((m > 0).sum())
        if area > best_area:
            best_area, best = area, f
    return best


# %% [markdown]
# ## 1. Side-by-side spectrogram overlays across attenuation
#
# For each attenuation we pick the frame with the most *detectable* energy (so the
# comparison is meaningful). Panels: **ground truth**, then each detector's mask
# overlaid on the same spectrogram.

# %%
SHOWCASE = ["attenuation_dB_0", "attenuation_dB_20", "attenuation_dB_40", "attenuation_dB_55"]
for stem in SHOWCASE:
    frame = most_visible_frame(stem)
    b = load_all(stem, frame)
    # relabel keys for pretty panel titles
    pretty = {DET_LABEL[k]: b.detector_masks[k] for k in DET_ORDER if k in b.detector_masks}
    b.detector_masks = pretty
    fig = v.plot_frame_panels(b, detectors=list(pretty.keys()))
    fig.suptitle(f"{stem}  ({ATTEN[stem]} dB attenuation)  — frame {frame}", y=1.02, fontsize=13)
    fig.savefig(FIG_DIR / f"panels_{stem}.png", dpi=95, bbox_inches="tight")
    plt.show()


# %% [markdown]
# ## 2. Aggregate metrics vs attenuation (the fair comparison)
#
# For each capture we sample several annotated frames, and for every ground-truth
# region measure the coverage each detector achieves. **Region detection rate**
# (coverage ≥ 0.30) is the headline; pixel-IoU is shown too, with the caveat that it
# rewards box-filling.

# %%
def region_bandwidth_bucket(item):
    bw = abs(float(item.get("freq_upper_hz", 0)) - float(item.get("freq_lower_hz", 0)))
    return mem.bucket_bandwidth({"occupied_bw_hz": bw})

K_FRAMES = 6          # annotated frames sampled per capture
rows_det, rows_pix = [], []
for stem, atten in ATTEN.items():
    ann, _ = v.classify_frames(BATCH_ROOT, stem)
    if not ann:
        continue
    picks = v.pick_spread(ann, K_FRAMES)
    for frame in picks:
        try:
            b = load_all(stem, frame)
        except Exception as e:
            print("skip", stem, frame, e); continue
        gt = (b.gt_mask > 0).astype(np.uint8)
        for k in DET_ORDER:
            if k not in b.detector_masks:
                continue
            pred = (b.detector_masks[k] > 0).astype(np.uint8)
            pm = mem.pixel_metrics(pred, gt)
            rows_pix.append({"stem": stem, "atten": atten, "det": k,
                             "iou": pm.iou, "recall": pm.recall, "precision": pm.precision})
            for it in b.gt_items:
                rr = mem.region_coverage(pred, it, b.fft_rows, b.fft_cols)
                rows_det.append({"stem": stem, "atten": atten, "det": k,
                                 "coverage": rr.coverage, "label": it.get("label"),
                                 "bw_bucket": region_bandwidth_bucket(it),
                                 "detected": (rr.coverage >= DET_THRESHOLD)})
    print(f"  {stem}: {len(picks)} frames")

import pandas as pd
det_df = pd.DataFrame(rows_det); pix_df = pd.DataFrame(rows_pix)
det_df.to_csv(FIG_DIR / "region_detection_3det.csv", index=False)
pix_df.to_csv(FIG_DIR / "pixel_metrics_3det.csv", index=False)
print("regions scored:", len(det_df), "| frames scored:", len(pix_df))


# %%
# Region detection rate vs attenuation (headline) + pixel IoU vs attenuation.
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for k in DET_ORDER:
    g = det_df[det_df.det == k].groupby("atten")["detected"].mean()
    axes[0].plot(g.index, g.values, "-o", label=DET_LABEL[k], color=DET_COLOR[k])
    gp = pix_df[pix_df.det == k].groupby("atten")["iou"].mean()
    axes[1].plot(gp.index, gp.values, "-o", label=DET_LABEL[k], color=DET_COLOR[k])
axes[0].set_title(f"Region detection rate vs attenuation (coverage ≥ {DET_THRESHOLD})")
axes[1].set_title("Pixel IoU vs attenuation (favors box-filling — see caveats)")
for ax in axes:
    ax.set_xlabel("attenuation (dB)  — higher = lower SNR"); ax.grid(alpha=0.3)
    ax.set_ylim(-0.02, 1.02); ax.legend()
axes[0].set_ylabel("detection rate"); axes[1].set_ylabel("mean IoU")
fig.tight_layout(); fig.savefig(FIG_DIR / "metrics_vs_atten_3det.png", dpi=110); plt.show()


# %%
# Detection rate by waveform class (which signals each detector misses).
import pandas as pd
piv = (det_df.assign(det=det_df.det.map(DET_LABEL))
       .pivot_table(index="label", columns="det", values="detected", aggfunc="mean"))
piv = piv.reindex(columns=[DET_LABEL[k] for k in DET_ORDER])
print("Detection rate by waveform class (all attenuations pooled):")
display(piv.round(2))
ax = piv.plot(kind="bar", figsize=(12, 5),
              color=[DET_COLOR[k] for k in DET_ORDER])
ax.set_title(f"Detection rate by waveform class (coverage ≥ {DET_THRESHOLD})")
ax.set_ylabel("detection rate"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig(FIG_DIR / "detection_by_class_3det.png", dpi=110); plt.show()


# %% [markdown]
# ## Caveats (read before quoting numbers)
#
# 1. **Ground truth includes buried signals.** Annotation boxes mark where a
#    transmitter *was*, even when it is below the noise floor at high attenuation.
#    So at 45–60 dB, large boxes cover pure-noise pixels that no detector can (or
#    should) flag — pixel recall/IoU there is low *for all three detectors*.
# 2. **Pixel-IoU favors the fine-tuned model.** It was trained to reproduce the
#    filled-box convention, so it "fills" detected regions; the coherent-power and
#    zero-shot detectors emit sparse energy-based masks and are penalized on IoU even
#    when they correctly localize the signal. **Use region detection rate for a fair
#    read.**
# 3. **Frame geometry differs.** Batch frames are 512×10240 (nfft=10240, ~21 ms);
#    the fine-tuned model runs at nfft=1024 / 256-row tiles and its mask is resized
#    onto the display grid — identical treatment to how `cuda_dino`'s internal
#    1024-wide mask is resampled.
# 4. **`most_visible_frame` biases the *visuals* toward frames with detectable
#    energy** so the overlays are informative; the aggregate metrics use a spread of
#    annotated frames (`pick_spread`) and are not cherry-picked.
