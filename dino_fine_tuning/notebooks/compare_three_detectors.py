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
# Aggregate numbers come from the SAME canonical tables as batch_eval_review_three_detectors.ipynb
# (produced by eval_detector_masks.py over all three detectors) — so both notebooks agree exactly.
TABLES_DIR  = FT_ROOT / "notebooks/compare_tables_canonical"

# ---- which fine-tuned model to compare ----
FT_CKPT   = FT_ROOT / "checkpoints/M1_ft/best.pt"     # best overall IoU/F1; swap to M2_ft for best 45-50 dB
FT_EVALMETA = FT_ROOT / "eval_out/M1_ft/eval_meta.json"
DET_THRESHOLD = 0.10   # region "detected" if coverage >= this (matches the original batch_eval_review notebook)

for p in (DINO_REPO, FT_ROOT / "src", EVAL_DIR):
    sys.path.insert(0, str(p))

import eval_viz as v
import mask_eval_metrics as mem
import plot_eval_results as pe
import finetuned_infer as fi
import yaml

# pretty display names + fixed colors/order for the four detectors
DET_ORDER  = ["coherent_power", "cuda_dino", "finetuned_dino", "finetuned_dino_m2"]
DET_LABEL  = {"coherent_power": "Coherent Power",
              "cuda_dino": "Zero-shot DINOv3",
              "finetuned_dino": "Fine-tuned DINOv3 (M1: ≤30 dB)",
              "finetuned_dino_m2": "Fine-tuned DINOv3 (M2: all dB)"}
DET_COLOR  = {"coherent_power": "#d95f02", "cuda_dino": "#7570b3",
              "finetuned_dino": "#1b9e77", "finetuned_dino_m2": "#e7298a"}

# %%
# Load both fine-tuned detectors (M1 = trained on ≤30 dB, M2 = trained on all dB).
train_cfg = yaml.safe_load(open(FT_ROOT / "configs/train.yaml"))
ds_meta   = json.loads((FT_ROOT / "data/dataset/dataset_meta.json").read_text())
detector    = fi.FinetunedDetector(str(FT_ROOT / "checkpoints/M1_ft/best.pt"), train_cfg, ds_meta,
                                   threshold=fi.load_threshold(FT_ROOT / "eval_out/M1_ft/eval_meta.json"))
detector_m2 = fi.FinetunedDetector(str(FT_ROOT / "checkpoints/M2_ft/best.pt"), train_cfg, ds_meta,
                                   threshold=fi.load_threshold(FT_ROOT / "eval_out/M2_ft/eval_meta.json"))
FT_MODELS = {"finetuned_dino": detector, "finetuned_dino_m2": detector_m2}
print(f"fine-tuned models: M1_ft (thr={detector.threshold:.2f}), M2_ft (thr={detector_m2.threshold:.2f})")

ATTEN = {  # stem -> attenuation dB
    "attenuation_dB_0": 0, "attenuation_dB_5": 5, "attenuation_dB_10": 10,
    "attenuation_dB_15": 15, "attenuation_dB_20": 20, "attenuation_dB_25": 25,
    "attenuation_dB_30": 30, "attenuation_dB_35": 35, "attenuation_dB_40": 40,
    "attenuation_dB_45": 45, "attenuation_dB_50": 50, "attenuation_dB_55": 55,
    "attenuation_dB_60": 60,
}


# %%
def add_finetuned(bundle, stem):
    """Run both fine-tuned models on this frame's IQ and attach their masks (display grid)."""
    dp = v.find_capture_data(stem, CAPTURE_DIRS)
    row = next(r for r in mem.load_manifest(BATCH_ROOT / "coherent_power" / stem)
               if int(r["frame_number"]) == bundle.frame_number)
    iq = v.read_frame_iq(dp, int(row["local_file_offset_complex"]), int(row["complex_samples_read"]))
    # max-pool onto the display/GT grid so thin short-time detections survive the
    # 10x time downsample (nearest would drop Zadoff-Chu-style pulses).
    for name, mdl in FT_MODELS.items():
        bundle.detector_masks[name] = fi.to_display_grid(mdl.mask_for_iq(iq), bundle.fft_rows, bundle.fft_cols)
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
    display(fig); plt.close(fig)   # close -> no duplicate auto-render at cell end


# %% [markdown]
# ## 2. Aggregate metrics vs attenuation (from the canonical 3-detector tables)
#
# Numbers come from `compare_tables_canonical/` — the **same** `eval_detector_masks.py`
# output used by `batch_eval_review_three_detectors.ipynb`, so both notebooks agree
# exactly. **Region detection rate** (coverage ≥ threshold) is the fair headline;
# pixel-IoU is shown too (it rewards box-filling — see caveats).

# %%
import pandas as pd
region = pd.DataFrame(pe.load_region(TABLES_DIR / "region_metrics.csv"))
frame  = pd.DataFrame(pe.load_frame(TABLES_DIR / "frame_pixel_metrics.csv"))
region["detected"] = region["coverage"].astype(float) >= DET_THRESHOLD
print("detectors:", sorted(region.detector.unique()),
      "| region rows:", len(region), "| frame rows:", len(frame))

# %%
# Region detection rate vs attenuation (headline) + pixel IoU vs attenuation.
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for k in DET_ORDER:
    g = region[region.detector == k].groupby("attenuation_db")["detected"].mean()
    axes[0].plot(g.index, g.values, "-o", label=DET_LABEL[k], color=DET_COLOR[k])
    fk = frame[(frame.detector == k) & (frame.mask_present)]
    gp = fk.groupby("attenuation_db")["iou"].mean()
    axes[1].plot(gp.index, gp.values, "-o", label=DET_LABEL[k], color=DET_COLOR[k])
axes[0].set_title(f"Region detection rate vs attenuation (coverage ≥ {DET_THRESHOLD})")
axes[1].set_title("Pixel IoU vs attenuation (favors box-filling — see caveats)")
for ax in axes:
    ax.set_xlabel("attenuation (dB)  — higher = lower SNR"); ax.grid(alpha=0.3)
    ax.set_ylim(-0.02, 1.02); ax.legend()
axes[0].set_ylabel("detection rate"); axes[1].set_ylabel("mean pixel IoU")
fig.tight_layout(); fig.savefig(FIG_DIR / "metrics_vs_atten_3det.png", dpi=110)
display(fig); plt.close(fig)


# %%
# Detection rate by waveform class (which signals each detector misses).
piv = region.pivot_table(index="signal_class", columns="detector", values="detected", aggfunc="mean")
piv = piv.reindex(columns=[k for k in DET_ORDER if k in piv.columns])
piv.columns = [DET_LABEL[c] for c in piv.columns]
print("Detection rate by waveform class (all attenuations pooled):")
display(piv.round(2))
ax = piv.plot(kind="bar", figsize=(12, 5), color=[DET_COLOR[k] for k in DET_ORDER])
ax.set_title(f"Detection rate by waveform class (coverage ≥ {DET_THRESHOLD})")
ax.set_ylabel("detection rate"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3, axis="y")
fig = ax.get_figure(); fig.tight_layout(); fig.savefig(FIG_DIR / "detection_by_class_3det.png", dpi=110)
display(fig); plt.close(fig)


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
