# %% [markdown]
# # Batch-eval review — three detectors (Coherent Power · Zero-shot DINOv3 · Fine-tuned DINOv3)
#
# This notebook mirrors the **rich graph set** of
# `infocom_evals/.../batch_eval_review.ipynb` (the `plot_eval_results` figures:
# detection rate & IoU vs power faceted by signal class / bandwidth / pulse length,
# and per-frame precision/recall/F1/IoU/FP vs power) — but with **three detectors**
# on every plot, adding our **fine-tuned DINOv3** (`M1_ft`) alongside the two deployed
# detectors from the `sweep_20260630` batch run.
#
# - Deployed masks (`coherent_power`, `cuda_dino` = zero-shot DINOv3) come from the batch sweep.
# - `finetuned_dino` is run offline on each frame's IQ (`src/finetuned_infer.py`, nfft=1024,
#   256-row tiles, max-pooled onto the 512×10240 display grid).
# - The tidy fact tables (`compare_tables/`) are produced by `src/gen_compare_tables.py`
#   using the same `mask_eval_metrics` primitives + bucketers as the deployed eval, so the
#   graphs are directly comparable.
#
# **See the caveats at the bottom** (GT marks buried signals at high attenuation; pixel-IoU
# rewards box-filling — region detection rate is the fair headline).

# %%
from pathlib import Path
import sys, json, warnings, subprocess
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
warnings.filterwarnings("ignore")

FT_ROOT      = Path.home() / "Holohub-Signal-Detection/dino_fine_tuning"
DINO_REPO    = Path.home() / "dinov3"
EVAL_DIR     = Path.home() / ("Holohub-Signal-Detection/applications/usrp_wideband_signal_detection"
                              "/infocom_evals/signal_detection_experiments")
BATCH_ROOT   = Path("/tmp/usrp_spectrograms/batch_eval/sweep_20260630")
CAPTURE_DIRS = [Path.home() / "captures"]
TABLES_DIR   = FT_ROOT / "notebooks/compare_tables"          # 3-detector tidy tables
FIG_DIR      = FT_ROOT / "reports/figs_compare"; FIG_DIR.mkdir(parents=True, exist_ok=True)
DET_THRESHOLD = 0.30                                          # coverage >= this == "detected"

for p in (DINO_REPO, FT_ROOT / "src", EVAL_DIR):
    sys.path.insert(0, str(p))
import eval_viz as v
import mask_eval_metrics as mem
import plot_eval_results as pe
import finetuned_infer as fi
import yaml

DET_LABEL = {"coherent_power": "Coherent Power", "cuda_dino": "Zero-shot DINOv3",
             "finetuned_dino": "Fine-tuned DINOv3"}

# %%
# Load the fine-tuned detector (for the spectrogram panels) and the tidy tables (for the graphs).
train_cfg = yaml.safe_load(open(FT_ROOT / "configs/train.yaml"))
ds_meta   = json.loads((FT_ROOT / "data/dataset/dataset_meta.json").read_text())
THRESH    = fi.load_threshold(FT_ROOT / "eval_out/M1_ft/eval_meta.json")
detector  = fi.FinetunedDetector(str(FT_ROOT / "checkpoints/M1_ft/best.pt"), train_cfg, ds_meta, threshold=THRESH)

if not (TABLES_DIR / "region_metrics.csv").exists():
    print("tables missing — regenerate with:")
    print(f"  python {FT_ROOT/'src/gen_compare_tables.py'} --out-dir {TABLES_DIR}")
region = pe.load_region(TABLES_DIR / "region_metrics.csv")
frame  = pe.load_frame(TABLES_DIR / "frame_pixel_metrics.csv")
print(f"detectors: {pe._detectors(region)} | {len(region)} region rows, {len(frame)} frame rows")


# %%
def load_bundle_3det(stem, frame_number):
    """Bundle with coherent_power + cuda_dino (from sweep) + fine-tuned (run live)."""
    b = v.load_frame_bundle_smart(BATCH_ROOT, frame_number, file_stem=stem, capture_dirs=CAPTURE_DIRS)
    row = next(r for r in mem.load_manifest(BATCH_ROOT / "coherent_power" / stem)
               if int(r["frame_number"]) == frame_number)
    dp = v.find_capture_data(stem, CAPTURE_DIRS)
    iq = v.read_frame_iq(dp, int(row["local_file_offset_complex"]), int(row["complex_samples_read"]))
    b.detector_masks["finetuned_dino"] = fi.to_display_grid(detector.mask_for_iq(iq), b.fft_rows, b.fft_cols)
    # order + pretty labels for panels
    b.detector_masks = {DET_LABEL[k]: b.detector_masks[k]
                        for k in ["coherent_power", "cuda_dino", "finetuned_dino"] if k in b.detector_masks}
    return b


# %% [markdown]
# ## 1. Single-frame overlay (spectrogram + each detector)

# %%
FILE_STEM = "attenuation_dB_20"
# pick an annotated frame with detectable energy
ann, _ = v.classify_frames(BATCH_ROOT, FILE_STEM)
FRAME = ann[len(ann) // 2]
b = load_bundle_3det(FILE_STEM, FRAME)
print(f"{FILE_STEM} frame {FRAME}: grid {b.fft_rows}x{b.fft_cols}, {len(b.gt_items)} GT regions")
fig = v.plot_frame_panels(b, detectors=list(b.detector_masks.keys()))
fig.suptitle(f"{FILE_STEM} — frame {FRAME}", y=1.02); display(fig)


# %% [markdown]
# ## 2. Reproducible frame review (annotated + noise-only), all three detectors

# %%
REVIEW_STEM = "attenuation_dB_40"
SAMPLE = v.sample_review_frames(BATCH_ROOT, REVIEW_STEM, n_annotated=3, n_noise=1, seed=7)
print(f"{SAMPLE['annotated_available']} annotated / {SAMPLE['noise_available']} noise-only; reviewing {SAMPLE['review_frames']}")
for fr in SAMPLE["review_frames"]:
    b = load_bundle_3det(REVIEW_STEM, fr)
    tag = "NOISE-ONLY" if fr in SAMPLE["noise_frames"] else f"{len(b.gt_items)} GT regions"
    print(f"--- frame {fr}: {tag} ---")
    display(v.plot_frame_panels(b, detectors=list(b.detector_masks.keys())))


# %% [markdown]
# ## 3. The rich graphs — detection & pixel metrics vs power, one line per detector
#
# These are the extra `plot_eval_results` graphs, now comparing all three detectors.

# %%
display(pe.fig_rate_vs_power_by(region, "signal_class", None, DET_THRESHOLD,
                                "Detection rate vs power, by signal class"))
display(pe.fig_rate_vs_power_by(region, "bandwidth", pe.BW_ORDER, DET_THRESHOLD,
                                "Detection rate vs power, by bandwidth"))
display(pe.fig_rate_vs_power_by(region, "pulse_length", pe.LEN_ORDER, DET_THRESHOLD,
                                "Detection rate vs power, by pulse length (duration)"))

# %%
display(pe.fig_metric_vs_bucket(region, "bandwidth", pe.BW_ORDER, DET_THRESHOLD,
                                "Detection rate / box-IoU / coverage vs bandwidth"))
display(pe.fig_metric_vs_bucket(region, "pulse_length", pe.LEN_ORDER, DET_THRESHOLD,
                                "Detection rate / box-IoU / coverage vs pulse length"))

# %%
# per-frame precision / recall / F1 / IoU / false-positive-area vs power (attenuation)
display(pe.fig_frame_metrics_vs_power(frame, "Per-frame pixel metrics vs attenuation"))


# %% [markdown]
# ## 4. Numeric summary — detection rate vs attenuation (coverage ≥ threshold)

# %%
import pandas as pd
rdf = pd.DataFrame(region)
rdf["detected"] = rdf["coverage"].astype(float) >= DET_THRESHOLD
piv = rdf.pivot_table(index="detector", columns="attenuation_db", values="detected", aggfunc="mean").round(2)
piv.index = [DET_LABEL.get(i, i) for i in piv.index]
print("Region detection rate vs attenuation:"); display(piv)
print("\nDetection rate by waveform class (all attenuations):")
cls = rdf.pivot_table(index="signal_class", columns="detector", values="detected", aggfunc="mean").round(2)
cls.columns = [DET_LABEL.get(c, c) for c in cls.columns]; display(cls)


# %% [markdown]
# ## Caveats (same as the three-detector notebook)
#
# 1. **GT includes buried signals** — annotation boxes mark where a transmitter was, even
#    below the noise floor at high attenuation, so pixel recall/IoU is low for *all* detectors there.
# 2. **Pixel-IoU favors the fine-tuned model** (trained to reproduce the filled-box convention);
#    coherent-power and zero-shot emit sparse energy masks. **Region detection rate is the fair metric.**
# 3. **Geometry** — batch frames are 512×10240 (nfft=10240); the fine-tuned model runs at nfft=1024
#    and its mask is max-pooled onto the display grid (preserves thin Zadoff-Chu pulses).
# 4. Tables are built from `--frames-per-capture 60` spread frames per capture; regenerate with
#    `src/gen_compare_tables.py` (adjust `--frames-per-capture` for more/less).
