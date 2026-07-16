# %% [markdown]
# # YOLO26 vs DINO/coherent — signal-detection comparison
#
# All detectors scored by the **same** `eval_detector_masks.py` over the same batch frames,
# masks materialized into one `sweep_all` root (deployed coherent/zero-shot-DINO from the
# container, fine-tuned DINO M1/M2 from the DINO pipeline, YOLO26 s/m via `gen_yolo_run`).
# Only the masks differ → apples-to-apples. Tables from `eval/compare_tables/`. Kernel: **Python (yolo)**.

# %%
from pathlib import Path
import os, sys, warnings
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
warnings.filterwarnings("ignore")

EVAL_DIR = Path.home() / ("Holohub-Signal-Detection/applications/usrp_wideband_signal_detection"
                          "/infocom_evals/signal_detection_experiments")
YOLO_EVAL_ROOT = Path(os.environ.get("YOLO_EVAL_ROOT",
                      str(Path.home() / "Holohub-Signal-Detection/yolo_training/eval")))
TABLES_DIR   = YOLO_EVAL_ROOT / "compare_tables"
DETS_ROOT    = YOLO_EVAL_ROOT / "sweeps/sweep_all"
CAPTURE_DIRS = [Path.home() / "captures"]
DET_THRESHOLD = 0.1

sys.path.insert(0, str(EVAL_DIR))
import eval_viz as v
import mask_eval_metrics as mem
import plot_eval_results as pe

# traditional/deployed + DINO + YOLO, grouped for the plots
DET_ORDER = ["coherent_power", "cuda_dino", "finetuned_dino", "finetuned_dino_m2", "yolo26s", "yolo26m"]
DET_LABEL = {"coherent_power": "Coherent Power", "cuda_dino": "Zero-shot DINOv3",
             "finetuned_dino": "Fine-tuned DINOv3 (M1)", "finetuned_dino_m2": "Fine-tuned DINOv3 (M2)",
             "yolo26s": "YOLO26s (fine-tuned)", "yolo26m": "YOLO26m (fine-tuned)"}


def show(fig):
    display(fig); plt.close(fig)


# %%
if not (TABLES_DIR / "region_metrics.csv").exists():
    print("Combined tables missing. After YOLO training finishes, build with:")
    print("  python src/assemble_yolo_eval.py    # materializes YOLO masks + symlinks DINO dirs + evals")
region = pe.load_region(TABLES_DIR / "region_metrics.csv")
frame  = pe.load_frame(TABLES_DIR / "frame_pixel_metrics.csv")
present = pe._detectors(region)
DET_ORDER = [d for d in DET_ORDER if d in present]
missing = [d for d in ("yolo26s", "yolo26m") if d not in present]
print("detectors present:", DET_ORDER)
if missing:
    print("NOT YET present (train + re-assemble to add):", missing)
print(f"{len(region)} region rows, {len(frame)} frame rows")


# %%
def load_bundle(stem, frame_number):
    """Spectrogram + every materialized detector mask from sweep_all, ordered + labeled."""
    b = v.load_frame_bundle_smart(DETS_ROOT, frame_number, file_stem=stem, capture_dirs=CAPTURE_DIRS)
    b.detector_masks = {DET_LABEL[k]: b.detector_masks[k] for k in DET_ORDER if k in b.detector_masks}
    return b


# %% [markdown]
# ## 1. Single-frame overlay (spectrogram + every detector)

# %%
FILE_STEM = "attenuation_dB_45"; FRAME = 100    # same frame as the DINO notebooks
b = load_bundle(FILE_STEM, FRAME)
print(f"{FILE_STEM} frame {FRAME}: grid {b.fft_rows}x{b.fft_cols}, {len(b.gt_items)} GT regions")
show(v.plot_frame_panels(b, detectors=list(b.detector_masks.keys())))

# %% [markdown]
# ## 2. Reproducible frame review (annotated + noise-only), seed 7 (same as the DINO notebooks)

# %%
REVIEW_STEM = "attenuation_dB_45"
SAMPLE = v.sample_review_frames(DETS_ROOT, REVIEW_STEM, n_annotated=5, n_noise=1, seed=7)
print(f"reviewing {SAMPLE['review_frames']}")
for fr in SAMPLE["review_frames"]:
    b = load_bundle(REVIEW_STEM, fr)
    tag = "NOISE-ONLY" if fr in SAMPLE["noise_frames"] else f"{len(b.gt_items)} GT regions"
    print(f"--- frame {fr}: {tag} ---")
    show(v.plot_frame_panels(b, detectors=list(b.detector_masks.keys())))

# %% [markdown]
# ## 3. Performance vs power, faceted by signal class / bandwidth / pulse length (line per detector)

# %%
show(pe.fig_rate_vs_power_by(region, "signal_class", None, DET_THRESHOLD, "Detection rate vs power, by signal class"))
show(pe.fig_rate_vs_power_by(region, "bandwidth", pe.BW_ORDER, DET_THRESHOLD, "Detection rate vs power, by bandwidth"))
show(pe.fig_rate_vs_power_by(region, "pulse_length", pe.LEN_ORDER, DET_THRESHOLD, "Detection rate vs power, by pulse length"))

# %% [markdown]
# ## 4. Performance vs bandwidth and vs pulse length

# %%
show(pe.fig_metric_vs_bucket(region, "bandwidth", pe.BW_ORDER, DET_THRESHOLD, "Performance vs signal bandwidth"))
show(pe.fig_metric_vs_bucket(region, "pulse_length", pe.LEN_ORDER, DET_THRESHOLD, "Performance vs pulse length"))

# %% [markdown]
# ## 5. Frame-level accuracy (P/R/F1/pixel-IoU) + false-positive area vs power

# %%
show(pe.fig_frame_metrics_vs_power(frame))

# %% [markdown]
# ## 6. Numeric summary — detection rate vs attenuation + by waveform class

# %%
rdf = pd.DataFrame(region); rdf["detected"] = rdf["coverage"].astype(float) >= DET_THRESHOLD
piv = rdf.pivot_table(index="detector", columns="attenuation_db", values="detected", aggfunc="mean").round(2)
piv = piv.reindex([d for d in DET_ORDER if d in piv.index]); piv.index = [DET_LABEL.get(i, i) for i in piv.index]
print(f"Detection rate vs attenuation (coverage ≥ {DET_THRESHOLD}):"); display(piv)
cls = rdf.pivot_table(index="signal_class", columns="detector", values="detected", aggfunc="mean").round(2)
cls = cls.reindex(columns=[d for d in DET_ORDER if d in cls.columns]); cls.columns = [DET_LABEL.get(c, c) for c in cls.columns]
print("\nDetection rate by waveform class (all attenuations):"); display(cls)
