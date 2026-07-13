# %% [markdown]
# # Batch-eval review — three detectors (Coherent Power · Zero-shot DINOv3 · Fine-tuned DINOv3)
#
# This is a faithful extension of
# `infocom_evals/.../batch_eval_review.ipynb`: it uses the **same metrics, buckets,
# thresholds, graph functions, and titles**, and simply adds a third detector — our
# **fine-tuned DINOv3** (`M1_ft`) — to every plot.
#
# **Metric consistency (important):** all four detectors are scored by the *same*
# canonical tool, `eval_detector_masks.py`, over the `sweep_20260630` batch run:
# - `coherent_power`, `cuda_dino` (zero-shot) masks are the deployed detectors' own outputs.
# - `finetuned_dino` masks are materialized by `src/gen_finetuned_run.py` into a
#   batch-format run dir, then scored by the identical code path.
#
# So bucketing (bandwidth/pulse-length from the source SigMF `wfgt:` attributes —
# e.g. ZC/METADATA report bandwidth `unknown`), coverage, box-IoU, and the pixel
# metrics are computed identically for every detector. `DET_THRESHOLD` matches the
# original (0.1). See the caveats at the bottom.

# %%
from pathlib import Path
import sys, json, subprocess, warnings
import matplotlib.pyplot as plt
from IPython.display import display
warnings.filterwarnings("ignore")

FT_ROOT      = Path.home() / "Holohub-Signal-Detection/dino_fine_tuning"
DINO_REPO    = Path.home() / "dinov3"
EVAL_DIR     = Path.home() / ("Holohub-Signal-Detection/applications/usrp_wideband_signal_detection"
                              "/infocom_evals/signal_detection_experiments")
BATCH_ROOT   = Path("/tmp/usrp_spectrograms/batch_eval/sweep_20260630")   # deployed masks (for panels)
DETS_ROOT    = FT_ROOT / "notebooks/sweep_detectors"                       # combined multi-detector run root
TABLES_DIR   = FT_ROOT / "notebooks/compare_tables_canonical"             # eval_detector_masks output
CAPTURE_DIRS = [Path.home() / "captures"]
DET_THRESHOLD = 0.1   # coverage>=this counts as 'detected' (matches the original; GT are filled boxes; try 0.05-0.3)

for p in (DINO_REPO, FT_ROOT / "src", EVAL_DIR):
    sys.path.insert(0, str(p))
import eval_viz as v
import mask_eval_metrics as mem
import plot_eval_results as pe
import finetuned_infer as fi
import yaml

# 4 detectors: two deployed baselines + two fine-tuned variants (M1 trained on ≤30 dB, M2 on all dB).
DET_ORDER = ["coherent_power", "cuda_dino", "finetuned_dino", "finetuned_dino_m2"]
DET_LABEL = {"coherent_power": "Coherent Power", "cuda_dino": "Zero-shot DINOv3",
             "finetuned_dino": "Fine-tuned DINOv3 (M1: ≤30 dB)",
             "finetuned_dino_m2": "Fine-tuned DINOv3 (M2: all dB)"}


def show(fig):
    """Display a figure exactly once (closing it prevents the inline backend from
    also auto-rendering it at cell end -> no duplicate graphs)."""
    display(fig); plt.close(fig)


# %%
# Both fine-tuned detectors (for the live spectrogram panels) + the canonical tables.
train_cfg = yaml.safe_load(open(FT_ROOT / "configs/train.yaml"))
ds_meta   = json.loads((FT_ROOT / "data/dataset/dataset_meta.json").read_text())
detector    = fi.FinetunedDetector(str(FT_ROOT / "checkpoints/M1_ft/best.pt"), train_cfg, ds_meta,
                                   threshold=fi.load_threshold(FT_ROOT / "eval_out/M1_ft/eval_meta.json"))
detector_m2 = fi.FinetunedDetector(str(FT_ROOT / "checkpoints/M2_ft/best.pt"), train_cfg, ds_meta,
                                   threshold=fi.load_threshold(FT_ROOT / "eval_out/M2_ft/eval_meta.json"))
FT_MODELS = {"finetuned_dino": detector, "finetuned_dino_m2": detector_m2}

if not (TABLES_DIR / "region_metrics.csv").exists():
    print("Canonical tables missing. Generate the fine-tuned run dirs + tables with:")
    print(f"  python {FT_ROOT/'src/gen_finetuned_run.py'}   # M1 -> finetuned_dino")
    print(f"  python {FT_ROOT/'src/gen_finetuned_run.py'} --ft-ckpt checkpoints/M2_ft/best.pt "
          f"--detector-name finetuned_dino_m2 --ft-eval-meta eval_out/M2_ft/eval_meta.json")
    print(f"  python {EVAL_DIR/'eval_detector_masks.py'} --batch-root {DETS_ROOT} "
          f"--out-dir {TABLES_DIR} --coverage-threshold {DET_THRESHOLD}")
region = pe.load_region(TABLES_DIR / "region_metrics.csv")
frame  = pe.load_frame(TABLES_DIR / "frame_pixel_metrics.csv")
print(f"detectors: {pe._detectors(region)}")
print(f"{len(region)} region rows, {len(frame)} frame rows")


# %%
def load_bundle_dets(stem, frame_number):
    """coherent_power + cuda_dino (from the sweep) + both fine-tuned models (run live), pretty-labeled."""
    b = v.load_frame_bundle_smart(BATCH_ROOT, frame_number, file_stem=stem, capture_dirs=CAPTURE_DIRS)
    row = next(r for r in mem.load_manifest(BATCH_ROOT / "coherent_power" / stem)
               if int(r["frame_number"]) == frame_number)
    dp = v.find_capture_data(stem, CAPTURE_DIRS)
    iq = v.read_frame_iq(dp, int(row["local_file_offset_complex"]), int(row["complex_samples_read"]))
    for name, mdl in FT_MODELS.items():
        b.detector_masks[name] = fi.to_display_grid(mdl.mask_for_iq(iq), b.fft_rows, b.fft_cols)
    b.detector_masks = {DET_LABEL[k]: b.detector_masks[k] for k in DET_ORDER if k in b.detector_masks}
    return b


# %% [markdown]
# ## 1. Single-frame overlay (spectrogram + each detector)

# %%
FILE_STEM = "attenuation_dB_45"   # same capture as the original batch_eval_review.ipynb
FRAME = 100                        # same frame number as the original
b = load_bundle_dets(FILE_STEM, FRAME)
print(f"{FILE_STEM} frame {FRAME}: grid {b.fft_rows}x{b.fft_cols}, {len(b.gt_items)} GT regions")
show(v.plot_frame_panels(b, detectors=list(b.detector_masks.keys())))


# %% [markdown]
# ## 2. Reproducible frame review (annotated + noise-only), all four detectors

# %%
REVIEW_STEM = "attenuation_dB_45"   # same capture, same sampling params/seed as the original
SAMPLE = v.sample_review_frames(BATCH_ROOT, REVIEW_STEM, n_annotated=5, n_noise=1, seed=7)
print(f"{SAMPLE['annotated_available']} annotated / {SAMPLE['noise_available']} noise-only; reviewing {SAMPLE['review_frames']}")
for fr in SAMPLE["review_frames"]:
    b = load_bundle_dets(REVIEW_STEM, fr)
    tag = "NOISE-ONLY" if fr in SAMPLE["noise_frames"] else f"{len(b.gt_items)} GT regions"
    print(f"--- frame {fr}: {tag} ---")
    show(v.plot_frame_panels(b, detectors=list(b.detector_masks.keys())))


# %% [markdown]
# ## 3. Performance vs power, faceted by signal class / bandwidth / pulse length
# (Same `plot_eval_results` graphs and titles as the original notebook, one line per detector.)

# %%
show(pe.fig_rate_vs_power_by(region, "signal_class", None, DET_THRESHOLD,
                             "Detection rate vs power, by signal class"))
show(pe.fig_rate_vs_power_by(region, "bandwidth", pe.BW_ORDER, DET_THRESHOLD,
                             "Detection rate vs power, by bandwidth"))
show(pe.fig_rate_vs_power_by(region, "pulse_length", pe.LEN_ORDER, DET_THRESHOLD,
                             "Detection rate vs power, by pulse length (time duration)"))

# %% [markdown]
# ## 4. Performance (detection rate / box-IoU / coverage) vs bandwidth and vs pulse length

# %%
show(pe.fig_metric_vs_bucket(region, "bandwidth", pe.BW_ORDER, DET_THRESHOLD,
                             "Performance vs signal bandwidth"))
show(pe.fig_metric_vs_bucket(region, "pulse_length", pe.LEN_ORDER, DET_THRESHOLD,
                             "Performance vs pulse length (time duration)"))

# %% [markdown]
# ## 5. Frame-level accuracy (precision / recall / F1 / pixel-IoU) + false-positive area vs power

# %%
show(pe.fig_frame_metrics_vs_power(frame))


# %% [markdown]
# ## 6. Numeric summary — detection rate vs attenuation (coverage ≥ threshold)

# %%
import pandas as pd
rdf = pd.DataFrame(region); rdf["detected"] = rdf["coverage"].astype(float) >= DET_THRESHOLD
piv = rdf.pivot_table(index="detector", columns="attenuation_db", values="detected", aggfunc="mean").round(2)
piv.index = [DET_LABEL.get(i, i) for i in piv.index]
print(f"Region detection rate vs attenuation (coverage ≥ {DET_THRESHOLD}):"); display(piv)
cls = rdf.pivot_table(index="signal_class", columns="detector", values="detected", aggfunc="mean").round(2)
cls.columns = [DET_LABEL.get(c, c) for c in cls.columns]
print("\nDetection rate by waveform class (all attenuations):"); display(cls)


# %% [markdown]
# ## Caveats (read before quoting numbers)
#
# 1. **Metrics are identical across detectors** — all scored by `eval_detector_masks.py`
#    over `sweep_20260630`; only the masks differ. Baselines are the deployed detectors' own
#    outputs, unmodified.
# 2. **GT includes buried signals** — annotation boxes mark where a transmitter was, even below
#    the noise floor at high attenuation; pixel recall/IoU is therefore low for *all* detectors there.
# 3. **Pixel-IoU favors the fine-tuned model** (trained to reproduce the filled-box convention);
#    the sparse energy masks of the other detectors are penalized on IoU even when they localize the
#    signal. **Region detection rate is the fair headline.**
# 4. **Geometry** — batch frames are 512×10240 (nfft=10240); the fine-tuned model runs at nfft=1024
#    and its mask is MAX-pooled onto the display grid (preserves thin Zadoff-Chu pulses).
# 5. `DET_THRESHOLD` matches the original notebook (0.1). Regenerate tables via
#    `src/gen_finetuned_run.py` + `eval_detector_masks.py` (see the setup cell).
