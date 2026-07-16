# %% [markdown]
# # LTE out-of-distribution evaluation
#
# The fine-tuned models were trained on non-LTE waveforms; **LTE is an unseen (OOD)
# class**. This notebook runs the batch-eval-style analysis (detection / recall /
# coverage / false-positive vs SNR + example panels) **focused on the LTE class**.
#
# **How do YOLO26 and fine-tuned DINOv3 generalize to OOD data?** Both were trained on
# non-LTE waveforms, so LTE is an unseen class. Detectors present: **YOLO26s/m
# (fine-tuned)** and **Fine-tuned DINO M1/M2**, all scored by the canonical
# `eval_detector_masks.py` over the LTE captures. (Coherent Power / Zero-shot DINOv3
# are unavailable for LTE — they'd need a container run over `~/captures/lte`; any dirs
# added under the LTE sweep auto-appear on every plot.)

# %%
from pathlib import Path
import os
import sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
warnings.filterwarnings("ignore")

EVAL_ROOT    = Path(os.environ.get("YOLO_EVAL_ROOT",
                    str(Path.home() / "Holohub-Signal-Detection/yolo_training/eval")))   # override via env YOLO_EVAL_ROOT
EVAL_DIR     = Path.home() / ("Holohub-Signal-Detection/applications/usrp_wideband_signal_detection"
                              "/infocom_evals/signal_detection_experiments")
DETS_ROOT    = EVAL_ROOT / "sweeps" / "sweep_lte"
TABLES_DIR   = EVAL_ROOT / "compare_tables_lte"
CAPTURE_DIRS = [Path.home() / "captures/lte"]
FIG_DIR      = Path.home() / "Holohub-Signal-Detection/yolo_training/reports/figs_lte"; FIG_DIR.mkdir(parents=True, exist_ok=True)
DET_THRESHOLD = 0.1
TARGET = "LTE"

sys.path.insert(0, str(EVAL_DIR))
import eval_viz as v
import mask_eval_metrics as mem

# fixed styling; only detectors actually present are plotted (baselines auto-appear later)
ALL_ORDER = ["coherent_power", "cuda_dino", "finetuned_dino", "finetuned_dino_m2", "yolo26s", "yolo26m"]
DET_LABEL = {"coherent_power": "Coherent Power", "cuda_dino": "Zero-shot DINOv3",
             "finetuned_dino": "Fine-tuned DINOv3 (M1: ≤30 dB)",
             "finetuned_dino_m2": "Fine-tuned DINOv3 (M2: all dB)",
             "yolo26s": "YOLO26s (fine-tuned)", "yolo26m": "YOLO26m (fine-tuned)"}
DET_COLOR = {"coherent_power": "#d95f02", "cuda_dino": "#7570b3",
             "finetuned_dino": "#1b9e77", "finetuned_dino_m2": "#e7298a",
             "yolo26s": "#66a61e", "yolo26m": "#a6761d"}
DET_SHORT = {"coherent_power": "Coherent", "cuda_dino": "Zero-shot",
             "finetuned_dino": "M1", "finetuned_dino_m2": "M2",
             "yolo26s": "YOLO-s", "yolo26m": "YOLO-m"}  # compact printouts


def show(fig):
    display(fig); plt.close(fig)


# %%
region = pd.read_csv(TABLES_DIR / "region_metrics.csv").rename(columns={"bucket_signal_class": "signal_class"})
frame  = pd.read_csv(TABLES_DIR / "frame_pixel_metrics.csv")
region["coverage"] = pd.to_numeric(region["coverage"], errors="coerce")
region["detected"] = region["coverage"] >= DET_THRESHOLD
frame["noise_only"] = frame.gt_pixels == 0
DET_ORDER = [d for d in ALL_ORDER if d in set(region.detector.unique())]
attn = sorted(region.attenuation_db.dropna().unique())
print("detectors present:", DET_ORDER)
print("classes:", sorted(region.signal_class.unique()))
lte = region[region.signal_class == TARGET].copy()
print(f"LTE region rows: {len(lte)} | per attenuation (one detector):")
print(lte[lte.detector == DET_ORDER[0]].groupby("attenuation_db").size().to_string())


# %%
def line_vs_snr(df, value, ylabel, title, fname, agg="mean"):
    fig, ax = plt.subplots(figsize=(9, 5))
    for k in DET_ORDER:
        g = df[df.detector == k].groupby("attenuation_db")[value].agg(agg)
        ax.plot(g.index, g.values, "-o", label=DET_LABEL[k], color=DET_COLOR[k])
    ax.set_title(title); ax.set_xlabel("attenuation (dB) — higher = lower SNR")
    ax.set_ylabel(ylabel); ax.grid(alpha=0.3); ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / fname, dpi=110); show(fig)


# %% [markdown]
# ## 1. LTE detection rate vs SNR  (coverage ≥ 0.1)

# %%
line_vs_snr(lte, "detected", "detection rate",
            f"LTE (OOD) detection rate vs attenuation (coverage ≥ {DET_THRESHOLD})",
            "lte_detection_vs_snr.png")

# %% [markdown]
# ## 2. LTE recall / coverage vs SNR  (mean fraction of each LTE box recovered)

# %%
line_vs_snr(lte, "coverage", "mean box coverage (recall of the box)",
            "LTE (OOD) box coverage vs attenuation", "lte_coverage_vs_snr.png")

# %% [markdown]
# ## 3. Frame-level pixel metrics on LTE-bearing frames
# (Frames containing ≥1 LTE region; note such frames may also contain ZC/METADATA.)

# %%
lte_frames = set(map(tuple, lte[["file_stem", "frame_number"]].drop_duplicates().values))
fkey = frame.set_index(["file_stem", "frame_number"]).index
fr_lte = frame[frame.set_index(["file_stem", "frame_number"]).index.isin(lte_frames)].copy()
fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
for ax, (col, lab) in zip(axes, [("recall", "recall"), ("precision", "precision"),
                                 ("f1", "F1"), ("iou", "pixel IoU")]):
    for k in DET_ORDER:
        g = fr_lte[(fr_lte.detector == k) & (fr_lte.mask_present)].groupby("attenuation_db")[col].mean()
        ax.plot(g.index, g.values, "-o", label=DET_LABEL[k], color=DET_COLOR[k])
    ax.set_title(f"{lab} (LTE-bearing frames)"); ax.set_xlabel("attenuation (dB)")
    ax.grid(alpha=0.3); ax.set_ylim(-0.02, 1.02)
axes[0].set_ylabel("value"); axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG_DIR / "lte_frame_metrics_vs_snr.png", dpi=110); show(fig)

# %% [markdown]
# ## 4. False positives on NOISE-ONLY frames vs SNR
# (Frames with no transmitted signal — every detection is a false positive.)

# %%
noise = frame[frame.noise_only]
if len(noise):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for k in DET_ORDER:
        g = noise[noise.detector == k].groupby("attenuation_db")["fp_area_fraction"].mean()
        axes[0].plot(g.index, g.values, "-o", label=DET_LABEL[k], color=DET_COLOR[k])
        gf = noise[noise.detector == k].assign(anyfp=noise.pred_pixels > 0).groupby("attenuation_db")["anyfp"].mean()
        axes[1].plot(gf.index, gf.values, "-o", label=DET_LABEL[k], color=DET_COLOR[k])
    axes[0].set_title("Mean false-positive area on noise-only frames"); axes[0].set_ylabel("fp_area_fraction")
    axes[1].set_title("Fraction of noise-only frames with ≥1 false positive"); axes[1].set_ylabel("fraction flagged")
    for ax in axes:
        ax.set_xlabel("attenuation (dB) — higher = lower SNR"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[1].set_ylim(-0.02, 1.05)
    fig.tight_layout(); fig.savefig(FIG_DIR / "lte_false_positives_vs_snr.png", dpi=110); show(fig)
else:
    print("No noise-only frames in the LTE set (every frame contains annotations).")

# %% [markdown]
# ## 5. Low-SNR summary table (LTE, 30–60 dB)

# %%
lo = [a for a in attn if a >= 30]
det_tbl = lte.pivot_table(index="detector", columns="attenuation_db", values="detected", aggfunc="mean").reindex(DET_ORDER)
det_tbl.index = [DET_LABEL[i] for i in det_tbl.index]
print(f"LTE detection rate (coverage ≥ {DET_THRESHOLD}):"); display(det_tbl[[a for a in lo if a in det_tbl.columns]].round(2))
cov_tbl = lte.pivot_table(index="detector", columns="attenuation_db", values="coverage", aggfunc="mean").reindex(DET_ORDER)
cov_tbl.index = [DET_LABEL[i] for i in cov_tbl.index]
print("LTE mean box coverage:"); display(cov_tbl[[a for a in lo if a in cov_tbl.columns]].round(2))


# %% [markdown]
# ## 6. Example panels: LTE across SNR
# For each attenuation we show the LTE-bearing frame where M2 recovers the most —
# spectrogram + each detector's mask.

# %%
def panel_lte(db):
    d = lte[(lte.attenuation_db == db) & (lte.detector == "finetuned_dino_m2")]
    if d.empty:
        return
    row = d.loc[d.coverage.idxmax()]
    stem, fr = row.file_stem, int(row.frame_number)
    b = v.load_frame_bundle_smart(DETS_ROOT, fr, file_stem=stem, capture_dirs=CAPTURE_DIRS)
    covs = {k: lte[(lte.detector == k) & (lte.file_stem == stem) & (lte.frame_number == fr)].coverage.mean()
            for k in DET_ORDER}
    print(f"[{db} dB] {stem} frame {fr} — LTE box coverage: " +
          ", ".join(f"{DET_SHORT[k]}={covs[k]:.2f}" for k in DET_ORDER))
    b.detector_masks = {DET_LABEL[k]: b.detector_masks[k] for k in DET_ORDER if k in b.detector_masks}
    fig = v.plot_frame_panels(b, detectors=list(b.detector_masks.keys()))
    fig.suptitle(f"LTE (OOD) — {stem} ({db} dB) frame {fr}", y=1.02); show(fig)

for db in [0, 30, 45, 55, 60]:
    dbs = [a for a in attn if abs(a - db) <= 3]
    if dbs:
        panel_lte(dbs[0])

# %% [markdown]
# ### Noise-only frame (no transmitted signal) — every detection here is a false positive.
# We pick the noise-only frame where M2 predicts the most (worst-case); if the models are
# clean it just shows an empty frame.

# %%
noise_fr = frame[frame.noise_only]
if len(noise_fr):
    m2n = noise_fr[noise_fr.detector == "finetuned_dino_m2"]
    row = (m2n.loc[m2n.pred_pixels.idxmax()] if m2n.pred_pixels.max() > 0
           else m2n.iloc[len(m2n) // 2])
    stem, fr = row.file_stem, int(row.frame_number)
    b = v.load_frame_bundle_smart(DETS_ROOT, fr, file_stem=stem, capture_dirs=CAPTURE_DIRS)
    on = {DET_SHORT[k]: 100 * (b.detector_masks[k] > 0).mean() for k in DET_ORDER if k in b.detector_masks}
    print(f"[NOISE-ONLY] {stem} frame {fr} ({row.attenuation_db:.0f} dB) — % of frame flagged: " +
          ", ".join(f"{k}={p:.2f}%" for k, p in on.items()))
    b.detector_masks = {DET_LABEL[k]: b.detector_masks[k] for k in DET_ORDER if k in b.detector_masks}
    fig = v.plot_frame_panels(b, detectors=list(b.detector_masks.keys()))
    fig.suptitle(f"LTE set — NOISE-ONLY {stem} frame {fr} (all detections are false positives)", y=1.02)
    show(fig)
else:
    print("No noise-only frames in the LTE set.")


# %% [markdown]
# ## 7. Interpretation & how to add the deployed baselines
#
# **OOD takeaway:** although LTE was never in training, the fine-tuned detectors treat
# it as ordinary signal energy — detection/coverage curves vs SNR look like the
# in-distribution wideband classes, with **M2 (all-dB training) again holding deepest
# into low SNR**. (Fill in the M1-vs-M2 crossover from the tables above.)
#
# **Adding Coherent Power + Zero-shot DINOv3 (collaborator):** run the deployed offline
# pipeline on `~/captures/lte` with the standard 512×10240 framing, then place the
# resulting `coherent_power/` and `cuda_dino/` run dirs under
# `notebooks/sweep_lte/` (siblings of `finetuned_dino*`). Regenerate the tables:
# ```
# python <infocom_evals>/eval_detector_masks.py --batch-root notebooks/sweep_lte \
#        --captures-dir ~/captures/lte --out-dir notebooks/compare_tables_lte \
#        --coverage-threshold 0.1
# ```
# Re-run this notebook — the two baselines will appear automatically on every plot.
# (GT is identical across detectors; scoring is the same code path for all four.)
