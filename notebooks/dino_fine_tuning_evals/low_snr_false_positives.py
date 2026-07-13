# %% [markdown]
# # Low-SNR false positives: where the M2 fine-tuned DINOv3 hallucinates
#
# The M2 fine-tuned model (trained on **all** attenuations) is the best *detector* at
# low SNR — but that sensitivity has a cost: at 55–60 dB it starts flagging **noise**
# as signal. This notebook characterizes and *shows* that failure mode, contrasting
# it with the conservative M1 model (trained on ≤30 dB, which never false-alarms on
# pure noise) and the deployed detectors.
#
# Detectors (masks read from `notebooks/sweep_detectors/`, the same masks scored by
# `eval_detector_masks.py`):
# `coherent_power`, `cuda_dino` (zero-shot), `finetuned_dino` (M1: ≤30 dB),
# `finetuned_dino_m2` (M2: all dB).
#
# **Definition.** A *false positive* is a predicted-signal pixel where the ground
# truth is noise. On a **noise-only frame** (no annotations) *every* on-pixel is a
# false positive. `fp_area_fraction` = FP pixels / (non-signal pixels).

# %%
from pathlib import Path
import os
import sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
warnings.filterwarnings("ignore")

FT_ROOT      = Path.home() / "Holohub-Signal-Detection/dino_fine_tuning"
EVAL_ROOT    = Path(os.environ.get("DINO_EVAL_ROOT",
                    str(Path.home() / "Holohub-Signal-Detection/notebooks/dino_fine_tuning_evals")))   # eval tables + mask-sweep dirs; override via env DINO_EVAL_ROOT
DINO_REPO    = Path.home() / "dinov3"
EVAL_DIR     = Path.home() / ("Holohub-Signal-Detection/applications/usrp_wideband_signal_detection"
                              "/infocom_evals/signal_detection_experiments")
DETS_ROOT    = EVAL_ROOT / "sweeps" / "sweep_detectors"          # 4 detectors' masks on disk
TABLES_DIR   = EVAL_ROOT / "compare_tables" / "compare_tables_canonical"
CAPTURE_DIRS = [Path.home() / "captures"]
FIG_DIR      = FT_ROOT / "reports/figs_false_positives"; FIG_DIR.mkdir(parents=True, exist_ok=True)

for p in (DINO_REPO, FT_ROOT / "src", EVAL_DIR):
    sys.path.insert(0, str(p))
import eval_viz as v
import mask_eval_metrics as mem

DET_ORDER = ["coherent_power", "cuda_dino", "finetuned_dino", "finetuned_dino_m2"]
DET_LABEL = {"coherent_power": "Coherent Power", "cuda_dino": "Zero-shot DINOv3",
             "finetuned_dino": "Fine-tuned DINOv3 (M1: ≤30 dB)",
             "finetuned_dino_m2": "Fine-tuned DINOv3 (M2: all dB)"}
DET_COLOR = {"coherent_power": "#d95f02", "cuda_dino": "#7570b3",
             "finetuned_dino": "#1b9e77", "finetuned_dino_m2": "#e7298a"}


def show(fig):
    display(fig); plt.close(fig)


def load_panel(stem, frame):
    """4-detector bundle straight from disk (no inference), pretty-labeled + ordered."""
    b = v.load_frame_bundle_smart(DETS_ROOT, frame, file_stem=stem, capture_dirs=CAPTURE_DIRS)
    b.detector_masks = {DET_LABEL[k]: b.detector_masks[k] for k in DET_ORDER if k in b.detector_masks}
    return b


# %%
frame = pd.read_csv(TABLES_DIR / "frame_pixel_metrics.csv")
frame["noise_only"] = frame.gt_pixels == 0
print("detectors:", sorted(frame.detector.unique()))
print("noise-only frames per attenuation (shared across detectors):")
print(frame[frame.detector == "coherent_power"].groupby("attenuation_db")["noise_only"].sum().astype(int).to_string())


# %% [markdown]
# ## 1. Quantitative: false-positive behavior vs SNR
#
# Left: mean FP area on **noise-only** frames. Right: mean FP area on **signal** frames
# (FP = predicted signal outside the annotation boxes). M1 stays at ~0 on noise; M2
# and the deployed detectors rise as SNR drops.

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
for pane, (mask_noise, title) in zip(axes, [(True, "NOISE-ONLY frames"), (False, "SIGNAL frames")]):
    sub = frame[frame.noise_only == mask_noise]
    for k in DET_ORDER:
        g = sub[sub.detector == k].groupby("attenuation_db")["fp_area_fraction"].mean()
        pane.plot(g.index, g.values, "-o", ms=4, label=DET_LABEL[k], color=DET_COLOR[k])
    pane.set_title(f"Mean false-positive area — {title}")
    pane.set_xlabel("attenuation (dB) — higher = lower SNR"); pane.grid(alpha=0.3)
axes[0].set_ylabel("mean fp_area_fraction"); axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG_DIR / "fp_area_vs_snr.png", dpi=110); show(fig)

# %%
# The clearest contrast: how OFTEN each detector fires on a pure-noise frame.
fig, ax = plt.subplots(figsize=(9, 5))
noise = frame[frame.noise_only]
for k in DET_ORDER:
    g = noise[noise.detector == k].assign(anyfp=noise.pred_pixels > 0).groupby("attenuation_db")["anyfp"].mean()
    ax.plot(g.index, g.values, "-o", label=DET_LABEL[k], color=DET_COLOR[k])
ax.set_title("Fraction of NOISE-ONLY frames with ≥1 false-positive pixel")
ax.set_xlabel("attenuation (dB) — higher = lower SNR"); ax.set_ylabel("fraction of noise frames flagged")
ax.grid(alpha=0.3); ax.set_ylim(-0.02, 1.05); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG_DIR / "noise_frame_fire_rate.png", dpi=110); show(fig)

# %%
# M2's hallucinations are EPISODIC: on most noise frames it predicts nothing, but on a
# growing minority it emits large blobs. Distribution of M2 on-pixels on noise frames.
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, db in zip(axes, [55, 60]):
    m2 = frame[(frame.detector == "finetuned_dino_m2") & (frame.noise_only) & (frame.attenuation_db == db)]
    ax.hist(m2.pred_pixels / m2.total_pixels.iloc[0] * 100, bins=30, color=DET_COLOR["finetuned_dino_m2"])
    ax.set_title(f"M2 predicted area on noise frames — {db} dB")
    ax.set_xlabel("% of frame predicted 'signal'"); ax.set_ylabel("# noise frames"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG_DIR / "m2_noise_blob_hist.png", dpi=110); show(fig)

# %%
# Summary table at the two lowest-SNR levels.
lo = frame[frame.attenuation_db.isin([55, 60]) & frame.noise_only]
tbl = lo.groupby(["attenuation_db", "detector"]).agg(
    mean_fp_area=("fp_area_fraction", "mean"),
    frac_frames_flagged=("pred_pixels", lambda s: (s > 0).mean()),
    mean_fp_pixels=("pred_pixels", "mean")).round(4)
print("False positives on NOISE-ONLY frames at 55 & 60 dB:"); display(tbl)


# %% [markdown]
# ## 2. Gallery: M2 hallucinating on pure-noise frames (M1 stays clean)
#
# These are **noise-only** frames (no signal present). We pick the frames where M2
# lights up the most while M1 predicts nothing — pure false alarms. Panels:
# spectrogram, then each detector's mask. M1 (green) is empty; M2 (pink) paints blobs
# on noise; coherent power shows only its low-level speckle.

# %%
def pick_noise_halluc(db, n=4):
    pv = frame.pivot_table(index=["file_stem", "frame_number"], columns="detector",
                           values="pred_pixels", aggfunc="first")
    gt = frame[frame.detector == "coherent_power"].set_index(["file_stem", "frame_number"])
    pv = pv.join(gt[["attenuation_db", "gt_pixels"]])
    cand = pv[(pv.attenuation_db == db) & (pv.gt_pixels == 0) & (pv.finetuned_dino == 0)]
    return cand.sort_values("finetuned_dino_m2", ascending=False).head(n).index.tolist()

for db in (60, 55):
    print(f"\n===== NOISE-ONLY hallucinations @ {db} dB =====")
    for stem, fr in pick_noise_halluc(db, n=4):
        b = load_panel(stem, fr)
        counts = {k: int((b.detector_masks[DET_LABEL[k]] > 0).sum()) for k in DET_ORDER}
        print(f"frame {fr}: FP pixels -> " + ", ".join(f"{DET_LABEL[k].split('(')[0].strip()}={counts[k]}" for k in DET_ORDER))
        fig = v.plot_frame_panels(b, detectors=list(b.detector_masks.keys()))
        fig.suptitle(f"{stem} ({db} dB) frame {fr} — NOISE-ONLY (all detections are false positives)", y=1.02)
        show(fig)


# %% [markdown]
# ## 3. Gallery: M2 over-filling on signal frames at 60 dB
#
# On frames that *do* contain a (buried) signal, M2 not only finds it but paints far
# beyond it — large false-positive area vs M1/coherent. GT boxes are drawn in red.

# %%
sigpv = frame.pivot_table(index=["file_stem", "frame_number"], columns="detector",
                          values="fp_area_fraction", aggfunc="first")
sg = frame[frame.detector == "coherent_power"].set_index(["file_stem", "frame_number"])
sigpv = sigpv.join(sg[["attenuation_db", "gt_pixels"]])
sig60 = sigpv[(sigpv.attenuation_db == 60) & (sigpv.gt_pixels > 0)].copy()
sig60["gap"] = sig60.finetuned_dino_m2 - sig60[["coherent_power", "finetuned_dino"]].max(axis=1)
for stem, fr in sig60.sort_values("gap", ascending=False).head(3).index.tolist():
    b = load_panel(stem, fr)
    print(f"{stem} frame {fr}: {len(b.gt_items)} GT regions")
    fig = v.plot_frame_panels(b, detectors=list(b.detector_masks.keys()))
    fig.suptitle(f"{stem} (60 dB) frame {fr} — M2 over-fills beyond the buried signal", y=1.02)
    show(fig)


# %% [markdown]
# ## 4. Interpretation
#
# - **M1 (≤30 dB training)** is *conservative*: it never fired on a pure-noise frame in
#   the whole sweep (0% flagged), but it also misses most signal below ~45 dB.
# - **M2 (all-dB training)** is *sensitive*: it detects far more real signal at low SNR,
#   but having learned to pull faint structure out of noise, it also latches onto
#   noise that superficially resembles it — episodic large false-positive blobs,
#   on ~7.5% of noise frames at 55 dB rising to ~35% at 60 dB.
# - **Coherent power** emits constant low-level speckle (fires on ~100% of noise frames
#   but with tiny area); **zero-shot DINOv3** scatters moderate false positives.
#
# This is the classic sensitivity/precision trade-off at the noise floor. Practical
# levers to tame M2's low-SNR false alarms (future work): raise its decision threshold
# at low SNR, add a coherence/CFAR gate, train with more/harder noise-only negatives,
# or ensemble M1∧M2 (require both to agree at very low SNR).
#
# **Caveat:** GT marks annotation boxes; some "signal-frame" FP area is energy just
# outside a box that is still real leakage. The noise-only gallery (§2) is the cleanest
# evidence since those frames contain no transmitted signal at all.
