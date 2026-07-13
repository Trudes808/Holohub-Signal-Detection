# %% [markdown]
# # Wide-bandwidth waveforms at low SNR: OFDM · 5G Downlink · 802.11ax
#
# Focused final comparison of the four detectors on the **wide-bandwidth** signal
# classes, where the question is: *how deep into low SNR does each method keep
# detecting?*
#
# Detectors: **Coherent Power**, **Zero-shot DINOv3** (`cuda_dino`),
# **Fine-tuned M1** (≤30 dB training), **Fine-tuned M2** (all-dB training).
# All scored by the canonical `eval_detector_masks.py` (masks in
# `notebooks/sweep_detectors/`; tables in `notebooks/compare_tables_canonical/`).
#
# Metrics: **detection rate** = fraction of GT regions with box coverage ≥ 0.1;
# **coverage** = mean fraction of each GT box the detector actually fills (how much of
# the wideband signal is recovered).

# %%
from pathlib import Path
import os
import sys, warnings
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
DETS_ROOT    = EVAL_ROOT / "sweeps" / "sweep_detectors"
TABLES_DIR   = EVAL_ROOT / "compare_tables" / "compare_tables_canonical"
CAPTURE_DIRS = [Path.home() / "captures"]
FIG_DIR      = FT_ROOT / "reports/figs_wideband"; FIG_DIR.mkdir(parents=True, exist_ok=True)
DET_THRESHOLD = 0.1

for p in (DINO_REPO, FT_ROOT / "src", EVAL_DIR):
    sys.path.insert(0, str(p))
import eval_viz as v
import mask_eval_metrics as mem

CLASSES   = ["OFDM", "5G_Downlink", "802_11ax"]
DET_ORDER = ["coherent_power", "cuda_dino", "finetuned_dino", "finetuned_dino_m2"]
DET_LABEL = {"coherent_power": "Coherent Power", "cuda_dino": "Zero-shot DINOv3",
             "finetuned_dino": "Fine-tuned DINOv3 (M1: ≤30 dB)",
             "finetuned_dino_m2": "Fine-tuned DINOv3 (M2: all dB)"}
DET_COLOR = {"coherent_power": "#d95f02", "cuda_dino": "#7570b3",
             "finetuned_dino": "#1b9e77", "finetuned_dino_m2": "#e7298a"}
DET_SHORT = {"coherent_power": "Coherent", "cuda_dino": "Zero-shot",
             "finetuned_dino": "M1", "finetuned_dino_m2": "M2"}  # for compact printouts


def show(fig):
    display(fig); plt.close(fig)


# %%
r = pd.read_csv(TABLES_DIR / "region_metrics.csv")
r = r.rename(columns={"bucket_signal_class": "signal_class"})
r["coverage"] = pd.to_numeric(r["coverage"], errors="coerce")
r["detected"] = r["coverage"] >= DET_THRESHOLD
wb = r[r.signal_class.isin(CLASSES)].copy()
attn = sorted(wb.attenuation_db.dropna().unique())
print("wide-bandwidth region rows:", len(wb))
print("occupied-bandwidth mix per class:")
print(wb.groupby("signal_class")["bucket_bandwidth"].value_counts().to_string())


# %% [markdown]
# ## 1. Detection rate vs attenuation — per waveform class

# %%
def faceted(metric, ylabel, title, fname, agg="mean"):
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(6 * len(CLASSES), 4.6), sharey=True)
    for ax, cls in zip(axes, CLASSES):
        d = wb[wb.signal_class == cls]
        for k in DET_ORDER:
            g = d[d.detector == k].groupby("attenuation_db")[metric].agg(agg)
            ax.plot(g.index, g.values, "-o", ms=4, label=DET_LABEL[k], color=DET_COLOR[k])
        n = d[d.detector == "coherent_power"].groupby("attenuation_db").size().sum()
        ax.set_title(f"{cls}  (n={n} regions)"); ax.set_xlabel("attenuation (dB) — higher = lower SNR")
        ax.grid(alpha=0.3); ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel(ylabel); axes[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=13); fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=110); show(fig)

faceted("detected", "detection rate", f"Detection rate vs attenuation (coverage ≥ {DET_THRESHOLD})",
        "detection_vs_snr_by_class.png")


# %% [markdown]
# ## 2. Coverage vs attenuation — how much of the wideband signal is recovered

# %%
faceted("coverage", "mean box coverage", "Mean box coverage vs attenuation (fraction of GT box filled)",
        "coverage_vs_snr_by_class.png")


# %% [markdown]
# ## 3. Pooled wide-bandwidth performance + split by occupied bandwidth

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
# pooled over the 3 wideband classes
for k in DET_ORDER:
    g = wb[wb.detector == k].groupby("attenuation_db")["detected"].mean()
    axes[0].plot(g.index, g.values, "-o", label=DET_LABEL[k], color=DET_COLOR[k])
axes[0].set_title("Pooled OFDM+5G+802.11ax — detection rate vs attenuation")
axes[0].set_ylabel("detection rate")
# focus on the wide (>=25 MHz) subset only
wide = wb[wb.bucket_bandwidth.isin(["25-60MHz", ">=60MHz"])]
for k in DET_ORDER:
    g = wide[wide.detector == k].groupby("attenuation_db")["detected"].mean()
    axes[1].plot(g.index, g.values, "-o", label=DET_LABEL[k], color=DET_COLOR[k])
axes[1].set_title("Only the ≥25 MHz-wide subset — detection rate vs attenuation")
for ax in axes:
    ax.set_xlabel("attenuation (dB) — higher = lower SNR"); ax.grid(alpha=0.3); ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG_DIR / "pooled_wideband.png", dpi=110); show(fig)


# %% [markdown]
# ## 4. Low-SNR summary tables (40–60 dB)

# %%
lo = [a for a in attn if a >= 40]
for cls in CLASSES:
    d = wb[wb.signal_class == cls]
    piv = d.pivot_table(index="detector", columns="attenuation_db", values="detected", aggfunc="mean")
    piv = piv.reindex(index=DET_ORDER)[[a for a in lo if a in piv.columns]].round(2)
    piv.index = [DET_LABEL[i] for i in piv.index]
    print(f"\n### {cls} — detection rate (coverage ≥ {DET_THRESHOLD})"); display(piv)


# %% [markdown]
# ## 5. Visual examples at 55 & 60 dB (one row = spectrogram + each detector)
#
# For each class we show the frame at that attenuation where **M2 recovers the most**
# of the (buried) wideband signal — illustrating detections the other methods miss.
# Per-region coverage for the target class is printed above each figure.

# %%
def best_frame(cls, db):
    d = wb[(wb.signal_class == cls) & (wb.attenuation_db == db) & (wb.detector == "finetuned_dino_m2")]
    if d.empty:
        return None
    row = d.loc[d.coverage.idxmax()]
    return row["file_stem"], int(row["frame_number"])

def panel(stem, frame, cls, db):
    b = v.load_frame_bundle_smart(DETS_ROOT, frame, file_stem=stem, capture_dirs=CAPTURE_DIRS)
    # per-class coverage each detector achieves on this frame
    covs = {}
    for k in DET_ORDER:
        rows = wb[(wb.detector == k) & (wb.file_stem == stem) & (wb.frame_number == frame) & (wb.signal_class == cls)]
        covs[k] = rows.coverage.mean() if len(rows) else float("nan")
    print(f"[{cls} @ {db} dB] {stem} frame {frame} — box coverage: " +
          ", ".join(f"{DET_SHORT[k]}={covs[k]:.2f}" for k in DET_ORDER))
    b.detector_masks = {DET_LABEL[k]: b.detector_masks[k] for k in DET_ORDER if k in b.detector_masks}
    fig = v.plot_frame_panels(b, detectors=list(b.detector_masks.keys()))
    fig.suptitle(f"{cls} — {stem} ({db} dB) frame {frame}", y=1.02)
    show(fig)

for cls in CLASSES:
    for db in (55, 60):
        pick = best_frame(cls, db)
        if pick:
            panel(pick[0], pick[1], cls, db)


# %% [markdown]
# ## 6. Widest-bandwidth extremes: 160 MHz 802.11ax and 98.28 MHz 5G NR
#
# The hardest cases — the widest channel of each. Energy is spread across (essentially)
# the whole band, so it clears the noise floor the least. Does detection survive?

# %%
r_all = pd.read_csv(TABLES_DIR / "region_metrics.csv").rename(columns={"bucket_signal_class": "signal_class"})
r_all["occ_mhz"] = (pd.to_numeric(r_all.occupied_bw_hz, errors="coerce") / 1e6).round(2)
r_all["coverage"] = pd.to_numeric(r_all.coverage, errors="coerce")
r_all["detected"] = r_all.coverage >= DET_THRESHOLD
EXTREMES = [("802_11ax", 160.0), ("5G_Downlink", 98.28)]
for cls, bw in EXTREMES:
    d = r_all[(r_all.signal_class == cls) & (r_all.occ_mhz == bw)]
    n = len(d[d.detector == "coherent_power"])
    piv = d.pivot_table(index="detector", columns="attenuation_db", values="detected", aggfunc="mean").reindex(DET_ORDER)
    piv.index = [DET_LABEL[i] for i in piv.index]
    print(f"### {cls} @ {bw:.0f} MHz (widest channel, n={n} regions) — detection rate (cov ≥ {DET_THRESHOLD})")
    display(piv[[a for a in [35, 40, 45, 50, 55, 60] if a in piv.columns]].round(2))

# %%
# Render the best-case (max M2 coverage) frame for each extreme at 55 and 60 dB.
def panel_widest(cls, bw, db):
    d = r_all[(r_all.signal_class == cls) & (r_all.occ_mhz == bw) &
              (r_all.attenuation_db == db) & (r_all.detector == "finetuned_dino_m2")]
    if d.empty:
        print(f"(no {cls} @ {bw:.0f} MHz at {db} dB)"); return
    row = d.loc[d.coverage.idxmax()]
    stem, fr = row.file_stem, int(row.frame_number)
    b = v.load_frame_bundle_smart(DETS_ROOT, fr, file_stem=stem, capture_dirs=CAPTURE_DIRS)
    covs = {k: r_all[(r_all.detector == k) & (r_all.file_stem == stem) & (r_all.frame_number == fr) &
                     (r_all.signal_class == cls) & (r_all.occ_mhz == bw)].coverage.mean() for k in DET_ORDER}
    print(f"[{cls} {bw:.0f} MHz @ {db} dB] {stem} frame {fr} — box coverage: " +
          ", ".join(f"{DET_SHORT[k]}={covs[k]:.2f}" for k in DET_ORDER))
    b.detector_masks = {DET_LABEL[k]: b.detector_masks[k] for k in DET_ORDER if k in b.detector_masks}
    fig = v.plot_frame_panels(b, detectors=list(b.detector_masks.keys()))
    fig.suptitle(f"{cls} {bw:.0f} MHz (widest) — {stem} ({db} dB) frame {fr}", y=1.02)
    show(fig)

for cls, bw in EXTREMES:
    for db in (55, 60):
        panel_widest(cls, bw, db)


# %% [markdown]
# ## 7. Interpretation
#
# On wide-bandwidth waveforms (OFDM / 5G / 802.11ax) at low SNR:
#
# - **Coherent power** and **zero-shot DINOv3** collapse by ~45–50 dB — they need
#   visible energy, which a wideband signal spreads too thin to clear the noise floor.
#   (Zero-shot fails on 802.11ax at essentially all low-SNR levels.)
# - **M1 (≤30 dB training)** extends useful detection to ~45–50 dB, then falls off — it
#   never saw signals this buried during training.
# - **M2 (all-dB training)** is the only method that keeps detecting deep into low SNR:
#   OFDM 0.74 / 0.43, 5G 0.67 / 0.34, 802.11ax 0.29 / 0.02 at 55 / 60 dB, versus ~0 for
#   everything else. Coverage (fraction of the box recovered) shows the same ordering.
# - **At the widest channels even M2 eventually breaks** (§6): 160 MHz 802.11ax holds
#   only to ~50 dB (0.31) then collapses (0.03 / 0.00 at 55 / 60 dB); 98.28 MHz 5G is
#   sturdier (0.45 / 0.10 at 55 / 60 dB). The wider the signal, the thinner its energy
#   per bin, so the harder the low-SNR detection — the widest 802.11ax is the limit.
# - Trade-off: M2's low-SNR sensitivity also raises its false-positive rate on noise —
#   see `low_snr_false_positives.ipynb`.
#
# **Caveat:** GT boxes span the full annotated bandwidth/time even when the signal is
# buried; detection rate (coverage ≥ 0.1) asks "did the detector fire meaningfully
# inside the box," and coverage asks "how much of it." Both are computed identically
# for all four detectors by `eval_detector_masks.py`.
