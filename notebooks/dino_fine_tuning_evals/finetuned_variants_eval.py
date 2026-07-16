# %% [markdown]
# # Fine-tuned variants: frozen head vs. backbone-adapted — is unfreezing worth it?
#
# Four fine-tuned DINOv3 detectors, on the same batch-eval grid + canonical scoring as
# the other comparison notebooks (isolated tables so nothing else is disturbed):
#
# | model | training data | what's trained | trainable params |
# |---|---|---|---|
# | **M1 frozen** | ≤30 dB | segmentation head only (backbone frozen) | ~2.0 M |
# | **M1 ft** | ≤30 dB | head **+ last-4 transformer blocks** | ~30 M |
# | **M2 frozen** | all dB | head only | ~2.0 M |
# | **M2 ft** | all dB | head + last-4 blocks | ~30 M |
#
# The question: does unfreezing the last transformer blocks (`ft`) beat the frozen-head
# probe (`frozen`) by enough to justify the extra **training** compute? (Note: inference
# cost is identical — both run a full ViT-B/16 forward; only training differs.)

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
DETS_ROOT    = EVAL_ROOT / "sweeps" / "sweep_finetuned"
TABLES_DIR   = EVAL_ROOT / "compare_tables" / "compare_tables_finetuned"
CAPTURE_DIRS = [Path.home() / "captures"]
FIG_DIR      = FT_ROOT / "reports/figs_finetuned"; FIG_DIR.mkdir(parents=True, exist_ok=True)
DET_THRESHOLD = 0.1

for p in (DINO_REPO, FT_ROOT / "src", EVAL_DIR):
    sys.path.insert(0, str(p))
import eval_viz as v
import mask_eval_metrics as mem
import plot_eval_results as pe

DET_ORDER = ["m1_frozen", "m1_ft", "m2_frozen", "m2_ft"]
DET_LABEL = {"m1_frozen": "M1 frozen (≤30 dB, head only)", "m1_ft": "M1 ft (≤30 dB, +last-4)",
             "m2_frozen": "M2 frozen (all dB, head only)", "m2_ft": "M2 ft (all dB, +last-4)"}
DET_COLOR = {"m1_frozen": "#66c2a4", "m1_ft": "#1b9e77", "m2_frozen": "#f4a3c6", "m2_ft": "#e7298a"}
DET_STYLE = {"m1_frozen": "--", "m1_ft": "-", "m2_frozen": "--", "m2_ft": "-"}
PAIRS = [("M1 (≤30 dB)", "m1_frozen", "m1_ft"), ("M2 (all dB)", "m2_frozen", "m2_ft")]


def show(fig):
    display(fig); plt.close(fig)


# %%
region = pd.read_csv(TABLES_DIR / "region_metrics.csv").rename(columns={"bucket_signal_class": "signal_class"})
frame  = pd.read_csv(TABLES_DIR / "frame_pixel_metrics.csv")
region["coverage"] = pd.to_numeric(region["coverage"], errors="coerce")
region["detected"] = region["coverage"] >= DET_THRESHOLD
frame["noise_only"] = frame.gt_pixels == 0
present = [d for d in DET_ORDER if d in set(region.detector.unique())]
attn = sorted(region.attenuation_db.dropna().unique())
print("models present:", present)
print(f"{len(region)} region rows, {len(frame)} frame rows")


# %%
def vs_snr(ax, df, value, agg="mean", subset=None):
    for k in present:
        d = df[df.detector == k]
        if subset is not None:
            d = d[subset(d)]
        g = d.groupby("attenuation_db")[value].agg(agg)
        ax.plot(g.index, g.values, DET_STYLE[k] + "o", ms=4, label=DET_LABEL[k], color=DET_COLOR[k])
    ax.set_xlabel("attenuation (dB) — higher = lower SNR"); ax.grid(alpha=0.3)


# %% [markdown]
# ## 1. Detection rate & coverage vs SNR (all four)

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
vs_snr(axes[0], region, "detected"); axes[0].set_title(f"Region detection rate (coverage ≥ {DET_THRESHOLD})")
axes[0].set_ylabel("detection rate"); axes[0].set_ylim(-0.02, 1.02); axes[0].legend(fontsize=8)
vs_snr(axes[1], region, "coverage"); axes[1].set_title("Mean box coverage"); axes[1].set_ylabel("coverage"); axes[1].set_ylim(-0.02, 1.02)
fig.tight_layout(); fig.savefig(FIG_DIR / "detection_coverage_vs_snr.png", dpi=110); show(fig)

# %% [markdown]
# ## 2. Frame-level pixel metrics + false positives vs SNR

# %%
fig, axes = plt.subplots(1, 5, figsize=(22, 4.2))
for ax, (col, lab, sub) in zip(axes, [
        ("iou", "pixel IoU", lambda d: d.gt_pixels > 0), ("recall", "recall", lambda d: d.gt_pixels > 0),
        ("precision", "precision", lambda d: d.gt_pixels > 0), ("f1", "F1", lambda d: d.gt_pixels > 0),
        ("fp_area_fraction", "FP area (noise frames)", lambda d: d.gt_pixels == 0)]):
    vs_snr(ax, frame, col, subset=sub); ax.set_title(lab)
    if col != "fp_area_fraction":
        ax.set_ylim(-0.02, 1.02)
axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG_DIR / "frame_metrics_vs_snr.png", dpi=110); show(fig)

# %% [markdown]
# ## 3. Rich breakdowns (canonical `plot_eval_results`, one line per model)

# %%
show(pe.fig_rate_vs_power_by(pe.load_region(TABLES_DIR / "region_metrics.csv"), "signal_class", None, DET_THRESHOLD,
                             "Detection rate vs power, by signal class"))
show(pe.fig_metric_vs_bucket(pe.load_region(TABLES_DIR / "region_metrics.csv"), "bandwidth", pe.BW_ORDER, DET_THRESHOLD,
                             "Performance vs signal bandwidth"))
show(pe.fig_frame_metrics_vs_power(pe.load_frame(TABLES_DIR / "frame_pixel_metrics.csv")))


# %% [markdown]
# ## 4. The key question — does unfreezing help?  Δ(ft − frozen)
# Positive means the backbone-adapted model beats the frozen-head probe at that SNR.

# %%
def curve(df, det, value, sub=None):
    d = df[df.detector == det]
    if sub is not None:
        d = d[sub(d)]
    return d.groupby("attenuation_db")[value].mean()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for name, fro, ftd in PAIRS:
    d_det = curve(region, ftd, "detected") - curve(region, fro, "detected")
    d_iou = (curve(frame, ftd, "iou", lambda d: d.gt_pixels > 0)
             - curve(frame, fro, "iou", lambda d: d.gt_pixels > 0))
    axes[0].plot(d_det.index, d_det.values, "-o", ms=4, label=name)
    axes[1].plot(d_iou.index, d_iou.values, "-o", ms=4, label=name)
for ax, t in zip(axes, ["Δ detection rate (ft − frozen)", "Δ pixel IoU (ft − frozen)"]):
    ax.axhline(0, color="k", lw=0.8); ax.set_title(t); ax.grid(alpha=0.3)
    ax.set_xlabel("attenuation (dB) — higher = lower SNR"); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(FIG_DIR / "ft_minus_frozen_delta.png", dpi=110); show(fig)


# %% [markdown]
# ## 5. Training compute: frozen vs ft
# Trainable parameters, per-epoch time, and total training time (from each run's
# `history.json`). **Inference cost is identical across all four** (full ViT-B forward).

# %%
import torch  # noqa
sys.path.insert(0, str(FT_ROOT / "src"))
from model import DinoSegmenter
W = str(FT_ROOT / "checkpoints/M1_ft/best.pt")  # weights_path only used for arch; count params per mode
wpath = "/home/bqn82/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
nparam = {}
for mode in ("frozen", "ft_lastN"):
    m = DinoSegmenter(wpath, mode=mode, unfreeze_last_n=4)
    nparam[mode] = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
    del m
print(f"trainable params: frozen={nparam['frozen']:.2f}M   ft_lastN={nparam['ft_lastN']:.2f}M "
      f"({nparam['ft_lastN']/nparam['frozen']:.1f}x)")

rows = []
for k in DET_ORDER:
    ck = {"m1_frozen": "M1_frozen", "m1_ft": "M1_ft", "m2_frozen": "M2_frozen", "m2_ft": "M2_ft"}[k]
    h = json.loads((FT_ROOT / f"checkpoints/{ck}/history.json").read_text())
    mode = "ft_lastN" if k.endswith("_ft") else "frozen"
    rows.append({"model": DET_LABEL[k], "epochs": len(h),
                 "trainable_M": round(nparam[mode], 2),
                 "sec_per_epoch": round(np.mean([e["time_s"] for e in h])),
                 "total_train_min": round(sum(e["time_s"] for e in h) / 60, 1),
                 "best_val_iou": round(max(e["iou"] for e in h), 3)})
comp = pd.DataFrame(rows); display(comp)
for name, fro, ftd in PAIRS:
    tf = comp.set_index("model")
    a = tf.loc[DET_LABEL[fro], "total_train_min"]; b = tf.loc[DET_LABEL[ftd], "total_train_min"]
    print(f"{name}: ft training cost = {b/a:.2f}x frozen  ({a:.0f} → {b:.0f} min)")


# %% [markdown]
# ## 6. Example panels: all four variants on the same frames (across SNR)

# %%
def panel(stem, frame_number, tag):
    b = v.load_frame_bundle_smart(DETS_ROOT, frame_number, file_stem=stem, capture_dirs=CAPTURE_DIRS)
    b.detector_masks = {DET_LABEL[k]: b.detector_masks[k] for k in present if k in b.detector_masks}
    fig = v.plot_frame_panels(b, detectors=list(b.detector_masks.keys()))
    fig.suptitle(f"{tag} — {stem} frame {frame_number}", y=1.02); show(fig)

# pick, per attenuation, the frame where the ft/frozen split is most visible (max |m2_ft − m2_frozen| coverage)
for db in [0, 45, 55, 60]:
    dd = [a for a in attn if abs(a - db) <= 3]
    if not dd:
        continue
    rr = region[(region.attenuation_db == dd[0])]
    piv = rr.pivot_table(index=["file_stem", "frame_number"], columns="detector", values="coverage", aggfunc="mean")
    if not len(piv) or "m2_ft" not in piv or "m2_frozen" not in piv:
        continue
    piv["gap"] = (piv["m2_ft"] - piv["m2_frozen"]).abs()
    stem, fr = piv["gap"].idxmax()
    panel(stem, int(fr), f"{int(dd[0])} dB")


# %% [markdown]
# ## 7. Verdict — is unfreezing worth it?
#
# _(Filled from the figures/tables above.)_ Summary of what to look for:
# - **§4 Δ curves** are the crux: if Δ(ft − frozen) hovers near 0 (or dips negative),
#   the frozen-head probe is as good as the backbone-adapted model and the extra
#   training compute (**§5**, ~1.5×) is **not** justified.
# - Watch the split by **data regime**: unfreezing may help the ≤30 dB model (M1) more
#   than the all-dB model (M2), or vice-versa.
# - Inference cost is the **same** for frozen and ft, so this is purely a
#   training-compute-vs-accuracy trade.
