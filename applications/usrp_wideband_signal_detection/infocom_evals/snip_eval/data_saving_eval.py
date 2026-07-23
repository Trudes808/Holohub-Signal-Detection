# %% [markdown]
# # Data-saving evaluation — per-SNR visualizations + tables
# Reduction (x vs naive save-all) + signal retention per detector, on a physical **SNR axis** mapped
# from attenuation via the baseline SNR calibration (`snr_measurement`, which `build_snr_results` wraps):
# `snr_db = snr0_ref - attenuation_db`, with `snr0_ref` = median calibrated 0 dB SNR of the signals.
# Numbers come from `plot_data_saving.py` (snipper frequency-mode byte accounting + thresholded time-slice).

# %%
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

BATCH = Path(os.environ.get("DS_BATCH_ROOT", "/tmp/ds_batch"))
OUT = BATCH / "data_saving_figs"
TABLE = OUT / "data_saving_table.csv"
CALIB = BATCH / "snr_calibration.json"
CAP0 = Path("/home/bqn82/captures/attenuation_dB_0.sigmf-data")
_BC = Path.home()/"Holohub-Signal-Detection/applications/usrp_wideband_signal_detection/infocom_evals/baseline_comparisons"

# %% [markdown]
# ## SNR mapping (baseline calibration -> snr0_ref)
# %%
if CALIB.exists():
    snr0_ref = float(json.load(open(CALIB))["snr0_ref_db"])
    print(f"loaded snr0_ref = {snr0_ref:.2f} dB  (from {CALIB.name})")
else:
    sys.path.insert(0, str(_BC)); import snr_measurement as sm
    calib = sm.calibrate_from_capture(CAP0, CAP0.with_suffix(".sigmf-meta"), sm.SnrConfig())
    snr0_ref = float(np.median([r["snr0_db"] for r in calib["calibration"]]))
    print(f"calibrated snr0_ref = {snr0_ref:.2f} dB")
print(f"SNR(dB) = {snr0_ref:.1f} - attenuation   (att 0 -> {snr0_ref:.0f}, 30 -> {snr0_ref-30:.0f}, 60 -> {snr0_ref-60:.0f})")

# %% [markdown]
# ## Data-saving table (SNR-mapped)
# %%
df = pd.read_csv(TABLE)
df["snr_db"] = (snr0_ref - df["attenuation_db"]).round(1)
df = df.sort_values(["detector", "snr_db"]).reset_index(drop=True)
has_ts = "timeslice_reduction_x" in df.columns
cols = ["detector", "attenuation_db", "snr_db", "reduction_x"] + (["timeslice_reduction_x"] if has_ts else []) + ["retention"]
print(f"detectors: {sorted(df.detector.unique())}  | time-slice column: {has_ts}")
display(df[cols].round(3))
display(df.pivot_table(index="detector", columns="snr_db", values="reduction_x").round(2))

# %% [markdown]
# ## Figures (SNR axis)
# %%
dets = sorted(df.detector.unique())
cmap = dict(zip(dets, plt.cm.tab10(np.linspace(0, 1, max(len(dets), 3)))))
naive = float(df.naive_TB_hr.iloc[0])

def _snr_fig(fname, ycol, ylab, title, log=True, ts=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    if log:
        ax.axhline(naive, color="k", lw=2, label=f"naive save-all ({naive:.2f} TB/hr)")
    for det in dets:
        d = df[df.detector == det].sort_values("snr_db")
        ax.plot(d.snr_db, d[ycol], "-o", color=cmap[det], ms=4, label=det)
        if ts and has_ts:
            ax.plot(d.snr_db, d.timeslice_TB_hr, "--", color=cmap[det], alpha=.5)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("SNR (dB)  [higher = stronger signal]"); ax.set_ylabel(ylab)
    ax.set_title(title); ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1, .5)); fig.tight_layout()
    fig.savefig(OUT/fname, dpi=110, bbox_inches="tight"); display(fig); plt.close(fig)

_snr_fig("stored_vs_snr_axis.png", "stored_TB_hr", "stored TB/hr (log)",
         "Data stored per hour vs SNR" + (" (solid=freq-mode, dashed=time-slice)" if has_ts else ""), log=True, ts=True)
_snr_fig("reduction_vs_snr.png", "reduction_x", "data-reduction factor (x vs naive)",
         "Data-reduction factor vs SNR (frequency-mode snip)", log=False)

# reduction vs retention (path per detector across SNR)
fig, ax = plt.subplots(figsize=(9, 6))
for det in dets:
    d = df[df.detector == det].sort_values("snr_db")
    ax.plot(d.reduction_x, 100*d.retention, "-o", color=cmap[det], ms=4, label=det)
ax.set_xlabel("data-reduction factor (x vs naive)"); ax.set_ylabel("signal-time retention (%)")
ax.set_title("Reduction vs retention (each path = one detector across SNR)")
ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(OUT/"reduction_vs_retention_snr.png", dpi=110, bbox_inches="tight"); display(fig); plt.close(fig)
print(f"wrote SNR-axis figures -> {OUT}")
