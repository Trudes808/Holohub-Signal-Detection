# %% [markdown]
# # Live data: M1 vs M2 fine-tuned DINOv3 masks (no ground truth)
#
# Unlabeled live captures in `~/captures/live_data/sigmf_out` (test_1/2/3). For 9
# randomly chosen frames (3 per file) we show, side by side: the **spectrogram**, the
# **M1** mask overlay, and the **M2** mask overlay. Frame geometry is **512 × 10240**
# (nfft 10240, ~21.3 ms), matching the other comparison notebooks; the fine-tuned
# models run at their native nfft=1024 and their masks are max-pooled onto this grid.

# %%
from pathlib import Path
import sys, json, warnings
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
warnings.filterwarnings("ignore")

FT_ROOT      = Path.home() / "Holohub-Signal-Detection/dino_fine_tuning"
DINO_REPO    = Path.home() / "dinov3"
EVAL_DIR     = Path.home() / ("Holohub-Signal-Detection/applications/usrp_wideband_signal_detection"
                              "/infocom_evals/signal_detection_experiments")
LIVE_DIR     = Path.home() / "captures/live_data/sigmf_out"
FIG_DIR      = FT_ROOT / "reports/figs_live_data"; FIG_DIR.mkdir(parents=True, exist_ok=True)

FRAME_ROWS, NFFT = 512, 10240
FRAME_SAMPLES = FRAME_ROWS * NFFT
FRAMES_PER_FILE = 3
SEED = 7

for p in (DINO_REPO, FT_ROOT / "src", EVAL_DIR):
    sys.path.insert(0, str(p))
import eval_viz as v
import finetuned_infer as fi
import yaml

# %%
train_cfg = yaml.safe_load(open(FT_ROOT / "configs/train.yaml"))
ds_meta   = json.loads((FT_ROOT / "data/dataset/dataset_meta.json").read_text())
M1 = fi.FinetunedDetector(str(FT_ROOT / "checkpoints/M1_ft/best.pt"), train_cfg, ds_meta,
                          threshold=fi.load_threshold(FT_ROOT / "eval_out/M1_ft/eval_meta.json"))
M2 = fi.FinetunedDetector(str(FT_ROOT / "checkpoints/M2_ft/best.pt"), train_cfg, ds_meta,
                          threshold=fi.load_threshold(FT_ROOT / "eval_out/M2_ft/eval_meta.json"))
SR = float(json.loads((LIVE_DIR / "test_1.sigmf-meta").read_text())["global"]["core:sample_rate"])
EXTENT = [-SR / 2e6, SR / 2e6, FRAME_SAMPLES / SR * 1e3, 0.0]  # freq MHz (x), time ms (y, top=0)
print(f"M1 thr={M1.threshold:.2f}  M2 thr={M2.threshold:.2f}  | frame = {FRAME_ROWS}x{NFFT} "
      f"({FRAME_SAMPLES/SR*1e3:.1f} ms), span ±{SR/2e6:.2f} MHz")

files = sorted(LIVE_DIR.glob("*.sigmf-data"))
rng = np.random.default_rng(SEED)
picks = []
for df in files:
    n_frames = df.stat().st_size // 8 // FRAME_SAMPLES
    for fidx in sorted(rng.choice(n_frames, FRAMES_PER_FILE, replace=False)):
        picks.append((df, int(fidx)))
print("selected frames:", [(p[0].stem, p[1]) for p in picks])


# %%
def spectrogram_and_masks(data_path, fidx):
    mm = np.memmap(data_path, dtype=np.complex64, mode="r")
    s = fidx * FRAME_SAMPLES
    iq = np.asarray(mm[s:s + FRAME_SAMPLES], dtype=np.complex64)
    db = v.spectrogram_db_from_iq(iq, FRAME_ROWS, NFFT)           # 512 x 10240 dB
    m1 = fi.to_display_grid(M1.mask_for_iq(iq), FRAME_ROWS, NFFT)
    m2 = fi.to_display_grid(M2.mask_for_iq(iq), FRAME_ROWS, NFFT)
    return db, m1, m2


def draw(ax, db, vmin, vmax, mask=None, title=""):
    ax.imshow(db, aspect="auto", extent=EXTENT, origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
    if mask is not None:
        rgba = np.zeros((*mask.shape, 4), np.float32)
        rgba[..., 0] = 1.0
        rgba[..., 3] = (mask > 0).astype(np.float32) * 0.45
        ax.imshow(rgba, aspect="auto", extent=EXTENT, origin="upper")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("frequency (MHz, baseband)", fontsize=8)
    ax.set_ylabel("time (ms)", fontsize=8)


# %% [markdown]
# ## Side-by-side: spectrogram · M1 mask · M2 mask (9 frames)

# %%
for data_path, fidx in picks:
    db, m1, m2 = spectrogram_and_masks(data_path, fidx)
    finite = db[np.isfinite(db)]
    vmin, vmax = np.percentile(finite, 5), np.percentile(finite, 99.5)
    fig, axes = plt.subplots(1, 3, figsize=(19, 4.6), constrained_layout=True)
    draw(axes[0], db, vmin, vmax, None, f"{data_path.stem} — frame {fidx}\nspectrogram")
    draw(axes[1], db, vmin, vmax, m1, f"M1 (≤30 dB) mask  (on={100*(m1>0).mean():.2f}%)")
    draw(axes[2], db, vmin, vmax, m2, f"M2 (all dB) mask  (on={100*(m2>0).mean():.2f}%)")
    fig.savefig(FIG_DIR / f"live_{data_path.stem}_f{fidx}.png", dpi=95, bbox_inches="tight")
    display(fig); plt.close(fig)


# %% [markdown]
# ## Notes
# - **No ground truth** — these are unlabeled live captures, so this is a qualitative
#   look at what each model flags as signal. Red overlay = predicted signal.
# - Geometry (512 × 10240, ~21.3 ms/frame, ±122.88 MHz) matches the comparison
#   notebooks; M1/M2 run at nfft=1024 and are max-pooled onto the display grid.
# - Frames are chosen at random (seed=7, 3 per file). Change `SEED` / `FRAMES_PER_FILE`
#   to resample. Panels are also saved to `reports/figs_live_data/`.
