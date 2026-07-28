# investigation — low-SNR footprint diagnosis & the min_mask_bandwidth_hz fix

The one-off diagnostic + fix-development scripts (and their CSV/figure outputs) that explain **why the
coherent-power detector appeared to save so little data at low SNR**, and that drove the operator-level
fix now shipped in `signal_snipper`. These are the historical record — the production path lives one
level up in `snip_eval/`; nothing here is on the live/reproduction chain except the two CSVs that feed
the before/after figures (see below).

## What the investigation found
1. **The "48 MHz streak" is a receiver clock spur, not a signal.** A persistent ~117 Hz-wide CW line
   sits at 2048 MHz absolute (= 2¹¹ MHz), with a spur family at 2048 ± k·20.48 MHz. An attenuation
   sweep is the proof: real transmitted signals drop ~1 dB per dB of attenuation, but this line is flat
   (−0.13 dB/dB) and the ground truth has no signal there. → `streak_forensics.py`,
   `streak_mask_presence.py`.
2. **Bounding-box fusion inflated the footprint.** Energy detectors (coherent_power, 3 dB) faithfully
   flag the spur. 4-connected component labeling then **fuses** the persistent full-height spur column
   with wide transient bursts into a single full-height, ~1.4 %-filled bounding box that the snipper
   stores whole — a ~51× over-count of real data. → `prove_coherent_artifact.py`, `quantify_fixes.py`.
3. **The fix: `signal_snipper.min_mask_bandwidth_hz`** — a per-row run-length **mask pre-filter** that
   zeros lit runs narrower than N columns *before* component labeling (immune to fusion, because it
   acts on pixel runs, not merged boxes). Prototyped here, then implemented in the operator and
   validated: coherent-power → 0 stored at low SNR (matching DINO's honest zero); ground-truth/DINO
   masks provably untouched. → `prototype_mask_filter.py`, `replicate_75k_before.py`.
4. **Mask provenance (why "the same detector" gave different masks).** The detector is fully
   deterministic (byte-identical reruns). Two coherent-power mask *families* exist at different
   operating points — the staged July batch used the legacy global-threshold config; fresh runs use the
   per-frequency calibrated floor. Each mask run records its exact config in `offline_eval_summary.json`;
   never compare footprints across families without stating the config.

## Scripts → outputs
| script | does | writes |
|---|---|---|
| `streak_forensics.py [atten]` | attenuation-sweep proof the 48 MHz line is an RX clock spur | `streak_forensics.csv`, `figs/streak_*.png` |
| `streak_mask_presence.py` | which detectors' masks contain the spur column, per attenuation | `streak_mask_presence.csv` |
| `prove_coherent_artifact.py [atten]` | quantifies the bbox-fusion over-count (fill %, over-count ×) | `figs/prove_coherent_artifact.png` |
| `quantify_fixes.py` | offline replication of the snipper scoring candidate fixes under all gate configs | `fix_quantification.csv` |
| `prototype_mask_filter.py` | prototype + prediction of the implemented `min_mask_bandwidth_hz` filter | `prototype_mask_filter.csv` |
| `replicate_75k_before.py` | offline replication of the 75 kHz/1 ms "before" curve | `real_snip_metrics_75k_before_replicated.csv` |
| `render_masks.py` / `render_spectrogram_overlay.py` / `visualize_bbox.py` | per-frame debug visuals (spectrogram + mask + snipper boxes) | `figs/{masks_coherent_lowsnr,debug_spectrogram_overlay,debug_bbox_overlay}.png` |

**Two outputs still feed the current figures:** `fix_quantification.csv` (GT ceilings) and
`real_snip_metrics_75k_before_replicated.csv` (the 75 kHz "before" curve) are read by
`../plot_maskfilter_figs.py`. Keep them in sync if you re-run.

## Running
```bash
cd <this folder>
~/miniforge3/envs/dinov3/bin/python streak_forensics.py       # etc.
```
Inputs: the **staged detector masks** under `../snip_run/<detector>/<stem>/mask_arrays/*.packed.npz`
(gitignored, regenerable via `../stage_snip_masks.py` / `../create_all_masks.sh`) and the SigMF
captures in `/home/bqn82/captures`. Scripts read those from the parent `snip_eval/` and write their
CSVs/figures here (`figs/`). `prototype_mask_filter.py` and `replicate_75k_before.py` import helpers
from `quantify_fixes.py`, so run them from this folder.
