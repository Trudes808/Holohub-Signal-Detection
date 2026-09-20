# adaptive_span_db x adaptive_floor_frac sweep — ground truth (2026-09-19)

Driver: `dino_fine_tuning/ab_span_floor_sweep.py`. Held-out SigMF (108 frames, 306 annotated signals),
threshold 0.6, everything else at the deployed baseline. Predicted [512,1024] mask scored against a GT
mask built from the annotations -> pixel P/R/F1/IoU (captures the recall-vs-false-positive tradeoff),
plus region-detection recall (overall + sparse/low-SNR).
Run: `PYTHONPATH=~/Documents/dinov3 <.venv-ml python> ab_span_floor_sweep.py`.

## Result: the surface is FLAT — the baseline is already optimal.
- Across the whole grid (span 24-48 x floor 0.08-0.20): F1 0.959-0.964, IoU 0.921-0.931, region
  detection 92-94%, low-SNR 90-92%. Total spread ~0.005 F1 / ~0.010 IoU.
- **Baseline (34, 0.12): F1 0.964, IoU 0.930, region 94%, low-SNR 92%** — top of the range.
- Best-on-grid (40, 0.16): F1 0.964, IoU 0.931 — a +0.001 IoU delta = noise; region/low-SNR identical.
- The predicted tradeoff is visible but small: smaller span = higher precision, lower recall
  (span 24: P 0.967 R 0.956; span 48: P 0.955 R 0.972). floor_frac barely moves anything.

## Recommendation: KEEP (34, 0.12).
No meaningful gain is available from tuning these on the held-out data, and the value sits in a broad
flat optimum (robust — good). Chasing the +0.001 to (40,0.16) would be noise/overfit to this set.
NOTE: synthetic held-out has no X410 band-edge artifacts or live occupancy, so this measures the model's
core sensitivity to the params, not live FP behavior. Heatmap: span_floor_sweep.png.
