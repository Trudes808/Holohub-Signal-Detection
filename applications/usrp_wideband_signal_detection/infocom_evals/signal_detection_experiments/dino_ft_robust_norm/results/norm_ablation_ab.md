# Normalization ablation A/B — ground-truth held-out (2026-09-19)

Verifies the "load-bearing" claim for the M3_491 normalization family. Driver:
`dino_fine_tuning/ab_norm_validate.py` (reuses eval_heldout.per_snr_eval on data/dataset_491/heldout_sigmf,
6 composites with per-signal wfgt:power_db truth). Fixed deployment threshold 0.6; toggle one front-end
option off at a time; metric = per-signal region detection rate (any mask pixel overlaps the GT box).
Run with: `PYTHONPATH=/home/genesys-dgx1/Documents/dinov3 <.venv-ml python> ab_norm_validate.py`.

| Front-end | dense all | dense lowSNR | sparse all | sparse lowSNR |
|---|---|---|---|---|
| baseline (flatten+adaptive+robust) | 94% (269/286) | 92% (123/133) | 90% (18/20) | 90% (9/10) |
| flatten_noise_floor OFF | 94% (268/286) | 92% (122/133) | 90% (18/20) | 90% (9/10) |
| adaptive_normalization OFF | 64% (184/286) | 59% (79/133) | 55% (11/20) | 40% (4/10) |
| adaptive_robust_floor OFF | 94% (270/286) | 92% (123/133) | 90% (18/20) | 90% (9/10) |
| ALL norm OFF (flatten+adaptive) | 69% (196/286) | 65% (87/133) | 55% (11/20) | 40% (4/10) |

## Conclusions
- **adaptive_normalization is strongly load-bearing**: OFF craters detection (dense 94->64%, sparse
  low-SNR 90->40%). This is the weak-signal recovery, proven on ground truth. Keep.
- **flatten_noise_floor and adaptive_robust_floor are NEUTRAL on synthetic held-out** (identical on/off):
  their value is LIVE-specific (flatten = X410 band-edge roll-off FP removal; robust = dense real-world
  occupancy floor) which the clean synthetic set doesn't exercise. They don't hurt, and M3 was fine-tuned
  WITH them (frontend.py), so keeping them is correct (removing = off-distribution, zero upside).
- No config change warranted; the deployed front-end is correctly configured and matches training.
