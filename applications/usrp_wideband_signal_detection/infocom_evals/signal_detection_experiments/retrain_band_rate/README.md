# Band/rate-invariant retrain — saved tests

Validation for the domain-randomized fine-tune (see
`../../../notes/retrain_band_rate_invariant_plan.md`). Each test is run for the **current** model and
the **retrained** model so the improvement is visible. Save outputs here.

## 1 & 3. Rate-sweep + quiet false-positive (headline) — `eval_band_rate.py`
Emulates a capture at each sample rate (same capture-chain emulation as training) and runs each model,
reporting **pixel F1 vs rate** on a labeled band and **false-positive rate vs rate** on a quiet band. A
band/rate-invariant model holds F1 flat on ISM and FP-rate ≈ 0 on the quiet band, at every rate.

```bash
CUR=/home/bqn82/.../M2_ft.ts            # current single-rate model
NEW=/home/bqn82/.../M2_dr.ts            # retrained domain-randomized model
STATS=/home/bqn82/captures/dino_finetune_x411_sweep/stats

# (1) 1150 MHz quiet zone -> false-positive rate vs rate (should collapse to ~0 for NEW):
python3 eval_band_rate.py --models $CUR:current $NEW:retrained \
    --capture /home/bqn82/captures/quiet_1150.sigmf-meta --sweep-stats $STATS \
    --out ./results_quiet_1150 --rates-hz 20.48e6 61.44e6 122.88e6 245.76e6

# (3) 2400 MHz ISM (labeled) -> pixel F1 vs rate (NEW should stay flat, no regression):
python3 eval_band_rate.py --models $CUR:current $NEW:retrained \
    --capture /home/bqn82/captures/attenuation_dB_20.sigmf-meta --sweep-stats $STATS \
    --out ./results_ism_2400 --rates-hz 20.48e6 61.44e6 122.88e6 245.76e6
```
Writes `band_rate_results.json` + `band_rate_{f1,fprate}.png` per out-dir. After the wideband-image
sweep, add `491.52e6 500e6` to `--rates-hz` (needs a capture the radio can produce at those rates).

## 4. 500 MHz HDTV (wideband coverage) — `eval_band_rate.py` on an HDTV capture
Run the labeled/quiet eval on a 500 MHz HDTV capture (qualitative + coverage). If unlabeled, it reports
FP-rate (here interpreted as "fraction of the band flagged"); eyeball that NEW covers the occupied
region solidly without fragmenting. Same command as above with `--capture <hdtv>.sigmf-meta`.

## 2 & 5. ISM detection-rate-vs-SNR + held-out synthetic SNR — reuse the existing harness
No new code — the standard offline harness scores region detection-rate/F1/IoU vs SNR on the labeled
attenuation sweep. Regenerate per model and compare (retrained must not regress on 2400 ISM):
```bash
cd dino_fine_tuning
python src/gen_finetuned_run.py --ft-ckpt checkpoints/M2_dr/best.pt \
       --detector-name finetuned_dino_dr --ft-eval-meta eval_out/M2_dr/eval_meta.json
python <infocom_evals>/eval_detector_masks.py --batch-root notebooks/sweep_detectors \
       --out-dir notebooks/compare_tables_dr --coverage-threshold 0.1
# then build_snr_results.py / plot_snr_results.py as in baseline_comparisons/
```

## Notebooks
- `compare_dr.ipynb` — M2_dr vs M2_ft on the **labeled** captures: threshold sweep, IoU-vs-SNR
  (attenuation), rate-invariance (capture-chain emulated across rates), per-band, sweep PSD.
- `compare_val_bands.ipynb` — M2_dr vs M2_ft vs **coherent_power** on **real OTA IQ** captured by
  `sweep_capture.py --centers-hz ...` at specific bands, at every rate the loaded FPGA image supports.
  Unlabeled → compares occupancy % and pairwise detector agreement (IoU), plus a per-band×rate mask
  gallery. Point `VAL_DIR` at the sweep out-dir; runs on synthetic frames until captures exist.
  Capture the val IQ with:
  ```bash
  # ~10 model frames (2560*1024 samples) of raw antenna IQ per band, every stock-image rate:
  python dino_fine_tuning/data_collection/sweep_capture.py --device-args "addr=192.168.10.2" \
         --out-dir /tmp/usrp_val_bands --centers-hz 100e6 500e6 1200e6 2400e6 \
         --frames-per-burst 2560 --save-iq-every 1 --gains-db 30
  # higher rates: reimage to the 400 MHz FPGA image, then append with the wideband clocks:
  python ... --out-dir /tmp/usrp_val_bands --centers-hz 100e6 500e6 1200e6 2400e6 \
         --frames-per-burst 2560 --save-iq-every 1 --gains-db 30 \
         --master-clocks-hz 491.52e6 500e6 --resume
  ```

## What "success" looks like
- **Quiet 1150**: FP-rate ~0 across all rates for retrained (current fires on the passband).
- **ISM 2400**: F1 / detection-rate flat across rate, no regression vs current at 245.76.
- **HDTV 500**: solid coverage of the wideband signal.
- **Rate sweep**: every metric roughly rate-independent — the core deliverable.
