# latency_comp_eval — per-frame latency + GPU compute load, six detectors, CPU vs GPU

Measures, for **one simulated frame** at **20 / 100 / 250 / 500 MHz**, for each of the six
signal detectors:

1. **Compute load** — FLOPs per frame and peak GPU memory per frame.
2. **Latency** — average per-frame wall-clock latency on **CPU** and on **GPU**, as
   clustered bars (one cluster per detector; a bar per sample rate × device, CPU then GPU),
   with the **real-time budget** drawn as a labeled horizontal dashed line.

It is the latency/compute counterpart to `../baseline_comparisons/` (same folder
conventions, detector colors, and serialize-then-plot workflow) but on the *timing/compute*
axis instead of the SNR/accuracy axis.

## Why a single frame, resampled

The 20 dB attenuation capture (`/home/bqn82/captures/attenuation_dB_20.sigmf-data`) is native
**245.76 MS/s**. `run_latency_eval.py` reads one frame's worth of IQ once and
`scipy.signal.resample_poly`-resamples it to each target rate (decimate for 20/100 MHz,
≈passthrough for 250 MHz, interpolate for 500 MHz), then trims to exactly the samples the
deployed pipeline would consume for one frame at that rate.

## FFT geometry & the real-time budget (`fft_sizing.py`)

`fft_sizing.py` is a faithful Python port of `resolve_fft_runtime_config` in
`applications/usrp_wideband_signal_detection/fft_runtime_config.hpp` — the logic the live
pipeline uses to auto-select an FFT size for a sample rate (power-of-two span snap around a
500 MHz / 20480-bin reference, quantized to 1024-sample packets). A "frame" is one detector
batch = `num_ffts_per_batch` (512) FFT rows.

| rate | auto FFT size | samples/frame (512 × fft) | real-time budget |
|---|---|---|---|
| 20 MHz  | 1024  | 524,288    | 26.21 ms |
| 100 MHz | 5120  | 2,621,440  | 26.21 ms |
| 250 MHz | 10240 | 5,242,880  | 20.97 ms |
| 500 MHz | 20480 | 10,485,760 | 20.97 ms |

`real-time budget = samples_per_frame / sample_rate` — the wall-clock a detector has to
finish one frame before the next frame's samples have arrived.

**The budget is set by FFT bin size, not sample rate.** Because a frame is a fixed
`num_ffts_per_batch = 512` FFT rows and the FFT *width* auto-scales with the rate to hold a
target frequency resolution, the budget reduces to

    budget = samples_per_frame / sample_rate = num_ffts_per_batch / bin_size

which is **independent of sample rate** — each FFT row spans `1/bin_size` seconds, so a frame
spans `512 / bin_size` seconds regardless of how fast you sample. (That is also why the
deployed budget is ~constant ~21–26 ms across 20–500 MHz: the auto-FFT holds ~20–24 kHz/bin, so
the frame's time duration barely moves; the 26.2→21.0 ms step is just the power-of-two FFT snap
at 20/100 MHz.)

The budget factors into two knobs — `budget = num_ffts_per_batch × (fft_size / sample_rate) =
512 × (FFT window time)`. The knob that varies here is the **FFT window / integration time**
(`dwell = fft_size/sample_rate = 1/bin_size`): a **longer FFT window buys more budget**, and it
is the one unit that is both monotonic-with-budget *and* sample-rate-independent (so it draws as
a single horizontal line). Bin size is its reciprocal (coarser kHz = shorter window = *less*
budget — the direction runs backwards, which is why the lines are labeled by window time):

| FFT window (dwell) | ≡ bin size | real-time budget = 512 × window |
|---|---|---|
| 50 µs | 20 kHz  | 25.60 ms |
| 20 µs | 50 kHz  | 10.24 ms |
| 10 µs | 100 kHz | 5.12 ms  |
| 5 µs  | 200 kHz | 2.56 ms  |

(FFT *size in points* also raises the budget but is rate-dependent — a 4096-pt FFT is 8 µs at
500 MHz vs 205 µs at 20 MHz — so it can't be a single horizontal line.) Configure the set via
`fig_latency_bars(..., budget_bin_sizes_hz=(...))`.

## The six detectors (`latency_detectors.py`)

Every detector is a **device-switchable torch reference implementation** so CPU and GPU are
timed on identical footing. The deployed `coherent_power` / `cuda_dino` are C++/CUDA
operators with **no CPU path**, so they are reimplemented here from their *operations* (not a
byte-for-byte CUDA port); `yolo` / `dino_finetuned` reuse the exact deployed-sweep model
classes (`yolo_infer` / `finetuned_infer`).

| detector | geometry | timed compute |
|---|---|---|
| `3dB_power` | full frame (512 × nfft) | FFT + single-scalar percentile threshold |
| `blob_detection` | full frame | FFT + gaussian/sobel conv + percentile edges + morphology |
| `coherent_power` | full frame | FFT + per-freq equalization + box-mean CFAR support + per-freq floor(+2 dB)/strong-rescue(+8 dB) power views OR-combined + majority filter + open/close |
| `cuda_dino` | 256×512 chunks of the full frame | frozen DINOv3 ViT-B/16 forward (zero-shot) + patch-score threshold |
| `yolo` | native nfft=1024, 256-row tiles | fine-tuned YOLO26-m per tile |
| `dino_finetuned` | native nfft=1024, 256-row tiles | fine-tuned DINOv3 segmenter per tile |

Power detectors scale with `nfft`; the ML detectors tile at their native geometry so their
cost scales with **tile count → sample rate**. Both scalings match what the deployed system
sees at each rate.

## Run it

```bash
cd applications/usrp_wideband_signal_detection/infocom_evals/latency_comp_eval

# full run (all six detectors, CPU + GPU, all four rates) -> saved_results/latency_run.{npz,json}
python3 run_latency_eval.py --config latency_config.yaml

# faster subsets
python3 run_latency_eval.py --devices cuda                       # GPU only (skips slow ML-on-CPU cells)
python3 run_latency_eval.py --detectors coherent_power 3dB_power --rates 20e6 500e6
python3 run_latency_eval.py --max-reps 50 --time-budget-s 4      # shorter histograms

# render PNGs from a saved run
python3 plot_latency_results.py --results saved_results/latency_run
```

Then open `latency_eval_review.ipynb` and point `RESULTS` at the run — it renders the
compute-load bars, the per-rate CPU-vs-GPU latency histograms (with the real-time budget
line), the across-rates histograms, and the latency-vs-rate summary.

## Timing methodology

- **Detector-only.** Each `prepare()` computes the shared FFT front-end (IQ → dB spectrogram)
  ONCE, outside the timed region; the timed `run()` starts from that spectrogram. So the
  latency is the *detector operator's* own compute (spectrogram → mask), NOT the FFT the
  deployed pipeline runs once upstream for every detector, and NOT host↔device transfer (IQ is
  already on-device). This is what isolates the detector in the latency plot. (The compute-load
  FLOPs still add the analytic FFT term — that plot is *total* per-frame work, a different
  scope; peak memory includes the spectrogram working set.) GPU cells `torch.cuda.synchronize()`
  around each rep.
- Per cell: `warmup_reps` warmups, then adaptive repetition until `min_reps` and
  `time_budget_s`, capped at `max_reps` / `hard_cap_s` (the cap bounds very slow ML-on-CPU
  cells at high rates — those cells get fewer samples; check `n_reps` in the table).

## FLOP accounting caveat

FLOPs = `torch.utils.flop_counter.FlopCounterMode` (counts aten **conv + matmul**, the 2×MAC
convention) **plus an analytic FFT term** (`5·N·log2(N)` per length-N FFT row), because the
flop counter does not model `aten::_fft_c2c`. Pooling and elementwise ops are **not** counted.
So for the power detectors — whose compute is FFT + pooling/elementwise — the reported GFLOPs
is essentially the FFT term (why `coherent_power` and `3dB_power` report similar FLOPs);
`blob_detection` adds its conv passes. For the ML detectors the conv/matmul that dominates is
fully counted. Peak GPU memory captures the parts FLOPs miss.

## Files

- `fft_sizing.py` — auto FFT size + real-time budget (port of `fft_runtime_config.hpp`).
- `latency_detectors.py` — the six device-switchable torch detectors + FLOP components.
- `latency_config.yaml` — rates, capture, model checkpoints, timing knobs, detector params.
- `run_latency_eval.py` — CLI: resample one frame per rate, time CPU/GPU, FLOPs + peak mem, serialize.
- `latency_results.py` — reloadable `LatencyResults` (`.npz` + `.json`), mirrors `SnrResults`.
- `plot_latency_results.py` — figures (importable + CLI):
  - `fig_max_rate` (**headline**) — detector-only latency vs sample rate (all detectors on one
    device), with the ~constant ~21 ms frame budget as a single horizontal line; each
    detector's crossing = its **max feasible sample rate** (`max_realtime_rate_mhz`, from a
    log-log latency-vs-rate fit — interpolated in-range, power-law *extrapolated* beyond
    500 MS/s, so treat e.g. 3dB_power ~2.8 GS/s as a first-order estimate).
  - `fig_latency_bars` — clustered average latency (detector × rate × CPU/GPU) with the
    FFT-window / bin-size real-time budget lines.
  - `fig_latency_vs_rate` (faceted CPU vs GPU), `fig_compute_load` (FLOPs + peak-mem, log).
  - Per-detector CPU/GPU latency histograms also available (`fig_latency_hist*`).
- `latency_eval_review.ipynb` — notebook review, mirrors `baseline_eval_review.ipynb`.
- `saved_results/` — serialized runs + `latency_plots/` PNGs.
