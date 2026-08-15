# GB10 code-level GPU optimization plan (dual-channel gap closure)

**Status: GOAL MET 2026-08-15.** Per-frame detection (`emit_stride: 1`) at full dual
2× 491.52 Msps, live over the air: two 60 s runs with zero NIC drops, zero shed, full
0.48 Mpps per channel, detector pipeline ~7 ms/frame, masks bit-exact throughout. It took
Tier A (de-serialization) + Tier B round 1 (fused input+power, tiled u8 transpose, gated
diagnostic counts). The dual config now ships with `emit_stride: 1`. Rounds 3a+ (fftshift
fold, morphology tiling at live geometry) remain available as optional headroom for latency
work or a future third channel, ranked in the results doc.

## Goal, bar, and scope (agreed)

- **Goal**: per-frame detection (`emit_stride: 1`) at full dual 2× 491.52 Msps rate — i.e.
  recover roughly **2× detector headroom** over the current operating point (which runs
  `emit_stride: 2` at 79.1% ingest coverage).
- **Validation bar**: every change must reproduce **byte-identical detector masks** on a frozen
  capture via the offline replay harness before it ships. This deliberately excludes FP16
  intermediates and reduction reordering for now; if Tier A+B stall short of the 2× target,
  relaxing to offline-eval equivalence is a *separate decision to revisit*, not a silent fallback.
- **Scope**: changes land **in-place** in the shared operators (they are wins on the x86 bench
  too). Known observable side effect: per-frame debug counters in logs become one frame stale.
- **Order**: Tier A (serialization/allocation removal) → Tier B (kernel fusion) → Tier C
  (CUDA graphs, cross-channel batching) — re-measure after each tier and stop when the goal is met.
- **Both modes benefit by construction**: single- and dual-channel run the same three operators
  (`operators/fft/fft.cu`, `operators/coherent_power_signal_detector/coherent_power_signal_detector.cu`,
  `applications/usrp_wideband_signal_detection/spectrogram_visualization.cu`); nothing in the hot
  kernels is channel-count-specific. Single-channel (97.8% coverage, near-clean) is the
  validation vehicle; only Tier C cross-channel batching is dual-only.

## Code audit findings (2026-08-13, all file:line refs at commit c711d42e)

### Tier A — pure overhead, zero numerical change

1. **Per-stage stream syncs in production.** `time_step_ms` does a `cudaStreamSynchronize`
   after *every* detector stage when `timing_summary_enable` is on
   (`coherent_power_signal_detector.cu:1849`) — and **both live configs set it to true**.
   That is ~8–10 full stream drains per frame per channel. The 14.5 ms/frame dual figure was
   measured *with* this serialization in place.
2. **Five blocking 4-byte counter readbacks per emitted frame** plus one sync
   (`coherent_power_signal_detector.cu:3139, 3200, 3205, 3215, 3225, 3235`). They feed only
   `fast_summary` debug logging and informational metadata
   (`coherent_emitted_mask_nonzero_pixels`) — grep confirms **no programmatic consumers**
   downstream. Safe to defer one frame via pooled pinned staging.
3. **Per-frame allocations**: the FFT op `make_tensor`s a fresh device output every frame
   (`fft.cu:193`, a synchronous `cudaMalloc`); the detector emit path calls
   `allocate_owned_u8_buffer`/`allocate_owned_u32_buffer` per frame
   (`coherent_power_signal_detector.cu:3125-3127, 3156`); the preview creates/destroys a
   `cudaEvent` per frame (`spectrogram_visualization.cu:1617-1621`). Pool all of these.
4. **Mask handoff sync is load-bearing today**: `DetectorMaskMessage`
   (`spectrogram_visualization.hpp:343`) carries no stream/event, so consumers (mask overlay
   path, `signal_snipper`) assume `device_pixels` is complete on receipt. Removing the last
   per-frame sync (`:3200`) requires adding a `cudaEvent` to the message and
   `cudaStreamWaitEvent` in consumers.
5. **D2H→CPU→H2D render round trips**: the preview reduces on-GPU (good) but then copies to
   pinned, syncs, `memcpy`s into a `std::vector`, and the compositor re-uploads the composed
   RGB canvas to HoloViz (`spectrogram_visualization.cu:1630-1642, 1995`). On unified memory
   these copies are pure DRAM overhead. (Lower priority: the reduced buffers are small; the
   canvas upload is ~MBs/frame.)

### Tier B — fusion, numerically identical or near-identical

6. **fftshift is a separate full-surface complex pass** (`fft.cu:207`, ~168 MB/frame of traffic
   at 512×20480 complex) — foldable into consumers as an index remap.
7. **~30–50 tiny full-surface kernels per detector frame** (power_db → 4–8 frontend passes →
   2 box-mean → score → dynamic-floor family → ~8 erode/dilate → persistence → majority smooth
   × iterations → several count_nonzero). Fuse the obvious chains: power+frontend+score;
   separable morphology pairs in shared-memory tiles; fold count_nonzero into producing kernels
   (atomicAdd at write time). Each avoided pass saves ~84 MB/frame (float) or ~21 MB (u8).
8. **The visualizer recomputes power+log10 from the complex tensor**
   (`reduce_complex_to_grayscale_kernel`, `spectrogram_visualization.cu:1399`) — a full complex
   read the detector's `power_db` pass already did. Share the power_db surface.

### Tier C — bigger levers

9. **CUDA graphs** over the de-synced detector chain (requires Tier A first) — kills launch
   overhead, which is pricier on the Arm cores.
10. **FP16 intermediates** — halves detector bandwidth; **blocked by the bit-exact bar**, revisit
    only if needed.
11. **Cross-channel batching** — one launch over both channels' surfaces; architectural (channels
    currently arrive as separate tagged frames), dual-only benefit. Last resort.

## Bandwidth arithmetic (why this should reach 2×)

Dual full rate demands ~48k FFTs/s; the pipeline sustains ~28–38k, i.e. we need ~1.3–1.7×.
The measured 3.7→14.5 ms/frame kernel inflation under dual load is contention + serialization,
exactly what Tiers A/B attack:

- Tier A removes ~10–15 CPU↔GPU stalls per frame per channel that both idle the GPU between
  stages and defeat cross-channel stream overlap.
- Tier B eliminates whole full-surface passes; at ~84 MB/frame/pass and 48 frames/s dual,
  each fused pass is ~4 GB/s of DRAM traffic back — against a ~225 GB/s shared budget the
  pipeline already saturates in bursts.

## Validation workflow (every change)

1. `sudo ./bash_scripts/rebuild_demo_container_app.sh`
2. Offline replay on the frozen capture, dump per-frame masks, `sha256sum` against the golden
   manifest (`infocom_evals/signal_detection_experiments/gpu_opt_golden_masks/`). Must match
   byte-for-byte.
3. Live single-channel sanity (`sudo ./bash_scripts/run_live_demo.sh single`).
4. Live dual 60 s loss budget (wire packets vs NIC out-of-buffers vs converter batches) —
   the method from `RUNNING_ON_DGX_SPARK.md` §6. Track detector ms/frame from the (now
   event-based) timing summary.
5. Record results under `infocom_evals/signal_detection_experiments/`.

## Landed changes

(updated as work lands)

- [x] A.1 detector de-serialization (event-based stage timing, pinned-staged counters,
      emit-path buffer pooling) — commit ab8e7f78, bit-exact PASS both variants
- [x] A.2 FFT output tensor async allocation — same commit, same gate
- [ ] A.3 event-carrying `DetectorMaskMessage` (drop the last per-frame sync) + preview event pool
- [ ] A.4 (new) one-shot startup-backlog drain in the converter→FFT queue (~300 ms standing
      latency at matched rates; see results doc)
- [x] re-measure + emit_stride 1 attempt — see
      `infocom_evals/signal_detection_experiments/gpu_opt_tier_a_results.md`.
      **Headline: dual-channel NIC loss 20.9% → 0% at stride 2 (100% ingest), no config change.**
      stride 1 still sheds ~15% (detector ~14.7 ms/frame/ch × 48 frames/s) → Tier B next.
- [x] B round 1 (commit 10a0807d, bit-exact PASS): fused input+power kernel (power dB straight
      from the FFT tensor; complex analysis scratch now snapshot-only), shared-memory tiled u8
      transpose (was the profile's #1 kernel at 16.8% — 182 → 37 µs, 5×), new
      `emit_mask_diagnostic_counts` param (audit-only counters; dual live config disables — 4
      full-mask count passes per frame gone). Live stride-1 with the fused kernel alone: ingest
      85 → 87.5%; the tiled transpose + gated counts landed after the X410 control link dropped,
      so their live effect is **pending a radio power-cycle**.
- [x] B round 2 fused rectangle morphology — **tried, measured 50% slower, reverted**
      (d2df3ca4 / 9a4a3e7a). Lesson: the ~10 MB u8 masks are L2-resident on GB10 (24 MB L2), so
      collapsing separable pairs saves no real DRAM traffic and pays O(w×h) vs O(w+h) scans.
      **Fusion effort must target the non-L2-resident float (42 MB) and complex (84 MB)
      surfaces only.**
- [ ] B round 3 candidates — RE-RANKED at live dual geometry (see results Addendum 2): fftshift fold (10%), power_db scatter-write tiling (11%), morphology tiling AT LIVE GEOMETRY (15%, masks no longer L2-resident at 10.5 MB), score-family fusion (8%). Previous dynamic-variant list:
      fftshift fold into consumers (8.4%, complex 84 MB round-trip; needs an fft-op flag +
      remaps in detector/preview/cuda_dino), score-family fusion (score 5.1% + per_freq_fill
      1.5% + strong_rescue 2.6% all re-read corrected_db), dynamic_floor_update +
      row_sampled_mean merge (4.4% + 3.2%, both full-surface row reductions), box-mean pair
      (7.4% combined).

## Saturation math (why per-kernel µs matter 3×)

Under dual stride-1 load, kernels run ~3× their uncontended time (offline detector ~4.4 ms/frame
equivalent → 14.7 ms measured live). So ~1 µs of uncontended kernel time saved returns ~3 µs of
live GPU headroom: round 1's ~390 µs/frame/ch ≈ 60 ms/s contended ≈ ~6 points of ingest.
The stride-1 gap is ~125 ms/s — round 3's targets (~300–400 µs/frame uncontended) are plausibly
enough, but each must be re-measured live before trusting the estimate.

## Post-measurement correction to the bottleneck story

The baseline "GPU memory-system contention" ceiling was **partly an artifact of host-side
serialization**: with the syncs/allocs removed, the same GPU sustains dual full rate with the
stride-2 detector cadence and empty queues. The bandwidth wall is still real — stride 1's
pipeline stage inflates 7.4 → 9.8 ms under the doubled kernel load — but it sits ~15% over
budget, not ~70%, and Tier B fusion attacks exactly that.
