# Tier A results — detector de-serialization + async FFT allocation

Commit under test: `ab8e7f78` (gpu-opt Tier A) on `live_demo`. Method: controlled 60 s OTA
dual-channel runs (2400 + 1000 MHz, 2× 491.52 Msps), identical to the c711d42e baseline runs in
`RUNNING_ON_DGX_SPARK.md` §6; NIC loss read from the cumulative `rx_out_of_buffer` counter on
`enp1s0f0np0`. Bit-exactness: offline masks byte-identical to
`gpu_opt_golden_masks/` for both coherent variants before any live run.

## Dual channel, emit_stride 2 (the shipped operating point) — 2026-08-13

| Metric | Baseline (c711d42e) | Tier A (ab8e7f78) |
| --- | --- | --- |
| NIC `rx_out_of_buffer` (60 s) | 12,031,807 (**20.9% lost**) | **0 (100% into the pipeline)** |
| Per-channel ingest rate | degraded windows | full 0.48 Mpps every window, `queued == emitted` |
| Converter partial drops / panic resets | 0 / 0 | 0 / 0 |
| Converter out-queue depth | pegged at max (4) | ~1 |
| FFT→detector queue wait (mean) | (not measured; queue full) | **1.4–1.6 ms** |
| Detector GPU per emitted frame | ~14.5 ms (wall, sync-inflated) | **input 2.0 + power_db 1.6 + pipeline 7.4–8.0 ms** (event-measured) |
| CHDR→FFT latency (mean) | — | ~322 ms (startup-fill queue backlog; see note) |

The 20.9% dual-channel loss is gone with no config change: it was serialization
(per-stage timing syncs, blocking counter readbacks, per-frame cudaMalloc/cudaFree)
masquerading as a GPU-bandwidth ceiling.

## Dual channel, emit_stride 1 (per-frame detection attempt) — 2026-08-13

Does **not** hold yet: channels shed to ~0.40–0.41 Mpps (~85% of wire rate, ~39–42k pkts/s/queue
dropped at assembly), out-queue pegged at 3–4, `empty_polls=0`, CHDR→FFT ~400 ms.
Detector GPU per frame: input 2.4–2.6 + power_db 2.3–2.4 + pipeline ~9.8 ms ≈ **14.7 ms/frame/ch**
→ at 2×24 frames/s ≈ 0.7 s of GPU per second on detection alone, ~15% over budget (and note the
contention inflation vs stride 2: pipeline 7.4 → 9.8 ms). Closing this is Tier B's job
(kernel fusion; the pipeline stage is ~25–40 small full-surface passes).

## Follow-up spotted

At stride 2 the pipeline consumes exactly at the production rate, so the ~15-batch backlog that
accumulates in the converter→FFT queue during Vulkan/app startup **never drains** — that is the
~322 ms standing CHDR→FFT latency. A one-shot stale-batch drain at stream start (or a brief
faster-than-realtime catch-up) would cut visualization latency by ~300 ms without touching
throughput. Candidate Tier A.4.

## Addendum — Tier B rounds 1–2 (2026-08-13, later the same day)

- **Round 1** (`10a0807d`, bit-exact PASS both variants): fused input+power kernel, tiled u8
  transpose (nsys: 182 → 37 µs, was the #1 GPU kernel at 16.8%), `emit_mask_diagnostic_counts`
  gate (dual live config now skips 4 audit-only full-mask count passes per frame).
  Live stride-1 with only the fused kernel: ingest 85% → 87.5% (0.405 → 0.42 Mpps/ch) and the
  detector *stage* times redistributed (input 2.5 → 0.05 ms; pipeline absorbed the rest) —
  clean demonstration that the regime is bandwidth-saturated: wall time follows total DRAM
  traffic, not kernel count.
- **Round 2** fused rectangle morphology: **reverted after measurement** (+50% kernel time,
  242 vs 161 ms aggregate). The u8 masks are L2-resident on GB10; only float/complex surfaces
  pay DRAM. This redirects all remaining fusion work.
- The live effect of the tiled transpose + count gating is **unmeasured**: the X410's control
  link (sfp1) dropped mid-session (host port shows NO-CARRIER; UHD "No devices found") and
  needs a physical power-cycle. Offline validation is unaffected and PASS.

Measurement scripts: `measure_dual_60s.sh` methodology (app detached → wait for DPDK arm →
settle 8 s → 60 s OTA stream → settle → graceful stop; loss from cumulative
`rx_out_of_buffer` deltas + CHDR per-window summaries).

## Addendum 2 — no-hardware benchmark stood up (2026-08-13, radio + QSFP unavailable)

With the X410 gone and no loopback cable, the optimization loop now runs entirely offline:

1. **Bit-exact gate** (unchanged): offline replay vs `gpu_opt_golden_masks/`.
2. **Kernel ranking**: nsys per-kernel sums over the benchmark config (primary signal).
3. **End-to-end sanity**: `bash_scripts/bench_gpu_contention.sh` — N concurrent offline
   replays in the binary's new loop-preload mode (zero per-frame host I/O) at live dual frame
   geometry (512×20480, per-frame detection, dual detector profile, static floor mode).
   Config: `gpu_bench/config_bench491_gpu_contention.yaml`; input: hardlink of the frozen
   capture relabeled to 491.52 Msps (same bytes, live FFT geometry).

**Baseline (commit ad83ddcc)**: 1 instance 36.3 f/s; 2 concurrent 27.8+27.7 = **55.5 aggregate**
(bar: 46.9 f/s per live channel; ~94 aggregate ≈ dual stride-1). Run-to-run noise ~±1 f/s.
Caveats: two processes time-slice CUDA contexts (live overlaps streams in one process), and the
offline graph carries ~20 ms/frame of serial host overhead the live app does not (kernel sums:
~7.1 ms/frame GPU vs 27.6 ms wall) — so absolute f/s undershoots live; use for *ranking*.

**Dual-profile kernel ranking at live geometry** (nsys, per frame, uncontended — this supersedes
the earlier dynamic-variant ranking for stride-1 work):

| Kernel | ms/frame | share | note |
| --- | --- | --- | --- |
| cuFFT (3 passes: 64·64·5 for 20480 pts) | 2.24 | 32% | untouchable |
| power_db_from_input (fused, scatter-write) | 0.76 | 11% | write side could tile like transpose_u8 |
| fftshift_rows | 0.70 | 10% | **pure overhead — fold into consumers** |
| emit morphology (8 separable passes) | 1.09 | 15% | L2-resident at small geometry, NOT at live 10.5 MB masks — revisit tiling *at this geometry* |
| score | 0.56 | 8% | fusable with strong_rescue reads |
| box_mean cols+rows | 0.78 | 11% | |
| frontend_correction + row_mean | 0.49 | 7% | |
| majority_smooth / transpose / persistence / count | 0.48 | 7% | |

Next round (3a): fftshift fold — detector side is one index remap in the fused kernel; preview
and cuda_dino need an `apply_fftshift` fft-op param (default true) so untouched paths keep the
shift.

## Addendum 3 — GOAL MET: per-frame detection at full dual rate (2026-08-15, radio restored)

Radio re-connected (topology re-verified: control 192.168.21.2 on enp1s0f1np1, data
192.168.10.2 on enp1s0f0np0). Live 60 s measurements on commit c6f29a42's binary — first live
test of the Tier B round-1 kernels (tiled transpose + gated diagnostic counts):

| Run | NIC `rx_out_of_buffer` delta | shed warnings | rate/window | detector (event-based) |
| --- | --- | --- | --- | --- |
| dual `emit_stride: 2` | **0** | 0 | 0.481 Mpps/ch, queued==emitted | input 0.08 + power 1.5 + pipeline **5.0 ms** |
| dual `emit_stride: 1` | **0** | 0 | 0.471–0.481 Mpps/ch, queued==emitted | input 0.1 + power 1.4 + pipeline **6.9–7.1 ms** |
| dual `emit_stride: 1` (confirm) | **0** | 0 | 0.481 Mpps/ch, out_q 0 | — |

Progression of the stride-1 gap: 85% (Tier A) → 87.5% (+ fused input+power) → **100%**
(+ tiled transpose + gated counts). Pipeline stage at stride 2 dropped 7.4–8.0 → 5.0 ms/frame
with the round-1 kernels. FFT→detector queue wait at stride 1 is ~22 ms (stable, not growing);
chdr→fft remains ~322 ms startup-fill backlog (the Tier A.4 latency candidate).

The committed dual config now ships `emit_stride: 1`: **every frame of both 500-class Msps
channels is examined by the detector, with zero loss anywhere in the chain.**

## Addendum 4 — single-channel re-measure + snipper smoke test (2026-08-15)

- **Single channel is now 100% too**: 60 s at full 491.52 Msps, cumulative `rx_out_of_buffer`
  delta **0** (baseline binary had 622,410 / 2.2%). chdr→fft latency improved ~200 → **160 ms**
  (max 162 — rock stable). Detector: input 0.002 + power 0.4 + pipeline 1.97 ms/frame.
- **Signal snipper smoke test**: first run showed 24 data-flushing panic resets/60 s — the
  snipper config had missed two fixes validated on the perf configs
  (`degraded_shutdown_on_rx_queue_warning_threshold: 3` → 0, RX `num_bufs` 131072 → 262144).
  With those applied: panic resets **0**, NIC drops **0**, `queued == emitted` in every window,
  127 SigMF snippet files written. The profile oscillates (falls ~1 s behind under snip+write
  bursts, catches up at ~0.67 Mpps through the RX pools) — loss-free, but running near its
  ceiling; benign "Fell behind" log lines during the catch-ups.
