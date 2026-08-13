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
