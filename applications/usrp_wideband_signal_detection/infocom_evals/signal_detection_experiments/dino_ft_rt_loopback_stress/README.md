# DINO-FT real-time vs native vs coherent -- loopback stress test on the Argus capture (2026-10-21)

**Question.** Does the "DINO-FT real time" method (`finetuned_dino_detector.real_time_downsample=true`)
keep up with a 491.52 MSps channel, and what accuracy does it give up versus the full-resolution
DINO-FT path and the coherent-power detector? Measured on the real pipeline over a loopback replay
of an Argus capture at its native 245.76 MSps, then at double speed.

**Bench.** DGX Spark GB10, container `usrp_x410_sig_det_sat3737`, branch `live_demo` @ `1a466103`
plus the uncommitted instrumentation below. Capture: `~/Datasets/ARGUS/main_captures/capture_snr_p24dB`
(245.76 MSps, +24 dB average SNR, 3,594 SigMF ground-truth annotations, 11 signal classes), trimmed
to 1,751,040 CHDR packets = 342 frames @245.76 / 171 frames @491.52 (frame-aligned at both rates).

**Status after the efficiency work (end of 2026-10-21).** The 245.76 MSps deficit of the full pipeline
reported below is fixed: with the stage 1-3 changes in
[Efficiency work](#efficiency-work-the-full-pipeline-with-the-dashboard-2026-10-21-night) the full
pipeline with the dashboard and the snipper sustains 245.76 MSps, and the new short-FFT front-end
(`config_live_v3_dino_ft_m5s_sf2048.yaml`) is also the most accurate DINO-FT RT variant. The
491.52 MSps result is a projection (~94% GPU), because the sender here cannot offer the full rate.

## What was tested

| leg | detector | what it is |
|---|---|---|
| `dinoft_rt_m5s` | `cuda_dino_finetuned`, **M5-S (ViT-S)**, `real_time_downsample: true` | **DINO-FT real time** = the deployed live detector: app-sized wide FFT (10240-pt @245.76, 20480-pt @491.52) bilinearly resized to the model's 1024 columns -> 2 ViT tiles per frame at any rate |
| `dinoft_native_m2dr` | `cuda_dino_finetuned`, **M2_dr (ViT-B)**, `real_time_downsample: false` | **DINO-FT** at full resolution: dedicated 1024-pt FFT, 240 kHz/bin @245.76, 20 tiles/frame @245.76, 40 @491.52 (M2_dr is the checkpoint trained at this native geometry) |
| `coherent` | `coherent_power`, dynamic floor | the production coherent detector |
| `dinoft_native_m5s`, `dinoft_rt_m2dr` | path ablations with the model held fixed | isolates "path" from "model" |
| `dinoft_rt_m5s_nosnip` | M5-S RT without the snipper/compression/sink branch | pipeline-load ablation |

Every leg runs the **full production pipeline** (DPDK ingest -> CHDR -> FFT -> detector -> snipper ->
compression -> SigMF sink + the HoloViz dashboard) at `emit_stride: 1`, with the dense-composite
relaxations the loopback demo configs already use (coherent `max_emit_occupancy: 0`, DINO
`invalid_frame_guard: false`, snipper `max_mask_occupancy: 0.90`). Configs are generated from the
production configs by `make_configs.py` (edits listed in its docstring).

## Method

1. **Offline references** (`run_offline_refs.sh`): the offline eval binary runs each leg config over
   the SigMF file -> per-frame ground-truth masks on the detector's grid + the deterministic offline
   masks. Done at 245.76 and, via a "2x twin" of the capture (same IQ bytes, `core:sample_rate` and
   annotation frequency edges doubled -> same fractions of the span), at the 491.52 geometry.
2. **Loopback legs** (`run_leg.sh`, `run_series.sh`): the capture is packetized to a CHDR pcap
   (`replay_rx_to_buff.py`, 64 B header line, dst MAC of the DPDK port) and replayed with
   `tcpreplay --pps 240000` (245.76 MSps) or `--pps 480000` (491.52, 6 loops). A new
   detector-agnostic **emitted-mask dump** (`mask_dump_sink.hpp`, env `USRP_MASK_DUMP_DIR`) records
   every mask the viz/snipper consume plus per-frame detector GPU time, host cadence and CHDR fill.
3. **Scoring** (`collect_metrics.py`, `score_leg.py`, `summarize.py`): ingest health from the DPDK /
   CHDR / FFT counters and the manifest; accuracy by pairing live masks with the offline ground
   truth frame-by-frame (`region det@t` = annotation box covered >= t; `effective` counts frames the
   live pipeline never emitted as misses).

**How the loopback was actually wired.** The request assumed the Spark was in cable-loopback mode. It
was not: both QSFP ports were cabled to the X410, which was still streaming two 491.52 MSps channels
into the DPDK port from the 10-19 dual demo (nobody consuming; ~1 Mpps, 32 Gbps). The Spark's four
ConnectX-7 netdevs are only **two physical ports x two PCIe functions** (same `phys_switch_id`,
identical PHY counters), so a cable-free loopback exists: frames sent from `enP2p1s0f0np0`
(PF 0002:01:00.0) with the DPDK PF's MAC are switched inside the NIC to `enp1s0f0np0`
(PF 0000:01:00.0). The replay pcap was port-mapped to UDP 49160 -> 1240 and the leg configs match
only that flow, so flow isolation drops the radio's 49153 -> 1234/1235 packets in hardware. At 13:15
someone at the console re-cabled the two ports to each other (kernel: "Cable plugged" on all four
modules; X410 unreachable since). The evening validation runs (`v_*`, `lean_*`) use that cable
(`SENDER_IFACE=enp1s0f1np1`); every earlier run used the in-NIC path.

## Results

### Per-frame detector cost (offline binary, same GPU; model path only, ms/frame)

| | 245.76 MSps | 491.52 MSps |
|---|---|---|
| coherent_power (fft -> detector done) | ~1.5 | -- |
| DINO-FT RT, M5-S (2 tiles) | 16.5 | **13.0** |
| DINO-FT RT, M2_dr ViT-B (2 tiles) | 26.9 | 23.9 |
| DINO-FT native, M5-S (20 / 40 tiles) | 121 | 242 |
| DINO-FT native, M2_dr ViT-B (20 / 40 tiles) | 205 | 414 |

Frame budget: 21.3 ms (512 FFT rows) for the DINO legs, 10.7 ms (256 rows) for coherent, at both
rates. The RT path's cost is rate-independent as designed; the native paths scale with the rate.

### 245.76 MSps (native capture rate), one pass (342 frames)

**"Sustained"** = no packet lost AND the NIC buffer pool (262,144 RX data buffers, ~1.1 s of input)
not draining. A replay this short can end loss-free while the pool drains, i.e. while the pipeline is
slower than real time; the pool's drain rate (robust slope of free buffers per second, as a fraction
of the packet arrival rate) exposes that. Column added after the first pass of this record, which
had reported the full-pipeline RT leg as keeping up.

| leg | frames emitted | packets shed | NIC pool drain | sustained | detector GPU wall p50 / p95 (ms) | GPU | live == offline |
|---|---|---|---|---|---|---|---|
| DINO-FT RT (M5-S), full pipeline, run 1 | 342/342 | 0 | **8.3% of arrivals** | **no** | 22.1 / 27.0 (model path 17.2) | 68% | bit-exact, 342/342 |
| DINO-FT RT (M5-S), full pipeline, run 2 (another GPU job running) | 334/342 | 2.3% | exhausted after ~5 s | no | 22.8 / 29.9 | 66% | 63% of frames (shifted after the drop) |
| **DINO-FT RT (M5-S), lean** (no snipper, no dashboard) | **342/342** | **0** | 0.6% / 0.3% (2 runs) | **yes** | 17.7 / 19.9 | 54% | **bit-exact, 342/342** |
| coherent, full pipeline (2 runs) | 681/684 | 0 | 0.0% | **yes** | 0.5 / 1.3 | 30-35% | bit-exact (IoU 1.0, 99% identical) |
| DINO-FT native (M2_dr) | 72/342 | 79% | exhausted | no | 208 / 272 | 65% | frames corrupt after drops |
| DINO-FT native (M5-S) | 106/342 | 69% | exhausted | no | 127 / 156 | 91% | " |
| DINO-FT RT (M2_dr ViT-B) | 276/342 | 19% | exhausted | no | 30.1 / 36.2 | 86% | " (IoU median 0.26) |

- **The RT detector keeps up at 245.76 MSps on its own** (lean: 17.7 ms per frame vs the 21.3 ms
  budget, pool steady, every mask bit-exact with offline). **The full production pipeline does not**:
  sharing the GPU with the 10240-pt FFT, the snipper and the dashboard pushes the detector to
  22-23 ms per frame, ~8% slower than real time; run 1 survived the 7.3 s replay on buffering alone and
  run 2, with another session's GPU job active, ran out of buffers after ~5 s. Coherent keeps up with
  large margin in both configurations.
- Packets are shed at the NIC queue (`rx_out_of_buffers`; DPDK `missed` stays 0) once the pool is
  empty. The RT path needs **both** the downsample front-end and the ViT-S backbone: the ViT-B model on
  the same RT path (30 ms) loses 19% of the stream, the native paths 69-79%.

### Accuracy vs SigMF ground truth at 245.76 MSps (+24 dB), region det@0.5 unless noted

| leg | det@0.1 "found" | det@0.5 | effective det@0.5 (live) | px precision / recall / IoU | FP area | Narrowband FM | <2 MHz | >=60 MHz | ZC (12 us bursts) | Bluetooth |
|---|---|---|---|---|---|---|---|---|---|---|
| coherent | 0.958 | **0.826** | 0.826 | 0.755 / 0.507 / 0.377 | 0.0029 | **0.975** | **0.93** | 0.15 | **0.835** | 0.93 |
| DINO-FT RT (M5-S) | 0.918 | 0.763 | 0.763 | 0.776 / 0.806 / 0.642 | 0.0034 | **0.00** | 0.64 | **0.83** | 0.41 | 0.85 |
| DINO-FT native (M2_dr) | 0.850 | 0.748 | 0.102 (72 frames live) | **0.968 / 0.912 / 0.887** | **0.0011** | 0.75 | 0.91 | 0.91 | 0.09 | 0.92 |
| DINO-FT native (M5-S) | 0.848 | 0.708 | 0.119 | 0.731 / 0.706 / 0.532 | 0.0048 | 0.69 | 0.92 | 0.73 | 0.02 | 0.97 |
| DINO-FT RT (M2_dr) | 0.936 | 0.781 | 0.297 | 0.953 / 0.746 / 0.716 | 0.0027 | 0.00 | 0.60 | 0.55 | 0.60 | 0.73 |

(Offline = detector capability; live values are identical for the two detectors that kept up.)
Full tables: `results/summary.md`, `results/summary.accuracy.csv`; per-leg details in
`/tmp/usrp_spectrograms/rt_stress/<run>/eval/summary.md`.

**The RT trade-off is exactly the narrowband one.** The 10:1 bilinear frequency resize (245.76) drops
Narrowband FM from 0.75 (native) to 0.00 and the <2 MHz bucket from 0.91 to 0.64, and it smears
12 us ZC bursts (0.41 vs native 0.09 -- RT actually finds more of them). In exchange it keeps the
segmenter's clean masks (FP area 0.003 vs coherent's filled-box-penalized 0.38 IoU) and wins on
wideband signals (>=60 MHz: 0.83 vs coherent 0.15). Overall "found" rate (det@0.1): coherent 0.958,
RT 0.918, native 0.850.

### 491.52 MSps (double speed), full pipeline, 6 loops (1026 frames)

| leg | offered rate | frames emitted | packets shed | detector GPU wall p50 (ms) | emit cadence p50 (ms) | CHDR->FFT latency mean / max | backlog events | GPU |
|---|---|---|---|---|---|---|---|---|
| **DINO-FT RT (M5-S)** | 402 / 424 kpps | **70% / 67%** | **30% / 33%** | 32.7 / 33.8 | 34 / 35 | 113 / 968 ms | 1 | 70-72% |
| DINO-FT RT (M5-S) **no snipper** | 414 / 411 kpps | 85.5% / 87% | 14.6% / 13% | 28.6 / 28.2 | 29 | 90 / 316 ms | 1 | 74-77% |
| DINO-FT RT (M5-S) **lean** (no snipper, no dashboard; cabled loopback) | 449 kpps | 84.8% | 15.2% | 26.1 | 27 | - | 0 | 65% |
| coherent | 412 / 409 kpps | 97.4% / 98.0% | 2.2% / 1.8% | 0.4 | 12 | 40 / 279 ms | 36 / 21 | 71-73% |
| DINO-FT native (M2_dr) | 368 kpps | 7.8% | 92% | 417 | 418 | 1504 / 6086 ms | 1 | 87% |
| DINO-FT native (M5-S) | 322 kpps | 13.9% | 86% | 254 | 256 | 838 / 2689 ms | 0 | 89% |
| DINO-FT RT (M2_dr) | 409 kpps | 63% | 37% | 39.5 | 40 | 127 / 315 ms | 1 | 80% |

Two runs per row where repeated (first series / core-isolated series `*_491c`, `*_491d`). No leg is
sustained at this rate: every one exhausts the NIC buffer pool. Even the lean RT detector takes
26 ms per frame here against the 21.3 ms budget (another session's GPU job was also running).

- **The sender could not hold 480 kpps while the DPDK port was receiving**: tcpreplay reaches
  exactly 480,000 pps on this path when nothing consumes, but 402-424 kpps during the legs
  (pinning and priority did not change it). The offered load was therefore **~85-88% of 491.52 MSps**;
  the firehose case is worse than the numbers above.
- **The deployed RT detector does not keep up at this load on the dense composite.** Its own model
  path is 13 ms, but its whole-operator GPU wall time is 33 ms because the GPU is shared with the
  20480-pt FFT, the snipper's DDC on ~18 snippets per frame (10,573 snippets over 593 masks), the
  compression/sink and the dashboard. Removing the snipper alone recovers half of the deficit
  (67% -> 87% of frames). Coherent keeps 98% with periodic backlog events. The Argus composite is far
  denser than the live 2.4 GHz scene the demo runs on, so the per-frame claim at 491.52 is
  scene-dependent, not a property of the detector alone.
- **Per-frame ground-truth scoring is invalid once packets drop.** The CHDR converter builds a batch
  from the next N packets regardless of sequence gaps, so after the first lost packet every frame
  holds a shifted slice of the stream (online-vs-offline IoU collapses to ~0.03-0.17); `score_leg.py`
  flags these runs. The one run that kept its first loop mostly intact (`nosnip_491d`, 71% of
  paired frames identical) scores det@0.5 0.469 live vs 0.712 offline as a lower bound. Offline at
  the 491.52 geometry (the M5-S training geometry) the RT detector reaches det@0.1 0.876 / det@0.5
  0.712 / pixel IoU 0.712, coherent 0.822 det@0.5, Narrowband FM again 0.00 for RT vs 0.94 coherent.

### Robustness findings and fixes (fixes validated 2026-10-21 evening, cabled loopback, headless)

1. **CHDR converter crash after a GPU-side failure: fixed.** The first pass of this record called this
   a crash "under a hard consumer stall". Re-reading the two crash logs corrects that. The converter had
   queued 4 batches and emitted 0, and its oldest batch had waited 4.4 s for its own gather kernel.
   `dmesg` shows Xid 109 (context-switch timeout) then Xid 31 (MMU fault) for the app in the same
   second. The GPU stopped executing the app's work, in runs whose sender was pinned at real-time
   priority. The converter then (a) ignored the failed `cudaStreamSynchronize` in its backlog handler,
   (b) kept refilling ring slots whose batches had never been emitted, and (c) on the sticky error
   called `exit(1)` from its receive thread, so static teardown ran under live worker threads and
   segfaulted. A consumer stall is different: Holoscan backpressure simply stops scheduling the
   converter and the NIC drops, before and after this change.
   Fixes in `applications/usrp_freq_detection/CHDR_converter/chdr_rx.{h,cu}`:
   - a failed backlog sync or batch-event query is now fatal (previously treated as "not ready");
   - `fatal_cuda_error()` logs one line naming the CUDA error and pointing at `sudo dmesg | grep -i xid`,
     then ends the process with `_Exit(1)`: no teardown race, no SIGSEGV;
   - a ring-full guard sheds (and counts) a burst remainder instead of refilling the oldest slot while
     its batch is not yet emitted, covering a batch that completes mid-burst while the GPU is >= 4
     batches behind; new `ring_full_shed_bursts/pkts` in the per-second CHDR summary and exit report;
   - test knob `CHDR_DEBUG_FATAL_AT_BATCH=N` simulates the fatal error without faulting the GPU.
   The GPU-side trigger itself was not reproduced; it has not recurred in the ~20 legs run since the
   harness stopped using a real-time-priority sender.
2. **First live DINO frames stalled: fixed.** The init warm-up primed batch 4 of the per-tile model
   only, so the exact live path (bilinear resize + batch-2 forward + nearest upsample) autotuned on
   frame 1. The operator now warms that path for the 10240/20480-wide geometries.
3. **`visualization.enable: false` with the DINO detector deadlocked the graph: fixed.** The detector
   and snipper tap raw IQ, so with no dashboard the FFT output had no consumer and the graph
   deadlock-stopped after ~4 batches. `main.cpp` now builds the FFT/spectrogram branch only when
   something consumes it; the lean DINO-FT pipeline runs straight off the converter.
4. `NV_ERR_NO_MEMORY` from `_memdescAllocInternal` in the driver log: unchanged. It also appears on days
   without these runs, and no run failed on it.

Validation of the fixes (all headless, `SENDER_IFACE=enp1s0f1np1` over the new cable):

| check | run | result |
|---|---|---|
| lean DINO-FT, dashboard off (was a deadlock) | `v_lean_245` | 342/342 frames, 0 loss, pool steady, every mask bit-exact with offline |
| simulated fatal CUDA error at batch 50 | `v_fatal_245` | one FATAL line naming the error, `app exit status=1`, no SIGSEGV; 49 batches emitted first |
| 6 s detector stall at frame 60 (`DINO_DEBUG_STALL_FRAME/MS`) | `v_stall_245` | no crash, exit status 0; the NIC shed the stalled interval (442/684 frames), then normal |
| first-frame cost after the warm-up fix | `v_lean_245` | 59 / 57 ms for frames 1-2 (was 211 / 1188 ms) |
| full-pipeline regression, coherent | `v_full_coherent_245` | 681/684, 0 loss, bit-exact: unchanged |
| full-pipeline regression, DINO-FT RT | `v_full_rt_m5s_245` | 334/342, 2.3% loss: the pre-existing 8% deficit exhausted the pool under extra load from another GPU job; ring guard shed 0, no backlog events |

The ring guard never had to shed in any validation run, including the stall: backpressure stops the
converter before its ring fills. Its shed path covers only a GPU falling >= 4 batches behind
mid-burst, which was not reproduced.

### Discarded runs

`*_491b` (FIFO-pinned sender starved the app -> crashes above; `nosnip_noviz` deadlock);
`dinoft_rt_m5s_491d` (X display I/O error at 13:10 while the console was in use -> `double free`;
Holoviz window on `:1`); `coherent_491d` (a second GPU job, a DroneDetect DINO evaluation, started
on the shared GPU at 13:12). `lean_noviz_245`, `stall_6s_245`, `lean_noviz_491`, `native_m2dr_lean_491`
ran on an intermediate build of the ring guard (it could emit from inside the receive loop); kept
as performance data, superseded by the `v_*` runs for validating the fixes. Their logs stay under
`/tmp/usrp_spectrograms/rt_stress/`.

## Efficiency work: the full pipeline with the dashboard (2026-10-21, night)

**Ask.** A configuration with the dashboard on and the snipper optional (ideally on); a shorter FFT
for the RT front-end, since its output is bilinearly squeezed to 1024 columns anyway; and an
end-to-end pass for inefficiencies, aiming at full-rate real time with the dashboard.

**Outcome.** The full pipeline (dashboard rendered off-screen, snipper, compression, SigMF sink) now
sustains 245.76 MSps with the deployed M5-S RT detector, and holds every 491.52-geometry load the
sender can offer (376-393 kpps, 78-82% of the true rate). The short-FFT front-end is both cheaper and
more accurate; it ships as `config_live_v3_dino_ft_m5s_sf2048.yaml` (snipper on;
`pipeline.enable_signal_snipper: false` turns it off and keeps the dashboard and overlay). At the true
491.52 rate a half-rate capacity test projects ~94% GPU busy for that config and ~100% for the current
front-end, so 491.52 still needs an over-the-air confirmation.

### Where the time went (nsys, 245.76 MSps, before the changes)

nsys profiled the live pipeline with the snipper and without the dashboard (it cannot attach to the
Vulkan dashboard on this box) and the offline binary.

- **The GPU sat idle 41% of the time while the pipeline fell behind** (busy 58.6%; 209 of 234 frames).
  The detector thread was blocked on the host: per frame 6 stream/event syncs (~5.5 ms), 27 small
  `cudaMemcpyAsync` calls (4.3 ms), `cudaMallocAsync` slow-path spikes (3.6 ms; the default memory
  pool handed memory back after every sync) and a `cudaFree` of the emitted mask (1.7 ms, a
  device-wide sync).
- The MatX FFT added a 67 MiB device-to-device copy per frame.
- The autocast-traced TorchScript cast every weight fp32 -> bf16 on each forward: 131 cast kernels,
  1.8 ms per frame.
- The snipper copied the whole wide mask (5-10 MB) to the host on the legacy stream and labeled it on
  the CPU, although that mask is the 1024-column model output replicated 10x / 20x.
- The dashboard operators (FFT op, spectrogram pass-through, preview op) were lean.

### Changes

- **Stage 1, bit-exact, applies to every existing DINO-FT config.** Detector: an in-place cuFFT plan
  replaces the MatX FFT and its copy; the normalization clip parameters and the invalid-frame guard
  are computed on the GPU; one sync per frame; pooled emit masks and persistent events replace the
  per-frame allocations; the model path writes a compact mask without an internal sync. The mask
  message carries `col_repeat`, so the snipper labels the compact 1024-column mask (async copy on its
  own stream into pinned memory) and maps the boxes back. The app sets the default CUDA memory pool's
  release threshold to the maximum, so `cudaMallocAsync` stays on its fast path.
- **Stage 2, bit-exact.** `torch::jit::freeze` at load folds the per-forward weight casts
  (`torch_freeze`, default on; M5-S logits identical).
- **Stage 3, opt-in, changes the model input.** `downsample_fft_size: 2048` with
  `downsample_time_pool: true` transforms at 2048 points and area-pools rows back onto the reference
  row grid (5 short rows per row at 245.76, 10 at 491.52). The model still sees 2 tiles of 256x1024 at
  the trained row time and the emitted mask grid is unchanged, but each model column now integrates
  2 FFT bins instead of a 10:1 / 20:1 bilinear sample. `downsample_time_pool_domain` is `power` (dB of
  the mean power) or `db` (mean of the dB values).

| check | result |
|---|---|
| stage 1 vs the pre-change offline reference (245.76) | identical masks on every frame except frame 1, which the old reference computed before the warm-up autotune |
| stage 2 (frozen) vs stage 1 | 343/343 identical masks |
| snipper, snippets emitted before vs after | coherent 4790 = 4790; DINO-FT 3480 vs 3481, the difference in frame 1 |
| stage 3 live leg vs offline (`s3_score_245`, full pipeline) | 342/342 identical masks, det@0.5 0.846 both |

### Front-end accuracy (offline, SigMF ground truth, region det@ unless noted)

245.76 MSps (342 frames):

| front-end | det@0.1 | det@0.5 | px prec / recall / IoU | FP area | Narrowband FM | <2 MHz | >=60 MHz | ZC | Bluetooth | 16QAM |
|---|---|---|---|---|---|---|---|---|---|---|
| current (10240-pt, bilinear 10:1) | 0.918 | 0.763 | 0.776 / 0.806 / 0.642 | 0.0034 | 0.000 | 0.642 | 0.834 | 0.411 | 0.855 | 0.797 |
| 2048-pt, mean of dB | 0.921 | 0.746 | 0.751 / 0.831 / 0.641 | 0.0039 | 0.562 | 0.780 | 0.882 | 0.223 | 0.838 | 0.854 |
| **2048-pt, power** | **0.955** | **0.846** | 0.743 / 0.855 / 0.647 | 0.0044 | 0.562 | **0.814** | **0.924** | **0.587** | **0.906** | **0.889** |
| 4096-pt, mean of dB | 0.910 | 0.739 | 0.772 / 0.834 / 0.659 | 0.0028 | 0.375 | 0.674 | 0.870 | 0.288 | 0.846 | 0.800 |
| 4096-pt, power | 0.941 | 0.811 | 0.763 / 0.855 / 0.666 | 0.0032 | 0.438 | 0.720 | 0.905 | 0.507 | 0.906 | 0.843 |

491.52 MSps geometry (2x twin, 171 frames):

| front-end | det@0.1 | det@0.5 | px prec / recall / IoU | FP area | Narrowband FM | <2 MHz | >=60 MHz | ZC | Bluetooth | 16QAM |
|---|---|---|---|---|---|---|---|---|---|---|
| current (20480-pt, bilinear 20:1) | 0.876 | 0.712 | **0.936** / 0.740 / **0.712** | 0.0042 | 0.000 | 0.289 | 0.728 | 0.411 | 0.816 | 0.701 |
| 2048-pt, mean of dB | 0.896 | 0.696 | 0.756 / 0.813 / 0.642 | 0.0048 | 0.375 | 0.606 | 0.845 | 0.251 | 0.827 | 0.777 |
| **2048-pt, power** | **0.926** | **0.789** | 0.743 / 0.838 / 0.646 | 0.0056 | 0.333 | 0.577 | **0.861** | **0.557** | **0.847** | **0.817** |
| 4096-pt, mean of dB | 0.892 | 0.707 | 0.774 / 0.821 / 0.659 | 0.0035 | 0.250 | 0.426 | 0.837 | 0.332 | 0.847 | 0.748 |
| 4096-pt, power | 0.916 | 0.770 | 0.763 / 0.843 / 0.665 | 0.0040 | 0.167 | 0.426 | 0.852 | 0.530 | 0.847 | 0.774 |

Coherent for reference: det@0.5 0.826 (245.76) and 0.822 (491.52). JSON: `results/frontend_variants_{245,491}.json`.

- **2048-pt power pooling is the best detector at both rates**: det@0.5 +0.083 (245.76) and +0.077
  (491.52) over the current front-end, ahead of coherent at 245.76. It recovers narrowband FM
  (0.00 -> 0.56 / 0.33) and the <2 MHz bucket, and finds more ZC bursts.
- **Its cost is mask tightness at 491.52**: pixel precision 0.936 -> 0.743 and FP area 0.0042 -> 0.0056,
  i.e. masks spill wider around signals. At 245.76 precision barely moves (0.776 -> 0.743).
- Mean-of-dB pooling detects less than power pooling everywhere except the narrowband buckets at
  491.52 (Narrowband FM 0.375 vs 0.333, <2 MHz 0.606 vs 0.577). A burst that fills one of the pooled
  rows is diluted by the mean of dB but keeps its energy in the dB of the mean (ZC 0.22 vs 0.59 at
  245.76).
- M5-S was trained on the current wide-FFT front-end (`dino_fine_tuning/src/frontend.py`), so these
  gains come without retraining; a model fine-tuned on the pooled front-end may tighten the masks.

### Live results (full pipeline: off-screen dashboard + snipper + compression + sink, no mask dump)

| stage | rate | offered | frames | packet loss | NIC pool drain | sustained | detector GPU p50 | GPU mean |
|---|---|---|---|---|---|---|---|---|
| before (`nodump_dinoft_rt_m5s_245`) | 245.76 | 240 kpps | 682/684 | 0.37% | 5.6%, exhausted | no | 21.7 ms | 69% |
| stage 1 (`s1_full_245`) | 245.76 | 240 kpps | 684/684 | 0 | 0 | **yes** | 16.6 ms | 71% |
| stage 2 (`s2_full_245`) | 245.76 | 240 kpps | 684/684 | 0 | 0 | **yes** | 16.1 ms | 66% |
| stage 3, 2048-pt power (`s3_full_245`) | 245.76 | 240 kpps | 684/684 | 0 | 0 | **yes** | 14.8 ms | 63% |
| before, mask dump on (`dinoft_rt_m5s_491`, `_491c`) | 491.52 | 402 / 424 kpps | 70% / 67% | 30% / 33% | exhausted | no | 32.7 / 33.8 ms | 70-72% |
| stage 1 (`s1_full_491`) | 491.52 | 391 kpps | 1024/1026 | 0.19% | 0 | no | 24.4 ms | 85% |
| stage 2 (`s2_full_491`) | 491.52 | 376 kpps | 1026/1026 | 0 | 0 | **yes** | 22.3 ms | 81% |
| stage 3, 2048-pt power (`s3_full_491`, `_pm`) | 491.52 | 393 / 385 kpps | 1026/1026 | 0 | 0 | **yes** | 20.8 ms | 75-78% |

Per-run CSV: `results/efficiency_stages.csv`. The stage runs and the 245.76 "before" run replayed
over the cable (`SENDER_IFACE=enp1s0f1np1`) with the app on cores 0-17 and the sender on 18-19 at
normal priority. The "before" 491.52 rows are the earlier in-NIC runs with the mask dump on, which
costs real time too (at 245.76 the dump-on run drained 8.3% of arrivals, the dump-off run 5.6%).

### Capacity at the true 491.52 rate

The sender cannot offer 480 kpps while the app runs: it peaks at ~385-393 kpps over the cable or the
in-NIC path, and `--pps-multi` (`_pm`) and core placement did not raise it. Instead, the 491.52
geometry was replayed at half the packet rate (240 kpps: same frames, one every 42.7 ms instead of
21.3 ms) and the GPU busy fraction was scaled to the true rate:

| config | GPU busy at half rate | GPU busy time per frame | projected GPU busy at 480 kpps |
|---|---|---|---|
| current front-end, frozen (`half_dinoft_rt_m5s_491geom`) | 49.9% | 21.3 ms | ~100% |
| 2048-pt power (`half_dinoft_rt_m5s_sf2048p_491geom`) | 46.9% | 20.0 ms | ~94% |

nvidia-smi utilization counts time with any kernel resident, so this is an estimate. The stage 3 run
at 393 kpps (82% of the rate) ran at 75% GPU, which scales to ~92%: consistent. The short-FFT config
should hold 491.52 with a thin margin on this dense composite, and the current front-end sits at the
edge. Live over-the-air scenes are sparser (fewer snippets), which adds margin.

### TensorRT is not a drop-in speed-up for M5-S

The model was exported to ONNX (opset 17, static batch 2x1x256x1024) and built with `trtexec`
(TensorRT 10.13 in the container, CUDA graphs, no host transfers):

| model forward, 2 tiles | ms |
|---|---|
| TensorRT FP16 | 17.9 |
| TensorRT BF16 | 18.3 |
| PyTorch eager, bf16 autocast | 13.9 |
| operator model path, frozen TorchScript bf16 (offline; resize + model + threshold + stitch) | 10.5 |

The engine's time is spread over ~200 fused kernels with no single hot spot, so the libtorch path
stays. Also rejected: `torch::jit::optimize_for_inference` (logits change, max abs diff 1.5, no
speed-up) and CUDA-graph capture of the TorchScript forward (the traced RoPE issues
`prim::NumToTensor` size ops and host-to-device copies; it would need a re-export).

## Code changes (uncommitted, rebuilt into the container 2026-10-21)

- `applications/usrp_freq_detection/CHDR_converter/chdr_rx.{h,cu}`: fatal CUDA error handling
  (`fatal_cuda_error`, checked backlog sync and event query, `_Exit(1)` instead of `exit(1)`),
  ring-full shed guard + counters, test knob `CHDR_DEBUG_FATAL_AT_BATCH` (finding 1).
- `applications/usrp_wideband_signal_detection/main.cpp`: FFT/spectrogram branch only when consumed
  (finding 3); wiring for the mask dump below.
- `applications/usrp_wideband_signal_detection/mask_dump_sink.hpp` (new) + `CMakeLists.txt`:
  detector-agnostic emitted-mask dump, off unless `mask_dump.dir` (config) or `USRP_MASK_DUMP_DIR`
  (env). D2H on a private non-blocking stream; background writer.
- `operators/finetuned_dino_detector/finetuned_dino_detector.{hpp,cu}`: exact-live-path warm-up
  (finding 2), native-path inference timing, whole-operator per-frame GPU time
  (`detector_inference_ms` / `detector_model_path_ms` metadata), test knob
  `DINO_DEBUG_STALL_FRAME` + `DINO_DEBUG_STALL_MS`.
- `operators/finetuned_dino_detector/finetuned_dino_mask_dump.{hpp,cpp}`: manifest gains
  `host_mono_ns, detector_ms, chdr_batch_index, chdr_pkts, chdr_expected_pkts, chdr_partial, nonzero`.
- Harness (this directory): leg configs render the dashboard off-screen
  (`visualization.holoviz.headless: true`); `run_leg.sh` records `app exit status` in `app.log` and
  passes the two test knobs; `collect_metrics.py` reports the NIC pool drain and a `sustained` flag.

Efficiency work (stages 1-3):

- `operators/finetuned_dino_detector/finetuned_dino_detector.{hpp,cu}` + `CMakeLists.txt` (`CUDA::cufft`):
  in-place cuFFT plan per channel, GPU clip parameters and GPU invalid-frame guard with a pinned
  readback, one `cudaEventSynchronize` per frame, pooled emit masks, persistent events, compact mask
  expanded by a nearest-column kernel and tagged with `col_repeat`; new params `torch_freeze`
  (default true), `downsample_time_pool` (default false) and `downsample_time_pool_domain`
  (`db` default, `power`) with the window and pooling kernels.
- `operators/finetuned_dino_detector/finetuned_dino_torch_helpers.{hpp,cpp}`: `load(..., freeze)`;
  `forward_downsampled` writes the compact mask into a caller buffer without syncing, skips the resize
  when the width already equals `nfft`, and times with persistent events.
- `applications/usrp_wideband_signal_detection/spectrogram_visualization.hpp`:
  `DetectorMaskMessage::col_repeat` (default 1).
- `applications/usrp_wideband_signal_detection/signal_snipper.{hpp,cu}`: compact-mask labeling when
  `col_repeat > 1` (gather kernel, async copy on the snipper stream into pinned memory, boxes mapped
  back to wide columns).
- `applications/usrp_wideband_signal_detection/main.cpp`: default CUDA memory pool release threshold
  set to the maximum at startup.
- New top-level config `config_live_v3_dino_ft_m5s_sf2048.yaml`; app README §6 / §7 and
  `dino_fine_tuning/README.md` updated.
- Harness: short-FFT legs `dinoft_rt_m5s_sf{2048,4096}{,p}` in `make_configs.py`; `run_leg.sh` knobs
  `MASK_DUMP=0`, `APP_CORES`, `SENDER_CORES`, `SENDER_EXTRA`, `NSYS_CAPTURE_S` / `NSYS_MODE`;
  `score_offline.py` (new) scores offline runs side by side; `collect_metrics.py` falls back to the
  sampled detector time when no mask dump exists.

## Reproduce

```bash
cd applications/usrp_wideband_signal_detection/infocom_evals/signal_detection_experiments/dino_ft_rt_loopback_stress
python3 make_configs.py                                    # leg configs from the production configs
# pcap (frame-aligned: 171 x 10240 packets), DPDK-port MAC, replay ports 49160->1240
python3 ../../../../usrp_freq_detection/replay_rx_to_buff.py --sigmf-data ~/Datasets/ARGUS/main_captures/capture_snr_p24dB.sigmf-data \
  --out-pcap $COMP/capture_snr_p24dB_171f.spark.pcap --dst-mac 4c:bb:47:2c:45:13 --src-mac 4c:bb:47:2c:45:14 --max-samples 1793064960
tcprewrite --portmap=49153:49160,1234:1240 --infile=$COMP/capture_snr_p24dB_171f.spark.pcap --outfile=$COMP/capture_snr_p24dB_171f.lb.pcap
./run_offline_refs.sh ~/Datasets/ARGUS/main_captures/capture_snr_p24dB.sigmf-data                       # 245.76 references
OUT=/tmp/usrp_spectrograms/rt_stress/offline_x2 ./run_offline_refs.sh ~/Datasets/ARGUS_x2_scratch/capture_snr_p24dB_x2.sigmf-data   # 491.52 references
./run_series.sh 245760000 240000 1 $COMP/capture_snr_p24dB_171f.lb.pcap ~/Datasets/ARGUS/main_captures/capture_snr_p24dB.sigmf-meta 245 \
    dinoft_rt_m5s coherent dinoft_native_m2dr dinoft_native_m5s dinoft_rt_m2dr
OFFLINE_ROOT=/tmp/usrp_spectrograms/rt_stress/offline_x2 ./run_series.sh 491520000 480000 6 $COMP/capture_snr_p24dB_171f.lb.pcap \
    ~/Datasets/ARGUS_x2_scratch/capture_snr_p24dB_x2.sigmf-meta 491 dinoft_rt_m5s dinoft_rt_m5s_nosnip coherent
python3 summarize.py --runs ... --out results/summary.md
# efficiency work: offline front-end variants + GT scoring, then full-pipeline legs without the mask dump
V="dinoft_rt_m5s dinoft_rt_m5s_sf2048 dinoft_rt_m5s_sf2048p dinoft_rt_m5s_sf4096 dinoft_rt_m5s_sf4096p"
OUT=/tmp/usrp_spectrograms/rt_stress/offline_v3 ./run_offline_refs.sh ~/Datasets/ARGUS/main_captures/capture_snr_p24dB.sigmf-data $V
python3 score_offline.py --root /tmp/usrp_spectrograms/rt_stress/offline_v3 --stem capture_snr_p24dB \
    --meta ~/Datasets/ARGUS/main_captures/capture_snr_p24dB.sigmf-meta --frames 342 --out results/frontend_variants_245.json $V
export SENDER_IFACE=enp1s0f1np1 MASK_DUMP=0
./run_leg.sh s3_full_245 dinoft_rt_m5s_sf2048p 245760000 240000 2 $COMP/capture_snr_p24dB_171f.lb.pcap
./run_leg.sh s3_full_491 dinoft_rt_m5s_sf2048p 491520000 480000 6 $COMP/capture_snr_p24dB_171f.lb.pcap
./run_leg.sh half_dinoft_rt_m5s_sf2048p_491geom dinoft_rt_m5s_sf2048p 491520000 240000 3 $COMP/capture_snr_p24dB_171f.lb.pcap
python3 collect_metrics.py /tmp/usrp_spectrograms/rt_stress/s3_full_491
```
`$COMP` = `~/Documents/holoscan_waveform_generation/composition/composites`. With the cabled
loopback in place, `SENDER_IFACE=enp1s0f1np1` replays over the wire instead of the in-NIC path
(dst MAC unchanged). Leg configs render the dashboard off-screen, so no window opens. Test knobs:
`CHDR_DEBUG_FATAL_AT_BATCH=N`, `DINO_DEBUG_STALL_FRAME=N DINO_DEBUG_STALL_MS=ms` (env of `run_leg.sh`).

## Next steps

- Confirm 491.52 MSps over the air with the X410 after re-cabling: `config_live_v3_dino_ft_m5s_sf2048.yaml`
  for about 2 minutes, watching the `RX worker summary` free-buffer count and `Fell behind` lines. The
  sender here tops out near 393 kpps, so the true 480 kpps load has only been projected (~94% GPU).
- Decide whether the short-FFT config becomes the demo's DINO target: one line in
  `infocom_evals/pycodec_e2e/demo_conductor.py` (`CONFIG_BY_DETECTOR_LIVE`) and one in
  `bash_scripts/loopback_channel_supervisor.sh`. Trade-off: +0.08 det@0.5 and less GPU, wider masks
  at 491.52.
- Fine-tune on the pooled front-end (port the pooling into `dino_fine_tuning/src/frontend.py`) to
  tighten the masks again.
- Remaining GPU margin levers, untested: a smaller dashboard FFT at 491.52, `emit_stride: 2` on the
  dual-channel config, and a re-export of the model without the RoPE size ops so the forward can be
  captured as a CUDA graph.
- Narrowband recall: max-pool frequency reduction before the model (`detector_mask_sanity/README.md`)
  remains untested; the pooled front-end already recovers part of it.
