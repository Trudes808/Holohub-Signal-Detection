# DGX Spark (GB10) vs. the x86 bench (RTX 4000 Ada): why dual-channel runs slower

**Context.** The original bench ran this pipeline on an x86 host with an NVIDIA RTX 4000 Ada
Generation GPU and sustained dual-channel 491.52 Msps with per-frame detection at full mask width
(the `emit_stride1` two-channel config is that bench's artifact-capture profile). On the DGX Spark
port, dual-channel runs at ~75% frame coverage with detection every 2nd frame (~435 ms latency);
single-channel runs essentially clean. This note explains why, with measured numbers.
GB10 figures marked *(measured)* were taken on this box (torch cu130 / CUDA probe, 2026-08-13);
RTX 4000 Ada figures are vendor specifications.

## 1. Hardware comparison

| | RTX 4000 Ada (x86 bench) | GB10 (DGX Spark) |
| --- | --- | --- |
| Architecture | Ada Lovelace (AD104), discrete PCIe | Blackwell iGPU in an Arm SoC (sm_121) |
| SMs / CUDA cores | 48 / 6,144 | **48 / 6,144** *(measured)* — identical |
| FP32 peak | ~26.7 TFLOPS (boost 2.61 GHz class) | ~30 TFLOPS theoretical @ 2.42 GHz; **19 TFLOPS sustained** *(measured, cuBLAS 8k matmul)* |
| L2 cache | **48 MB** | **24 MB** *(measured)* |
| Memory | 20 GB GDDR6, 160-bit, **360 GB/s dedicated** | 128 GB LPDDR5X unified, ~273 GB/s theoretical, **225 GB/s** *(measured, idle, large sequential copy)* — **shared** with 20 CPU cores, NIC DMA, OS, display |
| Power envelope | 130 W for the GPU alone | ~140 W for the **entire SoC** (GPU + 20 Arm cores + fabric) |
| NIC → GPU path | GPUDirect RDMA into dedicated VRAM (system RAM untouched) | NIC DMAs into the same LPDDR5X the GPU computes from (zero-copy, but shares the one memory system) |

The punchline: **the two GPUs have identical core counts.** The gap is not "a smaller GPU" — it is
the *memory system* and the *power/contention envelope* around the same number of SMs.

## 2. What the pipeline demands (dual-channel full rate)

Dual 491.52 Msps = 48,000 FFTs/s of 20,480 points. Rough sustained DRAM traffic:

| Stage | Approx. traffic |
| --- | --- |
| NIC ingest write + converter read (sc16) | ~8 GB/s |
| Converter writes complex float | ~8 GB/s |
| FFT read + write (multi-pass for 20,480 pts) | ~16–30 GB/s |
| Power/dB pass | ~12 GB/s |
| Detector (several passes: box-mean, morphology, persistence, mask) | ~15–25 GB/s |
| Spectrogram preview + waterfall history | ~10 GB/s |
| **Total** | **~70–100 GB/s sustained**, in many small, launch-heavy kernels |

- On the **RTX 4000 Ada**, that traffic ran against 360 GB/s of *private* bandwidth — and much of
  the inter-pass traffic never reached DRAM at all, because Ada's **48 MB L2** holds entire
  working sets (one 20,480-bin float row is only 80 KB; whole per-batch intermediates fit).
- On **GB10**, the same traffic runs against ~225 GB/s *shared* with the DPDK packet DMA
  (~4 GB/s), 20 CPU cores, the desktop, and the render path — with **half the L2** to absorb
  inter-pass reuse. Bandwidth-bound streaming kernels degrade roughly with effective bandwidth,
  and that is exactly what we measured.

## 3. The measured evidence (from the tuning sessions on this box)

1. **Kernel inflation under contention** — the same coherent-detector kernel costs
   **3.7 ms/frame single-channel but 14.5 ms/frame dual-channel** (identical per-channel work);
   spectrogram preview adds ~10–11 ms/frame. ~4× wall-time inflation with 2× the kernels in
   flight is the signature of a saturated shared memory system, not of insufficient SMs.
2. **Networking exonerated** — moving the DPDK RX workers onto the isolated CPU cores (5,7)
   changed throughput by 0%; the NIC delivers both flows at full wire rate with
   `ring_full_drops=0`. The converter's output queue sits pegged at its maximum: frames are
   *ready* and downstream won't take them.
3. **Render path exonerated** — removing display decimation entirely (render every frame,
   ~36 fps/channel) cost nothing measurable.
4. **Detection cadence is the lever that works** — `emit_stride: 2` (detect every 2nd frame)
   freed ~400 ms/s of GPU and bought +30% throughput and −25% latency. Only reductions in GPU
   work move the needle: conclusive that the pipeline is GPU-bound.
5. **Sustained ceiling** — the pipeline sustains ~28–38k FFT/s aggregate depending on batching,
   vs the 48k FFT/s dual full rate demands (~1.3–1.7× over budget). Single-channel (24k) fits.
6. Sustained compute also sits below peak (19 of ~30 TFLOPS in a pure matmul): the SoC shares one
   power/thermal budget across CPU+GPU, and the cu130/sm_121 software stack is first-generation.

## 4. Why the x86 bench didn't have this problem

- **Isolation**: NIC→VRAM over PCIe (GPUDirect/peermem) meant packet traffic, CPU work, and the
  desktop never touched the GPU's 360 GB/s. On the Spark, everything shares one 273 GB/s pool —
  the unified memory that makes the port *possible* (no GPUDirect needed) is also the contended
  resource.
- **Cache**: 48 MB vs 24 MB L2 — on Ada, consecutive pipeline passes over the same rows largely
  hit L2; on GB10 they round-trip DRAM.
- **Power**: 130 W dedicated to the GPU vs ~140 W across the whole SoC; sustained dual-channel
  load runs CPU cores (DPDK polling, scheduler) *and* GPU flat-out inside one envelope.
- Net: with identical SM counts, the Ada 4000 delivers roughly 2× the *effective* memory
  throughput to this workload — which matches the ~2× gap between dual-channel demand and what
  GB10 sustains.

## 5. What it means / what would close the gap

Current operating points on GB10 (committed configs; dual re-measured 2026-08-13 after the
Tier A code-level GPU work, commit `ab8e7f78` — see `gpu_optimization_plan.md`):

| | Ingest | Coverage | Latency | Visualization |
| --- | --- | --- | --- | --- |
| Single-channel | 491.52 Msps | **97.8% measured** | ~200 ms | full rate |
| Dual-channel | 2× 491.52 Msps | **100% measured** (zero NIC drops); detection every 2nd frame | ~320 ms | ~36 fps/ch |

**2026-08-13 revision to this note's conclusion.** The original diagnosis above attributed the
dual-channel gap entirely to the memory system. The Tier A optimization pass showed that a large
share of it was **host-side serialization inside the operators** (per-stage timing syncs,
blocking counter readbacks, per-frame cudaMalloc/cudaFree — stalls the baseline profiling was
itself subject to): removing them took dual stride-2 coverage from 79.1% to 100% with no config
change. The contention argument in §§1–4 still holds, but as a smaller effect: at
`emit_stride: 1` (2× detector work) the pipeline still sheds ~15%, and the detector's pipeline
stage inflates 7.4 → 9.8 ms/frame under the doubled kernel load — that residual is the true
memory-system ceiling on this SoC.

What closing the remaining (stride-1) gap takes:

1. **Kernel fusion** (Tier B, in progress): fuse the power/dB/frontend/score chain and the
   separable morphology pairs, fold the fftshift pass into consumers, share the power surface
   with the preview — cuts full-surface DRAM round-trips directly.
2. **Cheaper detection footprint**: the dual config's full-width (20,480-column) masks exist for
   snipper/artifact capture; a demo-only detector profile could halve detector traffic.
3. **Different hardware class**: any discrete GPU with dedicated GDDR restores the isolation the
   x86 bench had.

The Spark's trade after Tier A: a ~1 kW bench collapsed into a ~140 W box that ingests dual
500-class Msps at **100% coverage** with live detection (every 2nd frame) and visualization.
