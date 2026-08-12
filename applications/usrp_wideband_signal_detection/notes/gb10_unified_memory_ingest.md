# Unified-Memory Ingest Recipe for DGX Spark (GB10)

**Status:** design / proposal (2026-08-10). Supersedes the GPUDirect-RDMA ingest model on this box.
**One-liner:** GB10 is an integrated, cache-coherent CPU+GPU SoC with no discrete VRAM, so the
NIC must DMA into **host** memory and the GPU reads that *same* memory coherently. On this SoC that
is the zero-copy path — not a "CPU staging" penalty. This doc is the recipe to switch the
`usrp_wideband_signal_detection` DPDK ingest from `kind: device` (broken here) to a coherent
host-memory region.

---

## 1. Why the original path is dead on GB10

The app's `advanced_network` DPDK manager allocates the RX data region as `kind: device`
(`cuMemAlloc`, GPU VRAM) and registers it with the ConnectX-7 for DMA via `rte_dev_dma_map` →
mlx5 `ibv_reg_mr` on the GPU virtual address. That requires **GPUDirect RDMA** (NIC writing into a
GPU BAR), which needs a kernel peer-memory provider (`nvidia-peermem`) or a dma-buf export.

**GB10 supports neither**, because it is not a discrete GPU. Measured device attributes
(`cuDeviceGetAttribute`, driver enumerates it as *"NVIDIA Tegra NVIDIA GB10"*):

| Attribute | Value | Meaning |
| --- | --- | --- |
| `INTEGRATED` | **1** | CPU+GPU share physical LPDDR5X (one memory pool) |
| `UNIFIED_ADDRESSING` | 1 | one virtual address space for host+device pointers |
| `PAGEABLE_MEMORY_ACCESS` | **1** | GPU can access *any* pageable host memory directly |
| `PAGEABLE_MEM_ACCESS_USES_HOST_PAGE_TABLES` | 1 | ATS: GPU walks the host page tables |
| `CONCURRENT_MANAGED_ACCESS` | 1 | CPU+GPU can touch managed memory concurrently |
| `CAN_MAP_HOST_MEMORY` | 1 | pinned host memory is device-mappable (UVA) |
| `GPU_DIRECT_RDMA_SUPPORTED` | **0** | no classic GPUDirect RDMA |
| `GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED` | **0** | — |
| `DMA_BUF_SUPPORTED` | **0** | CUDA will not export device memory as a dma-buf |

Corollaries (all verified): `cuMemGetHandleForAddressRange(...DMA_BUF_FD...)` on `cuMemAlloc`
memory → `EINVAL`; `cuMemCreate(gpuDirectRDMACapable=1)` → "invalid device ordinal". So the earlier
candidate fixes (dma-buf patch to the DPDK manager, `nvidia-peermem`, DOCA GPUNetIO into device
memory) are **all moot** — there is no device BAR to DMA into. This is architectural.

## 2. The insight: on a coherent SoC, "host memory RX" IS the zero-copy GPU path

On a **discrete** GPU, "CPU staging" = NIC → host RAM → a slow **PCIe copy** → separate GPU VRAM.
That penalty is what we want to avoid. **GB10 has no separate GPU VRAM** — the bytes the NIC writes
into host memory are the exact bytes a CUDA kernel reads, over the shared LPDDR5X, coherently.
There is nothing to stage *to*. With `PAGEABLE_MEMORY_ACCESS=1`, a plain host pointer is a valid
device pointer inside a kernel (no copy, no registration).

Bandwidth sanity: the stream is ~7.9 Gb/s (single channel @ the negotiated 245.76 Msps, sc16) to
~32 Gb/s (dual 500 Msps). Shared LPDDR5X is ~273 GB/s. Even a redundant coherent copy would be
1–3 % of memory bandwidth; true zero-copy removes even that.

## 3. Relevant facts in the current code

**Manager memory kinds** (`operators/advanced_network/advanced_network/manager.cpp:124-168`):

| `kind:` | Backing | NIC registration (`register_mrs`) | GPU-accessible on GB10? |
| --- | --- | --- | --- |
| `device` | `cuMemAlloc` (VRAM) | `rte_dev_dma_map` → needs peermem → **FAILS (EINVAL)** | n/a (broken) |
| `host_pinned` | `cudaHostAlloc` (pinned) | `rte_extmem_register`+`rte_dev_dma_map` on **host** mem → works, no peermem | Yes — UVA device pointer (`CAN_MAP_HOST_MEMORY`) |
| `huge` | `rte_malloc_socket` (DPDK hugepage) | **skipped** — DPDK auto-registers its own hugepage heap | Yes — via ATS (`PAGEABLE_MEMORY_ACCESS`) |
| `host` | `malloc` | `rte_extmem_register`+`rte_dev_dma_map` | Yes — via ATS |

**Converter hot path** (`applications/usrp_freq_detection/CHDR_converter/chdr_rx.cu`): the datapath
kernel `place_packet_data_kernel` reads the received packet segment pointers
(`get_segment_packet_ptr(burst, seg, p)`) **directly on the GPU** and writes the complex output
tensor. Those pointers point into the RX memory region. So if the RX region is host memory, the
kernel dereferences host pointers — which GB10 permits (UVA + ATS) with no copy. (There are a few
`cudaMemcpyDeviceToHost` calls at lines ~161 and ~996, but those are debug/fingerprint/log paths,
not the throughput path.)

## 4. The recipe

**Core change:** set the RX **data** region's `kind` from `device` to a host kind, so the NIC DMAs
into coherent memory the GPU reads in place. The CHDR header regions are already `huge` (host) and
unaffected. NIC registration then uses a normal host MR — **no peermem, no dma-buf, works today**.

Config edit (per current single-channel config `advanced_network.cfg.memory_regions`):

```yaml
      - name: "CH1_Data_RX_GPU"     # rename to CH1_Data_RX_HOST for clarity
        kind: "host_pinned"          # was "device"   (or "huge" — see variants)
        affinity: 0
        access: [ local ]
        num_bufs: 131072
        buf_size: 4096
```

### Variant A — `host_pinned` (cudaHostAlloc)
- **Pros:** explicitly page-locked + UVA-mapped → unambiguous zero-copy device pointer; classic
  integrated-GPU zero-copy; DMA-safe (pinned).
- **Cons:** not hugepage-backed; must confirm `rte_extmem_register`/`rte_dev_dma_map` accept it with
  the manager's `GPU_PAGE_SIZE` page-size arg; potential TLB pressure vs hugepages.

### Variant B — `huge` (DPDK hugepage)
- **Pros:** most DPDK-idiomatic RX backing (mbuf pools love hugepages); auto-registered by DPDK; 1 GB
  pages already set up on this box; GPU reads it via ATS (`PAGEABLE_MEMORY_ACCESS=1`).
- **Cons:** GPU access relies on ATS/system-allocated-memory rather than an explicit pinned mapping
  — needs a runtime confirm that the converter kernel can dereference the hugepage pointer (expected
  to work on GB10, but verify). Not `cudaPointerGetAttributes`-registered.

**Recommendation:** try **Variant B (`huge`)** first as a config-only change (most DPDK-native, and
GB10's ATS should make it directly GPU-readable), with **Variant A (`host_pinned`)** as the fallback
if the converter kernel can't cleanly use the hugepage pointer.

## 5. Implementation plan

1. **Config-only spike:** flip the data region `kind` (Variant B) in
   `config_coherent_power_perf_dynamic_single_channel.yaml`; run live. Success = DPDK init passes MR
   mapping (no EINVAL), packets received, spectrogram/detector produce output, viz shows signal.
2. **Verify GPU access mode:** confirm the converter kernel reads the RX buffer with correct data
   (not garbage) — i.e. the host pointer is truly coherent on the GPU. If Variant B misbehaves,
   switch to Variant A.
3. **Zero-copy audit:** confirm the hot path has no hidden host↔device bounce for the RX buffer;
   keep the debug `cudaMemcpy` paths guarded/off in production configs.
4. **Propagate:** apply the same `kind` change to the other live configs (perfreq, cuda_dino,
   snipper, capture, replay, two-channel) once validated.
5. **Buffer sizing:** revisit `num_bufs`/`buf_size` for host backing (528 MB/ch today); ensure the
   hugepage pool (8×1 GB) covers single + dual channel.
6. **Document + (optionally) upstream:** note that on integrated GPUs the manager should prefer a
   host kind; consider a manager-level auto-fallback when `DMA_BUF_SUPPORTED==0 && INTEGRATED==1`.

## 6. Risks / to-verify
- `rte_extmem_register` page-size assumptions for `host_pinned` (Variant A).
- Whether the `advanced_network` operator hands the converter a pointer typed/annotated as device
  memory anywhere that asserts on `cudaPointerGetAttributes` (would need a tweak for host kinds).
- mlx5 MR for host memory is expected fine; confirm no residual `kind==DEVICE` assumptions in
  `register_mrs`/`map_mrs` gate the host path oddly.
- Coherency/ordering: ensure the manager's completion → kernel-launch ordering is a true
  happens-before on GB10 (it is for same-stream work; watch cross-stream).

## 7. Validation plan
- Frozen/loopback first if possible (cable-loopback replay config) to get determinism, then live.
- Compare live detector masks against the offline baseline we already validated
  (`comprehensive_ordered.sigmf-data`) for sanity.
- Watch shutdown stats: `Received packets` > 0 (nonzero) confirms the DMA path works.

## 8. Open questions (for the user)
1. Target **memory kind**: `huge` (DPDK-native, ATS zero-copy) vs `host_pinned` (explicit UVA
   zero-copy) — or should I benchmark both?
2. **First step**: fastest config-only spike to prove ingest, or a full zero-copy code audit first?
3. **Channel scope**: bring up single-channel first, or go straight to dual-channel (two 500 Msps)?
4. **Change scope**: OK to edit the shared `operators/advanced_network` manager if needed (e.g. a
   host-kind assertion), or keep everything app-local/config-only?
5. **Perf target**: is there a latency/throughput bar this must hit, or is "correct + real-time at
   245.76 Msps" the initial goal?

## 9. Spike result (2026-08-10) — RECIPE VALIDATED (init path)
Config-only change `kind: device` → `kind: huge` on `CH1_Data_RX_GPU` in
`config_coherent_power_perf_dynamic_single_channel.yaml`, run live single-channel:
- **NO `Fail to create MR` / `unable to DMA map` / EINVAL** — the GPUDirect blocker is GONE.
- DPDK EAL init OK, port 0 + 1 RX queue configured, RX flow (`udp_dst 1234`) added, advanced-network
  workers started, Holoscan graph activated, ran the full 30 s cleanly, no CUDA errors.
- `Received packets: 0 / Missed packets: 0` → no data arrived yet (sender not streaming / SFP
  data-plane not yet aligned). So the **NIC→host-memory ingest + graph init are proven**; the GPU's
  coherent read of the host RX buffer will be exercised once packets flow.
- Chosen options: `kind: huge`, config-only, single-channel first, app-local changes only.

## 10. Live data blocked by a UHD↔DPDK control conflict (NOT a memory issue) (2026-08-10)
Cabling confirmed: X410 QSFP0 (`192.168.21.2`) → host **`enp1s0f1np1` (`0000:01:00.1`, carrier=1,
MAC 4c:bb:47:2c:45:14)**; `enp1s0f0np0` (`01:00.0`) has NO cable. So configs + `after_reboot.sh`
were rebound from `01:00.0` → **`01:00.1`** (I had mis-defaulted to port 0 earlier).
With the app running (DPDK on `01:00.1`, `flow_isolation: true`), the X410 could not be commanded:
- Control over the QSFP (`addr=192.168.21.2`) → `No devices found`: `flow_isolation` makes mlx5 take
  the whole port from the kernel, so UHD can't reach the X410 over `192.168.21.x`.
- Control over 1 GbE mgmt (`mgmt_addr=192.168.30.3`) → device found but RFNoC graph fails
  (`Timed out getting recv buff for management transaction`): RFNoC CHDR control still rides the QSFP
  data plane, which is DPDK-isolated.
App RX stayed `bursts=0 packets=0` (idle-polling) because the X410 was never commanded to stream.
The host-memory ingest path itself is proven (DPDK init + graph OK) — this is an X410/UHD/DPDK
coexistence problem, orthogonal to the GB10 memory work.
**Candidate fixes:** (a) `flow_isolation: false` so the kernel keeps the port for UHD control while
DPDK steers only `udp_dst 1234` (bifurcated) — most likely; (b) command the X410 to start streaming
BEFORE the app isolates the port; (c) a dedicated control link / UHD's own DPDK transport. Needs the
bench's known-good sender+app coexistence recipe.

**Tried:** `flow_isolation: false` + promiscuous did NOT help — while the app's DPDK owns
`enp1s0f1np1`, the host kernel gets **100% loss** pinging the X410 (`192.168.21.2`); mlx5 routes all
RX to DPDK regardless of isolation. **Single-QSFP coexistence is not possible here.**

## 11. RESOLUTION: two-QSFP topology (matches the original bench) (2026-08-10)
Two physical links so control and data never share a port:
- **Control link** — X410 SFP0 (`192.168.21.2`) → host **`enp1s0f1np1` (`0000:01:00.1`)**, kept in the
  **kernel** (no DPDK). UHD controls the X410 here (`--args addr=192.168.21.2`).
- **Data link** — X410 **SFP1** → a **second** host CX7 port (recommend **`enp1s0f0np0` / `0000:01:00.0`**,
  free, same card, different PCIe function so it's independent of the kernel control port),
  **DPDK-bound** for the `udp_dst 1234` stream.
Then: rebind the config's `advanced_network` DPDK bind to the DATA port (`0000:01:00.0`); give the
data host port an IP on SFP1's subnet; run the sender with control over SFP0 and data out SFP1
(`--adapter sfp1`, `--dest-addr`/`--dest-mac` = the data host port). Exact subnets/MACs discovered
after the second cable is connected.

**State after connecting the 2nd cable (2026-08-10):** both CX7 ports carrier-up. Control link
confirmed: `ip route get 192.168.21.2` → `enp1s0f1np1`, ARP `192.168.21.2 = 00:80:2f:40:9f:f9`
(X410 SFP0). **Data-link addressing still unknown:** both host ports currently share `192.168.21.1/24`
(leftover; only `enp1s0f1np1` uses it for control); `enp1s0f0np0` (data) has carrier but NO ARP
neighbor on `192.168.21.x` → the X410 SFP1 is on a different subnet. UHD doesn't advertise SFP1; SSH
to the X410 (`root@192.168.21.2`) needs a password. BLOCKED pending the X410 SFP1 IP / data subnet
(user bench knowledge or X410 login). Data host port MAC = `4c:bb:47:2c:45:13`.

**Two-link coexistence PROVEN (2026-08-10):** with the app's DPDK on the DATA port (`01:00.0`),
`ping 192.168.21.2` via `enp1s0f1np1` stays **0% loss** — control is unaffected because it's a
different PCIe function. So control(kernel)+data(DPDK) on two links works with no conflict.
Sender `--adapter sfp1 --dest-mac 4c:bb:47:2c:45:13 --dest-addr 192.168.21.1` connected + set
491.52 Msps, but the app still saw `packets=0` — the X410 did not deliver to the data port.
**Last missing piece:** which physical X410 QSFP the data cable is in (→ correct `--adapter`) and the
data-link subnet / X410 SFP1 IP (so the X410 routes/ARPs the stream out that QSFP to `enp1s0f0np0`).
Needs bench data-link details or X410 SSH creds to inspect/set the SFP1 network config.

## 12. LIVE END-TO-END WORKING ✅ (2026-08-10)
X410 SFP map (read via `ssh root@192.168.21.2`, no password; BusyBox): **`sfp0=192.168.10.2`**
(MAC `00:80:2f:40:9f:f5`), **`sfp1=192.168.21.2`** (MAC `00:80:2f:40:9f:f9`), `eth0=192.168.30.3`
(1 GbE mgmt). The X410's `addr` (192.168.21.2) is on **sfp1**, not sfp0 — the earlier `--adapter sfp1`
streamed out the CONTROL cable, hence 0 packets. Correct topology:
- **CONTROL link**: X410 `sfp1` (192.168.21.2) ↔ host **`enp1s0f1np1`** (192.168.21.1, **kernel**).
- **DATA link**: X410 `sfp0` (192.168.10.2) ↔ host **`enp1s0f0np0`** (192.168.10.1 → **DPDK `0000:01:00.0`**).

**Known-good live recipe (single channel):**
1. Host data-port IP: `sudo ip addr flush dev enp1s0f0np0 && sudo ip addr add 192.168.10.1/24 dev enp1s0f0np0`
   (the host netplan currently mislabels both CX7 ports as 192.168.21.1 — fix persistently later).
2. App config `config_coherent_power_perf_dynamic_single_channel.yaml`: `CH1_Data_RX_GPU kind: huge`,
   DPDK bind `0000:01:00.0`, `flow_isolation: true`. Run it (DPDK owns the data port only; control
   port stays kernel — verified `ping 192.168.21.2` = 0% loss while DPDK owns 01:00.0).
3. Sender (control on sfp1, stream out sfp0):
   `python3 rx_to_remote_udp.py --args "addr=192.168.21.2" --freq 2400e6 --rate 500e6 --gain 30 \
     --channels 0 --adapter sfp0 --dest-addr 192.168.10.1 --dest-port 1234 \
     --dest-mac-addr 4c:bb:47:2c:45:13 --spp 1024`

**RESULT:** RX `0.480 Mpps` = **491.52 Msps** (full X410 rate), `malformed=0`, `ring_full_drops=0`;
`FFT ingress live`; the GPU reads the NIC-written host hugepage buffer **coherently, zero-copy** — the
unified-memory ingest recipe confirmed with real RF, no CUDA errors. The X410 negotiates 491.52 Msps
(not 500). **Tuning note:** chdr→fft latency ~321 ms (batching: `num_ffts_per_batch=512` ×
`num_simul_batches=4`) — functional but high; reduce batch sizes for lower live latency.

## 13. Latency tuning + cuda_dino live (2026-08-10)
**Coherent latency**: `num_ffts_per_batch` (must equal `fft.num_bursts`) 512 → **128**:
- 512 → chdr→fft ~321 ms (stable). 32 → ~60 ms BUT the converter backlogs (needs ~750 batch/s, tops
  out ~264) → `panic_resets` climb + `out_q` pegged (data loss) — too aggressive.
- **128 → ~115 ms**, converter keeps up (`queued≈emitted`, `out_q≈0`), NIC `ring_full_drops=0`, full
  491.52 Msps. **Landed here.** Residual latency is downstream-dominated (converter batch is only
  ~5–20 ms of it); further cuts need `num_simul_batches` / spectrogram-history / backpressure tuning.
- `panic_resets` (~1/s) = converter self-heal after 3 consecutive partial-batch timeouts
  (`degraded_reset_partial_flush_threshold=3`, 1 s cooldown) — in ALL configs, not backlog; raise the
  threshold / `partial_batch_drop_timeout_ms` to quiet it.

**cuda_dino live**: applied the same `kind: huge` fix (batch kept 512 — DINO-inference-bound, not
converter-bound). Runs live on GB10: TorchScript DINOv3 loads (backend `cuda_partial`, input
1024×1024), receives live 512×20480 tensors, spectrogram preview computes, no CUDA errors/crash. It is
**DINO-limited**: `allow_backpressure_valve` throttles ingest to ~0.03 Mpps (~4 batch/s = the DINO's
sustainable rate); fft→preview ~321 ms avg (up to ~1.4 s). Full-rate DINO on 491 Msps isn't feasible
(heavy ViT); the valve processes a steady subset gracefully. To push more through DINO: smaller model,
coarser chunking (`chunk_bandwidth_hz`/`max_tokens_per_inference`), or accept the subset.

## 14. Dual-channel live on GB10 (2026-08-10, validation + tuning session)
Two-channel config (`config_coherent_power_performance_emit_stride1_two_channel.yaml`) brought up
live: both X410 RF channels (`--channels 0 1`, e.g. 2400/1000 MHz) stream out **sfp0** as two UDP
flows (dst 1234/1235) into the single DPDK data port — **both flows measured at full rate
(0.48 Mpps ≈ 491.52 Msps each, ~983 Msps aggregate), ring_full_drops=0**, dual-panel HoloViz up,
detections on both channels. Empirical findings:
- **Both channels use `udp_src 49153`** (tcpdump-verified) — the checked-in flow matches are correct;
  the README's "ch1 src 49154" note is wrong for this bench.
- The sender supports only ONE `--adapter` — both channels' data share one X410 QSFP by design.
  Control keeps sfp1. (A per-QSFP data split would require a different control path.)
- **Converter/pipeline ceiling ≈ 24k FFT/s aggregate** (≈ one channel's full rate). Dual-channel at
  full wire rate therefore processes ~60% of frames per channel (graceful shed, no crash, no NIC
  drops). Sweep: batch 512/4w = symmetric, ~575 ms; 256/4w = asymmetric (one ch starves — avoid);
  384/4w = ~470 ms mild backlog; **256 + `worker_thread_number: 8` = symmetric ~350 ms (landing
  point)**. Raising workers fixed the asymmetry, not the ceiling. `render_every_n_frames: 3` keeps
  display load sane (~20-30 fps/ch); detection still every emitted frame.
- Restart hygiene: after killing the app, WAIT for the process to fully exit before relaunching
  (`EAL: Cannot create lock ... Is another primary process running?` otherwise), then clear
  `/dev/hugepages/nwlrbbmqbh*` and `/var/run/dpdk/nwlrbbmqbh`.

**Shed-mode addendum (2026-08-12).** Batch size decides WHERE the over-ceiling excess is shed:
- **512**: shed at batch assembly — out_q stays ≤5, RX mempools healthy (~250k/262k free), calm
  logs. Coverage varies run-to-run (~29/47 per ch, sometimes one channel reaches full 47/47 while
  the other drops to ~25 with NIC rx_missed bursts — scheduling-dependent). **Committed default.**
- **256**: shed via out_q backpressure — out_q pegs ~48 batches × 10240 pkts, which pins the ENTIRE
  262144-buffer RX mempool (`seg2 avail=0`) → NIC starves → rx_missed ~200k/s + continuous
  `Fell behind in processing on GPU!` error spam. Reverted; do not use for dual full rate.
- **No half-rate option**: the CG_400 FPGA image refuses lower rates (`Requesting invalid sampling
  rate ... Actual rate is: 491.52 MHz`) — dual lossless would need a different FPGA image.
- New wrapper `bash_scripts/start_radio_stream.sh` = the OTA radio-control step with bench
  defaults (it IS real over-the-air collection, not a synthetic source).

**Zero-out root cause (2026-08-12).** User-visible "spectrogram zeroes out then restarts" during
live runs = the CHDR converter's `dpdk-rx-queue-warning-threshold` soft resync (`chdr_rx.cu:886`):
`degraded_shutdown_on_rx_queue_warning_threshold: 3` triggered a FULL channel flush after every 3rd
NIC `rx_q_errors` micro-drop warning — losing far more data than the micro-drop and blanking the
display (~1 resync/s at full rate). Fixed in the dynamic single + two-channel configs:
threshold → 0 (partial-flush self-heal retained) and single-channel `num_bufs` 131072 → 262144.
Verified: `panic_resets=0` over a full-rate run (was ~1/s), continuous display. Single-channel
landing point: batch 256 (128 needs 187.5 batch/s > ~140 capacity → sheds ~25%; 256 needs 93.75
→ ~82% coverage limited by NIC micro-drop bursts, latency ~200 ms).

**Viz decimation (clarification):** `num_ffts_per_batch` is COMPUTE batching, NOT display decimation.
Display rate = `render_every_n_frames` (frame stride) + the spectrogram source→display row
downsample (shown in-UI as "Processing Ratio"/"Vis Ratio"). Detection runs `emit_stride: 1` (every
frame). So the batch change did not alter viz decimation or detection cadence.
