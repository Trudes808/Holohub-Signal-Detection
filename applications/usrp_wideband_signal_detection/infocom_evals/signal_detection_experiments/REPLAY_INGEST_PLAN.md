# Plan: SigMF Replay onto the Live DPDK Ingest Path (`replay_rx_to_buff`)

## Objective

Replace the live USRP (`applications/usrp_freq_detection/rx_to_remote_udp.py`) with a host-side
replay sender that reads a SigMF recording (`*.sigmf-data` + `*.sigmf-meta`), frames the samples
into the **exact same CHDR/UDP packets** the USRP emits, and pushes them onto the **same physical
DPDK ingest path** the live app already uses — so the unchanged `usrp_wideband_signal_detection`
app exercises the real `chdr_converter -> FFT -> detector` pipeline on recorded signals.

Tested via the standard flow: `./rebuild_demo_container_app.sh` + a `config*_replay.yaml` run
through the existing wrappers. **No app/operator code changes while working on the replay.**

## Chosen architecture (and why)

The DPDK `advanced_network` manager binds the physical NIC `0000:a2:00.1` exclusively
([adv_network_dpdk_mgr.cpp:700](../../../../operators/advanced_network/advanced_network/managers/dpdk/adv_network_dpdk_mgr.cpp#L700)),
DMAs IQ payload straight into **GPU memory** (`kind: device` + hardware buffer-split), and steers
packets with `rte_flow`. None of that can be reproduced by a host socket or a software DPDK vdev
without invasive patches to the shared operator + the CHDR converter.

**Decision: physical loopback.** An SFP cable connects the card's kernel-owned port
`0000:a2:00.0` (`ens4f0np0`) to the DPDK-owned port `0000:a2:00.1` (`ens4f1np1`). The replay
sender emits real Ethernet frames out `ens4f0np0`; they loop into the DPDK NIC exactly as USRP
traffic would. This preserves GPUDirect, buffer-split, and flow steering with the app untouched.

## What the live path actually requires (verified)

- **Wire packet = `[42B eth/ip/udp][32B CHDR header line][4096B sc16 payload]`** (decoded from a
  live capture — see `chdr_reference_header_bytes.txt`). Total CHDR length field = 4128 bytes.
  - CHDR header u64 (LE): `dst_epid=3`, `length=4128`, `seq_num` (+1/packet), `num_mdata=0`,
    `pkt_type=7` (DATA_WITH_TS), `eov/eob/vc=0`. Followed by an 8-byte timestamp, then 16 bytes of
    zero padding → 32-byte header line (`CHDR_W=256`). Payload (sc16 IQ) begins at offset 32.
  - The advanced_network manager splits each wire packet into 42B / 64B / remaining segments by the
    `buf_size`s; the converter consumes only segment 2 at
    [chdr_rx.cu:1032](../../../../applications/usrp_freq_detection/CHDR_converter/chdr_rx.cu#L1032).
    The 64B split over-captures the first 32B (8 samples) of payload into seg1, so the converter
    effectively drops the first 8 samples/packet — a pre-existing live quirk the replay inherits.
  - The converter ignores header *contents* (splits by fixed byte offset), so replay only needs the
    byte layout right, with a plausible incrementing `seq_num`/timestamp.
- **Payload = interleaved little-endian `int16` I,Q**, scaled by `1/0x7FFF` in the kernel
  ([chdr_rx.cu:56-86](../../../../applications/usrp_freq_detection/CHDR_converter/chdr_rx.cu#L56)).
  (The "network order" comment there is misleading — no byte-swap happens; validate empirically.)
- **`num_complex_samples_per_packet: 1024`** -> 4096-byte payload per packet (matches USRP `--spp 1024`).
- **Flow match:** UDP `src=49153`, `dst=1234` — replay packets must use these ports.
- **Sample rate / center freq come from config, not packets** — so the replay config must encode them.

## Inputs (from the SigMF meta — confirmed against `generated_inputs/`)

- `core:datatype: cf32_le` (complex float32 LE) -> convert to `int16` via `round(x * 0x7FFF)`, clip.
- `core:sample_rate: 245760000` (**245.76 MSps, NOT 500e6 for this replay**) -> drives FFT span + replay pacing.
- `core:num_channels: 1`; `captures[0].core:frequency` -> visualization center frequency.
- Annotations (`QPSK`, `ZC`, freq edges) -> ground truth for validating detector output.

## Deliverables

### 1. `applications/usrp_freq_detection/replay_rx_to_buff.py`  *(host-side, next to `rx_to_remote_udp.py`)*

Mirrors the USRP's role. Responsibilities:
1. Parse `*.sigmf-meta`; read `*.sigmf-data` with the declared dtype (`np.fromfile`).
2. Convert complex samples -> interleaved LE `int16` (sc16), scaled/clipped.
3. Chunk into 1024-sample (4096-byte) payloads; prepend a **64-byte CHDR header** with an
   incrementing sequence number (see "CHDR header" risk below).
4. Wrap each in UDP(`src=49153,dst=1234`) / IP(`dst=192.168.10.51`) / Ethernet
   (`dst=<DPDK port .1 MAC>`, `src=<ens4f0np0 MAC>`).
5. **Output a `.pcap`** (preferred) for `tcpreplay` egress, with an optional `--live` AF_PACKET
   raw-socket sender for convenience.
6. CLI: `--sigmf-data PATH` (meta inferred), `--dest-mac`, `--src-iface ens4f0np0`,
   `--rate-mbps` (pacing), `--out-pcap PATH`, `--loop N`.

Egress: `sudo tcpreplay -i ens4f0np0 --mbps=<rate> replay.pcap` (C-speed, paceable). Python
raw-socket send is the fallback; it cannot approach line rate.

### 2. `applications/usrp_wideband_signal_detection/config_coherent_power_performance_single_channel_replay.yaml`

Copy of the single-channel baseline, with **only** these deltas (top-level path so the wrappers sync it):
- `advanced_network` — **unchanged** (same DPDK NIC `0000:a2:00.1`, same flow match 49153/1234).
- `fft.span: 245760000`, `fft.reference_span_hz: 245760000`,
  `fft.resolution: 12000` (= round(245760000 / 20480)). `transform_points`/`window_points`
  stay 20480. (If a future SigMF uses a different rate, recompute these from its `core:sample_rate`.)
- `visualization.renderer.center_frequency_hz: <captures[0].core:frequency>`.
- `chdr_converter.partial_batch_drop_timeout_ms: 5000` (tolerate sub-real-time replay) and relax
  `degraded_shutdown_*` thresholds so slow replay doesn't trigger a panic reset.
- Detector params: leave at baseline for the first run.

## Throughput note

245.76 MSps × 4 B = ~7.9 Gbit/s. Python sockets can't pace that; `tcpreplay` can get close on this
NIC. For **correctness** validation we replay sub-real-time and rely on the bumped
`partial_batch_drop_timeout_ms` — the detector's persistence/frame logic is frame-count based, so
wall-clock rate only affects throughput metrics, not detection results.

## Bring-up sequence

1. **Capture ground-truth CHDR header** (de-risks framing): run any live/known-good config with
   `chdr_converter.log_packets: true`, grab one packet's segment-0/1/2 hex from the logs. Lock the
   replay script's 64-byte header to match. *(If no USRP is available, derive from the RFNoC CHDR
   spec for X410 and validate via the loopback + `log_packets`.)*
2. **Physical**: connect loopback cable `.0 <-> .1`; ensure `.1` is DPDK-bound (`after_reboot.sh`),
   `ens4f0np0` is link-up. Confirm the DPDK port MAC and whether promiscuous mode is on.
3. **Generate** `replay.pcap` from a chosen SigMF capture.
4. **Run the app**: `CONFIG_NAME=config_coherent_power_performance_single_channel_replay.yaml ./run_coherent_power_performance.sh`.
5. **Replay**: `sudo tcpreplay -i ens4f0np0 --mbps=<rate> replay.pcap`.
6. **Observe**: converter `ttl_pkts_recv` / avg Msps in the stop() report; visualization; saved
   masks under `/tmp/usrp_*`.

## Validation

- Packets received and batches completing (no partial-flush panic in converter logs).
- Spectrogram frequency axis correct (signals land at the SigMF annotation freq edges).
- Detector masks plausibly match annotated `QPSK`/`ZC` regions.
- Cross-check the same SigMF through the existing **offline** path
  (`run_offline_cuda_detector_eval.cpp`, which already reads SigMF) for consistency.

## Open risks / must-verify-at-implementation

1. **Exact 64-byte CHDR header layout** — resolve via `log_packets` capture (step 1). Highest risk.
2. **Payload endianness** — validate with a known single-tone capture.
3. **Dest MAC / promiscuous mode** — confirm the DPDK port MAC and whether L2 dest must match.
4. **Loopback link-up** between two ports of the same card.
5. **Rate vs converter timeout** — mitigated by `partial_batch_drop_timeout_ms` + `tcpreplay` pacing.

## Status (PROVEN end-to-end 2026-06-30)

First successful replay: 2000 synthetic packets from a `cf32_le` SigMF traversed the loopback
(`.0`->`.1`) into DPDK, were parsed by `chdr_converter` (0 malformed / 0 errored / 0 panic), and a
batch was emitted to FFT/detector. The approach works.

Tuning learned from the first run:
- A full FFT batch = `num_ffts_per_batch (512) * num_packets_per_fft (20)` = 10,240 packets
  (~10.5 M samples) — larger than a single ~5 M-sample recording. For sustained/complete batches
  either loop the pcap (`tcpreplay --loop 0 --mbps 500`) or lower `num_ffts_per_batch` so a batch
  fits one recording.
- Pace egress (`--mbps`) so packets arrive steadily; otherwise a short burst arrives in ~1 s and the
  converter idle-timeout-flushes the partial batch (benign).



- `applications/usrp_freq_detection/replay_rx_to_buff.py` — built. Verified locally: parses a
  `cf32_le` 245.76 MSps SigMF, emits frames whose bytes match the captured CHDR layout
  (epid=3, length=4128, pkt_type=7, 16B pad, payload@offset 32, udp 49153->1234, dst-MAC `…6b`,
  4170 B/frame). pcap output + `--live` AF_PACKET mode.
- `config_coherent_power_performance_single_channel_replay.yaml` — built. Baseline + `fft.span`/
  `reference_span_hz` = 245760000, `resolution` = 12000, `partial_batch_drop_timeout_ms` = 5000,
  `visualization.enable` = false. DPDK stays on committed `.1`.

### End-to-end replay procedure (remaining: loopback cabling)

1. Disconnect the USRP from `.0`; connect a loopback cable `ens4f0np0 (.0) <-> ens4f1np1 (.1)`.
   Confirm `ethtool ens4f1np1` shows `Link detected: yes`.
2. `python3 replay_rx_to_buff.py --sigmf-data <capture>.sigmf-data --out-pcap replay.pcap`
3. `cd usrp_wideband_signal_detection && ./run_torchscript_performance_test.sh config_coherent_power_performance_single_channel_replay.yaml`
4. `sudo ip link set ens4f0np0 mtu 9000` (frames are ~4170 B)
5. Replay at the true sample rate: `sudo tcpreplay --preload-pcap --loop 0 --pps 240000 -i ens4f0np0 replay.pcap`
   (240000 pps = 245.76 MSps ≈ 8 Gbit/s; `--pps` ties packet rate to sample rate, `--loop 0` sustains
   it past one file. The script prints the exact pps for the loaded recording.)
6. Watch the app's `RX worker summary` `packets=`/`rate` and `timeout_flushes`/`ring_full_drops`;
   verify detector output vs SigMF annotations.

## Dynamic range across attenuation levels (`attenuation_dB_*`)

The dataset spans `dB_0` (loudest) to `dB_60` (quietest). Measured int16 peaks at `--scale 32767`:
`dB_0`≈22925 (~14.5 bits), `dB_25`≈1358 (~10.4 bits), `dB_60`≈757 (~9.6 bits). Note `dB_0→dB_25`
drops the expected ~25 dB, but below ~`dB_25` the recorded amplitude is **noise-floor-dominated**
(signal buried), so peaks flatten.

Implications for handling the range:
- **Quantization is not the bottleneck.** Even `dB_60` uses ~9.6 bits; keep `--scale 32767`
  (preserves the true relative levels between files). The replay script prints per-file int16
  headroom and only warns when the peak is genuinely tiny (<64 counts).
- **`--gain-db` won't recover buried signals.** Boosting a noise-dominated capture scales signal and
  noise together — SNR unchanged. Use it only for genuinely quantization-limited (signal-limited) input.
- **Display must track level** via `visualization.renderer.db_floor`/`db_ceil`:
  `dB_0`≈`-22/+35` (baseline), `dB_25`≈`-40/+10` (verified good), quieter levels → lower both.
  Display-only; no effect on detection.
- **Detector across levels:** `coherent_power` is quantile/relative-based (`frontend_reference_q`,
  `frontend_row_q`), so it should adapt to level — verify detection holds from `dB_0` down to the
  noise-limited floor; tune `fast_power_floor_db`/`fast_score_threshold` only if it degrades.

## Out of scope

- No changes to `advanced_network` or the CHDR converter.
- Dual-channel replay (single-channel first).
- Real-time-rate replay (correctness-focused first).
