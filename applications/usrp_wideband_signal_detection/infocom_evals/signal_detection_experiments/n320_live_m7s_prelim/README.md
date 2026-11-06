# M7-S live preliminary test on the USRP N320 (2026-11-06)

**Question.** Do the new M7-S live configs run end to end in real time? The configs are
`config_live_v3_dino_ft_m7s.yaml`, `config_live_v3_dino_ft_m7s_sf2048.yaml` and
`config_live_v3_dino_ft_m7s_two_channel.yaml`. With the X410 powered off, the radio connected
was a USRP N320, serial 3459703. This tests function and real-time behaviour only. It does
**not** test accuracy (there is no ground truth) and it does **not** test the 491.52 MSps budget:

- the N320's top rate is 245.76 MSps per channel (master clock 245.76 MHz);
- it has one 10 GbE data link;
- so single-channel ran at 245.76 MSps (half the X410 rate) and dual-channel at 2 × 122.88 MSps.

## Setup that works (cable-free, nothing changed on the bench)

The N320's SFP0 (`192.168.10.2`) is cabled to physical port 1 of the Spark's CX7 NIC at 10 Gb/s.
Its RJ45 management port is `192.168.20.4` via `enP7s7`. Physical port 1 appears as two PCIe
functions with separate MACs:

- `0000:01:00.1`, kernel netdev `enp1s0f1np1`, MAC `:14`: carries UHD control (`addr=192.168.10.2`);
- `0002:01:00.1`, MAC `:18`: owned by DPDK. The N320 streams to `dest_mac=4c:bb:47:2c:45:18`, and
  the NIC delivers it to this function by MAC.

Radio settings (environment for `bash_scripts/start_radio_stream.sh`):

```
CTRL_ADDR="192.168.10.2,master_clock_rate=245.76e6" ADAPTER=sfp0 DEST_ADDR=192.168.10.1
DEST_MAC=4c:bb:47:2c:45:18 ANTENNA=TX/RX KEEP_HDR=1 RATE=245.76e6 (dual: 122.88e6)
```

Config changes compared with the production M7-S configs (`configs/`):

- DPDK `address: 0002:01:00.1`;
- the `CHn_CHDR_Headers_RX_CPU` region `buf_size: 16`;
- N320 centre frequencies;
- a dashboard screenshot.

Run each test with `run_leg.sh <leg> <rate> <channels> <freqs> <ports>`, which calls the real
`run_live_demo.sh` with `SKIP_TOPOLOGY_CHECK=1` and `USRP_SAMPLE_RATE_HZ`.

What did **not** work, and why:

1. **DPDK on the same PCIe function as UHD control** (`0000:01:00.1`). Once DPDK started in
   isolation mode with mlx5 hardware steering (`dv_flow_en=2`), the kernel stopped receiving the
   N320's replies. UHD then failed with "No devices found".
2. **UHD control over the RJ45.** UHD caps samples per packet by the control link's MTU, which is
   1,500 on the RJ45. Packets came out at 364 samples (1,456-byte payload). Setting `mtu=8000`,
   `spp=1024` in the stream arguments, or `recv_frame_size` on the device did not change this.
3. **`raw_payload` mode.** The app's DPDK buffer split is 42 B Ethernet/IP/UDP + a CHDR-header
   region + 4,096 B of IQ. The X410 uses 512-bit CHDR, so its header is 64 B (4,202-byte frames).
   The N320 uses 64-bit CHDR, and in `raw_payload` mode it sends no header, which doesn't fit that
   layout: the RX queue dropped every packet. With `full_packet` mode (`--keep-hdr`) the N320
   sends 16 B (8 B header + 8 B timestamp) + 4,096 B on every packet. With the header region set
   to 16 B, the layout matches. The CHDR converter reads only the payload segment, so it needs no
   code change.

Script changes made for this test (both backward compatible; the X410 path is unchanged):

- `usrp_freq_detection/rx_to_remote_udp.py`: new `--antenna` option. The spp port index is now
  `chan % radio.get_num_output_ports()`: the N320 has one channel per radio block, so the old
  `chan % 2` addressed a port that doesn't exist on channel 1.
- `bash_scripts/start_radio_stream.sh`: optional `ANTENNA` and `KEEP_HDR` settings.

## Results (60 s receive per test, over the air, gain 30, antennas on TX/RX)

| test | config | rate | packets received / sent | lost (all at start-up) | DINO-FT mean per inference | "Fell behind" | GPU busy mean / p95 |
|---|---|---|---|---|---|---|---|
| single, CH0 2.44 GHz (VERT2450) | `n320_m7s_single` | 245.76 | 14,349,115 / 14,400,000 (99.65 %) | 50,885 | **13.1 ms** (2,784 frames; budget 21.3 ms) | 1 | 74 % / 89 % |
| short FFT, same scene | `n320_m7s_sf2048` | 245.76 | 14,228,226 / 14,400,000 (98.8 %) | 171,774 | **13.2 ms** (2,752 frames) | 1 | 70 % / 87 % |
| dual, CH0 2.44 GHz + CH1 915 MHz (VERT900) | `n320_m7s_dual` | 2 × 122.88 | 13,194,542 / 14,400,000 (91.6 %); first run 95.1 % | 1.2 M; first run 0.71 M | 14.7–15.9 ms (emit_stride 4) | 2 | 60 % / 93 % |

- **Steady state is clean in every test.** Every DPDK drop warning comes before or right at the
  first DINO-FT inference, during pipeline and model warm-up. After that there are no drops for
  the rest of the run, and NIC `rx_out_of_buffer` / `rx_discards_phy` stay at 0. The longer dual
  warm-up costs about 3–5 s of data at start. This start-up loss is not specific to M7-S.
- Processing ratio is 1:1 (100 %). Average CHDR throughput was 248.7 MSps for single and
  122.6 MSps per channel for dual.
- In the dual test, the huge "RX out of buffers" figure is `rx_mbuf_allocation_errors`, which
  counts allocation retries during that warm-up. It is not a packet count; the real loss is
  `rx_missed`.
- **The mask dump is too heavy for live use at this rate.** A first single-channel test with
  `mask_dump` on wrote 512 × 10,240 masks (about 230 MB/s, 9.8 GB in 42 s). It lost about 12 % of
  packets, the detector p95 rose to 38 ms, and there were 26 "Fell behind" events. For live
  real-time verdicts, use the dashboard screenshot instead.

### What the dashboards show

- **Single** (`dashboard_single_245M.png`): the switch label reads "DINO v3 M7-S". The masks
  follow the 2.4 GHz Wi-Fi bursts, the narrow line near 2.447 GHz, and sparse emitters at
  2.50–2.56 GHz (the 2.5 GHz cellular band). The quiet span is not painted.
- **Short FFT** (`dashboard_sf2048_245M.png`): same scene. It marks more narrow emitters and gives
  denser, wider masks across the Wi-Fi channels, as M5-S did on this front end (lower pixel
  precision). Its accuracy relative to the standard front end still needs ground truth.
- **Dual** (`dashboard_dual_2x122M.png`): CH-1 at 915 MHz marks the 869–894 MHz LTE/CDMA downlink
  carriers as wide blocks with the gaps left clean, and the narrow 935–960 MHz downlink carriers
  as individual lines. CH-0 shows the 2.4 GHz Wi-Fi.

## Still open

- 491.52 MSps (single and dual): needs the X410, or two 10 GbE links. These tests ran at half the
  X410 per-channel rate (single) and a quarter of the dual load.
- Accuracy: these are visual checks only. Capture a labelled scene, or use the coherent detector as
  a teacher, to score M7-S against the short-FFT front end.
- Start-up loss: worth a warm-up (a dummy inference before arming DPDK) if the first seconds of a
  capture matter.
