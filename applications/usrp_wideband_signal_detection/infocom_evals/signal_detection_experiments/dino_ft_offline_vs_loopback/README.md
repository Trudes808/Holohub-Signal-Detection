# DINO-FT offline-vs-loopback A/B (2026-09-14)

**Question.** The live/real-time DINO-FT (M2_dr) output "looks atrocious." Is that the model, its
real-time geometry, or the real-time ingest path?

**Method.** Run the *identical* compiled `finetuned_dino_detector` operator two ways on the SAME
1 s capture, so the only variable is how IQ reaches the operator:

- **offline leg** — `run_cuda_dino_offline_file.py --detector cuda_dino_finetuned
  --config config_live_v3_dino_ft.yaml` reads the SigMF file directly (clean, deterministic).
- **loopback leg** — the same capture packetized to a CHDR pcap and replayed over DPDK at the LIVE
  491.52 MSps into the real `chdr_converter -> FFT -> detector` pipeline. The operator dumps every
  emitted mask via the new (default-off) `debug_mask_dump_dir` param
  (`config_loopback_eval_dino_ft_rt.yaml`).

Both legs use `real_time_downsample: true` (wide-FFT -> bilinear resize -> circular edge -> flatten),
`emit_stride: 4`, `threshold: 0.95` — byte-identical detector blocks. Masks are 512 x 20480 uint8 on
the same wide-FFT grid (frame = 20*1024*512 = 10,485,760 complex samples). Aggregate metrics are used
because the legs are not frame-aligned (loopback loops the 1 s capture).

Capture: `~/Documents/captures/x410_ota_2g4_gain10_20260908.sigmf-{data,meta}` (491.52 MSps, 2.4 GHz,
1 s OTA, ci16). pcap: `.../composites/x410_ota_2g4_gain10_20260908.spark.pcap`.

## Result — the model is fine; the real-time INGEST intermittently corrupts frames

| metric | offline (file) | loopback (DPDK @491.52) |
|---|---|---|
| frames | 46 | 250 |
| global occupancy | 0.293% | 0.314% (x1.07) |
| **max frame occupancy** | **1.35%** | **6.98%** |
| occupancy-spectrum Pearson r | — | **0.992** vs offline |
| full-width bar rows | 0.0% | 0.0% |
| edge occupancy | 0.0% | 0.067% |
| spike frames (> offline max) | — | 4 / 250 = 1.6% |

- The occupancy **spectrum** (which frequencies get detected) is nearly identical (r = 0.99): both
  legs light the same real signal region near 2.4 GHz. The model/geometry is sound.
- Neither leg produces full-width bars in the MASK (the live "yellow bar" is a RAW-spectrogram
  artifact, not a detection — consistent with the earlier finding that it never appears in the PSD).
- The ONE real difference: loopback has rare, dramatic **spike frames** — a corrupted ingest frame
  makes DINO-FT fire a large spurious off-band blob (frame 560: 6.98%, a solid blob across the low
  half of the band where there is no signal; frame 944: off-center blobs). ~1-2 genuinely-corrupt
  frames in 250, i.e. one spurious blob every few seconds at the live mask cadence -> reads as the
  intermittent "atrocious" look.
- This is with CLEAN uniform 1024-sample packets. The real radio streams mixed 1024/1008 native-DDC
  framing (~44-50% short packets), which drives MORE such corruptions -> live-on-radio is worse.

**Conclusion.** Do not touch the model or its downsample geometry.

## Root cause (confirmed 2026-09-14) + fix

Two "root fix" attempts in the CHDR converter were tried and REVERTED because they targeted the wrong
mechanism (the header's "torn frame = slot recycled mid-read" comment):
- **Detector copy-at-`compute()`** — copies data already recycled by dequeue time (deep input-queue lag);
  no effect, and the extra sync worsened lag.
- **Converter copy-at-queue (owned snapshot per batch)** — spikes UNCHANGED, and the extra 84 MB/batch
  copy on the ingest-critical path HALVED throughput (0.40 -> 0.20 Mpps) -> more drops.

By elimination (offline=no packets=clean; loopback=packets=spikes; owned-snapshot=still spikes) the real
cause is **dropped-packet fill corruption under GPU saturation**: at 491.52 MSps with viz+snipper+detector
the GB10 saturates (log: 131 "might get dropped", 311 "Fell behind", ~200-400k of 480k pps ingested), so
some batches are assembled from discontinuous/stale IQ -> a broadband transient -> a spurious off-band
blob. This is a PERF CEILING, not a converter logic bug; "lossless ingest at 491.52 full-load" is not
achievable by a code change.

**Fix (shipped): detector invalid-frame guard.** `finetuned_dino_detector` now suppresses (emits an empty
mask for) a frame whose occupancy is a gross outlier vs an adaptive per-channel baseline
(`invalid_frame_guard`=on, `invalid_frame_min_occupancy`=0.03, `invalid_frame_occupancy_k`=6). A
drop-corrupted frame is invalid, so the correct output is no detections. Verified on the same A/B:

| metric | offline | loopback pre-fix | loopback + guard |
|---|---|---|---|
| global occupancy | 0.293% | 0.314% | 0.292% |
| **max frame occupancy** | 1.35% | 6.98% | **1.55%** |
| occupancy-spectrum Pearson r | — | 0.992 | **0.996** |
| dramatic spikes (6-10%) | 0 | 4/250 | **0** (6 suppressed to empty) |

Guarded loopback is statistically indistinguishable from clean offline. Never fires offline (occ stays
far below the floor); tune `invalid_frame_min_occupancy` up for denser scenes (dense composites).

**Residual.** The guard clears the dramatic blobs but NOT subtle torn frames (thin-in-time /
wide-in-frequency streaks at ~1.4% whole-frame occupancy, just under the floor) — visible as faint brief
horizontal marks in the loopback raster (`results_guard/occupancy_raster.png`) but absent offline. An
occupancy threshold can't separate them from legitimate dense frames; a frequency-SPAN discriminator
(suppress frames whose detections span an anomalously wide band in few time rows) would catch them. Left
as a follow-up. The deeper cure is reducing drops (lighten the pipeline / lower rate) since the cause is
the saturation perf ceiling.

## Files
- `compare_offline_vs_loopback.py` — the analysis (aggregate metrics + plots).
- `results/summary.json`, `results/occupancy_spectrum.csv` — numbers.
- `results/occupancy_spectrum.png`, `results/occupancy_raster.png`,
  `results/offline_sample_frames.png`, `results/loopback_worst_frames.png` — figures.

## Reproduce
```bash
# offline leg
cd applications/usrp_wideband_signal_detection
python3 run_cuda_dino_offline_file.py ~/Documents/captures/x410_ota_2g4_gain10_20260908.sigmf-data \
  --detector cuda_dino_finetuned --config "$PWD/config_live_v3_dino_ft.yaml" \
  --output-root /tmp/usrp_spectrograms/offline_eval/cuda_dino_finetuned_rt/x410_ota_2g4_gain10_20260908
# pcap (once): python3 ../usrp_freq_detection/replay_rx_to_buff.py --sigmf-data <cap>.sigmf-data --out-pcap <...>.pcap
#   then fix dst-MAC for THIS host's DPDK port:  tcprewrite --enet-dmac=4c:bb:47:2c:45:13 --infile=<...>.pcap --outfile=<...>.spark.pcap
# loopback leg (network in loopback; NO sudo on the launcher)
RATE_HZ=491520000 PPS=480000 PCAP=x410_ota_2g4_gain10_20260908.spark.pcap \
  CONFIG_NAME=config_loopback_eval_dino_ft_rt.yaml ./bash_scripts/run_loopback_v3_demo.sh
# compare
cd infocom_evals/signal_detection_experiments/dino_ft_offline_vs_loopback
python3 compare_offline_vs_loopback.py --offline <offline out> --loopback /tmp/usrp_spectrograms/loopback_eval/dino_ft_rt/mask_arrays --out-dir ./results
```

## NOTE — loopback dst-MAC on the Spark
The composite pcaps carry the OLD x86 bench NIC MAC (`e0:9d:73:e0:5b:6b`). This Spark's DPDK data
port (`0000:01:00.0` = `enp1s0f0np0`) is `4c:bb:47:2c:45:13`, and the app runs the port in flow
isolation with promiscuous OFF, so frames with the wrong dst-MAC are dropped at L2 (rx_packets_phy
climbs, host/queue rx stays 0). Any loopback pcap on this host must be dst-MAC `4c:bb:47:2c:45:13`
(regenerate with `replay_rx_to_buff.py --dst-mac`, or rewrite with `tcprewrite --enet-dmac`).
