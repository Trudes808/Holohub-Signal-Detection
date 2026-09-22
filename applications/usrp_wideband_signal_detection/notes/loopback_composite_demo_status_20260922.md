# Loopback-composite demo + dual polish — status & plan (2026-09-22)

## RESTORE POINT (roll back here if the night's work makes things worse)
- **Branch:** `live_demo`
- **Commit:** `d2e6fd04` — *viz: carry the persisted mask onto folded backlog frames (fix ROI ribbon)*
- **Tag:** `demo-known-good-20260922`
- **Rollback:** `git checkout demo-known-good-20260922`  (or `git reset --hard demo-known-good-20260922`)
- After rollback, rebuild the container app: `sudo ./bash_scripts/rebuild_demo_container_app.sh`

## What is known-good at the restore point (OTA X410, 2.4 GHz, 491.52 MSps)
Two visualization fixes committed this session, both verified live:
1. **`d33ad6c6`** — backpressure valve no longer *drops* backlog frames; it folds every backlog
   frame's rows into the history ring in arrival order. Killed the ~68% frame drop
   (`drop_render_busy` 30180→0, `drop_rate` 2.12→0). Single-channel scroll now continuous.
   New stat `backlog_folded` in the "Visualizer timing:" log; HUD "Dropped" stays honest.
2. **`d2e6fd04`** — folded frames now carry the most-recent detector mask (front-path mask
   persistence mirrored into the fold loop; `kMaskPersistMaxFrames` hoisted to compute scope).
   Fixes the broken/blank ROI ribbon on dual. Verified: `drop_rate=0`, masks backfilled≈received.

### Known remaining issue (the reason for tonight's work)
- **Dual ROI striping** is the *real detection cadence*, not a viz bug:
  - single M5-S: `finetuned_dino_detector.emit_stride: 1` (per-frame) → smooth.
  - dual   M5-S: `finetuned_dino_detector.emit_stride: 4` (every 4th frame) → detection updates
    only every 4th frame, held across the gap → periodic horizontal banding.
  - Root: M5-S single-pass ~10.4 ms fits the 21.3 ms frame period for ONE channel but not two.

## Plan for tonight (user-approved approaches, ~3 h, all on loopback so OTA is untouched)
1. **Loopback from the release_v2 composite** (`/home/genesys-dgx1/Documents/release_v2/composite/composite.sigmf-data`,
   ci16 @ 100 MSps, 1 ch, ~1.29 s, 100 MHz-wide baseband):
   - Resample ×4.9152 → 491.52 MSps.
   - Package to a CHDR/UDP pcap **reusing the existing pcap-gen tool** in
     `holoscan_waveform_generation/composition` (same tool that made `comprehensive_ordered.pcap`,
     `snr_single_*.pcap`). Make a ch1 copy via `tcprewrite --portmap=1234:1235`.
   - **Both channels replay the composite** (ch0 udp 1234, ch1 udp 1235 on one wire; DPDK flows split).
2. **Dual "look good" = decimate FFT rows** (chosen over slowing replay / stride tweaks):
   feed DINO **every** frame but at 1/4 row resolution (every 4th FFT row) so per-frame cost ≈ the
   old stride-4 cost, keeping real-time 491.52 **and** per-frame detection (no striping).
   Hard part: map the 1/4-height mask back onto the full-res detection overlay.
3. **Single↔dual toggle button** in the top-left DEMO CONTROLS panel:
   - Reuse the demo_control.json plumbing. Add a `channel_mode` field; the button writes it; the
     **conductor** relaunches the app with the single vs dual config (`self.args.dual`), exactly like
     the current detector switch (~15 s reconfigure). No new fast-path.

## Key facts / paths (so this is resumable)
- **Loopback mechanism:** tcpreplay of a CHDR pcap into sender NIC (`enp1s0f1np1`, mtu 9000).
  Reference: `bash_scripts/run_loopback_dual_dino_ft.sh` (dual = two tcpreplay streams, ports 1234/1235).
- **Existing pcaps:** `/home/genesys-dgx1/Documents/holoscan_waveform_generation/composition/composites/`
  (`comprehensive_ordered.pcap`, `snr_single_*.pcap`, `x410_ota_*.spark.pcap`, ...).
- **Conductor control:** `infocom_evals/pycodec_e2e/demo_conductor.py` watches `demo_control.json`
  (fields: `detector`, `color_model`, `snr`). Config maps: `CONFIG_BY_DETECTOR_LIVE`,
  `CONFIG_BY_DETECTOR_LIVE_DUAL`, `CONFIG_BY_DETECTOR_LOOPBACK`; picks dual via `self.args.dual`.
  Detector switch = app relaunch (~15 s); SNR switch = tcpreplay restart (~1 s).
- **Viz control/buttons:** `spectrogram_visualization.cu` — `demo_control_write()` (writes json),
  `demo_control_load_once()` (reads), ImGui buttons e.g. "Reset Max Hold" (~L3033), `draw_toggle_button`
  (~L1273). demo_control path set via `demo_control_json` param.
- **DINO detector:** section `finetuned_dino_detector:` in the M5-S configs; `emit_stride`, plus
  `dino_chunk_rows: 256`, `dino_chunk_cols: 512`, `live_emit_mask_rows: 16`, `live_emit_mask_cols: 20480`.
- **Display geometry:** viz `rows_per_frame: 8` (override `USRP_ROWS_PER_FRAME`), `display_time_rows`,
  `output_height/width`. History ring in `ChannelVisualizationState` (`history_grayscale/mask/class_id`,
  `history_row_frame_numbers/indices_within_frame`). Mask patched by frame_number in
  `patch_history_mask_for_frame()`; incoming 20480-wide mask reduced to display width first
  (`reduce_mask_to_history_rows[_gpu]`).
- **Configs:** single `config_live_v3_dino_ft_m5s.yaml`, dual `config_live_v3_dino_ft_m5s_two_channel.yaml`;
  loopback maps in the conductor. run wrapper `bash_scripts/run_live_demo.sh` (presets v3/v5/v5dual...).

## Verify (run wrappers, not hand docker exec)
- Rebuild after code/config change: `sudo ./bash_scripts/rebuild_demo_container_app.sh`
- Success signal in the app log: `Visualizer timing: ... drop_render_busy=0 ... drop_rate=0.00`
  with `backlog_folded` climbing and `masks backfilled ≈ received`.
