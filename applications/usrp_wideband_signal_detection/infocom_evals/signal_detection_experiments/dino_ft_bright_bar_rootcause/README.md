# DINO-FT bright-bar root cause + fix (OTA, 2026-09-20)

Investigation of the intermittent **full-width bright yellow bar** in the raw spectrogram that also
pins the PSD **max-hold** line. Live OTA on the DGX Spark (GB10) + X410 @ 2.4 GHz, 491.52 MSps.

## TL;DR
The bright bar is a **uniform full-scale ("garbage") spectrogram frame** produced when the DINO-FT
GPU load intermittently oversubscribes the GB10, stalls the RX path (`Fell behind`), and a batch gets
assembled from **recycled DPDK mbuf memory** — a **full-count `10240/10240` batch with garbage
content** (NOT short packets, NOT partial, NOT a stale-buffer repeat). It is display-side only: the
detector's invalid-frame guard already zeros the corresponding mask.

Two independent things were also found and fixed:
1. The **invalid-frame guard was mis-calibrated** — its `0.03` occupancy floor sat *below* real busy
   2.4 GHz activity (4–15%), so it false-suppressed legitimate dense detections and deadlocked its
   cold-start baseline (it only updates on accepted frames).
2. The classic **short-packet** bright bar (`4032 B` truncated/status packets read past into garbage)
   was already fixed upstream on 2026-09-09 (`14b7ae67`, CHDR per-packet zero-fill) — **verified still
   working**: 1060+ short packets in the dual capture, zero bars from them.

## Evidence (see `logs/`)

### `single_dino_stride4_bars_captured.log` — the bar captured (config default at the time)
Full-rate viz sensor logged 7 broadband frames in 120 s (~1 / 17 s):
```
[bar_dbg] BROADBAND ch0 frame=698  mean=0.999 frac_hi=1.000 pkts=10240/10240 short=0 partial=0 fp=118e69e0...
[bar_dbg] BROADBAND ch0 frame=1346 mean=1.000 frac_hi=1.000 pkts=10240/10240 short=0 partial=0 fp=8a9edb35...
... (7 total, distinct fingerprints)
```
- `mean≈1.0, frac_hi=1.0` → every pixel maxed = uniform full-scale = garbage (real RF is never uniform
  across 500 MHz).
- `pkts=10240/10240 short=0 partial=0` → full-count batch, garbage content.
- 23 `Fell behind` events this run; bars cluster around them / high DPDK ring depth (mbuf pool ~90% used).
- Detector invalid-frame guard (floor already raised to 0.40 for this run) suppressions: **0**
  (was 10 false-positives at the old 0.03 floor — see below).

### `dual_dino_stride16_capture.log` — dual channel, the "target ingest"
- `Fell behind = 0`, `partial_drops = 0`, `panic_resets = 0`, full ~0.48 Mpps/ch. **No bars.**
- 1060+ short packets (`4032 B`) — all handled by the zero-fill; no bar.
- At the *old* guard floor 0.03: 10 false-suppressions on full-count 4–15% frames (legit dense 2.4 GHz).

### `single_dino_stride12_fixed.log` — the fix validated
`emit_stride: 4 -> 12`:
| metric | stride 4 | stride 12 |
|---|---|---|
| `Fell behind` | 23 | **0** |
| broadband garbage bars | 7 | **0** |
| guard false-suppressions | 10 (@0.03) | 0 (@0.40) |
| `partial_drops` / `panic_resets` | 0 / 0 | 0 / 0 |

Inference ~49 ms; stride 12 = ~3.9 Hz = ~19% GPU duty → the backlog (and thus the garbage) never
forms. The bar is eliminated **at the source**.

### `dual_dino_stride16_guardfix_confirm.log` — dual-channel, post-fix confirmation
Dual @ 2.4 + 1.0 GHz with the propagated fixes (guard floor 0.40, `emit_stride=16`):
- Detector guard false-suppressions: **0** (was 10 at floor 0.03).
- Ingest: `partial_drops=0`, `panic_resets=0`, only 2 startup `Fell behind` (steady state clean).
- The viz broadband-suppress guard **fired 2× on real garbage** — two startup ramp-up frames
  (`frac_hi=1.000` / `0.984`, full-count `10240/10240`) were skipped, so no bar reached the waterfall.
  Confirms the guard catches the unavoidable startup-transient garbage that `emit_stride` can't prevent.

## Fixes applied (live DINO configs + viz; eval/loopback configs deliberately untouched)
1. **`emit_stride: 4 -> 12`** in `config_live_v3_dino_ft.yaml` and `config_live_v3_dino_ft_sb.yaml`
   (dual is already 16). Root-cause fix: removes the GPU backlog that generates the garbage.
2. **Invalid-frame guard floor `0.03 -> 0.40`, `k 6 -> 2`** in the three live DINO configs
   (`config_live_v3_dino_ft.yaml`, `_two_channel`, `_sb`). Stops false-suppression of real dense
   frames and fixes the cold-start baseline deadlock; still catches broadband (~100%) garbage.
3. **Viz broadband-suppress guard** (`spectrogram_visualization.cu`, `SpectrogramPreviewOp`): a
   uniform-full-scale preview frame (>= `broadband_suppress_frac` near-saturated, default 0.85) is
   skipped entirely — no waterfall row, no PSD/max-hold/density update. Belt-and-suspenders for any
   residual transient; didn't need to fire at stride 12. `broadband_suppress_frac <= 0` disables it.
4. **"Reset Max Hold" button** (ImGui, Display Controls): clears the peak-hold trace on demand (global
   epoch consumed on the compute thread), for a manual clear if anything ever slips through.

## Diagnostics left in place (inert in production)
- Detector: `DINO_GUARD_DEBUG=1` env logs each invalid-frame-guard suppression with the ingest
  metadata (`pkts/expected`, `partial`, fingerprint vs last accepted). Read once at init; zero cost off.
- `run_live_demo.sh` forwards `-e DINO_GUARD_DEBUG`.

## How to reproduce
```
# capture (bars, before fix): temporarily set emit_stride: 4 in config_live_v3_dino_ft.yaml, then
sudo env DINO_GUARD_DEBUG=1 DURATION=120 GAIN=15 DISPLAY=:1 CHANNELS=0 FREQS=2400e6 DEST_PORTS=1234 \
  ./bash_scripts/run_live_demo.sh config_live_v3_dino_ft.yaml
# grep the container log /tmp/live_demo_app.log for 'Fell behind' and 'Skipping broadband-garbage'.
```
Note: an X410 `rx xport timed out getting a response from mgmt_portal` at streamer start is cleared by
`ssh root@192.168.21.2 systemctl restart usrp-hwd` (wait ~10 s for MPM), then re-run.
