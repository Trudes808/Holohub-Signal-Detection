# Class-color mask overlay — live validation (2026-09-19)

The "Color Mask by Class" dashboard toggle colors the detection-mask overlay by the classifier's
predicted modulation class. Validated live on the network loopback (aligned 491.52 MSps capture,
classify-only AMC daemon: VT-CNN2 + ResNet1D + T-PRIME, gate T-PRIME).

Driver: `capture_overlay.sh` (env CONFIG_NAME / DAEMON / LABEL). Artifact: RSdrKitPhtsFgSsyX5K3Ca.

## Conditions (results/)
- `comparison.png` — the three conditions, same 2.36–2.47 GHz window.
- `dashboard_clean.png`, `dino_cc_2.png` — full DINO-FT M3 + classifier dashboard.
- `*_zoom.png` — per-condition overlay zooms.

| Condition | Detector | Classifier | Overlay | recent_decodes |
|---|---|---|---|---|
| class colors | DINO-FT M3 | on | multi-color (OFDM blue dominant), 0 lime | 16 (OFDM 5, PSK 1, NOISE 10) |
| fallback | DINO-FT M3 | off | uniform lime (no markers) | 0 |
| detector-agnostic | coherent_power | on | class-colored noisier masks | 16 (OFDM 8, FSK 1, NOISE 7) |

Real-time at full radio rate (tcpreplay 480k pps = 491.52 MSps), GPU ~82%, CHDR partial_drops = 0 in
all three. Overlay pixel counts (DINO+cls zoom): OFDM 1157, FSK 47, QAM 32, PSK 12, lime 0.

## Requires (why it lights up)
- classify-only daemon now emits a per-signal frequency-tagged class marker (rt_decode_daemon.py); the
  viz matches lit mask columns to the nearest marker by frequency (±4% of span). Off by default; the
  config param `renderer.class_colors_enable: true` starts it on for a headless run.
