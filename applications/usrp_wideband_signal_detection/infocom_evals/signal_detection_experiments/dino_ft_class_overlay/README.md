# Class-color mask overlay — live validation (2026-09-19)

The "Color Mask by Class" dashboard toggle colors the detection-mask overlay by the classifier's
predicted modulation class. Validated live on the network loopback (aligned 491.52 MSps capture,
classify-only AMC daemon: VT-CNN2 + ResNet1D + T-PRIME, gate T-PRIME).

Palette (5 distinct hues, legible on the blue waterfall): PSK green, QAM magenta, FSK orange, OFDM cyan,
NOISE red; detected-but-unclassified columns show a neutral gray (no lime-vs-green clash, no blue-on-blue).

Driver: `capture_overlay.sh` (env CONFIG_NAME / DAEMON / LABEL). Artifact: RSdrKitPhtsFgSsyX5K3Ca.

## Conditions (results/)
- `comparison.png` — the three conditions, same 2.36–2.47 GHz window.
- `dashboard_clean.png`, `dino_cc_2.png` — full DINO-FT M3 + classifier dashboard.
- `*_zoom.png` — per-condition overlay zooms.

| Condition | Detector | Classifier | Overlay | recent_decodes |
|---|---|---|---|---|
| class colors | DINO-FT M3 | on | multi-color, all 5 classes distinct | 16 (OFDM 5, NOISE 6, FSK 3, PSK 2) |
| fallback | DINO-FT M3 | off | uniform lime (no markers) | 0 |
| detector-agnostic | coherent_power | on | class-colored noisier masks | 16 (OFDM 8, NOISE 5, FSK 2, PSK 1) |

Real-time at full radio rate (tcpreplay 480k pps = 491.52 MSps), GPU ~82%, CHDR partial_drops = 0 in
all three. All five class hues render distinctly; cyan (OFDM) and red (NOISE) stand off the blue background.

## Requires (why it lights up)
- classify-only daemon now emits a per-signal frequency-tagged class marker (rt_decode_daemon.py); the
  viz matches lit mask columns to the nearest marker by frequency (±4% of span). Off by default; the
  config param `renderer.class_colors_enable: true` starts it on for a headless run.
