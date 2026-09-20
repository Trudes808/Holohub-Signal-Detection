# Class-color mask overlay — live validation (2026-09-19)

The "Color Mask by Class" dashboard toggle colors the detection-mask overlay by the classifier's
predicted modulation class. Validated live on the network loopback (aligned 491.52 MSps capture,
classify-only AMC daemon: VT-CNN2 + ResNet1D + T-PRIME, gate T-PRIME).

Coloring is PER-SIGNAL REGION, separated in TIME and FREQUENCY: the daemon exports each decode's band
[f_lo,f_hi]; the viz colors a pixel by the decode whose band contains it, and FREEZES the class into a
parallel history ring at capture time so old waterfall rows keep their class (no whole-column recolor).
Palette (5 distinct hues, legible on the blue waterfall): PSK green, QAM magenta, FSK orange, OFDM cyan,
NOISE red; detected-but-unclassified pixels show a neutral gray. Commits: 477f52a1 (region+time), 53c96560 (palette).

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

## LIVE OTA validation (2026-09-20)
Ran the full pipeline on the real X410 at 2.4 GHz (GAIN 10, M3, classify-only daemon, class-color
overlay on): classifier produced 16 per-signal markers on real traffic (NOISE 9, OFDM 6, PSK 1 --
OFDM dominant = WiFi, physically correct), CHDR partial_drops=0, GPU ~77%. The WiFi channel at
~2.44-2.46 GHz shows cyan (OFDM); other live signals colored per class in time+frequency.
Images: live_ota_dashboard.png, live_ota_overlay_zoom.png.

NOTE (radio bring-up gotcha): after the loopback->OTA rewiring the two X410 SFP cables were SWAPPED
(host data NIC on X410 sfp1, control NIC on sfp0). Symptom: control/SSH/uhd_find_devices work, but
device open fails at RFNoC GSM init with "recv error on socket: Connection refused" and the data addr
(192.168.10.2) doesn't ping. Diagnosis: the device holding 10.2 (sfp0) has MAC ...f5, but the host
resolved 10.2 to ...f9 -> data NIC cabled to the wrong X410 port. check_radio_topology.sh does NOT
catch this (it only checks host IPs + control reachability). Fix = swap the two SFP cables.
