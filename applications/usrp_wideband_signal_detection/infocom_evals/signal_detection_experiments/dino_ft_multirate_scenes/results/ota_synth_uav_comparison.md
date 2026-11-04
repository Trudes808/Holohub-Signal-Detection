# Detector comparison: M5S vs M6S vs M7S

## Synthetic held-out scenes (120 scenes, 10 scenarios x 6 rate buckets, thr 0.6)

| metric | M5S | M6S | M7S |
|---|---|---|---|
| pixel precision | 0.751 | 0.748 | 0.797 |
| pixel recall | 0.314 | 0.920 | 0.937 |
| pixel IoU | 0.284 | 0.703 | 0.756 |
| region det@0.1 (headline) | 63.6% | 97.5% | 97.1% |
| region det@0.5 (headline) | 42.1% | 94.5% | 94.7% |
| scenario macro det@0.1 | 63.6% | 97.5% | 97.1% |
| scenario macro det@0.5 | 42.1% | 94.5% | 94.7% |
| false-alarm occupancy (noise_only) | 0.000% | 0.000% | 0.000% |

**by scenario** (det@0.1 / det@0.5, %)

| group | n | M5S | M6S | M7S |
|---|---|---|---|---|
| real_plus_synth_uav | 970 | 64 / 42 | 98 / 95 | 97 / 95 |

**by sample-rate bucket** (det@0.1 / det@0.5, %)

| group | n | M5S | M6S | M7S |
|---|---|---|---|---|
| 10-25M | 316 | 38 / 2 | 99 / 93 | 99 / 97 |
| 150-400M | 116 | 100 / 99 | 100 / 100 | 100 / 100 |
| 25-60M | 18 | 94 / 39 | 100 / 100 | 100 / 100 |
| 60-150M | 520 | 70 / 54 | 96 / 94 | 95 / 92 |

**by PSD SNR (dB above floor)** (det@0.1 / det@0.5, %)

| group | n | M5S | M6S | M7S |
|---|---|---|---|---|
| 0-6 | 138 | 14 / 1 | 89 / 78 | 88 / 80 |
| 12-20 | 294 | 69 / 22 | 100 / 99 | 98 / 98 |
| 6-12 | 168 | 40 / 15 | 97 / 95 | 98 / 92 |
| <0 | 43 | 5 / 0 | 93 / 81 | 98 / 93 |
| >=20 | 327 | 99 / 97 | 100 / 100 | 100 / 100 |

**by in-band fraction (band-edge cut)** (det@0.1 / det@0.5, %)

| group | n | M5S | M6S | M7S |
|---|---|---|---|---|
| 0.5-0.9 (cut) | 108 | 16 / 0 | 96 / 83 | 96 / 92 |
| >=0.9 (in band) | 858 | 70 / 48 | 98 / 96 | 97 / 95 |

**by bandwidth (model bins)** (det@0.1 / det@0.5, %)

| group | n | M5S | M6S | M7S |
|---|---|---|---|---|
| 8-32 bins | 132 | 62 / 49 | 89 / 83 | 85 / 80 |
| >=256 bins | 367 | 52 / 7 | 100 / 96 | 99 / 98 |
| >=32 bins | 471 | 73 / 68 | 98 / 96 | 99 / 96 |

**by duration (wide rows)** (det@0.1 / det@0.5, %)

| group | n | M5S | M6S | M7S |
|---|---|---|---|---|
| 3-16 rows | 268 | 57 / 32 | 94 / 90 | 95 / 92 |
| <3 rows | 30 | 40 / 37 | 93 / 90 | 77 / 73 |
| >=16 rows | 666 | 67 / 47 | 99 / 97 | 99 / 97 |

**by emitter behaviour** (det@0.1 / det@0.5, %)

| group | n | M5S | M6S | M7S |
|---|---|---|---|---|
| drone_ctrl | 189 | 48 / 35 | 90 / 84 | 89 / 85 |
| drone_video | 775 | 67 / 44 | 99 / 97 | 99 / 97 |

**by class label** (det@0.1 / det@0.5, %)

| group | n | M5S | M6S | M7S |
|---|---|---|---|---|
| DroneCtrl | 189 | 48 / 35 | 90 / 84 | 89 / 85 |
| DroneVideo | 775 | 67 / 44 | 99 / 97 | 99 / 97 |

**Pixel recall by scenario** (precision in brackets)

| scenario | M5S | M6S | M7S |
|---|---|---|---|
| real_plus_synth_uav | 0.31 (0.75) | 0.92 (0.75) | 0.94 (0.80) |
