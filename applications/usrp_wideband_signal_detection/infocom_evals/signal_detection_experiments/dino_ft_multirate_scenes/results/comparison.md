# Detector comparison: M5S vs M6S_multirate

## Synthetic held-out scenes (120 scenes, 10 scenarios x 6 rate buckets, thr 0.6)

| metric | M5S | M6S_multirate |
|---|---|---|
| pixel precision | 0.930 | 0.925 |
| pixel recall | 0.227 | 0.671 |
| pixel IoU | 0.223 | 0.636 |
| region det@0.1 (headline) | 51.4% | 85.7% |
| region det@0.5 (headline) | 39.0% | 81.8% |
| scenario macro det@0.1 | 50.6% | 80.6% |
| scenario macro det@0.5 | 40.6% | 75.0% |
| false-alarm occupancy (noise_only) | 0.114% | 0.021% |

**by scenario** (det@0.1 / det@0.5, %)

| group | n | M5S | M6S_multirate |
|---|---|---|---|
| band_edge_focus | 111 | 50 / 42 | 78 / 70 |
| cellular_adjacent | 2503 | 29 / 15 | 97 / 96 |
| drone_ism | 2882 | 66 / 52 | 95 / 92 |
| hoppers_focus | 1137 | 57 / 43 | 89 / 82 |
| ism_collisions | 3032 | 54 / 44 | 67 / 61 |
| legacy_dense | 233 | 63 / 52 | 81 / 77 |
| legacy_sparse | 68 | 74 / 65 | 84 / 84 |
| wideband_fill | 49 | 12 / 12 | 53 / 37 |

**by sample-rate bucket** (det@0.1 / det@0.5, %)

| group | n | M5S | M6S_multirate |
|---|---|---|---|
| 10-25M | 1624 | 34 / 21 | 95 / 90 |
| 150-400M | 1112 | 77 / 69 | 90 / 84 |
| 25-60M | 1350 | 72 / 62 | 97 / 94 |
| 400-500M | 516 | 56 / 47 | 70 / 65 |
| 5-10M | 3350 | 20 / 10 | 72 / 68 |
| 60-150M | 2063 | 88 / 66 | 96 / 93 |

**by PSD SNR (dB above floor)** (det@0.1 / det@0.5, %)

| group | n | M5S | M6S_multirate |
|---|---|---|---|
| 0-6 | 2320 | 23 / 12 | 82 / 75 |
| 12-20 | 2065 | 82 / 61 | 96 / 95 |
| 6-12 | 1865 | 56 / 41 | 98 / 95 |
| <0 | 1285 | 12 / 7 | 33 / 22 |
| >=20 | 2480 | 69 / 60 | 98 / 97 |

**by in-band fraction (band-edge cut)** (det@0.1 / det@0.5, %)

| group | n | M5S | M6S_multirate |
|---|---|---|---|
| 0.5-0.9 (cut) | 1314 | 39 / 24 | 97 / 95 |
| <0.5 (mostly cut) | 2038 | 25 / 18 | 52 / 45 |
| >=0.9 (in band) | 6663 | 62 / 48 | 94 / 90 |

**by bandwidth (model bins)** (det@0.1 / det@0.5, %)

| group | n | M5S | M6S_multirate |
|---|---|---|---|
| 2-8 bins | 961 | 55 / 29 | 85 / 80 |
| 8-32 bins | 3272 | 38 / 27 | 96 / 92 |
| <2 bins | 284 | 43 / 32 | 55 / 48 |
| >=256 bins | 3487 | 43 / 32 | 72 / 68 |
| >=32 bins | 2011 | 86 / 77 | 97 / 94 |

**by duration (wide rows)** (det@0.1 / det@0.5, %)

| group | n | M5S | M6S_multirate |
|---|---|---|---|
| 3-16 rows | 4779 | 47 / 33 | 94 / 92 |
| <3 rows | 3019 | 45 / 35 | 68 / 62 |
| >=128 rows | 459 | 54 / 36 | 79 / 70 |
| >=16 rows | 1758 | 73 / 62 | 95 / 92 |

**by emitter behaviour** (det@0.1 / det@0.5, %)

| group | n | M5S | M6S_multirate |
|---|---|---|---|
| ble_adv | 20 | 70 / 30 | 90 / 70 |
| ble_data | 137 | 73 / 42 | 87 / 75 |
| bt_hop | 839 | 61 / 45 | 81 / 72 |
| drone_ctrl | 209 | 63 / 51 | 98 / 96 |
| drone_video | 609 | 57 / 44 | 100 / 100 |
| drone_video_fm | 17 | 100 / 76 | 100 / 100 |
| edge | 88 | 47 / 39 | 80 / 70 |
| fast_hop | 600 | 29 / 17 | 87 / 77 |
| fill | 42 | 0 / 0 | 45 / 26 |
| gsm | 2258 | 27 / 14 | 98 / 98 |
| iot | 77 | 57 / 26 | 78 / 65 |
| legacy | 347 | 66 / 56 | 82 / 78 |
| lib5g_dl | 19 | 42 / 42 | 100 / 84 |
| lib5g_dl_tdd | 45 | 20 / 0 | 100 / 98 |
| lib5g_ul | 25 | 4 / 0 | 100 / 96 |
| lmr | 28 | 50 / 50 | 64 / 57 |
| lte_dl | 34 | 76 / 44 | 94 / 94 |
| lte_dl_tdd | 12 | 92 / 25 | 100 / 100 |
| nb | 12 | 17 / 8 | 67 / 67 |
| nr_dl | 13 | 100 / 31 | 100 / 100 |
| nr_dl_tdd | 31 | 55 / 45 | 81 / 74 |
| nr_ul | 17 | 6 / 6 | 59 / 41 |
| oven | 60 | 100 / 100 | 100 / 100 |
| slow_hop | 33 | 55 / 27 | 64 / 45 |
| wide_hop | 422 | 98 / 80 | 100 / 99 |
| wifi | 2479 | 58 / 49 | 75 / 71 |
| wifi_ack | 1450 | 56 / 45 | 76 / 72 |
| zigbee | 76 | 99 / 80 | 100 / 97 |

**by class label** (det@0.1 / det@0.5, %)

| group | n | M5S | M6S_multirate |
|---|---|---|---|
| 16QAM | 57 | 65 / 53 | 86 / 79 |
| 4FSK | 18 | 72 / 33 | 83 / 78 |
| 5G_Downlink | 103 | 49 / 37 | 100 / 96 |
| 5G_Downlink_UL | 25 | 4 / 0 | 100 / 96 |
| 802_11ax | 3154 | 53 / 46 | 70 / 65 |
| 802_11g | 790 | 75 / 57 | 99 / 97 |
| BLE | 157 | 73 / 41 | 87 / 75 |
| BPSK | 64 | 61 / 47 | 80 / 77 |
| Bluetooth | 849 | 61 / 45 | 81 / 72 |
| Broadband_FM | 31 | 81 / 61 | 81 / 81 |
| CW | 23 | 48 / 39 | 65 / 57 |
| DroneCtrl | 209 | 63 / 51 | 98 / 96 |
| DroneVideo | 609 | 57 / 44 | 100 / 100 |
| GFSK | 151 | 54 / 31 | 77 / 66 |
| GSM | 2258 | 27 / 14 | 98 / 98 |
| LMR | 28 | 50 / 50 | 64 / 57 |
| LTE | 46 | 80 / 39 | 96 / 96 |
| LoRa | 41 | 63 / 29 | 85 / 61 |
| Microwave | 60 | 100 / 100 | 100 / 100 |
| NR | 44 | 68 / 41 | 86 / 82 |
| NR_UL | 17 | 6 / 6 | 59 / 41 |
| Narrowband_FM | 26 | 23 / 8 | 46 / 38 |
| NoiseBand | 41 | 24 / 24 | 49 / 34 |
| OFDM | 88 | 61 / 56 | 88 / 83 |
| QPSK | 1028 | 57 / 44 | 91 / 86 |
| Zigbee | 76 | 99 / 80 | 100 / 97 |

**Pixel recall by scenario** (precision in brackets)

| scenario | M5S | M6S_multirate |
|---|---|---|
| band_edge_focus | 0.30 (0.94) | 0.95 (0.96) |
| cellular_adjacent | 0.23 (0.98) | 0.99 (0.95) |
| drone_ism | 0.38 (0.95) | 0.98 (0.96) |
| hoppers_focus | 0.70 (0.91) | 0.91 (0.60) |
| ism_collisions | 0.50 (0.92) | 0.74 (0.87) |
| legacy_dense | 0.70 (0.95) | 0.95 (0.97) |
| legacy_sparse | 0.56 (0.98) | 0.98 (0.98) |
| noise_only | occ 0.114% | occ 0.021% |
| radar_pulses | 0.35 (0.80) | 0.82 (0.75) |
| wideband_fill | 0.00 (1.00) | 0.25 (1.00) |

## Real hard test cases (B210 880 MHz cellular/LMR @40 MSps, X310 LTE-FDD + NR 10 MHz @15.36 MSps; continuous labels)

| metric | M5S | M6S_multirate |
|---|---|---|
| pixel precision | 0.988 | 0.987 |
| pixel recall | 0.330 | 0.948 |
| pixel IoU | 0.328 | 0.937 |
| region det@0.1 | 65.2% | 75.8% |
| region det@0.5 | 24.2% | 74.2% |

**by label** (det@0.1 / det@0.5, %)

| group | n | M5S | M6S_multirate |
|---|---|---|---|
| CDMA | 16 | 100 / 69 | 100 / 100 |
| LMR | 8 | 0 / 0 | 0 / 0 |
| LTE | 32 | 78 / 16 | 100 / 100 |
| LTE FDD Generation | 1 | 100 / 0 | 100 / 100 |
| NR Generation | 1 | 100 / 0 | 100 / 0 |
| Unknown | 8 | 0 / 0 | 0 / 0 |

## ARGUS LTE-only OOD captures (245.76 MSps, +46 / +26 / +6 / -4 dB)

| metric | M5S | M6S_multirate |
|---|---|---|
| pixel precision | 0.781 | 0.625 |
| pixel recall | 0.472 | 0.482 |
| pixel IoU | 0.417 | 0.374 |
| region det@0.1 | 81.5% | 83.7% |
| region det@0.5 | 70.7% | 78.3% |

**by scene** (det@0.1 / det@0.5, %)

| group | n | M5S | M6S_multirate |
|---|---|---|---|
| argus_lte_m04dB | 28 | 43 / 21 | 46 / 29 |
| argus_lte_p06dB | 4 | 100 / 100 | 100 / 100 |
| argus_lte_p26dB | 28 | 96 / 86 | 100 / 100 |
| argus_lte_p46dB | 32 | 100 / 97 | 100 / 100 |
