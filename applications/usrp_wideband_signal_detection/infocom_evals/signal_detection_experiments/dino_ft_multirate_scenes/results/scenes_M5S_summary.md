# Held-out scene evaluation: M5S (thr 0.6)

scenes 120, GT regions 19461, frames/scene <= 4

**Overall pixel**: precision 0.930 recall 0.227 IoU 0.223

**Pixel metrics by scenario**

| scenario | scenes | precision | recall | IoU | pred occ |
|---|---|---|---|---|---|
| band_edge_focus | 12 | 0.941 | 0.302 | 0.296 | 2.05% |
| cellular_adjacent | 12 | 0.984 | 0.229 | 0.228 | 12.23% |
| drone_ism | 12 | 0.948 | 0.384 | 0.376 | 19.53% |
| hoppers_focus | 12 | 0.906 | 0.696 | 0.649 | 1.88% |
| ism_collisions | 12 | 0.924 | 0.498 | 0.478 | 10.01% |
| legacy_dense | 12 | 0.954 | 0.699 | 0.676 | 6.35% |
| legacy_sparse | 12 | 0.984 | 0.556 | 0.551 | 2.77% |
| noise_only | 12 | 0.000 | 0.000 | 0.000 | 0.11% |
| radar_pulses | 12 | 0.802 | 0.351 | 0.323 | 8.61% |
| wideband_fill | 12 | 1.000 | 0.000 | 0.000 | 0.05% |

**False alarms on 12 annotation-free scenes**: predicted occupancy 0.114%

**Region detection by scenario (all GT boxes)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| band_edge_focus | 111 | 50.5% | 42.3% |
| cellular_adjacent | 2504 | 28.7% | 14.9% |
| drone_ism | 2882 | 65.6% | 52.2% |
| hoppers_focus | 1137 | 57.2% | 43.0% |
| ism_collisions | 3032 | 53.7% | 43.5% |
| legacy_dense | 236 | 62.3% | 51.3% |
| legacy_sparse | 68 | 73.5% | 64.7% |
| radar_pulses | 9442 | 27.8% | 17.5% |
| wideband_fill | 49 | 12.2% | 12.2% |

**Headline (excluding 9446 sub-pixel boxes and the radar_pulses scenario)**: 10015 regions, det@0.1 51.4% det@0.5 39.0%; scenario macro-average det@0.1 50.6% det@0.5 40.6%

**by scenario (excl. sub-pixel)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| band_edge_focus | 111 | 50.5% | 42.3% |
| cellular_adjacent | 2503 | 28.7% | 14.9% |
| drone_ism | 2882 | 65.6% | 52.2% |
| hoppers_focus | 1137 | 57.2% | 43.0% |
| ism_collisions | 3032 | 53.7% | 43.5% |
| legacy_dense | 233 | 63.1% | 51.9% |
| legacy_sparse | 68 | 73.5% | 64.7% |
| wideband_fill | 49 | 12.2% | 12.2% |

**by sample-rate bucket**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 10-25M | 1624 | 33.7% | 21.0% |
| 150-400M | 1112 | 76.9% | 68.8% |
| 25-60M | 1350 | 71.8% | 62.4% |
| 400-500M | 516 | 56.0% | 46.7% |
| 5-10M | 3350 | 19.9% | 10.3% |
| 60-150M | 2063 | 88.2% | 66.4% |

**by PSD SNR (dB above floor)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 0-6 | 2320 | 23.1% | 12.5% |
| 12-20 | 2065 | 81.7% | 61.4% |
| 6-12 | 1865 | 56.1% | 41.0% |
| <0 | 1285 | 12.0% | 6.7% |
| >=20 | 2480 | 69.4% | 60.3% |

**by in-band fraction (band-edge cut)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 0.5-0.9 (cut) | 1314 | 38.5% | 24.4% |
| <0.5 (mostly cut) | 2038 | 25.2% | 17.7% |
| >=0.9 (in band) | 6663 | 61.9% | 48.3% |

**by bandwidth (model bins)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 2-8 bins | 961 | 55.4% | 28.7% |
| 8-32 bins | 3272 | 38.1% | 27.4% |
| <2 bins | 284 | 43.3% | 31.7% |
| >=256 bins | 3487 | 43.1% | 31.5% |
| >=32 bins | 2011 | 86.5% | 76.6% |

**by duration (wide rows)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 3-16 rows | 4779 | 47.4% | 33.0% |
| <3 rows | 3019 | 44.8% | 35.3% |
| >=128 rows | 459 | 53.8% | 36.4% |
| >=16 rows | 1758 | 72.8% | 62.3% |

**by emitter behaviour**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| ble_adv | 20 | 70.0% | 30.0% |
| ble_data | 137 | 73.0% | 42.3% |
| bt_hop | 839 | 61.3% | 44.6% |
| drone_ctrl | 209 | 62.7% | 50.7% |
| drone_video | 609 | 56.7% | 44.2% |
| drone_video_fm | 17 | 100.0% | 76.5% |
| edge | 88 | 46.6% | 38.6% |
| fast_hop | 600 | 28.8% | 17.2% |
| fill | 42 | 0.0% | 0.0% |
| gsm | 2258 | 27.1% | 13.9% |
| iot | 77 | 57.1% | 26.0% |
| legacy | 347 | 66.3% | 55.9% |
| lib5g_dl | 19 | 42.1% | 42.1% |
| lib5g_dl_tdd | 45 | 20.0% | 0.0% |
| lib5g_ul | 25 | 4.0% | 0.0% |
| libofdm_dl | 4 | 100.0% | 0.0% |
| lmr | 28 | 50.0% | 50.0% |
| lte_dl | 34 | 76.5% | 44.1% |
| lte_dl_tdd | 12 | 91.7% | 25.0% |
| lte_ul | 5 | 0.0% | 0.0% |
| nb | 12 | 16.7% | 8.3% |
| nr_dl | 13 | 100.0% | 30.8% |
| nr_dl_tdd | 31 | 54.8% | 45.2% |
| nr_ul | 17 | 5.9% | 5.9% |
| on_fill | 7 | 85.7% | 85.7% |
| oven | 60 | 100.0% | 100.0% |
| slow_hop | 33 | 54.5% | 27.3% |
| wide_hop | 422 | 97.6% | 80.3% |
| wifi | 2479 | 57.9% | 49.3% |
| wifi_ack | 1450 | 55.9% | 45.2% |
| zigbee | 76 | 98.7% | 80.3% |

**by class label**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 16QAM | 57 | 64.9% | 52.6% |
| 4FSK | 18 | 72.2% | 33.3% |
| 5G_Downlink | 103 | 48.5% | 36.9% |
| 5G_Downlink_UL | 25 | 4.0% | 0.0% |
| 64QAM | 5 | 80.0% | 80.0% |
| 802_11ax | 3154 | 52.9% | 45.6% |
| 802_11g | 790 | 74.8% | 57.0% |
| 8PSK | 5 | 60.0% | 20.0% |
| BLE | 157 | 72.6% | 40.8% |
| BPSK | 64 | 60.9% | 46.9% |
| Bluetooth | 849 | 61.1% | 44.6% |
| Broadband_FM | 31 | 80.6% | 61.3% |
| CW | 23 | 47.8% | 39.1% |
| DroneCtrl | 209 | 62.7% | 50.7% |
| DroneVideo | 609 | 56.7% | 44.2% |
| FMCW | 7 | 85.7% | 42.9% |
| GFSK | 151 | 53.6% | 31.1% |
| GSM | 2258 | 27.1% | 13.9% |
| LMR | 28 | 50.0% | 50.0% |
| LTE | 46 | 80.4% | 39.1% |
| LTE_UL | 5 | 0.0% | 0.0% |
| LoRa | 41 | 63.4% | 29.3% |
| Microwave | 60 | 100.0% | 100.0% |
| NR | 44 | 68.2% | 40.9% |
| NR_UL | 17 | 5.9% | 5.9% |
| Narrowband_FM | 26 | 23.1% | 7.7% |
| NoiseBand | 41 | 24.4% | 24.4% |
| OFDM | 88 | 61.4% | 55.7% |
| QPSK | 1028 | 56.6% | 44.0% |
| Zigbee | 76 | 98.7% | 80.3% |
