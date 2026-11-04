# Held-out scene evaluation: M6S_multirate (thr 0.6)

scenes 120, GT regions 19461, frames/scene <= 4

**Overall pixel**: precision 0.925 recall 0.671 IoU 0.636

**Pixel metrics by scenario**

| scenario | scenes | precision | recall | IoU | pred occ |
|---|---|---|---|---|---|
| band_edge_focus | 12 | 0.962 | 0.948 | 0.914 | 6.31% |
| cellular_adjacent | 12 | 0.952 | 0.989 | 0.942 | 54.55% |
| drone_ism | 12 | 0.961 | 0.978 | 0.941 | 49.15% |
| hoppers_focus | 12 | 0.598 | 0.908 | 0.564 | 3.71% |
| ism_collisions | 12 | 0.875 | 0.743 | 0.672 | 15.79% |
| legacy_dense | 12 | 0.968 | 0.948 | 0.919 | 8.48% |
| legacy_sparse | 12 | 0.981 | 0.977 | 0.958 | 4.88% |
| noise_only | 12 | 0.000 | 0.000 | 0.000 | 0.02% |
| radar_pulses | 12 | 0.746 | 0.820 | 0.641 | 21.63% |
| wideband_fill | 12 | 0.997 | 0.248 | 0.248 | 24.81% |

**False alarms on 12 annotation-free scenes**: predicted occupancy 0.021%

**Region detection by scenario (all GT boxes)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| band_edge_focus | 111 | 78.4% | 70.3% |
| cellular_adjacent | 2504 | 97.3% | 96.4% |
| drone_ism | 2882 | 95.1% | 92.0% |
| hoppers_focus | 1137 | 89.2% | 82.3% |
| ism_collisions | 3032 | 67.2% | 61.3% |
| legacy_dense | 236 | 80.1% | 75.8% |
| legacy_sparse | 68 | 83.8% | 83.8% |
| radar_pulses | 9442 | 44.6% | 35.2% |
| wideband_fill | 49 | 53.1% | 36.7% |

**Headline (excluding 9446 sub-pixel boxes and the radar_pulses scenario)**: 10015 regions, det@0.1 85.7% det@0.5 81.8%; scenario macro-average det@0.1 80.6% det@0.5 75.0%

**by scenario (excl. sub-pixel)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| band_edge_focus | 111 | 78.4% | 70.3% |
| cellular_adjacent | 2503 | 97.3% | 96.4% |
| drone_ism | 2882 | 95.1% | 92.0% |
| hoppers_focus | 1137 | 89.2% | 82.3% |
| ism_collisions | 3032 | 67.2% | 61.3% |
| legacy_dense | 233 | 81.1% | 76.8% |
| legacy_sparse | 68 | 83.8% | 83.8% |
| wideband_fill | 49 | 53.1% | 36.7% |

**by sample-rate bucket**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 10-25M | 1624 | 94.8% | 90.2% |
| 150-400M | 1112 | 89.6% | 83.8% |
| 25-60M | 1350 | 96.8% | 93.8% |
| 400-500M | 516 | 70.2% | 64.7% |
| 5-10M | 3350 | 71.8% | 68.0% |
| 60-150M | 2063 | 95.8% | 93.1% |

**by PSD SNR (dB above floor)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 0-6 | 2320 | 82.2% | 75.4% |
| 12-20 | 2065 | 96.3% | 95.1% |
| 6-12 | 1865 | 97.5% | 95.4% |
| <0 | 1285 | 33.5% | 22.4% |
| >=20 | 2480 | 98.4% | 97.3% |

**by in-band fraction (band-edge cut)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 0.5-0.9 (cut) | 1314 | 97.3% | 95.2% |
| <0.5 (mostly cut) | 2038 | 52.0% | 45.4% |
| >=0.9 (in band) | 6663 | 93.8% | 90.3% |

**by bandwidth (model bins)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 2-8 bins | 961 | 84.9% | 79.8% |
| 8-32 bins | 3272 | 95.6% | 92.2% |
| <2 bins | 284 | 55.3% | 47.5% |
| >=256 bins | 3487 | 72.5% | 68.1% |
| >=32 bins | 2011 | 97.3% | 94.4% |

**by duration (wide rows)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 3-16 rows | 4779 | 94.4% | 91.5% |
| <3 rows | 3019 | 67.9% | 62.2% |
| >=128 rows | 459 | 78.9% | 70.2% |
| >=16 rows | 1758 | 94.7% | 92.0% |

**by emitter behaviour**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| ble_adv | 20 | 90.0% | 70.0% |
| ble_data | 137 | 86.9% | 75.2% |
| bt_hop | 839 | 80.8% | 72.5% |
| drone_ctrl | 209 | 97.6% | 95.7% |
| drone_video | 609 | 100.0% | 99.8% |
| drone_video_fm | 17 | 100.0% | 100.0% |
| edge | 88 | 79.5% | 70.5% |
| fast_hop | 600 | 86.7% | 76.8% |
| fill | 42 | 45.2% | 26.2% |
| gsm | 2258 | 98.4% | 98.0% |
| iot | 77 | 77.9% | 64.9% |
| legacy | 347 | 81.8% | 78.4% |
| lib5g_dl | 19 | 100.0% | 84.2% |
| lib5g_dl_tdd | 45 | 100.0% | 97.8% |
| lib5g_ul | 25 | 100.0% | 96.0% |
| libofdm_dl | 4 | 100.0% | 100.0% |
| lmr | 28 | 64.3% | 57.1% |
| lte_dl | 34 | 94.1% | 94.1% |
| lte_dl_tdd | 12 | 100.0% | 100.0% |
| lte_ul | 5 | 80.0% | 40.0% |
| nb | 12 | 66.7% | 66.7% |
| nr_dl | 13 | 100.0% | 100.0% |
| nr_dl_tdd | 31 | 80.6% | 74.2% |
| nr_ul | 17 | 58.8% | 41.2% |
| on_fill | 7 | 100.0% | 100.0% |
| oven | 60 | 100.0% | 100.0% |
| slow_hop | 33 | 63.6% | 45.5% |
| wide_hop | 422 | 99.5% | 98.8% |
| wifi | 2479 | 75.1% | 71.0% |
| wifi_ack | 1450 | 76.3% | 71.7% |
| zigbee | 76 | 100.0% | 97.4% |

**by class label**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 16QAM | 57 | 86.0% | 78.9% |
| 4FSK | 18 | 83.3% | 77.8% |
| 5G_Downlink | 103 | 100.0% | 96.1% |
| 5G_Downlink_UL | 25 | 100.0% | 96.0% |
| 64QAM | 5 | 100.0% | 100.0% |
| 802_11ax | 3154 | 69.8% | 65.0% |
| 802_11g | 790 | 99.1% | 97.1% |
| 8PSK | 5 | 100.0% | 100.0% |
| BLE | 157 | 87.3% | 74.5% |
| BPSK | 64 | 79.7% | 76.6% |
| Bluetooth | 849 | 80.8% | 72.4% |
| Broadband_FM | 31 | 80.6% | 80.6% |
| CW | 23 | 65.2% | 56.5% |
| DroneCtrl | 209 | 97.6% | 95.7% |
| DroneVideo | 609 | 100.0% | 99.8% |
| FMCW | 7 | 100.0% | 100.0% |
| GFSK | 151 | 76.8% | 65.6% |
| GSM | 2258 | 98.4% | 98.0% |
| LMR | 28 | 64.3% | 57.1% |
| LTE | 46 | 95.7% | 95.7% |
| LTE_UL | 5 | 80.0% | 40.0% |
| LoRa | 41 | 85.4% | 61.0% |
| Microwave | 60 | 100.0% | 100.0% |
| NR | 44 | 86.4% | 81.8% |
| NR_UL | 17 | 58.8% | 41.2% |
| Narrowband_FM | 26 | 46.2% | 38.5% |
| NoiseBand | 41 | 48.8% | 34.1% |
| OFDM | 88 | 87.5% | 83.0% |
| QPSK | 1028 | 91.1% | 85.5% |
| Zigbee | 76 | 100.0% | 97.4% |
