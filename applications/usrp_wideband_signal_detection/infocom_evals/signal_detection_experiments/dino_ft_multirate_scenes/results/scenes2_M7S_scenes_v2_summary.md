# Held-out scene evaluation: M7S_scenes_v2 (thr 0.6)

scenes 156, GT regions 25614, frames/scene <= 4

**Overall pixel**: precision 0.915 recall 0.728 IoU 0.682

**Pixel metrics by scenario**

| scenario | scenes | precision | recall | IoU | pred occ |
|---|---|---|---|---|---|
| band_edge_focus | 12 | 0.711 | 0.917 | 0.668 | 6.38% |
| broadcast_fm | 12 | 0.822 | 0.878 | 0.738 | 17.65% |
| cellular_adjacent | 12 | 0.947 | 0.979 | 0.928 | 41.12% |
| cellular_uplink | 12 | 0.873 | 0.866 | 0.769 | 13.85% |
| drone_ism | 12 | 0.967 | 0.943 | 0.914 | 44.82% |
| hoppers_focus | 12 | 0.938 | 0.931 | 0.877 | 5.43% |
| ism_collisions | 12 | 0.911 | 0.860 | 0.793 | 17.98% |
| legacy_dense | 12 | 0.947 | 0.925 | 0.880 | 7.19% |
| legacy_sparse | 12 | 0.849 | 0.977 | 0.833 | 8.40% |
| lmr_band | 12 | 0.806 | 0.688 | 0.590 | 1.94% |
| noise_only | 12 | 0.000 | 0.000 | 0.000 | 0.00% |
| radar_pulses | 12 | 0.796 | 0.692 | 0.588 | 25.98% |
| wideband_fill | 12 | 0.999 | 0.416 | 0.416 | 41.25% |

**False alarms on 13 annotation-free scenes**: predicted occupancy 0.004%

**Region detection by scenario (all GT boxes)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| band_edge_focus | 106 | 67.9% | 60.4% |
| broadcast_fm | 872 | 95.1% | 90.0% |
| cellular_adjacent | 1539 | 93.0% | 89.1% |
| cellular_uplink | 2262 | 76.7% | 70.1% |
| drone_ism | 2453 | 95.2% | 89.0% |
| hoppers_focus | 829 | 89.0% | 85.2% |
| ism_collisions | 1999 | 85.4% | 78.3% |
| legacy_dense | 233 | 77.7% | 70.0% |
| legacy_sparse | 69 | 84.1% | 81.2% |
| lmr_band | 457 | 64.1% | 56.9% |
| radar_pulses | 14739 | 83.7% | 69.7% |
| wideband_fill | 56 | 60.7% | 50.0% |

**Headline (excluding 14744 sub-pixel boxes and the radar_pulses scenario)**: 10870 regions, det@0.1 86.6% det@0.5 80.6%; scenario macro-average det@0.1 80.9% det@0.5 74.6%

**by scenario (excl. sub-pixel)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| band_edge_focus | 106 | 67.9% | 60.4% |
| broadcast_fm | 872 | 95.1% | 90.0% |
| cellular_adjacent | 1539 | 93.0% | 89.1% |
| cellular_uplink | 2261 | 76.8% | 70.1% |
| drone_ism | 2453 | 95.2% | 89.0% |
| hoppers_focus | 829 | 89.0% | 85.2% |
| ism_collisions | 1999 | 85.4% | 78.3% |
| legacy_dense | 230 | 78.3% | 70.4% |
| legacy_sparse | 68 | 83.8% | 80.9% |
| lmr_band | 457 | 64.1% | 56.9% |
| wideband_fill | 56 | 60.7% | 50.0% |

**by sample-rate bucket**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 10-25M | 1267 | 94.2% | 89.0% |
| 150-400M | 1518 | 71.6% | 65.2% |
| 25-60M | 1740 | 93.6% | 87.7% |
| 400-500M | 1689 | 72.4% | 62.7% |
| 5-10M | 2626 | 94.0% | 90.1% |
| 60-150M | 2030 | 89.3% | 83.6% |

**by PSD SNR (dB above floor)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 0-6 | 1670 | 78.5% | 70.5% |
| 12-20 | 1861 | 94.8% | 92.4% |
| 6-12 | 2336 | 89.3% | 84.5% |
| <0 | 1619 | 58.6% | 39.8% |
| >=20 | 3384 | 97.6% | 96.0% |

**by in-band fraction (band-edge cut)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 0.5-0.9 (cut) | 1095 | 93.4% | 86.8% |
| <0.5 (mostly cut) | 887 | 85.5% | 74.7% |
| >=0.9 (in band) | 8888 | 85.9% | 80.5% |

**by bandwidth (model bins)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 2-8 bins | 2370 | 86.1% | 81.0% |
| 8-32 bins | 1604 | 91.9% | 86.2% |
| <2 bins | 1306 | 56.7% | 49.3% |
| >=256 bins | 2469 | 94.6% | 87.4% |
| >=32 bins | 3121 | 90.5% | 85.3% |

**by duration (wide rows)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 3-16 rows | 4454 | 87.6% | 82.6% |
| <3 rows | 2211 | 88.5% | 81.5% |
| >=128 rows | 1812 | 82.1% | 75.7% |
| >=16 rows | 2393 | 86.5% | 80.0% |

**by emitter behaviour**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| ble_adv | 15 | 93.3% | 73.3% |
| ble_data | 80 | 87.5% | 80.0% |
| bt_hop | 852 | 82.3% | 75.5% |
| cdma_dl | 3 | 100.0% | 100.0% |
| drone_ctrl | 392 | 99.5% | 98.2% |
| drone_video | 499 | 100.0% | 100.0% |
| drone_video_fm | 15 | 100.0% | 100.0% |
| edge | 81 | 64.2% | 56.8% |
| fast_hop | 410 | 86.8% | 85.1% |
| fill | 42 | 52.4% | 38.1% |
| fm_station | 872 | 95.1% | 90.0% |
| gsm | 1081 | 99.8% | 98.0% |
| inband_interferer | 7 | 100.0% | 100.0% |
| iot | 65 | 86.2% | 72.3% |
| laa | 12 | 100.0% | 91.7% |
| legacy | 348 | 79.6% | 72.7% |
| lib5g_dl | 20 | 100.0% | 100.0% |
| lib5g_dl_tdd | 63 | 100.0% | 100.0% |
| lmr | 102 | 60.8% | 57.8% |
| lte_dl | 28 | 100.0% | 100.0% |
| lte_dl_tdd | 63 | 95.2% | 88.9% |
| nb | 5 | 80.0% | 80.0% |
| nbdata | 26 | 38.5% | 30.8% |
| nbfh | 143 | 53.8% | 45.5% |
| nbiot | 5 | 100.0% | 60.0% |
| nr_dl | 12 | 100.0% | 100.0% |
| nr_dl_tdd | 42 | 100.0% | 100.0% |
| on_fill | 14 | 85.7% | 85.7% |
| oven | 30 | 100.0% | 100.0% |
| pager | 42 | 76.2% | 61.9% |
| prach | 39 | 89.7% | 87.2% |
| ptt | 80 | 51.2% | 40.0% |
| pucch | 488 | 52.9% | 48.2% |
| pusch | 1793 | 81.3% | 73.7% |
| slow_hop | 3 | 0.0% | 0.0% |
| srs | 125 | 82.4% | 64.8% |
| tdma | 121 | 65.3% | 60.3% |
| wide_hop | 220 | 100.0% | 96.8% |
| wifi | 1645 | 90.8% | 82.0% |
| wifi_ack | 957 | 89.8% | 81.5% |
| zigbee | 30 | 93.3% | 83.3% |

**by class label**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 16QAM | 67 | 67.2% | 65.7% |
| 4FSK | 11 | 63.6% | 54.5% |
| 5G_Downlink | 117 | 98.3% | 97.4% |
| 64QAM | 8 | 100.0% | 87.5% |
| 802_11ax | 1605 | 87.4% | 77.9% |
| 802_11g | 1011 | 95.4% | 88.1% |
| 8PSK | 5 | 80.0% | 80.0% |
| BLE | 95 | 88.4% | 78.9% |
| BPSK | 83 | 72.3% | 63.9% |
| Bluetooth | 862 | 82.0% | 75.2% |
| Broadband_FM | 33 | 75.8% | 69.7% |
| Broadcast_FM | 872 | 95.1% | 90.0% |
| CDMA | 3 | 100.0% | 100.0% |
| CW | 10 | 90.0% | 90.0% |
| DMR | 123 | 65.9% | 60.2% |
| DroneCtrl | 392 | 99.5% | 98.2% |
| DroneVideo | 499 | 100.0% | 100.0% |
| FMCW | 5 | 100.0% | 80.0% |
| FSK | 89 | 57.3% | 49.4% |
| GFSK | 189 | 72.5% | 67.7% |
| GSM | 1081 | 99.8% | 98.0% |
| LAA | 12 | 100.0% | 91.7% |
| LMR | 22 | 54.5% | 54.5% |
| LTE | 91 | 96.7% | 92.3% |
| LTE-M | 3 | 100.0% | 100.0% |
| LTE_UL | 1230 | 68.2% | 61.1% |
| LoRa | 32 | 84.4% | 65.6% |
| Microwave | 30 | 100.0% | 100.0% |
| NB-IoT | 2 | 100.0% | 0.0% |
| NR | 54 | 100.0% | 100.0% |
| NR_UL | 1215 | 83.5% | 75.7% |
| NXDN | 29 | 51.7% | 51.7% |
| Narrowband_FM | 149 | 53.7% | 43.0% |
| NoiseBand | 36 | 47.2% | 36.1% |
| OFDM | 82 | 89.0% | 84.1% |
| P25 | 27 | 44.4% | 40.7% |
| Pager | 42 | 76.2% | 61.9% |
| QPSK | 592 | 92.9% | 90.0% |
| TETRA | 32 | 62.5% | 53.1% |
| Zigbee | 30 | 93.3% | 83.3% |
