# Held-out scene evaluation: M12S_v6q2 (thr 0.6)

scenes 156, GT regions 31365, frames/scene <= 4

**Overall pixel**: precision 0.946 recall 0.885 IoU 0.842

**Pixel metrics by scenario**

| scenario | scenes | precision | recall | IoU | pred occ |
|---|---|---|---|---|---|
| band_edge_focus | 12 | 0.664 | 0.766 | 0.552 | 5.62% |
| broadcast_fm | 12 | 0.885 | 0.771 | 0.700 | 16.32% |
| cellular_adjacent | 12 | 0.967 | 0.906 | 0.879 | 40.91% |
| cellular_uplink | 12 | 0.900 | 0.576 | 0.541 | 8.34% |
| drone_ism | 12 | 0.989 | 0.960 | 0.950 | 56.16% |
| hoppers_focus | 12 | 0.889 | 0.857 | 0.774 | 5.07% |
| ism_collisions | 12 | 0.942 | 0.915 | 0.866 | 19.50% |
| legacy_dense | 12 | 0.938 | 0.850 | 0.805 | 7.01% |
| legacy_sparse | 12 | 0.915 | 0.920 | 0.848 | 13.12% |
| lmr_band | 12 | 0.804 | 0.765 | 0.645 | 2.46% |
| noise_only | 12 | 0.742 | 0.442 | 0.383 | 3.55% |
| radar_pulses | 12 | 0.875 | 0.815 | 0.730 | 29.61% |
| wideband_fill | 12 | 0.994 | 0.966 | 0.960 | 73.87% |

**False alarms on 1 annotation-free scenes**: predicted occupancy 0.247%

**Region detection by scenario (all GT boxes)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| band_edge_focus | 147 | 74.1% | 61.9% |
| broadcast_fm | 1079 | 91.9% | 81.9% |
| cellular_adjacent | 1143 | 80.0% | 73.9% |
| cellular_uplink | 2078 | 71.7% | 60.8% |
| drone_ism | 2437 | 89.6% | 83.4% |
| hoppers_focus | 1818 | 88.7% | 83.4% |
| ism_collisions | 2993 | 88.7% | 81.7% |
| legacy_dense | 421 | 72.0% | 60.6% |
| legacy_sparse | 475 | 89.5% | 75.2% |
| lmr_band | 1093 | 81.6% | 68.0% |
| noise_only | 1121 | 56.1% | 42.8% |
| radar_pulses | 16493 | 93.8% | 80.9% |
| wideband_fill | 67 | 100.0% | 100.0% |

**Headline (excluding 16502 sub-pixel boxes and the radar_pulses scenario)**: 14863 regions, det@0.1 82.5% det@0.5 73.9%; scenario macro-average det@0.1 82.0% det@0.5 72.8%

**by scenario (excl. sub-pixel)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| band_edge_focus | 147 | 74.1% | 61.9% |
| broadcast_fm | 1079 | 91.9% | 81.9% |
| cellular_adjacent | 1143 | 80.0% | 73.9% |
| cellular_uplink | 2078 | 71.7% | 60.8% |
| drone_ism | 2436 | 89.7% | 83.5% |
| hoppers_focus | 1817 | 88.7% | 83.4% |
| ism_collisions | 2993 | 88.7% | 81.7% |
| legacy_dense | 418 | 72.2% | 60.8% |
| legacy_sparse | 474 | 89.5% | 75.1% |
| lmr_band | 1090 | 81.8% | 68.2% |
| noise_only | 1121 | 56.1% | 42.8% |
| wideband_fill | 67 | 100.0% | 100.0% |

**by sample-rate bucket**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 10-25M | 2399 | 96.4% | 92.7% |
| 150-400M | 2877 | 67.7% | 55.3% |
| 25-60M | 2270 | 90.7% | 83.3% |
| 400-500M | 2095 | 70.1% | 58.3% |
| 5-10M | 2422 | 87.3% | 76.5% |
| 60-150M | 2800 | 84.5% | 78.4% |

**by PSD SNR (dB above floor)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 0-6 | 3113 | 73.7% | 57.8% |
| 12-20 | 3023 | 89.9% | 85.3% |
| 6-12 | 3114 | 84.5% | 75.0% |
| <0 | 1582 | 49.1% | 32.7% |
| >=20 | 4031 | 95.5% | 92.9% |

**by in-band fraction (band-edge cut)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 0.5-0.9 (cut) | 1223 | 91.3% | 82.8% |
| <0.5 (mostly cut) | 941 | 87.2% | 82.6% |
| >=0.9 (in band) | 12699 | 81.4% | 72.3% |

**by bandwidth (model bins)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 2-8 bins | 2504 | 82.0% | 73.9% |
| 8-32 bins | 2244 | 93.2% | 86.6% |
| <2 bins | 4218 | 65.1% | 52.1% |
| >=256 bins | 2957 | 95.7% | 90.6% |
| >=32 bins | 2940 | 86.7% | 78.5% |

**by duration (wide rows)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 3-16 rows | 3647 | 80.7% | 72.7% |
| <3 rows | 2730 | 89.7% | 84.1% |
| >=128 rows | 4359 | 76.8% | 64.5% |
| >=16 rows | 4127 | 85.5% | 77.9% |

**by emitter behaviour**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| ble_adv | 9 | 77.8% | 44.4% |
| ble_data | 75 | 86.7% | 77.3% |
| bt_hop | 897 | 79.4% | 68.7% |
| drone_ctrl | 179 | 96.6% | 93.9% |
| drone_video | 396 | 99.7% | 99.7% |
| drone_video_fm | 27 | 100.0% | 100.0% |
| edge | 83 | 71.1% | 57.8% |
| fast_hop | 121 | 84.3% | 84.3% |
| fhss_net | 2385 | 86.6% | 77.0% |
| fill | 42 | 100.0% | 100.0% |
| fm_station | 880 | 96.6% | 87.3% |
| gsm | 535 | 94.4% | 89.9% |
| inband_interferer | 7 | 100.0% | 100.0% |
| iot | 55 | 81.8% | 45.5% |
| laa | 12 | 100.0% | 91.7% |
| legacy | 346 | 79.5% | 73.1% |
| lib5g_dl | 13 | 100.0% | 100.0% |
| lib5g_dl_tdd | 72 | 98.6% | 98.6% |
| lmr | 91 | 61.5% | 52.7% |
| lte_dl | 25 | 100.0% | 100.0% |
| lte_dl_tdd | 45 | 93.3% | 84.4% |
| nb | 7 | 85.7% | 85.7% |
| nbdata | 31 | 54.8% | 45.2% |
| nbfh | 157 | 68.2% | 56.7% |
| nbiot | 10 | 80.0% | 60.0% |
| nr_dl | 24 | 100.0% | 100.0% |
| nr_dl_tdd | 87 | 100.0% | 100.0% |
| on_fill | 10 | 100.0% | 100.0% |
| oven | 63 | 100.0% | 100.0% |
| pager | 23 | 95.7% | 95.7% |
| prach | 28 | 60.7% | 53.6% |
| ptt | 93 | 65.6% | 60.2% |
| pucch | 517 | 54.7% | 50.1% |
| pusch | 1635 | 72.5% | 59.0% |
| quiet_line | 14 | 57.1% | 42.9% |
| radar | 79 | 38.0% | 34.2% |
| rx_dc | 251 | 78.1% | 69.3% |
| rx_spur | 48 | 16.7% | 16.7% |
| slow_hop | 12 | 8.3% | 0.0% |
| spur_comb | 1931 | 67.9% | 53.0% |
| srs | 104 | 75.0% | 65.4% |
| tdma | 40 | 32.5% | 32.5% |
| wide_hop | 56 | 94.6% | 94.6% |
| wifi | 2092 | 93.9% | 89.1% |
| wifi_ack | 1244 | 92.9% | 86.8% |
| zigbee | 12 | 91.7% | 75.0% |

**by class label**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 16QAM | 65 | 81.5% | 75.4% |
| 4FSK | 16 | 75.0% | 62.5% |
| 5G_Downlink | 115 | 98.3% | 97.4% |
| 64QAM | 9 | 77.8% | 77.8% |
| 802_11ax | 2459 | 92.6% | 87.4% |
| 802_11g | 891 | 96.3% | 90.5% |
| 8PSK | 14 | 71.4% | 57.1% |
| BLE | 84 | 85.7% | 73.8% |
| BPSK | 83 | 73.5% | 66.3% |
| Bluetooth | 909 | 79.0% | 68.3% |
| Broadband_FM | 45 | 86.7% | 80.0% |
| Broadcast_FM | 880 | 96.6% | 87.3% |
| CW | 26 | 50.0% | 42.3% |
| DC_spur | 251 | 78.1% | 69.3% |
| DMR | 57 | 52.6% | 52.6% |
| DroneCtrl | 179 | 96.6% | 93.9% |
| DroneVideo | 396 | 99.7% | 99.7% |
| FMCW | 7 | 85.7% | 71.4% |
| FSK | 77 | 64.9% | 45.5% |
| GFSK | 2497 | 85.9% | 76.3% |
| GSM | 535 | 94.4% | 89.9% |
| LAA | 12 | 100.0% | 91.7% |
| LMR | 29 | 48.3% | 48.3% |
| LTE | 70 | 95.7% | 90.0% |
| LTE-M | 6 | 66.7% | 50.0% |
| LTE_UL | 1079 | 62.7% | 53.7% |
| LoRa | 34 | 85.3% | 47.1% |
| Microwave | 63 | 100.0% | 100.0% |
| NB-IoT | 4 | 100.0% | 75.0% |
| NR | 111 | 100.0% | 100.0% |
| NR_UL | 1205 | 73.5% | 60.4% |
| NXDN | 42 | 57.1% | 52.4% |
| Narrowband_FM | 153 | 65.4% | 56.9% |
| NoiseBand | 21 | 90.5% | 90.5% |
| OFDM | 98 | 92.9% | 89.8% |
| P25 | 27 | 74.1% | 55.6% |
| Pager | 23 | 95.7% | 95.7% |
| QPSK | 194 | 88.7% | 85.6% |
| RX_spur | 1979 | 66.6% | 52.1% |
| RadarCW | 79 | 38.0% | 34.2% |
| TETRA | 27 | 48.1% | 48.1% |
| Zigbee | 12 | 91.7% | 75.0% |
