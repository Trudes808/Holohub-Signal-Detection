# Held-out scene evaluation: M11S_v6 (thr 0.6)

scenes 156, GT regions 31365, frames/scene <= 4

**Overall pixel**: precision 0.939 recall 0.889 IoU 0.841

**Pixel metrics by scenario**

| scenario | scenes | precision | recall | IoU | pred occ |
|---|---|---|---|---|---|
| band_edge_focus | 12 | 0.652 | 0.776 | 0.549 | 5.80% |
| broadcast_fm | 12 | 0.854 | 0.768 | 0.679 | 16.83% |
| cellular_adjacent | 12 | 0.972 | 0.885 | 0.863 | 39.78% |
| cellular_uplink | 12 | 0.893 | 0.581 | 0.543 | 8.47% |
| drone_ism | 12 | 0.990 | 0.959 | 0.949 | 56.05% |
| hoppers_focus | 12 | 0.874 | 0.867 | 0.770 | 5.22% |
| ism_collisions | 12 | 0.940 | 0.919 | 0.868 | 19.60% |
| legacy_dense | 12 | 0.930 | 0.857 | 0.805 | 7.12% |
| legacy_sparse | 12 | 0.924 | 0.925 | 0.859 | 13.08% |
| lmr_band | 12 | 0.798 | 0.773 | 0.647 | 2.51% |
| noise_only | 12 | 0.711 | 0.461 | 0.388 | 3.87% |
| radar_pulses | 12 | 0.855 | 0.888 | 0.772 | 33.04% |
| wideband_fill | 12 | 0.993 | 0.958 | 0.952 | 73.40% |

**False alarms on 1 annotation-free scenes**: predicted occupancy 0.206%

**Region detection by scenario (all GT boxes)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| band_edge_focus | 147 | 74.8% | 62.6% |
| broadcast_fm | 1079 | 93.1% | 84.5% |
| cellular_adjacent | 1143 | 79.5% | 73.6% |
| cellular_uplink | 2078 | 71.5% | 61.3% |
| drone_ism | 2437 | 89.0% | 83.3% |
| hoppers_focus | 1818 | 88.8% | 83.9% |
| ism_collisions | 2993 | 88.6% | 82.4% |
| legacy_dense | 421 | 75.3% | 61.3% |
| legacy_sparse | 475 | 92.0% | 74.3% |
| lmr_band | 1093 | 82.2% | 71.8% |
| noise_only | 1121 | 57.6% | 44.4% |
| radar_pulses | 16493 | 96.5% | 90.0% |
| wideband_fill | 67 | 100.0% | 98.5% |

**Headline (excluding 16502 sub-pixel boxes and the radar_pulses scenario)**: 14863 regions, det@0.1 82.8% det@0.5 74.7%; scenario macro-average det@0.1 82.8% det@0.5 73.5%

**by scenario (excl. sub-pixel)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| band_edge_focus | 147 | 74.8% | 62.6% |
| broadcast_fm | 1079 | 93.1% | 84.5% |
| cellular_adjacent | 1143 | 79.5% | 73.6% |
| cellular_uplink | 2078 | 71.5% | 61.3% |
| drone_ism | 2436 | 89.1% | 83.3% |
| hoppers_focus | 1817 | 88.9% | 84.0% |
| ism_collisions | 2993 | 88.6% | 82.4% |
| legacy_dense | 418 | 75.8% | 61.7% |
| legacy_sparse | 474 | 92.0% | 74.3% |
| lmr_band | 1090 | 82.3% | 71.9% |
| noise_only | 1121 | 57.6% | 44.4% |
| wideband_fill | 67 | 100.0% | 98.5% |

**by sample-rate bucket**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 10-25M | 2399 | 96.4% | 92.7% |
| 150-400M | 2877 | 68.6% | 56.8% |
| 25-60M | 2270 | 90.5% | 84.1% |
| 400-500M | 2095 | 70.9% | 58.5% |
| 5-10M | 2422 | 87.4% | 77.8% |
| 60-150M | 2800 | 84.4% | 79.2% |

**by PSD SNR (dB above floor)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 0-6 | 3113 | 74.3% | 59.0% |
| 12-20 | 3023 | 90.3% | 86.0% |
| 6-12 | 3114 | 85.2% | 76.4% |
| <0 | 1582 | 49.1% | 33.6% |
| >=20 | 4031 | 95.2% | 93.0% |

**by in-band fraction (band-edge cut)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 0.5-0.9 (cut) | 1223 | 90.8% | 83.7% |
| <0.5 (mostly cut) | 941 | 87.2% | 83.2% |
| >=0.9 (in band) | 12699 | 81.7% | 73.2% |

**by bandwidth (model bins)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 2-8 bins | 2504 | 82.5% | 74.8% |
| 8-32 bins | 2244 | 92.6% | 87.1% |
| <2 bins | 4218 | 66.4% | 53.6% |
| >=256 bins | 2957 | 95.5% | 90.8% |
| >=32 bins | 2940 | 86.5% | 79.1% |

**by duration (wide rows)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 3-16 rows | 3647 | 80.3% | 73.1% |
| <3 rows | 2730 | 89.7% | 84.9% |
| >=128 rows | 4359 | 78.2% | 66.0% |
| >=16 rows | 4127 | 85.4% | 78.4% |

**by emitter behaviour**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| ble_adv | 9 | 77.8% | 44.4% |
| ble_data | 75 | 88.0% | 78.7% |
| bt_hop | 897 | 79.2% | 67.7% |
| drone_ctrl | 179 | 96.6% | 93.9% |
| drone_video | 396 | 99.7% | 99.7% |
| drone_video_fm | 27 | 100.0% | 100.0% |
| edge | 83 | 72.3% | 59.0% |
| fast_hop | 121 | 84.3% | 84.3% |
| fhss_net | 2385 | 86.5% | 78.5% |
| fill | 42 | 100.0% | 97.6% |
| fm_station | 880 | 96.9% | 88.8% |
| gsm | 535 | 91.0% | 88.0% |
| inband_interferer | 7 | 100.0% | 100.0% |
| iot | 55 | 81.8% | 47.3% |
| laa | 12 | 100.0% | 91.7% |
| legacy | 346 | 80.1% | 73.1% |
| lib5g_dl | 13 | 100.0% | 100.0% |
| lib5g_dl_tdd | 72 | 98.6% | 98.6% |
| lmr | 91 | 63.7% | 54.9% |
| lte_dl | 25 | 100.0% | 100.0% |
| lte_dl_tdd | 45 | 93.3% | 84.4% |
| nb | 7 | 85.7% | 71.4% |
| nbdata | 31 | 51.6% | 45.2% |
| nbfh | 157 | 65.6% | 56.7% |
| nbiot | 10 | 80.0% | 60.0% |
| nr_dl | 24 | 100.0% | 91.7% |
| nr_dl_tdd | 87 | 100.0% | 100.0% |
| on_fill | 10 | 100.0% | 100.0% |
| oven | 63 | 100.0% | 100.0% |
| pager | 23 | 95.7% | 95.7% |
| prach | 28 | 64.3% | 57.1% |
| ptt | 93 | 66.7% | 61.3% |
| pucch | 517 | 56.5% | 52.2% |
| pusch | 1635 | 72.4% | 59.3% |
| quiet_line | 14 | 57.1% | 50.0% |
| radar | 79 | 40.5% | 40.5% |
| rx_dc | 251 | 79.3% | 69.3% |
| rx_spur | 48 | 18.8% | 16.7% |
| slow_hop | 12 | 8.3% | 0.0% |
| spur_comb | 1931 | 70.4% | 55.3% |
| srs | 104 | 76.0% | 62.5% |
| tdma | 40 | 35.0% | 32.5% |
| wide_hop | 56 | 94.6% | 94.6% |
| wifi | 2092 | 93.9% | 89.8% |
| wifi_ack | 1244 | 92.5% | 87.5% |
| zigbee | 12 | 83.3% | 83.3% |

**by class label**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 16QAM | 65 | 81.5% | 76.9% |
| 4FSK | 16 | 75.0% | 62.5% |
| 5G_Downlink | 115 | 98.3% | 97.4% |
| 64QAM | 9 | 77.8% | 77.8% |
| 802_11ax | 2459 | 92.4% | 88.1% |
| 802_11g | 891 | 96.1% | 91.2% |
| 8PSK | 14 | 64.3% | 57.1% |
| BLE | 84 | 86.9% | 75.0% |
| BPSK | 83 | 74.7% | 68.7% |
| Bluetooth | 909 | 78.9% | 67.3% |
| Broadband_FM | 45 | 86.7% | 77.8% |
| Broadcast_FM | 880 | 96.9% | 88.8% |
| CW | 26 | 50.0% | 42.3% |
| DC_spur | 251 | 79.3% | 69.3% |
| DMR | 57 | 52.6% | 52.6% |
| DroneCtrl | 179 | 96.6% | 93.9% |
| DroneVideo | 396 | 99.7% | 99.7% |
| FMCW | 7 | 71.4% | 71.4% |
| FSK | 77 | 61.0% | 48.1% |
| GFSK | 2497 | 85.7% | 77.7% |
| GSM | 535 | 91.0% | 88.0% |
| LAA | 12 | 100.0% | 91.7% |
| LMR | 29 | 48.3% | 48.3% |
| LTE | 70 | 95.7% | 90.0% |
| LTE-M | 6 | 66.7% | 50.0% |
| LTE_UL | 1079 | 63.5% | 54.2% |
| LoRa | 34 | 85.3% | 52.9% |
| Microwave | 63 | 100.0% | 100.0% |
| NB-IoT | 4 | 100.0% | 75.0% |
| NR | 111 | 100.0% | 98.2% |
| NR_UL | 1205 | 73.7% | 61.1% |
| NXDN | 42 | 61.9% | 52.4% |
| Narrowband_FM | 153 | 66.0% | 56.9% |
| NoiseBand | 21 | 90.5% | 90.5% |
| OFDM | 98 | 93.9% | 88.8% |
| P25 | 27 | 70.4% | 59.3% |
| Pager | 23 | 95.7% | 95.7% |
| QPSK | 194 | 89.2% | 84.5% |
| RX_spur | 1979 | 69.2% | 54.3% |
| RadarCW | 79 | 40.5% | 40.5% |
| TETRA | 27 | 51.9% | 48.1% |
| Zigbee | 12 | 83.3% | 83.3% |
