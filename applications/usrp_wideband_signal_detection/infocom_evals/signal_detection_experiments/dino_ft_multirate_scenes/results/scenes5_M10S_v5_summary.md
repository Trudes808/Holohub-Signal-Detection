# Held-out scene evaluation: M10S_v5 (thr 0.6)

scenes 156, GT regions 29644, frames/scene <= 4

**Overall pixel**: precision 0.941 recall 0.842 IoU 0.800

**Pixel metrics by scenario**

| scenario | scenes | precision | recall | IoU | pred occ |
|---|---|---|---|---|---|
| band_edge_focus | 12 | 0.902 | 0.841 | 0.771 | 4.74% |
| broadcast_fm | 12 | 0.874 | 0.857 | 0.763 | 16.29% |
| cellular_adjacent | 12 | 0.954 | 0.741 | 0.716 | 34.76% |
| cellular_uplink | 12 | 0.927 | 0.840 | 0.788 | 10.30% |
| drone_ism | 12 | 0.967 | 0.947 | 0.917 | 52.63% |
| hoppers_focus | 12 | 0.950 | 0.921 | 0.878 | 7.63% |
| ism_collisions | 12 | 0.952 | 0.842 | 0.807 | 19.13% |
| legacy_dense | 12 | 0.961 | 0.894 | 0.863 | 7.63% |
| legacy_sparse | 12 | 0.959 | 0.905 | 0.871 | 8.79% |
| lmr_band | 12 | 0.818 | 0.692 | 0.599 | 1.89% |
| noise_only | 12 | 0.495 | 0.414 | 0.291 | 5.11% |
| radar_pulses | 12 | 0.891 | 0.721 | 0.663 | 25.10% |
| wideband_fill | 12 | 0.980 | 0.893 | 0.877 | 69.24% |

**False alarms on 2 annotation-free scenes**: predicted occupancy 8.620%

**Region detection by scenario (all GT boxes)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| band_edge_focus | 140 | 63.6% | 53.6% |
| broadcast_fm | 935 | 98.0% | 91.1% |
| cellular_adjacent | 1502 | 83.6% | 73.3% |
| cellular_uplink | 2414 | 78.6% | 71.9% |
| drone_ism | 3457 | 87.3% | 83.4% |
| hoppers_focus | 846 | 87.4% | 83.2% |
| ism_collisions | 2314 | 81.2% | 72.4% |
| legacy_dense | 360 | 74.4% | 65.8% |
| legacy_sparse | 227 | 41.9% | 34.8% |
| lmr_band | 474 | 65.6% | 57.0% |
| noise_only | 1128 | 53.6% | 42.1% |
| radar_pulses | 15772 | 91.4% | 72.0% |
| wideband_fill | 75 | 98.7% | 90.7% |

**Headline (excluding 15775 sub-pixel boxes and the radar_pulses scenario)**: 13869 regions, det@0.1 80.4% det@0.5 73.2%; scenario macro-average det@0.1 76.2% det@0.5 68.3%

**by scenario (excl. sub-pixel)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| band_edge_focus | 140 | 63.6% | 53.6% |
| broadcast_fm | 935 | 98.0% | 91.1% |
| cellular_adjacent | 1502 | 83.6% | 73.3% |
| cellular_uplink | 2414 | 78.6% | 71.9% |
| drone_ism | 3457 | 87.3% | 83.4% |
| hoppers_focus | 846 | 87.4% | 83.2% |
| ism_collisions | 2314 | 81.2% | 72.4% |
| legacy_dense | 357 | 74.8% | 66.1% |
| legacy_sparse | 227 | 41.9% | 34.8% |
| lmr_band | 474 | 65.6% | 57.0% |
| noise_only | 1128 | 53.6% | 42.1% |
| wideband_fill | 75 | 98.7% | 90.7% |

**by sample-rate bucket**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 10-25M | 1855 | 93.4% | 86.6% |
| 150-400M | 2193 | 60.8% | 52.0% |
| 25-60M | 1961 | 92.2% | 87.7% |
| 400-500M | 1892 | 70.6% | 60.0% |
| 5-10M | 2877 | 87.6% | 79.8% |
| 60-150M | 3091 | 78.1% | 72.9% |

**by PSD SNR (dB above floor)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 0-6 | 2385 | 64.2% | 48.8% |
| 12-20 | 2078 | 90.5% | 86.3% |
| 6-12 | 2940 | 83.3% | 76.6% |
| <0 | 1781 | 42.2% | 25.7% |
| >=20 | 4685 | 96.8% | 95.7% |

**by in-band fraction (band-edge cut)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 0.5-0.9 (cut) | 1659 | 96.3% | 92.2% |
| <0.5 (mostly cut) | 1789 | 63.9% | 57.5% |
| >=0.9 (in band) | 10421 | 80.6% | 72.9% |

**by bandwidth (model bins)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 2-8 bins | 2379 | 85.1% | 75.1% |
| 8-32 bins | 1442 | 89.3% | 83.8% |
| <2 bins | 2971 | 54.3% | 45.2% |
| >=256 bins | 3941 | 85.7% | 81.1% |
| >=32 bins | 3136 | 90.6% | 83.5% |

**by duration (wide rows)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 3-16 rows | 4540 | 85.6% | 80.2% |
| <3 rows | 3016 | 80.5% | 73.4% |
| >=128 rows | 3556 | 70.8% | 60.6% |
| >=16 rows | 2757 | 83.9% | 77.7% |

**by emitter behaviour**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| ble_adv | 26 | 96.2% | 65.4% |
| ble_data | 55 | 85.5% | 74.5% |
| bt_hop | 985 | 85.8% | 75.7% |
| cdma_dl | 3 | 100.0% | 100.0% |
| drone_ctrl | 160 | 100.0% | 98.8% |
| drone_video | 770 | 99.9% | 99.9% |
| drone_video_fm | 8 | 100.0% | 100.0% |
| edge | 81 | 66.7% | 55.6% |
| fast_hop | 342 | 80.1% | 78.4% |
| fill | 42 | 100.0% | 90.5% |
| fm_station | 908 | 99.0% | 92.3% |
| gsm | 884 | 97.5% | 93.2% |
| inband_interferer | 6 | 83.3% | 50.0% |
| iot | 58 | 74.1% | 43.1% |
| laa | 10 | 100.0% | 100.0% |
| legacy | 374 | 78.9% | 73.8% |
| lib5g_dl | 13 | 100.0% | 100.0% |
| lib5g_dl_tdd | 82 | 79.3% | 42.7% |
| lmr | 98 | 51.0% | 40.8% |
| lte_dl | 28 | 100.0% | 100.0% |
| lte_dl_tdd | 65 | 90.8% | 75.4% |
| nb | 8 | 87.5% | 87.5% |
| nbdata | 26 | 53.8% | 34.6% |
| nbfh | 160 | 65.6% | 56.9% |
| nbiot | 12 | 100.0% | 66.7% |
| nr_dl | 16 | 100.0% | 100.0% |
| nr_dl_tdd | 78 | 48.7% | 46.2% |
| on_fill | 10 | 100.0% | 80.0% |
| oven | 30 | 100.0% | 100.0% |
| pager | 42 | 71.4% | 59.5% |
| prach | 47 | 95.7% | 91.5% |
| ptt | 84 | 57.1% | 53.6% |
| pucch | 747 | 62.0% | 57.6% |
| pusch | 1768 | 82.1% | 72.0% |
| quiet_line | 14 | 57.1% | 50.0% |
| radar | 78 | 26.9% | 24.4% |
| rx_dc | 266 | 72.9% | 59.8% |
| rx_spur | 71 | 42.3% | 36.6% |
| slow_hop | 11 | 45.5% | 36.4% |
| spur_comb | 1348 | 51.3% | 39.5% |
| srs | 74 | 83.8% | 71.6% |
| tdma | 127 | 75.6% | 64.6% |
| wide_hop | 220 | 99.5% | 98.6% |
| wifi | 2231 | 82.8% | 77.9% |
| wifi_ack | 1348 | 81.0% | 75.2% |
| zigbee | 55 | 96.4% | 85.5% |

**by class label**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 16QAM | 65 | 80.0% | 69.2% |
| 4FSK | 14 | 50.0% | 35.7% |
| 5G_Downlink | 129 | 85.3% | 61.2% |
| 64QAM | 7 | 85.7% | 85.7% |
| 802_11ax | 2217 | 78.6% | 72.8% |
| 802_11g | 1378 | 87.9% | 83.4% |
| 8PSK | 8 | 87.5% | 62.5% |
| BLE | 81 | 88.9% | 71.6% |
| BPSK | 83 | 71.1% | 67.5% |
| Bluetooth | 998 | 85.5% | 75.5% |
| Broadband_FM | 28 | 71.4% | 64.3% |
| Broadcast_FM | 908 | 99.0% | 92.3% |
| CDMA | 3 | 100.0% | 100.0% |
| CW | 21 | 66.7% | 61.9% |
| DC_spur | 266 | 72.9% | 59.8% |
| DMR | 122 | 75.4% | 63.9% |
| DroneCtrl | 160 | 100.0% | 98.8% |
| DroneVideo | 770 | 99.9% | 99.9% |
| FMCW | 7 | 85.7% | 71.4% |
| FSK | 81 | 50.6% | 40.7% |
| GFSK | 105 | 45.7% | 32.4% |
| GSM | 884 | 97.5% | 93.2% |
| LAA | 10 | 100.0% | 100.0% |
| LMR | 18 | 22.2% | 11.1% |
| LTE | 93 | 93.5% | 82.8% |
| LTE-M | 9 | 100.0% | 66.7% |
| LTE_UL | 1470 | 72.4% | 64.1% |
| LoRa | 34 | 79.4% | 50.0% |
| Microwave | 30 | 100.0% | 100.0% |
| NB-IoT | 3 | 100.0% | 66.7% |
| NR | 94 | 57.4% | 55.3% |
| NR_UL | 1166 | 82.0% | 73.5% |
| NXDN | 25 | 44.0% | 32.0% |
| Narrowband_FM | 178 | 63.5% | 59.0% |
| NoiseBand | 22 | 90.9% | 86.4% |
| OFDM | 99 | 91.9% | 86.9% |
| P25 | 26 | 53.8% | 38.5% |
| Pager | 42 | 71.4% | 59.5% |
| QPSK | 615 | 90.9% | 88.6% |
| RX_spur | 1419 | 50.9% | 39.4% |
| RadarCW | 78 | 26.9% | 24.4% |
| TETRA | 48 | 79.2% | 66.7% |
| Zigbee | 55 | 96.4% | 85.5% |
