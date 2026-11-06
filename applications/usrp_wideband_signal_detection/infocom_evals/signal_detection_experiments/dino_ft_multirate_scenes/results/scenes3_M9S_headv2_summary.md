# Held-out scene evaluation: M9S_headv2 (thr 0.6)

scenes 156, GT regions 26773, frames/scene <= 4

**Overall pixel**: precision 0.949 recall 0.928 IoU 0.883

**Pixel metrics by scenario**

| scenario | scenes | precision | recall | IoU | pred occ |
|---|---|---|---|---|---|
| band_edge_focus | 12 | 0.881 | 0.941 | 0.835 | 7.42% |
| broadcast_fm | 12 | 0.888 | 0.836 | 0.757 | 16.03% |
| cellular_adjacent | 12 | 0.967 | 0.923 | 0.895 | 40.07% |
| cellular_uplink | 12 | 0.931 | 0.837 | 0.788 | 9.76% |
| drone_ism | 12 | 0.950 | 0.961 | 0.915 | 49.90% |
| hoppers_focus | 12 | 0.926 | 0.883 | 0.824 | 2.88% |
| ism_collisions | 12 | 0.934 | 0.889 | 0.836 | 24.92% |
| legacy_dense | 12 | 0.950 | 0.941 | 0.896 | 7.85% |
| legacy_sparse | 12 | 0.887 | 0.954 | 0.851 | 4.32% |
| lmr_band | 12 | 0.810 | 0.702 | 0.603 | 1.99% |
| noise_only | 12 | 0.710 | 0.728 | 0.561 | 0.15% |
| radar_pulses | 12 | 0.898 | 0.914 | 0.828 | 29.02% |
| wideband_fill | 12 | 0.994 | 0.965 | 0.960 | 73.81% |

**False alarms on 4 annotation-free scenes**: predicted occupancy 0.023%

**Region detection by scenario (all GT boxes)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| band_edge_focus | 144 | 76.4% | 62.5% |
| broadcast_fm | 900 | 95.1% | 86.9% |
| cellular_adjacent | 1212 | 84.8% | 80.6% |
| cellular_uplink | 2386 | 82.1% | 74.1% |
| drone_ism | 2859 | 83.0% | 77.8% |
| hoppers_focus | 753 | 87.5% | 84.7% |
| ism_collisions | 3049 | 88.2% | 79.3% |
| legacy_dense | 254 | 79.5% | 70.9% |
| legacy_sparse | 93 | 82.8% | 78.5% |
| lmr_band | 489 | 68.1% | 60.7% |
| noise_only | 32 | 81.2% | 75.0% |
| radar_pulses | 14531 | 94.4% | 88.8% |
| wideband_fill | 71 | 100.0% | 100.0% |

**Headline (excluding 14534 sub-pixel boxes and the radar_pulses scenario)**: 12239 regions, det@0.1 84.8% det@0.5 77.9%; scenario macro-average det@0.1 84.1% det@0.5 77.6%

**by scenario (excl. sub-pixel)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| band_edge_focus | 144 | 76.4% | 62.5% |
| broadcast_fm | 900 | 95.1% | 86.9% |
| cellular_adjacent | 1212 | 84.8% | 80.6% |
| cellular_uplink | 2386 | 82.1% | 74.1% |
| drone_ism | 2859 | 83.0% | 77.8% |
| hoppers_focus | 753 | 87.5% | 84.7% |
| ism_collisions | 3049 | 88.2% | 79.3% |
| legacy_dense | 251 | 80.1% | 71.3% |
| legacy_sparse | 93 | 82.8% | 78.5% |
| lmr_band | 489 | 68.1% | 60.7% |
| noise_only | 32 | 81.2% | 75.0% |
| wideband_fill | 71 | 100.0% | 100.0% |

**by sample-rate bucket**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 10-25M | 1216 | 93.4% | 90.0% |
| 150-400M | 1631 | 70.9% | 62.4% |
| 25-60M | 2248 | 94.3% | 87.2% |
| 400-500M | 1710 | 75.5% | 63.6% |
| 5-10M | 1990 | 90.8% | 85.2% |
| 60-150M | 3444 | 83.5% | 77.9% |

**by PSD SNR (dB above floor)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 0-6 | 1822 | 71.9% | 61.0% |
| 12-20 | 2194 | 96.4% | 93.2% |
| 6-12 | 2565 | 90.3% | 83.3% |
| <0 | 1952 | 51.1% | 33.6% |
| >=20 | 3706 | 98.3% | 96.8% |

**by in-band fraction (band-edge cut)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 0.5-0.9 (cut) | 1544 | 90.2% | 81.5% |
| <0.5 (mostly cut) | 1851 | 71.8% | 65.8% |
| >=0.9 (in band) | 8844 | 86.6% | 79.9% |

**by bandwidth (model bins)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 2-8 bins | 2697 | 87.4% | 80.4% |
| 8-32 bins | 1218 | 93.3% | 88.1% |
| <2 bins | 1478 | 61.1% | 49.8% |
| >=256 bins | 4137 | 86.7% | 80.5% |
| >=32 bins | 2709 | 88.6% | 82.4% |

**by duration (wide rows)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 3-16 rows | 3929 | 84.6% | 79.1% |
| <3 rows | 3422 | 84.6% | 76.6% |
| >=128 rows | 2133 | 85.5% | 76.3% |
| >=16 rows | 2755 | 85.1% | 79.3% |

**by emitter behaviour**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| ble_adv | 14 | 92.9% | 64.3% |
| ble_data | 63 | 85.7% | 76.2% |
| bt_hop | 982 | 83.8% | 75.5% |
| cdma_dl | 3 | 100.0% | 100.0% |
| drone_ctrl | 347 | 98.3% | 98.3% |
| drone_video | 388 | 99.5% | 99.5% |
| drone_video_fm | 15 | 100.0% | 100.0% |
| edge | 79 | 68.4% | 58.2% |
| fast_hop | 300 | 81.7% | 80.7% |
| fill | 42 | 100.0% | 100.0% |
| fm_station | 872 | 95.8% | 87.7% |
| gsm | 676 | 97.9% | 97.3% |
| inband_interferer | 8 | 100.0% | 100.0% |
| iot | 57 | 89.5% | 61.4% |
| laa | 10 | 100.0% | 100.0% |
| legacy | 343 | 80.5% | 73.2% |
| lib5g_dl | 20 | 100.0% | 100.0% |
| lib5g_dl_tdd | 54 | 100.0% | 100.0% |
| lmr | 114 | 56.1% | 48.2% |
| lte_dl | 28 | 100.0% | 100.0% |
| lte_dl_tdd | 43 | 93.0% | 88.4% |
| nb | 7 | 85.7% | 85.7% |
| nbdata | 26 | 50.0% | 30.8% |
| nbfh | 157 | 65.0% | 52.2% |
| nbiot | 4 | 100.0% | 50.0% |
| nr_dl | 16 | 100.0% | 100.0% |
| nr_dl_tdd | 59 | 100.0% | 100.0% |
| on_fill | 10 | 100.0% | 100.0% |
| oven | 87 | 100.0% | 100.0% |
| pager | 41 | 80.5% | 73.2% |
| prach | 51 | 92.2% | 90.2% |
| ptt | 79 | 55.7% | 54.4% |
| pucch | 555 | 62.5% | 55.0% |
| pusch | 1874 | 82.2% | 73.4% |
| quiet_line | 14 | 64.3% | 64.3% |
| rx_dc | 293 | 88.4% | 74.1% |
| slow_hop | 4 | 25.0% | 25.0% |
| srs | 93 | 83.9% | 73.1% |
| tdma | 149 | 75.2% | 70.5% |
| wide_hop | 220 | 100.0% | 97.3% |
| wifi | 2504 | 83.8% | 76.9% |
| wifi_ack | 1503 | 82.8% | 73.5% |
| zigbee | 35 | 85.7% | 80.0% |

**by class label**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 16QAM | 66 | 80.3% | 68.2% |
| 4FSK | 12 | 83.3% | 83.3% |
| 5G_Downlink | 102 | 99.0% | 97.1% |
| 64QAM | 8 | 100.0% | 87.5% |
| 802_11ax | 2932 | 83.1% | 74.5% |
| 802_11g | 1091 | 84.3% | 79.0% |
| 8PSK | 10 | 100.0% | 100.0% |
| BLE | 77 | 87.0% | 74.0% |
| BPSK | 74 | 73.0% | 70.3% |
| Bluetooth | 994 | 83.4% | 75.1% |
| Broadband_FM | 32 | 75.0% | 68.8% |
| Broadcast_FM | 872 | 95.8% | 87.7% |
| CDMA | 3 | 100.0% | 100.0% |
| CW | 17 | 82.4% | 82.4% |
| DC_spur | 293 | 88.4% | 74.1% |
| DMR | 136 | 78.7% | 70.6% |
| DroneCtrl | 347 | 98.3% | 98.3% |
| DroneVideo | 388 | 99.5% | 99.5% |
| FMCW | 7 | 100.0% | 85.7% |
| FSK | 96 | 55.2% | 43.8% |
| GFSK | 101 | 49.5% | 35.6% |
| GSM | 676 | 97.9% | 97.3% |
| LAA | 10 | 100.0% | 100.0% |
| LMR | 20 | 70.0% | 65.0% |
| LTE | 71 | 95.8% | 93.0% |
| LTE-M | 1 | 100.0% | 0.0% |
| LTE_UL | 1262 | 69.3% | 60.4% |
| LoRa | 28 | 89.3% | 57.1% |
| Microwave | 87 | 100.0% | 100.0% |
| NB-IoT | 3 | 100.0% | 66.7% |
| NR | 75 | 100.0% | 100.0% |
| NR_UL | 1311 | 86.8% | 78.8% |
| NXDN | 31 | 41.9% | 41.9% |
| Narrowband_FM | 164 | 58.5% | 51.8% |
| NoiseBand | 23 | 91.3% | 87.0% |
| OFDM | 94 | 93.6% | 90.4% |
| P25 | 37 | 56.8% | 40.5% |
| Pager | 41 | 80.5% | 73.2% |
| QPSK | 566 | 92.2% | 89.9% |
| TETRA | 46 | 73.9% | 67.4% |
| Zigbee | 35 | 85.7% | 80.0% |
