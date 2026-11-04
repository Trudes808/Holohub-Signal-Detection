# Held-out scene evaluation: M8S_scenes_v3 (thr 0.6)

scenes 156, GT regions 26773, frames/scene <= 4

**Overall pixel**: precision 0.919 recall 0.930 IoU 0.860

**Pixel metrics by scenario**

| scenario | scenes | precision | recall | IoU | pred occ |
|---|---|---|---|---|---|
| band_edge_focus | 12 | 0.868 | 0.942 | 0.825 | 7.53% |
| broadcast_fm | 12 | 0.846 | 0.848 | 0.735 | 17.08% |
| cellular_adjacent | 12 | 0.954 | 0.941 | 0.900 | 41.45% |
| cellular_uplink | 12 | 0.858 | 0.850 | 0.745 | 10.75% |
| drone_ism | 12 | 0.931 | 0.969 | 0.905 | 51.35% |
| hoppers_focus | 12 | 0.914 | 0.905 | 0.834 | 2.99% |
| ism_collisions | 12 | 0.882 | 0.902 | 0.805 | 26.75% |
| legacy_dense | 12 | 0.889 | 0.947 | 0.847 | 8.43% |
| legacy_sparse | 12 | 0.865 | 0.962 | 0.836 | 4.47% |
| lmr_band | 12 | 0.801 | 0.689 | 0.588 | 1.97% |
| noise_only | 12 | 0.655 | 0.825 | 0.575 | 0.18% |
| radar_pulses | 12 | 0.814 | 0.933 | 0.769 | 32.68% |
| wideband_fill | 12 | 0.995 | 0.941 | 0.937 | 71.96% |

**False alarms on 4 annotation-free scenes**: predicted occupancy 0.047%

**Region detection by scenario (all GT boxes)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| band_edge_focus | 144 | 76.4% | 66.7% |
| broadcast_fm | 900 | 94.6% | 87.9% |
| cellular_adjacent | 1212 | 86.6% | 82.9% |
| cellular_uplink | 2386 | 80.1% | 72.3% |
| drone_ism | 2859 | 83.0% | 78.4% |
| hoppers_focus | 753 | 87.3% | 84.7% |
| ism_collisions | 3049 | 87.6% | 79.6% |
| legacy_dense | 254 | 78.3% | 74.4% |
| legacy_sparse | 93 | 87.1% | 84.9% |
| lmr_band | 489 | 66.3% | 57.1% |
| noise_only | 32 | 84.4% | 84.4% |
| radar_pulses | 14531 | 94.6% | 90.0% |
| wideband_fill | 71 | 100.0% | 100.0% |

**Headline (excluding 14534 sub-pixel boxes and the radar_pulses scenario)**: 12239 regions, det@0.1 84.4% det@0.5 78.2%; scenario macro-average det@0.1 84.3% det@0.5 79.5%

**by scenario (excl. sub-pixel)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| band_edge_focus | 144 | 76.4% | 66.7% |
| broadcast_fm | 900 | 94.6% | 87.9% |
| cellular_adjacent | 1212 | 86.6% | 82.9% |
| cellular_uplink | 2386 | 80.1% | 72.3% |
| drone_ism | 2859 | 83.0% | 78.4% |
| hoppers_focus | 753 | 87.3% | 84.7% |
| ism_collisions | 3049 | 87.6% | 79.6% |
| legacy_dense | 251 | 78.9% | 74.9% |
| legacy_sparse | 93 | 87.1% | 84.9% |
| lmr_band | 489 | 66.3% | 57.1% |
| noise_only | 32 | 84.4% | 84.4% |
| wideband_fill | 71 | 100.0% | 100.0% |

**by sample-rate bucket**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 10-25M | 1216 | 93.6% | 89.6% |
| 150-400M | 1631 | 70.3% | 63.3% |
| 25-60M | 2248 | 94.0% | 87.5% |
| 400-500M | 1710 | 71.5% | 62.5% |
| 5-10M | 1990 | 91.8% | 86.3% |
| 60-150M | 3444 | 83.5% | 78.2% |

**by PSD SNR (dB above floor)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 0-6 | 1822 | 71.4% | 63.3% |
| 12-20 | 2194 | 95.6% | 92.0% |
| 6-12 | 2565 | 88.3% | 82.5% |
| <0 | 1952 | 53.1% | 37.2% |
| >=20 | 3706 | 97.8% | 95.8% |

**by in-band fraction (band-edge cut)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 0.5-0.9 (cut) | 1544 | 90.3% | 81.5% |
| <0.5 (mostly cut) | 1851 | 72.1% | 67.0% |
| >=0.9 (in band) | 8844 | 85.9% | 79.9% |

**by bandwidth (model bins)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 2-8 bins | 2697 | 86.0% | 79.7% |
| 8-32 bins | 1218 | 93.0% | 87.4% |
| <2 bins | 1478 | 58.2% | 49.5% |
| >=256 bins | 4137 | 87.0% | 81.0% |
| >=32 bins | 2709 | 89.1% | 83.8% |

**by duration (wide rows)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 3-16 rows | 3929 | 83.1% | 78.3% |
| <3 rows | 3422 | 84.8% | 77.3% |
| >=128 rows | 2133 | 85.6% | 78.2% |
| >=16 rows | 2755 | 84.7% | 79.0% |

**by emitter behaviour**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| ble_adv | 14 | 85.7% | 78.6% |
| ble_data | 63 | 84.1% | 81.0% |
| bt_hop | 982 | 80.9% | 73.9% |
| cdma_dl | 3 | 100.0% | 100.0% |
| drone_ctrl | 347 | 98.3% | 97.1% |
| drone_video | 388 | 99.5% | 99.5% |
| drone_video_fm | 15 | 100.0% | 100.0% |
| edge | 79 | 62.0% | 50.6% |
| fast_hop | 300 | 81.3% | 80.3% |
| fill | 42 | 100.0% | 100.0% |
| fm_station | 872 | 95.4% | 88.8% |
| gsm | 676 | 97.9% | 97.9% |
| inband_interferer | 8 | 100.0% | 100.0% |
| iot | 57 | 89.5% | 66.7% |
| laa | 10 | 100.0% | 100.0% |
| legacy | 343 | 80.2% | 77.0% |
| lib5g_dl | 20 | 100.0% | 100.0% |
| lib5g_dl_tdd | 54 | 100.0% | 100.0% |
| lmr | 114 | 53.5% | 43.9% |
| lte_dl | 28 | 100.0% | 100.0% |
| lte_dl_tdd | 43 | 93.0% | 86.0% |
| nb | 7 | 100.0% | 85.7% |
| nbdata | 26 | 38.5% | 30.8% |
| nbfh | 157 | 65.6% | 52.9% |
| nbiot | 4 | 100.0% | 75.0% |
| nr_dl | 16 | 100.0% | 100.0% |
| nr_dl_tdd | 59 | 100.0% | 100.0% |
| on_fill | 10 | 100.0% | 100.0% |
| oven | 87 | 100.0% | 100.0% |
| pager | 41 | 78.0% | 73.2% |
| prach | 51 | 94.1% | 90.2% |
| ptt | 79 | 54.4% | 50.6% |
| pucch | 555 | 55.7% | 48.5% |
| pusch | 1874 | 82.7% | 74.4% |
| quiet_line | 14 | 64.3% | 64.3% |
| rx_dc | 293 | 93.2% | 84.3% |
| slow_hop | 4 | 25.0% | 25.0% |
| srs | 93 | 80.6% | 63.4% |
| tdma | 149 | 73.8% | 63.1% |
| wide_hop | 220 | 99.5% | 98.6% |
| wifi | 2504 | 83.9% | 77.0% |
| wifi_ack | 1503 | 83.0% | 75.8% |
| zigbee | 35 | 85.7% | 68.6% |

**by class label**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 16QAM | 66 | 77.3% | 71.2% |
| 4FSK | 12 | 83.3% | 83.3% |
| 5G_Downlink | 102 | 99.0% | 98.0% |
| 64QAM | 8 | 87.5% | 87.5% |
| 802_11ax | 2932 | 83.3% | 75.6% |
| 802_11g | 1091 | 84.6% | 79.5% |
| 8PSK | 10 | 100.0% | 100.0% |
| BLE | 77 | 84.4% | 80.5% |
| BPSK | 74 | 73.0% | 67.6% |
| Bluetooth | 994 | 80.6% | 73.7% |
| Broadband_FM | 32 | 75.0% | 68.8% |
| Broadcast_FM | 872 | 95.4% | 88.8% |
| CDMA | 3 | 100.0% | 100.0% |
| CW | 17 | 82.4% | 82.4% |
| DC_spur | 293 | 93.2% | 84.3% |
| DMR | 136 | 77.9% | 66.9% |
| DroneCtrl | 347 | 98.3% | 97.1% |
| DroneVideo | 388 | 99.5% | 99.5% |
| FMCW | 7 | 100.0% | 85.7% |
| FSK | 96 | 56.2% | 42.7% |
| GFSK | 101 | 47.5% | 35.6% |
| GSM | 676 | 97.9% | 97.9% |
| LAA | 10 | 100.0% | 100.0% |
| LMR | 20 | 70.0% | 65.0% |
| LTE | 71 | 95.8% | 91.5% |
| LTE-M | 1 | 100.0% | 100.0% |
| LTE_UL | 1262 | 65.8% | 57.1% |
| LoRa | 28 | 89.3% | 75.0% |
| Microwave | 87 | 100.0% | 100.0% |
| NB-IoT | 3 | 100.0% | 66.7% |
| NR | 75 | 100.0% | 100.0% |
| NR_UL | 1311 | 87.9% | 80.0% |
| NXDN | 31 | 45.2% | 41.9% |
| Narrowband_FM | 164 | 56.7% | 50.0% |
| NoiseBand | 23 | 87.0% | 87.0% |
| OFDM | 94 | 91.5% | 88.3% |
| P25 | 37 | 48.6% | 43.2% |
| Pager | 41 | 78.0% | 73.2% |
| QPSK | 566 | 91.9% | 90.6% |
| TETRA | 46 | 71.7% | 50.0% |
| Zigbee | 35 | 85.7% | 68.6% |
