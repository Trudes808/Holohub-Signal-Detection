# Held-out scene evaluation: M13S_v7 (thr 0.6)

scenes 156, GT regions 26275, frames/scene <= 4

**Overall pixel**: precision 0.945 recall 0.863 IoU 0.821

**Pixel metrics by scenario**

| scenario | scenes | precision | recall | IoU | pred occ |
|---|---|---|---|---|---|
| band_edge_focus | 12 | 0.896 | 0.768 | 0.705 | 3.95% |
| broadcast_fm | 12 | 0.896 | 0.847 | 0.772 | 15.55% |
| cellular_adjacent | 12 | 0.971 | 0.890 | 0.866 | 40.46% |
| cellular_uplink | 12 | 0.899 | 0.734 | 0.678 | 7.98% |
| drone_ism | 12 | 0.943 | 0.951 | 0.899 | 47.11% |
| hoppers_focus | 12 | 0.937 | 0.614 | 0.589 | 5.36% |
| ism_collisions | 12 | 0.954 | 0.859 | 0.825 | 21.76% |
| legacy_dense | 12 | 0.941 | 0.840 | 0.798 | 7.11% |
| legacy_sparse | 12 | 0.864 | 0.843 | 0.744 | 6.03% |
| lmr_band | 12 | 0.872 | 0.772 | 0.694 | 3.90% |
| noise_only | 12 | 0.698 | 0.552 | 0.446 | 3.32% |
| radar_pulses | 12 | 0.893 | 0.918 | 0.827 | 26.72% |
| wideband_fill | 12 | 0.995 | 0.854 | 0.850 | 65.24% |

**Region detection by scenario (all GT boxes)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| band_edge_focus | 153 | 63.4% | 54.2% |
| broadcast_fm | 925 | 96.2% | 88.2% |
| cellular_adjacent | 835 | 82.2% | 75.6% |
| cellular_uplink | 2221 | 73.9% | 62.1% |
| drone_ism | 2940 | 85.6% | 79.1% |
| hoppers_focus | 1571 | 91.7% | 86.3% |
| ism_collisions | 2625 | 84.7% | 76.1% |
| legacy_dense | 428 | 75.5% | 60.3% |
| legacy_sparse | 470 | 87.0% | 70.9% |
| lmr_band | 1031 | 82.9% | 68.7% |
| noise_only | 754 | 66.4% | 48.3% |
| radar_pulses | 12247 | 91.9% | 87.8% |
| wideband_fill | 75 | 96.0% | 94.7% |

**Headline (excluding 12253 sub-pixel boxes and the radar_pulses scenario)**: 14022 regions, det@0.1 83.1% det@0.5 73.6%; scenario macro-average det@0.1 82.2% det@0.5 72.1%

**by scenario (excl. sub-pixel)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| band_edge_focus | 153 | 63.4% | 54.2% |
| broadcast_fm | 925 | 96.2% | 88.2% |
| cellular_adjacent | 835 | 82.2% | 75.6% |
| cellular_uplink | 2220 | 73.9% | 62.1% |
| drone_ism | 2939 | 85.7% | 79.1% |
| hoppers_focus | 1570 | 91.7% | 86.2% |
| ism_collisions | 2625 | 84.7% | 76.1% |
| legacy_dense | 425 | 76.0% | 60.7% |
| legacy_sparse | 470 | 87.0% | 70.9% |
| lmr_band | 1031 | 82.9% | 68.7% |
| noise_only | 754 | 66.4% | 48.3% |
| wideband_fill | 75 | 96.0% | 94.7% |

**by sample-rate bucket**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 10-25M | 2177 | 93.0% | 87.1% |
| 150-400M | 2936 | 70.0% | 55.7% |
| 25-60M | 1942 | 90.3% | 83.7% |
| 400-500M | 1728 | 67.5% | 55.5% |
| 5-10M | 2539 | 91.0% | 85.2% |
| 60-150M | 2700 | 86.8% | 75.6% |

**by PSD SNR (dB above floor)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 0-6 | 3097 | 75.7% | 56.0% |
| 12-20 | 2474 | 90.5% | 86.7% |
| 6-12 | 2737 | 86.8% | 77.5% |
| <0 | 1551 | 46.3% | 27.5% |
| >=20 | 4163 | 95.5% | 93.5% |

**by in-band fraction (band-edge cut)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 0.5-0.9 (cut) | 961 | 94.8% | 89.7% |
| <0.5 (mostly cut) | 1545 | 88.7% | 83.9% |
| >=0.9 (in band) | 11516 | 81.4% | 70.9% |

**by bandwidth (model bins)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 2-8 bins | 2651 | 84.6% | 74.6% |
| 8-32 bins | 1720 | 86.7% | 79.4% |
| <2 bins | 3965 | 68.6% | 53.3% |
| >=256 bins | 2594 | 95.3% | 90.3% |
| >=32 bins | 3092 | 88.3% | 81.6% |

**by duration (wide rows)**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 3-16 rows | 3066 | 81.6% | 75.0% |
| <3 rows | 2337 | 86.3% | 78.0% |
| >=128 rows | 4210 | 80.1% | 65.2% |
| >=16 rows | 4409 | 85.3% | 78.3% |

**by emitter behaviour**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| ble_adv | 12 | 58.3% | 33.3% |
| ble_data | 98 | 89.8% | 72.4% |
| bt_hop | 878 | 78.6% | 64.6% |
| drone_ctrl | 219 | 99.1% | 99.1% |
| drone_video | 732 | 99.7% | 99.6% |
| drone_video_fm | 11 | 100.0% | 100.0% |
| edge | 81 | 66.7% | 55.6% |
| fast_hop | 217 | 89.9% | 88.5% |
| fhss_net | 2235 | 82.0% | 73.2% |
| fill | 42 | 92.9% | 90.5% |
| fm_station | 904 | 96.9% | 88.7% |
| gsm | 323 | 95.0% | 91.6% |
| inband_interferer | 12 | 100.0% | 100.0% |
| iot | 56 | 91.1% | 58.9% |
| laa | 22 | 86.4% | 72.7% |
| legacy | 343 | 79.6% | 72.9% |
| lib5g_dl | 20 | 100.0% | 95.0% |
| lib5g_dl_tdd | 63 | 100.0% | 100.0% |
| lmr | 50 | 66.0% | 54.0% |
| lte_dl | 28 | 89.3% | 85.7% |
| lte_dl_tdd | 56 | 94.6% | 87.5% |
| nb | 4 | 75.0% | 75.0% |
| nbdata | 28 | 60.7% | 53.6% |
| nbfh | 81 | 51.9% | 46.9% |
| nbiot | 4 | 50.0% | 25.0% |
| nr_dl | 16 | 100.0% | 100.0% |
| nr_dl_tdd | 72 | 100.0% | 100.0% |
| on_fill | 10 | 100.0% | 100.0% |
| oven | 87 | 100.0% | 100.0% |
| pager | 34 | 76.5% | 76.5% |
| prach | 43 | 93.0% | 86.0% |
| ptt | 42 | 71.4% | 57.1% |
| pucch | 565 | 62.8% | 54.5% |
| pusch | 1332 | 71.8% | 61.1% |
| quiet_line | 61 | 65.6% | 63.9% |
| radar | 79 | 46.8% | 46.8% |
| rx_dc | 282 | 73.8% | 68.4% |
| rx_spur | 63 | 50.8% | 44.4% |
| slow_hop | 22 | 13.6% | 4.5% |
| spur_comb | 1663 | 75.8% | 52.3% |
| srs | 118 | 89.8% | 76.3% |
| tdma | 58 | 56.9% | 55.2% |
| wide_burst | 16 | 100.0% | 100.0% |
| wide_hop | 59 | 96.6% | 96.6% |
| wifi | 1784 | 91.0% | 84.4% |
| wifi_ack | 1077 | 89.9% | 82.2% |
| zigbee | 20 | 90.0% | 75.0% |

**by class label**

| group | regions | det@0.1 | det@0.5 |
|---|---|---|---|
| 16QAM | 61 | 80.3% | 72.1% |
| 4FSK | 11 | 72.7% | 72.7% |
| 5G_Downlink | 115 | 98.3% | 96.5% |
| 64QAM | 9 | 100.0% | 88.9% |
| 802_11ax | 1451 | 83.8% | 75.9% |
| 802_11g | 1426 | 97.5% | 91.5% |
| 8PSK | 10 | 80.0% | 80.0% |
| BLE | 110 | 86.4% | 68.2% |
| BPSK | 79 | 72.2% | 68.4% |
| Bluetooth | 888 | 78.2% | 64.3% |
| Broadband_FM | 31 | 67.7% | 54.8% |
| Broadcast_FM | 904 | 96.9% | 88.7% |
| CW | 63 | 55.6% | 50.8% |
| DC_spur | 282 | 73.8% | 68.4% |
| DMR | 40 | 57.5% | 55.0% |
| DroneCtrl | 219 | 99.1% | 99.1% |
| DroneVideo | 732 | 99.7% | 99.6% |
| FMCW | 9 | 88.9% | 77.8% |
| FSK | 57 | 57.9% | 50.9% |
| GFSK | 2364 | 81.3% | 72.6% |
| GSM | 323 | 95.0% | 91.6% |
| LAA | 22 | 86.4% | 72.7% |
| LMR | 16 | 62.5% | 62.5% |
| LTE | 84 | 92.9% | 86.9% |
| LTE-M | 1 | 100.0% | 100.0% |
| LTE_UL | 700 | 64.7% | 51.3% |
| LoRa | 34 | 94.1% | 55.9% |
| Microwave | 87 | 100.0% | 100.0% |
| NB-IoT | 3 | 33.3% | 0.0% |
| NR | 88 | 100.0% | 100.0% |
| NR_UL | 1367 | 74.1% | 65.8% |
| NXDN | 22 | 72.7% | 59.1% |
| Narrowband_FM | 86 | 62.8% | 46.5% |
| NoiseBand | 29 | 93.1% | 93.1% |
| OFDM | 95 | 90.5% | 85.3% |
| P25 | 33 | 48.5% | 45.5% |
| Pager | 34 | 76.5% | 76.5% |
| QPSK | 286 | 92.3% | 89.5% |
| RX_spur | 1726 | 74.9% | 52.0% |
| RadarCW | 79 | 46.8% | 46.8% |
| TETRA | 26 | 61.5% | 61.5% |
| Zigbee | 20 | 90.0% | 75.0% |
