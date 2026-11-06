# N320 OTA stress captures — detector occupancy

30 captures under `/home/genesys-dgx1/Datasets/ARGUS/OOD_captures/OTA`; masks for M5S, M7S, M10S, M11S in `<capture>/masks/<model>.npz`, panels in `<capture>/panels/`.
Occupancy = fraction of the 512 x 1024 model grid marked signal, averaged over the evaluated frames (threshold 0.6).
Agreement = IoU of the two masks over those frames.

| capture | rate MSps | gain dB | frames | occ M5S | occ M7S | occ M10S | occ M11S | IoU M5S|M7S | IoU M5S|M10S | IoU M5S|M11S | IoU M7S|M10S | IoU M7S|M11S | IoU M10S|M11S | expected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `n320_adsb_dme_1078-1102MHz_c1090M_r24p576M_g60_ch1` | 24.58 | 60.0 | 12/46 | 0.23% | 14.17% | 4.86% | 6.50% | 0.01 | 0.04 | 0.03 | 0.07 | 0.18 | 0.30 | ADS-B / Mode S replies (1090 MHz pulses), DME, radar |
| `n320_cell1900_pcs_downlink_1929-1991MHz_c1960M_r61p44M_g55_ch0` | 61.44 | 55.0 | 12/39 | 22.52% | 75.45% | 75.04% | 72.55% | 0.30 | 0.30 | 0.31 | 0.92 | 0.90 | 0.91 | PCS band 2 downlinks (VERT2450 off-resonance) |
| `n320_cell2100_aws_downlink_2107-2157MHz_c2132M_r50M_g55_ch0` | 50.00 | 55.0 | 12/31 | 12.82% | 64.47% | 76.78% | 56.06% | 0.19 | 0.17 | 0.22 | 0.77 | 0.73 | 0.70 | AWS band 4/66 downlinks (VERT2450 off-resonance) |
| `n320_cell700_downlink_726-758MHz_c742M_r31p25M_g40_ch1` | 31.25 | 40.0 | 12/59 | 26.18% | 67.83% | 70.59% | 65.08% | 0.38 | 0.37 | 0.40 | 0.93 | 0.95 | 0.91 | LTE band 12/13/17 downlinks (5/10 MHz carriers) + part of band 13 |
| `n320_cell700_uplink_697-717MHz_c707M_r20M_g60_ch1` | 20.00 | 60.0 | 12/76 | 6.17% | 15.86% | 6.63% | 9.28% | 0.27 | 0.40 | 0.40 | 0.31 | 0.39 | 0.61 | LTE band 12/17 uplinks: sporadic narrow UE bursts, low duty |
| `n320_cell850_downlink_850-900MHz_offcentre_c875M_r50M_g55_ch1` | 50.00 | 55.0 | 12/15 | 12.03% | 60.48% | 74.39% | 59.04% | 0.19 | 0.16 | 0.20 | 0.78 | 0.81 | 0.77 | band 5 downlink in a wide off-centre look: UL/DL guard + ISM edge |
| `n320_cell850_downlink_869-894MHz_c881p5M_r25M_g55_ch1` | 25.00 | 55.0 | 12/47 | 2.68% | 94.08% | 94.65% | 96.54% | 0.03 | 0.03 | 0.03 | 0.95 | 0.95 | 0.95 | cellular band 5 downlinks (LTE/NR, CDMA legacy) centred on the band |
| `n320_cell850_downlink_884-896MHz_edge_c890M_r12p5M_g55_ch1` | 12.50 | 55.0 | 12/47 | 2.63% | 70.82% | 71.57% | 70.80% | 0.04 | 0.03 | 0.04 | 0.95 | 0.94 | 0.94 | upper band-5 downlink carriers, one cut by the band edge |
| `n320_cell850_uplink_814-876MHz_wide_c845M_r61p44M_g45_ch1` | 61.44 | 45.0 | 12/19 | 3.89% | 13.56% | 18.56% | 16.44% | 0.23 | 0.18 | 0.19 | 0.64 | 0.67 | 0.70 | band 5 uplink + guard + lower downlink in one wide look |
| `n320_cell850_uplink_824-849MHz_c836p5M_r25M_g60_ch1` | 25.00 | 60.0 | 12/47 | 0.48% | 3.23% | 5.27% | 3.44% | 0.09 | 0.06 | 0.09 | 0.33 | 0.20 | 0.42 | band 5 uplinks: low-duty UE bursts, PUCCH edges, PRACH |
| `n320_cell850_uplink_825-835MHz_narrow_c830M_r10p24M_g60_ch1` | 10.24 | 60.0 | 12/39 | 2.04% | 22.50% | 9.71% | 20.41% | 0.08 | 0.19 | 0.09 | 0.37 | 0.52 | 0.42 | band 5 uplink, narrow look, 10.24 MSps |
| `n320_fm_broadcast_69-119MHz_wide_c94M_r50M_g40_ch1` | 50.00 | 40.0 | 12/15 | 2.42% | 3.44% | 6.20% | 6.14% | 0.36 | 0.21 | 0.23 | 0.34 | 0.36 | 0.59 | FM band inside a wide 50 MHz look (band edges in view) |
| `n320_fm_broadcast_88-108MHz_c98M_r24p576M_g40_ch1` | 24.58 | 40.0 | 12/46 | 3.47% | 6.91% | 6.35% | 7.71% | 0.44 | 0.42 | 0.37 | 0.63 | 0.59 | 0.68 | FM broadcast stations on the 200 kHz raster (75 kHz dev stereo + RDS); VERT900 is off-resonance here |
| `n320_fm_broadcast_98-108MHz_partial_c103M_r10M_g55_ch1` | 10.00 | 55.0 | 12/38 | 2.24% | 13.86% | 5.05% | 2.73% | 0.16 | 0.33 | 0.38 | 0.30 | 0.17 | 0.44 | upper half of the FM band, narrow look |
| `n320_hdtv_uhf_480-580MHz_scan_c530M_r100M_g40_ch1` | 100.00 | 40.0 | 12/19 | 17.32% | 18.97% | 18.49% | 18.68% | 0.91 | 0.93 | 0.92 | 0.95 | 0.96 | 0.96 | ATSC 6 MHz HDTV channels (UHF 14-36), wide scan |
| `n320_hdtv_uhf_499-531MHz_c515M_r31p25M_g55_ch1` | 31.25 | 55.0 | 12/59 | 1.04% | 14.38% | 44.23% | 47.12% | 0.06 | 0.02 | 0.02 | 0.32 | 0.30 | 0.86 | ATSC HDTV channels, 31.25 MSps |
| `n320_hdtv_uhf_565-585MHz_c575M_r20M_g55_ch1` | 20.00 | 55.0 | 12/38 | 3.94% | 5.48% | 4.88% | 4.65% | 0.71 | 0.80 | 0.84 | 0.82 | 0.78 | 0.95 | ATSC HDTV channels, 20 MSps, likely one channel cut at the edge |
| `n320_ism2g4_2350-2550MHz_wide_c2450M_r200M_g50_ch0` | 200.00 | 50.0 | 12/19 | 5.97% | 8.59% | 8.87% | 7.24% | 0.56 | 0.50 | 0.57 | 0.63 | 0.63 | 0.70 | 2.4 GHz ISM inside a 200 MHz look (band edges and out-of-band in view) |
| `n320_ism2g4_2387-2487MHz_full_c2437M_r100M_g50_ch0` | 100.00 | 50.0 | 12/19 | 8.27% | 12.81% | 10.26% | 10.89% | 0.58 | 0.62 | 0.61 | 0.74 | 0.75 | 0.86 | 2.4 GHz ISM: WiFi ch 1-14, Bluetooth, BLE, Zigbee, microwave ovens |
| `n320_ism2g4_2392-2432MHz_ch1edge_c2412M_r40M_g45_ch0` | 40.00 | 45.0 | 12/76 | 5.67% | 7.87% | 6.30% | 6.27% | 0.61 | 0.62 | 0.65 | 0.69 | 0.69 | 0.80 | WiFi channel 1 at the centre, channel 6 cut at the edge |
| `n320_ism2g4_2411-2473MHz_c2442M_r61p44M_g50_ch0` | 61.44 | 50.0 | 12/39 | 7.16% | 13.38% | 12.11% | 11.65% | 0.46 | 0.47 | 0.48 | 0.72 | 0.71 | 0.82 | WiFi ch 1-11 at 61.44 MSps (odd rate for the detector) |
| `n320_ism2g4_2419-2431MHz_narrow_c2425M_r12p5M_g50_ch0` | 12.50 | 50.0 | 12/47 | 6.73% | 7.99% | 8.58% | 8.58% | 0.48 | 0.45 | 0.42 | 0.58 | 0.61 | 0.71 | between WiFi ch 3 and 5: BT hops and packet edges, 12.5 MSps |
| `n320_ism2g4_2452-2472MHz_ch11_c2462M_r20M_g45_ch0` | 20.00 | 45.0 | 12/38 | 3.72% | 8.73% | 8.37% | 8.03% | 0.33 | 0.38 | 0.38 | 0.54 | 0.60 | 0.73 | WiFi channel 11 only, 20 MSps |
| `n320_ism900_900-930MHz_c915M_r30p72M_g60_ch1` | 30.72 | 60.0 | 12/58 | 0.74% | 38.32% | 18.28% | 8.07% | 0.01 | 0.02 | 0.05 | 0.26 | 0.13 | 0.38 | 900 MHz ISM: LoRa, FHSS smart meters, Z-Wave, pagers nearby |
| `n320_ism900_914-926MHz_narrow_c920M_r12p5M_g60_ch1` | 12.50 | 60.0 | 12/47 | 1.52% | 3.50% | 4.75% | 5.00% | 0.31 | 0.26 | 0.26 | 0.58 | 0.58 | 0.76 | 900 MHz ISM, narrow look |
| `n320_quiet_1338-1362MHz_c1350M_r25M_g60_ch1` | 25.00 | 60.0 | 12/47 | 0.44% | 0.46% | 0.63% | 4.76% | 0.45 | 0.64 | 0.09 | 0.51 | 0.09 | 0.12 | expected quiet band (L-band radar region): false-alarm reference |
| `n320_wifi5g_5150-5250MHz_unii1_c5200M_r100M_g45_ch0` | 100.00 | 45.0 | 12/19 | 3.13% | 4.27% | 4.06% | 3.80% | 0.60 | 0.55 | 0.54 | 0.69 | 0.65 | 0.71 | 5 GHz UNII-1 WiFi (ch 36-48, 80 MHz BSSs) |
| `n320_wifi5g_5170-5190MHz_ch36_c5180M_r20M_g60_ch0` | 20.00 | 60.0 | 12/38 | 1.46% | 4.17% | 3.80% | 3.19% | 0.32 | 0.35 | 0.41 | 0.58 | 0.61 | 0.71 | single 5 GHz channel 36, 20 MSps |
| `n320_wifi5g_5400-5600MHz_unii2e_c5500M_r200M_g60_ch0` | 200.00 | 60.0 | 12/19 | 0.73% | 2.10% | 2.17% | 1.84% | 0.26 | 0.25 | 0.28 | 0.62 | 0.60 | 0.64 | 5 GHz UNII-2 extended (DFS) in a 200 MHz look |
| `n320_wifi5g_5744-5806MHz_unii3_c5775M_r62p5M_g60_ch0` | 62.50 | 60.0 | 12/39 | 3.50% | 5.30% | 4.87% | 4.54% | 0.63 | 0.67 | 0.69 | 0.77 | 0.76 | 0.85 | 5 GHz UNII-3 WiFi (ch 149-165) |
