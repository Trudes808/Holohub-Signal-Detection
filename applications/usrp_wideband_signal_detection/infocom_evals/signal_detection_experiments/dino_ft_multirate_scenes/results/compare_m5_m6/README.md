# M5-S vs M6-S side-by-side panels

Rows per panel: model input with ground truth (red), M5-S mask, M6-S mask, difference (white both / red only M5-S / green only M6-S). Threshold 0.6. Rendered by `dino_fine_tuning/src/compare_overlays.py`.

## B210 @ 880 MHz: LTE + CDMA + LMR downlinks (user test case)

`cellular_downlink_880MHz_small_blocks` frame 0 @ 40.00 MSps. M5-S: pred occ 20.9%  |  GT covered 30%  precision 100%. M6-S: pred occ 68.8%  |  GT covered 99%  precision 99%.

![B210 @ 880 MHz: LTE + CDMA + LMR downlinks (user test case)](real_labelled_00_cellular_downlink_880MHz_small_blocks_f0.jpg)

## X310: LTE FDD 10 MHz test model (user test case)

`rx-waveform-td-rec-0-2023_02_23-17_03_59_287` frame 0 @ 15.36 MSps. M5-S: pred occ 24.8%  |  GT covered 41%  precision 97%. M6-S: pred occ 60.2%  |  GT covered 100%  precision 97%.

![X310: LTE FDD 10 MHz test model (user test case)](real_labelled_01_rx-waveform-td-rec-0-2023_02_23-17_03_59_287_f0.jpg)

## X310: NR FR1 10 MHz, 30 kHz SCS, TM2 (user test case)

`rx-waveform-td-rec-0-2023_02_23-17_01_38_653` frame 0 @ 15.36 MSps. M5-S: pred occ 19.6%  |  GT covered 26%  precision 78%. M6-S: pred occ 14.5%  |  GT covered 22%  precision 89%.

![X310: NR FR1 10 MHz, 30 kHz SCS, TM2 (user test case)](real_labelled_02_rx-waveform-td-rec-0-2023_02_23-17_01_38_653_f0.jpg)

## DroneDetect: DJI Mavic Pro flying + Bluetooth + WiFi interference (2.4375 GHz, 60 MSps)

`dronedetect_BOTH_MP1_FY` frame 0 @ 60.00 MSps. M5-S: pred occ 1.20%  (no labels). M6-S: pred occ 9.02%  (no labels).

![DroneDetect: DJI Mavic Pro flying + Bluetooth + WiFi interference (2.4375 GHz, 60 MSps)](real_unlabelled_00_dronedetect_BOTH_MP1_FY_f0.jpg)

## DroneDetect: DJI Mavic Air 2S flying + WiFi (2.4375 GHz, 60 MSps)

`dronedetect_WIFI_AIR_FY` frame 0 @ 60.00 MSps. M5-S: pred occ 10.74%  (no labels). M6-S: pred occ 37.61%  (no labels).

![DroneDetect: DJI Mavic Air 2S flying + WiFi (2.4375 GHz, 60 MSps)](real_unlabelled_01_dronedetect_WIFI_AIR_FY_f0.jpg)

## T-PRIME: two WiFi transmitters overlapping 50% (802.11ax @2.442 + 802.11n @2.452 GHz, 62.5 MSps)

`tprime_SR625_OVERLAP50_axn` frame 0 @ 62.50 MSps. M5-S: pred occ 40.02%  (no labels). M6-S: pred occ 46.46%  (no labels).

![T-PRIME: two WiFi transmitters overlapping 50% (802.11ax @2.442 + 802.11n @2.452 GHz, 62.5 MSps)](real_unlabelled_02_tprime_SR625_OVERLAP50_axn_f0.jpg)

## Synthetic: adjacent cellular carriers, 10.69 MSps

`cellular_adjacent_10-25M_10.69M_s200033` frame 0 @ 10.69 MSps. M5-S: pred occ 13.8%  |  GT covered 14%  precision 100%. M6-S: pred occ 100.0%  |  GT covered 100%  precision 100%.

![Synthetic: adjacent cellular carriers, 10.69 MSps](synthetic_00_cellular_adjacent_10-25M_10.69M_s200033_f0.jpg)

## Synthetic: adjacent cellular carriers, 62.5 MSps

`cellular_adjacent_60-150M_62.50M_s200029` frame 0 @ 62.50 MSps. M5-S: pred occ 17.0%  |  GT covered 30%  precision 98%. M6-S: pred occ 58.2%  |  GT covered 100%  precision 94%.

![Synthetic: adjacent cellular carriers, 62.5 MSps](synthetic_01_cellular_adjacent_60-150M_62.50M_s200029_f0.jpg)

## Synthetic: drone in a congested ISM band, 11.17 MSps

`drone_ism_10-25M_11.17M_s200056` frame 0 @ 11.17 MSps. M5-S: pred occ 23.1%  |  GT covered 23%  precision 100%. M6-S: pred occ 100.0%  |  GT covered 100%  precision 100%.

![Synthetic: drone in a congested ISM band, 11.17 MSps](synthetic_02_drone_ism_10-25M_11.17M_s200056_f0.jpg)

## Synthetic: frequency hoppers at the full 500 MSps rate

`hoppers_focus_400-500M_500.00M_s200060` frame 0 @ 500.00 MSps. M5-S: pred occ 0.3%  |  GT covered 3%  precision 93%. M6-S: pred occ 8.0%  |  GT covered 99%  precision 97%.

![Synthetic: frequency hoppers at the full 500 MSps rate](synthetic_03_hoppers_focus_400-500M_500.00M_s200060_f0.jpg)

## Synthetic: frequency hoppers, 10.42 MSps

`hoppers_focus_10-25M_10.42M_s200069` frame 1 @ 10.42 MSps. M5-S: pred occ 0.4%  |  GT covered 7%  precision 42%. M6-S: pred occ 30.8%  |  GT covered 85%  precision 7%.

![Synthetic: frequency hoppers, 10.42 MSps](synthetic_04_hoppers_focus_10-25M_10.42M_s200069_f1.jpg)

## Synthetic: ISM collisions (WiFi + BT + Zigbee), 22.34 MSps

`ism_collisions_10-25M_22.34M_s200045` frame 3 @ 22.34 MSps. M5-S: pred occ 30.2%  |  GT covered 66%  precision 98%. M6-S: pred occ 45.1%  |  GT covered 97%  precision 96%.

![Synthetic: ISM collisions (WiFi + BT + Zigbee), 22.34 MSps](synthetic_05_ism_collisions_10-25M_22.34M_s200045_f3.jpg)

## Synthetic: band-edge cut-off signals, 10.69 MSps

`band_edge_focus_10-25M_10.69M_s200081` frame 2 @ 10.69 MSps. M5-S: pred occ 7.9%  |  GT covered 10%  precision 100%. M6-S: pred occ 80.1%  |  GT covered 99%  precision 100%.

![Synthetic: band-edge cut-off signals, 10.69 MSps](synthetic_06_band_edge_focus_10-25M_10.69M_s200081_f2.jpg)

## Synthetic: dense random emitters at 491.52 MSps

`legacy_dense_400-500M_491.52M_s200001` frame 1 @ 491.52 MSps. M5-S: pred occ 4.3%  |  GT covered 83%  precision 99%. M6-S: pred occ 5.3%  |  GT covered 97%  precision 94%.

![Synthetic: dense random emitters at 491.52 MSps](synthetic_07_legacy_dense_400-500M_491.52M_s200001_f1.jpg)
