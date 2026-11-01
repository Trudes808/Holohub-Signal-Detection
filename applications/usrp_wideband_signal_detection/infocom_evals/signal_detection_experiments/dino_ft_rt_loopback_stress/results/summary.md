## Speed / real-time health (loopback legs)

Sustained = no packet lost AND the NIC buffer pool not draining (drain < 1% of arrivals); a finite replay can end loss-free while the pool drains, i.e. while the pipeline is slower than real time.

| run | leg | rate MSps | loops | budget ms/frame | frames emitted/expected | det ms p50 / p95 / max | emit cadence p50 ms | CHDR->FFT lat mean / max ms | pkt loss | NIC pool drain (% of arrivals) | sustained | backlog / fell-behind | GPU mean / max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dinoft_rt_m5s_245 | DINO-FT RT (M5-S, ViT-S) -- deployed | 245.76 | 1 | 21.3 | 342/342 (100.0%) | 22.1 / 27.0 / 72.4 | 23.1 | 72.8 / 162.7 | 0.000% | 8.3% | NO | 0 / 0 | 68% / 78% |
| v_full_rt_m5s_245 | DINO-FT RT (M5-S, ViT-S) -- deployed | 245.76 | 1 | 21.3 | 334/342 (97.7%) | 22.8 / 29.9 / 120.7 | 23.6 | 76.4 / 238.5 | 2.299% | exhausted | NO | 0 / 0 | 66% / 77% |
| lean_noviz_245 | DINO-FT RT (M5-S), lean: no snipper, no dashboard | 245.76 | 1 | 21.3 | 342/342 (100.0%) | 17.7 / 19.8 / 165.6 | 20.6 | - / - | 0.000% | 0.6% | yes | 0 / 0 | 54% / 68% |
| v_lean_245 | DINO-FT RT (M5-S), lean: no snipper, no dashboard | 245.76 | 1 | 21.3 | 342/342 (100.0%) | 17.7 / 19.9 / 59.4 | 20.9 | - / - | 0.000% | 0.3% | yes | 0 / 0 | 54% / 72% |
| coherent_245 | coherent_power (dynamic floor) | 245.76 | 1 | 10.7 | 681/684 (99.6%) | 0.5 / 1.3 / 8.8 | 10.5 | 32.6 / 44.6 | 0.000% | 0.0% | yes | 0 / 0 | 30% / 46% |
| v_full_coherent_245 | coherent_power (dynamic floor) | 245.76 | 1 | 10.7 | 681/684 (99.6%) | 0.5 / 1.2 / 11.6 | 10.6 | 32.5 / 47.9 | 0.000% | 0.0% | yes | 0 / 0 | 35% / 48% |
| dinoft_native_m2dr_245 | DINO-FT native (M2_dr, ViT-B) | 245.76 | 1 | 21.3 | 72/342 (21.1%) | 207.8 / 271.5 / 1637.7 | 208.7 | 770.7 / 3518.2 | 79.123% | exhausted | NO | 1 / 1 | 65% / 96% |
| dinoft_native_m5s_245 | DINO-FT native (M5-S, ViT-S) | 245.76 | 1 | 21.3 | 106/342 (31.0%) | 127.0 / 155.7 / 221.4 | 127.8 | 397.3 / 616.6 | 69.181% | exhausted | NO | 0 / 0 | 91% / 95% |
| dinoft_rt_m2dr_245 | DINO-FT RT (M2_dr, ViT-B) | 245.76 | 1 | 21.3 | 276/342 (80.7%) | 30.1 / 36.2 / 72.9 | 30.9 | 97.9 / 332.2 | 19.472% | exhausted | NO | 0 / 0 | 86% / 88% |
| dinoft_rt_m5s_491 | DINO-FT RT (M5-S, ViT-S) -- deployed | 491.52 | 6 | 21.3 | 715/1026 (69.7%) | 32.7 / 45.4 / 366.7 | 34.0 | 113.8 / 968.2 | 30.383% | exhausted | NO | 1 / 1 | 70% / 84% |
| dinoft_rt_m5s_491c | DINO-FT RT (M5-S, ViT-S) -- deployed | 491.52 | 6 | 21.3 | 685/1026 (66.8%) | 33.8 / 48.0 / 96.7 | 35.0 | 113.3 / 277.7 | 33.286% | exhausted | NO | 1 / 1 | 72% / 83% |
| dinoft_rt_m5s_nosnip_491c | DINO-FT RT (M5-S), no snipper | 491.52 | 6 | 21.3 | 877/1026 (85.5%) | 28.6 / 31.9 / 97.1 | 29.4 | 91.1 / 303.7 | 14.600% | exhausted | NO | 1 / 1 | 74% / 82% |
| dinoft_rt_m5s_nosnip_491d | DINO-FT RT (M5-S), no snipper | 491.52 | 6 | 21.3 | 893/1026 (87.0%) | 28.2 / 31.2 / 104.7 | 29.1 | 89.9 / 315.6 | 13.004% | exhausted | NO | 1 / 1 | 77% / 83% |
| lean_noviz_491 | DINO-FT RT (M5-S), lean: no snipper, no dashboard | 491.52 | 6 | 21.3 | 870/1026 (84.8%) | 26.1 / 30.6 / 83.9 | 27.0 | - / - | 15.226% | exhausted | NO | 0 / 0 | 65% / 74% |
| coherent_491 | coherent_power (dynamic floor) | 491.52 | 6 | 10.7 | 1999/2052 (97.4%) | 0.4 / 1.6 / 13.3 | 11.9 | 40.1 / 104.4 | 2.185% | exhausted | NO | 36 / 36 | 73% / 89% |
| coherent_491c | coherent_power (dynamic floor) | 491.52 | 6 | 10.7 | 2011/2052 (98.0%) | 0.4 / 2.2 / 14.6 | 12.0 | 39.8 / 279.1 | 1.822% | exhausted | NO | 21 / 21 | 71% / 93% |
| dinoft_native_m2dr_491 | DINO-FT native (M2_dr, ViT-B) | 491.52 | 6 | 21.3 | 80/1026 (7.8%) | 417.2 / 777.6 / 2681.9 | 418.0 | 1503.8 / 6086.4 | 92.281% | exhausted | NO | 1 / 1 | 87% / 96% |
| native_m2dr_lean_491 | DINO-FT native (M2_dr), lean | 491.52 | 6 | 21.3 | 87/1026 (8.5%) | 414.7 / 507.7 / 2562.7 | 415.7 | - / - | 91.598% | exhausted | NO | 0 / 0 | 82% / 96% |
| dinoft_native_m5s_491 | DINO-FT native (M5-S, ViT-S) | 491.52 | 6 | 21.3 | 143/1026 (13.9%) | 254.4 / 315.2 / 1729.5 | 255.5 | 837.8 / 2688.6 | 86.121% | exhausted | NO | 0 / 0 | 89% / 96% |
| dinoft_rt_m2dr_491 | DINO-FT RT (M2_dr, ViT-B) | 491.52 | 6 | 21.3 | 648/1026 (63.2%) | 39.5 / 47.8 / 89.9 | 40.4 | 126.8 / 315.0 | 36.920% | exhausted | NO | 1 / 1 | 80% / 90% |

## Accuracy vs SigMF ground truth (region = annotation box; det@t = box coverage >= t; effective = unprocessed frames count as misses; pixel metrics on processed frames)

| run | leg | side | frames w/ mask | det@0.1 | det@0.5 | effective det@0.5 | px prec | px recall | px F1 | px IoU | FP area | NBFM det | Bluetooth det | ZC det | <2 MHz det | online==offline IoU (k) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dinoft_rt_m5s_245 | DINO-FT RT (M5-S, ViT-S) -- deployed | loopback | 342/342 | 0.918 | 0.763 | 0.763 | 0.776 | 0.806 | 0.855 | 0.642 | 0.0034 | 0.00 | 0.85 | 0.41 | 0.64 | 1.000 (+0) |
| dinoft_rt_m5s_245 | DINO-FT RT (M5-S, ViT-S) -- deployed | offline | 342/342 | 0.918 | 0.763 | 0.763 | 0.776 | 0.806 | 0.855 | 0.642 | 0.0034 | 0.00 | 0.85 | 0.41 | 0.64 | - |
| v_full_rt_m5s_245 | DINO-FT RT (M5-S, ViT-S) -- deployed | loopback | 334/342 | 0.597 | 0.500 | 0.500 | 0.604 | 0.618 | 0.801 | 0.507 | 0.0081 | 0.06 | 0.52 | 0.23 | 0.48 | 1.000 (+0) |
| v_full_rt_m5s_245 | DINO-FT RT (M5-S, ViT-S) -- deployed | offline | 342/342 | 0.918 | 0.763 | 0.763 | 0.776 | 0.806 | 0.855 | 0.642 | 0.0034 | 0.00 | 0.85 | 0.41 | 0.64 | - |
| lean_noviz_245 | DINO-FT RT (M5-S), lean: no snipper, no dashboard | loopback | 342/342 | 0.918 | 0.763 | 0.763 | 0.776 | 0.806 | 0.855 | 0.642 | 0.0034 | 0.00 | 0.85 | 0.41 | 0.64 | 1.000 (+0) |
| lean_noviz_245 | DINO-FT RT (M5-S), lean: no snipper, no dashboard | offline | 342/342 | 0.918 | 0.763 | 0.763 | 0.776 | 0.806 | 0.855 | 0.642 | 0.0034 | 0.00 | 0.85 | 0.41 | 0.64 | - |
| v_lean_245 | DINO-FT RT (M5-S), lean: no snipper, no dashboard | loopback | 342/342 | 0.918 | 0.763 | 0.763 | 0.776 | 0.806 | 0.855 | 0.642 | 0.0034 | 0.00 | 0.85 | 0.41 | 0.64 | 1.000 (+0) |
| v_lean_245 | DINO-FT RT (M5-S), lean: no snipper, no dashboard | offline | 342/342 | 0.918 | 0.763 | 0.763 | 0.776 | 0.806 | 0.855 | 0.642 | 0.0034 | 0.00 | 0.85 | 0.41 | 0.64 | - |
| coherent_245 | coherent_power (dynamic floor) | loopback | 681/684 | 0.958 | 0.826 | 0.826 | 0.755 | 0.507 | 0.580 | 0.377 | 0.0029 | 0.97 | 0.93 | 0.83 | 0.93 | 1.000 (+0) |
| coherent_245 | coherent_power (dynamic floor) | offline | 684/684 | 0.958 | 0.827 | 0.827 | 0.753 | 0.508 | 0.581 | 0.377 | 0.0027 | 0.97 | 0.93 | 0.83 | 0.93 | - |
| v_full_coherent_245 | coherent_power (dynamic floor) | loopback | 681/684 | 0.956 | 0.825 | 0.825 | 0.755 | 0.508 | 0.580 | 0.378 | 0.0028 | 0.97 | 0.93 | 0.83 | 0.93 | 1.000 (+0) |
| v_full_coherent_245 | coherent_power (dynamic floor) | offline | 684/684 | 0.958 | 0.827 | 0.827 | 0.753 | 0.508 | 0.581 | 0.377 | 0.0027 | 0.97 | 0.93 | 0.83 | 0.93 | - |
| dinoft_native_m2dr_245 | DINO-FT native (M2_dr, ViT-B) | loopback | 72/342 | 0.713 | 0.640 | 0.102 | 0.816 | 0.741 | 0.789 | 0.707 | 0.0162 | 0.00 | 0.00 | 0.07 | 0.66 | 1.000 (+0) |
| dinoft_native_m2dr_245 | DINO-FT native (M2_dr, ViT-B) | offline | 342/342 | 0.850 | 0.748 | 0.748 | 0.968 | 0.912 | 0.937 | 0.887 | 0.0011 | 0.75 | 0.92 | 0.09 | 0.91 | - |
| dinoft_native_m5s_245 | DINO-FT native (M5-S, ViT-S) | loopback | 106/342 | 0.635 | 0.520 | 0.119 | 0.692 | 0.507 | 0.535 | 0.443 | 0.0247 | 0.12 | 0.29 | 0.11 | 0.82 | 1.000 (+0) |
| dinoft_native_m5s_245 | DINO-FT native (M5-S, ViT-S) | offline | 342/342 | 0.848 | 0.708 | 0.708 | 0.731 | 0.706 | 0.746 | 0.532 | 0.0048 | 0.69 | 0.97 | 0.02 | 0.92 | - |
| dinoft_rt_m2dr_245 | DINO-FT RT (M2_dr, ViT-B) | loopback | 276/342 | 0.416 | 0.321 | 0.297 | 0.626 | 0.401 | 0.540 | 0.368 | 0.0322 | 0.04 | 0.34 | 0.22 | 0.36 | 0.263 (+0) |
| dinoft_rt_m2dr_245 | DINO-FT RT (M2_dr, ViT-B) | offline | 342/342 | 0.936 | 0.781 | 0.781 | 0.953 | 0.746 | 0.822 | 0.716 | 0.0027 | 0.00 | 0.73 | 0.60 | 0.60 | - |
| dinoft_rt_m5s_491 | DINO-FT RT (M5-S, ViT-S) -- deployed | loopback | 168/171 | n/a: frame alignment lost after packet drops (online-vs-offline IoU 0.040) | | | | | | | | | | | | - |
| dinoft_rt_m5s_491 | DINO-FT RT (M5-S, ViT-S) -- deployed | offline | 171/171 | 0.876 | 0.712 | 0.712 | 0.936 | 0.740 | 0.819 | 0.712 | 0.0042 | 0.00 | 0.82 | 0.41 | 0.29 | - |
| dinoft_rt_m5s_491c | DINO-FT RT (M5-S, ViT-S) -- deployed | loopback | 168/171 | n/a: frame alignment lost after packet drops (online-vs-offline IoU 0.031) | | | | | | | | | | | | - |
| dinoft_rt_m5s_491c | DINO-FT RT (M5-S, ViT-S) -- deployed | offline | 171/171 | 0.876 | 0.712 | 0.712 | 0.936 | 0.740 | 0.819 | 0.712 | 0.0042 | 0.00 | 0.82 | 0.41 | 0.29 | - |
| dinoft_rt_m5s_nosnip_491c | DINO-FT RT (M5-S), no snipper | loopback | 170/171 | n/a: frame alignment lost after packet drops (online-vs-offline IoU 0.099) | | | | | | | | | | | | - |
| dinoft_rt_m5s_nosnip_491c | DINO-FT RT (M5-S), no snipper | offline | 171/171 | 0.876 | 0.712 | 0.712 | 0.936 | 0.740 | 0.819 | 0.712 | 0.0042 | 0.00 | 0.82 | 0.41 | 0.29 | - |
| dinoft_rt_m5s_nosnip_491d | DINO-FT RT (M5-S), no snipper | loopback | 171/171 | 0.573 | 0.469 | 0.469 | 0.695 | 0.587 | 0.746 | 0.530 | 0.0200 | 0.00 | 0.47 | 0.24 | 0.25 | 1.000 (+0) |
| dinoft_rt_m5s_nosnip_491d | DINO-FT RT (M5-S), no snipper | offline | 171/171 | 0.876 | 0.712 | 0.712 | 0.936 | 0.740 | 0.819 | 0.712 | 0.0042 | 0.00 | 0.82 | 0.41 | 0.29 | - |
| coherent_491 | coherent_power (dynamic floor) | loopback | 342/342 | n/a: frame alignment lost after packet drops (online-vs-offline IoU 0.056) | | | | | | | | | | | | - |
| coherent_491 | coherent_power (dynamic floor) | offline | 342/342 | 0.952 | 0.822 | 0.822 | 0.766 | 0.464 | 0.560 | 0.356 | 0.0026 | 0.94 | 0.93 | 0.89 | 0.92 | - |
| coherent_491c | coherent_power (dynamic floor) | loopback | 340/342 | n/a: frame alignment lost after packet drops (online-vs-offline IoU 0.041) | | | | | | | | | | | | - |
| coherent_491c | coherent_power (dynamic floor) | offline | 342/342 | 0.952 | 0.822 | 0.822 | 0.766 | 0.464 | 0.560 | 0.356 | 0.0026 | 0.94 | 0.93 | 0.89 | 0.92 | - |
| dinoft_native_m2dr_491 | DINO-FT native (M2_dr, ViT-B) | loopback | 78/171 | n/a: frame alignment lost after packet drops (online-vs-offline IoU 0.167) | | | | | | | | | | | | - |
| dinoft_native_m2dr_491 | DINO-FT native (M2_dr, ViT-B) | offline | 171/171 | 0.758 | 0.701 | 0.701 | 0.970 | 0.873 | 0.916 | 0.850 | 0.0012 | 0.83 | 0.85 | 0.04 | 0.90 | - |
| dinoft_native_m5s_491 | DINO-FT native (M5-S, ViT-S) | loopback | 143/171 | n/a: frame alignment lost after packet drops (online-vs-offline IoU 0.088) | | | | | | | | | | | | - |
| dinoft_native_m5s_491 | DINO-FT native (M5-S, ViT-S) | offline | 171/171 | 0.761 | 0.668 | 0.668 | 0.745 | 0.662 | 0.733 | 0.522 | 0.0049 | 0.71 | 0.88 | 0.01 | 0.85 | - |
| dinoft_rt_m2dr_491 | DINO-FT RT (M2_dr, ViT-B) | loopback | 168/171 | n/a: frame alignment lost after packet drops (online-vs-offline IoU 0.038) | | | | | | | | | | | | - |
| dinoft_rt_m2dr_491 | DINO-FT RT (M2_dr, ViT-B) | offline | 171/171 | 0.861 | 0.675 | 0.675 | 0.842 | 0.627 | 0.706 | 0.533 | 0.0069 | 0.00 | 0.65 | 0.44 | 0.25 | - |
