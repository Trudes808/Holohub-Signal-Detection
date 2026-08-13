# Golden mask manifests for the GPU optimization work

Bit-exact regression gate for the code-level GPU optimizations
(see `notes/gpu_optimization_plan.md`). Every operator change must reproduce these masks
byte-for-byte before it ships.

- **Baseline commit**: c711d42e (live_demo), binary rebuilt from that HEAD 2026-08-13.
- **Input capture**: `comprehensive_ordered.sigmf-data`
  (from `~/Documents/holoscan_waveform_generation/composition/composites/`; staged copy lives at
  `/tmp/usrp_spectrograms/offline_inputs/comprehensive_ordered/`).
- **Determinism**: both variants verified run-to-run byte-identical (two independent runs each)
  before being recorded, so any hash mismatch is a real behavior change, not noise.

| Manifest | Variant | Config | Frames |
| --- | --- | --- | --- |
| `golden_perfreq_sha256.txt` | per-freq calibrated | `config_coherent_power_perf_perfreq_single_channel.yaml` (harness default for `--detector coherent_power`) | 286 × 512×10240 |
| `golden_dynamic_sha256.txt` | dynamic floor | `config_coherent_power_perf_dynamic_single_channel.yaml` | 573 × 256×10240 |

## Regenerate / verify

```bash
cd applications/usrp_wideband_signal_detection
sudo ./bash_scripts/rebuild_demo_container_app.sh

# per-freq variant
python3 run_cuda_dino_offline_file.py \
  ~/Documents/holoscan_waveform_generation/composition/composites/comprehensive_ordered.sigmf-data \
  --detector coherent_power --no-tensors \
  --output-root /tmp/usrp_spectrograms/gpu_opt_check_perfreq

# dynamic-floor variant
python3 run_cuda_dino_offline_file.py \
  ~/Documents/holoscan_waveform_generation/composition/composites/comprehensive_ordered.sigmf-data \
  --detector coherent_power \
  --config config_coherent_power_perf_dynamic_single_channel.yaml --no-tensors \
  --output-root /tmp/usrp_spectrograms/gpu_opt_check_dynamic

# compare (must produce no output)
(cd /tmp/usrp_spectrograms/gpu_opt_check_perfreq/mask_arrays && sha256sum *.npy | sort -k2) \
  | diff infocom_evals/signal_detection_experiments/gpu_opt_golden_masks/golden_perfreq_sha256.txt -
(cd /tmp/usrp_spectrograms/gpu_opt_check_dynamic/mask_arrays && sha256sum *.npy | sort -k2) \
  | diff infocom_evals/signal_detection_experiments/gpu_opt_golden_masks/golden_dynamic_sha256.txt -
```
