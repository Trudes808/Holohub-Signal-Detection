# Signal Detection Experiment Workflow

This folder contains the lightweight notebook workflow we have been using to:

- select a SigMF capture window
- render the matching spectrogram and annotations
- run the offline CUDA DINO detector on that exact window
- compare detector output against SigMF ground truth
- inspect intermediate detector pathway and post-processing artifacts

## Files

- `signal_detection_eval.ipynb`
  - main notebook entry point
- `signal_detection_eval.py`
  - thin notebook helper module
- `offline_cuda_detector_eval_review_helpers.py`
  - manifest and saved-artifact review helpers
- `config_cuda_dino_performance_single_channel_offline_eval.yaml`
  - offline-eval config used by the maintained C++ harness

## Typical Workflow

### 1. Open the notebook

Open:

- `/home/sat3737/holohub-dev/applications/usrp_wideband_signal_detection/infocom_evals/signal_detection_experiments/signal_detection_eval.ipynb`

The notebook is intentionally thin. Most logic lives in `signal_detection_eval.py`.

### 2. Set the input capture and run the setup cells

The notebook is set up to work with a SigMF capture, for example:

- data: `/home/bqn82/captures/attenuation_dB_0.sigmf-data`
- meta: `/home/bqn82/captures/attenuation_dB_0.sigmf-meta`

Run the notebook cells from top to bottom through the offline-run helper cell.

That will:

- load the SigMF bundle
- choose an offline-compatible single-frame window
- show the corresponding spectrogram and annotation overlays
- prepare the exact offline input slice under:
  - `/home/sat3737/holohub-dev/applications/usrp_wideband_signal_detection/generated_inputs`

### 3. Rebuild if C++ changed

If you changed any of these C++ files, rebuild before running offline:

- `/home/sat3737/holohub-dev/applications/usrp_wideband_signal_detection/run_offline_cuda_detector_eval.cpp`
- `/home/sat3737/holohub-dev/operators/fft/fft.cu`
- `/home/sat3737/holohub-dev/operators/cuda_dino_detector/cuda_dino_detector.cu`

### 4. Use the notebook helper to print the manual sudo commands

The notebook offline cell calls:

- `/home/sat3737/holohub-dev/applications/usrp_wideband_signal_detection/run_cuda_dino_offline_file.py`

Keep `OFFLINE_DRY_RUN = True` when you want the notebook to print the exact manual commands without executing them.

We do this so privileged commands stay manual.

### 5. Run the printed commands in a terminal

The helper prints commands of this form:

```bash
sudo mkdir -p /tmp/usrp_spectrograms/offline_inputs/<slice_stem>
sudo cp -f <slice>.sigmf-data /tmp/usrp_spectrograms/offline_inputs/<slice_stem>/<slice>.sigmf-data
sudo cp -f <slice>.sigmf-meta /tmp/usrp_spectrograms/offline_inputs/<slice_stem>/<slice>.sigmf-meta
sudo docker exec -i usrp_x410_signal_detection_demo bash -lc '/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection/run_offline_cuda_detector_eval --config <generated_offline_config> --input-file <container_slice_path> --output-root <container_output_root>'
```

For the current single-frame example window, the command sequence has been:

```bash
sudo rm -rf /tmp/usrp_spectrograms/offline_cuda_dino/attenuation_dB_0_samples_30781694_36024574
sudo mkdir -p /tmp/usrp_spectrograms/offline_inputs/attenuation_dB_0_samples_30781694_36024574
sudo cp -f /home/sat3737/holohub-dev/applications/usrp_wideband_signal_detection/generated_inputs/attenuation_dB_0_samples_30781694_36024574.sigmf-data /tmp/usrp_spectrograms/offline_inputs/attenuation_dB_0_samples_30781694_36024574/attenuation_dB_0_samples_30781694_36024574.sigmf-data
sudo cp -f /home/sat3737/holohub-dev/applications/usrp_wideband_signal_detection/generated_inputs/attenuation_dB_0_samples_30781694_36024574.sigmf-meta /tmp/usrp_spectrograms/offline_inputs/attenuation_dB_0_samples_30781694_36024574/attenuation_dB_0_samples_30781694_36024574.sigmf-meta
sudo docker exec -i usrp_x410_signal_detection_demo bash -lc '/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection/run_offline_cuda_detector_eval --config /workspace/holohub/applications/usrp_wideband_signal_detection/generated_configs/config_cuda_dino_performance_single_channel_attenuation_dB_0_samples_30781694_36024574_offline_eval.yaml --input-file /workspace/spectrograms/offline_inputs/attenuation_dB_0_samples_30781694_36024574/attenuation_dB_0_samples_30781694_36024574.sigmf-data --output-root /workspace/spectrograms/offline_cuda_dino/attenuation_dB_0_samples_30781694_36024574'
```

## Output Layout

The offline run writes artifacts under:

- `/tmp/usrp_spectrograms/offline_cuda_dino/<slice_stem>`

Important outputs:

- `frame_manifest.csv`
- `offline_eval_summary.json`
- `spectrogram_tensors/`
- `aligned_spectrogram_tensors/`
- `mask_arrays/`
- `gt_masks/`

When detector debug artifacts are enabled, the same output root also contains:

- `offline_validation_summary.json`
- `chunk_debug/chunk_debug_summary.json`
- `chunk_debug/chunk_corrected_resized.npy`
- `chunk_debug/chunk_dino_score_raw.npy`
- `chunk_debug/chunk_dino_score_raw_deweighted.npy`
- `chunk_debug/chunk_coherence_gate.npy`
- `chunk_debug/chunk_hybrid_keep_freq.npy`
- `chunk_debug/chunk_hybrid_keep_res.npy`
- `chunk_debug/chunk_hybrid_seed_mask.npy`
- `chunk_debug/chunk_hybrid_closed_mask.npy`
- `chunk_debug/chunk_hybrid_filled_mask.npy`
- `chunk_debug/chunk_hybrid_component_filtered_mask.npy`
- `chunk_debug/chunk_grouped_mask.npy`
- `chunk_debug/chunk_final_mask.npy`
- `offline_projected_grouped_mask.npy`
- `offline_projected_grouped_score.npy`
- `offline_merged_box_mask.npy`
- `offline_final_mask.npy`

## Notebook Visualization Flow

After the manual offline run finishes, go back to the notebook and run the later cells.

These cells now provide three layers of review:

### Saved detector vs ground truth

The main comparison cells show:

- SigMF spectrogram with GT boxes
- saved offline GT mask
- saved detector mask over the same spectrogram region
- standalone GT and detector binary masks

### Detector pathway views

The debug cell at the bottom now loads the detector intermediate artifact bundle and renders:

- corrected spectrogram input
- raw DINO score
- deweighted DINO score
- coherence gate
- hybrid keep-frequency pathway
- hybrid keep-residual pathway
- combined score
- projected combined score

### Post-processing views

The same debug cell also renders the post-processing stages:

- hybrid seed mask
- hybrid closed mask
- hybrid filled mask
- hybrid component-filtered mask
- grouped mask
- projected grouped mask
- merged box mask
- final global mask

This is the quickest way to see where the detector path is losing signal quality.

## Notes

- The notebook is designed around the exact same offline C++ harness used in the application path.
- The helper currently stages a single exact offline-compatible frame for the selected annotation region.
- `run_cuda_dino_offline_file.py` now generates an `offline_eval` overlay config automatically when needed.
- Detector intermediate debug artifacts are enabled through the offline wrapper-generated config.
- All `sudo` commands remain manual by design.