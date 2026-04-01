# Sage Development Notes

Last updated: 2026-04-01

## Important Recent Changes (2026-03-24)

### New Application: `usrp_wideband_signal_detection`

- Added a new application parallel to `usrp_freq_detection`:
  - `applications/usrp_wideband_signal_detection/main.cpp`
  - `applications/usrp_wideband_signal_detection/config.yaml`
  - `applications/usrp_wideband_signal_detection/CMakeLists.txt`
  - `applications/usrp_wideband_signal_detection/README.md`
  - `applications/usrp_wideband_signal_detection/metadata.json`
- Graph flow in new app:
  - `chdrConverterOp -> fftOp -> spectrogramOp -> dinoV3SignalDetectorOp`
  - plus side branch `fftOp -> logOp`

### New Operator: `dinov3_signal_detector`

- Added operator scaffold and registration:
  - `operators/dinov3_signal_detector/dinov3_signal_detector.hpp`
  - `operators/dinov3_signal_detector/dinov3_signal_detector.cu`
  - `operators/dinov3_signal_detector/CMakeLists.txt`
  - `operators/dinov3_signal_detector/README.md`
  - `operators/dinov3_signal_detector/metadata.json`
  - `operators/CMakeLists.txt` updated to include `dinov3_signal_detector`

### Spectrogram Handoff Update

- Updated `spectrogram` operator to emit passthrough output (`out`) so downstream detector stages can consume the same GPU tensor stream without extra host copies.
- Files updated:
  - `operators/spectrogram/spectrogram.hpp`
  - `operators/spectrogram/spectrogram.cu`

### DINO Detector Backend Modes (Current)

- `dinov3_signal_detector` now supports configurable backend behavior:
  - `torchscript`: attempts TorchScript model forward on GPU.
  - `pytorch_placeholder`: runs GPU PyTorch tensor preprocessing and placeholder mask generation.
  - `cuda_threshold_fallback`: runs CUDA-only threshold masking path.
- New config knobs include:
  - `use_pytorch_backend`
  - `inference_backend`
  - `model_name`
  - `model_repo_path`
  - `weights_path` (placeholder while download completes)
  - `model_script_path` (placeholder TorchScript path)
  - `strict_model_forward`

### Model Artifacts Status

- Full DINOv3 model-forward integration is still in progress.
- The local host-side DINOv3 repo is now known:
  - `/home/sat3737/holoscan_demo_workspace/dinov3`
- The selected host-side weight file is now known:
  - `/home/sat3737/holoscan_demo_workspace/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`
- The runtime source of truth remains the Holohub container path:
  - `/workspace/models/dinov3`
- Current implementation includes a TorchScript hook with strict validation available via `strict_model_forward=true`.
- Remaining work is container staging plus TorchScript export under `/workspace/models/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.ts`.

## FFT Operator Flow

### Current End-to-End Data Path

1. **Network ingest + packet steering (Advanced Network Operator)**
   - VITA49 UDP packets are split across CPU/GPU memory regions using queue/flow rules.
   - File locations:
     - `applications/psd_pipeline/config.yaml`
     - `applications/psd_pipeline/README.md`

2. **VITA connector (VRT parsing + IQ staging to GPU tensor)**
   - Context packets are parsed and converted into metadata.
   - Data packets are byte-swapped and converted from int16 IQ to complex float GPU tensors.
   - Emits `tuple<tensor_t<complex, 2>, cudaStream_t>`.
   - File locations:
     - `applications/psd_pipeline/advanced_network_connectors/vita49_rx.h`
     - `applications/psd_pipeline/advanced_network_connectors/vita49_rx.cu`
     - `applications/psd_pipeline/advanced_network_connectors/README.md`

3. **FFT operator**
   - Applies `fftshift1D(fft(input))` on GPU.
   - Uses `channel_number` metadata to select per-channel output slice.
   - Passes VITA49-related fields via metadata (spectrum/window/indices/etc.).
   - File locations:
     - `operators/fft/fft.hpp`
     - `operators/fft/fft.cu`
     - `operators/fft/README.md`

4. **High-rate PSD operator**
   - Converts complex FFT output to linear power via `abs2(...) * (1/N^2)`.
   - File locations:
     - `operators/high_rate_psd/high_rate_psd.hpp`
     - `operators/high_rate_psd/high_rate_psd.cu`
     - `operators/high_rate_psd/README.md`

5. **Low-rate PSD operator**
   - Averages over burst dimension, converts to dB, clamps and casts to int8.
   - This is optimized for transport/telemetry output, not ML feature fidelity.
   - File locations:
     - `operators/low_rate_psd/low_rate_psd.hpp`
     - `operators/low_rate_psd/low_rate_psd.cu`
     - `operators/low_rate_psd/README.md`

6. **VITA49 packetizer**
   - Copies int8 PSD from GPU to host and sends VITA49.2 spectral/context UDP packets.
   - File locations:
     - `operators/vita49_psd_packetizer/vita49_psd_packetizer.hpp`
     - `operators/vita49_psd_packetizer/vita49_psd_packetizer.cu`
     - `operators/vita49_psd_packetizer/README.md`

7. **App graph wiring**
   - `vitaConnectorOp -> fftOp -> highRatePsdOp -> lowRatePsdOp -> packetizerOp`
   - File location:
     - `applications/psd_pipeline/main.cpp`

### Important Considerations Identified

- **No direct UHD/USRP driver path in this repo** was found; ingest is currently VITA49-over-network oriented.
- **FFT configuration coupling matters**:
  - `vita_connector` and `fft` must agree on effective samples-per-FFT (`num_packets_per_fft * num_complex_samples_per_packet == burst_size`).
- **Metadata is a core control plane**:
  - `channel_number`, timestamps, stream ID, and VITA context values are required downstream.
- **Current low-rate PSD path is lossy for ML**:
  - `10*log10`, clamping, and int8 cast can discard useful spectral detail.
- **Packetizer introduces host transfer overhead**:
  - Suitable for UDP export; should be kept off critical ML path.
- **Code quality note**:
  - `spectrum_type` param appears duplicated in FFT setup registration (`operators/fft/fft.cu`), likely harmless but should be cleaned.

### Runtime State and Metadata Inventory (Current)

#### Connector State (`Vita49ConnectorOpRx`)

- **Per-channel persistent state** (in `Channel` struct):
  - `cur_idx`, `rf_data`, stream/event arrays, output queue (`out_q`), packet counters
  - `current_context`, `current_meta`, `context_received`, `meta_set`
- **Batching state**:
  - `num_packets_per_batch`, `cur_msg`, aggregation count (`aggr_pkts_recv`)
- Files:
  - `applications/psd_pipeline/advanced_network_connectors/vita49_rx.h`
  - `applications/psd_pipeline/advanced_network_connectors/vita49_rx.cu`

#### Metadata Keys in Active Use

- Ingest/connector adds:
  - `channel_number`, `integer_timestamp`, `fractional_timestamp`, `stream_id`
  - `bandwidth_hz`, `rf_ref_freq_hz`, `reference_level_dbm`, `gain_stage_1_db`, `gain_stage_2_db`, `sample_rate_hz`, `change_indicator`
- FFT adds/passes:
  - `spectrum_type`, `averaging_type`, `window_time_delta_interpretation`, `window_type`, `num_transform_points`, `num_window_points`, `resolution`, `span`, `weighting_factor`, `f1_index`, `f2_index`, `window_time_delta`
- Low-rate PSD adds:
  - `num_averages`


## Proposed Spectrogram and Dinov3 Signal Detection Operators

### Development Goal

Add two new operators while preserving high-throughput behavior:

1. **Spectrogram Operator** (GPU-native, high-rate to model-ready time-frequency tensor)
2. **DINOv3 Signal Detection Operator** (inference stage using pre-trained DINOv3 backend)

### Proposed Operator 1: Spectrogram Operator

#### Primary Role

- Convert FFT stream into model-ready spectrogram tensors.
- Preserve precision and avoid host copies.

#### Proposed Input/Output Contract

- **Input**: `tuple<tensor_t<complex, 2>, cudaStream_t>` from `fft` (or optional `tuple<tensor_t<float,2>, cudaStream_t>` from high-rate PSD variant).
- **Output**:
  - Main: spectrogram tensor for inference (`tensor_t<half|float, 3 or 4>`, e.g. `C x H x W` or `N x C x H x W`).
  - Optional debug/telemetry port for visualization/export branch.

#### Required Internal State

- Per-channel rolling buffer/ring state:
  - Time-window history, write index, frame-ready flag.
- Config state:
  - `freq_bins`, `time_bins`, `hop_size`, `overlap`, `window_type`, `log_mode`, `normalization`, `output_dtype`.
- Performance state:
  - pre-allocated output tensors, optional CUDA stream/event bookkeeping.

#### Suggested File Locations

- New operator directory:
  - `operators/spectrogram/`
    - `spectrogram.hpp`
    - `spectrogram.cu`
    - `CMakeLists.txt`
    - `README.md`
    - `metadata.json`
- Registration update:
  - `operators/CMakeLists.txt`

### Proposed Operator 2: DINOv3 Signal Detection Operator

#### Primary Role

- Run spectrogram tensors through DINOv3 model and emit detection/classification embeddings/results.

#### Integration Approach (Preferred)

- **Preferred first pass**: leverage `holoscan::ops::InferenceOp` style path with CUDA tensors and TensorRT backend (same pattern used in multiple apps).
- **Alternative**: custom GXF/TensorRT wrapped operator if DINOv3 graph requires specialized pre/post not cleanly handled by stock InferenceOp.

#### Required Internal/Configuration State

- Model runtime state:
  - model path / engine cache / precision mode (FP16/FP32) / dynamic shape handling.
- Input mapping state:
  - tensor name mapping between spectrogram op output and model input.
- Optional post-processing state:
  - thresholds, class map, embedding output mode.

#### Suggested File Locations

- If using existing InferenceOp in app graph:
  - Add app-level config and wiring in:
    - `applications/psd_pipeline/main.cpp` (or a new dedicated SDR+ML app)
    - `applications/psd_pipeline/config.yaml` (or new config file)
- If custom operator needed:
  - `operators/dinov3_signal_detector/`
    - `dinov3_signal_detector.hpp/.cpp` or `.cu`
    - `CMakeLists.txt`, `README.md`, `metadata.json`

### Proposed Graph Wiring (with Branching)

#### Keep Existing Export Branch

- `vitaConnectorOp -> fftOp -> highRatePsdOp -> lowRatePsdOp -> packetizerOp`

#### Add New ML Branch

- `fftOp -> spectrogramOp -> dinov3InferenceOp -> (postprocess/alert/publish)`

This avoids host-copy bottlenecks from the packetizer branch and allows ML-specific rate control.

### Key Proposed Changes to Streamline Flow for High Throughput Use Case

1. **Split transport path from ML path**
   - Keep packetizer branch as-is for standards-compliant UDP output.
   - Keep inference path fully GPU-resident.

2. **Move spectrogram generation to GPU and fuse operations where possible**
   - Fuse windowing/log/normalization to reduce extra memory traffic.

3. **Introduce explicit inference rate control**
   - Add stride/drop/window policy before DINOv3 (not every FFT frame should trigger inference).

4. **Fix model input geometry early**
   - Make spectrogram operator emit fixed-size tensors directly aligned to DINOv3 expected shape.

5. **Use pre-allocated pools and static tensor layouts**
   - Avoid runtime allocations in hot path for both new operators.

6. **Preserve and propagate minimal metadata needed for decisions**
   - Channel/time/frequency context should flow into inference outputs for downstream actionability.

7. **Plan for 2-channel scheduling explicitly**
   - Maintain per-channel state isolation and deterministic channel tagging in output messages.

### Suggested Implementation Phases

1. Build `spectrogram` operator + unit-level validation against known FFT inputs.
2. Wire spectrogram branch in app with synthetic data and throughput telemetry.
3. Add DINOv3 inference integration and verify tensor mapping/latency.
4. Add gating/decimation controls and stress test at target ingest rates.
5. Optimize kernel and buffer strategy based on profiling.
