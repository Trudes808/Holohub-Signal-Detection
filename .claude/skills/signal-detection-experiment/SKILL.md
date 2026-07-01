---
description: Plan, run, and summarize USRP wideband signal-detection experiments. Use when sweeping detector configs, comparing coherent and DINO behavior, or evaluating offline versus live signal-detection quality.
disable-model-invocation: true
argument-hint: "[goal or sweep description]"
---

# Signal Detection Experiment

Work in `applications/usrp_wideband_signal_detection`.

Goal: $ARGUMENTS

## Choose the cheapest useful path first

- Prefer frozen-input or offline validation if the goal can be answered without a live SDR run.
- Prefer single-channel live configs before dual-channel throughput configs unless the goal explicitly requires dual-channel behavior.
- Start from an existing baseline config before creating a new one.

Recommended baselines:

- Coherent live single-channel: `config_coherent_power_performance_single_channel.yaml`
- Coherent validation: `config_coherent_power_validation.yaml`
- Coherent frozen-input capture: `old_configs/config_coherent_power_debug_capture.yaml`
- DINO live single-channel: `config_torchscript_performance_single_channel.yaml`
- DINO live two-channel: `config_torchscript_performance.yaml`

## Execution rules

- If you create a new runnable live config, keep it at the app root and name it `config*.yaml`.
- Do not use host-local CMake or raw `./holohub build` for this app.
- Use the documented wrappers:
  - `./rebuild_demo_container_app.sh`
  - `CONFIG_NAME=<config>.yaml ./run_coherent_power_performance.sh`
  - `CONFIG_NAME=<config>.yaml ./run_torchscript_performance_test.sh`
  - `./run_offline_coherent_power_validator_from_tensor.sh --latest-snapshot`
  - `./dino_cuda_validation.sh --tensor-npy <path> ...`
- Preserve validation and production parity by keeping the detector backend consistent. Vary `emit_stride`, logging, saves, or timing flags before changing the validated backend.

## Experiment record

Create or update a run note under `applications/usrp_wideband_signal_detection/infocom_evals/signal_detection_experiments/` using `TEMPLATE.md`.

Each run note should capture:

1. Goal and hypothesis.
2. Baseline config and edited config.
3. Exact command or commands used.
4. Key metrics and artifact paths.
5. Whether the result supported the hypothesis.
6. The next recommended sweep or rollback decision.

## Output

Return a short summary with:

- chosen baseline
- config changes
- command or commands run
- key measurements
- next recommended experiment