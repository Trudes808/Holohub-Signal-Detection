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

- Coherent live (calibrated per-freq): `config_coherent_power_perf_perfreq_single_channel.yaml`
- Coherent live (dynamic floor): `config_coherent_power_perf_dynamic_single_channel.yaml`
- Coherent frozen-input capture: `config_coherent_power_capture_chdr_single_channel.yaml`
- Cable-loopback replay: `config_coherent_power_performance_single_channel_replay.yaml`
- DINO live: `config_cuda_dino_performance_single_channel.yaml`

## Execution rules

- If you create a new runnable live config, keep it at the app root and name it `config*.yaml`. Shell wrappers live in `bash_scripts/`, calibration in `calibration/`, superseded configs in `old_configs/`, notes in `notes/` — keep that layout.
- Do not use host-local CMake or raw `./holohub build` for this app.
- Use the documented wrappers (run from the app root; container identity from `bash_scripts/container_env.sh`):
  - `sudo ./bash_scripts/rebuild_demo_container_app.sh`
  - `CONFIG_NAME=<config>.yaml sudo ./bash_scripts/run_coherent_power_performance.sh`
  - `CONFIG_NAME=<config>.yaml sudo ./bash_scripts/run_torchscript_performance_test.sh`
  - Offline eval: `python3 run_cuda_dino_offline_file.py <file.sigmf-data> --detector {coherent_power|cuda_dino} --config <cfg> --output-root <dir>`
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