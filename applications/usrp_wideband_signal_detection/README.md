# USRP Wideband Signal Detection

## Overview

This application mirrors the high-rate USRP ingest path and adds the new DINOv3 signal detector stage.

Flow:

`chdrConverterOp -> fftOp -> spectrogramOp -> dinoV3SignalDetectorOp`

A side logger branch is kept from `fftOp` for throughput visibility.

## Run

From the build directory for this application:

```bash
./usrp_wideband_signal_detection config.yaml
```

Use the same external USRP stream command used by `usrp_freq_detection`.

## Notes

- `spectrogramOp` currently saves debug spectrogram images.
- `dinoV3SignalDetectorOp` is a C++/CUDA scaffold that emits a deterministic mask tensor and metadata.
- This stage is the integration point for TensorRT-backed DINOv3 inference in future iterations.
