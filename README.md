# Argus: Real-Time Wideband RF Signal Detection on Holoscan

Argus is a GPU-native pipeline that detects and extracts signals of interest from a wideband
software-defined-radio stream in real time. A USRP X410 streams up to two 491.52 MSps channels
over 100 GbE; NVIDIA Holoscan ingests the samples with DPDK, computes a live wide FFT and
spectrogram on the GPU, runs a detector (coherent power, zero-shot DINOv3, or a fine-tuned DINOv3
segmenter), cuts every detected region out of the raw IQ as SigMF, and renders a
spectrum-analyzer dashboard. The same operators run offline, so a directory of captures can be
segmented with no radio attached.

This repository is a fork of NVIDIA's [HoloHub](https://github.com/nvidia-holoscan/holohub) with
the RF/AI application and its operators added. The upstream HoloHub documentation is kept
unchanged in [`doc/holohub_upstream_README.md`](doc/holohub_upstream_README.md).

## Pick your task

| I want to… | Start here |
| --- | --- |
| **Install Holoscan and run the live demo** on a USRP X410 (dashboard, detector switch, signal classifier) | [`applications/usrp_wideband_signal_detection/README.md`](applications/usrp_wideband_signal_detection/README.md) §1 Install, §2 Live demo |
| **Segment the regions of interest in an offline dataset** (a directory of SigMF captures → `<dir>_preprocessed` of per-signal SigMF cutouts) | [`applications/usrp_wideband_signal_detection/README.md`](applications/usrp_wideband_signal_detection/README.md) §3 Preprocess an offline dataset |
| **Retrain the fine-tuned DINOv3 detector** (the M2 recipe on the Argus dataset, and how the deployed M5-S was trained and exported) | [`dino_fine_tuning/README.md`](dino_fine_tuning/README.md) |

Quick version of each:

```bash
# 1. live demo (DGX Spark + X410; container built once per §1 of the app README)
cd applications/usrp_wideband_signal_detection
sudo ./bash_scripts/after_reboot.sh            # host prerequisites, once per boot
sudo ./bash_scripts/run_live_demo.sh v5        # or v5dual for two channels

# 2. offline dataset -> regions of interest
sudo python3 preprocess_dataset.py /data/my_captures --detector coherent_power   # -> /data/my_captures_preprocessed

# 3. fine-tune DINOv3 (M2 recipe)
cd ../../dino_fine_tuning
python src/build_dataset.py --config configs/dataset.yaml
python src/train.py --config configs/train.yaml --dataset data/dataset --mode ft_lastN --name M2_ft --out checkpoints/M2_ft
```

## What is in this fork

| Path | Contents |
| --- | --- |
| [`applications/usrp_wideband_signal_detection/`](applications/usrp_wideband_signal_detection/) | The application: configs, container/run wrappers, offline drivers, visualization, snipper + SigMF sink, evaluation harnesses, notes. |
| [`operators/coherent_power_signal_detector/`](operators/coherent_power_signal_detector/), [`operators/cuda_dino_detector/`](operators/cuda_dino_detector/), [`operators/finetuned_dino_detector/`](operators/finetuned_dino_detector/) | The three detector operators (plus the shared `fft`, `spectrogram`, `advanced_network` ingest operators). |
| [`applications/usrp_freq_detection/`](applications/usrp_freq_detection/) | The X410 radio sender (`rx_to_remote_udp.py`), replay packetizer and the x86 container `Dockerfile`. |
| [`dino_fine_tuning/`](dino_fine_tuning/) | Dataset builders, training, evaluation and TorchScript export for the fine-tuned DINOv3 detector. |
| [`yolo_training/`](yolo_training/) | YOLO26 baseline trained on the same data for the detector comparison. |
| [`notebooks/`](notebooks/) | Validation and evaluation notebooks (detector comparisons, low-SNR and OOD studies). |

## Platforms

- **Validated bench:** NVIDIA DGX Spark (GB10, aarch64, CUDA 13, ConnectX-7) with a USRP X410 running the `CG_400` FPGA image. Container image: `applications/usrp_wideband_signal_detection/Dockerfile.spark_cuda13`.
- **Original bench:** x86_64 with an RTX 4000 Ada (CUDA 12.6, GPUDirect RDMA). Container image: `applications/usrp_freq_detection/Dockerfile`. Platform deltas: [`notes/RUNNING_ON_DGX_SPARK.md`](applications/usrp_wideband_signal_detection/notes/RUNNING_ON_DGX_SPARK.md) §1.

Model weights are not in git. The app README §1 lists what to download and where it goes
(DINOv3 backbone, the fine-tuned `M5-S` TorchScript export, optional classifier weights).

## License

The fork keeps HoloHub's Apache-2.0 [LICENSE](LICENSE); the RF/AI additions carry
`SPDX-FileCopyrightText: 2026 National Instruments Corporation` headers under the same license.
DINOv3 weights are distributed by Meta under their own license.
