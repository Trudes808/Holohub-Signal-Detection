# M4 domain-adaptation fine-tune — results (2026-09-20)

Fixed the DINO-FT real-OTA "coarse blob" domain gap by fine-tuning **M4** from base DINOv3 on a mix of
the synthetic `dataset_491` + real OTA captures (coherent-teacher pseudo-labels + antenna-off noise
negatives, 2.4 & 1.0 GHz). Pipeline: `dino_fine_tuning/label_ota.sh` -> `src/build_dataset_ota.py` ->
`src/combine_datasets.py` -> `src/train.py` -> `export_dinov3_finetuned_torchscript.py`.

## Training data
- Synthetic `dataset_491`: 2265 train / 277 val tiles (per-signal SNR -3..32 dB, dense+sparse).
- OTA `dataset_ota`: 3360 train / 1112 val tiles from 12 antenna-ON captures (coherent teacher, raised
  threshold + despeckle) + 4 antenna-OFF noise captures (all-zero masks). Both bands, gains 10/15.
- Combined: **5625 train / 1389 val** (~40% synthetic / 60% OTA).
- Fine-tune: `train.py --mode ft_lastN` (30.32M trainable), 18 epochs, best val IoU 0.8365 (combined).
- Export: `weights/finetuned_dino_m4_491_bf16.ts`, eager-vs-traced IoU 1.000, deploy threshold 0.6.

## Result 1 -- real OTA (the problem): M3 vs M4 on x410_ota_2g4_gain10_20260908 (`m3_vs_m4_ota.png`)
| | mask occ | quiet-pixel false positives |
|---|---|---|
| M3 (synthetic-only) | 0.67% | 15.6% of detections |
| **M4 (synthetic+OTA)** | **0.40%** | **6.6%** |
M4's coarse blobs resolve into **tight per-signal segments** that hug the real signals; the spurious
off-band blob (~2.62 GHz) is gone. False-firing on quiet regions **more than halved**.

## Result 2 -- synthetic held-out (anti-forgetting / SNR): M4 == M3 (`m4_heldout_snr.png`)
Per-SNR region detection, deploy thr 0.6 (test-split IoU 0.933, F1 0.966):
| kind | -3..3 | 3..10 | 10..20 | 20..32 |
|---|---|---|---|---|
| dense M3 | 92 | 92 | 91 | 100 |
| dense **M4** | 92 | 92 | **95** | 100 |
| sparse M3/M4 | 83 | 100 | 86 | 100 |
Identical across SNR (incl. low-SNR, the teacher's blind spot -- covered by the synthetic mix), **+4%
at dense 10-20 dB**. No regression: the OTA gain came free of synthetic forgetting.

## Reproduce
```
cd dino_fine_tuning
./label_ota.sh                                    # coherent teacher labels for *_on_* captures
PYTHONPATH=src .venv-ml/python src/build_dataset_ota.py --labels-dir /tmp/usrp_spectrograms/ota_labels
PYTHONPATH=src .venv-ml/python src/combine_datasets.py --inputs data/dataset_491 data/dataset_ota --out data/dataset_491_plus_ota
PYTHONPATH=~/Documents/dinov3:src .venv-ml/python src/train.py --config configs/train_491.yaml \
    --dataset data/dataset_491_plus_ota --mode ft_lastN --name M4_491_ota --out checkpoints/M4_491_ota
# export + validate: export_dinov3_finetuned_torchscript.py (--threshold 0.6); compare via scratchpad scripts here.
```

## Deploy
Point the live DINO configs' `model_script_path` at `finetuned_dino_m4_491_bf16.ts` (threshold 0.6).
Not yet deployed pending a live OTA confirmation run.
