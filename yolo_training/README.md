# YOLO training / eval (Ultralytics YOLO26)

Bring **Ultralytics YOLO26** in as another signal-detection model alongside the DINO work,
emitting masks in the **same format as every other detector** so it scores through the
identical `eval_detector_masks.py` pipeline (consistent metrics/buckets/thresholds).

## Goal (parallels `dino_fine_tuning/`)
- **YOLO26 fine-tuned** for signal-vs-noise on RF spectrograms (the primary model, à la DINO M1/M2).
- ~~YOLO26 zero-shot~~ — **dropped**: stock YOLO is closed-vocabulary (COCO), no signal class; not a
  meaningful zero-shot detector for spectrograms. Fine-tuned only.
- YOLO masks materialized into the batch-eval format (like `gen_finetuned_run.py`), then scored
  by the canonical `eval_detector_masks.py` → `compare_tables_*`.
- **A YOLO-only eval notebook** + **a comparison notebook** (YOLO vs coherent / zero-shot DINO /
  fine-tuned DINO), reusing `eval_viz` / `plot_eval_results` like the DINO notebooks.

## Environment
Conda env **`yolo`** (torch cu124 to match the RTX 4000 Ada / dinov3 env). Create with:
```bash
mamba env create -f environment.yml         # or: bash scripts/setup_env.sh
conda activate yolo
```
`scripts/setup_env.sh` also registers a Jupyter kernel "Python (yolo)" and freezes exact
versions into `requirements.txt`.

## Layout
- `environment.yml` / `requirements.txt` — env spec (loose / pinned).
- `scripts/setup_env.sh` — create env + kernel + freeze.
- `src/train_yolo.py`, `src/predict_yolo.py` — thin Ultralytics wrappers.
- `configs/dataset.yaml.example` — YOLO dataset config template (binary signal class).
- `data/`, `runs/`, `weights/` — datasets / training outputs / checkpoints (git-ignored).

## Design decisions (locked)
1. **Fine-tuned only** — no zero-shot YOLO (closed-vocab COCO head can't detect RF signals; the DINO
   backbone can transfer, a YOLO head can't). One or more fine-tuned YOLO26 detectors, à la DINO M1/M2.
2. **Masks via detection boxes** — train YOLO26 *detection*; fill each predicted box into the binary
   mask grid. Matches the GT filled-box convention + box-IoU/pixel metrics, apples-to-apples with DINO.
3. **Reuse DINO framing** — build the YOLO dataset from the same 256 time-row x 1024 freq-bin frames
   (nfft=1024) as `dino_fine_tuning`, so labels, geometry, and the eval display grid align.


## Pipeline status
- [x] **Dataset builder** (`src/build_yolo_dataset.py`) — DINO frames+boxes+splits → YOLO format (14588/3038/3150).
- [~] **Train** fine-tuned YOLO26 detection (signal/noise): `scripts/train_both.sh` runs yolo26s + yolo26m — *in progress*.
- [x] **Mask emitter** (`src/yolo_infer.py` + `src/gen_yolo_run.py`) — YOLO boxes → binary masks on the batch grid
      (`to_display_grid`, verbatim from the DINO path). Geometry validated.
- [x] **Assembler + eval** (`src/assemble_yolo_eval.py`) — materialize YOLO masks + symlink the DINO/deployed
      detectors into one sweep root, run `eval_detector_masks.py` → `eval/compare_tables/`.
- [x] **Notebooks** — `notebooks/yolo_eval.ipynb` (YOLO-only) + `notebooks/yolo_vs_dino_comparison.ipynb`
      (vs coherent / zero-shot DINO / fine-tuned DINO). Kernel "Python (yolo)".

## Run the eval (after training finishes)
```bash
conda activate yolo
python src/assemble_yolo_eval.py          # YOLO masks (both sizes) + DINO dirs -> eval_detector_masks -> eval/compare_tables
```
Then open the two notebooks (kernel **Python (yolo)**) and Run All. Weights are auto-located by glob
(`runs/**/yolo26{s,m}_signal/weights/best.pt`), so the nested Ultralytics run path doesn't matter.
Relocate eval artifacts by setting `YOLO_EVAL_ROOT`.
